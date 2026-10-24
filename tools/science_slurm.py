"""Native Slurm adapter for the existing bounded science supervisor.

The CPU controller survives GPU leases. Each GPU holder runs a small independent
process manager; checkpoints, identities and host profiles remain outside Git.
"""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import time
import urllib.request

from science_control import Controller, run, save, utc


def fields(job):
    result = run(['scontrol', 'show', 'job', str(job), '-o'])
    record = dict(re.findall(r'(\w+)=([^\s]+)', result))
    if not record.get('UserId', '').endswith('(' + str(os.getuid()) + ')'):
        raise PermissionError('Foreign Slurm allocation')
    return record


class Native(Controller):
    def enabled(self):
        self.git('fetch', '--quiet', 'origin', 'refs/heads/experiment-queue')
        text = self.git('show', 'FETCH_HEAD:.agents/experiment-orchestration.md')
        return re.findall(r'^Status: (.+)$', text, re.M) == ['enabled'] and bool(re.search(r'^Consent: \S', text, re.M))

    def allocation(self):
        output = run(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i|%T|%N|%e'])
        rows = [line.split('|') for line in output.splitlines() if line.split('|')[0] == str(self.p['job_id'])]
        if not rows:
            return 'ABSENT', 0
        if len(rows) != 1:
            raise ValueError('Ambiguous allocation')
        _, state, node, end = rows[0]
        remaining = 0
        if state == 'RUNNING':
            allocation = fields(self.p['job_id'])
            if 'gres/gpu=1' not in allocation['AllocTRES'].split(','):
                raise ValueError('Expected exactly one allocated GPU')
            self.p['node'] = node
            self.s['allocation_expiry_utc'] = end
            remaining = __import__('datetime').datetime.fromisoformat(end).replace(
                tzinfo=__import__('datetime').timezone.utc).timestamp() - time.time()
        return state, remaining

    def node(self, argv, timeout=60):
        # Parent resource probes are replaced with the GPU manager's actual
        # memory/device observations; no Docker or ROCm commands are executed.
        if argv[:4] == ['sudo', '-n', 'docker', 'stats']:
            return json.dumps(self.health().get('memory', {}))
        if argv[0].endswith('rocm-smi'):
            return json.dumps(self.health().get('gpu', {}))
        raise ValueError('Native manager guard is stale; diagnose before recovery')

    def health(self):
        path = self.root / 'supervisor/worker.json'
        return json.loads(path.read_text()) if path.exists() else {}

    def container(self):
        health = self.health()
        if health.get('job_id') != self.p['job_id'] or health.get('token') != self.s.get('container_id'):
            return None
        return 'running' if health.get('running') and time.time() - health.get('checked_epoch', 0) < 30 else None

    def stop_container(self):
        token = self.s.get('container_id')
        if not token:
            return
        save(self.root / 'supervisor/stop.json', dict(token=token))
        for _ in range(30):
            h = self.health()
            if h.get('token') != token or not h.get('running'):
                return
            if self.allocation()[0] == 'ABSENT':
                return  # Slurm has ended the exact job and its process steps.
            time.sleep(1)
        raise RuntimeError('Exact native workload stop is unverified')

    def launch(self, recovery=False):
        if not self.queue() or (self.root / 'STOP').exists():
            return False
        step, checkpoint = self.checkpoints()
        if self.s['completed_updates'] and not step:
            raise ValueError('Complete recovery checkpoint is unavailable')
        if step:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
            subprocess.run(['srun', '--jobid=' + str(self.p['job_id']), '--overlap', '--exact',
                '--nodes=1', '--ntasks=1', '--cpus-per-task=2', '--mem=48G', '--gres=none',
                self.p['python'], self.p['code'] + '/tools/verify_science_checkpoint.py',
                '--checkpoint', checkpoint, '--model', self.p['model'],
                '--output', str(self.root / 'supervisor/verified-checkpoint.json')],
                env=env, check=True, timeout=1200, stdout=subprocess.DEVNULL)
        self.s['attempt'] += 1
        if recovery:
            self.s['failures'].append(dict(time=time.time(), checkpoint=step))
        token = secrets.token_hex(32)
        name = self.p['run_id'] + '-a' + str(self.s['attempt'])
        self.s.update(completed_updates=step, resume_checkpoint=checkpoint, container=name,
                      container_id=token, attempt_started=time.time(), last_progress=time.time(),
                      status='starting', log_offset=0, log_tail='')
        self.s.setdefault('started_utc', utc())
        self.persist()
        save(self.root / 'supervisor/start.json', dict(token=token, name=name, checkpoint=checkpoint,
                                                       job_id=self.p['job_id'], claim=self.p['claim']))
        return True

    def renew(self):
        if self.s['renewals'] >= self.p['max_renewals']:
            raise ValueError('Finite GPU lease budget exhausted')
        if not self.enabled() or (self.root / 'STOP').exists() or self.allocation()[0] != 'ABSENT':
            return
        token = self.p['claim'] + '-lease-' + str(self.s['renewals'] + 1)
        matches = [line.split('|')[0] for line in run(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i|%k']).splitlines()
                   if line.endswith('|' + token)]
        if len(matches) > 1:
            raise ValueError('Duplicate GPU submission token')
        if self.s.get('pending_submission') == token and not matches:
            raise ValueError('Uncertain submission requires scheduler reconciliation')
        if matches:
            job = matches[0]
        else:
            self.s['pending_submission'] = token
            self.persist()
            job = run(['sbatch', '--parsable', '--comment=' + token, *self.p['sbatch_args'],
                       self.p['holder'], self.p['profile']]).split(';')[0]
        if not job.isdigit():
            raise ValueError('Uncertain allocation response')
        self.p['job_id'] = int(job)
        save(self.p['profile'], self.p)
        self.s['renewals'] += 1
        self.s.pop('pending_submission', None)
        self.s.pop('container', None)
        self.s.pop('container_id', None)
        self.s['status'] = 'pending_allocation'

    def row(self):
        values = [str(self.p.get('priority', 10)), '[' + self.p['name'] + '](' + self.p['record'] + ')',
                  self.p['resources'], self.p['owner'], self.p['claim'], str(self.p['job_id']),
                  self.s.get('started_utc', '—'), self.s.get('ended_utc', '—'),
                  '[run](' + self.p['run_url'] + ')', self.s['status'] + '; ' + str(self.s['completed_updates']) + '/480 updates']
        return '| ' + ' | '.join(values) + ' |'

    def bootstrap(self):
        machine = 'machine:sha256:' + hashlib.sha256(Path('/etc/machine-id').read_bytes().strip()).hexdigest()
        if self.p['machine_id'] != machine:
            raise ValueError('Profile belongs to another CPU controller')
        if (self.root / 'STOP').exists() or self.s['status'] in ('user_stopped', 'completed', 'needs_attention'):
            return False
        if self.p.get('logging_ready'):
            return True
        state, _ = self.allocation()
        if state == 'ABSENT':
            self.renew()
            self.persist()
            return False
        if state != 'RUNNING':
            return False
        for _ in range(5):
            self.git('fetch', '--quiet', 'origin', 'refs/heads/experiment-queue')
            head = self.git('rev-parse', 'FETCH_HEAD')
            consent = self.git('show', head + ':.agents/experiment-orchestration.md')
            if re.findall(r'^Status: (.+)$', consent, re.M) != ['enabled']:
                return False
            index = self.git('show', head + ':experiments/README.md') + '\n'
            marker = '](' + self.p['record'] + ')'
            matches = [line for line in index.splitlines() if marker in line and line.startswith('|')]
            if len(matches) != 1:
                raise ValueError('Expected one pending scientific protocol')
            cells = [v.strip() for v in matches[0].strip('|').split('|')]
            if cells[4] not in ('—', self.p['claim']):
                raise ValueError('Another allocated winner owns this condition')
            if self.allocation()[0] != 'RUNNING':
                return False
            self.s.setdefault('started_utc', utc())
            self.s['status'] = 'allocated; logging initialization'
            replacement = self.row().replace('[run](' + self.p['run_url'] + ')', '—')
            index = index.replace(matches[0], replacement)
            # Put the allocated winner in Running before any W&B creation.
            index = index.replace(replacement + '\n', '')
            index = re.sub(r'(## Running Experiments\n\n\|[^\n]*\n\|[^\n]*\n)',
                           lambda m: m[1] + replacement + '\n', index, count=1)
            self.git('read-tree', head)
            blob = self.git('hash-object', '-w', '--stdin', data=index)
            self.git('update-index', '--cacheinfo', '100644,' + blob + ',experiments/README.md')
            tree = self.git('write-tree')
            commit = self.git('commit-tree', tree, '-p', head, data='Claim allocated science condition\n')
            try:
                self.git('push', '--quiet', 'origin', commit + ':refs/heads/experiment-queue')
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError('Allocated claim publication failed')
        # The private credential file is never passed as a command-line argument.
        secret = Path(self.p['wandb_auth']).read_text().strip()
        match = re.search(r'(?:WANDB_API_KEY|api_key)\s*[=:]\s*["\']?([^\s"\']+)', secret, re.I)
        key = match[1] if match else secret
        query = 'mutation($name:String,$config:JSONString){upsertBucket(input:{name:$name,modelName:"Noisy-RLVR",entityName:"alliance",config:$config}){bucket{name}}}'
        request = urllib.request.Request('https://api.wandb.ai/graphql', data=json.dumps(dict(query=query,
            variables=dict(name=self.p['run_id'], config=json.dumps({k:dict(value=v) for k,v in self.p['provenance'].items()})))).encode(),
            headers={'Content-Type':'application/json','Authorization':'Basic '+base64.b64encode(('api:'+key).encode()).decode()})
        result = json.load(urllib.request.urlopen(request, timeout=45))
        if result.get('errors') or result['data']['upsertBucket']['bucket']['name'] != self.p['run_id']:
            raise RuntimeError('W&B initialization failed')
        # Publish assigned child and genuine W&B identity atomically with the row.
        self.git('fetch', '--quiet', 'origin', 'refs/heads/experiment-queue')
        head = self.git('rev-parse', 'FETCH_HEAD')
        index = self.git('show', head + ':experiments/README.md') + '\n'
        line = next(line for line in index.splitlines() if marker in line and line.startswith('|'))
        if self.p['claim'] not in line or self.allocation()[0] != 'RUNNING':
            raise ValueError('Allocated claim changed before protocol publication')
        child_path = 'experiments/' + self.p['record']
        child = self.git('show', head + ':' + child_path)
        if not re.search(r'^Machine ID: (unassigned|`?' + re.escape(machine) + r'`?)$', child, re.M):
            raise ValueError('Foreign scientific record')
        child = re.sub(r'^Machine ID: .+$', 'Machine ID: `' + machine + '`', child, flags=re.M)
        child = re.sub(r'^W&B: .+$', 'W&B: [run](' + self.p['run_url'] + ')', child, flags=re.M)
        child = child.replace('Queued; no allocation or execution claim.', 'Allocated on ' + self.p['owner'] + '; optimizer startup pending.')
        self.git('read-tree', head)
        for path, data in [('experiments/README.md', index.replace(line, self.row())), (child_path, child)]:
            blob = self.git('hash-object', '-w', '--stdin', data=data)
            self.git('update-index', '--cacheinfo', '100644,' + blob + ',' + path)
        tree = self.git('write-tree')
        commit = self.git('commit-tree', tree, '-p', head, data='Publish allocated science identity and protocol\n')
        self.git('push', '--quiet', 'origin', commit + ':refs/heads/experiment-queue')
        self.p['logging_ready'] = True
        save(self.p['profile'], self.p)
        return True

    def tick(self):
        if not self.bootstrap():
            self.persist()
            return
        super().tick()


def worker(profile):
    p = json.loads(Path(profile).read_text())
    root = Path(p['root'])
    job = int(os.environ['SLURM_JOB_ID'])
    if fields(job)['JobState'] != 'RUNNING':
        raise RuntimeError('Worker allocation is unavailable')
    child = None
    token = None
    seen = set()
    last_gpu = 0
    gpu = {}
    while True:
        request = root / 'supervisor/start.json'
        if request.exists():
            action = json.loads(request.read_text())
            if action['job_id'] == job and action['token'] not in seen and not (root / 'STOP').exists():
                if child is not None and child.poll() is None:
                    raise RuntimeError('A native workload is already active')
                token = action['token']; seen.add(token)
                env = dict(os.environ, **p['env'], RUN_ROOT=str(root), BASE_MODEL=p['model'],
                           WANDB_RUN_ID=p['run_id'], EXPERIMENT_NAME=p['name'],
                           RESUME_CHECKPOINT=action['checkpoint'], RLVR_WORKLOAD_TOKEN=token)
                secret = Path(p['wandb_auth']).read_text().strip()
                match = re.search(r'(?:WANDB_API_KEY|api_key)\s*[=:]\s*["\']?([^\s"\']+)', secret, re.I)
                env['WANDB_API_KEY'] = match[1] if match else secret
                log = (root / 'logs' / (action['name'] + '.log')).open('a')
                child = subprocess.Popen(p['command'], env=env, cwd=str(root), stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
        stopped = (root / 'STOP').exists()
        stop = root / 'supervisor/stop.json'
        if stop.exists() and json.loads(stop.read_text()).get('token') == token:
            stopped = True
        memory = {}
        group = Path('/proc/self/cgroup').read_text().split('0::',1)[1].strip().lstrip('/')
        cgroup = Path('/sys/fs/cgroup') / group
        for candidate in [cgroup, *cgroup.parents]:
            try:
                cap = (candidate / 'memory.max').read_text().strip()
                if cap != 'max':
                    stats = dict(line.split() for line in (candidate/'memory.stat').read_text().splitlines())
                    used = int((candidate/'memory.current').read_text()) - int(stats.get('inactive_file', 0))
                    observation = dict(limit_bytes=int(cap), used_bytes=used, fraction=used/int(cap))
                    if not memory or int(cap)<memory['limit_bytes']:memory=observation
            except (OSError, ValueError):pass
        if not memory:
            raise RuntimeError('Finite native cgroup memory limit unavailable')
        if memory['fraction'] >= .90:
            stopped = True
        if stopped and child is not None:
            marker = ('RLVR_WORKLOAD_TOKEN=' + token).encode()
            for sig in (signal.SIGTERM, signal.SIGKILL):
                for proc in Path('/proc').iterdir():
                    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
                    try:
                        if proc.stat().st_uid==os.getuid() and marker in (proc/'environ').read_bytes().split(b'\0'):
                            os.kill(int(proc.name),sig)
                    except OSError:pass
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
            if child.poll() is None:raise RuntimeError('Exact workload termination failed')
        if time.time() - last_gpu >= 60:
            identity = os.environ.get('SLURM_JOB_GPUS', '')
            if not identity or ',' in identity:raise RuntimeError('Expected one scheduler-assigned physical GPU')
            telemetry = subprocess.run(['nvidia-smi', '--id=' + identity,
                '--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,text=True,timeout=15,check=True)
            gpu = dict(physical_id=identity, telemetry=telemetry.stdout.strip())
            last_gpu=time.time()
        health = dict(job_id=job, token=token, running=child is not None and child.poll() is None,
                      child_pid=child.pid if child else None, returncode=child.poll() if child else None,
                      checked_epoch=time.time(), checked_utc=utc(), memory=memory, gpu=gpu)
        save(root/'supervisor/worker.json', health)
        if token:
            save(root/'supervisor'/ (action['name']+'-memory.json'), health)
        time.sleep(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--worker', action='store_true')
    args=parser.parse_args()
    os.environ['TZ']='UTC';time.tzset()
    if args.worker:
        worker(args.profile)
    else:
        controller=Native(args.profile)
        lock=(controller.root/'supervisor/controller.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                controller.tick();controller.s['last_success_utc']=utc();controller.persist()
            except ValueError as exc:
                try:controller.stop_container()
                except Exception as stop_error:controller.s['protective_stop_error']=str(stop_error)
                controller.s.update(status='needs_attention',reason=str(exc),ended_utc=utc())
                controller.persist()
                print(utc(),type(exc).__name__,str(exc),flush=True)
            except Exception as exc:
                controller.s['observation_error']=type(exc).__name__+': '+str(exc)
                controller.persist()
                print(utc(),type(exc).__name__,str(exc),flush=True)
            time.sleep(60)


if __name__ == '__main__':
    main()
