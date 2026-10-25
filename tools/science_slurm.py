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


def working_memory(text=None, mount=Path('/sys/fs/cgroup')):
    text=Path('/proc/self/cgroup').read_text() if text is None else text
    entries=[line.split(':',2) for line in text.splitlines()]
    unified=next((row[2] for row in entries if row[0]=='0' and row[1]==''),None)
    legacy=next((row[2] for row in entries if 'memory' in row[1].split(',')),None)
    version=1 if legacy is not None else 2
    relative=legacy if version==1 else unified
    if relative is None:raise RuntimeError('Memory cgroup is unavailable')
    base=mount if version==2 else mount/'memory'
    current=base/relative.lstrip('/');observations=[]
    for path in [current,*current.parents]:
        if path!=base and base not in path.parents:continue
        try:
            cap=(path/('memory.max' if version==2 else 'memory.limit_in_bytes')).read_text().strip()
            if cap=='max' or int(cap)>=2**60:continue
            usage=int((path/('memory.current' if version==2 else 'memory.usage_in_bytes')).read_text())
            stats=dict(line.split() for line in (path/'memory.stat').read_text().splitlines())
            cache=sum(int(stats.get(('total_' if version==1 else '')+key,stats.get(key,0)))
                      for key in ('inactive_file','active_file'))
            used=max(0,usage-cache)
            observations.append(dict(limit_bytes=int(cap),used_bytes=used,fraction=used/int(cap),
                                     cgroup_version=version,path=str(path)))
        except (OSError,ValueError):continue
    if not observations:raise RuntimeError('Finite native cgroup memory limit unavailable')
    return max(observations,key=lambda item:item['fraction'])


class Native(Controller):
    def remaining_gpu_seconds(self):
        if 'gpu_hour_cap' not in self.p:
            return None
        jobs=set(self.s.get('native_allocations',[]))
        if self.p['job_id']:jobs.add(int(self.p['job_id']))
        used=0
        for job in jobs:
            rows=[line.split('|') for line in run(['sacct','-nP','--allocations','--jobs='+str(job),
                '--format=JobIDRaw,UID,ElapsedRaw,State']).splitlines() if line.split('|')[0]==str(job)]
            if len(rows)!=1 or rows[0][1]!=str(os.getuid()):
                raise ValueError('Owned allocation accounting is unavailable')
            if rows[0][3].split()[0] not in ['COMPLETED','CANCELLED','FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE']:
                raise ValueError('Previous GPU allocation is not confirmed terminal')
            used+=int(rows[0][2])
        self.s.update(native_allocations=sorted(jobs),allocated_gpu_seconds=used)
        return int(self.p['gpu_hour_cap']*3600)-used

    def ensure_manager(self):
        error=self.root/'supervisor/manager-error.json'
        if error.exists() and json.loads(error.read_text()).get('job_id')==self.p['job_id']:
            raise ValueError('Native GPU manager failed; inspect manager-error.json before recovery')
        probe='''import json,os,sys
from pathlib import Path
found=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  a=(p/'cmdline').read_bytes().split(b'\\0'); e=(p/'environ').read_bytes().split(b'\\0')
  if p.stat().st_uid==os.getuid() and b'--worker' in a and os.fsencode(sys.argv[1]) in a and ('SLURM_JOB_ID='+sys.argv[2]).encode() in e:found.append(int(p.name))
 except OSError:pass
print(json.dumps(found))
'''
        environment={key:value for key,value in os.environ.items()
                     if not key.startswith('SLURM_') and key not in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')}
        prefix=['srun','--jobid='+str(self.p['job_id']),'--overlap','--exact','--nodes=1','--ntasks=1']
        observed=subprocess.run(prefix+['--cpus-per-task=1','--mem=256M','--gres=none','/usr/bin/python3','-S','-c',probe,self.p['profile'],str(self.p['job_id'])],
            env=environment,capture_output=True,text=True,timeout=60,check=True)
        managers=json.loads(observed.stdout.strip())
        if len(managers)>1:raise ValueError('More than one exact native GPU manager exists')
        if managers:return
        starts=self.s.setdefault('manager_starts',[])
        recent=[value for value in starts if time.time()-value<21600]
        if len(recent)>=3:raise ValueError('Bounded native manager recovery exhausted')
        starts.append(time.time());self.persist()
        cpus=next(value.split('=',1)[1] for value in self.p['sbatch_args'] if value.startswith('--cpus-per-task='))
        memory=next(value.split('=',1)[1] for value in self.p['sbatch_args'] if value.startswith('--mem='))
        with (self.root/'supervisor/manager-restart.log').open('a') as log:
            subprocess.Popen(prefix+['--cpus-per-task='+cpus,'--mem='+memory,'--gres=gpu:1',
                '/usr/bin/python3','-S',self.p['code']+'/tools/science_slurm.py','--worker','--profile',self.p['profile']],
                env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)

    def publish_pending(self):
        for _ in range(5):
            self.git('fetch','--quiet','origin','refs/heads/experiment-queue')
            head=self.git('rev-parse','FETCH_HEAD')
            consent=self.git('show',head+':.agents/experiment-orchestration.md')
            if re.findall(r'^Status: (.+)$',consent,re.M)!=['enabled'] or not re.search(r'^Consent: \S',consent,re.M):return
            index=self.git('show',head+':experiments/README.md')+'\n'
            marker=']('+self.p['record']+')'
            rows=[line for line in index.splitlines() if line.startswith('|') and marker in line]
            if len(rows)!=1:raise ValueError('Pending protocol is missing or duplicated')
            cells=[value.strip() for value in rows[0].strip('|').split('|')]
            if cells[4]!='—':return
            cells[2]=self.p['resources']
            cells[9]='Candidate '+self.p['claim'].split(':')[0]+'/'+str(self.p['job_id'])+' submitted; awaiting allocation and calibrated startup'
            replacement='| '+' | '.join(cells)+' |'
            if replacement==rows[0]:return
            self.git('read-tree',head)
            blob=self.git('hash-object','-w','--stdin',data=index.replace(rows[0],replacement))
            self.git('update-index','--cacheinfo','100644,'+blob+',experiments/README.md')
            tree=self.git('write-tree')
            commit=self.git('commit-tree',tree,'-p',head,data='Publish pending science candidate\n')
            try:self.git('push','--quiet','origin',commit+':refs/heads/experiment-queue');return
            except RuntimeError:continue
        raise RuntimeError('Pending candidate publication failed')

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
        sequence = health.get('sequence')
        if sequence != self.s.get('worker_sequence'):
            self.s.update(worker_sequence=sequence, worker_observed_epoch=time.time())
            if self.s.get('container'):
                save(self.root/'supervisor'/(self.s['container']+'-memory.json'),
                     dict(health, checked_epoch=time.time()))
        return 'running' if health.get('running') and time.time() - self.s.get('worker_observed_epoch', 0) < 180 else None

    def stop_container(self):
        token = self.s.get('container_id')
        if not token:
            return
        save(self.root / 'supervisor/stop.json', dict(token=token))
        for _ in range(60):
            h = self.health()
            if h.get('token') != token or not h.get('running'):
                return
            if self.allocation()[0] == 'ABSENT':
                return  # Slurm has ended the exact job and its process steps.
            time.sleep(1)
        raise RuntimeError('Exact native workload stop is unverified')

    def launch(self, recovery=False):
        if self.allocation()[0] != 'RUNNING' or not self.queue() or (self.root / 'STOP').exists():
            return False
        step, checkpoint = self.checkpoints()
        if self.s['completed_updates'] and not step:
            raise ValueError('Complete recovery checkpoint is unavailable')
        if step:
            env = {key:value for key,value in os.environ.items()
                   if not key.startswith('SLURM_') and key not in
                   ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL')}
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
        if self.p['job_id']:
            history = run(['sacct','-nP','--allocations','--jobs='+str(self.p['job_id']),
                           '--format=JobIDRaw,UID,State'])
            rows = [line.split('|') for line in history.splitlines()
                    if line.split('|')[0] == str(self.p['job_id'])]
            if len(rows) != 1 or rows[0][1] != str(os.getuid()):
                raise ValueError('Exact ended allocation is unverified')
            if rows[0][2].startswith('CANCELLED'):
                if (self.root/'supervisor'/f'idle-release-{self.p["job_id"]}.json').exists():
                    self.s.update(status='needs_attention',reason='Sustained GPU idleness required allocation release')
                else:
                    (self.root/'STOP').touch()
                    self.s.update(status='user_stopped',reason='Allocation cancelled; no automatic restart')
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
            arguments=list(self.p['sbatch_args'])
            remaining=self.remaining_gpu_seconds()
            if remaining is not None:
                if remaining<=180:raise ValueError('Allocated GPU-hour budget exhausted')
                duration=next(a.split('=',1)[1] for a in arguments if a.startswith('--time='))
                day,clock=duration.split('-',1) if '-' in duration else ('0',duration)
                parts=[int(x) for x in clock.split(':')]
                requested=int(day)*86400+(sum(v*60**(len(parts)-i-1) for i,v in enumerate(parts)) if len(parts)>1 else parts[0]*60)
                arguments=[a for a in arguments if not a.startswith('--time=')]
                arguments.append('--time='+str(min(requested,remaining)//60))
            self.s['pending_submission'] = token
            self.persist()
            job = run(['sbatch', '--parsable', '--comment=' + token, *arguments,
                       self.p['holder'], self.p['profile']]).split(';')[0]
        if not job.isdigit():
            raise ValueError('Uncertain allocation response')
        self.p['job_id'] = int(job)
        if 'gpu_hour_cap' in self.p:
            self.s['native_allocations']=sorted(set(self.s.get('native_allocations',[]))|{int(job)})
        save(self.p['profile'], self.p)
        self.s['renewals'] += 1
        self.s.pop('manager_starts', None)  # Recovery bounds apply to this allocation.
        self.s.pop('pending_submission', None)
        self.s.pop('container', None)
        self.s.pop('container_id', None)
        self.s['status'] = 'pending_allocation'
        self.publish_pending()

    def row(self):
        preparation=('; preparing '+str(self.s.get('verified_noisy_rows',0))+'/1843 verified noisy labels') \
            if self.p.get('noise_generation') and not (self.root/'data/NOISY_DATA_READY').exists() else ''
        values = [str(self.p.get('priority', 10)), '[' + self.p['name'] + '](' + self.p['record'] + ')',
                  self.p['resources'], self.p['owner'], self.p['claim'], str(self.p['job_id']),
                  self.s.get('started_utc', '—'), self.s.get('ended_utc', '—'),
                  '[run](' + self.p['run_url'] + ')', self.s['status'] + preparation + '; ' + str(self.s['completed_updates']) + '/480 updates']
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
            if state == 'PENDING':self.publish_pending()
            return False
        for _ in range(5):
            self.git('fetch', '--quiet', 'origin', 'refs/heads/experiment-queue')
            head = self.git('rev-parse', 'FETCH_HEAD')
            consent = self.git('show', head + ':.agents/experiment-orchestration.md')
            if re.findall(r'^Status: (.+)$', consent, re.M) != ['enabled'] or not re.search(r'^Consent: \S', consent, re.M):
                return False
            index = self.git('show', head + ':experiments/README.md') + '\n'
            marker = '](' + self.p['record'] + ')'
            matches = [line for line in index.splitlines() if marker in line and line.startswith('|')]
            if len(matches) != 1:
                raise ValueError('Expected one pending scientific protocol')
            cells = [v.strip() for v in matches[0].strip('|').split('|')]
            if cells[4] not in ('—', self.p['claim']):
                fields(self.p['job_id'])  # Release only this verified losing candidate.
                run(['scancel',str(self.p['job_id'])])
                raise ValueError('Another allocated winner owns this condition')
            if self.allocation()[0] != 'RUNNING':
                return False
            actual=fields(self.p['job_id'])
            self.p['provenance']['actual_allocation']={k:actual.get(k) for k in
                ['JobId','NodeList','AllocTRES','StartTime','EndTime','Partition']}
            resolved=self.root/'supervisor/resolved-launch.yaml'
            self.p['provenance']['resolved_config_yaml']=resolved.read_text()
            self.s.setdefault('started_utc', utc())
            self.s['status'] = 'allocated; logging initialization'
            replacement = self.row().replace('[run](' + self.p['run_url'] + ')', '—')
            index = index.replace(matches[0], replacement)
            # Put the allocated winner in Running before any W&B creation.
            index = index.replace(replacement + '\n', '')
            index, inserted = re.subn(r'(## Running Experiments\n\n\|[^\n]*\n\|[^\n]*\n)',
                           lambda m: m[1] + replacement + '\n', index, count=1)
            if inserted != 1:
                raise ValueError('Running table shape is invalid; no queue edit published')
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
        if [v.strip() for v in line.strip('|').split('|')][4] != self.p['claim'] or self.allocation()[0] != 'RUNNING':
            raise ValueError('Allocated claim changed before protocol publication')
        child_path = 'experiments/' + self.p['record']
        child = self.git('show', head + ':' + child_path)
        if not re.search(r'^Machine ID: (unassigned|`?' + re.escape(machine) + r'`?)$', child, re.M):
            raise ValueError('Foreign scientific record')
        child = re.sub(r'^Machine ID: .+$', 'Machine ID: `' + machine + '`', child, flags=re.M)
        child = re.sub(r'^W&B: .+$', 'W&B: [run](' + self.p['run_url'] + ')', child, flags=re.M)
        child = re.sub(r'## Main Finding\n.*?(?=\n## Events)',
            '## Main Finding (Running)\n\nAllocated on '+self.p['owner']+'; optimizer startup pending. Native verl 0.5.0/CUDA execution is a runtime adaptation of the existing ROCm science protocol; compare with that version difference disclosed.\n', child, flags=re.S)
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
        if self.allocation()[0]=='RUNNING' and self.s['status'] not in ('completed','user_stopped','needs_attention'):
            self.ensure_manager()
            health=self.health()
            if health.get('job_id')!=self.p['job_id'] or not health.get('manager_ready'):
                self.s['status']='GPU preflight; optimizer startup pending'
                self.persist()
                self.queue(publish=True)
                return
            if self.p.get('noise_generation') and not (self.root/'data/NOISY_DATA_READY').exists():
                progress=self.root/'data/noise-progress.json'
                if progress.exists():
                    actual=json.loads(progress.read_text())
                    if actual.get('verified_rows',0)>self.s.get('verified_noisy_rows',0):
                        self.s.update(verified_noisy_rows=actual['verified_rows'],last_progress=time.time())
                    self.s['preparation_phase']=actual['phase']
        super().tick()


def stop_token(token):
    if not token:return
    marker=('RLVR_WORKLOAD_TOKEN='+token).encode()
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
            try:
                if proc.stat().st_uid==os.getuid() and marker in (proc/'environ').read_bytes().split(b'\0'):
                    os.kill(int(proc.name),sig)
            except OSError:pass
        time.sleep(2)


def worker_loop(profile):
    p = json.loads(Path(profile).read_text())
    root = Path(p['root'])
    job = int(os.environ['SLURM_JOB_ID'])
    if fields(job)['JobState'] != 'RUNNING':
        raise RuntimeError('Worker allocation is unavailable')
    previous_token=None
    previous_path=root/'supervisor/worker.json'
    if previous_path.exists():
        previous=json.loads(previous_path.read_text())
        if previous.get('job_id')==job and previous.get('token'):
            previous_token=previous['token']
            stop_token(previous_token)
    save(root/'supervisor/worker.json',dict(job_id=job,token=None,running=False,
        manager_pid=os.getpid(),manager_ready=False,phase='gpu_preflight',checked_epoch=time.time()))
    probe='''import json,torch,uuid
assert torch.cuda.device_count()==1
p=torch.cuda.get_device_properties(0);free,total=torch.cuda.mem_get_info()
assert total>=79_000_000_000 and free>=total*.90
x=torch.ones((1024,1024),device='cuda',dtype=torch.bfloat16)
assert torch.isfinite(x@x).all();torch.cuda.synchronize()
raw=p.uuid
device_uuid=uuid.UUID(bytes=raw) if isinstance(raw,bytes) and len(raw)==16 else uuid.UUID(str(raw).removeprefix('GPU-'))
print(json.dumps(dict(name=p.name,total_bytes=total,free_bytes=free,uuid='GPU-'+str(device_uuid),logical_device=0,torch=torch.__version__,cuda=torch.version.cuda)))
'''
    proof=subprocess.run([p['python'],'-c',probe],capture_output=True,text=True,
                         timeout=300,check=True,env=dict(os.environ,**p['env']))
    cuda_proof=json.loads(proof.stdout.strip().splitlines()[-1])
    save(root/'supervisor/gpu-preflight.json',cuda_proof)
    child = None
    token = None
    seen = set()
    if previous_token:seen.add(previous_token)  # CPU supervision owns checkpoint-safe recovery.
    last_gpu = 0
    gpu = {}
    sequence = 0
    last_use = time.time()
    cleaned = set()
    while True:
        if child is not None and child.poll() is not None and token not in cleaned:
            stop_token(token)
            cleaned.add(token)
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
                save(root/'supervisor/worker.json',dict(job_id=job,token=token,running=True,checked_epoch=time.time()))
        stopped = (root / 'STOP').exists()
        stop = root / 'supervisor/stop.json'
        if stop.exists() and json.loads(stop.read_text()).get('token') == token:
            stopped = True
        memory = working_memory()
        if memory['fraction'] >= .90:
            stopped = True
        if stopped and child is not None:
            stop_token(token)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:pass
            if child.poll() is None:raise RuntimeError('Exact workload termination failed')
        if time.time() - last_gpu >= 60:
            identity = os.environ.get('SLURM_STEP_GPUS') or os.environ.get('SLURM_JOB_GPUS') or 'unavailable'
            if ',' in identity:raise RuntimeError('Expected one scheduler-assigned physical GPU')
            telemetry = subprocess.run(['nvidia-smi',
                '--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,text=True,timeout=15,check=True)
            if len(telemetry.stdout.strip().splitlines()) != 1:
                raise RuntimeError('NVML exposure does not match the one-GPU allocation')
            if telemetry.stdout.strip().split(',')[0].strip()!=cuda_proof['uuid']:
                raise RuntimeError('CUDA and NVML identify different devices')
            gpu = dict(physical_id=identity, telemetry=telemetry.stdout.strip(),
                       peak_used_mib=max(gpu.get('peak_used_mib',0),float(telemetry.stdout.strip().split(',')[-2])))
            if float(telemetry.stdout.strip().split(',')[-1]) > 0:last_use=time.time()
            last_gpu=time.time()
        if time.time()-last_use >= 3300:
            save(root/'supervisor'/f'idle-release-{job}.json',dict(checked_utc=utc(),idle_seconds=time.time()-last_use))
            fields(job)
            run(['scancel',str(job)])
            return
        sequence += 1
        health = dict(job_id=job, token=token, running=child is not None and child.poll() is None,
                      child_pid=child.pid if child else None, returncode=child.poll() if child else None,
                      checked_epoch=time.time(), checked_utc=utc(), memory=memory, gpu=gpu, sequence=sequence,
                      manager_pid=os.getpid(),manager_ready=True)
        save(root/'supervisor/worker.json', health)
        time.sleep(2)


def worker(profile):
    p=json.loads(Path(profile).read_text())
    # A delayed srun can race the controller's process probe. Never let a second
    # manager touch the first manager's token, health file or workload.
    lock=Path('/dev/shm',f'rlvr-science-manager-{os.getuid()}-{os.environ["SLURM_JOB_ID"]}.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    try:
        worker_loop(profile)
    except Exception as exc:
        save(Path(p['root'])/'supervisor/manager-error.json',dict(job_id=int(os.environ['SLURM_JOB_ID']),
            checked_utc=utc(),error=type(exc).__name__+': '+str(exc)))
        raise
    finally:
        path=Path(p['root'])/'supervisor/worker.json'
        if path.exists():
            health=json.loads(path.read_text())
            if health.get('job_id') == int(os.environ['SLURM_JOB_ID']):stop_token(health.get('token'))


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
                if controller.p.get('logging_ready'):
                    try:controller.queue(publish=True)
                    except Exception as publication:controller.s['publication_error']=str(publication);controller.persist()
                print(utc(),type(exc).__name__,str(exc),flush=True)
            except Exception as exc:
                controller.s['observation_error']=type(exc).__name__+': '+str(exc)
                controller.persist()
                print(utc(),type(exc).__name__,str(exc),flush=True)
            time.sleep(60)


if __name__ == '__main__':
    main()
