import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
import science_slurm


class SlurmControls(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        (self.root/'supervisor').mkdir();(self.root/'checkpoints').mkdir()
        self.profile=self.root/'profile.json'
        self.profile.write_text(json.dumps(dict(root=str(self.root),job_id=0,max_renewals=4,
            claim='owned',run_id='identity',run_url='https://wandb.ai/alliance/Noisy-RLVR/runs/identity')))
        self.controller=science_slurm.Native(self.profile)

    def test_disabled_consent_and_stop_cannot_submit(self):
        self.controller.enabled=Mock(return_value=False)
        with patch.object(science_slurm,'run') as run:
            self.controller.renew();run.assert_not_called()
        self.controller.enabled=Mock(return_value=True)
        (self.root/'STOP').touch()
        with patch.object(science_slurm,'run') as run:
            self.controller.renew();run.assert_not_called()

    def test_lost_submission_response_cannot_create_a_second_job(self):
        self.controller.enabled=Mock(return_value=True)
        self.controller.allocation=Mock(return_value=('ABSENT',0))
        self.controller.s['pending_submission']='owned-lease-1'
        with patch.object(science_slurm,'run',return_value='') as run:
            with self.assertRaisesRegex(ValueError,'Uncertain submission'):
                self.controller.renew()
            self.assertEqual(run.call_count,1)

    def test_foreign_or_stale_worker_is_never_reported_running(self):
        self.controller.p['job_id']=42
        self.controller.s['container_id']='owned'
        self.controller.health=Mock(return_value=dict(job_id=43,token='owned',running=True,checked_epoch=10**20))
        self.assertIsNone(self.controller.container())
        self.controller.health=Mock(return_value=dict(job_id=42,token='other',running=True,checked_epoch=10**20))
        self.assertIsNone(self.controller.container())
        self.controller.health=Mock(return_value=dict(job_id=42,token='owned',running=True,checked_epoch=1))
        self.assertIsNone(self.controller.container())

    def test_no_checkpoint_cannot_resume_a_progressed_run(self):
        self.controller.queue=Mock(return_value=True)
        self.controller.allocation=Mock(return_value=('RUNNING',3600))
        self.controller.s['completed_updates']=25
        with self.assertRaisesRegex(ValueError,'Complete recovery checkpoint'):
            self.controller.launch(recovery=True)
        self.assertFalse((self.root/'supervisor/start.json').exists())

    def test_ambiguous_gpu_managers_never_create_another_producer(self):
        result=type('Result',(),{'stdout':'[111, 222]'})()
        with patch.object(science_slurm.subprocess,'run',return_value=result), \
             patch.object(science_slurm.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'More than one exact'):
                self.controller.ensure_manager()
            launch.assert_not_called()

    def test_diagnosed_manager_error_blocks_automatic_replay(self):
        (self.root/'supervisor/manager-error.json').write_text(json.dumps(dict(job_id=0,error='GPU preflight failed')))
        with patch.object(science_slurm.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'inspect manager-error'):
                self.controller.ensure_manager()
            launch.assert_not_called()

    def test_v1_and_v2_memory_limits_both_enforce_working_set(self):
        for version in [1,2]:
            base=self.root/str(version)
            path=base/('memory/job' if version==1 else 'job');path.mkdir(parents=True)
            (path/('memory.limit_in_bytes' if version==1 else 'memory.max')).write_text('4096')
            (path/('memory.usage_in_bytes' if version==1 else 'memory.current')).write_text('3584')
            prefix='total_' if version==1 else ''
            (path/'memory.stat').write_text(prefix+'inactive_file 512\n'+prefix+'active_file 512\n')
            result=science_slurm.working_memory('8:memory:/job\n' if version==1 else '0::/job\n',base)
            self.assertEqual(result['cgroup_version'],version)
            self.assertEqual(result['used_bytes'],2560)
            self.assertEqual(result['fraction'],.625)


if __name__=='__main__':unittest.main()
