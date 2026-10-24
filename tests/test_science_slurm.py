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


if __name__=='__main__':unittest.main()
