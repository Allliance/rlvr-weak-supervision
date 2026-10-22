"""CPU checks for ownership, checkpoint completeness, stop and consent controls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location("science_control", Path(__file__).parents[1] / "tools/science_control.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Controls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "supervisor").mkdir()
        (self.root / "checkpoints").mkdir()
        self.profile = self.root / "profile.json"
        self.profile.write_text(json.dumps(dict(root=str(self.root), job_id=42, run_url="https://example.test/run",
                                               claim="owned", stall_seconds=900)))
        self.controller = module.Controller(self.profile)

    def test_incomplete_checkpoint_is_never_a_recovery_point(self):
        root = self.root / "checkpoints/global_step_5"
        (root / "actor").mkdir(parents=True)
        for f in [root / "data.pt", root / "actor/model_world_size_1_rank_0.pt",
                  root / "actor/optim_world_size_1_rank_0.pt"]:
            with zipfile.ZipFile(f, "w") as archive:
                archive.writestr("archive/data.pkl", b"fixture")
        self.assertEqual(self.controller.checkpoints(), (0, ""))
        with zipfile.ZipFile(root / "actor/extra_state_world_size_1_rank_0.pt", "w") as archive:
            archive.writestr("archive/data.pkl", b"fixture")
        # Even closed shard files are not committed until the tracker advances.
        self.assertEqual(self.controller.checkpoints(), (0, ""))
        (self.root / "checkpoints/latest_checkpointed_iteration.txt").write_text("5")
        self.assertEqual(self.controller.checkpoints(), (5, str(root)))

    def test_stop_works_when_scheduler_is_unavailable(self):
        (self.root / "STOP").touch()
        self.controller.stop_container = Mock()
        self.controller.allocation = Mock(side_effect=RuntimeError("scheduler unavailable"))
        with self.assertRaisesRegex(RuntimeError, "scheduler unavailable"):
            self.controller.tick()
        self.controller.stop_container.assert_called_once()
        self.assertEqual(self.controller.s["status"], "user_stopped")

    def test_active_transitional_allocation_cannot_renew(self):
        self.controller.p.update(max_renewals=3)
        self.controller.s["renewals"] = 0
        self.controller.queue = Mock(return_value=True)
        for state in ["CONFIGURING", "SUSPENDED", "REQUEUED", "COMPLETING"]:
            self.controller.s["allocation_state"] = state
            self.controller.renew()
            self.assertEqual(self.controller.s["renewals"], 0)

    def test_progress_survives_large_warning_bursts_and_numpy_metrics(self):
        (self.root / "logs").mkdir()
        self.controller.s["container"] = "owned-attempt"
        path = self.root / "logs/owned-attempt.log"
        path.write_text("step:3 - actor/grad_norm:np.float64(0.25) - timing_s/step:80.5\n" + "warning\n" * 400000)
        self.assertEqual(self.controller.metrics(), [])
        self.assertEqual(self.controller.s["completed_updates"], 3)
        self.assertEqual(self.controller.s["finite_metrics"]["grad_norm"], 0.25)
        with path.open("a") as f:
            f.write("step:4 - actor/grad_norm:np.float64(-inf)\n")
        self.assertIn("Non-finite optimizer metric", self.controller.metrics())

    def test_failed_launch_consumes_recovery_budget(self):
        self.controller.p.update(container_prefix="owned", node_launcher="known-launcher")
        self.controller.queue = Mock(return_value=True)
        self.controller.checkpoints = Mock(return_value=(5, "/known/checkpoint"))
        self.controller.node = Mock(side_effect=RuntimeError("lost launch response"))
        with self.assertRaisesRegex(RuntimeError, "lost launch response"):
            self.controller.launch(recovery=True)
        self.assertEqual(len(self.controller.s["failures"]), 1)
        self.assertEqual(self.controller.s["failures"][0]["checkpoint"], 5)

    def test_disabled_and_invalid_consent_prevent_launch(self):
        for consent in ["Status: disabled\nConsent: explicitly disabled", "Status: enabled", ""]:
            with self.subTest(consent=consent):
                self.controller.git = Mock(side_effect=["", "head", consent])
                self.assertFalse(self.controller.queue())
                self.assertEqual(self.controller.git.call_count, 3)

    def test_foreign_claim_never_authorizes_launch(self):
        self.controller.git = Mock(side_effect=["", "head", "Status: enabled\nConsent: user requested",
                                               "| foreign | https://example.test/run |"])
        with self.assertRaisesRegex(ValueError, "claim"):
            self.controller.queue()

    def test_stop_retains_allocation_and_never_renews(self):
        (self.root / "STOP").touch()
        self.controller.allocation = Mock(return_value=("RUNNING", 3600))
        self.controller.stop_container = Mock()
        self.controller.renew = Mock()
        self.controller.launch = Mock()
        self.controller.queue = Mock(return_value=True)
        self.controller.tick()
        self.controller.stop_container.assert_called_once()
        self.controller.renew.assert_not_called()
        self.controller.launch.assert_not_called()
        self.assertEqual(self.controller.s["status"], "user_stopped")
        self.assertEqual(self.controller.s["allocation_state"], "RUNNING")

    def test_failed_queue_publication_does_not_publish_success(self):
        self.controller.s["status"] = "training"
        self.controller.p.update(owner="host")
        index = "\n".join("## " + s + " Experiments\n\n| Header |\n|---|\n" for s in ["Pending", "Running", "Finished"])
        index += "| owned | https://example.test/run |\n"
        def git(*args, **kwargs):
            if args[0] == "fetch": return ""
            if args[0] == "rev-parse": return "head"
            if args[0] == "show": return ("Status: enabled\nConsent: user requested"
                if "experiment-orchestration.md" in args[1] else index)
            if args[0] == "push": raise RuntimeError("rejected")
            return "object"
        self.controller.git = git
        with self.assertRaisesRegex(RuntimeError, "five"):
            self.controller.queue(publish=True)
        self.assertNotIn("queue_synced_utc", self.controller.s)


if __name__ == "__main__":
    unittest.main()
