"""Bounded single-experiment Slurm/ROCm controller; all host settings stay local."""
import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import time
import zipfile


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(argv, *, data=None, timeout=60):
    result = subprocess.run(argv, input=data, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        # Never include an environment, credentials, or secret-bearing arguments.
        raise RuntimeError(f"{Path(argv[0]).name}: {result.stderr[-500:]}")
    return result.stdout.strip()


def notify():
    address = os.environ.get("NOTIFY_SOCKET")
    if address:
        if address.startswith("@"):
            address = "\0" + address[1:]
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as stream:
            stream.connect(address)
            stream.sendall(b"READY=1\nWATCHDOG=1")


class Controller:
    def __init__(self, profile):
        self.p = json.loads(Path(profile).read_text())
        self.p["profile"] = str(Path(profile).resolve())
        self.root = Path(self.p["root"])
        self.path = self.root / "supervisor/state.json"
        self.s = json.loads(self.path.read_text()) if self.path.exists() else {
            "attempt": 0, "failures": [], "renewals": 0,
            "completed_updates": 0, "last_progress": time.time(), "status": "preparing"}

    def node(self, argv, timeout=60):
        return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                    self.p["node"], shlex.join(argv)], timeout=timeout)

    def git(self, *argv, data=None):
        return run(["git", "--git-dir=" + self.p["queue_git"], *argv], data=data)

    def queue(self, *, publish=False):
        """A successful fast-forward push is required before a new launch."""
        for _ in range(5):
            self.git("fetch", "--quiet", "origin", "refs/heads/experiment-queue")
            head = self.git("rev-parse", "FETCH_HEAD")
            consent = self.git("show", head + ":.agents/experiment-orchestration.md")
            if re.findall(r"^Status: (.+)$", consent, re.M) != ["enabled"]:
                return False
            if not re.search(r"^Consent: \S", consent, re.M):
                return False
            index = self.git("show", head + ":experiments/README.md") + "\n"
            matches = [line for line in index.splitlines() if self.p["run_url"] in line]
            if len(matches) != 1:
                raise ValueError("Expected exactly one owned experiment in the live queue")
            cells = [v.strip() for v in matches[0].strip("|").split("|")]
            if self.p["claim"] not in cells:
                raise ValueError("Published claim does not belong to this controller")
            if not publish:
                return True
            row = self.row()
            # Each table has the same schema, so terminal transitions are atomic.
            index = index.replace(matches[0] + "\n", "")
            section = ("Finished" if self.s["status"] in ("completed", "user_stopped", "needs_attention")
                       else "Running" if self.s.get("allocation_state") == "RUNNING" else "Pending")
            pattern = r"(## " + section + r" Experiments\n\n\|[^\n]*\n\|[^\n]*\n)"
            index, count = re.subn(pattern, lambda m: m[1] + row + "\n", index)
            if count != 1:
                raise ValueError("Invalid live queue table")
            if index.strip() == self.git("show", head + ":experiments/README.md").strip():
                return True
            self.git("read-tree", head)
            blob = self.git("hash-object", "-w", "--stdin", data=index)
            self.git("update-index", "--cacheinfo", "100644," + blob + ",experiments/README.md")
            tree = self.git("write-tree")
            commit = self.git("commit-tree", tree, "-p", head, data="Update science experiment state\n")
            try:
                self.git("push", "--quiet", "origin", commit + ":refs/heads/experiment-queue")
                self.s["queue_synced_utc"] = utc()
                return True
            except RuntimeError:
                continue
        raise RuntimeError("Could not publish experiment state after five fetch/push attempts")

    def row(self):
        status = self.s["status"] + "; " + str(self.s["completed_updates"]) + "/480 updates"
        if self.s.get("reason"):
            status += "; " + self.s["reason"].replace("|", "/").replace("\n", " ")[:200]
        values = ["1", "[Qwen3-1.7B science 2048](qwen3-science-2048.md)",
                  "gpu=1; vram=288GiB; cpu=16; ram=128GiB", self.p["owner"],
                  self.p["claim"], str(self.p["job_id"]), self.s.get("started_utc", "—"),
                  self.s.get("ended_utc", "—"), "[run](" + self.p["run_url"] + ")", status]
        return "| " + " | ".join(values) + " |"

    def allocation(self):
        output = run(["squeue", "-h", "-u", self.p["user"], "-o", "%i|%u|%T|%N|%e"])
        rows = [line.split("|") for line in output.splitlines()
                if line.strip() and line.split("|")[0] == str(self.p["job_id"])]
        if not rows:
            return "ABSENT", 0
        if len(rows) != 1 or rows[0][1] != self.p["user"]:
            raise ValueError("Ambiguous or foreign allocation")
        _, _, state, node, end = rows[0]
        if state == "RUNNING" and node != self.p["node"]:
            raise ValueError("Unexpected allocation node")
        remaining = (dt.datetime.fromisoformat(end).replace(tzinfo=dt.timezone.utc).timestamp()
                     - time.time()) if state == "RUNNING" else 0
        self.s["allocation_expiry_utc"] = end
        return state, remaining

    def checkpoints(self):
        tracker = self.root / "checkpoints/latest_checkpointed_iteration.txt"
        if not tracker.is_file():
            return 0, ""
        value = tracker.read_text().strip()
        if not value.isdecimal():
            return 0, ""  # A writer may be between truncate and the small tracker write.
        committed = int(value)
        valid = []
        for path in (self.root / "checkpoints").glob("global_step_*"):
            step = int(path.name.rsplit("_", 1)[1])
            if step > committed:
                continue
            required = [path / "data.pt", *[path / "actor" / (kind + "_world_size_1_rank_0.pt")
                        for kind in ("model", "optim", "extra_state")]]
            if all(f.is_file() and zipfile.is_zipfile(f) for f in required):
                valid.append((step, str(path)))
        return max(valid, default=(0, ""))

    def container(self):
        name = self.s.get("container")
        if not name:
            return None
        output = self.node(["sudo", "-n", "docker", "ps", "-a", "--filter", "name=^/" + name + "$",
                            "--format", "{{.ID}}|{{.State}}"])
        if not output:
            return None
        rows = output.splitlines()
        if len(rows) != 1:
            raise ValueError("Ambiguous exact container identity")
        identity, state = rows[0].split("|")
        if not self.s.get("container_id"):
            label = self.node(["sudo", "-n", "docker", "inspect", "--format",
                               '{{index .Config.Labels "rlvr.claim"}}', name])
            if label != self.p["claim"]:
                raise ValueError("Exact container name is not owned by this published claim")
            self.s["container_id"] = self.node(["sudo", "-n", "docker", "inspect", "--format", "{{.Id}}", name])
        if self.s.get("container_id") and not self.s["container_id"].startswith(identity):
            raise ValueError("Container identity changed")
        return state

    def stop_container(self):
        identity = self.s.get("container_id")
        if not identity and self.s.get("container"):
            self.container()  # Reconcile a lost launch response using exact name + claim label.
            identity = self.s.get("container_id")
        if not identity:
            return
        output = self.node(["sudo", "-n", "docker", "ps", "--filter", "id=" + identity,
                            "--format", "{{.ID}}"])
        if output:
            if len(output.splitlines()) != 1 or not identity.startswith(output):
                raise ValueError("Unverified exact workload identity")
            self.node(["sudo", "-n", "docker", "stop", "--time", "30", identity], timeout=50)
            if self.node(["sudo", "-n", "docker", "ps", "--filter", "id=" + identity, "--format", "{{.ID}}"]):
                raise RuntimeError("Exact workload failed to stop")

    def launch(self, recovery=False):
        if not self.queue() or (self.root / "STOP").exists():
            return False
        if self.s["completed_updates"] and not self.checkpoints()[0]:
            raise ValueError("Recovery requires a complete checkpoint")
        self.s["attempt"] += 1
        step, path = self.checkpoints()
        if recovery:
            # Record before starting: failed/ambiguous launch responses also consume budget.
            self.s["failures"].append({"time": time.time(), "checkpoint": step})
        self.s["completed_updates"] = step
        self.s["resume_checkpoint"] = path
        self.s["attempt_started"] = time.time()
        self.s["last_progress"] = time.time()
        self.s["container"] = self.p["container_prefix"] + "-a" + str(self.s["attempt"])
        self.s["log_offset"] = 0
        self.s["log_tail"] = ""
        self.s.pop("container_id", None)
        self.persist()
        # The launcher verifies scheduler GPU ownership, exposes one render device,
        # and returns only after Docker has accepted the detached workload.
        identity = self.node(["bash", self.p["node_launcher"], str(self.p["job_id"]),
                              self.s["container"]], timeout=90).splitlines()[-1]
        if not re.fullmatch(r"[0-9a-f]{64}", identity):
            raise ValueError("Uncertain Docker launch; reconcile the exact name before retry")
        self.s["container_id"] = identity
        self.s.setdefault("started_utc", utc())
        self.s["status"] = "starting"
        return True

    def renew(self):
        if self.s["renewals"] >= self.p["max_renewals"]:
            raise ValueError("Finite allocation renewal budget exhausted")
        if not self.queue() or (self.root / "STOP").exists():
            return
        # Never overlap GPU reservations; old workload is already checkpointed/stopped.
        if self.s.get("allocation_state") not in {
                "ABSENT", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}:
            return
        token = self.p["claim"] + "-lease-" + str(self.s["renewals"] + 1)
        output = run(["squeue", "-h", "-u", self.p["user"], "-o", "%i|%k"])
        matches = [line.split("|")[0] for line in output.splitlines() if line.endswith("|" + token)]
        if len(matches) > 1:
            raise ValueError("Duplicate allocation request")
        if matches:
            job = matches[0]
        else:
            job = run(["sbatch", "--parsable", "--job-name=Qwen3-1.7B-science-2048",
                       "--partition=" + self.p["partition"], "--nodelist=" + self.p["node"],
                       "--nodes=1", "--ntasks=1", "--cpus-per-task=16", "--mem=128G", "--gres=gpu:1",
                       "--time=24:00:00", "--time-min=01:00:00", "--comment=" + token,
                       "--output=" + str(self.root / "logs/allocation-%j.log"),
                       self.p["holder"]]).split(";")[0]
        if not job.isdigit():
            raise ValueError("Uncertain allocation submission")
        self.p["job_id"] = int(job)
        save(self.p["profile"], self.p)
        self.s["renewals"] += 1
        self.s["status"] = "pending_allocation"
        self.s.pop("container", None)
        self.s.pop("container_id", None)

    def metrics(self):
        name = self.s.get("container")
        if not name:
            return []
        path = self.root / "logs" / (name + ".log")
        if not path.exists():
            return []
        with path.open(errors="replace") as f:
            offset = self.s.get("log_offset", 0)
            f.seek(offset if offset <= path.stat().st_size else 0)
            text = self.s.get("log_tail", "") + f.read()
            self.s["log_offset"] = f.tell()
            self.s["log_tail"] = text[-8192:]
        progress = [int(v) for v in re.findall(r"step:(\d+)\s+-", text)]
        progress += [int(v) for v in re.findall(r"training/global_step[:=]\s*(\d+)", text)]
        if progress and max(progress) > self.s["completed_updates"]:
            self.s["completed_updates"] = max(progress)
            self.s["last_progress"] = time.time()
            self.s["status"] = "training"
        faults = []
        scalars = re.findall(r"actor/(grad_norm|pg_loss|kl_loss|lr):\s*(?:np\.\w+\()?([-+\deE.nanif]+)", text)
        for key, value in scalars:
            self.s.setdefault("finite_metrics", {})[key] = float(value) if math.isfinite(float(value)) else value
        if any(not math.isfinite(float(v)) for _, v in scalars):
            faults.append("Non-finite optimizer metric")
        durations = re.findall(r"timing_s/step:([\d.]+)", text)
        if durations:
            self.s["longest_iteration_seconds"] = max(float(x) for x in durations[-10:])
        if "Error executing job with overrides" in text or "OutOfMemoryError" in text:
            faults.append("Workload exception; inspect exact attempt log")
        if "Disk quota exceeded" in text or "No space left on device" in text:
            faults.append("Storage write failure")
        return faults

    def persist(self):
        self.s["checked_utc"] = utc()
        save(self.path, self.s)

    def tick(self):
        if self.s["status"] == "needs_attention":
            self.stop_container()
        if (self.root / "STOP").exists():
            self.stop_container()
            if self.s["status"] != "completed":
                self.s.update(status="user_stopped", ended_utc=self.s.get("ended_utc", utc()))
            self.persist()
        state, remaining = self.allocation()
        self.s["allocation_state"] = state
        if not (self.root / "STOP").exists() and self.s["status"] not in ("completed", "user_stopped", "needs_attention"):
            faults = self.metrics()
            checkpoint, path = self.checkpoints()
            self.s["checkpoint_step"], self.s["checkpoint"] = checkpoint, path
            running = self.container() == "running"
            if running:
                guard = self.root / "supervisor" / (self.s["container"] + "-memory.json")
                heartbeat = json.loads(guard.read_text()).get("checked_epoch", 0) if guard.exists() else 0
                if time.time() - heartbeat > 30 and time.time() - self.s["attempt_started"] > 90:
                    self.s["guard_restarts"] = self.s.get("guard_restarts", 0) + 1
                    if self.s["guard_restarts"] > 2:
                        self.stop_container()
                        raise ValueError("Memory guard failed to recover a fresh heartbeat")
                    self.node(["sudo", "-n", "systemctl", "restart",
                               "rlvr-science-guard-" + self.s["container"] + ".service"])
                else:
                    self.s["guard_restarts"] = 0
            self.s["storage_free_bytes"] = shutil.disk_usage(self.root).free
            if self.s["storage_free_bytes"] < 50 * 1024**3:
                self.stop_container()
                raise ValueError("Less than 50 GiB free; preserve checkpoints and diagnose storage")
            if running and time.time() - self.s.get("last_full_check", 0) >= 900:
                self.s["container_stats"] = self.node(["sudo", "-n", "docker", "stats", "--no-stream",
                    "--format", "{{json .}}", self.s["container_id"]])
                self.s["gpu_telemetry"] = self.node(["/opt/rocm/bin/rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"])
                self.s["last_full_check"] = time.time()
            if self.s["completed_updates"] >= 480 and checkpoint >= 480 and not running:
                self.s.update(status="completed", ended_utc=utc())
            elif state == "RUNNING" and remaining < 180:
                self.stop_container()
                self.s["status"] = "awaiting_allocation_expiry"
            elif state in {"ABSENT", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                           "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}:
                self.stop_container()
                self.renew()
            elif state == "RUNNING":
                startup_age = time.time() - self.s.get("attempt_started", time.time())
                threshold = (self.p["stall_seconds"] if not self.s["completed_updates"]
                             else max(900, 4 * self.s.get("longest_iteration_seconds", 0)))
                stalled = running and time.time() - self.s["last_progress"] > threshold
                if faults or stalled or (self.s.get("container") and not running and startup_age > 60):
                    if self.s["completed_updates"] >= 480 and checkpoint >= 480:
                        self.s.update(status="completed", ended_utc=utc())
                    else:
                        recent = [f for f in self.s["failures"] if time.time() - f["time"] < 21600]
                        if faults and "Non-finite optimizer metric" in faults:
                            self.stop_container()
                            raise ValueError("Non-finite metrics require diagnosis before recovery")
                        if "Storage write failure" in faults:
                            self.stop_container()
                            raise ValueError("Storage write failed; preserve checkpoints and diagnose quota/capacity")
                        if len(recent) >= 3 or sum(f["checkpoint"] >= checkpoint for f in recent) >= 2:
                            self.stop_container()
                            raise ValueError("Bounded recovery exhausted")
                        if not checkpoint:
                            self.stop_container()
                            raise ValueError("Failed startup requires diagnosis; no complete recovery checkpoint")
                        if not self.queue() or (self.root / "STOP").exists():
                            self.s["status"] = "recovery_parked"
                            self.persist()
                            return
                        self.stop_container()
                        self.launch(recovery=True)
                elif not self.s.get("container"):
                    self.launch()
        self.persist()
        if time.time() - self.s.get("last_queue_check", 0) >= 600 or self.s["status"] != self.s.get("published_status"):
            if self.queue(publish=True):
                self.s["last_queue_check"] = time.time()
                self.s["published_status"] = self.s["status"]
            self.persist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    controller = Controller(args.profile)
    with (controller.root / "supervisor/controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                controller.tick()
                controller.s.pop("observation_error", None)
                controller.s["last_success_utc"] = utc()
                controller.persist()
                notify()
            except ValueError as exc:
                # Only the previously verified immutable container ID may be stopped.
                # Never act on a replacement bearing the same name.
                try:
                    controller.stop_container()
                except Exception as stop_error:
                    controller.s["protective_stop_error"] = str(stop_error)
                controller.s.update(status="needs_attention", reason=str(exc), ended_utc=utc())
                controller.persist()
                print(utc(), type(exc).__name__, str(exc), flush=True)
            except Exception as exc:
                # An unavailable observation never authorizes stopping/launching.
                controller.s["observation_error"] = type(exc).__name__ + ": " + str(exc)
                controller.persist()
                print(utc(), type(exc).__name__, str(exc), flush=True)
            if args.once:
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
