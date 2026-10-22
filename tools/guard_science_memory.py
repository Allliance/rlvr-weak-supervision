"""Two-second host cgroup guard for one exact, independently retained workload."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    peak = 0
    while True:
        result = subprocess.run(["sudo", "-n", "docker", "inspect", "--format",
                                 "{{.Id}}|{{.State.Running}}|{{.State.Pid}}", args.container],
                                text=True, capture_output=True, timeout=10)
        if result.returncode:
            if "No such object" in result.stderr or "No such container" in result.stderr:
                return
            raise RuntimeError("Docker observation unavailable; service must restart the guard")
        identity, running, pid = result.stdout.strip().split("|")
        if identity != args.container:
            raise ValueError("Exact container identity mismatch")
        if running != "true":
            return
        group = next(line.split(":", 2)[2] for line in Path("/proc", pid, "cgroup").read_text().splitlines()
                     if line.startswith("0::"))
        root = Path("/sys/fs/cgroup") / group.lstrip("/")
        current = int((root / "memory.current").read_text())
        limit = int((root / "memory.max").read_text())
        stats = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
        working_set = max(0, current - int(stats.get("inactive_file", "0")))
        peak = max(peak, working_set)
        pressure = working_set >= limit * 0.9
        output = Path(args.output)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(checked_epoch=time.time(), container_id=identity,
                                            current_bytes=current, working_set_bytes=working_set,
                                            peak_bytes=peak, limit_bytes=limit,
                                            protective_stop=pressure)) + "\n")
        temporary.replace(output)
        if pressure:
            subprocess.run(["sudo", "-n", "docker", "stop", "--time", "20", identity],
                           check=True, capture_output=True, timeout=40)
            return
        time.sleep(2)


if __name__ == "__main__":
    main()
