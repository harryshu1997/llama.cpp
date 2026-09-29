"""Wait for the shared rig, then execute the already staged finite Pixel sweep."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path("/mnt/storage/s42-pixel10pro-dense-gemv-20260923-v1")
LOCK = "/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock"


def record(state, **values):
    row = {"at": datetime.now(timezone.utc).isoformat(), "state": state, **values}
    with (ROOT / "QUEUE_EVENTS.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def main():
    if any((ROOT / name).exists() for name in ("run1", "RUN.log", "DONE.json")):
        raise RuntimeError("sweep output already exists; preserve it")
    attempt = 0
    while True:
        attempt += 1
        record("WAITING_FOR_LOCK", attempt=attempt)
        result = subprocess.run(
            ["flock", "-w", "900", "-E", "75", LOCK, "bash", str(ROOT / "RUN.sh")],
            stdin=subprocess.DEVNULL, check=False,
        )
        if result.returncode == 75:
            record("LOCK_WAIT_EXPIRED", attempt=attempt)
            continue
        record("SWEEP_FINISHED", attempt=attempt, exit_code=result.returncode)
        if result.returncode:
            return result.returncode
        with (ROOT / "SWEEP_AUDIT.json").open("x") as output:
            audit = subprocess.run(
                [sys.executable, str(ROOT / "analyze_pixel_gemv.py"), str(ROOT / "run1")],
                stdin=subprocess.DEVNULL, stdout=output, check=False,
            )
        record("AUDIT_FINISHED", exit_code=audit.returncode)
        return audit.returncode


if __name__ == "__main__":
    raise SystemExit(main())
