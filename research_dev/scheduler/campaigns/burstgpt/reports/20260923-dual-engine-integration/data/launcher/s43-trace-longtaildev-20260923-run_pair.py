#!/usr/bin/env python3
"""Dev-trace pair on the S43 dual-engine phone stack (NPU+GPU zero-copy FFN worker).

One rig lock for the whole pair. Order: baseline preflight, treatment preflight, baseline run
(desktop-baseline), treatment run (energy-aware, phone-assisted). Phone battery / OPLUS notify code
are logged before and after every run stage (no guard; notify 512 = charger latched off -> stop and
report, per the user's rule). After every run the phone session root (worker.log with the S43DUALFFN
lines, router.log, session.log, diagnostics) is copied into <run dir>/phone-session-root.tar.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE = Path("/mnt/storage/s42-trace-v2-20260921-prep/source")
ROOT = Path("/home/zhihao")
LOCK = "/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock"
STAGES = (("baseline", "preflight"), ("treatment", "preflight"), ("baseline", "run"), ("treatment", "run"))
STATUS = ROOT / "s43-trace-longtaildev-20260923-PAIR_STATUS.jsonl"
ATTEMPT = os.environ.get("PAIR_ATTEMPT", "1")
ADB = ["/usr/bin/adb", "-P", "5037", "-s", "3C15AU002CL00000"]
SESSION_ROOT = "/data/local/tmp/s43-dual-session-root-20260923-v1"
LOCK_TIMEOUT_EXIT = 75


def log(**fields):
    fields["at"] = datetime.now(timezone.utc).isoformat()
    with STATUS.open("a") as f:
        f.write(json.dumps(fields) + "\n")


def adb(*args, timeout=120, binary=False):
    return subprocess.run([*ADB, *args], stdin=subprocess.DEVNULL, capture_output=True,
                          text=not binary, timeout=timeout)


def phone_state(label):
    fields = {}
    try:
        for line in adb("shell", "dumpsys", "battery").stdout.splitlines():
            line = line.strip()
            for key in ("level", "status", "voltage", "temperature", "USB powered", "Charge counter",
                        "Max charging current", "Battery current", "Charger voltage", "PlugType"):
                if line.startswith(key + ":") or line.startswith(key + " :"):
                    fields[key] = line.split(":", 1)[1].strip()
        notify = adb("shell", "su -c 'cat /sys/class/oplus_chg/battery/battery_notify_code'").stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        log(event="phone_state", label=label, error=repr(error))
        return None
    log(event="phone_state", label=label, battery=fields, notify_code=notify)
    return notify


def pull_session_root(destination):
    listing = adb("shell", "su -c 'ls -la " + SESSION_ROOT + " 2>/dev/null'").stdout
    (destination / "phone-session-root.ls.txt").write_text(listing)
    result = adb("exec-out", "su -c 'tar -cf - -C " + SESSION_ROOT + " . 2>/dev/null'", timeout=600, binary=True)
    (destination / "phone-session-root.tar").write_bytes(result.stdout)
    log(event="session_logs_pulled", destination=str(destination), tar_bytes=len(result.stdout),
        returncode=result.returncode)


def main():
    if "--locked" not in sys.argv:
        code = subprocess.run(["flock", "-E", str(LOCK_TIMEOUT_EXIT), "-w", "3600", LOCK, "python3", "-u",
                               str(Path(__file__).resolve()), "--locked", *sys.argv[1:]],
                              stdin=subprocess.DEVNULL).returncode
        if code == LOCK_TIMEOUT_EXIT:
            log(status="LOCK_TIMEOUT", attempt=ATTEMPT)
        return code
    os.environ.update(LANG="C.UTF-8", S42_UNIFIED_REPO_ROOT=str(SOURCE), PYTHONDONTWRITEBYTECODE="1",
                      LD_LIBRARY_PATH="/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64")
    kinds = [k for k in ("baseline", "treatment") if k in sys.argv] or ["baseline", "treatment"]
    log(status="LOCK_ACQUIRED", attempt=ATTEMPT, kinds=kinds, pid=os.getpid())
    for kind, stage in STAGES:
        if kind not in kinds:
            continue
        base = ROOT / f"s42-trace-longtaildev-{kind}-20260923-inputs"
        if (base / "CANCEL").exists():
            log(kind=kind, stage=stage, status="CANCELLED")
            return 1
        output = base / (f"preflight-{ATTEMPT}" if stage == "preflight" else f"run-{kind}-{ATTEMPT}")
        if output.exists():
            raise RuntimeError(f"output already exists: {output}")
        if stage == "run":
            if phone_state(f"{kind}-before") == "512":
                log(kind=kind, stage=stage, status="STOPPED_NOTIFY_512")
                return 3
        command = ["python3", "research_dev/scheduler/campaigns/burstgpt/launch.py", str(base / "campaign.json"),
                   str(output)]
        if stage == "preflight":
            command.append("--preflight-only")
        log(kind=kind, stage=stage, status="STARTED", attempt=ATTEMPT)
        with (base / f"{stage.upper()}-{ATTEMPT}.log").open("x") as out:
            code = subprocess.run(command, cwd=SOURCE, stdin=subprocess.DEVNULL, stdout=out,
                                  stderr=subprocess.STDOUT).returncode
        (base / f"{stage.upper()}_EXIT-{ATTEMPT}.txt").write_text(f"{code}\n")
        log(kind=kind, stage=stage, exit_code=code, attempt=ATTEMPT)
        if stage == "run":
            phone_state(f"{kind}-after")
            if output.is_dir():
                try:
                    pull_session_root(output)
                except (OSError, subprocess.TimeoutExpired) as error:
                    log(event="session_logs_pulled", error=repr(error))
        if code:
            return code
    log(status="PAIR_DONE", attempt=ATTEMPT, kinds=kinds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
