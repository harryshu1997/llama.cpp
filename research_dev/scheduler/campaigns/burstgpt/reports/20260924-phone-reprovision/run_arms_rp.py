#!/usr/bin/env python3
"""Under the rig lock: sync the staged main tree into the deploy source, then per arm preflight, admission, run.

    python3 run_arms_rp.py dev2coherentRP dev2coherentEF2

Adapted from ../20260924-coherent-policy-coalesced/run_arms_ef.py. The sync may change only the files listed in
SYNC_MANIFEST (sha256 per path, taken from the local tree), the deploy must match the manifest afterwards, and a
second rsync dry run must find no difference between the staged tree and the deploy. Every launcher stage runs with cwd and
PYTHONPATH = the deploy source and writes RESULT.json or FAILURE.json under <inputs>/run-<arm>-<attempt>/run/.
A CANCEL file in an arm's input directory stops the chain before that arm's next stage. Battery and
battery_notify_code are logged before/after every stage; notify code 512 (charger latched off) stops the chain.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

DEPLOY = Path("/mnt/storage/s42-trace-v2-20260921-prep")
SOURCE = DEPLOY / "source"
STAGING = Path("/home/zhihao/s42-reprovision-20260924-staging")
RIG = Path("/home/zhihao/s42-reprovision-20260924-rig")
SYNC_MANIFEST = RIG / "SYNC_MANIFEST_RP.sha256"
ROOT = Path("/home/zhihao")
LOCK = "/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock"
STATUS = ROOT / "s42-trace-longtaildev2-RP-20260924-STATUS.jsonl"
ATTEMPT = os.environ.get("ARM_ATTEMPT", "1")
REUSE_PREFLIGHT = os.environ.get("REUSE_PREFLIGHT") == "1"
ADB = ["/usr/bin/adb", "-P", "5037", "-s", "3C15AU002CL00000"]
ARMS = {
    "dev2coherentRP": (ROOT / "s42-trace-longtaildev2-coherentRP-20260924-inputs",
                       [str(RIG / "check_admission_both.py"), "{inputs}", "{preflight}"]),
    "dev2coherentEF2": (ROOT / "s42-trace-longtaildev2-coherentEF2-20260924-inputs",
                        [str(RIG / "check_admission_both.py"), "{inputs}", "{preflight}"]),
}


def log(**fields):
    fields["at"] = datetime.now(timezone.utc).isoformat()
    with STATUS.open("a") as f:
        f.write(json.dumps(fields) + "\n")
    print(json.dumps(fields), flush=True)


def battery():
    try:
        dumpsys = subprocess.run([*ADB, "shell", "dumpsys", "battery"], capture_output=True, text=True,
                                 timeout=20, stdin=subprocess.DEVNULL).stdout
        code = subprocess.run([*ADB, "shell", "su", "-c", "cat /sys/class/oplus_chg/battery/battery_notify_code"],
                              capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL).stdout.strip()
    except Exception as error:  # noqa: BLE001 - reported, never fatal
        return {"error": repr(error)}
    fields = {}
    for line in dumpsys.splitlines():
        key, _, value = line.strip().partition(":")
        if key in ("level", "status", "plugged", "temperature", "voltage", "USB powered", "AC powered",
                   "Charger voltage", "Battery current", "PhoneTemp", "Max charging current"):
            fields[key] = value.strip()
    fields["battery_notify_code"] = code
    return fields


def latched(stage, arm=None):
    state = battery()
    log(arm=arm, stage=stage, status="BATTERY", battery=state)
    if state.get("battery_notify_code") == "512":
        log(arm=arm, stage=stage, status="STOPPED_CHARGER_LATCHED", battery=state)
        return True
    return False


def manifest():
    rows = {}
    for line in SYNC_MANIFEST.read_text().splitlines():
        digest, path = line.split(None, 1)
        rows[path.strip()] = digest
    return rows


def digests(root, paths):
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() if (root / path).exists() else None
            for path in paths}


def sync():
    """rsync -rc staging -> deploy; refuse unless exactly the manifest files differ and staging matches it."""
    expected = manifest()
    staged = digests(STAGING / "research_dev/scheduler", expected)
    if staged != expected:
        log(stage="sync", status="REFUSED_STAGING_DIFFERS_FROM_MANIFEST",
            files=sorted(p for p in expected if staged[p] != expected[p]))
        return False
    command = ["rsync", "-rc", "--itemize-changes", "--exclude", "__pycache__", "--exclude", "reports",
               "--exclude", "*.pyc", str(STAGING / "research_dev/scheduler") + "/",
               str(SOURCE / "research_dev/scheduler") + "/"]
    dry = subprocess.run([command[0], "-n", *command[1:]], capture_output=True, text=True, check=True,
                         stdin=subprocess.DEVNULL).stdout
    changed = sorted(line.split(None, 1)[1] for line in dry.splitlines() if line[:1] in "<>")
    unexpected = [path for path in changed if path not in expected]
    if unexpected:
        log(stage="sync", status="REFUSED_UNEXPECTED_FILES", files=unexpected[:80])
        return False
    before = digests(SOURCE / "research_dev/scheduler", expected)
    out = subprocess.run(command, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL).stdout
    (RIG / f"SYNC_RP-{ATTEMPT}.txt").write_text(out)
    after = digests(SOURCE / "research_dev/scheduler", expected)
    if after != expected:
        log(stage="sync", status="FAILED_DEPLOY_DIFFERS_FROM_MANIFEST",
            files=sorted(p for p in expected if after[p] != expected[p]))
        return False
    residual = subprocess.run([command[0], "-n", *command[1:]], capture_output=True, text=True, check=True,
                              stdin=subprocess.DEVNULL).stdout
    residual = sorted(line.split(None, 1)[1] for line in residual.splitlines() if line[:1] in "<>")
    if residual:
        log(stage="sync", status="FAILED_DEPLOY_DIFFERS_FROM_STAGING", files=residual[:80])
        return False
    log(stage="sync", status="DONE", changed_files=changed,
        unchanged_manifest_files=sorted(p for p in expected if before[p] == expected[p]))
    return True


def main():
    if "--locked" not in sys.argv:
        return subprocess.run(["flock", "-w", "7200", LOCK, "python3", "-u", str(Path(__file__).resolve()), "--locked",
                               *sys.argv[1:]], stdin=subprocess.DEVNULL).returncode
    arms = [a for a in sys.argv[1:] if a != "--locked"] or list(ARMS)
    os.environ.update(LANG="C.UTF-8", S42_UNIFIED_REPO_ROOT=str(SOURCE), PYTHONDONTWRITEBYTECODE="1",
                      LD_LIBRARY_PATH="/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64")
    log(stage="lock", status="ACQUIRED", arms=arms, attempt=ATTEMPT, pid=os.getpid())
    if latched("sync"):
        return 3
    if not sync():
        return 4
    for arm in arms:
        base, admission = ARMS[arm]
        passed = sorted(path.parent / ("preflight-" + path.name[len("PREFLIGHT_EXIT-"):-len(".txt")])
                        for path in base.glob("PREFLIGHT_EXIT-*.txt") if path.read_text().strip() == "0")
        for stage in ("preflight", "run"):
            if (base / "CANCEL").exists():
                log(arm=arm, stage=stage, status="CANCELLED")
                return 1
            output = base / (f"preflight-{ATTEMPT}" if stage == "preflight" else f"run-{arm}-{ATTEMPT}")
            if stage == "preflight" and REUSE_PREFLIGHT and passed and passed[-1].is_dir():
                output = passed[-1]
                log(arm=arm, stage=stage, status="REUSED", output=str(output))
            elif output.exists():
                raise RuntimeError(f"output already exists: {output}")
            else:
                command = ["python3", "research_dev/scheduler/campaigns/burstgpt/launch.py", str(base / "campaign.json"),
                           str(output)]
                if stage == "preflight":
                    command.append("--preflight-only")
                if latched(stage, arm):
                    return 3
                log(arm=arm, stage=stage, status="STARTED", attempt=ATTEMPT, output=str(output))
                with (base / f"{stage.upper()}-{ATTEMPT}.log").open("x") as out:
                    code = subprocess.run(command, cwd=SOURCE, stdin=subprocess.DEVNULL, stdout=out,
                                          stderr=subprocess.STDOUT).returncode
                (base / f"{stage.upper()}_EXIT-{ATTEMPT}.txt").write_text(f"{code}\n")
                log(arm=arm, stage=stage, exit_code=code, attempt=ATTEMPT, battery=battery())
                if code:
                    return code
            if stage == "preflight":
                argv = [part.format(inputs=base, preflight=output) for part in admission]
                check = subprocess.run(["python3", *argv], cwd=SOURCE, capture_output=True, text=True,
                                       stdin=subprocess.DEVNULL,
                                       env={**os.environ, "PYTHONPATH": str(SOURCE), "PYTHONDONTWRITEBYTECODE": "1"})
                (base / f"ADMISSION-{ATTEMPT}.log").write_text(check.stdout + check.stderr)
                (base / f"ADMISSION_EXIT-{ATTEMPT}.txt").write_text(f"{check.returncode}\n")
                log(arm=arm, stage="admission_check", exit_code=check.returncode,
                    output=(check.stdout if not check.returncode else check.stderr)[-1500:])
                if check.returncode:
                    return check.returncode
    latched("end")
    log(status="ARMS_DONE", arms=arms, attempt=ATTEMPT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
