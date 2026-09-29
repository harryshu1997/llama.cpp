#!/usr/bin/env python3
"""Under the rig lock: sync the staged scheduler tree into the shared deploy, then preflight + run each arm.

Arms are named on the command line (default: coherent plain). Every stage waits for its own exit; the
launcher writes RESULT.json or FAILURE.json under <inputs>/run-<arm>-<attempt>/run/.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

DEPLOY = Path("/mnt/storage/s42-trace-v2-20260921-prep")
SOURCE = DEPLOY / "source"
STAGING = Path("/home/zhihao/s42-coherent-20260924-staging")
ROOT = Path("/home/zhihao")
LOCK = "/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock"
STATUS = ROOT / "s42-trace-longtaildev-coherent-20260924-STATUS.jsonl"
ATTEMPT = os.environ.get("ARM_ATTEMPT", "1")
# "physical": launch.py --preflight-only; "resolve": launch.py --resolve-only (catalog and manifests only, no
# hardware), used when the rig and deploy were just preflighted by an earlier arm of the same chain.
PREFLIGHT_MODE = os.environ.get("PREFLIGHT_MODE", "physical")
ADB = ["/usr/bin/adb", "-P", "5037", "-s", "3C15AU002CL00000"]


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
        if key in ("level", "status", "plugged", "temperature", "voltage", "USB powered", "Charger voltage",
                   "Battery current", "PhoneTemp"):
            fields[key] = value.strip()
    fields["battery_notify_code"] = code
    return fields


# Content that may change in the deploy: this change set plus the two newer trace-builder files of the local tree.
SYNC_ALLOWED = frozenset({
    "_internal/adaptive_decode_state.py", "_internal/adaptive_decode_ops/coherence.py",
    "_internal/adaptive_decode_ops/completion.py", "_internal/adaptive_decode_ops/reporting.py",
    "_internal/adaptive_decode_ops/windows.py", "_unified/adaptive_decode_control.py",
    "campaigns/burstgpt/build_realistic_trace.py", "campaigns/burstgpt/prepare_trace_inputs_v2.py",
    "tests/test_adaptive_coherence.py", "tests/test_build_realistic_trace.py", "tests/test_prepare_trace_inputs_v2.py",
})


def sync():
    """rsync -c the staged tree into the deploy; refuse if anything outside the change set would change."""
    command = ["rsync", "-c", "-rlt", "--itemize-changes", "--exclude", "__pycache__", "--exclude", "reports",
               str(STAGING / "research_dev/scheduler") + "/", str(SOURCE / "research_dev/scheduler") + "/"]
    dry = subprocess.run([command[0], "-n", *command[1:]], capture_output=True, text=True, check=True,
                         stdin=subprocess.DEVNULL).stdout
    changed = sorted(line.split(None, 1)[1] for line in dry.splitlines() if line[:1] in "<>")
    unexpected = [path for path in changed if path not in SYNC_ALLOWED]
    if unexpected:
        log(stage="sync", status="REFUSED_UNEXPECTED_FILES", files=unexpected[:80])
        return False
    out = subprocess.run(command, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL).stdout
    (ROOT / f"s42-trace-longtaildev-coherent-20260924-SYNC-{ATTEMPT}.txt").write_text(out)
    log(stage="sync", status="DONE", changed_files=changed)
    return True


def main():
    if "--locked" not in sys.argv:
        return subprocess.run(["flock", "-w", "7200", LOCK, "python3", "-u", str(Path(__file__).resolve()), "--locked",
                               *sys.argv[1:]], stdin=subprocess.DEVNULL).returncode
    arms = [a for a in sys.argv[1:] if a != "--locked"] or ["coherent", "plain"]
    os.environ.update(LANG="C.UTF-8", S42_UNIFIED_REPO_ROOT=str(SOURCE), PYTHONDONTWRITEBYTECODE="1",
                      LD_LIBRARY_PATH="/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64")
    log(stage="lock", status="ACQUIRED", arms=arms, attempt=ATTEMPT)
    if not sync():
        return 4
    for arm in arms:
        base = (ROOT / f"s42-trace-v2a-{arm[3:]}-20260924-inputs" if arm.startswith("v2a")
                else ROOT / "s42-trace-longtaildev2-baseline-20260924-inputs" if arm == "dev2base"
                else ROOT / f"s42-trace-longtaildev2-{arm[4:]}-20260924-inputs" if arm.startswith("dev2")
                else ROOT / f"s42-trace-longtaildev-{arm}-20260924-inputs")
        passed = sorted(path.parent / ("preflight-" + path.name[len("PREFLIGHT_EXIT-"):-len(".txt")])
                        for path in base.glob("PREFLIGHT_EXIT-*.txt") if path.read_text().strip() == "0")
        for stage in ("preflight", "run"):
            if (base / "CANCEL").exists():
                log(arm=arm, stage=stage, status="CANCELLED")
                return 1
            label = "resolve" if stage == "preflight" and PREFLIGHT_MODE == "resolve" else stage
            output = base / (f"{label}-{ATTEMPT}" if stage == "preflight" else f"run-{arm}-{ATTEMPT}")
            if stage == "preflight" and passed and passed[-1].is_dir():
                # A passed preflight of these unchanged inputs is reused; only its admission check runs again.
                output = passed[-1]
                log(arm=arm, stage=stage, status="REUSED", output=str(output))
            elif output.exists():
                raise RuntimeError(f"output already exists: {output}")
            else:
                command = ["python3", "research_dev/scheduler/campaigns/burstgpt/launch.py", str(base / "campaign.json"),
                           str(output)]
                if stage == "preflight":
                    command.append("--resolve-only" if label == "resolve" else "--preflight-only")
                before = battery()
                if before.get("battery_notify_code") == "512":
                    log(arm=arm, stage=stage, status="STOPPED_CHARGER_LATCHED", battery=before)
                    return 3
                log(arm=arm, stage=label, status="STARTED", attempt=ATTEMPT, battery=before)
                with (base / f"{label.upper()}-{ATTEMPT}.log").open("x") as out:
                    code = subprocess.run(command, cwd=SOURCE, stdin=subprocess.DEVNULL, stdout=out,
                                          stderr=subprocess.STDOUT).returncode
                (base / f"{label.upper()}_EXIT-{ATTEMPT}.txt").write_text(f"{code}\n")
                log(arm=arm, stage=label, exit_code=code, attempt=ATTEMPT, battery=battery())
                if code:
                    return code
            if stage == "preflight":
                plan = "coalesced-batch" if arm in {"coherent", "v2acoherent", "dev2coherent"} else "split-row"
                check = subprocess.run(["python3", str(ROOT / "s42-coherent-20260924-rig/check_admission.py"),
                                        str(base), str(output), plan], cwd=SOURCE, capture_output=True, text=True,
                                       stdin=subprocess.DEVNULL,
                                       env={**os.environ, "PYTHONPATH": str(SOURCE), "PYTHONDONTWRITEBYTECODE": "1"})
                log(arm=arm, stage="admission_check", exit_code=check.returncode,
                    output=(check.stdout or check.stderr)[-2000:])
                if check.returncode:
                    return check.returncode
    log(status="ARMS_DONE", arms=arms, attempt=ATTEMPT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
