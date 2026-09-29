#!/usr/bin/env python3
"""Under the rig lock: sync the staged CURRENT main tree into the deploy source, then per arm preflight, admission,
run; then (still under the same lock hold) generate, verify and push the Gemma 26-layer FFN shard set.

    python3 run_arms_allon.py dev2baseDP dev2allon --shards

Adapted from ../20260924-phone-reprovision/run_arms_rp.py. The sync may change only the files listed in SYNC_MANIFEST
(sha256 per path, taken from the staged tree), the deploy must match the manifest afterwards, and a second rsync dry
run must find no difference between the staged tree and the deploy. Every launcher stage runs with cwd and
PYTHONPATH = the deploy source and writes RESULT.json or FAILURE.json under <inputs>/run-<arm>-<attempt>/run/.
A CANCEL file in an arm's input directory stops the chain before that arm's next stage. `dumpsys battery` and
battery_notify_code are logged before/after every stage (full dumpsys text under RIG/battery/); notify code 512
(charger latched off) stops the chain. The shard stage runs after the arms whatever their exit codes (not after a
512 stop or a CANCEL); it never deletes anything on the phone and refuses a non-empty target directory whose
content differs.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

DEPLOY = Path("/mnt/storage/s42-trace-v2-20260921-prep")
SOURCE = DEPLOY / "source"
STAGING = Path("/mnt/storage/s42-allon-20260924-staging")
RIG = Path("/home/zhihao/s42-allon-20260924-rig")
SYNC_MANIFEST = RIG / "SYNC_MANIFEST_ALLON.sha256"
ROOT = Path("/home/zhihao")
LOCK = "/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock"
LOCK_WAIT_S = "14400"
STATUS = ROOT / "s42-trace-longtaildev2-ALLON-20260924-STATUS.jsonl"
ATTEMPT = os.environ.get("ARM_ATTEMPT", "1")
REUSE_PREFLIGHT = os.environ.get("REUSE_PREFLIGHT") == "1"
ADB = ["/usr/bin/adb", "-P", "5037", "-s", "3C15AU002CL00000"]
ARMS = {
    "dev2baseDP": (ROOT / "s42-trace-longtaildev2-baseDP-20260924-inputs",
                   [str(RIG / "check_admission.py"), "{inputs}", "{preflight}", "split-row"]),
    "dev2allon": (ROOT / "s42-trace-longtaildev2-allon-20260924-inputs",
                  [str(RIG / "check_admission_both.py"), "{inputs}", "{preflight}"]),
}
# Gemma 26-layer shard set (reports/20260924-phone-reprovision/README.md, "Shards to generate"), output on /mnt/storage.
GEMMA_MODEL = Path("/home/zhihao/models/gemma-4-12B-Q40-dequant-f16.gguf")
GEMMA_SHA = "sha256:ed76f2183d2d1d65091986033023e6c78d27f6276c1b0c5826cc92acf73538cf"
SHARD_OUT = Path("/mnt/storage/s42-ffn-shards-20260924-v3-gemma26/gemma26")
SHARD_SPECS = ("HTP0=0-8:15360", "HTP1=9-17:15360", "HTP2=18-25:15360")
PHONE_DIR = "/data/local/tmp/s43-ffn-shards-20260924-v3/gemma26"
SHARD_FILES = ("HTP0.ffn.gguf", "HTP1.ffn.gguf", "HTP2.ffn.gguf", "FFN_SHARDS.json")
PUSH_MARGIN_BYTES = 512 * 1024 * 1024


def log(**fields):
    fields["at"] = datetime.now(timezone.utc).isoformat()
    with STATUS.open("a") as f:
        f.write(json.dumps(fields, default=str) + "\n")
    print(json.dumps(fields, default=str), flush=True)


def adb(*args, timeout=60):
    return subprocess.run([*ADB, *args], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def battery(stage, arm=None):
    try:
        dumpsys = adb("shell", "dumpsys", "battery", timeout=20).stdout
        code = adb("shell", "su", "-c", "cat /sys/class/oplus_chg/battery/battery_notify_code", timeout=20).stdout.strip()
    except Exception as error:  # noqa: BLE001 - reported, never fatal
        return {"error": repr(error)}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (RIG / "battery").mkdir(exist_ok=True)
    (RIG / "battery" / f"{stamp}-{arm or 'chain'}-{stage}.txt").write_text(dumpsys + f"\nbattery_notify_code: {code}\n")
    fields = {}
    for line in dumpsys.splitlines():
        key, _, value = line.strip().partition(":")
        if key in ("level", "status", "plugged", "temperature", "voltage", "USB powered", "AC powered",
                   "Charger voltage", "Battery current", "PhoneTemp", "Max charging current"):
            fields[key] = value.strip()
    fields["battery_notify_code"] = code
    return fields


def latched(stage, arm=None):
    state = battery(stage, arm)
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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def digests(root, paths):
    return {path: sha256_file(root / path) if (root / path).exists() else None for path in paths}


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
    (RIG / f"SYNC_ALLON-{ATTEMPT}.txt").write_text(out)
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


def run_arm(arm):
    """Preflight, admission check, run. Returns the first non-zero exit code, else 0; None on CANCEL / 512."""
    base, admission = ARMS[arm]
    passed = sorted(path.parent / ("preflight-" + path.name[len("PREFLIGHT_EXIT-"):-len(".txt")])
                    for path in base.glob("PREFLIGHT_EXIT-*.txt") if path.read_text().strip() == "0")
    for stage in ("preflight", "run"):
        if (base / "CANCEL").exists():
            log(arm=arm, stage=stage, status="CANCELLED")
            return None
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
                return None
            log(arm=arm, stage=stage, status="STARTED", attempt=ATTEMPT, output=str(output))
            with (base / f"{stage.upper()}-{ATTEMPT}.log").open("x") as out:
                code = subprocess.run(command, cwd=SOURCE, stdin=subprocess.DEVNULL, stdout=out,
                                      stderr=subprocess.STDOUT).returncode
            (base / f"{stage.upper()}_EXIT-{ATTEMPT}.txt").write_text(f"{code}\n")
            log(arm=arm, stage=stage, exit_code=code, attempt=ATTEMPT, battery=battery(stage + "_end", arm))
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
    return 0


def phone_free_bytes():
    out = adb("shell", "df", "/data", timeout=30).stdout
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 4 and parts[3].isdigit():
            return int(parts[3]) * 1024, out
    raise RuntimeError("cannot parse df /data: " + out)


def shards():
    """Generate the Gemma 26-layer shard set on /mnt/storage, verify, push to a NEW phone dir, verify on the phone."""
    if latched("shards_begin"):
        return None
    index_path = SHARD_OUT / "FFN_SHARDS.json"
    if SHARD_OUT.exists() and any(SHARD_OUT.iterdir()) and not index_path.exists():
        log(stage="shards", status="REFUSED_PARTIAL_OUTPUT_DIR", output=str(SHARD_OUT))
        return 5
    if not index_path.exists():
        SHARD_OUT.parent.mkdir(parents=True, exist_ok=True)
        command = ["python3", "research_dev/scheduler/native/ffn_shard_gguf.py", str(GEMMA_MODEL),
                   "--parent-sha256", GEMMA_SHA, "--out-dir", str(SHARD_OUT), "--verify-parent"]
        for spec in SHARD_SPECS:
            command += ["--shard", spec]
        log(stage="shards_generate", status="STARTED", command=command)
        with (RIG / f"SHARDS_GENERATE-{ATTEMPT}.log").open("a") as out:
            code = subprocess.run(command, cwd=SOURCE, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                  env={**os.environ, "PYTHONPATH": f"{SOURCE}:{SOURCE / 'gguf-py'}",
                                       "PYTHONDONTWRITEBYTECODE": "1"}).returncode
        log(stage="shards_generate", exit_code=code)
        if code:
            return code
    else:
        log(stage="shards_generate", status="REUSED_EXISTING_INDEX", output=str(index_path))
    index = json.loads(index_path.read_text())
    expected = {row["path"]: (row["shard_sha256"].split(":", 1)[1], row["shard_bytes"]) for row in index["shards"]}
    local = {}
    for name, (sha, size) in sorted(expected.items()):
        path = SHARD_OUT / name
        local[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size,
                       "index_sha256": sha, "index_bytes": size}
        local[name]["ok"] = local[name]["sha256"] == sha and local[name]["bytes"] == size
    (SHARD_OUT / "LOCAL_SHA256.json").write_text(json.dumps(local, indent=1) + "\n")
    log(stage="shards_verify_local", ok=all(v["ok"] for v in local.values()), files=local,
        layers={row["session_id"]: row["layer_spec"] for row in index["shards"]})
    if not all(v["ok"] for v in local.values()):
        return 6
    total = sum(v["bytes"] for v in local.values()) + index_path.stat().st_size
    free, df_text = phone_free_bytes()
    log(stage="shards_phone_space", free_bytes=free, needed_bytes=total, df=df_text.strip().splitlines()[-1])
    if free < total + PUSH_MARGIN_BYTES:
        log(stage="shards_push", status="REFUSED_INSUFFICIENT_PHONE_SPACE", free_bytes=free, needed_bytes=total)
        return 7
    listing = adb("shell", "ls", "-la", PHONE_DIR, timeout=30)
    existing = [l.split()[-1] for l in listing.stdout.splitlines() if l.startswith("-")] if listing.returncode == 0 else []
    if existing and set(existing) - {"FFN_SHARDS.json"}:
        log(stage="shards_push", status="REFUSED_PHONE_DIR_NOT_EMPTY", existing=existing, output=PHONE_DIR)
        return 8
    mk = adb("shell", "mkdir", "-p", PHONE_DIR, timeout=30)
    if mk.returncode:
        log(stage="shards_push", status="MKDIR_FAILED", output=mk.stdout + mk.stderr)
        return 9
    transfer = []
    for name in SHARD_FILES:
        started = datetime.now(timezone.utc)
        push = subprocess.run([*ADB, "push", str(SHARD_OUT / name), f"{PHONE_DIR}/{name}"], capture_output=True,
                              text=True, stdin=subprocess.DEVNULL, timeout=3600)
        seconds = (datetime.now(timezone.utc) - started).total_seconds()
        transfer.append({"file": name, "exit": push.returncode, "seconds": round(seconds, 1),
                         "output": (push.stdout + push.stderr).strip()[-300:]})
        log(stage="shards_push", file=name, exit_code=push.returncode, seconds=round(seconds, 1))
        if push.returncode:
            (SHARD_OUT / "PHONE_TRANSFER.log").write_text(json.dumps(transfer, indent=1) + "\n")
            return 10
    (SHARD_OUT / "PHONE_TRANSFER.log").write_text(json.dumps(transfer, indent=1) + "\n")
    remote = adb("shell", f"cd {PHONE_DIR} && sha256sum " + " ".join(n for n in SHARD_FILES if n != "FFN_SHARDS.json")
                 + " && ls -la", timeout=1800)
    (SHARD_OUT / "PHONE_SHA256.txt").write_text(remote.stdout + remote.stderr)
    phone = dict(re.findall(r"^([0-9a-f]{64})\s+\*?(\S+)$", remote.stdout, re.M))
    phone_by_name = {name: sha for sha, name in phone.items()}
    verified = {name: phone_by_name.get(name) == expected[name][0] for name in expected}
    log(stage="shards_verify_phone", ok=all(verified.values()), verified=verified, phone_dir=PHONE_DIR,
        index=str(index_path), listing=remote.stdout.strip()[-600:])
    latched("shards_end")
    return 0 if all(verified.values()) else 11


def main():
    if "--locked" not in sys.argv:
        return subprocess.run(["flock", "-w", LOCK_WAIT_S, LOCK, "python3", "-u", str(Path(__file__).resolve()),
                               "--locked", *sys.argv[1:]], stdin=subprocess.DEVNULL).returncode
    want_shards = "--shards" in sys.argv
    arms = [a for a in sys.argv[1:] if a not in ("--locked", "--shards")]
    os.environ.update(LANG="C.UTF-8", S42_UNIFIED_REPO_ROOT=str(SOURCE), PYTHONDONTWRITEBYTECODE="1",
                      LD_LIBRARY_PATH="/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64")
    log(stage="lock", status="ACQUIRED", arms=arms, shards=want_shards, attempt=ATTEMPT, pid=os.getpid())
    if latched("sync"):
        return 3
    if not sync():
        return 4
    arm_codes = {}
    for arm in arms:
        code = run_arm(arm)
        arm_codes[arm] = code
        log(arm=arm, stage="arm_done", exit_code=code)
        if code is None:
            log(status="CHAIN_STOPPED", arms=arm_codes)
            return 1
    shard_code = None
    if want_shards:
        shard_code = shards()
        log(stage="shards_done", exit_code=shard_code)
    latched("end")
    log(status="ARMS_DONE", arms=arm_codes, shards=shard_code, attempt=ATTEMPT)
    return 0 if all(c == 0 for c in arm_codes.values()) and shard_code in (0, None) else 2


if __name__ == "__main__":
    raise SystemExit(main())
