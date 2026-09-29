"""Exclusive bounded launch; no restoration or process-kill fallback in this wrapper."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, "/mnt/storage/s42-dmabuf-cancel-20260914-v1-gBJdFx/validator-v2")
from src.run_direct_usb_device import execution_lock, preflight, shell

mode = sys.argv[1]
assert mode in {"preflight", "run"}
attempt = sys.argv[2] if len(sys.argv) > 2 else "v1"
assert attempt.isascii() and attempt.isalnum()
command_name = sys.argv[3] if len(sys.argv) > 3 else "GATE_COMMAND.json"
assert Path(command_name).name == command_name
command = json.loads((ROOT / command_name).read_text())
if mode == "preflight":
    command += ["--preflight-only"]
command[command.index("--output") + 1] = str(ROOT / ("gate-" + mode + "-" + attempt))
with execution_lock():
    check = ROOT / ("gate-" + mode + "-" + attempt + "-idle-check")
    check.mkdir()
    candidate = json.loads((ROOT / "TRANSPORT_BOOT.json").read_text())["candidate"]
    assert preflight(check) == candidate["boot_id"]
    hashes = shell("sha256sum /sys/kernel/notes /sys/kernel/btf/vmlinux").splitlines()
    assert [row.split()[0] for row in hashes] == [candidate["identity"]["notes"], candidate["identity"]["btf"]]
    with (ROOT / ("gate-" + mode + "-" + attempt + "-resolved-command.json")).open("x") as stream:
        json.dump(command, stream, sort_keys=True, indent=2)
    result = subprocess.run(command)
    if result.returncode:
        raise SystemExit(result.returncode)
    after = ROOT / ("gate-" + mode + "-" + attempt + "-postflight")
    after.mkdir()
    assert preflight(after) == candidate["boot_id"]
