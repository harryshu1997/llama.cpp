"""Run the existing bounded relocation gate with token-probability observations."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS = Path("/mnt/storage/s42-remote-resident-phone-20260914-v3b-r9Rdsh")
sys.path.insert(0, "/mnt/storage/s42-dmabuf-cancel-20260914-v1-gBJdFx/validator-v2")
from src.run_direct_usb_device import execution_lock, preflight, shell


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def main():
    mode = sys.argv[1]
    assert mode in {"preflight", "run"}
    command = json.loads((PREVIOUS / "gate-run-v4-resolved-command.json").read_text())
    original = list(command)
    command[1] = str(ROOT / "source/research_dev/scheduler/campaigns/burstgpt/remote_resident_gate.py")
    replacements = {
        "--output": str(ROOT / ("gate-" + mode + "-v1")),
        "--phone-session-root": "/data/local/tmp/" + ROOT.name,
    }
    for option in ("--phone-remote-hash-cache", "--gguf-manifest-cache"):
        source = Path(command[command.index(option) + 1])
        target = ROOT / source.name
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(source.read_bytes())
        replacements[option] = str(target)
    for name, value in replacements.items():
        command[command.index(name) + 1] = value
    command += ["--diagnostic-top-logprobs", "32"]
    if mode == "preflight":
        command += ["--preflight-only"]
    candidate = json.loads((PREVIOUS / "TRANSPORT_BOOT.json").read_text())["candidate"]
    launch = {"mode": mode, "started_epoch_ns": time.time_ns(), "status": "PENDING"}
    with execution_lock():
        check = ROOT / (mode + "-idle-check")
        check.mkdir()
        assert preflight(check) == candidate["boot_id"]
        hashes = shell("sha256sum /sys/kernel/notes /sys/kernel/btf/vmlinux").splitlines()
        assert [line.split()[0] for line in hashes] == [candidate["identity"]["notes"], candidate["identity"]["btf"]]
        write_new(ROOT / (mode + "-command.json"), {
            "original": original, "resolved": command,
            "diagnostic_only": True, "native_rebuild": False,
            "preserved_result_sha256": hashlib.sha256(
                (PREVIOUS / "gate-run-v4/REMOTE_RESIDENT_GATE.json").read_bytes()).hexdigest(),
        })
        result = subprocess.run(command)
        launch["returncode"] = result.returncode
        launch["status"] = "CHILD_FINISHED"
        after = ROOT / (mode + "-postflight")
        after.mkdir()
        try:
            assert preflight(after) == candidate["boot_id"]
            launch["postflight"] = "PASS"
        except Exception as error:
            launch["postflight"] = f"{type(error).__name__}: {error}"
        launch["finished_epoch_ns"] = time.time_ns()
        write_new(ROOT / (mode + "-launch.json"), launch)
        return result.returncode or (0 if launch["postflight"] == "PASS" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
