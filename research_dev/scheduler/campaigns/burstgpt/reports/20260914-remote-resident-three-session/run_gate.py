"""Exclusive bounded launch of the existing gate with a frozen experiment configuration."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, "/mnt/storage/s42-dmabuf-cancel-20260914-v1-gBJdFx/validator-v2")
from src.run_direct_usb_device import execution_lock, preflight, shell


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def main():
    mode = sys.argv[1]
    assert mode in {"preflight", "run"}
    attempt = sys.argv[2] if len(sys.argv) > 2 else "v1"
    assert attempt.isascii() and attempt.isalnum()
    label = mode + "-" + attempt
    spec = json.loads((ROOT / "EXPERIMENT.json").read_text())
    previous = Path(spec["prior_artifacts"])
    command = json.loads((previous / "gate-run-v4-resolved-command.json").read_text())
    original = list(command)
    command[1] = str(ROOT / "source/research_dev/scheduler/campaigns/burstgpt/remote_resident_gate.py")
    replacements = {
        "--output": str(ROOT / ("gate-" + label)),
        "--phone-session-root": "/data/local/tmp/" + ROOT.name + "-" + attempt,
        "--maximum-phone-sessions": str(spec["maximum_phone_sessions"]),
        "--remote-layer-mask": spec["remote_layer_mask"],
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
    command += ["--fixed-phone-residency-json", json.dumps(spec["fixed_phone_residency"]),
                "--output-comparison", spec["output_comparison"],
                "--diagnostic-top-logprobs", str(spec["diagnostic_top_logprobs"])]
    if mode == "preflight":
        command += ["--preflight-only"]
    candidate = json.loads((previous / "TRANSPORT_BOOT.json").read_text())["candidate"]
    launch = {"mode": mode, "started_epoch_ns": time.time_ns(), "status": "PENDING"}
    with execution_lock():
        check = ROOT / (label + "-idle-check")
        check.mkdir()
        assert preflight(check) == candidate["boot_id"]
        hashes = shell("sha256sum /sys/kernel/notes /sys/kernel/btf/vmlinux").splitlines()
        assert [row.split()[0] for row in hashes] == [candidate["identity"]["notes"], candidate["identity"]["btf"]]
        write_new(ROOT / (label + "-command.json"), {
            "original": original, "resolved": command, "experiment": spec,
            "preserved_result_sha256": hashlib.sha256(
                (previous / "gate-run-v4/REMOTE_RESIDENT_GATE.json").read_bytes()).hexdigest(),
        })
        result = subprocess.run(command)
        launch["returncode"], launch["status"] = result.returncode, "CHILD_FINISHED"
        after = ROOT / (label + "-postflight")
        after.mkdir()
        try:
            assert preflight(after) == candidate["boot_id"]
            launch["postflight"] = "PASS"
        except Exception as error:
            launch["postflight"] = f"{type(error).__name__}: {error}"
        launch["finished_epoch_ns"] = time.time_ns()
        write_new(ROOT / (label + "-launch.json"), launch)
        return result.returncode or (0 if launch["postflight"] == "PASS" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
