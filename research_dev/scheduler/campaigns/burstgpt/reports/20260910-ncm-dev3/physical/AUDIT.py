"""Archive read-only cleanup observations and hashes after the three-request run."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    root = args.root
    result = json.loads((root / "run/RESULT.json").read_text())
    assert result["status"] == "PASS" and result["counts"]["requests"] == 3
    commands = []
    for command in (
        ["adb", "-s", "3C15AU002CL00000", "shell", "ps -A"],
        ["adb", "-s", "3C15AU002CL00000", "shell",
         "getprop sys.usb.config; uname -r; cat /sys/kernel/btf/vmlinux | sha256sum"],
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader"],
        ["ps", "-p", "6871", "-o", "pid,comm,args"],
    ):
        started = time.time_ns()
        response = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
        commands.append({"command": command, "started_epoch_ns": started,
                         "finished_epoch_ns": time.time_ns(), "returncode": response.returncode,
                         "stdout": response.stdout, "stderr": response.stderr})
    remaining = [line for line in commands[0]["stdout"].splitlines()
                 if "llama-" in line or "ffn-split" in line]
    assert not remaining, remaining
    assert commands[1]["stdout"].startswith("ptp,adb\n6.12.23-android16-5-o-g227664cbe007-4k\n")
    source = json.loads((root / "inputs/SOURCE_MANIFEST_EXECUTION.json").read_text())
    mismatches = [row["path"] for row in source["files"] if hashlib.sha256(
        (args.repo / row["path"]).read_bytes()).hexdigest() != row["sha256"].removeprefix("sha256:")]
    assert not mismatches, mismatches
    write_new(root / "POST_GATE_AUDIT.json", {
        "status": "PASS", "commands": commands, "remaining_phone_workers": remaining,
        "source_file_count": len(source["files"]), "source_mismatches": mismatches,
        "scope": "Read-only observations; no trace, baseline, reset or process intervention",
    })
    artifacts = {str(path.relative_to(root)): {"bytes": path.stat().st_size,
                 "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}
                 for path in sorted(root.rglob("*")) if path.is_file()
                 and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    write_new(root / "ARTIFACTS.json", artifacts)
    print(json.dumps({"status": "PASS", "artifact_count": len(artifacts),
                      "source_file_count": len(source["files"])}))


if __name__ == "__main__":
    main()
