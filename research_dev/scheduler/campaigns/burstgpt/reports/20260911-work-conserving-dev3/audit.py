"""Read-only post-run source, process, and USB audit."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    rig = json.loads((root / "inputs-v3/rig.json").read_text())
    repo = Path(rig["repo_root"])
    source = json.loads((root / "inputs-v3/SOURCE_MANIFEST.json").read_text())
    mismatches = [row["path"] for row in source["files"] if "sha256:" + hashlib.sha256(
        (repo / row["path"]).read_bytes()).hexdigest() != row["sha256"]]
    adb = [rig["binaries"]["adb"], "-P", str(rig["phone"]["adb_port"]), "-s", rig["phone"]["serial"]]
    rows = []
    for command in (
        adb + ["shell", "ps -A"],
        adb + ["shell", "getprop sys.usb.config; uname -r"],
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader"],
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        ["ps", "-eo", "pid,ppid,comm,args"],
    ):
        started = time.time_ns()
        response = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
        rows.append({"command": command, "started_epoch_ns": started,
                     "finished_epoch_ns": time.time_ns(), "stdout": response.stdout,
                     "stderr": response.stderr, "returncode": response.returncode})
    phone_workers = [line for line in rows[0]["stdout"].splitlines()
                     if "llama-" in line or "ffn-split" in line]
    desktop_workers = [line for line in rows[-1]["stdout"].splitlines()
                       if str(root / "cuda-build/bin/llama-server") in line]
    status = "PASS" if not (mismatches or phone_workers or desktop_workers) else "FAIL"
    status = status if rows[1]["stdout"].startswith("ptp,adb\n") else "FAIL"
    payload = {"status": status, "commands": rows, "source_mismatches": mismatches,
               "source_files_checked": len(source["files"]), "remaining_phone_workers": phone_workers,
               "remaining_experiment_desktop_workers": desktop_workers,
               "scope": "read-only; no resets, process termination, or system changes"}
    with (root / "POST_GATE_AUDIT.json").open("x") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "commands"}))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
