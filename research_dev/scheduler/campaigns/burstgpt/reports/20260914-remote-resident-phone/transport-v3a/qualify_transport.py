"""Fresh bounded measurements of the existing FFN USB client on the candidate boot."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CANDIDATE = Path("/mnt/storage/s42-dmabuf-cancel-20260914-v1-gBJdFx")
sys.path.insert(0, str(CANDIDATE / "validator-v2"))
from src.run_direct_usb_device import execution_lock, preflight, shell


def save(name, value):
    with (ROOT / name).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def main():
    with execution_lock():
        before = ROOT / "transport-preflight"
        before.mkdir()
        boot = preflight(before)
        candidate = json.loads((CANDIDATE / "CANDIDATE_BOOT_RESULT.json").read_text())
        assert boot == candidate["boot_id"]
        actual = shell("sha256sum /sys/kernel/notes /sys/kernel/btf/vmlinux").splitlines()
        assert [row.split()[0] for row in actual] == [
            candidate["identity"]["notes"], candidate["identity"]["btf"]]
        with (ROOT / "ffs_dmabuf_host").open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == (
                "61bc2907310c3bf2d13ae10234d7b633fd6b613bdf6cf347a9fc6e113995904f")
        save("TRANSPORT_BOOT.json", {"candidate": candidate, "actual_identity": actual})
        for payload in (7680, 10240, 3932160):
            for direction in ("h2d", "d2h", "duplex"):
                name = f"rrphone-20260914-v3-{payload}-{direction}"
                assert shell("cat /proc/sys/kernel/random/boot_id") == boot
                request = 64 if direction == "d2h" else payload
                response = 64 if direction == "h2d" else payload
                command = ["bash", str(ROOT / "run_transport_case.sh"), "dmabuf", "async",
                           "devmem", str(request), str(response), "5", "100", "1",
                           name, str(ROOT / "transport")]
                save(name + "-command.json", {"command": command, "started_epoch_ns": time.time_ns(),
                                            "environment": {"S41_STAGE_ROOT": str(ROOT)}})
                print("START", name, flush=True)
                with (ROOT / (name + ".log")).open("x") as log:
                    result = subprocess.run(command, env={**os.environ, "S41_STAGE_ROOT": str(ROOT)},
                                            stdout=log, stderr=subprocess.STDOUT, timeout=180)
                if result.returncode:
                    raise RuntimeError("transport case failed; stop: " + name)
                assert shell("cat /proc/sys/kernel/random/boot_id") == boot
                print("PASS", name, flush=True)
        after = ROOT / "transport-postflight"
        after.mkdir()
        assert preflight(after) == boot
        save("TRANSPORT_RESULT.json", {"status": "PASS", "boot_id": boot,
                                      "cases": 9, "model_execution": False})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        save("TRANSPORT_FAILURE.json", {"error": repr(error), "finished_epoch_ns": time.time_ns()})
        raise
