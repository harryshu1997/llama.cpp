"""Measure the current USB stack at the coalesced Qwen capacity; run under the rig lock."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

QUALIFIER = Path("/mnt/storage/s42-ffn-microbatch-20260916-v4-arwuw3")
PHONE_SESSION = "/data/local/tmp/s42-rrphone-20260914-v3b/functionfs_transport_session.sh"
PHONE_RESTORE = "/data/local/tmp/s42-rrphone-20260914-v3b/restore_android_usb.sh"
ADB = ["/usr/bin/adb", "-P", "5037", "-s", "3C15AU002CL00000"]


def output(command):
    return subprocess.check_output(command, text=True).strip()


def phone_root(command):
    return output([*ADB, "shell", "su", "-c", shlex.quote(command)])


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main(root):
    root.mkdir(exist_ok=False)
    processes = output([*ADB, "shell", "ps", "-A"])
    active = [row for row in processes.splitlines() if any(
        word in row for word in ("llama-ffn", "ffs_dmabuf", "phone_gadget", "direct_phone"))]
    retained = []
    for row in tuple(active):
        if row.split()[1] != "25713" or row.split()[-1] != "direct_phone_service":
            continue
        log = phone_root("tail -1 /data/local/tmp/"
                         "gemma_direct_v4_latency_cpu6_01_policy_c_1789885037040872396/service.log")
        descriptors = phone_root("ls -l /proc/25713/fd")
        if (log == "DIRECT_USB_HELD reason=lifecycle_failed endpoints_and_buffers_retained=1"
                and "/dev/usb-ffs/gemma_b1_20260910/ep0" in descriptors
                and "/dev/usb-ffs/s41/" not in descriptors):
            retained.append({"process": row, "terminal_log": log, "descriptors": descriptors})
            active.remove(row)
    if active:
        raise RuntimeError("phone worker already active: " + repr(active))
    if (phone_root("cat /config/usb_gadget/g1/UDC") != "a600000.dwc3"
            or phone_root("cat /config/usb_gadget/g2/UDC")
            or phone_root("find /config/usb_gadget/g2/configs -type l")):
        raise RuntimeError("experiment USB gadget is already in use")
    gpu = output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                  "--format=csv,noheader"])
    if gpu:
        raise RuntimeError("GPU already in use: " + gpu)
    before = {
        "boot_id": output([*ADB, "shell", "cat", "/proc/sys/kernel/random/boot_id"]),
        "kernel_release": output([*ADB, "shell", "uname", "-r"]),
        "kernel_hashes": phone_root("sha256sum /sys/kernel/notes /sys/kernel/btf/vmlinux"),
        "unrelated_retained_services_left_untouched": retained,
        "qualification_binary_sha256": "sha256:" + hashlib.sha256(
            (QUALIFIER / "ffs_dmabuf_host").read_bytes()).hexdigest(),
    }
    save(root / "BEFORE.json", before)
    environment = {**os.environ, "S41_STAGE_ROOT": str(QUALIFIER),
                   "S41_PHONE_SESSION": PHONE_SESSION, "S41_PHONE_RESTORE": PHONE_RESTORE}
    for payload in (7680, 10240, 40960):
        for direction in ("h2d", "d2h", "duplex"):
            name = f"task1-20260922-{payload}-{direction}"
            command = ["bash", str(QUALIFIER / "run_transport_case.sh"), "dmabuf", "async",
                       "devmem", str(64 if direction == "d2h" else payload),
                       str(64 if direction == "h2d" else payload), "8", "64", "4",
                       name, str(root / "receipts")]
            save(root / (name + "-command.json"), {"argv": command,
                 "started_epoch_ns": time.time_ns(), "stage": str(QUALIFIER),
                 "phone_session": PHONE_SESSION, "phone_restore": PHONE_RESTORE})
            print("START", name, flush=True)
            with (root / (name + ".log")).open("x") as log:
                result = subprocess.run(command, env=environment, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=180)
            if result.returncode:
                raise RuntimeError("qualification failed: " + name)
            if output([*ADB, "shell", "cat", "/proc/sys/kernel/random/boot_id"]) != before["boot_id"]:
                raise RuntimeError("phone rebooted during qualification")
            print("PASS", name, flush=True)
    save(root / "RESULT.json", {"status": "PASS", "cases": 9, "maximum_payload_bytes": 40960,
                               "queue_depth": 4, "boot_id": before["boot_id"],
                               "kernel_changed_by_this_test": False})


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    try:
        main(destination)
    except BaseException as error:
        if destination.is_dir() and not (destination / "FAILURE.json").exists():
            save(destination / "FAILURE.json", {"error": repr(error)})
        raise
