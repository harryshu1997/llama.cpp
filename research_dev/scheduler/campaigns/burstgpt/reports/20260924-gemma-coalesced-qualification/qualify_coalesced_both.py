#!/usr/bin/env python3
"""Coalesced-both transport qualification: 61,440- and 38,400-byte receipts with the 2026-09-22 task1 method.

Same qualifier binary (stage ffs_dmabuf_host sha256 61bc2907...), same phone session/restore scripts
(s42-rrphone-20260914-v3b), same phone worker (ffs_dmabuf_phone.android e2c66e6b...), same arguments
(dmabuf async devmem, warmup 8, 64 iterations, queue depth 4, both directions + duplex). The only
difference: the phone session root is a NEW directory (S41_PHONE_ROOT) holding a byte-identical copy
of the phone worker, so no production /data/local/tmp/s41-* or s42-* directory is modified.

Run under the rig lock:  flock -w 7200 <lock> env S43_UNDER_LOCK=1 python3 qualify_coalesced_both.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/home/zhihao/s43-transport-receipts-61440-20260924")
RECEIPTS = OUT / "receipts"
STAGE = Path("/mnt/storage/s42-ffn-microbatch-20260916-v4-arwuw3")
QUALIFIER = STAGE / "ffs_dmabuf_host"
QUALIFIER_SHA = "61bc2907310c3bf2d13ae10234d7b633fd6b613bdf6cf347a9fc6e113995904f"
PHONE_BINARY_SOURCE = Path("/home/zhihao/s41-ffs-dmabuf-fixed-v1/ffs_dmabuf_phone.android")
PHONE_BINARY_SHA = "e2c66e6b11ba35d247f05dc94b68863971b7ea06e6e234ad4983880f7ba64be1"
PHONE_ROOT = "/data/local/tmp/s43-transport-qual-20260924"
PHONE_SESSION = "/data/local/tmp/s42-rrphone-20260914-v3b/functionfs_transport_session.sh"
PHONE_SESSION_SHA = "fc9e91c0ca9f930f06670ab83e96952a55a140b7c74aa0e1dea99188a5111c93"
PHONE_RESTORE = "/data/local/tmp/s42-rrphone-20260914-v3b/restore_android_usb.sh"
PHONE_RESTORE_SHA = "4c749c8587bdae3e6b0f1ba9062a617737692aa7859566afd3cbee552bcbe44b"
SERIAL = "3C15AU002CL00000"
ADB = ["/usr/bin/adb", "-P", "5037", "-s", SERIAL]
SYSFS = Path("/sys/bus/usb/devices/2-2")
EXPECTED_KERNEL = "6.12.23-android16-5-o-g227664cbe007-4k"
EXPECTED_NOTES = "40bbacf73c0d35195693c566ae695803f59fbf7ec8ce817a1921fb44b3807b51"
EXPECTED_BTF = "77a8ce5acc215506e8dff1bf84e4561ef316680bcba803e9dfc16acecc449735"
PAYLOADS = (61440, 38400)
DIRECTIONS = ("h2d", "d2h", "duplex")
WARMUP, ITERATIONS, DEPTH = 8, 64, 4
PREFIX = "coalesced-both-20260924"
GENERATION = "functionfs-dmabuf-async-ring-v2"


def run(command, timeout=60, **kwargs):
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL, **kwargs)


def shell(command, timeout=30):
    result = run(ADB + ["shell", "-n", "su -c " + shlex.quote(command)], timeout)
    return result.stdout.replace("\r", "").strip()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(name, value):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def battery():
    fields = {}
    dumpsys = run(ADB + ["shell", "-n", "dumpsys", "battery"], 20).stdout
    for line in dumpsys.splitlines():
        key, _, value = line.strip().partition(":")
        if key in ("level", "status", "plugged", "temperature", "voltage", "USB powered",
                   "Charger voltage", "Battery current", "PhoneTemp", "Max charging current",
                   "Charge counter"):
            fields[key] = value.strip()
    fields["battery_notify_code"] = shell("cat /sys/class/oplus_chg/battery/battery_notify_code")
    fields["dumpsys_battery_raw"] = dumpsys
    return fields


def check_notify(stage, fields):
    if fields.get("battery_notify_code") == "512":
        raise RuntimeError(f"battery_notify_code 512 at {stage}; stopping")


def phone_identity():
    boot_id = shell("cat /proc/sys/kernel/random/boot_id")
    kernel = shell("uname -r")
    hashes = shell("sha256sum /sys/kernel/notes /sys/kernel/btf/vmlinux")
    return {"boot_id": boot_id, "kernel_release": kernel, "kernel_hashes": hashes}


def gadget_state():
    return {
        "g1_udc": shell("cat /config/usb_gadget/g1/UDC"),
        "g2_udc": shell("cat /config/usb_gadget/g2/UDC"),
        "current_speed": shell("cat /sys/class/udc/a600000.dwc3/current_speed"),
        "g2_functions": shell("ls /config/usb_gadget/g2/functions"),
    }


def assert_idle_rig(where):
    devices = run(["/usr/bin/adb", "-P", "5037", "devices"]).stdout
    if SERIAL + "\tdevice" not in devices:
        raise RuntimeError("OP15 is not attached to adb -P 5037: " + devices)
    if SYSFS.joinpath("serial").read_text().strip() != SERIAL:
        raise RuntimeError("sysfs 2-2 is not the OP15")
    speed = SYSFS.joinpath("speed").read_text().strip()
    if speed != "5000":
        raise RuntimeError("OP15 link is not SuperSpeed 5000: " + speed)
    if run(["lsusb", "-d", "18d1:2d00"]).returncode == 0:
        raise RuntimeError("a FunctionFS qualification gadget is already enumerated")
    active = run(["pgrep", "-af", "campaigns/burstgpt/(runner|launch)\\.py|llama-server|ffs_dmabuf_host"])
    if active.returncode == 0:
        raise RuntimeError("desktop campaign/server/qualifier process active: " + active.stdout)
    workers = shell("ps -A | grep -E 'llama-ffn|resident-router|resident-workers|ffs_dmabuf|llama-server' | grep -v grep")
    if workers:
        raise RuntimeError("phone FFN worker/qualifier processes are running: " + workers)
    state = gadget_state()
    if state["g1_udc"] != "a600000.dwc3" or state["g2_udc"] != "" or state["current_speed"] != "super-speed":
        raise RuntimeError(f"phone gadget not in normal Android state at {where}: {state}")
    if "ffs.s41" in state["g2_functions"].split():
        raise RuntimeError("ffs.s41 function still present on g2")
    return state


def verify_receipt(name, request, response):
    value = json.loads((RECEIPTS / (name + ".json")).read_text(encoding="ascii"))
    problems = []
    for key, expected in (("schema", "s41_ffs_dmabuf_transport_v2"), ("device_mode", "dmabuf"),
                          ("host_mode", "async"), ("host_allocator", "devmem"),
                          ("transport_generation", GENERATION), ("reset_recoveries", 0),
                          ("request_bytes", request), ("response_bytes", response),
                          ("configured_queue_depth", DEPTH), ("queue_depth", DEPTH),
                          ("warmup", WARMUP), ("iterations", ITERATIONS),
                          ("usbfs_available_bytes", 16777216), ("slot_safety_bytes", 65536)):
        if value.get(key) != expected:
            problems.append(f"{key}={value.get(key)!r} != {expected!r}")
    for key, payload in (("h2d_payload_MBps", request), ("d2h_payload_MBps", response)):
        rate = value.get(key)
        if payload > 64 and (type(rate) not in (int, float) or rate < 1.0):
            problems.append(f"{key}={rate!r} < 1.0")
    terminal = (RECEIPTS / (name + ".terminal.log")).read_text().replace("\r", "").splitlines()
    if terminal[:1] != ["a600000.dwc3"] or (len(terminal) > 1 and terminal[1] != "") or terminal[2:3] != ["super-speed"]:
        problems.append("terminal log is not (g1 bound, g2 empty, super-speed): " + repr(terminal))
    if (RECEIPTS / (name + ".kernel_faults.log")).stat().st_size:
        problems.append("kernel faults logged")
    if problems:
        raise RuntimeError(f"receipt {name} rejected: " + "; ".join(problems))
    return {k: v for k, v in value.items() if not k.endswith("samples_ms")}


def main():
    if os.environ.get("S43_UNDER_LOCK") != "1":
        raise RuntimeError("run under flock on the rig lock with S43_UNDER_LOCK=1")
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.time_ns()
    if digest(QUALIFIER) != QUALIFIER_SHA:
        raise RuntimeError("qualifier binary changed")
    if digest(PHONE_BINARY_SOURCE) != PHONE_BINARY_SHA:
        raise RuntimeError("host copy of the phone worker differs from the qualified one")
    usbfs_mb = Path("/sys/module/usbcore/parameters/usbfs_memory_mb").read_text().strip()
    idle = assert_idle_rig("preflight")
    identity = phone_identity()
    if identity["kernel_release"] != EXPECTED_KERNEL:
        raise RuntimeError("phone kernel release changed: " + identity["kernel_release"])
    hashes = [row.split()[0] for row in identity["kernel_hashes"].splitlines()]
    if hashes != [EXPECTED_NOTES, EXPECTED_BTF]:
        raise RuntimeError("phone kernel notes/BTF differ from the candidate boot identity: " + identity["kernel_hashes"])
    remote = shell(f"sha256sum {PHONE_SESSION} {PHONE_RESTORE}").splitlines()
    remote_shas = {row.split()[1]: row.split()[0] for row in remote}
    if remote_shas.get(PHONE_SESSION) != PHONE_SESSION_SHA or remote_shas.get(PHONE_RESTORE) != PHONE_RESTORE_SHA:
        raise RuntimeError("phone session/restore scripts changed: " + repr(remote_shas))
    if shell(f"test -e {PHONE_ROOT} && echo EXISTS") == "EXISTS":
        raise RuntimeError("new phone root already exists: " + PHONE_ROOT)
    retained = shell("ps -A | grep -E 'direct_phone_service' | grep -v grep")
    boot_id = identity["boot_id"]
    before_battery = battery()
    check_notify("before", before_battery)

    # NEW phone root with a byte-identical copy of the qualified phone worker (shell-owned, like the original).
    push = run(ADB + ["shell", "-n", "mkdir", PHONE_ROOT], 20)
    if push.returncode:
        raise RuntimeError("mkdir phone root failed: " + push.stderr)
    push = run(ADB + ["push", str(PHONE_BINARY_SOURCE), PHONE_ROOT + "/ffs_dmabuf_phone.android"], 120)
    if push.returncode:
        raise RuntimeError("push failed: " + push.stderr)
    run(ADB + ["shell", "-n", "chmod", "755", PHONE_ROOT + "/ffs_dmabuf_phone.android"], 20)
    pushed_sha = shell(f"sha256sum {PHONE_ROOT}/ffs_dmabuf_phone.android").split()[0]
    if pushed_sha != PHONE_BINARY_SHA:
        raise RuntimeError("pushed phone worker digest differs: " + pushed_sha)

    save("BEFORE.json", {
        **identity,
        "qualification_binary_sha256": "sha256:" + QUALIFIER_SHA,
        "qualification_binary_path": str(QUALIFIER),
        "phone_worker_sha256": "sha256:" + PHONE_BINARY_SHA,
        "phone_worker_path": PHONE_ROOT + "/ffs_dmabuf_phone.android",
        "phone_session": PHONE_SESSION, "phone_session_sha256": "sha256:" + PHONE_SESSION_SHA,
        "phone_restore": PHONE_RESTORE, "phone_restore_sha256": "sha256:" + PHONE_RESTORE_SHA,
        "host_usbfs_memory_mb": usbfs_mb,
        "host_usb_sysfs_speed": "5000",
        "gadget_state": idle,
        "battery": before_battery,
        "unrelated_retained_services_left_untouched": retained,
        "started_epoch_ns": started,
    })

    summary = {}
    environment = {
        "S41_STAGE_ROOT": str(STAGE),
        "S41_PHONE_ROOT": PHONE_ROOT,
        "S41_PHONE_SESSION": PHONE_SESSION,
        "S41_PHONE_RESTORE": PHONE_RESTORE,
    }
    for payload in PAYLOADS:
        for direction in DIRECTIONS:
            name = f"{PREFIX}-{payload}-{direction}"
            request = 64 if direction == "d2h" else payload
            response = 64 if direction == "h2d" else payload
            if shell("cat /proc/sys/kernel/random/boot_id") != boot_id:
                raise RuntimeError("phone rebooted before " + name)
            command = ["bash", str(STAGE / "run_transport_case.sh"), "dmabuf", "async", "devmem",
                       str(request), str(response), str(WARMUP), str(ITERATIONS), str(DEPTH),
                       name, str(RECEIPTS)]
            save(name + "-command.json", {"argv": command, "environment": environment,
                                          "stage": str(STAGE), "phone_root": PHONE_ROOT,
                                          "phone_session": PHONE_SESSION, "phone_restore": PHONE_RESTORE,
                                          "started_epoch_ns": time.time_ns()})
            print("START", name, flush=True)
            with (OUT / (name + ".log")).open("x") as log:
                result = subprocess.run(command, env={**os.environ, **environment}, stdout=log,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=180)
            if result.returncode:
                raise RuntimeError(f"transport case failed (exit {result.returncode}); stop: {name}")
            if shell("cat /proc/sys/kernel/random/boot_id") != boot_id:
                raise RuntimeError("phone rebooted during " + name)
            after = battery()
            check_notify(name, after)
            summary[name] = verify_receipt(name, request, response)
            summary[name]["battery_after"] = {k: after[k] for k in ("level", "battery_notify_code", "temperature")}
            print("PASS", name, (OUT / (name + ".log")).read_text().strip(), flush=True)

    final_identity = phone_identity()
    final_state = assert_idle_rig("postflight")
    after_battery = battery()
    check_notify("after", after_battery)
    save("AFTER.json", {**final_identity, "gadget_state": final_state, "battery": after_battery,
                        "finished_epoch_ns": time.time_ns()})
    save("RESULT.json", {
        "status": "PASS",
        "boot_id": boot_id,
        "kernel_changed_by_this_test": final_identity["kernel_hashes"] != identity["kernel_hashes"]
                                        or final_identity["boot_id"] != boot_id,
        "cases": len(summary),
        "payload_bytes": sorted(PAYLOADS),
        "maximum_payload_bytes": max(PAYLOADS),
        "queue_depth": DEPTH,
        "warmup": WARMUP,
        "iterations": ITERATIONS,
        "receipts": summary,
    })
    print("DONE", json.dumps({k: (v["response_ready_median_ms"], v["h2d_payload_MBps"], v["d2h_payload_MBps"])
                             for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:  # noqa: BLE001 - recorded, then re-raised
        try:
            save("FAILURE.json", {"error": repr(error), "finished_epoch_ns": time.time_ns()})
        except Exception:
            pass
        raise
