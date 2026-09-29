"""Finite matched transport/FFN tests; USB selection is restricted to the Pixel port."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shlex
import socket
import statistics
import struct
import subprocess
import time

import aoa_bench


SERIAL = "5A040DLCH004ES"
USB_PATH = Path("/sys/bus/usb/devices/2-9.2")
PHONE = "/data/local/tmp/s42-pixel10pro-aoa-20260924-v1"
ADB = ["/usr/bin/adb", "-P", "5037", "-s", SERIAL]
PORT = 27149
ARTIFACT = "940f5f1f2ce0c68d726713e0b1ec86808334c7ca769feac07cd3fa8581c4eae9"
HELLO_REQ = struct.Struct("<IHHQIIHH32s4x")
HELLO_RSP = struct.Struct("<IHHHHIIIIII4xQQIHH32s")
EXEC_REQ = struct.Struct("<IHHIiIIIII")
EXEC_RSP = struct.Struct("<IHHHHIiIIIIIQ")
ECHO = struct.Struct("<QQIIQ")
SENTINEL = 0x9e3779b97f4a7c15
MAGIC = 0x46534631


def run(args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True,
                          stdin=subprocess.DEVNULL,
                          timeout=kwargs.pop("timeout", 30), **kwargs).stdout


def shell(command):
    return run(ADB + ["shell", "su", "-c", shlex.quote(command)])


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def identity():
    result = {name: (USB_PATH / name).read_text().strip()
              for name in ("serial", "idVendor", "idProduct", "busnum", "devnum", "speed")}
    if result["serial"] != SERIAL or result["idVendor"] != "18d1":
        raise RuntimeError(f"Pixel physical-port identity changed: {result}")
    return result


def open_pixel():
    before = identity()
    lu, ctx = aoa_bench.load()
    handle = aoa_bench.open_by_bus_addr(lu, ctx, int(before["busnum"]), int(before["devnum"]))
    if not handle or identity() != before:
        raise RuntimeError("Cannot open stable Pixel bus/address")
    return lu, ctx, handle, before


def accessory_endpoints():
    raw = (USB_PATH / "descriptors").read_bytes()
    offset = 0
    selected = False
    endpoints = []
    while offset < len(raw):
        length, kind = raw[offset:offset + 2]
        if length < 2:
            raise RuntimeError("Invalid USB descriptor")
        part = raw[offset:offset + length]
        if kind == 4:
            selected = part[2] == 0 and part[3] == 0 and part[5] == 255
        if kind == 5 and selected and part[3] & 3 == 2:
            endpoints.append(part[2])
        offset += length
    incoming = [ep for ep in endpoints if ep & 128]
    outgoing = [ep for ep in endpoints if not ep & 128]
    if len(incoming) != 1 or len(outgoing) != 1:
        raise RuntimeError(f"Unexpected accessory endpoints: {endpoints}")
    return outgoing[0], incoming[0]


def change_mode(command):
    lu, ctx, handle, before = open_pixel()
    try:
        if command == "switch":
            if int(before["idProduct"], 16) in aoa_bench.ACCESSORY_PIDS:
                return before
            buf = (ctypes.c_ubyte * 2)()
            rc = lu.libusb_control_transfer(handle, 0xC0, 51, 0, 0, buf, 2, 2000)
            if rc != 2 or buf[0] < 1:
                raise RuntimeError(f"AOA GET_PROTOCOL failed: {rc}")
            values = ["S42", "PixelFFNProbe", "Finite private FFN benchmark", "1.0",
                      "https://example.invalid", SERIAL]
            for index, text in enumerate(values):
                data = text.encode() + b"\0"
                value = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
                rc = lu.libusb_control_transfer(handle, 0x40, 52, 0, index, value, len(data), 2000)
                if rc != len(data):
                    raise RuntimeError(f"AOA string {index}: {rc}")
            rc = lu.libusb_control_transfer(handle, 0x40, 53, 0, 0, None, 0, 2000)
        else:
            rc = lu.libusb_reset_device(handle)
        if rc not in (0, -4):
            raise RuntimeError(f"USB {command}: {rc}")
    finally:
        lu.libusb_close(handle)
        lu.libusb_exit.argtypes = [aoa_bench.CTXP]
        lu.libusb_exit(ctx)
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        try:
            after = identity()
            is_accessory = int(after["idProduct"], 16) in aoa_bench.ACCESSORY_PIDS
            if is_accessory == (command == "switch") and run(ADB + ["get-state"]).strip() == "device":
                return {"before": before, "after": after}
        except (OSError, RuntimeError, subprocess.SubprocessError):
            pass
        time.sleep(0.5)
    raise RuntimeError("Pixel did not return with ADB; inspect target before further actions")


class Transport:
    def __init__(self, mode):
        self.mode = mode
        self.pending = bytearray()
        if mode == "adb":
            self.stream = socket.create_connection(("127.0.0.1", PORT), timeout=20)
            self.stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        else:
            self.lu, self.ctx, self.handle, device = open_pixel()
            if int(device["idProduct"], 16) not in aoa_bench.ACCESSORY_PIDS:
                raise RuntimeError("Pixel is not in accessory mode")
            self.out_ep, self.in_ep = accessory_endpoints()
            rc = self.lu.libusb_claim_interface(self.handle, 0)
            if rc:
                raise RuntimeError(f"Pixel interface claim: {rc}")

    def send(self, data):
        if self.mode == "adb":
            self.stream.sendall(data)
            return
        value = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        done = ctypes.c_int()
        rc = self.lu.libusb_bulk_transfer(self.handle, self.out_ep, value, len(data),
                                          ctypes.byref(done), 10000)
        if rc or done.value != len(data):
            raise RuntimeError(f"USB OUT {rc}, {done.value}/{len(data)}")

    def receive(self, length):
        while len(self.pending) < length:
            if self.mode == "adb":
                part = self.stream.recv(max(65536, length - len(self.pending)))
            else:
                buffer = (ctypes.c_ubyte * 65536)()
                done = ctypes.c_int()
                rc = self.lu.libusb_bulk_transfer(self.handle, self.in_ep, buffer, len(buffer),
                                                  ctypes.byref(done), 10000)
                if rc:
                    raise RuntimeError(f"USB IN {rc}, bytes={done.value}")
                part = ctypes.string_at(buffer, done.value)
            if not part:
                raise RuntimeError("EOF before complete frame")
            self.pending.extend(part)
        result = bytes(self.pending[:length])
        del self.pending[:length]
        return result

    def close(self):
        if self.mode == "adb":
            self.stream.close()
        else:
            self.lu.libusb_release_interface(self.handle, 0)
            self.lu.libusb_close(self.handle)
            self.lu.libusb_exit.argtypes = [aoa_bench.CTXP]
            self.lu.libusb_exit(self.ctx)


def distribution(samples):
    ordered = sorted(samples)
    return {"count": len(samples), "median_ms": statistics.median(samples),
            "p90_ms": ordered[int(len(ordered) * 0.9)],
            "p99_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]}


def fnv(data):
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xffffffff
    return value


def start_worker(args, output, count):
    target = f"{PHONE}/{args.tag}"
    if args.kind == "echo":
        env = f"S42_PIXEL_ECHO_PORT={PORT} " if args.mode == "adb" else ""
        command = f"{env}{PHONE}/aoa-echo serial {args.request} {args.response} {count} {args.warmup} 1"
    else:
        command = " ".join([
            f"LD_LIBRARY_PATH={PHONE}", "S42_PIXEL_CPU_THREADS=6", "S42_PIXEL_CPU_MASK=fc",
            "S42_PIXEL_CPU_POOL=1", "S42_PIXEL_PACKED_WEIGHTS=1", "S42_PIXEL_CPU_QUANT_RESIDUAL=1",
            "S42_PIXEL_FUSED_RESIDUAL=1", "S42_PIXEL_CPU_PAIR_DOT=1", "S42_PIXEL_CPU_ROW_CHUNK=64",
            "S42_PIXEL_CPU_ROW_PROFILE=0", f"S42_PIXEL_AOA={int(args.mode == 'aoa')}",
            f"{PHONE}/llama-ffn-split-worker -m {PHONE}/QWEN_PACKED.ffn.gguf",
            f"--artifact-sha256 sha256:{ARTIFACT} --layers 18,19,20,21,22,23",
            f"--columns 17408 --column-quantum 4352 --backend CPU --port {PORT} --bind 127.0.0.1",
            f"--f16-io --max-tokens 4 --max-requests {count}",
        ])
    script = f"""#!/system/bin/sh
exec 9>/data/local/tmp/.s42-pixel-ffn-kernels.lock
flock -n 9 9>&9 || {{ echo 73 > {target}/EXIT.txt; exit 73; }}
cat /proc/sys/kernel/random/boot_id > {target}/BOOT.txt
id > {target}/UID.txt
dumpsys battery > {target}/BATTERY_BEFORE.txt
{command} > {target}/worker.log 2>&1 &
worker_pid=$!
echo $worker_pid > {target}/PID.txt
wait $worker_pid
status=$?
dumpsys battery > {target}/BATTERY_AFTER.txt
echo $status > {target}/EXIT.txt
exit $status
"""
    (output / "RUN_PHONE.sh").write_text(script)
    run(ADB + ["shell", "mkdir", target])
    run(ADB + ["push", str(output / "RUN_PHONE.sh"), f"{target}/RUN_PHONE.sh"])
    if args.mode == "adb":
        run(ADB + ["forward", "--no-rebind", f"tcp:{PORT}", f"tcp:{PORT}"])
    shell(f"setsid sh {target}/RUN_PHONE.sh </dev/null >{target}/launcher.log 2>&1 &")
    ready = "[ffn-worker] ready backend=" if args.kind == "ffn" else (
        "TCP listening" if args.mode == "adb" else "endpoint open")
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        log = shell(f"cat {target}/worker.log 2>/dev/null || true")
        if ready in log:
            details = shell(f"worker_pid=$(cat {target}/PID.txt); "
                            'cat /proc/$worker_pid/status /proc/$worker_pid/cgroup; '
                            'for item in /proc/$worker_pid/task/*/status; do '
                            "grep -E '^(Name|Pid|Cpus_allowed_list):' $item; done; "
                            'for item in /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq; '
                            'do echo $item; cat $item; done')
            (output / "WORKER_STATUS.txt").write_text(details)
            return target
        status = shell(f"cat {target}/EXIT.txt 2>/dev/null || true").strip()
        if status:
            if args.mode == "adb":
                run(ADB + ["forward", "--remove", f"tcp:{PORT}"])
            raise RuntimeError(f"Worker exited {status}: {log[-2000:]}")
        time.sleep(0.25)
    raise TimeoutError("Worker readiness")


def finish_worker(target, output):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        status = shell(f"cat {target}/EXIT.txt 2>/dev/null || true").strip()
        if status:
            run(ADB + ["pull", target, str(output / "phone")])
            if status != "0":
                raise RuntimeError(f"Finite worker exit {status}")
            return
        time.sleep(0.25)
    raise TimeoutError("Worker completion; left untouched for inspection")


def measure(args):
    output = Path(args.output) / args.tag
    output.mkdir(parents=True, exist_ok=False)
    save(output / "CONFIG.json", vars(args))
    save(output / "USB_BEFORE.json", identity())
    count = (args.warmup + args.repeats) * (6 if args.kind == "ffn" else 1)
    try:
        target = start_worker(args, output, count)
    except BaseException as error:
        save(output / "FAILURE.json", {"error": repr(error), "phase": "startup",
                                        "worker_not_force_killed": True})
        raise
    transport = None
    records = []
    try:
        transport = Transport(args.mode)
        if args.kind == "ffn":
            artifact = bytes.fromhex(ARTIFACT)
            mask = sum(1 << layer for layer in range(18, 24))
            transport.send(HELLO_REQ.pack(MAGIC, 6, 1, mask, 5120, 17408, 3, 4, artifact))
            fields = HELLO_RSP.unpack(transport.receive(HELLO_RSP.size))
            if (fields[:9] != (MAGIC, 6, 2, 0, 3, 5120, 17408, 0, 17408) or
                    fields[10:12] != (6, mask) or fields[13:] != (4352, 4, 0, artifact)):
                raise RuntimeError(f"HELLO mismatch: {fields}")
            save(output / "HELLO.json", {"fields": [x.hex() if isinstance(x, bytes) else x for x in fields]})
        payloads = {}
        for ident in range(1, count + 1):
            if args.kind == "echo":
                request = bytearray(args.request)
                request[:ECHO.size] = ECHO.pack(0x5134414f41524551, ident, args.request,
                                                args.response, ident ^ SENTINEL)
                request[-8:] = struct.pack("<Q", ident ^ SENTINEL)
                expected = bytearray(args.response)
                expected[:ECHO.size] = ECHO.pack(0x5134414f41525350, ident, args.request,
                                                 args.response, ident ^ SENTINEL)
                expected[-8:] = struct.pack("<Q", ident ^ SENTINEL)
                started = time.perf_counter_ns()
                transport.send(request)
                response = transport.receive(args.response)
                rpc_ms = (time.perf_counter_ns() - started) / 1e6
                if response != expected:
                    raise RuntimeError(f"Echo mismatch at {ident}")
                record = {"id": ident, "rpc_ms": rpc_ms, "warmup": ident <= args.warmup}
            else:
                layer = 18 + (ident - 1) % 6
                if layer not in payloads:
                    raw = (Path(args.inputs) / f"input-layer{layer}.f16").read_bytes()
                    payloads[layer] = (raw * args.tokens)[:10240 * args.tokens]
                payload = payloads[layer]
                request = EXEC_REQ.pack(MAGIC, 6, 3, ident, layer, 5120 * args.tokens,
                                        len(payload), fnv(payload), args.columns, args.tokens) + payload
                started = time.perf_counter_ns()
                transport.send(request)
                raw = transport.receive(EXEC_RSP.size + len(payload))
                rpc_ms = (time.perf_counter_ns() - started) / 1e6
                fields = EXEC_RSP.unpack(raw[:EXEC_RSP.size])
                response = raw[EXEC_RSP.size:]
                if (fields[:9] != (MAGIC, 6, 4, 0, 0, ident, layer, 5120 * args.tokens, len(payload)) or
                        fields[9] != fnv(response) or fields[10:12] != (args.columns, args.tokens)):
                    raise RuntimeError(f"EXEC mismatch at {ident}: {fields}")
                name = f"output-layer{layer}.f16"
                reference = output / name
                if reference.exists() and reference.read_bytes() != response:
                    raise RuntimeError("Repeated output changed")
                reference.write_bytes(response)
                if args.reference and (Path(args.reference) / name).read_bytes() != response:
                    raise RuntimeError(f"Transport output mismatch for layer {layer}")
                compute_ms = fields[12] / 1000
                record = {"id": ident, "layer": layer, "rpc_ms": rpc_ms,
                          "compute_ms": compute_ms, "outside_compute_ms": rpc_ms - compute_ms,
                          "input_sha256": hashlib.sha256(payload).hexdigest(),
                          "output_sha256": hashlib.sha256(response).hexdigest(),
                          "warmup": ident <= args.warmup * 6}
            records.append(record)
            if args.gap_ms:
                time.sleep(args.gap_ms / 1000)
        transport.close()
        transport = None
        finish_worker(target, output)
        measured = [r for r in records if not r["warmup"]]
        keys = ("rpc_ms",) if args.kind == "echo" else ("rpc_ms", "compute_ms", "outside_compute_ms")
        result = {"status": "PASS", "completed": count, "all_frames_valid": True,
                  "timing": {key: distribution([r[key] for r in measured]) for key in keys}}
        save(output / "RESULT.json", result)
        print(args.tag, json.dumps(result), flush=True)
    except BaseException as error:
        save(output / "FAILURE.json", {"error": repr(error), "completed": len(records),
                                        "worker_not_force_killed": True})
        raise
    finally:
        save(output / "CALLS.json", records)
        if transport:
            transport.close()
        if args.mode == "adb":
            run(ADB + ["forward", "--remove", f"tcp:{PORT}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "switch", "reset", "measure"])
    parser.add_argument("--mode", choices=["adb", "aoa"], default="adb")
    parser.add_argument("--kind", choices=["echo", "ffn"], default="echo")
    parser.add_argument("--tag", default="smoke")
    parser.add_argument("--output", default="results")
    parser.add_argument("--request", type=int, default=10280)
    parser.add_argument("--response", type=int, default=10288)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--tokens", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--columns", type=int, choices=[8704, 17408], default=17408)
    parser.add_argument("--gap-ms", type=float, default=0)
    parser.add_argument("--inputs", default="inputs")
    parser.add_argument("--reference")
    args = parser.parse_args()
    if not args.tag.replace("-", "").replace("_", "").isalnum():
        parser.error("Tag must contain only letters, digits, dashes and underscores")
    if args.command == "inspect":
        print(json.dumps(identity(), indent=2))
    elif args.command in ("switch", "reset"):
        print(json.dumps(change_mode(args.command), indent=2))
    else:
        measure(args)


if __name__ == "__main__":
    main()
