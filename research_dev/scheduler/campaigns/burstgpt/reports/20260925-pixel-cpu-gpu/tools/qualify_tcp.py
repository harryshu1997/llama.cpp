"""Over-TCP qualification of Pixel worker variants with the production AdbTcpPhoneWorkerSession.

Each arm launches the worker rooted under the Pixel phone lock with an exact finite budget, drives it
from the desktop over the adb forward at server-like cadence (6 consecutive layer calls per token, then an
idle gap), records host RPC / worker compute per call and the response payloads, and stops it through the
session (normal budget exit, forward removed, boot unchanged). Run under the rig lock.

usage: S42_UNIFIED_REPO_ROOT=<repo> python3 qualify_tcp.py CONFIG.json OUTPUT_DIR
"""

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.environ["S42_UNIFIED_REPO_ROOT"])
from research_dev.scheduler.adapters.phone_tcp_session import (  # noqa: E402
    AdbTcpPhoneWorkerSession, AdbTcpWorkerConfiguration, HELLO_REQUEST, HELLO_RESPONSE, EXECUTE_REQUEST,
    EXECUTE_RESPONSE, PROTOCOL_MAGIC, PROTOCOL_VERSION, _fnv32,
)

LAYERS = [18, 19, 20, 21, 22, 23]
N_EMBD = 5120


def battery(serial):
    return subprocess.run(["/usr/bin/adb", "-P", "5037", "-s", serial, "shell", "dumpsys", "battery"],
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60).stdout


def run_arm(config, arm, output, inputs):
    phone = config["phone"]
    segments = config["segments"]
    calls = sum((s["warmup"] + s["steps"]) * len(LAYERS) for s in segments)
    hashes = dict(phone["common_sha256"])
    hashes[arm["worker_path"]] = arm["worker_sha256"]
    configuration = AdbTcpWorkerConfiguration(
        device_id="pixel10pro-phone", serial=phone["serial"], adb_port=5037, adb_path=Path("/usr/bin/adb"),
        worker_path=arm["worker_path"], library_directories=(phone["library_directory"],),
        shard_path=phone["shard_path"], artifact_sha256=phone["artifact_sha256"], layer_mask=sum(1 << l for l in LAYERS),
        n_embd=N_EMBD, columns=17408, column_quantum=4352, max_tokens=4, swiglu=True, backend="CPU",
        phone_port=phone["phone_port"], forward_port=0, max_requests=calls,
        worker_environment=arm["environment"], expected_sha256_by_path=hashes,
        as_root=True, phone_lock_path=phone["phone_lock_path"])
    session = AdbTcpPhoneWorkerSession(configuration)
    directory = output / arm["name"]
    directory.mkdir()
    (directory / "BATTERY_BEFORE.txt").write_text(battery(phone["serial"]))
    (directory / "PREFLIGHT.json").write_text(json.dumps(session.preflight().to_json(), indent=2) + "\n")
    (directory / "LAUNCH.json").write_text(json.dumps(session.start(directory / "worker.log").to_json(), indent=2) + "\n")
    rows, dumps = [], {}
    try:
        port = session.transport_contract().control_port
        with socket.create_connection(("127.0.0.1", port), timeout=60) as stream:
            stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            stream.sendall(HELLO_REQUEST.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, 1, configuration.layer_mask,
                                             N_EMBD, 17408, 3, 4, bytes.fromhex(phone["artifact_sha256"][7:])))
            hello = HELLO_RESPONSE.unpack(session._receive(stream, HELLO_RESPONSE.size))
            assert hello[:4] == (PROTOCOL_MAGIC, PROTOCOL_VERSION, 2, 0)
            request_id = 0
            for seg in segments:
                rows_per_call = seg["rows"]
                dump = bytearray()
                for step in range(seg["warmup"] + seg["steps"]):
                    for index, layer in enumerate(LAYERS):
                        payload = b"".join(inputs[(layer, r)] for r in range(rows_per_call))
                        request_id += 1
                        started = time.monotonic_ns()
                        stream.sendall(EXECUTE_REQUEST.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, 3, request_id, layer,
                                                            N_EMBD * rows_per_call, len(payload), _fnv32(payload),
                                                            17408, rows_per_call) + payload)
                        response = EXECUTE_RESPONSE.unpack(session._receive(stream, EXECUTE_RESPONSE.size))
                        data = session._receive(stream, len(payload))
                        elapsed_us = (time.monotonic_ns() - started) / 1000
                        assert response[:6] == (PROTOCOL_MAGIC, PROTOCOL_VERSION, 4, 0, 0, request_id)
                        assert response[8] == len(payload) and response[9] == _fnv32(data)
                        rows.append({"segment": seg["name"], "step": step - seg["warmup"], "layer": layer,
                                     "rows": rows_per_call, "rpc_us": elapsed_us, "compute_us": response[-1],
                                     "overhead_us": elapsed_us - response[-1], "hash": response[9]})
                        if step == seg["warmup"]:
                            dump.extend(data)
                        if index + 1 < len(LAYERS) and seg["gap_call_us"]:
                            time.sleep(seg["gap_call_us"] / 1e6)
                    time.sleep(seg["gap_token_us"] / 1e6)
                dumps[seg["name"]] = bytes(dump)
                (directory / f"{seg['name']}.f16").write_bytes(dump)
        stop = session.stop(served_calls=len(rows)).to_json()
        result = {"status": "PASS", "arm": arm["name"], "calls": len(rows), "stop": stop,
                  "worker_sha256": arm["worker_sha256"], "environment": arm["environment"],
                  "dump_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in dumps.items()}}
        (directory / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        (directory / "CALLS.json").write_text(json.dumps(rows) + "\n")
        (directory / "BATTERY_AFTER.txt").write_text(battery(phone["serial"]))
        if session.active:
            (directory / "CLEANUP.json").write_text(json.dumps(session.stop(served_calls=len(rows)).to_json(), indent=2) + "\n")
    return result


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    output = Path(sys.argv[2])
    output.mkdir()
    inputs = {}
    for layer in LAYERS:
        for r in range(4):
            data = (Path(config["input_dir"]) / f"input-layer{layer}-row{r}.f16").read_bytes()
            assert len(data) == 2 * N_EMBD
            inputs[(layer, r)] = data
    summary = []
    for arm in config["arms"]:
        summary.append(run_arm(config, arm, output, inputs))
        time.sleep(config.get("cooldown_s", 10))
    (output / "SUITE_RESULT.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PASS", [s["arm"] for s in summary])


if __name__ == "__main__":
    main()
