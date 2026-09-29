"""Finite ADB TCP FFN timing calibration using the qualified worker lifecycle."""

import fcntl
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time

sys.path.insert(0, os.environ["S42_UNIFIED_REPO_ROOT"])
from research_dev.scheduler.campaigns.burstgpt.two_phone_gate import helper_configuration  # noqa: E402
from research_dev.scheduler.adapters.phone_tcp_session import (  # noqa: E402
    AdbTcpPhoneWorkerSession, HELLO_REQUEST, HELLO_RESPONSE, EXECUTE_REQUEST, EXECUTE_RESPONSE,
    PROTOCOL_MAGIC, PROTOCOL_VERSION, _fnv32,
)


def main():
    root = Path(sys.argv[1])
    config = json.loads((root / "GATE_CONFIG.json").read_text())
    output = root / sys.argv[2]
    output.mkdir()
    config["helper_phone"]["max_requests"] = 216
    artifact = json.loads(Path(config["manifest"]).read_text())["artifact_sha256"]
    session = AdbTcpPhoneWorkerSession(helper_configuration(config, artifact, 5120, True))
    rows = []
    with Path(config["execution_lock"]).open("a") as lock:
        deadline = time.monotonic() + 900
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        (output / "PREFLIGHT.json").write_text(json.dumps(session.preflight().to_json(), indent=2) + "\n")
        (output / "LAUNCH.json").write_text(json.dumps(session.start(output / "worker.log").to_json(), indent=2) + "\n")
        try:
            with socket.create_connection(("127.0.0.1", session.transport_contract().control_port), timeout=60) as stream:
                stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                stream.sendall(HELLO_REQUEST.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, 1, 16515072,
                                                 5120, 17408, 3, 4, bytes.fromhex(artifact[7:])))
                hello = HELLO_RESPONSE.unpack(session._receive(stream, HELLO_RESPONSE.size))
                assert hello[:4] == (PROTOCOL_MAGIC, PROTOCOL_VERSION, 2, 0)
                for repeat in range(6):
                    for batch in (1, 2, 4):
                        payload = b"".join(struct.pack("<e", ((index % 29) - 14) / 16)
                                           for index in range(5120 * batch))
                        for columns in (8704, 17408):
                            for layer in range(18, 24):
                                request_id = len(rows) + 1
                                time.sleep(0.005)
                                started = time.monotonic_ns()
                                stream.sendall(EXECUTE_REQUEST.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, 3,
                                    request_id, layer, 5120 * batch, len(payload), _fnv32(payload), columns, batch) + payload)
                                response = EXECUTE_RESPONSE.unpack(session._receive(stream, EXECUTE_RESPONSE.size))
                                data = session._receive(stream, len(payload))
                                elapsed = (time.monotonic_ns() - started) / 1000
                                assert response[:6] == (PROTOCOL_MAGIC, PROTOCOL_VERSION, 4, 0, 0, request_id)
                                assert response[8] == len(payload) and response[9] == _fnv32(data)
                                rows.append({"repeat": repeat, "tokens": batch, "columns": columns, "layer": layer,
                                             "payload_bytes": len(payload), "rpc_us": elapsed,
                                             "compute_us": response[-1], "overhead_us": elapsed - response[-1]})
            stop = session.stop(served_calls=len(rows)).to_json()
            (output / "RESULT.json").write_text(json.dumps({"status": "PASS", "calls": rows,
                "stop": stop, "note": "FNV framing checked; timing calibration, not independent numerical qualification."}, indent=2) + "\n")
        finally:
            (output / "CALLS.json").write_text(json.dumps(rows, indent=2) + "\n")
            if session.active:
                (output / "CLEANUP.json").write_text(json.dumps(session.stop(served_calls=len(rows)).to_json(), indent=2) + "\n")


if __name__ == "__main__":
    main()
