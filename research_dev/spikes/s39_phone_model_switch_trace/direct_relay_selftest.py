#!/usr/bin/env python3
"""Protocol self-test for llama-stage-direct-relay with mock workers."""

from __future__ import annotations

import argparse
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
if str(S22) not in sys.path:
    sys.path.insert(0, str(S22))

from stage_v3_client import BatchRow, StageV3Client


STAGE_STOP = -1
STAGE_V3_HELLO = -8
STAGE_V3_BATCH = -9
STAGE_V3_SEQ_REMOVE = -10
STAGE_V3_STATUS = -11
STAGE_V3_DRAIN = -12
STAGE_V3_IDENTITY = -13
STAGE_V3_MAGIC = 0x4C535633
STAGE_V3_VERSION = 3
STAGE_IDENTITY_MAGIC = 0x4C534944
STAGE_IDENTITY_VERSION = 1
CAPABILITIES = 1 | 2 | 4 | 8 | 32
CAP_TERMINAL = 16
MODEL_DIGEST = bytes.fromhex("12" * 32)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = sock.recv(size - len(result))
        if not chunk:
            raise RuntimeError("mock worker unexpected EOF")
        result.extend(chunk)
    return bytes(result)


def recv_i32(sock: socket.socket, count: int) -> tuple[int, ...]:
    return struct.unpack(f"<{count}i", recv_exact(sock, 4 * count))


def recv_i64(sock: socket.socket, count: int) -> tuple[int, ...]:
    return struct.unpack(f"<{count}q", recv_exact(sock, 8 * count))


def send_i32(sock: socket.socket, values: list[int] | tuple[int, ...]) -> None:
    sock.sendall(struct.pack(f"<{len(values)}i", *values))


def send_i64(sock: socket.socket, values: tuple[int, ...]) -> None:
    sock.sendall(struct.pack(f"<{len(values)}q", *values))


class MockWorker:
    def __init__(
        self,
        layer_start: int,
        layer_end: int,
        terminal: bool,
        *,
        allow_disconnect: bool = False,
        n_batch: int = 2048,
        status_max_streams: int = 32,
    ):
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.terminal = terminal
        self.allow_disconnect = allow_disconnect
        self.n_batch = n_batch
        self.status_max_streams = status_max_streams
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.port = 0
        self.active: set[int] = set()
        self.activation_checks = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(timeout=5):
            raise RuntimeError("mock worker did not listen")

    def join(self) -> None:
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("mock worker did not stop")
        if self.error is not None:
            raise RuntimeError("mock worker failed") from self.error

    def _status(self, sock: socket.socket, draining: bool) -> None:
        send_i32(
            sock,
            [
                0,
                STAGE_V3_VERSION,
                len(self.active),
                self.status_max_streams,
                int(draining),
            ],
        )

    def _run(self) -> None:
        server: socket.socket | None = None
        client: socket.socket | None = None
        try:
            server = socket.socket()
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            self.port = server.getsockname()[1]
            self.ready.set()
            client, _ = server.accept()
            draining = False
            while True:
                try:
                    command = recv_i32(client, 1)[0]
                except RuntimeError as error:
                    if self.allow_disconnect and "unexpected EOF" in str(error):
                        return
                    raise
                if command == STAGE_V3_HELLO:
                    capabilities = CAPABILITIES | (
                        CAP_TERMINAL if self.terminal else 0
                    )
                    send_i32(
                        client,
                        [
                            STAGE_V3_MAGIC,
                            STAGE_V3_VERSION,
                            self.layer_start,
                            self.layer_end,
                            40,
                            4,
                            32,
                            512,
                            self.n_batch,
                            512,
                            capabilities,
                        ],
                    )
                elif command == STAGE_V3_IDENTITY:
                    send_i32(
                        client,
                        [
                            STAGE_IDENTITY_MAGIC,
                            STAGE_IDENTITY_VERSION,
                            15,
                        ],
                    )
                    client.sendall(MODEL_DIGEST)
                elif command in (STAGE_V3_STATUS, STAGE_V3_DRAIN):
                    version = recv_i32(client, 1)[0]
                    if version != STAGE_V3_VERSION:
                        raise RuntimeError("mock status version")
                    if command == STAGE_V3_DRAIN:
                        draining = True
                    self._status(client, draining)
                elif command == STAGE_V3_SEQ_REMOVE:
                    version, seq_id = recv_i32(client, 2)
                    _request_id, _route_epoch = recv_i64(client, 2)
                    if version != STAGE_V3_VERSION or seq_id not in self.active:
                        raise RuntimeError("mock invalid remove")
                    self.active.remove(seq_id)
                    self._status(client, draining)
                elif command == STAGE_V3_BATCH:
                    version, n_rows, hidden_width = recv_i32(client, 3)
                    if version != STAGE_V3_VERSION or n_rows <= 0:
                        raise RuntimeError("mock invalid batch")
                    request_ids = recv_i64(client, n_rows)
                    route_epochs = recv_i64(client, n_rows)
                    seq_ids = recv_i32(client, n_rows)
                    positions = recv_i32(client, n_rows)
                    tokens = recv_i32(client, n_rows)
                    hidden = ()
                    if hidden_width:
                        hidden = struct.unpack(
                            f"<{n_rows * hidden_width}f",
                            recv_exact(client, n_rows * hidden_width * 4),
                        )
                    self.active.update(seq_ids)
                    send_i32(client, [0, n_rows, 0 if self.terminal else 4])
                    send_i64(client, request_ids)
                    send_i64(client, route_epochs)
                    send_i32(client, seq_ids)
                    send_i32(client, positions)
                    if self.terminal:
                        expected = tuple(
                            float(token + column)
                            for token in tokens
                            for column in range(4)
                        )
                        if hidden != expected:
                            raise RuntimeError("activation did not arrive directly")
                        self.activation_checks += n_rows
                        send_i32(
                            client,
                            tuple(
                                token + 1000 + position
                                for token, position in zip(tokens, positions)
                            ),
                        )
                    else:
                        values = tuple(
                            float(token + column)
                            for token in tokens
                            for column in range(4)
                        )
                        client.sendall(
                            struct.pack(f"<{len(values)}f", *values)
                        )
                elif command == STAGE_STOP:
                    return
                else:
                    raise RuntimeError(f"mock unknown command {command}")
        except BaseException as error:
            self.error = error
        finally:
            if client is not None:
                client.close()
            if server is not None:
                server.close()
            self.ready.set()


def connect_with_retry(port: int) -> StageV3Client:
    deadline = time.monotonic() + 5
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            return StageV3Client.connect("127.0.0.1", port, 5)
        except OSError as error:
            last_error = error
            time.sleep(0.02)
    raise RuntimeError("relay did not listen") from last_error


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_relay(
    relay_binary: Path,
    head: MockWorker,
    tail: MockWorker,
) -> tuple[int, subprocess.Popen[str]]:
    relay_port = free_port()
    process = subprocess.Popen(
        [
            str(relay_binary),
            "--listen",
            str(relay_port),
            "--head",
            f"127.0.0.1:{head.port}",
            "--tail",
            f"127.0.0.1:{tail.port}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return relay_port, process


def run_happy(relay_binary: Path) -> None:
    head = MockWorker(0, 30, False)
    tail = MockWorker(30, 40, True)
    head.start()
    tail.start()
    relay_port, process = start_relay(relay_binary, head, tail)
    client = connect_with_retry(relay_port)
    hello = client.hello()
    if hello.layer_start != 0 or hello.layer_end != 40:
        raise RuntimeError("composite hello range")
    rows = (
        BatchRow(10, 1, 0, 0, 7),
        BatchRow(11, 1, 1, 0, 9),
    )
    results = client.batch(rows)
    if tuple(result.token for result in results) != (1007, 1009):
        raise RuntimeError("composite token result")
    client.remove(0, 10, 1)
    client.remove(1, 11, 1)
    status = client.drain()
    if status.active_sequences != 0 or not status.draining:
        raise RuntimeError("composite drain")
    client.stop()
    client.close()
    stdout, stderr = process.communicate(timeout=5)
    if process.returncode != 0:
        raise RuntimeError(f"relay failed: {stdout}\n{stderr}")
    if (
        tail.activation_checks != 2
        or '"activation_payload_bytes":32' not in stderr
        or '"host_activation_payload_bytes":0' not in stderr
        or '"status":"DIRECT_RELAY_OK"' not in stderr
    ):
        raise RuntimeError(f"relay certificate mismatch: {stderr}")
    head.join()
    tail.join()


def connect_raw_with_retry(port: int) -> socket.socket:
    deadline = time.monotonic() + 5
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=5)
        except OSError as error:
            last_error = error
            time.sleep(0.02)
    raise RuntimeError("relay did not listen") from last_error


def run_batch_limit(relay_binary: Path) -> None:
    head = MockWorker(0, 30, False, allow_disconnect=True, n_batch=1)
    tail = MockWorker(30, 40, True, allow_disconnect=True, n_batch=1)
    head.start()
    tail.start()
    relay_port, process = start_relay(relay_binary, head, tail)
    client = connect_raw_with_retry(relay_port)
    client.sendall(struct.pack("<i", STAGE_V3_HELLO))
    hello = recv_i32(client, 11)
    if hello[8] != 1:
        raise RuntimeError("composite n_batch did not use the route minimum")
    client.sendall(struct.pack("<4i", STAGE_V3_BATCH, STAGE_V3_VERSION, 2, 0))
    if client.recv(1):
        raise RuntimeError("relay accepted a batch larger than n_batch")
    client.close()
    _stdout, stderr = process.communicate(timeout=5)
    if process.returncode != 3 or '"status":"DIRECT_RELAY_ERROR"' not in stderr:
        raise RuntimeError(f"batch-limit gate did not fail closed: {stderr}")
    head.join()
    tail.join()


def run_status_limit(relay_binary: Path) -> None:
    head = MockWorker(
        0,
        30,
        False,
        allow_disconnect=True,
        status_max_streams=31,
    )
    tail = MockWorker(
        30,
        40,
        True,
        allow_disconnect=True,
        status_max_streams=31,
    )
    head.start()
    tail.start()
    relay_port, process = start_relay(relay_binary, head, tail)
    client = connect_with_retry(relay_port)
    client.hello()
    result = client.batch((BatchRow(10, 1, 0, 0, 7),))
    if result[0].token != 1007:
        raise RuntimeError("status-limit setup batch failed")
    try:
        client.remove(0, 10, 1)
    except (OSError, RuntimeError):
        pass
    else:
        raise RuntimeError("relay accepted a mismatched status capacity")
    client.close()
    _stdout, stderr = process.communicate(timeout=5)
    if process.returncode != 3 or '"status":"DIRECT_RELAY_ERROR"' not in stderr:
        raise RuntimeError(f"status-limit gate did not fail closed: {stderr}")
    head.join()
    tail.join()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relay", type=Path, required=True)
    args = parser.parse_args()
    if not args.relay.is_file():
        parser.error("relay binary does not exist")
    relay = args.relay.resolve()
    run_happy(relay)
    run_batch_limit(relay)
    run_status_limit(relay)
    print("DIRECT_RELAY_SELFTEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
