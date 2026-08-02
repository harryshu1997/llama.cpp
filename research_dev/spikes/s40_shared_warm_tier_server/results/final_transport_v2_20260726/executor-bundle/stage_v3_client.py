#!/usr/bin/env python3
"""Strict client and lifecycle gate for the LayerSplit StageNet V3 protocol."""

from __future__ import annotations

import argparse
import json
import socket
import struct
import time
from dataclasses import asdict, dataclass, replace
from typing import Iterable, Mapping, Sequence


STAGE_STOP = -1
STAGE_DETACH = -7
STAGE_V3_HELLO = -8
STAGE_V3_BATCH = -9
STAGE_V3_SEQ_REMOVE = -10
STAGE_V3_STATUS = -11
STAGE_V3_DRAIN = -12
STAGE_V3_IDENTITY = -13
STAGE_V3_RANGE_BATCH = -14

STAGE_V3_MAGIC = 0x4C535633
STAGE_V3_VERSION = 3
STAGE_IDENTITY_MAGIC = 0x4C534944
STAGE_IDENTITY_VERSION = 1
STAGE_V3_BASE_CAPABILITIES = 0x0F
STAGE_V3_CAP_TERMINAL = 0x10
STAGE_V3_CAP_IDENTITY = 0x20
STAGE_V3_CAP_RANGE = 0x40

MAX_ROWS = 64
MAX_EMBEDDING = 1 << 20


class ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hello:
    layer_start: int
    layer_end: int
    n_layer: int
    n_embd: int
    max_streams: int
    n_ctx_seq: int
    n_batch: int
    n_ubatch: int
    capabilities: int
    file_type: int | None = None
    model_sha256: str | None = None


@dataclass(frozen=True)
class ModelIdentity:
    file_type: int
    model_sha256: str


@dataclass(frozen=True)
class Status:
    active_sequences: int
    max_streams: int
    draining: bool


@dataclass(frozen=True)
class BatchRow:
    request_id: int
    route_epoch: int
    seq_id: int
    position: int
    token: int
    hidden: Sequence[float] | None = None


@dataclass(frozen=True)
class BatchResult:
    request_id: int
    route_epoch: int
    seq_id: int
    position: int
    hidden: tuple[float, ...] | None
    token: int | None


def _pack_i32(values: Iterable[int]) -> bytes:
    values = tuple(values)
    return struct.pack(f"<{len(values)}i", *values)


def _pack_i64(values: Iterable[int]) -> bytes:
    values = tuple(values)
    return struct.pack(f"<{len(values)}q", *values)


def _pack_f32(values: Iterable[float]) -> bytes:
    values = tuple(values)
    return struct.pack(f"<{len(values)}f", *values)


class StageV3Client:
    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._hello: Hello | None = None
        self._identity: ModelIdentity | None = None

    @classmethod
    def connect(cls, host: str, port: int, timeout_s: float) -> "StageV3Client":
        sock = socket.create_connection((host, port), timeout=timeout_s)
        sock.settimeout(timeout_s)
        return cls(sock)

    def close(self) -> None:
        self._sock.close()

    def _recv_exact(self, size: int) -> bytes:
        if size < 0:
            raise ProtocolError("negative receive size")
        out = bytearray()
        while len(out) < size:
            chunk = self._sock.recv(size - len(out))
            if not chunk:
                raise ProtocolError("unexpected EOF")
            out.extend(chunk)
        return bytes(out)

    def _recv_i32(self, count: int) -> tuple[int, ...]:
        return struct.unpack(f"<{count}i", self._recv_exact(4 * count))

    def _recv_i64(self, count: int) -> tuple[int, ...]:
        return struct.unpack(f"<{count}q", self._recv_exact(8 * count))

    def hello(self) -> Hello:
        self._sock.sendall(_pack_i32([STAGE_V3_HELLO]))
        words = self._recv_i32(11)
        if words[0] != STAGE_V3_MAGIC or words[1] != STAGE_V3_VERSION:
            raise ProtocolError("StageNet V3 hello mismatch")
        hello = Hello(*words[2:])
        if not (0 <= hello.layer_start < hello.layer_end <= hello.n_layer):
            raise ProtocolError("invalid worker layer range")
        if not (0 < hello.n_embd <= MAX_EMBEDDING):
            raise ProtocolError("invalid embedding width")
        if not (0 < hello.max_streams <= MAX_ROWS):
            raise ProtocolError("invalid stream capacity")
        if hello.n_ctx_seq <= 0 or hello.n_batch <= 0 or hello.n_ubatch <= 0:
            raise ProtocolError("invalid worker capacity")
        if hello.capabilities & STAGE_V3_BASE_CAPABILITIES != STAGE_V3_BASE_CAPABILITIES:
            raise ProtocolError("unsupported StageNet V3 capabilities")
        known_capabilities = (
            STAGE_V3_BASE_CAPABILITIES
            | STAGE_V3_CAP_TERMINAL
            | STAGE_V3_CAP_IDENTITY
            | STAGE_V3_CAP_RANGE
        )
        if hello.capabilities & ~known_capabilities:
            raise ProtocolError("unknown StageNet V3 capability")
        terminal = bool(hello.capabilities & STAGE_V3_CAP_TERMINAL)
        if terminal != (hello.layer_end == hello.n_layer):
            raise ProtocolError("terminal capability and layer range disagree")
        self._hello = hello
        if hello.capabilities & STAGE_V3_CAP_IDENTITY:
            identity = self.identity()
            hello = replace(
                hello,
                file_type=identity.file_type,
                model_sha256=identity.model_sha256,
            )
            self._hello = hello
        return hello

    def identity(self) -> ModelIdentity:
        if self._identity is not None:
            return self._identity
        if self._hello is None:
            raise ProtocolError("hello must precede identity")
        if not self._hello.capabilities & STAGE_V3_CAP_IDENTITY:
            raise ProtocolError("worker omitted model identity")
        self._sock.sendall(_pack_i32([STAGE_V3_IDENTITY]))
        magic, version, file_type = self._recv_i32(3)
        digest = self._recv_exact(32)
        if magic != STAGE_IDENTITY_MAGIC or version != STAGE_IDENTITY_VERSION:
            raise ProtocolError("StageNet model identity mismatch")
        if file_type < 0:
            raise ProtocolError("invalid GGUF file type")
        self._identity = ModelIdentity(file_type, digest.hex())
        return self._identity

    def _recv_status(self) -> tuple[int, Status]:
        words = self._recv_i32(5)
        if words[1] != STAGE_V3_VERSION:
            raise ProtocolError("status version mismatch")
        status = Status(words[2], words[3], bool(words[4]))
        if status.active_sequences < 0 or status.max_streams <= 0:
            raise ProtocolError("invalid status counters")
        if status.active_sequences > status.max_streams or words[4] not in (0, 1):
            raise ProtocolError("invalid status state")
        if self._hello is not None and status.max_streams != self._hello.max_streams:
            raise ProtocolError("status stream capacity changed")
        return words[0], status

    def status(self) -> Status:
        self._sock.sendall(_pack_i32([STAGE_V3_STATUS, STAGE_V3_VERSION]))
        code, status = self._recv_status()
        if code != 0:
            raise ProtocolError(f"worker status failed: {code}")
        return status

    def drain(self) -> Status:
        self._sock.sendall(_pack_i32([STAGE_V3_DRAIN, STAGE_V3_VERSION]))
        code, status = self._recv_status()
        if code != 0 or not status.draining:
            raise ProtocolError("worker drain failed")
        return status

    def remove(self, seq_id: int, request_id: int, route_epoch: int) -> Status:
        payload = _pack_i32([STAGE_V3_SEQ_REMOVE, STAGE_V3_VERSION, seq_id])
        payload += _pack_i64([request_id, route_epoch])
        self._sock.sendall(payload)
        code, status = self._recv_status()
        if code != 0:
            raise ProtocolError(f"sequence removal failed: {code}")
        return status

    def batch(self, rows: Sequence[BatchRow]) -> tuple[BatchResult, ...]:
        return self._batch(rows, None)

    def range_batch(
        self,
        rows: Sequence[BatchRow],
        layer_start: int,
        layer_end: int,
    ) -> tuple[BatchResult, ...]:
        if self._hello is None:
            raise ProtocolError("hello must precede range batch")
        if not self._hello.capabilities & STAGE_V3_CAP_RANGE:
            raise ProtocolError("worker does not support dynamic layer cuts")
        if (
            type(layer_start) is not int
            or type(layer_end) is not int
            or layer_start < self._hello.layer_start
            or layer_end > self._hello.layer_end
            or layer_start >= layer_end
        ):
            raise ProtocolError("active layer range is outside worker residency")
        if self._hello.layer_start == 0 and layer_start != 0:
            raise ProtocolError("head range must start at layer zero")
        if (
            self._hello.capabilities & STAGE_V3_CAP_TERMINAL
            and layer_end != self._hello.n_layer
        ):
            raise ProtocolError("terminal range must end at the final layer")
        return self._batch(rows, (layer_start, layer_end))

    def _batch(
        self,
        rows: Sequence[BatchRow],
        active_range: tuple[int, int] | None,
    ) -> tuple[BatchResult, ...]:
        if self._hello is None:
            raise ProtocolError("hello must precede batch")
        if not rows or len(rows) > min(self._hello.n_batch, self._hello.n_ubatch):
            raise ProtocolError("batch size exceeds worker capacity")
        hidden_widths = {0 if row.hidden is None else len(row.hidden) for row in rows}
        if len(hidden_widths) != 1:
            raise ProtocolError("mixed hidden widths")
        hidden_width = hidden_widths.pop()
        required_width = self._hello.n_embd if self._hello.layer_start > 0 else 0
        if hidden_width != required_width:
            raise ProtocolError("hidden width does not match worker layer range")

        opcode = STAGE_V3_RANGE_BATCH if active_range is not None else STAGE_V3_BATCH
        header = [opcode, STAGE_V3_VERSION, len(rows), hidden_width]
        if active_range is not None:
            header.extend(active_range)
        payload = _pack_i32(header)
        payload += _pack_i64(row.request_id for row in rows)
        payload += _pack_i64(row.route_epoch for row in rows)
        payload += _pack_i32(row.seq_id for row in rows)
        payload += _pack_i32(row.position for row in rows)
        payload += _pack_i32(row.token for row in rows)
        if hidden_width:
            payload += _pack_f32(value for row in rows for value in row.hidden or ())
        self._sock.sendall(payload)

        code = self._recv_i32(1)[0]
        if code != 0:
            raise ProtocolError(f"worker rejected batch: {code}")
        n_rows, n_embd = self._recv_i32(2)
        if active_range is not None:
            echoed_range = self._recv_i32(2)
            if echoed_range != active_range:
                raise ProtocolError("batch response layer range mismatch")
        terminal = bool(self._hello.capabilities & STAGE_V3_CAP_TERMINAL)
        expected_width = 0 if terminal else self._hello.n_embd
        if n_rows != len(rows) or n_embd != expected_width:
            raise ProtocolError("batch response shape mismatch")
        request_ids = self._recv_i64(n_rows)
        route_epochs = self._recv_i64(n_rows)
        seq_ids = self._recv_i32(n_rows)
        positions = self._recv_i32(n_rows)
        if terminal:
            tokens = self._recv_i32(n_rows)
            hidden: tuple[float, ...] = ()
        else:
            tokens = tuple()
            hidden_raw = self._recv_exact(n_rows * n_embd * 4)
            hidden = struct.unpack(f"<{n_rows * n_embd}f", hidden_raw)

        expected = tuple(
            (row.request_id, row.route_epoch, row.seq_id, row.position)
            for row in rows
        )
        actual = tuple(zip(request_ids, route_epochs, seq_ids, positions))
        if actual != expected:
            raise ProtocolError("batch response lineage mismatch")
        return tuple(
            BatchResult(
                request_ids[index], route_epochs[index], seq_ids[index],
                positions[index],
                None if terminal else tuple(hidden[index * n_embd:(index + 1) * n_embd]),
                tokens[index] if terminal else None,
            )
            for index in range(n_rows)
        )

    def stop(self) -> None:
        self._sock.sendall(_pack_i32([STAGE_STOP]))

    def detach(self) -> None:
        self._sock.sendall(_pack_i32([STAGE_DETACH]))
        if self._recv_i32(1)[0] != 0:
            raise ProtocolError("worker detach failed")


def require_same_model(
    hellos: Mapping[str, Hello],
    expected_model_sha256: str | None = None,
    expected_file_type: int | None = None,
) -> ModelIdentity:
    if not hellos:
        raise ProtocolError("route has no workers")
    if expected_file_type is not None and (
        type(expected_file_type) is not int or expected_file_type < 0
    ):
        raise ProtocolError("invalid requested GGUF file type")
    if expected_model_sha256 is not None and (
        type(expected_model_sha256) is not str
        or len(expected_model_sha256) != 64
        or any(value not in "0123456789abcdef" for value in expected_model_sha256)
    ):
        raise ProtocolError("invalid requested model SHA-256")
    identities: dict[str, ModelIdentity] = {}
    for name, hello in hellos.items():
        if (
            type(hello.file_type) is not int
            or hello.file_type < 0
            or type(hello.model_sha256) is not str
            or len(hello.model_sha256) != 64
            or any(value not in "0123456789abcdef" for value in hello.model_sha256)
        ):
            raise ProtocolError(f"{name} omitted model identity")
        identities[name] = ModelIdentity(hello.file_type, hello.model_sha256)
    file_types = {identity.file_type for identity in identities.values()}
    if len(file_types) != 1:
        raise ProtocolError("route quantization mismatch")
    model_sha256s = {identity.model_sha256 for identity in identities.values()}
    if len(model_sha256s) != 1:
        raise ProtocolError("route model SHA-256 mismatch")
    identity = next(iter(identities.values()))
    if expected_file_type is not None and identity.file_type != expected_file_type:
        raise ProtocolError("route GGUF file type differs from the requested type")
    if (
        expected_model_sha256 is not None
        and identity.model_sha256 != expected_model_sha256
    ):
        raise ProtocolError("route model SHA-256 differs from the requested model")
    return identity


def run_lifecycle(client: StageV3Client, token: int, session_end: str) -> dict:
    started_ns = time.monotonic_ns()
    hello = client.hello()
    if hello.layer_start != 0:
        raise ProtocolError("lifecycle gate requires a head stage")
    if hello.max_streams < 2:
        raise ProtocolError("lifecycle gate requires at least two sequence slots")

    client.batch([BatchRow(101, 1, 0, 0, token)])
    after_first = client.status()
    client.batch([BatchRow(102, 1, 1, 0, token)])
    after_admit = client.status()
    client.batch([
        BatchRow(101, 1, 0, 1, token),
        BatchRow(102, 1, 1, 1, token),
    ])
    after_batch = client.status()
    after_remove = client.remove(0, 101, 1)
    client.batch([BatchRow(103, 1, 0, 0, token)])
    after_reuse = client.status()
    client.remove(0, 103, 1)
    client.remove(1, 102, 1)
    after_drain = client.drain()

    if [after_first.active_sequences, after_admit.active_sequences,
        after_batch.active_sequences, after_remove.active_sequences,
        after_reuse.active_sequences, after_drain.active_sequences] != [1, 2, 2, 1, 2, 0]:
        raise ProtocolError("lifecycle active-sequence counts differ from contract")
    if not after_drain.draining:
        raise ProtocolError("worker did not enter draining state")

    if session_end == "stop":
        client.stop()
    elif session_end == "detach":
        client.detach()

    return {
        "schema": "ls-stagenet-v3-lifecycle-v1",
        "verdict": "PASS",
        "hello": asdict(hello),
        "active_counts": [1, 2, 2, 1, 2, 0],
        "session_end": session_end,
        "elapsed_us": (time.monotonic_ns() - started_ns) // 1000,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--token", type=int, default=2)
    parser.add_argument("--session-end", choices=("stop", "detach", "leave"), default="stop")
    args = parser.parse_args()
    if not (1 <= args.port <= 65535) or args.timeout <= 0:
        parser.error("invalid port or timeout")

    client = StageV3Client.connect(args.host, args.port, args.timeout)
    try:
        report = run_lifecycle(client, args.token, args.session_end)
    finally:
        client.close()
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ProtocolError) as exc:
        print(json.dumps({"verdict": "FAIL", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2)
