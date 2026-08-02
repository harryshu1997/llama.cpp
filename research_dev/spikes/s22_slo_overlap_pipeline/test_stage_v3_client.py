#!/usr/bin/env python3

from __future__ import annotations

import socket
import struct
import threading
import unittest

from stage_v3_client import (
    BatchRow,
    Hello,
    ProtocolError,
    STAGE_IDENTITY_MAGIC,
    STAGE_IDENTITY_VERSION,
    STAGE_V3_BASE_CAPABILITIES,
    STAGE_V3_CAP_IDENTITY,
    STAGE_V3_CAP_RANGE,
    STAGE_V3_CAP_TERMINAL,
    STAGE_V3_HELLO,
    STAGE_V3_IDENTITY,
    STAGE_V3_RANGE_BATCH,
    STAGE_V3_MAGIC,
    STAGE_V3_VERSION,
    StageV3Client,
    require_same_model,
)


def i32(*values: int) -> bytes:
    return struct.pack(f"<{len(values)}i", *values)


def i64(*values: int) -> bytes:
    return struct.pack(f"<{len(values)}q", *values)


class ScriptedPeer:
    def __init__(self, response: bytes):
        self.client, self.server = socket.socketpair()
        self.response = response
        self.received = bytearray()
        self.thread = threading.Thread(target=self._run)

    def _run(self) -> None:
        try:
            self.received.extend(self.server.recv(1 << 20))
            self.server.sendall(self.response)
        finally:
            self.server.close()

    def __enter__(self) -> "ScriptedPeer":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.client.close()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise AssertionError("scripted peer did not stop")


def hello_bytes(
    layer_start: int = 0,
    layer_end: int = 8,
    terminal: bool = False,
    identity: bool = False,
    dynamic_range: bool = False,
) -> bytes:
    return i32(
        STAGE_V3_MAGIC, STAGE_V3_VERSION, layer_start, layer_end, 48, 4, 4,
        128, 64, 64,
        STAGE_V3_BASE_CAPABILITIES
        | (STAGE_V3_CAP_TERMINAL if terminal else 0)
        | (STAGE_V3_CAP_IDENTITY if identity else 0)
        | (STAGE_V3_CAP_RANGE if dynamic_range else 0),
    )


def hello_with_identity(file_type: int, digest: str) -> Hello:
    return Hello(
        0, 8, 48, 4, 4, 128, 64, 64,
        STAGE_V3_BASE_CAPABILITIES | STAGE_V3_CAP_IDENTITY,
        file_type,
        digest,
    )


class StageV3ClientTests(unittest.TestCase):
    def test_hello_parses_frozen_shape(self) -> None:
        with ScriptedPeer(hello_bytes()) as peer:
            client = StageV3Client(peer.client)
            hello = client.hello()
            self.assertEqual((hello.layer_start, hello.layer_end), (0, 8))
            self.assertEqual(hello.max_streams, 4)

    def test_hello_rejects_bad_magic(self) -> None:
        response = bytearray(hello_bytes())
        response[0:4] = i32(0)
        with ScriptedPeer(bytes(response)) as peer:
            with self.assertRaisesRegex(ProtocolError, "hello mismatch"):
                StageV3Client(peer.client).hello()

    def test_hello_reads_model_identity(self) -> None:
        digest = "ab" * 32
        client_sock, server_sock = socket.socketpair()

        def server() -> None:
            try:
                self.assertEqual(server_sock.recv(4), i32(STAGE_V3_HELLO))
                server_sock.sendall(hello_bytes(identity=True))
                self.assertEqual(server_sock.recv(4), i32(STAGE_V3_IDENTITY))
                server_sock.sendall(i32(
                    STAGE_IDENTITY_MAGIC, STAGE_IDENTITY_VERSION, 7,
                ) + bytes.fromhex(digest))
            finally:
                server_sock.close()

        thread = threading.Thread(target=server)
        thread.start()
        client = StageV3Client(client_sock)
        try:
            hello = client.hello()
            self.assertEqual(hello.file_type, 7)
            self.assertEqual(hello.model_sha256, digest)
            self.assertEqual(client.identity().model_sha256, digest)
        finally:
            client.close()
            thread.join(timeout=2)

    def test_route_rejects_quantization_mismatch(self) -> None:
        digest = "ab" * 32
        with self.assertRaisesRegex(ProtocolError, "quantization mismatch"):
            require_same_model({
                "desktop": hello_with_identity(7, digest),
                "phone": hello_with_identity(2, digest),
            })

    def test_route_rejects_model_mismatch(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "SHA-256 mismatch"):
            require_same_model({
                "desktop": hello_with_identity(7, "ab" * 32),
                "phone": hello_with_identity(7, "cd" * 32),
            })

    def test_route_accepts_exact_identity(self) -> None:
        digest = "ab" * 32
        identity = require_same_model(
            {
                "desktop": hello_with_identity(7, digest),
                "op12": hello_with_identity(7, digest),
                "op15": hello_with_identity(7, digest),
            },
            expected_model_sha256=digest,
            expected_file_type=7,
        )
        self.assertEqual((identity.file_type, identity.model_sha256), (7, digest))

    def test_status_rejects_changed_capacity(self) -> None:
        responses = hello_bytes() + i32(0, STAGE_V3_VERSION, 0, 3, 0)
        client_sock, server_sock = socket.socketpair()

        def server() -> None:
            try:
                server_sock.recv(4)
                server_sock.sendall(responses[:44])
                server_sock.recv(8)
                server_sock.sendall(responses[44:])
            finally:
                server_sock.close()

        thread = threading.Thread(target=server)
        thread.start()
        client = StageV3Client(client_sock)
        try:
            client.hello()
            with self.assertRaisesRegex(ProtocolError, "capacity changed"):
                client.status()
        finally:
            client.close()
            thread.join(timeout=2)

    def test_batch_requires_hello(self) -> None:
        left, right = socket.socketpair()
        try:
            with self.assertRaisesRegex(ProtocolError, "hello must precede"):
                StageV3Client(left).batch([BatchRow(1, 1, 0, 0, 2)])
        finally:
            left.close()
            right.close()

    def test_batch_rejects_response_lineage_mutation(self) -> None:
        client_sock, server_sock = socket.socketpair()

        def server() -> None:
            try:
                server_sock.recv(4)
                server_sock.sendall(hello_bytes())
                server_sock.recv(1 << 20)
                server_sock.sendall(i32(0, 1, 4))
                server_sock.sendall(i64(99))
                server_sock.sendall(i64(1))
                server_sock.sendall(i32(0, 0))
                server_sock.sendall(struct.pack("<4f", 0.0, 0.0, 0.0, 0.0))
            finally:
                server_sock.close()

        thread = threading.Thread(target=server)
        thread.start()
        client = StageV3Client(client_sock)
        try:
            client.hello()
            with self.assertRaisesRegex(ProtocolError, "lineage mismatch"):
                client.batch([BatchRow(1, 1, 0, 0, 2)])
        finally:
            client.close()
            thread.join(timeout=2)

    def test_mid_stage_requires_hidden_width(self) -> None:
        with ScriptedPeer(hello_bytes(layer_start=2)) as peer:
            client = StageV3Client(peer.client)
            client.hello()
            with self.assertRaisesRegex(ProtocolError, "hidden width"):
                client.batch([BatchRow(1, 1, 0, 0, 2)])

    def test_range_batch_round_trip(self) -> None:
        client_sock, server_sock = socket.socketpair()
        received = bytearray()

        def server() -> None:
            try:
                server_sock.recv(4)
                server_sock.sendall(hello_bytes(dynamic_range=True))
                received.extend(server_sock.recv(1 << 20))
                server_sock.sendall(i32(0, 1, 4, 0, 4))
                server_sock.sendall(i64(1, 1))
                server_sock.sendall(i32(0, 0))
                server_sock.sendall(struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))
            finally:
                server_sock.close()

        thread = threading.Thread(target=server)
        thread.start()
        client = StageV3Client(client_sock)
        try:
            client.hello()
            result = client.range_batch(
                [BatchRow(1, 1, 0, 0, 2)], 0, 4,
            )
            self.assertEqual(result[0].hidden, (1.0, 2.0, 3.0, 4.0))
            self.assertEqual(struct.unpack("<6i", received[:24]), (
                STAGE_V3_RANGE_BATCH, STAGE_V3_VERSION, 1, 0, 0, 4,
            ))
        finally:
            client.close()
            thread.join(timeout=2)

    def test_range_batch_requires_capability(self) -> None:
        with ScriptedPeer(hello_bytes()) as peer:
            client = StageV3Client(peer.client)
            client.hello()
            with self.assertRaisesRegex(ProtocolError, "does not support"):
                client.range_batch([BatchRow(1, 1, 0, 0, 2)], 0, 4)

    def test_head_range_must_start_at_zero(self) -> None:
        with ScriptedPeer(hello_bytes(dynamic_range=True)) as peer:
            client = StageV3Client(peer.client)
            client.hello()
            with self.assertRaisesRegex(ProtocolError, "head range"):
                client.range_batch([BatchRow(1, 1, 0, 0, 2)], 1, 4)

    def test_terminal_batch_returns_token(self) -> None:
        client_sock, server_sock = socket.socketpair()

        def server() -> None:
            try:
                server_sock.recv(4)
                server_sock.sendall(hello_bytes(8, 48, terminal=True))
                server_sock.recv(1 << 20)
                server_sock.sendall(i32(0, 1, 0))
                server_sock.sendall(i64(1, 1))
                server_sock.sendall(i32(0, 0, 7))
            finally:
                server_sock.close()

        thread = threading.Thread(target=server)
        thread.start()
        client = StageV3Client(client_sock)
        try:
            client.hello()
            result = client.batch([BatchRow(1, 1, 0, 0, 2, [0.0] * 4)])
            self.assertEqual(result[0].token, 7)
            self.assertIsNone(result[0].hidden)
        finally:
            client.close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
