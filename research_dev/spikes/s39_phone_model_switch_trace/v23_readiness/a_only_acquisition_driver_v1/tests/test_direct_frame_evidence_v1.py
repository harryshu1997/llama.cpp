#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "producers_v1" / "direct_frame_evidence_v1.py"


def load_source():
    spec = importlib.util.spec_from_file_location("s39_direct_frame_test", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load direct frame parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frames = load_source()


def frame(call_index, payload, position):
    return {
        "activation_payload_bytes": len(payload),
        "call_index": call_index,
        "hidden_width": 2,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "positions": [position, position],
        "request_ids": [101, 102],
        "route_epochs": [201, 202],
        "rows": 2,
        "schema": frames.SCHEMA,
        "seq_ids": [0, 1],
    }


def line(value):
    return frames.PREFIX + frames.canonical_bytes(value)


def shape(value):
    return {key: value[key] for key in frames.SHAPE_KEYS}


class DirectFrameEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.payloads = [
            b"\x00\x00\x00\x00\x00\x00\x80?" * 2,
            b"\x00\x00\x00@\x00\x00@@" * 2,
        ]
        self.records = [
            frame(0, self.payloads[0], 0),
            frame(1, self.payloads[1], 1),
        ]
        self.raw = (
            b"[direct-relay] ready\n"
            + line(self.records[0])
            + line(self.records[1])
            + b"DIRECTCERT {}\n"
        )

    def test_exact_frames_and_payloads_pass(self):
        result = frames.parse_direct_frames(
            self.raw,
            [shape(value) for value in self.records],
            self.payloads,
        )
        self.assertEqual(result, self.records)

    def test_payload_hash_mutation_is_rejected(self):
        payloads = copy.deepcopy(self.payloads)
        payloads[1] = b"x" + payloads[1][1:]
        with self.assertRaisesRegex(frames.DirectFrameError, "PAYLOAD_SHA256"):
            frames.parse_direct_frames(self.raw, None, payloads)

    def test_lineage_mutation_is_rejected(self):
        expected = [shape(value) for value in self.records]
        expected[0]["request_ids"][0] = 999
        with self.assertRaisesRegex(frames.DirectFrameError, "CALL_SHAPE"):
            frames.parse_direct_frames(self.raw, expected)

    def test_missing_duplicate_and_reordered_calls_are_rejected(self):
        invalid = (
            b"no frame\n",
            line(self.records[0]) + line(self.records[0]),
            line(self.records[1]) + line(self.records[0]),
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(frames.DirectFrameError):
                    frames.parse_direct_frames(raw)

    def test_noncanonical_duplicate_key_and_near_prefix_are_rejected(self):
        noncanonical = json.dumps(self.records[0]).encode("ascii") + b"\n"
        duplicate = (
            b'{"activation_payload_bytes":16,"activation_payload_bytes":16,'
            b'"call_index":0,"hidden_width":2,"payload_sha256":"'
            + self.records[0]["payload_sha256"].encode("ascii")
            + b'","positions":[0,0],"request_ids":[101,102],'
            b'"route_epochs":[201,202],"rows":2,'
            b'"schema":"ls-stage-direct-frame-v1","seq_ids":[0,1]}\n'
        )
        invalid = (
            frames.PREFIX + noncanonical,
            frames.PREFIX + duplicate,
            b"x DIRECTFRAME {}\n",
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(frames.DirectFrameError):
                    frames.parse_direct_frames(raw)

    def test_payload_size_and_lineage_reuse_are_rejected(self):
        for mutate, message in (
            (
                lambda value: value.__setitem__("activation_payload_bytes", 8),
                "PAYLOAD_BYTES",
            ),
            (
                lambda value: value.__setitem__("seq_ids", [0, 0]),
                "LINEAGE_REUSE",
            ),
        ):
            with self.subTest(message=message):
                value = copy.deepcopy(self.records[0])
                mutate(value)
                with self.assertRaisesRegex(frames.DirectFrameError, message):
                    frames.parse_direct_frames(line(value))


if __name__ == "__main__":
    unittest.main()
