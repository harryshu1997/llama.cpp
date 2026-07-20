#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import physical_mux as mux


def placement(**changes) -> bytes:
    value = {
        "schema": "layersplit-scheduled-placement-v2",
        "role": "host_tail",
        "mode": "pipedriver",
        "layer_start": 8,
        "layer_end": 48,
        "n_layer": 48,
        "pid": 123,
        "run_rc": 0,
        "compute_nodes": 3,
        "copy_nodes": 0,
        "metadata_nodes": 2,
        "missing_buffer_compute_nodes": 0,
        "compute_by_buffer_type": {"CUDA0": 3},
        "compute_by_op": {"ADD": 1, "MUL_MAT": 2},
        "compute_by_op_and_buffer": {
            "ADD": {"CUDA0": 1},
            "MUL_MAT": {"CUDA0": 2},
        },
        "copy_by_buffer_type": {},
        "status": "SCHEDULED_PLACEMENT_OK",
    }
    value.update(changes)
    return (json.dumps(value, separators=(",", ":")) + "\n").encode("ascii")


class PhysicalMuxTests(unittest.TestCase):
    def test_real_ordered_host_placement_passes(self) -> None:
        value = mux.validate_host_placement(placement(), 123, 8, 48, "CUDA0")
        self.assertEqual(value["compute_nodes"], 3)

    def test_phone_certificate_need_not_sort_keys(self) -> None:
        payload = b'{"session_id":1,"schema":"ls-stagenet-session-v2"}\n'
        self.assertEqual(mux.parse_object(payload, "cert")["session_id"], 1)
        with self.assertRaisesRegex(mux.MuxError, "not canonical"):
            mux.strict_line(payload, "cert")

    def test_duplicate_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(mux.MuxError, "duplicate key"):
            mux.parse_object(b'{"session_id":1,"session_id":2}\n', "cert")

    def test_cpu_or_wrong_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(mux.MuxError, "unexpected backend"):
            mux.validate_host_placement(
                placement(compute_by_buffer_type={"CPU": 3}), 123, 8, 48, "CUDA0",
            )
        with self.assertRaisesRegex(mux.MuxError, "identity or status"):
            mux.validate_host_placement(placement(layer_start=7), 123, 8, 48, "CUDA0")

    def test_missing_compute_and_bool_counts_are_rejected(self) -> None:
        with self.assertRaisesRegex(mux.MuxError, "compute coverage"):
            mux.validate_host_placement(placement(compute_nodes=0), 123, 8, 48, "CUDA0")
        with self.assertRaisesRegex(mux.MuxError, "compute coverage"):
            mux.validate_host_placement(placement(compute_nodes=True), 123, 8, 48, "CUDA0")

    def test_config_path_type_and_existing_artifact_root_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "schema": mux.CONFIG_SCHEMA,
                "host_command": ["host"],
                "phone_command": ["phone"],
                "host_env": {"A": "B"},
                "artifact_root": 7,
                "host_layer_start": 8,
                "host_layer_end": 48,
                "host_backend": "CUDA0",
            }
            path = root / "config.json"
            path.write_bytes(mux.canonical(config))
            with self.assertRaisesRegex(mux.MuxError, "must be a string"):
                mux.load_config(path)
            config["artifact_root"] = str(root)
            path.write_bytes(mux.canonical(config))
            with self.assertRaisesRegex(mux.MuxError, "absent absolute path"):
                mux.load_config(path)


if __name__ == "__main__":
    unittest.main()
