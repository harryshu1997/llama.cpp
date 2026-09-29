"""FFN shard index resolution and its use in the phone session manifest."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from research_dev.scheduler.adapters.ffn_shards import (
    FfnShardIndex,
    FfnShardIndexError,
    remote_hash_entries,
    resolve_ffn_shard,
    verify_remote_hashes,
)
from research_dev.scheduler.adapters.phone_session import (
    DirectPhoneFfnSession,
    PhysicalAdapterError,
)

PARENT = "sha256:" + "a" * 64
OTHER = "sha256:" + "b" * 64
SHA0 = "sha256:" + "0" * 64
SHA1 = "sha256:" + "1" * 64


def _index(parent: str = PARENT) -> dict:
    def row(session, path, mask, columns, size, sha):
        return {
            "session_id": session, "path": path, "shard_sha256": sha,
            "parent_sha256": parent, "layer_mask": f"{mask:016x}",
            "columns": columns, "n_ff": 1024, "shard_bytes": size,
            "weight_type": "F16", "layers": [], "layer_spec": "",
            "column_offset": 1024 - columns,
        }
    return {
        "schema": "s42-ffn-shard-index-v1",
        "parent_sha256": parent,
        "shards": [
            row("HTP0", "HTP0.ffn.gguf", 0b000011, 512, 1000, SHA0),
            row("HTP1", "HTP1.ffn.gguf", 0b001100, 768, 1500, SHA1),
        ],
    }


class FfnShardIndexTests(unittest.TestCase):
    def test_overlay_preflight_verifies_the_same_parent_and_deployed_shards(self):
        from research_dev.scheduler.campaigns.burstgpt.preflight import (
            RigPreflightError, _ffn_shard_preflight,
        )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "FFN_SHARDS.json"
            path.write_text(json.dumps(_index()))
            args = SimpleNamespace(
                qwen_ffn_shards=None, gemma_ffn_shards=None,
                llama_ffn_shards=str(path) + "=/phone/shards", adb=Path("/adb"),
                adb_port=5037, phone_usb_serial="test-phone",
            )
            other = SimpleNamespace(artifact_sha256=OTHER)
            parent = SimpleNamespace(artifact_sha256=PARENT)
            output = SimpleNamespace(stdout=(
                SHA0[7:] + "  /phone/shards/HTP0.ffn.gguf\n"
                + SHA1[7:] + "  /phone/shards/HTP1.ffn.gguf\n"
            ))
            with patch("research_dev.scheduler.campaigns.burstgpt.preflight.subprocess.run",
                       return_value=output) as physical:
                indexes, checks = _ffn_shard_preflight(args, other, other, parent)
                self.assertEqual(set(indexes), {PARENT})
                self.assertEqual(len(checks), 1)
                from research_dev.scheduler.campaigns.burstgpt.runner import _ffn_shard_storage
                stored = _ffn_shard_storage(indexes)
                self.assertEqual([row.session_id for row in stored], ["HTP0", "HTP1"])
                self.assertEqual([row.layer_mask for row in stored], [0b11, 0b1100])
                self.assertEqual([row.maximum_columns for row in stored], [512, 768])
                self.assertTrue(all(row.parent_artifact_sha256 == PARENT for row in stored))
                physical.assert_called_once()
                with self.assertRaisesRegex(RigPreflightError, "parent differs"):
                    _ffn_shard_preflight(args, other, other, other)
                output.stdout = output.stdout.replace(SHA1[7:], SHA0[7:])
                with self.assertRaisesRegex(RigPreflightError, "differs from its index hash"):
                    _ffn_shard_preflight(args, other, other, parent)

    def test_resolves_smallest_covering_shard(self):
        index = FfnShardIndex.from_index(_index(), "/data/local/tmp/shards/qwen")
        self.assertEqual(index.parent_sha256, PARENT)
        record = index.resolve(PARENT, 0b000001, 256)
        self.assertIsNotNone(record)
        self.assertEqual(record.remote_path, "/data/local/tmp/shards/qwen/HTP0.ffn.gguf")
        self.assertEqual(record.shard_sha256, SHA0)
        # layer subset and smaller served suffix of the stored slice
        self.assertEqual(index.resolve(PARENT, 0b001000, 768).session_hint, "HTP1")
        # not covered: layers span both shards, too many columns, wrong parent
        self.assertIsNone(index.resolve(PARENT, 0b000110, 512))
        self.assertIsNone(index.resolve(PARENT, 0b000011, 640))
        self.assertIsNone(index.resolve(OTHER, 0b000011, 512))
        self.assertIsNone(resolve_ffn_shard({}, PARENT, 0b1, 512))
        self.assertEqual(
            remote_hash_entries({PARENT: index}),
            {
                "ffn-shard:" + SHA0: "/data/local/tmp/shards/qwen/HTP0.ffn.gguf",
                "ffn-shard:" + SHA1: "/data/local/tmp/shards/qwen/HTP1.ffn.gguf",
            },
        )
        verify_remote_hashes({PARENT: index}, {"ffn-shard:" + SHA0: SHA0, "ffn-shard:" + SHA1: SHA1})
        with self.assertRaisesRegex(FfnShardIndexError, "differs from its index hash"):
            verify_remote_hashes({PARENT: index}, {"ffn-shard:" + SHA0: SHA0, "ffn-shard:" + SHA1: SHA0})
        # only the records a launch opens are checked when given explicitly
        verify_remote_hashes({PARENT: index}, {"ffn-shard:" + SHA0: SHA0}, records=[index.records[0]])

    def test_load_rejects_invalid_indexes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "FFN_SHARDS.json"
            path.write_text(json.dumps(_index()))
            index = FfnShardIndex.load(path, "/data/local/tmp/shards/qwen")
            self.assertEqual(len(index.records), 2)
            with self.assertRaisesRegex(FfnShardIndexError, "phone path"):
                FfnShardIndex.load(path, "relative/dir")
            bad = _index()
            bad["shards"][1]["parent_sha256"] = OTHER
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(FfnShardIndexError, "invalid"):
                FfnShardIndex.load(path, "/data/local/tmp/shards/qwen")
            bad = _index()
            bad["schema"] = "other"
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(FfnShardIndexError, "schema"):
                FfnShardIndex.load(path, "/data/local/tmp/shards/qwen")
            with self.assertRaisesRegex(FfnShardIndexError, "cannot read"):
                FfnShardIndex.load(Path(root) / "missing.json", "/data/local/tmp/x")


class ManifestShardSubstitutionTests(unittest.TestCase):
    def _session(self, ffn_shards):
        session = object.__new__(DirectPhoneFfnSession)
        session.configuration = SimpleNamespace(
            multi_session_port_base=18000,
            multi_session_device_count=3,
            model_paths_by_artifact={PARENT: "/data/local/tmp/models/qwen.gguf"},
            ffn_shards_by_artifact=ffn_shards,
        )
        return session

    def _shard(self, session_id, mask, columns):
        return SimpleNamespace(
            session_id=session_id,
            artifact_sha256=PARENT,
            session_generation=1,
            endpoint=f"session://op15-phone/{session_id}",
            layer_mask=mask,
            maximum_columns=columns,
            resident_bytes=4096,
            resident_geometry_sha256=SHA0,
            operator_plan_sha256=SHA1,
        )

    def test_manifest_uses_covering_shards_and_fails_closed(self):
        index = FfnShardIndex.from_index(_index(), "/data/local/tmp/shards/qwen")
        covered = (
            self._shard("HTP0", 0b000011, 512),   # covered by HTP0.ffn.gguf
            self._shard("HTP1", 0b001100, 512),   # covered by HTP1.ffn.gguf (768 stored)
        )
        with_shards, sha_with = self._session({PARENT: index})._multi_session_manifest(covered)
        without, sha_without = self._session({})._multi_session_manifest(covered)
        rows_with = [row.split(",") for row in with_shards.split(";")]
        rows_without = [row.split(",") for row in without.split(";")]
        self.assertEqual(
            [row[3] for row in rows_with],
            [
                "/data/local/tmp/shards/qwen/HTP0.ffn.gguf",
                "/data/local/tmp/shards/qwen/HTP1.ffn.gguf",
            ],
        )
        self.assertEqual([row[3] for row in rows_without], ["/data/local/tmp/models/qwen.gguf"] * 2)
        # every other column (layers, columns, ports, identities, generation) is unchanged
        for left, right in zip(rows_with, rows_without):
            self.assertEqual(left[:3] + left[4:], right[:3] + right[4:])
        self.assertNotEqual(sha_with, sha_without)
        sources = self._session({PARENT: index})._phone_weight_sources(covered)
        self.assertEqual([row.weight_source for row in sources], ["ffn_shard"] * 2)
        self.assertEqual([row.index_sha256 for row in sources], [index.index_sha256] * 2)
        with self.assertRaisesRegex(
            PhysicalAdapterError, "does not cover the scheduled assignment"
        ):
            self._session({PARENT: index})._multi_session_manifest((
                self._shard("HTP2", 0b110000, 512),
            ))
        with self.assertRaisesRegex(
            PhysicalAdapterError, "does not cover the scheduled assignment"
        ):
            self._session({PARENT: index})._multi_session_manifest((
                self._shard("HTP1", 0b000011, 512),
            ))

    def test_manifest_rejects_unregistered_model(self):
        session = self._session({})
        session.configuration.model_paths_by_artifact = {}
        with self.assertRaises(PhysicalAdapterError):
            session._multi_session_manifest((self._shard("HTP0", 0b1, 512),))


if __name__ == "__main__":
    unittest.main()
