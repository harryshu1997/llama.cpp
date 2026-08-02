#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
BUILDER_PATH = HERE.parent / "build_cuda_monolithic_launch_v1.py"
PRODUCER_PATH = HERE.parent / "cuda_monolithic_v1.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = load_module("build_cuda_monolithic_launch_v1", BUILDER_PATH)
PRODUCER = load_module("cuda_monolithic_v1_for_launch_test", PRODUCER_PATH)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LaunchFixture:
    def __init__(self, test: unittest.TestCase):
        self.temporary = tempfile.TemporaryDirectory()
        test.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.launcher = self.bundle / "llama-layersplit"
        self.launcher.write_bytes(b"fake launcher\n")
        self.launcher.chmod(0o755)
        self.library = self.bundle / "libggml.so.0"
        self.library.write_bytes(b"fake library\n")
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"fake model payload\n")
        self.config = BUILDER.LaunchConfig(
            allowed_system_roots=BUILDER.ALLOWED_SYSTEM_ROOTS,
            bundle_root=self.bundle,
            components=(
                BUILDER.ComponentSpec(
                    "cuda-mono.bin",
                    self.launcher,
                    sha256(self.launcher),
                ),
                BUILDER.ComponentSpec(
                    "cuda-mono.lib",
                    self.library,
                    sha256(self.library),
                ),
            ),
            cwd=self.root,
            environment=(
                ("CUDA_VISIBLE_DEVICES", "0"),
                ("HOME", "/home/zhihao"),
                ("LAYERSPLIT_MEMORY_CERT", "1"),
                ("LAYERSPLIT_MODEL_SHA256", sha256(self.model)),
                ("LAYERSPLIT_PLACEMENT_CERT", "1"),
                ("LC_ALL", "C"),
                ("LD_LIBRARY_PATH", str(self.bundle)),
            ),
            expected_file_type=15,
            expected_n_embd=5120,
            launcher_component_id="cuda-mono.bin",
            model_bytes=self.model.stat().st_size,
            model_path=self.model,
            model_sha256=sha256(self.model),
            port=39124,
        )


class LaunchBuilderTests(unittest.TestCase):
    def test_builds_canonical_plan_accepted_by_producer(self):
        fixture = LaunchFixture(self)
        value = BUILDER.build_launch(fixture.config, 17)
        raw = BUILDER.canonical_bytes(value)
        output = fixture.root / "launch.json"
        BUILDER.write_exclusive(output, raw)
        loaded, loaded_raw = PRODUCER.load_launch(
            output,
            fixture.config.model_sha256,
        )
        self.assertEqual(loaded, value)
        self.assertEqual(loaded_raw, raw)
        self.assertEqual(value["expected_n_ctx_seq"], 512)
        self.assertEqual(
            value["allowed_system_roots"],
            list(BUILDER.ALLOWED_SYSTEM_ROOTS),
        )
        self.assertEqual(
            value["command"],
            [
                str(fixture.launcher),
                "-m",
                str(fixture.model),
                "--mode",
                "monov3",
                "--port",
                "39124",
                "--devices",
                "CUDA0",
                "--driver-batch",
                "8",
                "--driver-context",
                "512",
                "--driver-max-prefill",
                "8",
            ],
        )

    def test_bundle_digest_is_derived_from_hashed_components(self):
        fixture = LaunchFixture(self)
        value = BUILDER.build_launch(fixture.config, 17)
        self.assertEqual(
            value["bundle_sha256"],
            BUILDER.bundle_digest(
                value["required_components"],
                value["launcher_component_id"],
            ),
        )
        changed = copy.deepcopy(value["required_components"])
        changed[0]["sha256"] = "0" * 64
        self.assertNotEqual(
            value["bundle_sha256"],
            BUILDER.bundle_digest(changed, value["launcher_component_id"]),
        )

    def test_component_and_model_hash_mismatch_refuse(self):
        fixture = LaunchFixture(self)
        components = list(fixture.config.components)
        components[0] = BUILDER.ComponentSpec(
            components[0].component_id,
            components[0].path,
            "0" * 64,
        )
        config = copy.copy(fixture.config)
        object.__setattr__(config, "components", tuple(components))
        with self.assertRaisesRegex(BUILDER.BuildError, "E_FILE_SHA256"):
            BUILDER.build_launch(config, 17)

        fixture = LaunchFixture(self)
        config = copy.copy(fixture.config)
        object.__setattr__(config, "model_sha256", "0" * 64)
        environment = dict(config.environment)
        environment["LAYERSPLIT_MODEL_SHA256"] = "0" * 64
        object.__setattr__(config, "environment", tuple(environment.items()))
        with self.assertRaisesRegex(BUILDER.BuildError, "E_FILE_SHA256"):
            BUILDER.build_launch(config, 17)

    def test_reopen_detects_post_hash_mutation(self):
        fixture = LaunchFixture(self)

        def mutate() -> None:
            fixture.library.write_bytes(b"changed after hash\n")

        with self.assertRaisesRegex(BUILDER.BuildError, "E_REOPEN_MUTATION"):
            BUILDER.snapshot_file(
                fixture.library,
                sha256(fixture.library),
                after_hash=mutate,
            )

    def test_symlink_and_wrong_model_size_refuse(self):
        fixture = LaunchFixture(self)
        link = fixture.root / "model-link.gguf"
        link.symlink_to(fixture.model)
        with self.assertRaisesRegex(BUILDER.BuildError, "E_OPEN"):
            BUILDER.snapshot_file(link, sha256(fixture.model))

        config = copy.copy(fixture.config)
        object.__setattr__(config, "model_bytes", fixture.model.stat().st_size + 1)
        with self.assertRaisesRegex(BUILDER.BuildError, "E_FILE_BYTES"):
            BUILDER.build_launch(config, 17)

    def test_component_order_roots_epoch_and_output_are_fail_closed(self):
        fixture = LaunchFixture(self)
        config = copy.copy(fixture.config)
        object.__setattr__(
            config,
            "components",
            tuple(reversed(config.components)),
        )
        with self.assertRaisesRegex(BUILDER.BuildError, "E_COMPONENT_IDS"):
            BUILDER.build_launch(config, 17)

        config = copy.copy(fixture.config)
        object.__setattr__(config, "allowed_system_roots", ("/",))
        with self.assertRaisesRegex(BUILDER.BuildError, "E_SYSTEM_ROOTS"):
            BUILDER.build_launch(config, 17)

        with self.assertRaisesRegex(BUILDER.BuildError, "E_ROUTE_EPOCH"):
            BUILDER.build_launch(fixture.config, 0)

        output = fixture.root / "occupied.json"
        output.write_bytes(b"occupied")
        with self.assertRaisesRegex(BUILDER.BuildError, "E_OUTPUT"):
            BUILDER.write_exclusive(output, b"replacement")
        self.assertEqual(output.read_bytes(), b"occupied")

    def test_production_configuration_is_exact(self):
        config = BUILDER.production_config()
        self.assertEqual(config.port, 39124)
        self.assertEqual(config.model_path, BUILDER.MODEL_PATH)
        self.assertEqual(config.model_sha256, BUILDER.MODEL_SHA256)
        self.assertEqual(config.model_bytes, 9001752960)
        self.assertEqual(len(config.components), 7)
        self.assertEqual(
            [component.component_id for component in config.components],
            sorted(component.component_id for component in config.components),
        )
        self.assertEqual(
            dict(config.environment),
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "HOME": "/home/zhihao",
                "LAYERSPLIT_MEMORY_CERT": "1",
                "LAYERSPLIT_MODEL_SHA256": BUILDER.MODEL_SHA256,
                "LAYERSPLIT_PLACEMENT_CERT": "1",
                "LC_ALL": "C",
                "LD_LIBRARY_PATH": str(BUILDER.BUNDLE_ROOT),
            },
        )
        for component in config.components:
            self.assertEqual(len(component.sha256), 64)
            int(component.sha256, 16)

    def test_canonical_output_has_no_duplicate_or_float_values(self):
        fixture = LaunchFixture(self)
        raw = BUILDER.canonical_bytes(BUILDER.build_launch(fixture.config, 17))
        value = json.loads(raw)
        self.assertEqual(
            raw,
            (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        )
        self.assertNotIn(b"NaN", raw)
        self.assertNotIn(b"Infinity", raw)


if __name__ == "__main__":
    unittest.main()
