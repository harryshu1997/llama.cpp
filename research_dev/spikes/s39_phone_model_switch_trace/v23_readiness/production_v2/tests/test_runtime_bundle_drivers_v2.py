#!/usr/bin/env python3

import copy
import hashlib
import importlib.util
from pathlib import Path
import py_compile
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent
V1_DIR = V2_DIR.parent / "production_v1"
S39 = V2_DIR.parents[1]
sys.path.insert(0, str(V1_DIR))
sys.path.insert(0, str(V1_DIR / "tests"))

import driver_common_v1 as base
from test_readiness_drivers_v1 import (
    QueueRunner,
    ReadinessDriverTests,
    TickingClock,
)


def load_v2():
    namespace = {
        "BASE": base,
        "__file__": str(V2_DIR / "driver_common_v2.py"),
    }
    source = (V2_DIR / "driver_common_v2.py").read_text(encoding="ascii")
    exec(compile(source, namespace["__file__"], "exec"), namespace)
    return type("V2", (), namespace)


v2 = load_v2()
overlay_spec = importlib.util.spec_from_file_location(
    "runtime_bundle_overlay_v1",
    V2_DIR / "runtime_bundle_overlay_v1.py",
)
overlay = importlib.util.module_from_spec(overlay_spec)
overlay_spec.loader.exec_module(overlay)
prep_spec = importlib.util.spec_from_file_location(
    "prepare_zero_swap_v1",
    V2_DIR / "prepare_zero_swap_v1.py",
)
prep = importlib.util.module_from_spec(prep_spec)
prep_spec.loader.exec_module(prep)


class CounterClock:
    def __init__(self, start=0.0, step=0.01):
        self.value = start
        self.step = step

    def __call__(self):
        value = self.value
        self.value += self.step
        return value


class RuntimeBundleDriverTests(unittest.TestCase):
    def setUp(self):
        fixture = ReadinessDriverTests(
            "test_artifact_snapshot_is_exact_and_digest_bound"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        fixture.op15_worker = (
            "/data/local/tmp/s39-v23/op15-worker/llama-layersplit"
        )
        fixture.op12_worker = (
            "/data/local/tmp/s39-v23/op12-worker/llama-layersplit"
        )
        self.runtime_plan_path = fixture.root / "runtime-bundle-plan.json"
        roots = {
            "cuda_monolithic": "/home/zhihao/s40-runtime/cuda-monolithic",
            "cuda_route": "/home/zhihao/s40-runtime/cuda-route",
            "op12_stagenet": "/data/local/tmp/s39-v23/op12-worker",
            "op15_direct_relay": "/data/local/tmp/s39-v23/op15-relay",
            "op15_stagenet": "/data/local/tmp/s39-v23/op15-worker",
        }
        endpoints = {
            "cuda_monolithic": ("cuda", "cuda_monolithic"),
            "cuda_route": ("cuda", "cuda_route"),
            "op12_stagenet": ("op12", "stagenet_worker"),
            "op15_direct_relay": ("op15", "direct_relay"),
            "op15_stagenet": ("op15", "stagenet_worker"),
        }
        components = []
        bundles = []
        self.runtime_stats = {}
        for bundle_index, bundle_id in enumerate(sorted(roots)):
            endpoint, process_role = endpoints[bundle_id]
            component_ids = []
            for kind_index, (suffix, role) in enumerate(
                (("launcher", "executable"), ("runtime", "shared_library"))
            ):
                component_id = f"{bundle_id}.{suffix}"
                component_ids.append(component_id)
                if suffix == "launcher":
                    filename = (
                        "llama-layersplit"
                        if bundle_id in ("op12_stagenet", "op15_stagenet")
                        else (
                            "stage-direct-relay"
                            if bundle_id == "op15_direct_relay"
                            else "llama-layersplit"
                        )
                    )
                else:
                    filename = "libllama.so"
                size = 1_000_000 + 10 * bundle_index + kind_index
                checksum = hashlib.sha256(component_id.encode("ascii")).hexdigest()
                components.append({
                    "bundle_id": bundle_id,
                    "bytes": size,
                    "component_id": component_id,
                    "endpoint": endpoint,
                    "path": roots[bundle_id] + "/" + filename,
                    "role": role,
                    "sha256": checksum,
                })
                self.runtime_stats[component_id] = {
                    "ctime_ns": 1_800_000_000_000_000_000 + len(self.runtime_stats),
                    "device_id": 100 + len(self.runtime_stats),
                    "inode": 200 + len(self.runtime_stats),
                    "mode": stat.S_IFREG | 0o755,
                    "mtime_ns": 1_700_000_000_000_000_000 + len(self.runtime_stats),
                    "size": size,
                }
            bundles.append({
                "bundle_id": bundle_id,
                "endpoint": endpoint,
                "launcher_component_id": component_ids[0],
                "process_role": process_role,
                "required_component_ids": component_ids,
            })
        contract_raw = fixture.contract_path.read_bytes()
        candidate_raw = fixture.candidate_path.read_bytes()
        self.plan = {
            "bundle_roots": roots,
            "bundles": bundles,
            "candidate_sha256": hashlib.sha256(candidate_raw).hexdigest(),
            "components": components,
            "contract_sha256": hashlib.sha256(contract_raw).hexdigest(),
            "model_id": base.MODEL_ID,
            "phase": base.PHASE,
            "schema": "s39-cp0-r1-runtime-bundle-plan-v1",
        }
        self.write_plan()

    def write_plan(self):
        self.runtime_plan_path.write_bytes(base.canonical_bytes(self.plan))

    def component(self, component_id):
        return next(
            value
            for value in self.plan["components"]
            if value["component_id"] == component_id
        )

    def component_outputs(self, include_digest):
        values = []
        for component in self.plan["components"]:
            record = self.runtime_stats[component["component_id"]]
            checksum = component["sha256"] if include_digest else None
            if component["endpoint"] == "cuda":
                values.append(self.fixture.remote_record(record, checksum))
            else:
                values.append(self.fixture.android_block(record, checksum))
        return values

    def inventory_outputs(self, extra=None):
        extra = extra or {}
        values = []
        components = {
            bundle["bundle_id"]: sorted(
                self.component(component_id)["path"]
                for component_id in bundle["required_component_ids"]
            )
            for bundle in self.plan["bundles"]
        }
        for bundle in self.plan["bundles"]:
            files = components[bundle["bundle_id"]] + extra.get(
                bundle["bundle_id"],
                [],
            )
            files = sorted(files)
            if bundle["endpoint"] == "cuda":
                values.append(
                    base.canonical_bytes({"bad": [], "files": files})
                )
            else:
                values.append(
                    "".join(f"FILE={path}\n" for path in files).encode("ascii")
                )
        return values

    def artifact_outputs(self, extra=None):
        return (
            self.fixture.artifact_outputs()
            + self.component_outputs(True)
            + self.inventory_outputs(extra)
        )

    def fresh_outputs(self, extra=None):
        return (
            self.fixture.fresh_outputs()
            + self.component_outputs(False)
            + self.inventory_outputs(extra)
        )

    def run_artifact(self, outputs=None):
        runner = QueueRunner(outputs or self.artifact_outputs())
        value = v2.artifact_driver(
            contract_path=self.fixture.contract_path,
            candidate_path=self.fixture.candidate_path,
            runtime_bundle_plan_path=self.runtime_plan_path,
            pre_dir=self.fixture.pre,
            output_dir=self.fixture.artifact_out,
            phase_id=self.fixture.phase_id,
            op15_worker=self.fixture.op15_worker,
            op12_worker=self.fixture.op12_worker,
            timeout=30,
            runner=runner,
            now_ns=TickingClock(100, 200, 300, 400),
        )
        return value, runner

    def run_fresh(self, outputs=None):
        runner = QueueRunner(outputs or self.fresh_outputs())
        value = v2.fresh_driver(
            contract_path=self.fixture.contract_path,
            candidate_path=self.fixture.candidate_path,
            runtime_bundle_plan_path=self.runtime_plan_path,
            pre_dir=self.fixture.pre,
            output_dir=self.fixture.fresh_out,
            phase_id=self.fixture.phase_id,
            op15_worker=self.fixture.op15_worker,
            op12_worker=self.fixture.op12_worker,
            timeout=30,
            runner=runner,
            now_ns=CounterClock(start=500, step=1),
        )
        return value, runner

    def test_artifact_and_fresh_preserve_base_and_bind_bundle(self):
        snapshot, artifact_runner = self.run_artifact()
        self.assertEqual(
            snapshot["schema"],
            "s39-cp0-r1-runtime-bundle-snapshot-v1",
        )
        self.assertTrue(
            (self.fixture.artifact_out / "artifact_snapshot.json").is_file()
        )
        self.assertEqual(len(snapshot["runtime_bundles"]), 5)
        self.assertEqual(
            len(artifact_runner.calls),
            5 + len(self.plan["components"]) + 5,
        )
        (lock, fresh), fresh_runner = self.run_fresh()
        self.assertEqual(
            lock["schema"],
            "s39-cp0-r1-runtime-bundle-readiness-lock-v1",
        )
        self.assertEqual(
            fresh["schema"],
            "s39-cp0-r1-runtime-bundle-fresh-v1",
        )
        self.assertTrue(
            (self.fixture.fresh_out / "fresh_snapshot.json").is_file()
        )
        self.assertEqual(
            len(fresh_runner.calls),
            8 + len(self.plan["components"]) + 5,
        )

    def test_omitted_library_is_exposed_by_directory_inventory(self):
        bundle = self.plan["bundles"][0]
        omitted = bundle["required_component_ids"].pop()
        self.write_plan()
        outputs = self.fixture.artifact_outputs()
        outputs += [
            value
            for component, value in zip(
                [
                    item
                    for item in self.plan["components"]
                    if item["component_id"] != omitted
                ],
                self.component_outputs(True),
            )
            if component["component_id"] != omitted
        ]
        with self.assertRaisesRegex(base.DriverError, "E_BUNDLE"):
            self.run_artifact(outputs)

    def test_substituted_component_digest_is_rejected(self):
        outputs = self.artifact_outputs()
        offset = 5
        component = self.plan["components"][0]
        record = self.runtime_stats[component["component_id"]]
        outputs[offset] = self.fixture.remote_record(record, "0" * 64)
        with self.assertRaisesRegex(base.DriverError, "E_SHA256"):
            self.run_artifact(outputs)

    def test_extra_directory_file_is_rejected(self):
        bundle = self.plan["bundles"][0]
        outputs = self.artifact_outputs({
            bundle["bundle_id"]: [
                self.plan["bundle_roots"][bundle["bundle_id"]] + "/unbound.so"
            ]
        })
        with self.assertRaisesRegex(base.DriverError, "DIRECTORY_SET"):
            self.run_artifact(outputs)

    def test_fresh_component_stat_drift_is_rejected(self):
        self.run_artifact()
        outputs = self.fresh_outputs()
        changed = copy.deepcopy(
            self.runtime_stats[self.plan["components"][0]["component_id"]]
        )
        changed["ctime_ns"] += 1
        outputs[8] = self.fixture.remote_record(changed)
        with self.assertRaisesRegex(base.DriverError, "COMPONENT_CHANGED"):
            self.run_fresh(outputs)

    def test_bundle_roots_must_be_isolated(self):
        self.plan["bundle_roots"]["cuda_route"] = (
            self.plan["bundle_roots"]["cuda_monolithic"] + "/nested"
        )
        for component in self.plan["components"]:
            if component["bundle_id"] == "cuda_route":
                component["path"] = (
                    self.plan["bundle_roots"]["cuda_route"]
                    + "/"
                    + Path(component["path"]).name
                )
        self.write_plan()
        with self.assertRaisesRegex(base.DriverError, "ROOT_OVERLAP"):
            self.run_artifact([])

    def test_cached_support_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for name in (
                "artifact_snapshot_driver_v2.py",
                "source_entry_v2.py",
                "driver_common_v2.py",
            ):
                paths[name] = root / name
                shutil.copyfile(V2_DIR / name, paths[name])
            paths["driver_common_v1.py"] = root / "driver_common_v1.py"
            shutil.copyfile(V1_DIR / "driver_common_v1.py", paths["driver_common_v1.py"])
            marker = root / "cached.py"
            marker.write_text(
                "print('CACHED_SUPPORT_USED')\n"
                "def artifact_main():\n"
                "    return 73\n",
                encoding="ascii",
            )
            for name in ("driver_common_v1.py", "driver_common_v2.py"):
                cache = Path(importlib.util.cache_from_source(str(paths[name])))
                cache.parent.mkdir(parents=True, exist_ok=True)
                py_compile.compile(
                    str(marker),
                    cfile=str(cache),
                    doraise=True,
                    invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
                )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(paths["artifact_snapshot_driver_v2.py"]),
                    "--entry-support",
                    str(paths["source_entry_v2.py"]),
                    "--base-support",
                    str(paths["driver_common_v1.py"]),
                    "--support",
                    str(paths["driver_common_v2.py"]),
                    "--help",
                ],
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertNotIn("CACHED_SUPPORT_USED", completed.stdout)
            self.assertIn("--runtime-bundle-plan", completed.stdout)

    def test_runtime_loaded_component_set_is_exact(self):
        bundle = self.plan["bundles"][0]
        expected = bundle["required_component_ids"]
        overlay.validate_loaded_repo_components(
            expected,
            bundle,
            base,
            bundle["bundle_id"],
        )
        for value in (
            expected[:-1],
            expected + ["unbound.component"],
            [expected[0], "substituted.component"],
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(base.DriverError, "LOADED_COMPONENTS"):
                    overlay.validate_loaded_repo_components(
                        value,
                        bundle,
                        base,
                        bundle["bundle_id"],
                    )

    def test_zero_swap_prepare_reboots_both_before_phase(self):
        def status(boot_id):
            return (
                f"BOOT_ID={boot_id}\n"
                "BOOT_COMPLETED=1\n"
                "MEM_AVAILABLE_KB=2000000\n"
                "SWAP_TOTAL_KB=100\n"
                "SWAP_FREE_KB=100\n"
                "THERMAL_STATUS=0\n"
            ).encode("ascii")

        outputs = []
        identities = (
            (
                "11111111-1111-4111-8111-111111111111",
                "21111111-1111-4111-8111-111111111111",
            ),
            (
                "31111111-1111-4111-8111-111111111111",
                "41111111-1111-4111-8111-111111111111",
            ),
        )
        for before, after in identities:
            outputs.extend([
                (before + "\n").encode("ascii"),
                b"",
                b"",
                status(after),
                status(after),
            ])
        output = self.fixture.root / "zero-swap"
        result = prep.prepare(
            contract_path=self.fixture.contract_path,
            output_dir=output,
            confirmation=prep.CONFIRM,
            timeout_seconds=60,
            poll_seconds=0.01,
            stable_samples=2,
            runner=QueueRunner(outputs),
            monotonic=CounterClock(),
            clock_ns=iter(range(100, 1000)).__next__,
            sleep=lambda _: None,
        )
        self.assertEqual(result["status"], "ZERO_SWAP_IDLE_PREP_PASS")
        self.assertEqual(set(result["phones"]), {"op12", "op15"})
        self.assertTrue((output / "zero_swap_prepare.json").is_file())

    def test_zero_swap_prepare_requires_explicit_confirmation(self):
        with self.assertRaisesRegex(prep.PrepareError, "E_CONFIRM"):
            prep.prepare(
                contract_path=self.fixture.contract_path,
                output_dir=self.fixture.root / "must-not-exist",
                confirmation="NO",
                timeout_seconds=60,
                poll_seconds=1,
                stable_samples=2,
                runner=QueueRunner([]),
            )


if __name__ == "__main__":
    unittest.main()
