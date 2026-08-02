#!/usr/bin/env python3

import hashlib
from pathlib import Path
import py_compile
import shutil
import stat
import subprocess
import sys
import tempfile
import types
import unittest


HERE = Path(__file__).resolve().parent
PRODUCTION = HERE.parent
V23 = PRODUCTION.parent
V1 = V23 / "production_v1"
V1_TESTS = V1 / "tests"
V23_TESTS = V23 / "tests"
S39 = V23.parent

sys.path.insert(0, str(V1))
sys.path.insert(0, str(V1_TESTS))
sys.path.insert(0, str(V23))
sys.path.insert(0, str(V23_TESTS))
sys.path.insert(0, str(S39))

import driver_common_v1 as base
import test_acquire_a_only_v23 as outer_tests
import test_readiness_drivers_v1 as v1_tests


QueueRunner = v1_tests.QueueRunner
TickingClock = v1_tests.TickingClock


class StepClock:
    def __init__(self, start=500, step=1):
        self.value = start
        self.step = step

    def __call__(self):
        value = self.value
        self.value += self.step
        return value


def load_v2():
    module = types.ModuleType("runtime_bundle_driver_v2_test")
    module.__file__ = str(PRODUCTION / "driver_common_v2.py")
    module.__dict__["BASE"] = base
    source = Path(module.__file__).read_text(encoding="ascii")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


v2 = load_v2()


def load_overlay():
    module = types.ModuleType("runtime_bundle_overlay_v1_test")
    module.__file__ = str(PRODUCTION / "runtime_bundle_overlay_v1.py")
    source = Path(module.__file__).read_text(encoding="ascii")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


overlay = load_overlay()


class RuntimeBundleFixture:
    def __init__(self, owner):
        self.owner = owner
        self.v1 = v1_tests.ReadinessDriverTests(
            "test_artifact_snapshot_is_exact_and_digest_bound"
        )
        self.v1.setUp()
        owner.addCleanup(self.v1.doCleanups)
        self.root = self.v1.root
        self.pre = self.v1.pre
        self.artifact_out = self.v1.artifact_out
        self.fresh_out = self.v1.fresh_out
        self.phase_id = self.v1.phase_id

        self.roots = {
            "cuda_monolithic": "/runtime/cuda-monolithic",
            "cuda_route": "/runtime/cuda-route",
            "op12_stagenet": "/runtime/op12-stagenet",
            "op15_direct_relay": "/runtime/op15-relay",
            "op15_stagenet": "/runtime/op15-stagenet",
        }
        self.v1.op15_worker = self.roots["op15_stagenet"] + "/worker"
        self.v1.op12_worker = self.roots["op12_stagenet"] + "/worker"
        self.components = self._make_components()
        self.bundles = self._make_bundles()
        contract_raw = self.v1.contract_path.read_bytes()
        candidate_raw = self.v1.candidate_path.read_bytes()
        self.plan = {
            "bundle_roots": self.roots,
            "bundles": self.bundles,
            "candidate_sha256": hashlib.sha256(candidate_raw).hexdigest(),
            "components": self.components,
            "contract_sha256": hashlib.sha256(contract_raw).hexdigest(),
            "model_id": v2.BASE.MODEL_ID,
            "phase": v2.BASE.PHASE,
            "schema": "s39-cp0-r1-runtime-bundle-plan-v1",
        }
        self.plan_path = self.root / "runtime-plan.json"
        self.write_plan()
        self.component_stats = {
            component["component_id"]: {
                "ctime_ns": 1_800_000_000_000_000_000 + index,
                "device_id": 1000 + index,
                "inode": 2000 + index,
                "mode": stat.S_IFREG | 0o755,
                "mtime_ns": 1_700_000_000_000_000_000 + index,
                "size": component["bytes"],
            }
            for index, component in enumerate(self.components)
        }

    def _make_components(self):
        values = []
        specs = (
            ("cuda_monolithic", "cuda"),
            ("cuda_route", "cuda"),
            ("op12_stagenet", "op12"),
            ("op15_direct_relay", "op15"),
            ("op15_stagenet", "op15"),
        )
        for index, (bundle_id, endpoint) in enumerate(specs):
            root = self.roots[bundle_id]
            launcher_path = root + "/launcher"
            if bundle_id == "op12_stagenet":
                launcher_path = self.v1.op12_worker
            elif bundle_id == "op15_stagenet":
                launcher_path = self.v1.op15_worker
            values.extend(
                [
                    {
                        "bundle_id": bundle_id,
                        "bytes": 10000 + index,
                        "component_id": bundle_id + ".launcher",
                        "endpoint": endpoint,
                        "path": launcher_path,
                        "role": "executable",
                        "sha256": f"{index + 1:x}" * 64,
                    },
                    {
                        "bundle_id": bundle_id,
                        "bytes": 20000 + index,
                        "component_id": bundle_id + ".runtime",
                        "endpoint": endpoint,
                        "path": root + "/runtime.so",
                        "role": "shared_library",
                        "sha256": f"{index + 6:x}" * 64,
                    },
                ]
            )
        return sorted(values, key=lambda value: value["component_id"])

    def _make_bundles(self):
        result = []
        for bundle_id in sorted(self.roots):
            endpoint, process_role = v2.REQUIRED_BUNDLES[bundle_id]
            required = sorted(
                component["component_id"]
                for component in self.components
                if component["bundle_id"] == bundle_id
            )
            result.append(
                {
                    "bundle_id": bundle_id,
                    "endpoint": endpoint,
                    "launcher_component_id": bundle_id + ".launcher",
                    "process_role": process_role,
                    "required_component_ids": required,
                }
            )
        return result

    def write_plan(self):
        self.plan_path.write_bytes(base.canonical_bytes(self.plan))

    def component_outputs(self, include_digest):
        outputs = []
        for component in self.components:
            record = self.component_stats[component["component_id"]]
            checksum = component["sha256"] if include_digest else None
            if component["endpoint"] == "cuda":
                outputs.append(self.v1.remote_record(record, checksum))
            else:
                outputs.append(self.v1.android_block(record, checksum))
        return outputs

    def inventory_outputs(self, extra_by_bundle=None):
        extra_by_bundle = extra_by_bundle or {}
        outputs = []
        by_id = {
            component["component_id"]: component
            for component in self.components
        }
        for bundle in self.bundles:
            paths = sorted(
                [
                    by_id[component_id]["path"]
                    for component_id in bundle["required_component_ids"]
                ]
                + list(extra_by_bundle.get(bundle["bundle_id"], []))
            )
            if bundle["endpoint"] == "cuda":
                outputs.append(base.canonical_bytes({"bad": [], "files": paths}))
            else:
                outputs.append(
                    ("".join(f"FILE={path}\n" for path in paths)).encode("ascii")
                )
        return outputs

    def run_artifact(self):
        outputs = (
            self.v1.artifact_outputs()
            + self.component_outputs(True)
            + self.inventory_outputs()
        )
        return v2.artifact_driver(
            contract_path=self.v1.contract_path,
            candidate_path=self.v1.candidate_path,
            runtime_bundle_plan_path=self.plan_path,
            pre_dir=self.pre,
            output_dir=self.artifact_out,
            phase_id=self.phase_id,
            op15_worker=self.v1.op15_worker,
            op12_worker=self.v1.op12_worker,
            timeout=30,
            runner=QueueRunner(outputs),
            now_ns=TickingClock(100, 200, 300, 400),
        )

    def fresh_outputs(self):
        return (
            self.v1.fresh_outputs()
            + self.component_outputs(False)
            + self.inventory_outputs()
        )

    def run_fresh(self, clock):
        return v2.fresh_driver(
            contract_path=self.v1.contract_path,
            candidate_path=self.v1.candidate_path,
            runtime_bundle_plan_path=self.plan_path,
            pre_dir=self.pre,
            output_dir=self.fresh_out,
            phase_id=self.phase_id,
            op15_worker=self.v1.op15_worker,
            op12_worker=self.v1.op12_worker,
            timeout=30,
            runner=QueueRunner(self.fresh_outputs()),
            now_ns=clock,
        )

    def overlay_args(self):
        runtime_snapshot = self.run_artifact()
        _, runtime_fresh = self.run_fresh(StepClock())
        base_artifact = self.artifact_out / "artifact_snapshot.json"
        base_lock = self.fresh_out / "readiness_lock.json"
        base_fresh = self.fresh_out / "fresh_snapshot.json"
        runtime_snapshot_path = (
            self.artifact_out / "runtime_bundle_snapshot.json"
        )
        runtime_lock = (
            self.fresh_out / "runtime_bundle_readiness_lock.json"
        )
        runtime_fresh_path = self.fresh_out / "runtime_bundle_fresh.json"
        base_fresh_value, _ = base.read_canonical(base_fresh)

        base_runtime = {
            "executors": [
                {
                    "executor_id": "GPU",
                    "host_boot_id": base_fresh_value["cuda"]["host_boot_id"],
                },
                {
                    "boot_id": base_fresh_value["phones"]["op12"]["boot_id"],
                    "executor_id": "PHONE_OP12",
                    "worker_executable_path": self.v1.op12_worker,
                    "worker_pid": 1200,
                    "worker_start_ticks": 12000,
                },
                {
                    "boot_id": base_fresh_value["phones"]["op15"]["boot_id"],
                    "executor_id": "PHONE_OP15",
                    "worker_executable_path": self.v1.op15_worker,
                    "worker_pid": 1500,
                    "worker_start_ticks": 15000,
                },
            ]
        }
        base_runtime_path = self.root / "base-runtime.json"
        base_runtime_raw = base.canonical_bytes(base_runtime)
        base_runtime_path.write_bytes(base_runtime_raw)

        role_sha256s = {
            role: hashlib.sha256(role.encode("ascii")).hexdigest()
            for role in overlay.EVIDENCE_ROLES.values()
        }
        manifest = {
            "acquisition_started_ns": 600,
            "artifacts": [
                {"role": role, "sha256": role_sha256s[role]}
                for role in sorted(role_sha256s)
            ],
            "phase": "A_ONLY",
            "phase_closed_ns": 800,
            "phase_id": self.phase_id,
        }
        manifest_path = self.root / "manifest.json"
        manifest_path.write_bytes(base.canonical_bytes(manifest))

        components = {
            value["component_id"]: value
            for value in self.components
        }
        bundle_snapshots = {
            value["bundle_id"]: value
            for value in runtime_snapshot["runtime_bundles"]
        }
        process_rows = []
        for bundle in self.bundles:
            bundle_id = bundle["bundle_id"]
            endpoint = bundle["endpoint"]
            if endpoint == "cuda":
                boot_id = base_fresh_value["cuda"]["host_boot_id"]
                pid = 4000 + len(process_rows)
                start_ticks = 40000 + len(process_rows)
                dependency_path = "/usr/lib/libc.so"
            else:
                boot_id = base_fresh_value["phones"][endpoint]["boot_id"]
                pid = 1200 if endpoint == "op12" else 1500
                start_ticks = 12000 if endpoint == "op12" else 15000
                dependency_path = "/vendor/lib64/libc.so"
            launcher = components[bundle["launcher_component_id"]]
            role = overlay.EVIDENCE_ROLES[bundle_id]
            process_rows.append(
                {
                    "boot_id": boot_id,
                    "bundle_id": bundle_id,
                    "bundle_sha256": bundle_snapshots[bundle_id][
                        "bundle_sha256"
                    ],
                    "endpoint": endpoint,
                    "evidence_role": role,
                    "evidence_sha256": role_sha256s[role],
                    "identity_probe_sha256": hashlib.sha256(
                        f"{bundle_id}-probe".encode("ascii")
                    ).hexdigest(),
                    "launcher_component_id": bundle[
                        "launcher_component_id"
                    ],
                    "launcher_path": launcher["path"],
                    "loaded_repo_component_ids": bundle[
                        "required_component_ids"
                    ],
                    "observed_ns": 650,
                    "pid": pid,
                    "start_ticks": start_ticks,
                    "system_dependencies": [
                        {
                            "build_id": None,
                            "ctime_ns": 10,
                            "device_id": 11,
                            "inode": 12,
                            "mode": stat.S_IFREG | 0o644,
                            "mtime_ns": 13,
                            "path": dependency_path,
                            "size": 14,
                        }
                    ],
                }
            )
        identity = {
            "base_runtime_identity_sha256": hashlib.sha256(
                base_runtime_raw
            ).hexdigest(),
            "completed_ns": 700,
            "phase": "A_ONLY",
            "phase_id": self.phase_id,
            "processes": process_rows,
            "runtime_bundle_fresh_sha256": hashlib.sha256(
                runtime_fresh_path.read_bytes()
            ).hexdigest(),
            "runtime_bundle_plan_sha256": hashlib.sha256(
                self.plan_path.read_bytes()
            ).hexdigest(),
            "runtime_bundle_snapshot_sha256": hashlib.sha256(
                runtime_snapshot_path.read_bytes()
            ).hexdigest(),
            "schema": "s39-cp0-r1-runtime-bundle-runtime-identity-v1",
            "started_ns": 610,
        }
        identity_path = self.root / "runtime-bundle-identity.json"
        identity_path.write_bytes(base.canonical_bytes(identity))

        argv = [
            "--base-support",
            str((V1 / "driver_common_v1.py").resolve()),
            "--driver-support",
            str((PRODUCTION / "driver_common_v2.py").resolve()),
            "--contract",
            str(self.v1.contract_path.resolve()),
            "--candidate",
            str(self.v1.candidate_path.resolve()),
            "--runtime-bundle-plan",
            str(self.plan_path.resolve()),
            "--manifest",
            str(manifest_path.resolve()),
            "--base-artifact-snapshot",
            str(base_artifact.resolve()),
            "--base-readiness-lock",
            str(base_lock.resolve()),
            "--base-fresh-snapshot",
            str(base_fresh.resolve()),
            "--base-runtime-identity",
            str(base_runtime_path.resolve()),
            "--runtime-bundle-snapshot",
            str(runtime_snapshot_path.resolve()),
            "--runtime-bundle-readiness-lock",
            str(runtime_lock.resolve()),
            "--runtime-bundle-fresh",
            str(runtime_fresh_path.resolve()),
            "--runtime-bundle-identity",
            str(identity_path.resolve()),
        ]
        return overlay.parse_args(argv)


class RuntimeBundleRegressionTests(unittest.TestCase):
    def outer_fixture(self):
        fixture = outer_tests.AcquireAOnlyV23Tests(
            "test_three_stage_order_and_independent_verdict"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_unchecked_stale_pyc_cannot_replace_captured_support_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = root / "artifact_snapshot_driver_v2.py"
            source_entry = root / "source_entry_v2.py"
            base_support = root / "driver_common_v1.py"
            support = root / "driver_common_v2.py"
            for source, target in (
                (PRODUCTION / "artifact_snapshot_driver_v2.py", entry),
                (PRODUCTION / "source_entry_v2.py", source_entry),
                (V1 / "driver_common_v1.py", base_support),
                (PRODUCTION / "driver_common_v2.py", support),
            ):
                shutil.copyfile(source, target)
            entry.chmod(0o755)

            genuine = support.read_bytes()
            support.write_text(
                "raise SystemExit('STALE_PYC_EXECUTED')\n",
                encoding="ascii",
            )
            py_compile.compile(
                str(support),
                invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
                doraise=True,
            )
            support.write_bytes(genuine)

            completed = subprocess.run(
                [
                    str(entry),
                    "--entry-support",
                    str(source_entry),
                    "--base-support",
                    str(base_support),
                    "--support",
                    str(support),
                    "--help",
                ],
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
            self.assertIn("--runtime-bundle-plan", completed.stdout)
            self.assertNotIn(
                "STALE_PYC_EXECUTED",
                completed.stdout + completed.stderr,
            )

    def test_launcher_only_stagenet_bundle_is_rejected(self):
        fixture = RuntimeBundleFixture(self)
        target = next(
            bundle
            for bundle in fixture.plan["bundles"]
            if bundle["bundle_id"] == "op15_stagenet"
        )
        target["required_component_ids"] = [target["launcher_component_id"]]
        fixture.plan["components"] = [
            component
            for component in fixture.plan["components"]
            if component["component_id"] != "op15_stagenet.runtime"
        ]
        fixture.write_plan()
        with self.assertRaises(v2.BASE.DriverError):
            v2.load_runtime_plan(
                fixture.plan_path,
                fixture.v1.contract_path.read_bytes(),
                fixture.v1.candidate_path.read_bytes(),
                fixture.v1.op15_worker,
                fixture.v1.op12_worker,
            )

    def test_omitted_runtime_dependency_is_rejected_by_inventory(self):
        fixture = RuntimeBundleFixture(self)
        omitted = "op15_stagenet.runtime"
        target = next(
            bundle
            for bundle in fixture.bundles
            if bundle["bundle_id"] == "op15_stagenet"
        )
        target["required_component_ids"].remove(omitted)
        fixture.components = [
            component
            for component in fixture.components
            if component["component_id"] != omitted
        ]
        fixture.plan["components"] = fixture.components
        fixture.plan["bundles"] = fixture.bundles
        fixture.write_plan()
        outputs = (
            fixture.v1.artifact_outputs()
            + fixture.component_outputs(True)
            + fixture.inventory_outputs(
                {
                    "op15_stagenet": [
                        fixture.roots["op15_stagenet"] + "/runtime.so"
                    ]
                }
            )
        )
        with self.assertRaises(v2.BASE.DriverError):
            v2.artifact_driver(
                contract_path=fixture.v1.contract_path,
                candidate_path=fixture.v1.candidate_path,
                runtime_bundle_plan_path=fixture.plan_path,
                pre_dir=fixture.pre,
                output_dir=fixture.artifact_out,
                phase_id=fixture.phase_id,
                op15_worker=fixture.v1.op15_worker,
                op12_worker=fixture.v1.op12_worker,
                timeout=30,
                runner=QueueRunner(outputs),
                now_ns=TickingClock(100, 200, 300, 400),
            )

    def test_deleted_runtime_dependency_is_rejected_before_fresh_publish(self):
        fixture = RuntimeBundleFixture(self)
        fixture.run_artifact()
        snapshot_path = fixture.artifact_out / "runtime_bundle_snapshot.json"
        snapshot, _ = base.read_canonical(snapshot_path)
        snapshot["runtime_components"] = snapshot["runtime_components"][:-1]
        snapshot_path.write_bytes(base.canonical_bytes(snapshot))
        with self.assertRaisesRegex(
            v2.BASE.DriverError,
            "E_RUNTIME_COMPONENT_COUNT",
        ):
            fixture.run_fresh(
                TickingClock(500, 510, 520, 530, 540, 550)
            )
        self.assertFalse(
            (fixture.fresh_out / "runtime_bundle_fresh.json").exists()
        )

    def test_whole_fresh_interval_over_contract_limit_is_rejected(self):
        fixture = RuntimeBundleFixture(self)
        fixture.run_artifact()
        maximum = fixture.v1.contract["readiness_v2_3"][
            "fresh_snapshot_maximum_age_ns"
        ]
        with self.assertRaises(v2.BASE.DriverError):
            fixture.run_fresh(
                StepClock(step=maximum + 1)
            )
        self.assertFalse(
            (fixture.fresh_out / "runtime_bundle_fresh.json").exists()
        )

    def test_each_fresh_probe_carries_bounded_capture_interval(self):
        fixture = RuntimeBundleFixture(self)
        fixture.run_artifact()
        lock, fresh = fixture.run_fresh(
            StepClock()
        )
        maximum = fixture.v1.contract["readiness_v2_3"][
            "fresh_snapshot_maximum_age_ns"
        ]
        self.assertLessEqual(
            fresh["completed_ns"] - lock["event_ns"],
            maximum,
        )
        intervals = fresh.get("probe_intervals")
        self.assertIsInstance(intervals, list)
        self.assertEqual(
            len(intervals),
            len(fixture.v1.fresh_outputs())
            + len(fixture.components)
            + len(fixture.bundles),
            "the ledger must cover BASE and V2 probes",
        )
        previous_completed = lock["event_ns"]
        for index, record in enumerate(intervals):
            with self.subTest(index=index):
                started = record["probe_started_ns"]
                completed = record["probe_completed_ns"]
                self.assertGreaterEqual(started, previous_completed)
                self.assertLessEqual(started, completed)
                self.assertLessEqual(completed - started, maximum)
                self.assertLessEqual(fresh["completed_ns"] - completed, maximum)
                previous_completed = completed
        self.assertLessEqual(previous_completed, fresh["completed_ns"])

    def test_exit_overlay_reopens_every_runtime_bundle_root(self):
        fixture = RuntimeBundleFixture(self)
        args = fixture.overlay_args()
        result = overlay.validate(args)
        self.assertEqual(result["status"], "RUNTIME_BUNDLE_PROVENANCE_PASS")

        paths = (
            args.runtime_bundle_plan,
            args.runtime_bundle_snapshot,
            args.runtime_bundle_readiness_lock,
            args.runtime_bundle_fresh,
            args.runtime_bundle_identity,
        )
        for path in paths:
            original = path.read_bytes()
            with self.subTest(path=path.name, mutation="deleted"):
                path.unlink()
                with self.assertRaises((OSError, ValueError)):
                    overlay.validate(args)
                path.write_bytes(original)
            with self.subTest(path=path.name, mutation="changed"):
                value, _ = base.read_canonical(path)
                if (
                    value.get("schema")
                    == "s39-cp0-r1-runtime-bundle-runtime-identity-v1"
                ):
                    value["runtime_bundle_fresh_sha256"] = "0" * 64
                elif "completed_ns" in value:
                    value["completed_ns"] += 1
                elif "event_ns" in value:
                    value["event_ns"] += 1
                elif "components" in value:
                    value["components"][0]["bytes"] += 1
                else:
                    self.fail(f"no mutation target for {path}")
                path.write_bytes(base.canonical_bytes(value))
                with self.assertRaises(ValueError):
                    overlay.validate(args)
                path.write_bytes(original)

    def test_exit_overlay_revalidates_probe_intervals(self):
        fixture = RuntimeBundleFixture(self)
        args = fixture.overlay_args()
        fresh, _ = base.read_canonical(args.runtime_bundle_fresh)
        identity, _ = base.read_canonical(args.runtime_bundle_identity)
        fresh["probe_intervals"][0]["probe_completed_ns"] = (
            fresh["probe_intervals"][0]["probe_started_ns"] - 1
        )
        fresh_raw = base.canonical_bytes(fresh)
        args.runtime_bundle_fresh.write_bytes(fresh_raw)
        identity["runtime_bundle_fresh_sha256"] = hashlib.sha256(
            fresh_raw
        ).hexdigest()
        args.runtime_bundle_identity.write_bytes(base.canonical_bytes(identity))
        with self.assertRaises(ValueError):
            overlay.validate(args)

    def test_exit_overlay_rejects_prelock_and_stale_evidence(self):
        for mutation in ("prelock", "stale"):
            with self.subTest(mutation=mutation):
                fixture = RuntimeBundleFixture(self)
                args = fixture.overlay_args()
                lock, _ = base.read_canonical(
                    args.runtime_bundle_readiness_lock
                )
                fresh, _ = base.read_canonical(args.runtime_bundle_fresh)
                identity, _ = base.read_canonical(
                    args.runtime_bundle_identity
                )
                manifest, _ = base.read_canonical(args.manifest)
                if mutation == "prelock":
                    lock["event_ns"] = fresh["started_ns"] + 1
                    lock_raw = base.canonical_bytes(lock)
                    args.runtime_bundle_readiness_lock.write_bytes(lock_raw)
                    fresh["readiness_lock_sha256"] = hashlib.sha256(
                        lock_raw
                    ).hexdigest()
                else:
                    maximum = fixture.v1.contract["readiness_v2_3"][
                        "fresh_snapshot_maximum_age_ns"
                    ]
                    acquisition = fresh["completed_ns"] + maximum + 1
                    manifest["acquisition_started_ns"] = acquisition
                    manifest["phase_closed_ns"] = acquisition + 100
                    args.manifest.write_bytes(base.canonical_bytes(manifest))
                    identity["started_ns"] = acquisition + 1
                    identity["completed_ns"] = acquisition + 2
                    for process in identity["processes"]:
                        process["observed_ns"] = acquisition + 1
                fresh_raw = base.canonical_bytes(fresh)
                args.runtime_bundle_fresh.write_bytes(fresh_raw)
                identity["runtime_bundle_fresh_sha256"] = hashlib.sha256(
                    fresh_raw
                ).hexdigest()
                args.runtime_bundle_identity.write_bytes(
                    base.canonical_bytes(identity)
                )
                with self.assertRaises(ValueError):
                    overlay.validate(args)

    def test_outer_default_exit_is_the_runtime_bundle_overlay(self):
        fixture = self.outer_fixture()
        runner = outer_tests.FakeRunner(
            fixture.plan,
            fixture.phase_id,
        )
        with self.assertRaises((OSError, SyntaxError, ValueError)):
            outer_tests.acquire.acquire(
                fixture.plan_path,
                fixture.root / "output",
                fixture.phase_id,
                runner=runner,
                now_ns=outer_tests.TickingClock(),
                preflight_collector=fixture.fake_preflight,
                plan_validator=lambda path, plan: None,
                v22_evaluator=fixture.fake_v22,
                v23_validator=fixture.fake_v23,
            )
        self.assertFalse((fixture.root / "output" / "RESULT.json").exists())

    def test_outer_runtime_bundle_roots_are_load_bearing(self):
        fixture = self.outer_fixture()
        seen = {}

        def rejecting_validator(**paths):
            seen.update(paths)
            paths["runtime_bundle_snapshot"].unlink()
            raise ValueError("E_TEST_RUNTIME_BUNDLE_ROOT_DELETED")

        with self.assertRaisesRegex(
            ValueError,
            "E_TEST_RUNTIME_BUNDLE_ROOT_DELETED",
        ):
            outer_tests.acquire.acquire(
                fixture.plan_path,
                fixture.root / "output",
                fixture.phase_id,
                runner=outer_tests.FakeRunner(
                    fixture.plan,
                    fixture.phase_id,
                ),
                now_ns=outer_tests.TickingClock(),
                preflight_collector=fixture.fake_preflight,
                plan_validator=lambda path, plan: None,
                v22_evaluator=fixture.fake_v22,
                v23_validator=fixture.fake_v23,
                runtime_bundle_validator=rejecting_validator,
            )
        self.assertEqual(
            set(seen),
            {
                "base_artifact_snapshot",
                "base_fresh_snapshot",
                "base_readiness_lock",
                "base_runtime_identity",
                "base_support",
                "candidate",
                "contract",
                "driver_support",
                "manifest",
                "runtime_bundle_fresh",
                "runtime_bundle_identity",
                "runtime_bundle_plan",
                "runtime_bundle_readiness_lock",
                "runtime_bundle_snapshot",
            },
        )
        self.assertFalse((fixture.root / "output" / "RESULT.json").exists())


if __name__ == "__main__":
    unittest.main()
