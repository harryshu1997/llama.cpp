#!/usr/bin/env python3

import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
V23 = HERE.parent
S39 = V23.parent
sys.path.insert(0, str(V23))
sys.path.insert(0, str(S39))

import acquire_a_only_v23 as acquire
import cp0_r1_evidence_v2 as v2
import cp0_r1_evidence_v22 as v22
import cp0_r1_evidence_v23 as v23
import v23_common as common


class TickingClock:
    def __init__(self):
        self.value = 100

    def __call__(self):
        value = self.value
        self.value += 10
        return value


class FakeRunner:
    def __init__(self, plan, phase_id, *, fail_stage=None, stderr_stage=None):
        self.plan = plan
        self.phase_id = phase_id
        self.fail_stage = fail_stage
        self.stderr_stage = stderr_stage
        self.calls = []

    def _row(self, role, event_ns=165):
        return {
            "acquisition_id": self.phase_id,
            "event_ns": event_ns,
            "kind": "fake",
            "phase": "A_ONLY",
            "phase_id": self.phase_id,
            "role": role,
        }

    def run(self, argv, *, timeout):
        del timeout
        stage = ("artifact", "fresh", "acquisition")[len(self.calls)]
        self.calls.append(list(argv))
        output = Path(argv[argv.index("--output") + 1])
        if stage == "artifact":
            common.write_exclusive(
                output / self.plan["output_files"]["artifact_snapshot"],
                {"stage": "artifact"},
            )
            common.write_exclusive(
                output / "runtime_bundle_snapshot.json",
                {"stage": "runtime-bundle-artifact"},
            )
        elif stage == "fresh":
            common.write_exclusive(
                output / self.plan["output_files"]["fresh_snapshot"],
                {"stage": "fresh"},
            )
            common.write_exclusive(
                output / self.plan["output_files"]["readiness_lock"],
                {"stage": "lock"},
            )
            common.write_exclusive(
                output / "runtime_bundle_readiness_lock.json",
                {"stage": "runtime-bundle-lock"},
            )
            common.write_exclusive(
                output / "runtime_bundle_fresh.json",
                {"stage": "runtime-bundle-fresh"},
            )
        else:
            for role, relative in self.plan["payload_roles"].items():
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(v2.canonical_line(self._row(role)))
            common.write_exclusive(
                output / self.plan["output_files"]["runtime_identity"],
                {"stage": "runtime"},
            )
            common.write_exclusive(
                output / self.plan["output_files"]["runtime_bundle_identity"],
                {"stage": "runtime-bundle-identity"},
            )
        return subprocess.CompletedProcess(
            argv,
            2 if stage == self.fail_stage else 0,
            b"",
            b"bad\n" if stage == self.stderr_stage else b"",
        )


class AcquireAOnlyV23Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.phase_id = "cp0-r1-v23-a-only-test"
        self.contract, contract_raw, _, candidate_raw = v23.validate_inputs(
            v23.DEFAULT_CONTRACT,
            v23.DEFAULT_CANDIDATE,
        )
        (
            v22_contract,
            _,
            _,
            _,
            _,
            _,
        ) = v22.validate_inputs(v22.DEFAULT_CONTRACT, v23.DEFAULT_CANDIDATE)
        roles = v22_contract["phase_protocol"]["phase_roles"]["A_ONLY"]
        payload_roles = sorted(set(roles) - acquire.PRE_ROLES)
        driver = self.root / "driver"
        driver.write_bytes(b"driver\n")
        driver.chmod(0o755)
        base_support = self.root / "base-support.py"
        entry_support = self.root / "entry-support.py"
        support = self.root / "support.py"
        runtime_plan = self.root / "runtime-plan.json"
        command_plan = self.root / "command-plan.json"
        base_support.write_bytes(b"base support\n")
        entry_support.write_bytes(b"entry support\n")
        support.write_bytes(b"support\n")
        runtime_plan.write_bytes(b"runtime plan\n")
        command_plan.write_bytes(b"command plan\n")
        source_manifest = self.root / "source-manifest.sha256"
        source_manifest.write_bytes(b"source manifest\n")

        def binding(path, argv_index):
            return {
                "argv_index": argv_index,
                "bytes": path.stat().st_size,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        readiness_argv = [
            str(driver),
            "--base-support",
            str(base_support),
            "--support",
            str(support),
            "--runtime-bundle-plan",
            str(runtime_plan),
            "--candidate",
            str(v23.DEFAULT_CANDIDATE.resolve()),
            "--contract",
            str(v23.DEFAULT_CONTRACT.resolve()),
            "--entry-support",
            str(entry_support),
            "--phase-id",
            "{phase_id}",
            "--pre",
            "{pre_dir}",
            "--output",
            "{output_dir}",
        ]
        readiness_bindings = [
            binding(driver, 0),
            binding(base_support, 2),
            binding(support, 4),
            binding(runtime_plan, 6),
            binding(v23.DEFAULT_CANDIDATE.resolve(), 8),
            binding(v23.DEFAULT_CONTRACT.resolve(), 10),
            binding(entry_support, 12),
        ]
        self.plan = {
            "candidate_sha256": common.sha256_bytes(candidate_raw),
            "contract_sha256": common.sha256_bytes(contract_raw),
            "drivers": {
                "acquisition": {
                    "argv_template": [
                        str(driver),
                        "--candidate",
                        str(v23.DEFAULT_CANDIDATE.resolve()),
                        "--command-plan",
                        str(command_plan),
                        "--contract",
                        str(v23.DEFAULT_CONTRACT.resolve()),
                        "--phase-id",
                        "{phase_id}",
                        "--pre",
                        "{pre_dir}",
                        "--started",
                        "{acquisition_started_ns}",
                        "--output",
                        "{output_dir}",
                    ],
                    "executed_files": [
                        binding(driver, 0),
                        binding(v23.DEFAULT_CANDIDATE.resolve(), 2),
                        binding(command_plan, 4),
                        binding(v23.DEFAULT_CONTRACT.resolve(), 6),
                    ],
                    "timeout_seconds": 60,
                },
                "artifact": {
                    "argv_template": copy.deepcopy(readiness_argv),
                    "executed_files": copy.deepcopy(readiness_bindings),
                    "timeout_seconds": 60,
                },
                "fresh": {
                    "argv_template": copy.deepcopy(readiness_argv),
                    "executed_files": copy.deepcopy(readiness_bindings),
                    "timeout_seconds": 60,
                },
            },
            "model_id": "qwen3-14b-q4_k_m",
            "output_files": {
                "artifact_snapshot": "artifact.json",
                "fresh_snapshot": "fresh.json",
                "readiness_lock": "lock.json",
                "runtime_bundle_identity": "runtime-bundle.json",
                "runtime_identity": "runtime.json",
            },
            "payload_roles": {
                role: f"roles/{index:02d}.jsonl"
                for index, role in enumerate(payload_roles)
            },
            "phase": "A_ONLY",
            "schema": acquire.PLAN_SCHEMA,
            "source_manifest_path": str(source_manifest),
            "source_manifest_sha256": hashlib.sha256(
                source_manifest.read_bytes()
            ).hexdigest(),
            "source_root": str(self.root),
        }
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_bytes(common.canonical_bytes(self.plan))

    def fake_preflight(self, contract, candidate, phase, phase_id, locks, timeout):
        del contract, candidate, locks, timeout
        row = {
            "acquisition_id": phase_id,
            "completed_ns": 145,
            "event_ns": 145,
            "forbidden_work_executed": False,
            "kind": "meta",
            "phase": phase,
            "phase_id": phase_id,
            "probe_labels": [],
            "role": "phase.preflight",
        }
        return v2.canonical_line(row)

    def fake_v22(self, root, manifest_name, *args):
        del args
        manifest = v2.parse_json(
            (root / manifest_name).read_bytes(),
            "manifest",
        )
        self.assertEqual(manifest["phase"], "A_ONLY")
        self.assertEqual(len(manifest["artifacts"]), 14)
        self.assertEqual(
            {row["role"] for row in manifest["artifacts"]},
            set(self.contract["phase_protocol"]["phase_roles"]["A_ONLY"]),
        )
        return {"status": "MODEL_A_QUALIFICATION_PASS"}

    def fake_v23(self, contract, candidate, root, artifact, lock, fresh, runtime, **kwargs):
        del contract, candidate, root, kwargs
        self.assertEqual(
            [path.read_text().strip() for path in (artifact, lock, fresh, runtime)],
            [
                '{"stage":"artifact"}',
                '{"stage":"lock"}',
                '{"stage":"fresh"}',
                '{"stage":"runtime"}',
            ],
        )
        return {"runtime": {"executor_count": 3}}

    def fake_runtime_bundle(self, **paths):
        self.assertEqual(
            set(paths),
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
        for name, path in paths.items():
            self.assertTrue(path.is_file(), name)
        return {
            "runtime_bundle_fresh_sha256": "1" * 64,
            "runtime_bundle_identity_sha256": "2" * 64,
            "runtime_bundle_plan_sha256": "3" * 64,
            "runtime_bundle_snapshot_sha256": "4" * 64,
            "schema": "s39-cp0-r1-runtime-bundle-validation-v1",
            "status": "RUNTIME_BUNDLE_PROVENANCE_PASS",
        }

    def run_acquisition(self, runner):
        return acquire.acquire(
            self.plan_path,
            self.root / "output",
            self.phase_id,
            runner=runner,
            now_ns=TickingClock(),
            preflight_collector=self.fake_preflight,
            plan_validator=lambda path, plan: None,
            v22_evaluator=self.fake_v22,
            v23_validator=self.fake_v23,
            runtime_bundle_validator=self.fake_runtime_bundle,
        )

    def test_three_stage_order_and_independent_verdict(self):
        runner = FakeRunner(self.plan, self.phase_id)
        result = self.run_acquisition(runner)
        self.assertEqual(result["status"], "MODEL_A_QUALIFICATION_PASS")
        self.assertEqual(
            result["runtime_bundle"]["status"],
            "RUNTIME_BUNDLE_PROVENANCE_PASS",
        )
        self.assertEqual(len(runner.calls), 3)
        self.assertIn("/artifact", runner.calls[0][-1])
        self.assertIn("/fresh", runner.calls[1][-1])
        self.assertIn("/acquisition", runner.calls[2][-1])
        self.assertTrue((self.root / "output" / "RESULT.json").is_file())

    def test_driver_failure_is_retained_and_refused(self):
        runner = FakeRunner(self.plan, self.phase_id, fail_stage="acquisition")
        with self.assertRaisesRegex(common.ReadinessError, "E_DRIVER_EXIT"):
            self.run_acquisition(runner)
        output = self.root / "output"
        self.assertTrue((output / "DRIVER_RECEIPT.json").is_file())
        self.assertFalse((output / "RESULT.json").exists())

    def test_driver_stderr_is_refused(self):
        runner = FakeRunner(self.plan, self.phase_id, stderr_stage="fresh")
        with self.assertRaisesRegex(common.ReadinessError, "E_FRESH_DRIVER_STDERR"):
            self.run_acquisition(runner)

    def test_mutated_driver_is_rejected_before_any_stage(self):
        Path(self.plan["drivers"]["artifact"]["argv_template"][0]).write_bytes(
            b"changed\n"
        )
        runner = FakeRunner(self.plan, self.phase_id)
        with self.assertRaisesRegex(common.ReadinessError, "bytes|sha256"):
            self.run_acquisition(runner)
        self.assertEqual(runner.calls, [])

    def test_plan_source_binding_is_required_and_contained(self):
        for mutate, message in (
            (
                lambda plan: plan.pop("source_manifest_sha256"),
                "plan",
            ),
            (
                lambda plan: plan.__setitem__(
                    "source_manifest_path",
                    str(self.root.parent / "outside.sha256"),
                ),
                "plan.source_manifest_path",
            ),
        ):
            with self.subTest(message=message):
                plan = copy.deepcopy(self.plan)
                mutate(plan)
                path = self.root / f"bad-source-{message.replace('.', '-')}.json"
                path.write_bytes(common.canonical_bytes(plan))
                with self.assertRaisesRegex(common.ReadinessError, message):
                    acquire._load_plan(
                        path,
                        v23.DEFAULT_CONTRACT.read_bytes(),
                        v23.DEFAULT_CANDIDATE.read_bytes(),
                        self.contract["phase_protocol"]["phase_roles"]["A_ONLY"],
                    )

    def test_every_required_driver_file_flag_is_fail_closed(self):
        roles = self.contract["phase_protocol"]["phase_roles"]["A_ONLY"]
        for name, flags in acquire.DRIVER_FILE_FLAGS.items():
            for flag in sorted(flags):
                with self.subTest(driver=name, flag=flag):
                    plan = copy.deepcopy(self.plan)
                    template = plan["drivers"][name]["argv_template"]
                    template[template.index(flag)] = "--renamed-file-flag"
                    path = self.root / (
                        f"missing-{name}-{flag.removeprefix('--')}.json"
                    )
                    path.write_bytes(common.canonical_bytes(plan))
                    with self.assertRaisesRegex(
                        common.ReadinessError,
                        "E_DRIVER_FLAG",
                    ):
                        acquire._load_plan(
                            path,
                            v23.DEFAULT_CONTRACT.read_bytes(),
                            v23.DEFAULT_CANDIDATE.read_bytes(),
                            roles,
                        )

    def test_unbound_literal_regular_file_is_fail_closed(self):
        plan = copy.deepcopy(self.plan)
        extra = self.root / "extra-input.json"
        extra.write_bytes(b"extra\n")
        plan["drivers"]["acquisition"]["argv_template"].extend(
            ["--extra-input", str(extra)]
        )
        path = self.root / "unbound-literal.json"
        path.write_bytes(common.canonical_bytes(plan))
        with self.assertRaisesRegex(
            common.ReadinessError,
            "executed_file_indexes",
        ):
            acquire._load_plan(
                path,
                v23.DEFAULT_CONTRACT.read_bytes(),
                v23.DEFAULT_CANDIDATE.read_bytes(),
                self.contract["phase_protocol"]["phase_roles"]["A_ONLY"],
            )

    def test_late_created_literal_file_cannot_escape_capture(self):
        late = self.root / "late-input.json"
        template = ["/not/a/file", "--late", str(late), "{output_dir}"]
        late.write_bytes(b"late\n")
        with self.assertRaisesRegex(
            common.ReadinessError,
            "E_UNCAPTURED_DRIVER_FILE",
        ):
            acquire._substitute_argv(
                template,
                self.phase_id,
                self.root / "output",
                self.root / "pre",
                None,
                {},
            )

    def test_executed_file_directory_is_rejected_from_open_descriptor(self):
        plan = copy.deepcopy(self.plan)
        driver = plan["drivers"]["artifact"]
        driver["argv_template"][0] = str(self.root)
        driver["executed_files"][0] = {
            "argv_index": 0,
            "bytes": 1,
            "path": str(self.root),
            "sha256": "0" * 64,
        }
        output = self.root / "capture-output"
        output.mkdir()
        with self.assertRaisesRegex(common.ReadinessError, "E_DRIVER_FILE_TYPE"):
            acquire._capture_executed_files(plan, output)

    def test_paid_cli_rejects_mutated_manifest_source_before_imports(self):
        bootstrap_root = self.root / "bootstrap"
        bootstrap_root.mkdir()
        entrypoint = bootstrap_root / "acquire_a_only_v23.py"
        entrypoint.write_bytes(Path(acquire.__file__).read_bytes())
        support = bootstrap_root / "support.py"
        support.write_bytes(b"value = 1\n")
        records = []
        for path in (entrypoint, support):
            records.append(
                (
                    path.name,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
        manifest = bootstrap_root / "SOURCE_SHA256SUMS.txt"
        manifest_raw = "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(records)
        ).encode("ascii")
        manifest.write_bytes(manifest_raw)
        plan = {
            "source_manifest_path": str(manifest),
            "source_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "source_root": str(bootstrap_root),
        }
        plan_path = bootstrap_root / "plan.json"
        plan_path.write_bytes(common.canonical_bytes(plan))
        support.write_bytes(b"value = 2\n")

        completed = subprocess.run(
            [
                sys.executable,
                str(entrypoint),
                "--plan",
                str(plan_path),
                "--output-root",
                str(bootstrap_root / "output"),
                "--phase-id",
                "cp0-r1-v23-a-only-bootstrap-test",
            ],
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        stdout = completed.stdout.decode("ascii")
        self.assertIn("CP0_R1_A_ONLY_V2_3_BOOTSTRAP_REFUSED", stdout)
        self.assertIn("E_BOOTSTRAP_SOURCE_SHA256: support.py", stdout)
        self.assertNotIn("Traceback", completed.stderr.decode("ascii"))

    def test_paid_semantic_load_uses_bootstrapped_plan_not_replaced_path(self):
        bootstrapped = copy.deepcopy(self.plan)
        bootstrapped_raw = common.canonical_bytes(bootstrapped)
        replacement = copy.deepcopy(self.plan)
        replacement["model_id"] = "substituted-model"
        self.plan_path.write_bytes(common.canonical_bytes(replacement))
        with mock.patch.multiple(
            acquire,
            _BOOTSTRAP_PLAN=bootstrapped,
            _BOOTSTRAP_PLAN_PATH=self.plan_path.absolute(),
            _BOOTSTRAP_PLAN_RAW=bootstrapped_raw,
        ):
            loaded = acquire._load_plan(
                self.plan_path,
                v23.DEFAULT_CONTRACT.read_bytes(),
                v23.DEFAULT_CANDIDATE.read_bytes(),
                self.contract["phase_protocol"]["phase_roles"]["A_ONLY"],
            )
        self.assertEqual(loaded["model_id"], "qwen3-14b-q4_k_m")

    def test_paid_exit_validator_uses_bootstrapped_overlay_module(self):
        class BoundOverlay:
            @staticmethod
            def validate(args):
                return {"bound": str(args.marker)}

        with (
            mock.patch.object(
                acquire,
                "_BOOTSTRAP_SOURCES",
                {"bound": (self.root, b"bound")},
            ),
            mock.patch.object(
                acquire,
                "_bound_modules",
                {"runtime_bundle_overlay_v1": BoundOverlay},
                create=True,
            ),
            mock.patch.object(
                acquire,
                "_load_source",
                side_effect=AssertionError("live source reread"),
            ),
        ):
            result = acquire._default_runtime_bundle_validator(
                marker=self.root / "marker"
            )
        self.assertEqual(result, {"bound": str(self.root / "marker")})

    def test_paid_plan_validator_rederives_plan_graph_and_manifest(self):
        calls = []
        expected_sources = [self.root / "one.py", self.root / "two.json"]
        plan = copy.deepcopy(self.plan)

        class BoundPlan:
            @staticmethod
            def build_plan():
                calls.append("build")
                return copy.deepcopy(plan)

            @staticmethod
            def exact(value, expected, field):
                calls.append(("exact", field))
                self.assertEqual(value, expected)

            @staticmethod
            def canonical_bytes(value):
                return common.canonical_bytes(value)

            @staticmethod
            def read_regular(path):
                calls.append(("read", path))
                return path.read_bytes()

            @staticmethod
            def source_paths(value):
                calls.append(("sources", value["schema"]))
                return expected_sources

            @staticmethod
            def validate_bound_source_manifest(value, sources):
                calls.append(("manifest", value["schema"]))
                self.assertEqual(sources, expected_sources)

        with (
            mock.patch.object(
                acquire,
                "_BOOTSTRAP_SOURCES",
                {"bound": (self.root, b"bound")},
            ),
            mock.patch.object(
                acquire,
                "_bound_modules",
                {"acquisition_plan_common_v1": BoundPlan},
                create=True,
            ),
            mock.patch.object(acquire, "_BOOTSTRAP_PLAN_RAW", None),
        ):
            acquire._default_plan_validator(self.plan_path, plan)
        self.assertIn("build", calls)
        self.assertIn(("sources", acquire.PLAN_SCHEMA), calls)
        self.assertIn(("manifest", acquire.PLAN_SCHEMA), calls)

    def test_original_driver_mutation_after_capture_cannot_change_execution(self):
        source = Path(
            self.plan["drivers"]["artifact"]["argv_template"][0]
        )

        class MutatingRunner(FakeRunner):
            def __init__(inner, plan, phase_id):
                super().__init__(plan, phase_id)
                inner.executed_bytes = []

            def run(inner, argv, *, timeout):
                source.write_bytes(b"changed-after-capture\n")
                inner.executed_bytes.append(Path(argv[0]).read_bytes())
                return super(MutatingRunner, inner).run(
                    argv,
                    timeout=timeout,
                )

        runner = MutatingRunner(self.plan, self.phase_id)
        result = self.run_acquisition(runner)
        self.assertEqual(result["status"], "MODEL_A_QUALIFICATION_PASS")
        self.assertEqual(runner.executed_bytes, [b"driver\n"] * 3)
        self.assertTrue(
            all(
                str(self.root / "output" / "executed") in argv[0]
                for argv in runner.calls
            )
        )
        self.assertEqual(source.read_bytes(), b"changed-after-capture\n")

    def test_missing_payload_role_is_refused(self):
        class MissingRunner(FakeRunner):
            def run(inner, argv, *, timeout):
                result = super(MissingRunner, inner).run(argv, timeout=timeout)
                if len(inner.calls) == 3:
                    missing = next(iter(inner.plan["payload_roles"].values()))
                    (Path(argv[-1]) / missing).unlink()
                return result

        with self.assertRaisesRegex(common.ReadinessError, "E_OUTPUT"):
            self.run_acquisition(MissingRunner(self.plan, self.phase_id))

    def test_payload_event_before_paid_interval_is_refused(self):
        class StaleRunner(FakeRunner):
            def _row(inner, role, event_ns=175):
                return super(StaleRunner, inner)._row(role, 1)

        with self.assertRaisesRegex(
            common.ReadinessError,
            "E_ACQUISITION_INTERVAL",
        ):
            self.run_acquisition(StaleRunner(self.plan, self.phase_id))

    def test_path_escape_and_unknown_placeholder_are_rejected(self):
        for mutate, message in (
            (
                lambda plan: plan["payload_roles"].__setitem__(
                    next(iter(plan["payload_roles"])),
                    "../escape.jsonl",
                ),
                "E_PATH",
            ),
            (
                lambda plan: plan["drivers"]["artifact"]["argv_template"].append(
                    "{verdict}"
                ),
                "placeholders",
            ),
        ):
            with self.subTest(message=message):
                plan = copy.deepcopy(self.plan)
                mutate(plan)
                path = self.root / f"bad-{message}.json"
                path.write_bytes(common.canonical_bytes(plan))
                with self.assertRaisesRegex(common.ReadinessError, message):
                    acquire._load_plan(
                        path,
                        v23.DEFAULT_CONTRACT.read_bytes(),
                        v23.DEFAULT_CANDIDATE.read_bytes(),
                        self.contract["phase_protocol"]["phase_roles"]["A_ONLY"],
                    )


if __name__ == "__main__":
    unittest.main()
