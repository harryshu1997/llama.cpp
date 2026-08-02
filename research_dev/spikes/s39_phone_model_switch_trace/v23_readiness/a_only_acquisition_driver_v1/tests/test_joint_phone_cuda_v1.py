#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


HERE = Path(__file__).resolve().parent
PRODUCER = (
    HERE.parent
    / "producers_v1"
    / "joint_phone_cuda_v1.py"
)


def load_producer():
    spec = importlib.util.spec_from_file_location(
        "s39_joint_phone_cuda_v1_test",
        PRODUCER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load producer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


joint = load_producer()


def marker(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


FAKE_CAPTURE = """#!/usr/bin/python3 -I
import argparse
import json
import os
from pathlib import Path
import time

def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\\n").encode("ascii")

def write_new(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)

def stamp(rows, start):
    for index, row in enumerate(rows):
        row["event_ns"] = start + index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--barrier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--pre-dir", required=True)
    parser.add_argument("--acquisition-started-ns", required=True)
    parser.add_argument("--command-plan-sha256", required=True)
    args = parser.parse_args()
    value = json.loads(args.template.read_text(encoding="ascii"))
    started = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    if args.kind == "cuda":
        write_new(args.barrier.with_suffix(".cuda"), str(started).encode("ascii"))
        deadline = time.monotonic() + 2
        while not args.barrier.with_suffix(".phone").exists():
            if time.monotonic() >= deadline:
                return 3
            time.sleep(0.001)
        base = started + 10
        stamp(value["cuda_route_rows"], base)
        stamp(value["cuda_memory_rows"], base + 20)
        stamp(value["quality_cuda_rows"], base + 30)
        value["bridge_start_row"]["event_ns"] = started + 1
        ready = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        value["bridge_ready_row"]["event_ns"] = ready
        value["runtime_process"]["observed_ns"] = ready
        value["started_ns"] = started
        value["completed_ns"] = ready + 1
    else:
        deadline = time.monotonic() + 2
        while not args.barrier.with_suffix(".cuda").exists():
            if time.monotonic() >= deadline:
                return 3
            time.sleep(0.001)
        base = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        stamp(value["mechanics_rows"], base)
        stamp(value["quality_phone_rows"], base + 20)
        stamp(value["placement_op15_rows"], base + 100)
        stamp(value["placement_op12_rows"], base + 200)
        stamp(value["route_transfer_rows"], base + 300)
        stamp(value["bridge_publication_rows"], base + 400)
        for index, process in enumerate(value["runtime_processes"]):
            process["observed_ns"] = base + 450 + index
        value["started_ns"] = started
        value["completed_ns"] = base + 500
        write_new(args.barrier.with_suffix(".phone"), b"ready")
    write_new(args.output, canonical(value))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""


class JointFixture:
    def __init__(self, test):
        temporary = tempfile.TemporaryDirectory()
        test.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pre = self.root / "pre"
        self.pre.mkdir()
        self.output = self.root / "joint.json"
        self.model_sha = marker("model")
        self.command_plan_sha = marker("command-plan")
        self.phase_id = "cp0-r1-v23-a-only-joint-test"
        self.mechanism = {
            "desktop": [["cuda-route"]],
            "op12": [["stage-mid"]],
            "op15": [["stage-head"]],
        }
        self.mechanism_sha = joint.sha256_bytes(
            joint.canonical_bytes(self.mechanism)
        )
        self.script = self.root / "fake_capture.py"
        self.script.write_text(FAKE_CAPTURE, encoding="ascii")
        self.script.chmod(0o755)
        self.barrier = self.root / "barrier"
        self.phone_template = self.root / "phone-template.json"
        self.cuda_template = self.root / "cuda-template.json"
        self.phone_template.write_bytes(canonical(self.phone_fragment()))
        self.cuda_template.write_bytes(canonical(self.cuda_fragment()))
        self.plan_path = self.root / "capture-plan.json"
        self.plan = self.capture_plan()
        self.plan_path.write_bytes(canonical(self.plan))

    @staticmethod
    def rows(kind, count):
        return [{"event_ns": 0, "kind": kind} for _ in range(count)]

    def common(self, schema):
        return {
            "completed_ns": 0,
            "mechanism_commands_sha256": self.mechanism_sha,
            "model_id": joint.MODEL_ID,
            "model_sha256": self.model_sha,
            "phase_id": self.phase_id,
            "route_epoch": 9,
            "schema": schema,
            "started_ns": 0,
        }

    @staticmethod
    def dependency(endpoint):
        root = "/usr/lib" if endpoint == "cuda" else "/vendor/lib64"
        return {
            "build_id": None,
            "ctime_ns": 1,
            "device_id": 2,
            "inode": 3,
            "mode": 33261,
            "mtime_ns": 4,
            "path": root + "/libc.so",
            "size": 5,
        }

    def runtime_process(self, bundle_id, endpoint):
        root = "/tmp" if endpoint == "cuda" else "/data/local/tmp"
        return {
            "boot_id": marker(endpoint),
            "bundle_id": bundle_id,
            "endpoint": endpoint,
            "identity_probe_sha256": marker(bundle_id + "-probe"),
            "launcher_path": root + "/" + bundle_id,
            "loaded_repo_component_ids": [
                bundle_id + ".launcher",
                bundle_id + ".runtime",
            ],
            "observed_ns": 0,
            "pid": len(bundle_id) + 100,
            "start_ticks": len(bundle_id) + 200,
            "system_dependencies": [self.dependency(endpoint)],
        }

    def phone_fragment(self):
        value = {
            **self.common(joint.PHONE_SCHEMA),
            "bridge_publication_rows": self.rows(
                "phone_publication_received",
                8,
            ),
            "mechanics_rows": [
                *self.rows("meta", 1),
                *self.rows("request", 8),
            ],
            "op12_runtime": {"phone": "op12"},
            "op15_runtime": {"phone": "op15"},
            "placement_op12_rows": [
                *self.rows("meta", 1),
                *self.rows("node", 1),
            ],
            "placement_op15_rows": [
                *self.rows("meta", 1),
                *self.rows("node", 1),
            ],
            "quality_phone_rows": self.rows("output", 64),
            "route_transfer_rows": [
                *self.rows("meta", 1),
                *self.rows("transfer", 1),
            ],
            "runtime_processes": [
                self.runtime_process("op12_stagenet", "op12"),
                self.runtime_process("op15_direct_relay", "op15"),
                self.runtime_process("op15_stagenet", "op15"),
            ],
        }
        return value

    def cuda_fragment(self):
        return {
            **self.common(joint.CUDA_SCHEMA),
            "bridge_ready_row": {"event_ns": 0, "kind": "cuda_ready"},
            "bridge_start_row": {"event_ns": 0, "kind": "cuda_load_start"},
            "cuda_memory_rows": [
                *self.rows("before", 1),
                *self.rows("ready", 1),
                *self.rows("after", 1),
            ],
            "cuda_route_rows": [
                *self.rows("meta", 1),
                *self.rows("request", 8),
            ],
            "gpu_runtime": {"executor": "cuda"},
            "quality_cuda_rows": self.rows("output", 64),
            "runtime_process": self.runtime_process("cuda_route", "cuda"),
        }

    def binding(self, path, index):
        raw = path.read_bytes()
        return {
            "argv_index": index,
            "bytes": len(raw),
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def command(self, kind, template, result):
        argv = [
            str(self.script),
            "--kind",
            kind,
            "--template",
            str(template),
            "--barrier",
            str(self.barrier),
            "--output",
            "{output_path}",
            "--phase-id",
            "{phase_id}",
            "--pre-dir",
            "{pre_dir}",
            "--acquisition-started-ns",
            "{acquisition_started_ns}",
            "--command-plan-sha256",
            "{command_plan_sha256}",
        ]
        return {
            "argv_template": argv,
            "executed_files": [
                self.binding(self.script, 0),
                self.binding(template, 4),
            ],
            "result_filename": result,
            "timeout_seconds": 5,
        }

    def capture_plan(self):
        return {
            "commands": {
                "cuda": self.command(
                    "cuda",
                    self.cuda_template,
                    "cuda-fragment.json",
                ),
                "phone": self.command(
                    "phone",
                    self.phone_template,
                    "phone-fragment.json",
                ),
            },
            "mechanism_commands": copy.deepcopy(self.mechanism),
            "model_id": joint.MODEL_ID,
            "model_sha256": self.model_sha,
            "phase": joint.PHASE,
            "schema": joint.PLAN_SCHEMA,
        }

    def argv(self):
        return [
            sys.executable,
            "-I",
            "-B",
            str(PRODUCER),
            "--capture-plan",
            str(self.plan_path),
            "--output",
            str(self.output),
            "--phase-id",
            self.phase_id,
            "--pre-dir",
            str(self.pre),
            "--acquisition-started-ns",
            str(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - 1_000_000),
            "--command-plan-sha256",
            self.command_plan_sha,
        ]

    def run(self):
        return subprocess.run(
            self.argv(),
            capture_output=True,
            check=False,
            timeout=10,
        )


class JointProducerTests(unittest.TestCase):
    def fixture(self):
        return JointFixture(self)

    def test_concurrent_pair_emits_raw_joint_fragment(self):
        fixture = self.fixture()
        completed = fixture.run()
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertEqual(completed.stdout, b"")
        self.assertEqual(completed.stderr, b"")
        value = json.loads(fixture.output.read_text(encoding="ascii"))
        self.assertEqual(value["schema"], joint.OUTPUT_SCHEMA)
        self.assertEqual(len(value["mechanics_rows"]), 9)
        self.assertEqual(len(value["quality_cuda_rows"]), 64)
        self.assertEqual(len(value["quality_phone_rows"]), 64)
        self.assertEqual(
            [row["bundle_id"] for row in value["runtime_processes"]],
            [
                "cuda_route",
                "op12_stagenet",
                "op15_direct_relay",
                "op15_stagenet",
            ],
        )
        self.assertEqual(
            [row["kind"] for row in value["bridge_rows"]],
            [
                "cuda_load_start",
                *(["phone_publication_received"] * 8),
                "cuda_ready",
            ],
        )
        evidence = fixture.output.parent / f".{fixture.output.name}.evidence"
        self.assertTrue((evidence / "phone.receipt.json").is_file())
        self.assertTrue((evidence / "cuda.receipt.json").is_file())

    def test_inline_plan_requires_matching_digest(self):
        commands = {
            "desktop": [["/launcher", "--plan-json", "{}"]],
            "op12": [["/op12"]],
            "op15": [["/op15"]],
        }
        with self.assertRaisesRegex(
            joint.CaptureError,
            "plan_sha.count",
        ):
            joint.validate_mechanism_commands(commands)

        commands["desktop"][0].extend(["--plan-sha256", "0" * 64])
        with self.assertRaisesRegex(
            joint.CaptureError,
            "plan_sha",
        ):
            joint.validate_mechanism_commands(commands)

    def test_plan_digest_without_inline_plan_is_rejected(self):
        commands = {
            "desktop": [["/launcher", "--plan-sha256", "0" * 64]],
            "op12": [["/op12"]],
            "op15": [["/op15"]],
        }
        with self.assertRaisesRegex(
            joint.CaptureError,
            "plan_sha.count",
        ):
            joint.validate_mechanism_commands(commands)

    def test_changed_executed_source_is_rejected(self):
        fixture = self.fixture()
        fixture.script.write_text(FAKE_CAPTURE + "\n", encoding="ascii")
        completed = fixture.run()
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"E_EXECUTED_BYTES", completed.stdout)
        self.assertFalse(fixture.output.exists())

    def test_nested_launch_plan_must_be_captured(self):
        fixture = self.fixture()
        command = fixture.plan["commands"]["cuda"]
        command["argv_template"].extend(
            ["--launch-plan", str(fixture.cuda_template)]
        )
        with self.assertRaisesRegex(
            joint.CaptureError,
            "E_EXECUTED_FILE_MISSING",
        ):
            joint.validate_command(command, "commands.cuda")

    def test_new_file_argument_must_be_captured_without_flag_allowlist(self):
        fixture = self.fixture()
        command = fixture.plan["commands"]["cuda"]
        command["executed_files"] = [command["executed_files"][0]]
        with self.assertRaisesRegex(
            joint.CaptureError,
            "E_EXECUTED_FILE_MISSING",
        ):
            joint.validate_command(command, "commands.cuda")

    def test_nested_child_must_be_self_contained(self):
        fixture = self.fixture()
        unsafe = fixture.root / "unsafe.py"
        unsafe.write_text(
            "#!/usr/bin/python3 -I\nimport live_repo_support\n",
            encoding="ascii",
        )
        unsafe.chmod(0o755)
        command = fixture.plan["commands"]["cuda"]
        command["argv_template"][0] = str(unsafe)
        command["executed_files"][0] = fixture.binding(unsafe, 0)
        with self.assertRaisesRegex(
            joint.CaptureError,
            "E_PRODUCER_IMPORT",
        ):
            joint.validate_command(command, "commands.cuda")

    def test_child_failure_is_fail_closed(self):
        fixture = self.fixture()
        failing = fixture.root / "failing.py"
        failing.write_text(
            "#!/usr/bin/python3 -I\n"
            "raise SystemExit(7)\n",
            encoding="ascii",
        )
        failing.chmod(0o755)
        command = fixture.plan["commands"]["phone"]
        command["argv_template"][0] = str(failing)
        command["executed_files"][0] = fixture.binding(failing, 0)
        fixture.plan_path.write_bytes(canonical(fixture.plan))
        completed = fixture.run()
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"E_COMMAND_EXIT", completed.stdout)
        self.assertFalse(fixture.output.exists())

    def test_fragment_mechanism_mismatch_is_rejected(self):
        fixture = self.fixture()
        fixture.plan["mechanism_commands"]["desktop"][0].append("--changed")
        fixture.plan_path.write_bytes(canonical(fixture.plan))
        completed = fixture.run()
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"mechanism_commands_sha256", completed.stdout)
        self.assertFalse(fixture.output.exists())

    def test_runtime_process_outside_capture_interval_is_rejected(self):
        fixture = self.fixture()
        value = fixture.runtime_process("op12_stagenet", "op12")
        value["observed_ns"] = 9
        with self.assertRaisesRegex(
            joint.CaptureError,
            "E_RUNTIME_PROCESS_INTERVAL",
        ):
            joint.validate_runtime_process(
                value,
                joint.PHONE_RUNTIME_BUNDLES,
                10,
                20,
                "process",
            )

    def test_runtime_process_wrong_membership_is_rejected(self):
        fixture = self.fixture()
        value = json.loads(fixture.phone_template.read_bytes())
        value["runtime_processes"][0]["bundle_id"] = "cuda_route"
        fixture.phone_template.write_bytes(canonical(value))
        fixture.plan["commands"]["phone"]["executed_files"][1] = fixture.binding(
            fixture.phone_template,
            4,
        )
        fixture.plan_path.write_bytes(canonical(fixture.plan))
        completed = fixture.run()
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"E_RUNTIME_BUNDLE", completed.stdout)
        self.assertFalse(fixture.output.exists())

    def test_runtime_process_order_is_rejected(self):
        fixture = self.fixture()
        value = json.loads(fixture.phone_template.read_bytes())
        value["runtime_processes"][0], value["runtime_processes"][1] = (
            value["runtime_processes"][1],
            value["runtime_processes"][0],
        )
        fixture.phone_template.write_bytes(canonical(value))
        fixture.plan["commands"]["phone"]["executed_files"][1] = fixture.binding(
            fixture.phone_template,
            4,
        )
        fixture.plan_path.write_bytes(canonical(fixture.plan))
        completed = fixture.run()
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"phone.runtime_processes.order", completed.stdout)
        self.assertFalse(fixture.output.exists())

    def test_runtime_process_noninteger_pid_is_rejected(self):
        fixture = self.fixture()
        value = json.loads(fixture.cuda_template.read_bytes())
        value["runtime_process"]["pid"] = 1.0
        fixture.cuda_template.write_bytes(canonical(value))
        fixture.plan["commands"]["cuda"]["executed_files"][1] = fixture.binding(
            fixture.cuda_template,
            4,
        )
        fixture.plan_path.write_bytes(canonical(fixture.plan))
        completed = fixture.run()
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"runtime_process.pid", completed.stdout)
        self.assertFalse(fixture.output.exists())


if __name__ == "__main__":
    unittest.main()
