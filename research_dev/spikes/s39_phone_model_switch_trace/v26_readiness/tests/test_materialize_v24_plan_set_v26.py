#!/usr/bin/env python3

import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import tempfile
import unittest


HERE = Path(__file__).resolve().parents[1]
S39 = HERE.parent
V24 = S39 / "v24_readiness"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ps = load("ps", HERE / "materialize_v24_plan_set_v26.py")
prod = ps.prod

V26_RECORD = json.loads(
    (ps.DEFAULT_V26_ROOT / "RUNTIME_INVENTORY_V2_6.json").read_bytes()
)
BODY = V26_RECORD["inventory"]
CONTRACT_V26 = json.loads(
    (HERE / "CP0_R1_EVIDENCE_CONTRACT_V2_6.json").read_bytes()
)
CONTRACT_V24 = json.loads(
    (V24 / "CP0_R1_EVIDENCE_CONTRACT_V2_4.json").read_bytes()
)
MONO_LAUNCH = json.loads(
    (
        V24 / "results" / "prephase_20260726T0915Z"
        / "cuda-monolithic-launch.json"
    ).read_bytes()
)
TOPOLOGY_PATH = HERE / "results" / "no_model_topology_final.json"

MIRROR_SOURCES = (
    "v24_readiness/CP0_R1_EVIDENCE_CONTRACT_V2_4.json",
    "v24_readiness/build_contract_v24.py",
    "v24_readiness/cp0_r1_evidence_v24.py",
    "v24_readiness/v24_common.py",
    "v24_readiness/desktop_deployment_v1/managed_runtime_launcher_usb_v1.py",
    "v24_readiness/desktop_deployment_v1/materialize_a_only_inputs_v1.py",
    "v24_readiness/desktop_deployment_v1/verify_topology_v1.py",
    "v24_readiness/producers_v1/cuda_route_v1.py",
    "v24_readiness/producers_v1/phone_route_v1.py",
    "v24_readiness/production_plan_v1/originate_runtime_v1.py",
    "v24_readiness/production_plan_v1/production_common_v1.py",
    "v23_readiness/a_only_acquisition_driver_v1/producers_v1/"
    "managed_runtime_launcher_v1.py",
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def synth_stat(size: int, inode: int) -> dict:
    return {
        "ctime_ns": 1785000000000000000,
        "device_id": 64769,
        "inode": inode,
        "mode": 0o100755,
        "mtime_ns": 1785000000000000000,
        "size": size,
    }


class Fixture:
    def __init__(self, case: unittest.TestCase):
        temporary = tempfile.TemporaryDirectory()
        case.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.mirror = self.root / "mirror"
        for relative in MIRROR_SOURCES:
            target = self.mirror / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(S39 / relative, target)
        for relative, source in ps.EXTRA_MIRROR_SOURCES.items():
            target = self.mirror / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            target.chmod(0o755)
        probe_target = self.mirror / ps.SNAPSHOT_PROBE_RELATIVE
        launcher_target = self.mirror / ps.SNAPSHOT_LAUNCHER_RELATIVE
        (
            self.mirror
            / "v24_readiness/desktop_deployment_v1"
            / "managed_runtime_launcher_usb_v1.py"
        ).chmod(0o755)
        self.joint_cwd = self.root / "joint-cwd"
        self.joint_cwd.mkdir()
        self.inputs_dir = self.root / "inputs"
        self.inputs_dir.mkdir()
        self.prephase = self.root / "prephase"
        self.prephase.mkdir()
        self.work_dir_patch = str(self.joint_cwd)
        case.addCleanup(
            setattr, ps, "DESKTOP_WORK_DIR", ps.DESKTOP_WORK_DIR
        )
        ps.DESKTOP_WORK_DIR = self.work_dir_patch

        usb_path = str(launcher_target)
        usb_raw = launcher_target.read_bytes()
        probe_raw = probe_target.read_bytes()
        adb_path = Path("/usr/lib/android-sdk/platform-tools/adb")
        adb_raw = adb_path.read_bytes()
        mono_launcher = next(
            item
            for item in MONO_LAUNCH["required_components"]
            if item["component_id"] == MONO_LAUNCH["launcher_component_id"]
        )
        pins = ps.record_component_pins(BODY)
        self.desktop_pins = {
            "adb": {
                "bytes": len(adb_raw),
                "path": str(adb_path),
                "sha256": sha(adb_raw),
                "stat": synth_stat(len(adb_raw), 700),
            },
            "codec": ps._artifact_pin({**pins["cuda-tokenize"]}),
            "cuda_runtime": ps._artifact_pin({**pins["cuda-route.bin"]}),
            "model": {
                "bytes": 9001752960,
                "path": "/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf",
                "sha256": ps.MODEL_SHA256,
                "stat": {**synth_stat(9001752960, 60), "mode": 0o100644},
            },
            "mono_command": list(MONO_LAUNCH["command"]),
            "mono_launcher": {
                "bytes": mono_launcher["stat"]["size"],
                "path": mono_launcher["path"],
                "sha256": mono_launcher["sha256"],
                "stat": dict(mono_launcher["stat"]),
            },
            "nvidia_smi": {
                "bytes": 1181528,
                "path": "/usr/bin/nvidia-smi",
                "sha256": "a" * 64,
                "stat": synth_stat(1181528, 701),
            },
            "probe": {
                "bytes": len(probe_raw),
                "path": str(probe_target),
                "sha256": sha(probe_raw),
            },
            "python": {
                "bytes": 1000,
                "path": "/usr/bin/python3",
                "sha256": "b" * 64,
                "stat": synth_stat(1000, 702),
            },
            "quality_corpus": {
                "bytes": CONTRACT_V26["quality"]["corpus"]["bytes"],
                "path": str(S39 / "CP0_R1_MMLU64_CORPUS_V2_2.jsonl"),
                "sha256": CONTRACT_V26["quality"]["corpus"]["sha256"],
            },
            "ssh": {
                "bytes": 1000,
                "path": "/usr/bin/ssh",
                "sha256": "c" * 64,
                "stat": synth_stat(1000, 703),
            },
            "usb_launcher": {
                "bytes": len(usb_raw),
                "path": usb_path,
                "sha256": sha(usb_raw),
            },
        }
        self.closure = ps.build_closure_input(BODY, MONO_LAUNCH)
        self.phone_static = ps.build_phone_static(
            self.closure, CONTRACT_V24, self.desktop_pins
        )
        self.cuda_static = ps.build_cuda_static(
            self.closure, CONTRACT_V24, self.desktop_pins, self.phone_static
        )
        self.runtime_static = ps.build_runtime_static(
            self.closure, BODY, CONTRACT_V24
        )

    def write_canonical(self, name: str, value) -> Path:
        path = self.inputs_dir / name
        path.write_bytes(ps.canonical_bytes(value))
        return path

    def inventory(self):
        closure_path = self.write_canonical(
            "runtime-bundle-inventory.json", self.closure
        )
        topology_path = self.inputs_dir / "topology-receipt.json"
        if not topology_path.exists():
            shutil.copy(TOPOLOGY_PATH, topology_path)
        input_pins = {
            "candidate": self.pin(S39 / "CP0_R1_CANDIDATE.json"),
            "contract": self.pin(V24 / "CP0_R1_EVIDENCE_CONTRACT_V2_4.json"),
            "cuda_monolithic_launch": self.pin(
                V24
                / "results/prephase_20260726T0915Z/cuda-monolithic-launch.json"
            ),
            "operator_input": {},
            "runtime_bundle_inventory": self.pin(closure_path),
            "token_history": self.pin(
                V24 / "results/prephase_20260726T0915Z/token-history.json"
            ),
            "tokenizer_plan": self.pin(
                V24 / "results/prephase_20260726T0915Z/tokenizer-plan.json"
            ),
            "topology_receipt": self.pin(topology_path),
        }
        operator = ps.build_operator_input(
            input_pins, self.desktop_pins, CONTRACT_V24
        )
        operator["directories"]["cuda_bundle_root"] = prod.CUDA_ROUTE_ROOT
        operator["directories"]["joint_cwd"] = str(self.joint_cwd)
        operator_path = self.write_canonical("operator-input.json", operator)
        input_pins["operator_input"] = self.pin(operator_path)
        inventory = ps.build_desktop_inventory(
            input_pins,
            operator,
            {
                "cuda_route": self.cuda_static,
                "phone_route": self.phone_static,
                "runtime": self.runtime_static,
            },
            CONTRACT_V24,
        )
        inventory["static"]["joint_cwd"] = str(self.joint_cwd)
        return inventory

    @staticmethod
    def pin(path: Path) -> dict:
        raw = path.read_bytes()
        return {"bytes": len(raw), "path": str(path), "sha256": sha(raw)}

    def run_mat(self, inventory) -> Path:
        inventory_path = self.write_canonical(
            "desktop-inventory.json", inventory
        )
        driver = load(
            "driver_under_test",
            self.mirror / ps.DRIVER_RELATIVE,
        )
        mat = driver.load_module(
            "mat_under_test",
            self.mirror
            / "v24_readiness/desktop_deployment_v1"
            / "materialize_a_only_inputs_v1.py",
        )
        spec = mat.build_spec(
            inventory_path, sha(inventory_path.read_bytes())
        )
        driver.validate_snapshot_processes(mat, spec["phone_route_static"])
        mat.write_new(
            self.inputs_dir / "prospective-runtime-spec.json", spec
        )
        originator = mat._load_originator()
        originator.materialize(
            spec_path=self.inputs_dir / "prospective-runtime-spec.json",
            output_paths={
                "cuda_route_launch": self.prephase / "cuda-route-launch.json",
                "joint_capture_plan": (
                    self.prephase / "joint-capture-plan.json"
                ),
                "phone_route_launch": (
                    self.prephase / "phone-route-launch.json"
                ),
                "runtime_plan": self.prephase / "runtime-bundle-plan.json",
            },
            prospective_root_output=(
                self.prephase / "prospective-runtime-root.json"
            ),
            report_output=self.inputs_dir / "dry-run-report.json",
        )
        return self.prephase


class BuilderTests(unittest.TestCase):
    def test_closure_input_shape(self):
        fixture = Fixture(self)
        closure = fixture.closure
        self.assertEqual(len(closure["components"]), 34)
        self.assertTrue(closure["closure_complete"])
        self.assertEqual(
            closure["bundle_roots"]["cuda_monolithic"],
            MONO_LAUNCH["bundle_root"],
        )
        mono_ids = [
            row["component_id"]
            for row in closure["components"]
            if row["bundle_id"] == "cuda_monolithic"
        ]
        self.assertEqual(
            sorted(mono_ids),
            sorted(
                item["component_id"]
                for item in MONO_LAUNCH["required_components"]
            ),
        )
        for row in closure["components"]:
            self.assertNotIn("build_id", row["stat"])

    def test_mechanism_matrix_shape(self):
        fixture = Fixture(self)
        matrix = fixture.phone_static["mechanism_commands"]
        self.assertEqual(len(matrix["desktop"]), 9)
        self.assertEqual(matrix["desktop"][1][0].endswith("llama-layersplit"), True)
        self.assertEqual(matrix["desktop"][8], MONO_LAUNCH["command"])
        self.assertEqual(len(matrix["op12"]), 3)
        self.assertEqual(len(matrix["op15"]), 4)
        self.assertEqual(
            fixture.cuda_static["mechanism_commands"], matrix
        )

    def test_phone_plans_have_no_ssh_key_and_no_boot_id(self):
        fixture = Fixture(self)
        for name, process in fixture.phone_static["processes"].items():
            self.assertEqual(len(process["argv"]), 5, name)
            plan = json.loads(process["argv"][2])
            self.assertNotIn("ssh", plan)
            self.assertNotIn("--boot-id", process["argv"])
            for component in plan["components"]:
                self.assertNotIn("build_id", component["stat"])

    def test_full_frozen_chain_accepts_adapter_inputs(self):
        fixture = Fixture(self)
        prephase = fixture.run_mat(fixture.inventory())
        for name in ps.PLAN_NAMES:
            self.assertTrue((prephase / name).exists(), name)
        result = ps.validate_plan_set(prephase)
        self.assertEqual(result["status"], "V24_PLAN_SET_VALIDATION_PASS")
        plan = json.loads((prephase / "runtime-bundle-plan.json").read_bytes())
        self.assertEqual(len(plan["components"]), 38)
        kinds = [row["kind"] for row in plan["capture_entrypoints"]]
        self.assertEqual(
            kinds,
            [
                "artifact_root",
                "cuda_monolithic",
                "fast_fresh_readiness",
                "joint_phone_cuda",
            ],
        )

    def test_live_wifi_address_in_static_is_rejected(self):
        fixture = Fixture(self)
        topology = json.loads(TOPOLOGY_PATH.read_bytes())
        live = topology["observed"]["phones"]["op12"]["wifi_ipv4"]
        fixture.phone_static["processes"]["op15_direct_relay"]["argv"][2] = (
            fixture.phone_static["processes"]["op15_direct_relay"]["argv"][2]
            .replace("0.0.0.12", live)
        )
        inventory = fixture.inventory()
        with self.assertRaisesRegex(Exception, "E_LIVE_IDENTITY|plan.sha256"):
            fixture.run_mat(inventory)

    def test_prebound_boot_id_is_rejected(self):
        fixture = Fixture(self)
        fixture.phone_static["processes"]["op12_stagenet"]["argv"] = (
            fixture.phone_static["processes"]["op12_stagenet"]["argv"]
            + ["--boot-id", "00000000-0000-0000-0000-000000000001"]
        )
        inventory = fixture.inventory()
        with self.assertRaisesRegex(
            Exception, "E_PREBOUND_BOOT_ID|mechanism"
        ):
            fixture.run_mat(inventory)

    def test_ssh_key_in_plan_is_rejected(self):
        fixture = Fixture(self)
        process = fixture.phone_static["processes"]["op15_stagenet"]
        plan = json.loads(process["argv"][2])
        plan["ssh"] = None
        raw = json.dumps(
            plan, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        process["argv"][2] = raw
        process["argv"][4] = sha(raw.encode("ascii"))
        inventory = fixture.inventory()
        with self.assertRaisesRegex(Exception, "E_KEYS|E_USB_LAUNCHER"):
            fixture.run_mat(inventory)

    def test_mono_component_tamper_is_rejected(self):
        fixture = Fixture(self)
        for row in fixture.closure["components"]:
            if row["component_id"] == "cuda-mono.bin":
                row["sha256"] = "f" * 64
        inventory = fixture.inventory()
        with self.assertRaisesRegex(Exception, "monolithic|cuda-mono"):
            fixture.run_mat(inventory)

    def test_short_mechanism_matrix_is_rejected(self):
        fixture = Fixture(self)
        matrix = fixture.phone_static["mechanism_commands"]
        matrix["desktop"] = matrix["desktop"][:8]
        fixture.cuda_static["mechanism_commands"] = matrix
        inventory = fixture.inventory()
        with self.assertRaisesRegex(Exception, "DESKTOP|desktop"):
            fixture.run_mat(inventory)

    def test_capture_component_tamper_is_rejected(self):
        fixture = Fixture(self)
        for row in fixture.runtime_static["components"]:
            if row["component_id"] == "capture.cuda_monolithic":
                row["sha256"] = "f" * 64
        inventory = fixture.inventory()
        with self.assertRaisesRegex(Exception, "capture"):
            fixture.run_mat(inventory)

    def test_duplicate_materialization_is_rejected(self):
        fixture = Fixture(self)
        fixture.run_mat(fixture.inventory())
        with self.assertRaisesRegex(Exception, "E_OUTPUT_EXISTS"):
            fixture.run_mat(fixture.inventory())

    def test_validate_rejects_byte_tamper(self):
        fixture = Fixture(self)
        prephase = fixture.run_mat(fixture.inventory())
        path = prephase / "runtime-bundle-plan.json"
        value = json.loads(path.read_bytes())
        value["phase"] = "B_ONLY"
        path.write_bytes(ps.canonical_bytes(value))
        with self.assertRaisesRegex(Exception, "runtime_plan.phase|E_VALUE"):
            ps.validate_plan_set(prephase)

    def test_validate_rejects_v23_reference(self):
        fixture = Fixture(self)
        prephase = fixture.run_mat(fixture.inventory())
        path = prephase / "prospective-runtime-root.json"
        value = json.loads(path.read_bytes())
        value["identity_placeholders"]["cuda"] = (
            "00000000-0000-0000-0000-000000000000"
        )
        value["model_id"] = "qwen3-14b-q4_k_m"
        value["schema"] = value["schema"]
        raw = ps.canonical_bytes(value).replace(
            b"POST_REBOOT_IDENTITY_BINDING_REQUIRED",
            b"POST_REBOOT_IDENTITY_BINDING_REQUIRED",
        )
        path.write_bytes(raw)
        tampered = json.loads(path.read_bytes())
        tampered["desktop_control"]["cuda_ssh_target"] = (
            "zhihao@172.20.74.85"
        )
        path.write_bytes(
            ps.canonical_bytes(tampered).replace(
                b"zhihao@172.20.74.85",
                b"v23_readiness_target",
                1,
            )
        )
        with self.assertRaisesRegex(Exception, "E_V23_REFERENCE|E_JSON|canonical"):
            ps.validate_plan_set(prephase)


class RemoteDriverTests(unittest.TestCase):
    class FakeRunner:
        def __init__(self):
            self.responses = {}
            self.calls = []

        def add(self, argv, stdout):
            self.responses[tuple(argv)] = stdout

        def run(self, argv, timeout):
            del timeout
            self.calls.append(list(argv))
            key = tuple(argv)
            if key not in self.responses:
                raise prod.inv.InventoryError(f"E_FAKE_COMMAND: {argv!r}")
            return self.responses[key]

    def test_mirror_conflict_is_rejected(self):
        runner = self.FakeRunner()
        capture = prod.Capture(runner, itertools.count(100, 3).__next__)
        relative = "CP0_R1_CANDIDATE.json"
        remote = f"{ps.MIRROR_ROOT}/{relative}"
        import shlex

        quoted = shlex.quote(remote)
        runner.add(
            prod.ssh_argv(
                f"if test -e {quoted}; then sha256sum -- {quoted};"
                f" else echo ABSENT; fi"
            ),
            ("f" * 64 + "  " + remote + "\n").encode("ascii"),
        )
        original = ps.MIRROR_FILES
        ps.MIRROR_FILES = (relative,)
        try:
            with self.assertRaisesRegex(
                ps.PlanSetError, "E_MIRROR_CONFLICT"
            ):
                ps.sync_mirror(capture)
        finally:
            ps.MIRROR_FILES = original

    def test_push_input_refuses_existing_remote(self):
        runner = self.FakeRunner()
        capture = prod.Capture(runner, itertools.count(100, 3).__next__)
        import shlex

        with tempfile.TemporaryDirectory() as temp:
            local = Path(temp) / "input.json"
            local.write_bytes(b"{}\n")
            remote = f"{ps.INPUTS_ROOT}/input.json"
            runner.add(
                prod.ssh_argv(
                    "mkdir -p -- "
                    + shlex.quote(str(Path(remote).parent))
                ),
                b"",
            )
            runner.add(
                prod.ssh_argv(
                    f"if test -e {shlex.quote(remote)}; then echo PRESENT;"
                    f" else echo ABSENT; fi"
                ),
                b"PRESENT\n",
            )
            with self.assertRaisesRegex(ps.PlanSetError, "E_INPUT_EXISTS"):
                ps.push_input(capture, local, remote, "input.json")


if __name__ == "__main__":
    unittest.main()
