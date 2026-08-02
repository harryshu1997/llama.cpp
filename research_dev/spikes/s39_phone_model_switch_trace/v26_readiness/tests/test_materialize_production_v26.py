#!/usr/bin/env python3

import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shlex
import tempfile
import unittest


HERE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prod = load("prod", HERE / "materialize_production_v26.py")

CONTRACT, _CONTRACT_RAW = prod.load_contract()
V24_CONTRACT = prod.load_v24_contract(CONTRACT)
TOPOLOGY, _TOPOLOGY_SHA = prod.load_topology(V24_CONTRACT)
MONO_LAUNCH, _MONO_SHA = prod.load_mono_launch()
MONO_PINS = prod.mono_component_pins(MONO_LAUNCH)

BOOTS = {
    "cuda": TOPOLOGY["observed"]["cuda"]["boot_id"],
    "op12": TOPOLOGY["observed"]["phones"]["op12"]["boot_id"],
    "op15": TOPOLOGY["observed"]["phones"]["op15"]["boot_id"],
}
WIFI = {
    "op12": TOPOLOGY["observed"]["phones"]["op12"]["wifi_ipv4"],
    "op15": TOPOLOGY["observed"]["phones"]["op15"]["wifi_ipv4"],
}
ENTRYPOINT_PINS = {
    "artifact_root_capture_v1.py": (
        CONTRACT["composition"]["capture_producers"]["artifact_root"]
    ),
    "cuda_monolithic_v1.py": (
        V24_CONTRACT["producer_requirements"]["source_programs"][
            "cuda_monolithic"
        ]
    ),
    "fast_fresh_capture_v1.py": (
        CONTRACT["composition"]["capture_producers"]["fast_fresh_readiness"]
    ),
    "joint_phone_cuda_v1.py": (
        V24_CONTRACT["producer_requirements"]["source_programs"][
            "joint_phone_cuda"
        ]
    ),
}
DESKTOP_ADB_SHA = "d" * 64
STAT_LINE = "DEV=64769|INO={inode}|SIZE={size}|MODE={mode}|MTIME=1785|CTIME=1785"


def sha256_text(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def build_world():
    world = {}
    inode = itertools.count(100)
    for bundle_id, table in prod.bundle_tables().items():
        root = prod.BUNDLE_ROOTS[bundle_id]
        files = {}
        for filename, (suffix, _role) in sorted(table.items()):
            component_id = suffix
            if bundle_id not in {"cuda_monolithic", "cuda_route"}:
                if not suffix.startswith(bundle_id):
                    component_id = f"{bundle_id}.{suffix}"
            if filename in ENTRYPOINT_PINS and bundle_id in {
                "cuda_monolithic",
                "cuda_route",
            }:
                size = ENTRYPOINT_PINS[filename]["bytes"]
                digest = ENTRYPOINT_PINS[filename]["sha256"]
            elif (
                bundle_id == "cuda_monolithic"
                and component_id in MONO_PINS
            ):
                size = MONO_PINS[component_id]["bytes"]
                digest = MONO_PINS[component_id]["sha256"]
            else:
                size = 1000 + len(files)
                digest = sha256_text(f"{bundle_id}/{filename}")
            files[filename] = {
                "bytes": size,
                "path": f"{root}/{filename}",
                "sha256": digest,
                "stat": {
                    "build_id": None,
                    "ctime_ns": 1785000000000000000,
                    "device_id": 64769,
                    "inode": next(inode),
                    "mode": 0o100755,
                    "mtime_ns": 1785000000000000000,
                    "size": size,
                },
            }
        world[root] = files
    return world


class FakeObserver:
    def __init__(self, world, launchers):
        self.world = world
        self.launchers = launchers
        self.visits = {}
        self.mutate_after_first = None

    def observe(self, *, endpoint, root, files):
        del endpoint
        known = self.world[root]
        expected = sorted(item["filename"] for item in files)
        if sorted(known) != expected:
            raise prod.inv.InventoryError(f"E_FAKE_CLOSURE: {root}")
        self.visits[root] = self.visits.get(root, 0) + 1
        rows = []
        for item in files:
            pin = copy.deepcopy(known[item["filename"]])
            if (
                self.mutate_after_first is not None
                and self.visits[root] > 1
                and (root, item["filename"]) == self.mutate_after_first
            ):
                pin["sha256"] = "e" * 64
            rows.append(pin)
        return rows

    def pin_launcher(self, path):
        if path not in self.launchers:
            raise prod.inv.InventoryError(f"E_FAKE_LAUNCHER: {path}")
        return copy.deepcopy(self.launchers[path])


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


def stat_stdout(size, mode="81ed", inode=7):
    return (
        STAT_LINE.format(inode=inode, size=size, mode=mode).encode("ascii")
        + b"\n"
    )


class Fixture:
    def __init__(self):
        self.world = build_world()
        launcher_stat = {
            "build_id": None,
            "ctime_ns": 1785000000000000000,
            "device_id": 64769,
            "inode": 900,
            "mode": 0o100755,
            "mtime_ns": 1785000000000000000,
            "size": 8358,
        }
        self.launchers = {
            prod.DESKTOP_USB_LAUNCHER: {
                "bytes": 8358,
                "path": prod.DESKTOP_USB_LAUNCHER,
                "sha256": prod.USB_LAUNCHER_SHA256,
                "stat": dict(launcher_stat),
            },
            prod.DESKTOP_FROZEN_LAUNCHER: {
                "bytes": 98969,
                "path": prod.DESKTOP_FROZEN_LAUNCHER,
                "sha256": prod.FROZEN_LAUNCHER_SHA256,
                "stat": {**launcher_stat, "inode": 901, "size": 98969},
            },
        }
        self.observer = FakeObserver(self.world, self.launchers)
        self.runner = FakeRunner()
        self.boots = dict(BOOTS)
        self.wifi = dict(WIFI)
        self.shards = {
            endpoint: CONTRACT["model_route_lock"]["geometry"]["known_shards"][
                endpoint
            ]
            for endpoint in ("op12", "op15")
        }
        self.model = CONTRACT["candidate_lock"]["model"]["artifact"]
        self.install_responses()

    def install_responses(self):
        contract = CONTRACT
        devices_lines = ["List of devices attached"]
        for endpoint in ("op15", "op12"):
            row = contract["devices"][endpoint]
            devices_lines.append(
                f"{row['serial']}       device usb:1-1"
                f" product:{row['product']} model:{row['model']}"
                f" device:{row['device']} transport_id:1"
            )
        self.runner.add(
            [prod.LOCAL_ADB_PATH, "-P", "5038", "devices", "-l"],
            ("\n".join(devices_lines) + "\n\n").encode("ascii"),
        )
        for endpoint in ("op12", "op15"):
            serial = contract["devices"][endpoint]["serial"]
            self.runner.add(
                prod.adb_argv(
                    serial, "shell", "cat /proc/sys/kernel/random/boot_id"
                ),
                (self.boots[endpoint] + "\n").encode("ascii"),
            )
            for prop, key in (
                ("ro.product.name", "product"),
                ("ro.product.model", "model"),
                ("ro.product.device", "device"),
            ):
                self.runner.add(
                    prod.adb_argv(serial, "shell", f"getprop {prop}"),
                    (contract["devices"][endpoint][key] + "\n").encode("ascii"),
                )
            self.runner.add(
                prod.adb_argv(
                    serial,
                    "shell",
                    "ip -4 -o addr show wlan0 | head -n 1"
                    " | tr -s ' ' | cut -d ' ' -f 4",
                ),
                (self.wifi[endpoint] + "/24\n").encode("ascii"),
            )
            shard = self.shards[endpoint]
            stat_script = "sh -c " + shlex.quote(
                prod._remote_stat_script(prod.SHARD_PATH)
            )
            self.runner.add(
                prod.adb_argv(serial, "shell", stat_script),
                stat_stdout(shard["bytes"], mode="81a4", inode=50),
            )
            self.runner.add(
                prod.adb_argv(
                    serial,
                    "shell",
                    "sha256sum -- " + shlex.quote(prod.SHARD_PATH),
                ),
                f"{shard['sha256']}  {prod.SHARD_PATH}\n".encode("ascii"),
            )
        cuda = contract["devices"]["cuda"]
        self.runner.add(
            prod.ssh_argv("hostname"),
            (cuda["host"] + "\n").encode("ascii"),
        )
        self.runner.add(
            prod.ssh_argv("cat /proc/sys/kernel/random/boot_id"),
            (self.boots["cuda"] + "\n").encode("ascii"),
        )
        memory_mib = cuda["memory_total_bytes"] // (1024 * 1024)
        self.runner.add(
            prod.ssh_argv(
                "nvidia-smi --id " + shlex.quote(cuda["uuid"])
                + " --query-gpu=uuid,name,memory.total"
                + " --format=csv,noheader,nounits"
            ),
            f"{cuda['uuid']}, {cuda['name']}, {memory_mib}\n".encode("ascii"),
        )
        self.runner.add(
            prod.ssh_argv(prod._remote_stat_script(prod.DESKTOP_MODEL_PATH)),
            stat_stdout(self.model["bytes"], mode="81a4", inode=60),
        )
        self.runner.add(
            prod.ssh_argv(
                "sha256sum -- " + shlex.quote(prod.DESKTOP_MODEL_PATH)
            ),
            f"{self.model['sha256']}  {prod.DESKTOP_MODEL_PATH}\n".encode(
                "ascii"
            ),
        )
        self.runner.add(
            prod.ssh_argv(prod._remote_stat_script(prod.DESKTOP_ADB_PATH)),
            stat_stdout(6795936, mode="81ed", inode=70),
        )
        self.runner.add(
            prod.ssh_argv(
                "sha256sum -- " + shlex.quote(prod.DESKTOP_ADB_PATH)
            ),
            f"{DESKTOP_ADB_SHA}  {prod.DESKTOP_ADB_PATH}\n".encode("ascii"),
        )
        for label, remote, digest in (
            ("usb", prod.DESKTOP_USB_LAUNCHER, prod.USB_LAUNCHER_SHA256),
            (
                "frozen",
                prod.DESKTOP_FROZEN_LAUNCHER,
                prod.FROZEN_LAUNCHER_SHA256,
            ),
        ):
            quoted = shlex.quote(remote)
            self.runner.add(
                prod.ssh_argv(
                    f"if test -e {quoted}; then echo PRESENT;"
                    f" else echo ABSENT; fi"
                ),
                b"PRESENT\n",
            )
            self.runner.add(
                prod.ssh_argv(
                    "; ".join(
                        (
                            "set -eu",
                            f"test -f {quoted}",
                            f"test ! -L {quoted}",
                            f"test -x {quoted}",
                            f"sha256sum -- {quoted}",
                        )
                    )
                ),
                f"{digest}  {remote}\n".encode("ascii"),
            )

    def materialize(self, output_root, suffix="test0001"):
        return prod.materialize(
            confirmation=prod.CONFIRMATION,
            output_root=output_root,
            runner=self.runner,
            local_observer=self.observer,
            android_observer=self.observer,
            clock_ns=itertools.count(1000, 7).__next__,
            wall_ns=lambda: 1785000000000000000,
            phase_suffix=suffix,
        )


def rewrite_consistent(root, mutate):
    """Apply mutate() to the published records, then repair every internal
    digest an attacker could also recompute (inventory self-digest, lock
    digests, materialization record, manifest)."""

    names = list(prod.PUBLISHED_FILES)
    values = {
        name: json.loads((root / name).read_bytes()) for name in names
    }
    mutate(values)
    spec_raw = prod.canonical_bytes(values["RUNTIME_INVENTORY_SPEC_V2_6.json"])
    inventory = values["RUNTIME_INVENTORY_V2_6.json"]
    inventory["inventory"]["spec_sha256"] = prod.sha256(spec_raw)
    inventory["inventory_sha256"] = prod.sha256(
        prod.canonical_bytes(inventory["inventory"])
    )
    inventory_raw = prod.canonical_bytes(inventory)
    lock = values["PHASE_LOCK_V2_6.json"]
    lock["spec_sha256"] = prod.sha256(spec_raw)
    lock["inventory_sha256"] = prod.sha256(inventory_raw)
    lock_raw = prod.canonical_bytes(lock)
    record = values["PRODUCTION_MATERIALIZATION_V2_6.json"]
    record["spec_sha256"] = lock["spec_sha256"]
    record["inventory_sha256"] = lock["inventory_sha256"]
    record["lock_sha256"] = prod.sha256(lock_raw)
    record_raw = prod.canonical_bytes(record)
    raws = {
        "RUNTIME_INVENTORY_SPEC_V2_6.json": spec_raw,
        "RUNTIME_INVENTORY_V2_6.json": inventory_raw,
        "PHASE_LOCK_V2_6.json": lock_raw,
        "PRODUCTION_MATERIALIZATION_V2_6.json": record_raw,
    }
    manifest = "".join(
        f"{prod.sha256(raws[name])}  {name}\n" for name in names
    )
    for name in names:
        (root / name).write_bytes(raws[name])
    (root / prod.MANIFEST_NAME).write_bytes(manifest.encode("ascii"))


class MaterializeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "publication"
        self.fixture = Fixture()

    def tearDown(self):
        self.temp.cleanup()

    def refuse(self, code, suffix="test0001"):
        with self.assertRaisesRegex(
            (prod.ProductionError, prod.inv.InventoryError), code
        ):
            self.fixture.materialize(self.root, suffix)
        self.assertFalse(self.root.exists())

    def test_materialize_and_validate_pass(self):
        result = self.fixture.materialize(self.root)
        self.assertEqual(
            result["validation"]["status"],
            "V2_6_PRODUCTION_MATERIALIZATION_PASS",
        )
        self.assertTrue(
            result["phase_id"].startswith("cp0-r1-v26-a-only-")
        )
        inventory = json.loads(
            (self.root / "RUNTIME_INVENTORY_V2_6.json").read_bytes()
        )
        self.assertEqual(
            len(inventory["inventory"]["managed_processes"]), 4
        )
        self.assertEqual(
            result["validation"]["runtime_component_count"], 40
        )

    def test_wrong_confirmation_is_rejected(self):
        with self.assertRaisesRegex(prod.ProductionError, "confirmation"):
            prod.materialize(
                confirmation="RUN_SOMETHING_ELSE",
                output_root=self.root,
                runner=self.fixture.runner,
                local_observer=self.fixture.observer,
                android_observer=self.fixture.observer,
                clock_ns=itertools.count(1000, 7).__next__,
                wall_ns=lambda: 1,
                phase_suffix="test0001",
            )

    def test_invalid_phase_suffix_is_rejected(self):
        self.refuse("E_PHASE_ID", suffix="bad suffix")

    def test_altered_phone_boot_id_is_rejected(self):
        serial = CONTRACT["devices"]["op12"]["serial"]
        self.fixture.runner.add(
            prod.adb_argv(
                serial, "shell", "cat /proc/sys/kernel/random/boot_id"
            ),
            b"11111111-2222-4333-8444-555555555555\n",
        )
        self.refuse("op12.boot_id.topology")

    def test_altered_desktop_boot_id_is_rejected(self):
        self.fixture.runner.add(
            prod.ssh_argv("cat /proc/sys/kernel/random/boot_id"),
            b"11111111-2222-4333-8444-555555555555\n",
        )
        self.refuse("cuda.boot_id.topology")

    def test_wrong_serial_identity_is_rejected(self):
        serial = CONTRACT["devices"]["op15"]["serial"]
        self.fixture.runner.add(
            prod.adb_argv(serial, "shell", "getprop ro.product.model"),
            b"CPH0000\n",
        )
        self.refuse("op15.model")

    def test_changed_wifi_address_is_rejected(self):
        serial = CONTRACT["devices"]["op15"]["serial"]
        self.fixture.runner.add(
            prod.adb_argv(
                serial,
                "shell",
                "ip -4 -o addr show wlan0 | head -n 1"
                " | tr -s ' ' | cut -d ' ' -f 4",
            ),
            b"10.9.9.9/24\n",
        )
        self.refuse("op15.wifi_ipv4.topology")

    def test_altered_shard_hash_is_rejected(self):
        serial = CONTRACT["devices"]["op12"]["serial"]
        self.fixture.runner.add(
            prod.adb_argv(
                serial,
                "shell",
                "sha256sum -- " + shlex.quote(prod.SHARD_PATH),
            ),
            ("f" * 64 + "  " + prod.SHARD_PATH + "\n").encode("ascii"),
        )
        self.refuse("op12.shard.sha256")

    def test_altered_desktop_model_hash_is_rejected(self):
        self.fixture.runner.add(
            prod.ssh_argv(
                "sha256sum -- " + shlex.quote(prod.DESKTOP_MODEL_PATH)
            ),
            ("f" * 64 + "  " + prod.DESKTOP_MODEL_PATH + "\n").encode("ascii"),
        )
        self.refuse("cuda.model.sha256")

    def test_altered_capture_entrypoint_hash_is_rejected(self):
        root = prod.BUNDLE_ROOTS["cuda_monolithic"]
        entry = self.fixture.world[root]["artifact_root_capture_v1.py"]
        entry["sha256"] = "f" * 64
        self.refuse("entrypoint.artifact_root_capture_v1.py")

    def test_altered_mono_component_hash_is_rejected(self):
        root = prod.BUNDLE_ROOTS["cuda_monolithic"]
        entry = self.fixture.world[root]["llama-layersplit"]
        entry["sha256"] = "f" * 64
        self.refuse("mono.cuda-mono.bin")

    def test_missing_bundle_file_fails_closure(self):
        root = prod.BUNDLE_ROOTS["op15_stagenet"]
        del self.fixture.world[root]["libggml.so"]
        self.refuse("E_FAKE_CLOSURE")

    def test_component_mutation_between_passes_is_rejected(self):
        root = prod.BUNDLE_ROOTS["op12_stagenet"]
        self.fixture.observer.mutate_after_first = (root, "libllama.so")
        self.refuse("op12_stagenet.libllama.sha256")

    def test_wrong_remote_launcher_hash_is_rejected(self):
        quoted = shlex.quote(prod.DESKTOP_USB_LAUNCHER)
        self.fixture.runner.add(
            prod.ssh_argv(
                "; ".join(
                    (
                        "set -eu",
                        f"test -f {quoted}",
                        f"test ! -L {quoted}",
                        f"test -x {quoted}",
                        f"sha256sum -- {quoted}",
                    )
                )
            ),
            ("f" * 64 + "  " + prod.DESKTOP_USB_LAUNCHER + "\n").encode(
                "ascii"
            ),
        )
        self.refuse("launcher.remote.usb")

    def test_duplicate_publication_is_rejected(self):
        self.fixture.materialize(self.root)
        with self.assertRaisesRegex(prod.ProductionError, "E_OUTPUT_EXISTS"):
            self.fixture.materialize(self.root, "test0002")


class ValidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "publication"
        self.fixture = Fixture()
        self.fixture.materialize(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def refuse(self, code):
        with self.assertRaisesRegex(
            (prod.ProductionError, prod.inv.InventoryError), code
        ):
            prod.validate_production(self.root)

    def test_partial_output_missing_file_is_rejected(self):
        (self.root / "PHASE_LOCK_V2_6.json").unlink()
        self.refuse("PHASE_LOCK_V2_6.json")

    def test_partial_output_missing_manifest_is_rejected(self):
        (self.root / prod.MANIFEST_NAME).unlink()
        self.refuse("manifest")

    def test_extra_file_is_rejected(self):
        (self.root / "EXTRA.json").write_bytes(b"{}\n")
        self.refuse("root.extras")

    def test_byte_tamper_is_rejected_by_manifest(self):
        path = self.root / "PHASE_LOCK_V2_6.json"
        value = json.loads(path.read_bytes())
        value["event_ns"] += 1
        path.write_bytes(prod.canonical_bytes(value))
        self.refuse("manifest.PHASE_LOCK_V2_6.json")

    def test_consistent_phase_id_rewrite_is_rejected(self):
        def mutate(values):
            values["PHASE_LOCK_V2_6.json"]["phase_id"] = (
                "cp0-r1-v26-a-only-other"
            )
            values["PRODUCTION_MATERIALIZATION_V2_6.json"]["phase_id"] = (
                "cp0-r1-v26-a-only-other"
            )

        rewrite_consistent(self.root, mutate)
        self.refuse("lock.inventory_phase_id")

    def test_consistent_timestamp_rewrite_is_rejected(self):
        def mutate(values):
            lock = values["PHASE_LOCK_V2_6.json"]
            lock["event_ns"] = lock["started_ns"] - 1

        rewrite_consistent(self.root, mutate)
        self.refuse("E_LOCK_ORDER")

    def test_consistent_boot_id_rewrite_is_rejected(self):
        def mutate(values):
            lock = values["PHASE_LOCK_V2_6.json"]
            lock["device_boot_ids"]["op12"] = (
                "11111111-2222-4333-8444-555555555555"
            )

        rewrite_consistent(self.root, mutate)
        self.refuse("lock.bound_boot.op12_stagenet")

    def test_consistent_argv_rewrite_is_rejected(self):
        def tamper(process):
            expectation = process["expectation"]
            expectation["cuda_flags"]["--port"] = "39999"
            for side in ("prospective_route", "bound_route"):
                argv = expectation[side]["argv"]
                argv[argv.index("--port") + 1] = "39999"
            for side in ("prospective", "bound"):
                plan = json.loads(process[side]["argv"][2])
                argv = plan["route"]["argv"]
                argv[argv.index("--port") + 1] = "39999"
                raw = json.dumps(
                    plan,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                process[side]["argv"][2] = raw
                process[side]["argv"][4] = prod.sha256(raw.encode("ascii"))

        def mutate(values):
            gate = load("gate", HERE / "managed_plan_gate_v1.py")
            inventory = values["RUNTIME_INVENTORY_V2_6.json"]
            spec = values["RUNTIME_INVENTORY_SPEC_V2_6.json"]
            for process in spec["managed_processes"]:
                if process["bundle_id"] == "cuda_route":
                    tamper(process)
            for process in inventory["inventory"]["managed_processes"]:
                if process["bundle_id"] != "cuda_route":
                    continue
                tamper(process)
                process["gate_result"] = gate.validate_managed_plan_pair(
                    process["prospective"]["argv"],
                    process["bound"]["argv"],
                    expectation=process["expectation"],
                    prospective_boot_id=process["prospective"]["boot_id"],
                    bound_boot_id=process["bound"]["boot_id"],
                )

        rewrite_consistent(self.root, mutate)
        self.refuse("managed.recompute")

    def test_consistent_producer_rewrite_is_rejected(self):
        def mutate(values):
            values["PHASE_LOCK_V2_6.json"]["producer"]["sha256"] = "f" * 64

        rewrite_consistent(self.root, mutate)
        self.refuse("lock.producer.sha256")

    def test_stale_topology_digest_is_rejected(self):
        def mutate(values):
            values["PHASE_LOCK_V2_6.json"]["topology_receipt_sha256"] = (
                "f" * 64
            )

        rewrite_consistent(self.root, mutate)
        self.refuse("lock.topology_receipt_sha256")

    def test_live_check_passes_on_matching_boots(self):
        runner = FakeRunner()
        lock = json.loads(
            (self.root / "PHASE_LOCK_V2_6.json").read_bytes()
        )
        for endpoint in ("op12", "op15"):
            serial = lock["device_identities"][endpoint]["serial"]
            runner.add(
                prod.adb_argv(
                    serial, "shell", "cat /proc/sys/kernel/random/boot_id"
                ),
                (lock["device_boot_ids"][endpoint] + "\n").encode("ascii"),
            )
        runner.add(
            prod.ssh_argv("cat /proc/sys/kernel/random/boot_id"),
            (lock["device_boot_ids"]["cuda"] + "\n").encode("ascii"),
        )
        result = prod.live_check(
            self.root,
            runner=runner,
            clock_ns=itertools.count(5000, 3).__next__,
        )
        self.assertEqual(
            result["status"], "V2_6_PRODUCTION_BOOT_IDENTITY_LIVE"
        )

    def test_live_check_rejects_rebooted_device(self):
        runner = FakeRunner()
        lock = json.loads(
            (self.root / "PHASE_LOCK_V2_6.json").read_bytes()
        )
        for endpoint in ("op12", "op15"):
            serial = lock["device_identities"][endpoint]["serial"]
            value = lock["device_boot_ids"][endpoint]
            if endpoint == "op12":
                value = "11111111-2222-4333-8444-555555555555"
            runner.add(
                prod.adb_argv(
                    serial, "shell", "cat /proc/sys/kernel/random/boot_id"
                ),
                (value + "\n").encode("ascii"),
            )
        with self.assertRaisesRegex(
            prod.ProductionError, "live.op12.boot_id"
        ):
            prod.live_check(
                self.root,
                runner=runner,
                clock_ns=itertools.count(5000, 3).__next__,
            )


if __name__ == "__main__":
    unittest.main()
