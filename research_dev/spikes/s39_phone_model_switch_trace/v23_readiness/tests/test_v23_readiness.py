import copy
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
S39 = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(S39))
sys.path.insert(0, str(S39 / "tests"))

import cp0_r1_evidence_v22 as v22
import v23_common as common
import cp0_r1_evidence_v23 as v23
from test_cp0_r1_evidence_v22 import V22PhaseFixture


class ReadinessFixture:
    def __init__(self, test: unittest.TestCase):
        self.temporary = tempfile.TemporaryDirectory()
        test.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (
            self.v22_contract,
            self.v22_contract_raw,
            self.candidate,
            self.candidate_raw,
            self.parent,
            self.corpus,
        ) = v22.validate_inputs(
            S39 / "CP0_R1_EVIDENCE_CONTRACT_V2_2.json",
            S39 / "CP0_R1_CANDIDATE.json",
        )
        (
            self.contract,
            _,
            candidate,
            _,
        ) = v23.validate_inputs(
            HERE / "CP0_R1_EVIDENCE_CONTRACT_V2_3.json",
            S39 / "CP0_R1_CANDIDATE.json",
        )
        common.exact(candidate, self.candidate, "candidate")
        self.phase = V22PhaseFixture(
            self.root / "bundle",
            "A_ONLY",
            self.v22_contract,
            self.v22_contract_raw,
            self.parent,
            self.candidate,
            self.candidate_raw,
            [],
            [],
            frozen_corpus=self.corpus,
        )
        transfer_role = (
            f"model.{self.candidate['models'][0]['model_id']}.route_transfer"
        )
        self.direct_payload_bytes = sum(
            row["payload_bytes"]
            for row in self.phase.rows[transfer_role]
            if row["kind"] == "transfer"
        )
        self.manifest, _ = v23.load_v22_manifest(self.phase.root)
        self.model = self.candidate["models"][0]
        self.route = self.phase.lock
        self.paths = {
            "cuda": self.route["cuda_model_path"],
            "op15": self.route["op15_shard_path"],
            "op12": self.route["op12_shard_path"],
            "op15_worker": "/data/local/tmp/s39-v23/llama-layersplit",
            "op12_worker": "/data/local/tmp/s39-v23/llama-layersplit",
        }
        self.worker_digests = {
            "op15_worker": "1" * 64,
            "op12_worker": "2" * 64,
        }
        self.stats = {
            endpoint: {
                "ctime_ns": 100_000_000_000 + index,
                "device_id": index + 1,
                "inode": 1000 + index,
                "mode": 33188,
                "mtime_ns": 90_000_000_000 + index,
                "size": (
                    self.model["artifact"]["bytes"]
                    if endpoint == "cuda"
                    else 50_000_000
                    if endpoint.endswith("_worker")
                    else self.route[f"{endpoint}_shard_bytes"]
                ),
            }
            for index, endpoint in enumerate(
                ("cuda", "op15", "op12", "op15_worker", "op12_worker")
            )
        }
        self.artifact = self.make_artifact()
        self.artifact_raw = common.canonical_bytes(self.artifact)
        phase_lock_sha = next(
            item["sha256"]
            for item in self.manifest["artifacts"]
            if item["role"] == "phase.lock"
        )
        self.lock = {
            "artifact_snapshot_sha256": common.sha256_bytes(self.artifact_raw),
            "event_ns": self.phase.opened_ns + 400,
            "phase": "A_ONLY",
            "phase_id": self.phase.phase_id,
            "schema": "s39-cp0-r1-readiness-lock-v2.3",
            "v2_2_phase_lock_sha256": phase_lock_sha,
        }
        self.lock_raw = common.canonical_bytes(self.lock)
        self.fresh = self.make_fresh()
        self.fresh_raw = common.canonical_bytes(self.fresh)
        self.runtime = self.make_runtime()
        self.write()

    def make_artifact(self):
        artifacts = []
        for endpoint in ("cuda", "op15", "op12", "op15_worker", "op12_worker"):
            if endpoint == "cuda":
                size = self.model["artifact"]["bytes"]
                digest = self.model["artifact"]["sha256"]
            elif endpoint.endswith("_worker"):
                size = self.stats[endpoint]["size"]
                digest = self.worker_digests[endpoint]
            else:
                size = self.route[f"{endpoint}_shard_bytes"]
                digest = self.route[f"{endpoint}_shard_sha256"]
            artifacts.append({
                "bytes": size,
                "endpoint": endpoint,
                "path": self.paths[endpoint],
                "sha256": digest,
                "stat": copy.deepcopy(self.stats[endpoint]),
            })
        return {
            "artifacts": artifacts,
            "completed_ns": self.phase.opened_ns + 200,
            "model_id": self.model["model_id"],
            "phase": "A_ONLY",
            "route_lock_sha256": next(
                item["sha256"]
                for item in self.manifest["artifacts"]
                if item["role"] == f"model.{self.model['model_id']}.route_lock"
            ),
            "schema": "s39-cp0-r1-artifact-snapshot-v2.3",
            "slot": "A",
            "started_ns": self.phase.opened_ns + 100,
        }

    def make_fresh(self):
        return {
            "artifact_stats": [
                {
                    "endpoint": endpoint,
                    "path": self.paths[endpoint],
                    "stat": copy.deepcopy(self.stats[endpoint]),
                }
                for endpoint in (
                    "cuda",
                    "op15",
                    "op12",
                    "op15_worker",
                    "op12_worker",
                )
            ],
            "completed_ns": self.phase.started_ns - 100,
            "cuda": {
                "host": "zhihao-Z690-C-ac",
                "host_boot_id": "11111111-1111-4111-8111-111111111111",
                "memory_total_bytes": 17175674880,
                "name": "NVIDIA GeForce RTX 4060 Ti",
                "pci_bus_id": "00000000:01:00.0",
                "uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
            },
            "phase": "A_ONLY",
            "phase_id": self.phase.phase_id,
            "phones": {
                "op15": self.phone_fresh(
                    "3C15AU002CL00000",
                    "CPH2749",
                    "OP611FL1",
                    "21111111-1111-4111-8111-111111111111",
                    "192.0.2.15",
                ),
                "op12": self.phone_fresh(
                    "5ae7a43d",
                    "CPH2583",
                    "OP595DL1",
                    "31111111-1111-4111-8111-111111111111",
                    "192.0.2.12",
                ),
            },
            "readiness_lock_sha256": common.sha256_bytes(self.lock_raw),
            "schema": "s39-cp0-r1-fresh-identity-v2.3",
            "started_ns": self.phase.started_ns - 200,
        }

    @staticmethod
    def phone_fresh(serial, model, device, boot_id, ipv4):
        return {
            "available_bytes": 2_000_000_000,
            "boot_id": boot_id,
            "device": device,
            "interfaces": {
                "wlan0": {
                    "ipv4": ipv4,
                    "rx_bytes": 1000,
                    "tx_bytes": 2000,
                }
            },
            "model": model,
            "product": model,
            "serial": serial,
            "swap_total_bytes": 0,
            "swap_used_bytes": 0,
            "thermal_status": 0,
        }

    def phone_runtime(self, phone, local, peer, pid):
        prefix = f"model.{self.model['model_id']}"
        digests = v23.manifest_digests(self.manifest)
        fresh = self.fresh["phones"][phone]
        return {
            "active_sequences_after_cleanup": 0,
            "available_bytes": 1_500_000_000,
            "boot_id": fresh["boot_id"],
            "direct_peer": {
                "interface": "wlan0",
                "local_ipv4": local,
                "peer_ipv4": peer,
                "socket_peer_observed": True,
            },
            "executor_id": f"PHONE_{phone.upper()}",
            "gpu_max_millic": 65000,
            "interface_after": {
                "interface": "wlan0",
                "rx_bytes": 1000 + self.direct_payload_bytes,
                "tx_bytes": 2000 + self.direct_payload_bytes,
            },
            "interface_before": {
                "interface": "wlan0",
                "rx_bytes": 1000,
                "tx_bytes": 2000,
            },
            "loaded_shard_path": self.paths[phone],
            "loaded_shard_sha256": self.route[f"{phone}_shard_sha256"],
            "loaded_shard_stat": copy.deepcopy(self.stats[phone]),
            "mechanics_sha256": digests[f"{prefix}.mechanics.phone"],
            "model_id": self.model["model_id"],
            "placement_sha256": digests[f"{prefix}.placement.{phone}"],
            "process_swap_bytes": 0,
            "route_epoch": 7,
            "route_transfer_sha256": digests[f"{prefix}.route_transfer"],
            "serial": fresh["serial"],
            "worker_model_sha256": self.model["artifact"]["sha256"],
            "worker_boot_nonce": "0123456789abcdef",
            "worker_executable_path": self.paths[f"{phone}_worker"],
            "worker_executable_sha256": self.worker_digests[f"{phone}_worker"],
            "worker_executable_stat": copy.deepcopy(
                self.stats[f"{phone}_worker"]
            ),
            "worker_pid": pid,
            "worker_start_ticks": 123456 + pid,
            "session_protocol_version": 2,
        }

    def make_runtime(self):
        return {
            "completed_ns": self.phase.started_ns + 2_000_000,
            "executors": [
                {
                    "artifact_path": self.paths["cuda"],
                    "artifact_sha256": self.model["artifact"]["sha256"],
                    "artifact_stat": copy.deepcopy(self.stats["cuda"]),
                    "executor_id": "GPU",
                    "gpu_uuid": self.fresh["cuda"]["uuid"],
                    "host_boot_id": self.fresh["cuda"]["host_boot_id"],
                    "model_id": self.model["model_id"],
                    "route_epoch": 7,
                },
                self.phone_runtime(
                    "op15",
                    "192.0.2.15",
                    "192.0.2.12",
                    101,
                ),
                self.phone_runtime(
                    "op12",
                    "192.0.2.12",
                    "192.0.2.15",
                    102,
                ),
            ],
            "fresh_snapshot_sha256": common.sha256_bytes(self.fresh_raw),
            "phase": "A_ONLY",
            "phase_id": self.phase.phase_id,
            "route_epoch": 7,
            "schema": "s39-cp0-r1-runtime-identity-v2.3",
            "started_ns": self.phase.started_ns + 1,
        }

    def write(self):
        self.artifact_raw = common.canonical_bytes(self.artifact)
        self.lock["artifact_snapshot_sha256"] = common.sha256_bytes(self.artifact_raw)
        self.lock_raw = common.canonical_bytes(self.lock)
        self.fresh["readiness_lock_sha256"] = common.sha256_bytes(self.lock_raw)
        self.fresh_raw = common.canonical_bytes(self.fresh)
        self.runtime["fresh_snapshot_sha256"] = common.sha256_bytes(self.fresh_raw)
        self.paths_out = {
            "artifact": self.root / "artifact.json",
            "lock": self.root / "lock.json",
            "fresh": self.root / "fresh.json",
            "runtime": self.root / "runtime.json",
        }
        self.paths_out["artifact"].write_bytes(self.artifact_raw)
        self.paths_out["lock"].write_bytes(self.lock_raw)
        self.paths_out["fresh"].write_bytes(self.fresh_raw)
        self.paths_out["runtime"].write_bytes(common.canonical_bytes(self.runtime))

    def validate(self):
        self.write()
        return v23.validate_readiness(
            self.contract,
            self.candidate,
            self.phase.root,
            self.paths_out["artifact"],
            self.paths_out["lock"],
            self.paths_out["fresh"],
            self.paths_out["runtime"],
        )

    def mutate_bundle_role(self, role):
        artifact = next(
            item for item in self.manifest["artifacts"] if item["role"] == role
        )
        path = self.phase.root / artifact["path"]
        path.write_bytes(path.read_bytes() + b" ")

    def mutate_manifest_digest(self, role):
        path = self.phase.root / v22.MANIFEST_NAME
        manifest, _ = common.read_canonical(path)
        artifact = next(
            item for item in manifest["artifacts"] if item["role"] == role
        )
        artifact["sha256"] = "0" * 64
        path.write_bytes(common.canonical_bytes(manifest))


class V23ReadinessTests(unittest.TestCase):
    def fixture(self):
        return ReadinessFixture(self)

    def test_contract_separates_wifi_selectors_from_physical_serials(self):
        contract = v23.builder.build_contract()
        phones = contract["readiness_v2_3"]["phone_identity"]
        self.assertEqual(phones["op15"]["adb_selector"], "172.20.173.218:5555")
        self.assertEqual(phones["op15"]["serial"], "3C15AU002CL00000")
        self.assertEqual(phones["op12"]["adb_selector"], "172.20.59.72:5555")
        self.assertEqual(phones["op12"]["serial"], "5ae7a43d")

    def test_contract_rejects_missing_or_swapped_wifi_selector(self):
        contract = v23.builder.build_contract()
        mutations = []
        missing = copy.deepcopy(contract)
        del missing["readiness_v2_3"]["phone_identity"]["op15"]["adb_selector"]
        mutations.append(missing)
        swapped = copy.deepcopy(contract)
        identities = swapped["readiness_v2_3"]["phone_identity"]
        identities["op15"]["adb_selector"], identities["op12"]["adb_selector"] = (
            identities["op12"]["adb_selector"],
            identities["op15"]["adb_selector"],
        )
        mutations.append(swapped)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                path = self.fixture().root / f"mutated-contract-{index}.json"
                path.write_bytes(common.canonical_bytes(mutation))
                with self.assertRaisesRegex(common.ReadinessError, "contract"):
                    v23.validate_inputs(
                        path,
                        S39 / "CP0_R1_CANDIDATE.json",
                    )

    def test_valid_readiness_passes(self):
        result = self.fixture().validate()
        self.assertEqual(result["runtime"]["executor_count"], 3)
        self.assertRegex(result["v2_2_result_sha256"], r"^[0-9a-f]{64}$")

    def test_mutated_route_lock_bytes_are_rejected(self):
        fixture = self.fixture()
        fixture.mutate_bundle_role(
            f"model.{fixture.model['model_id']}.route_lock"
        )
        with self.assertRaisesRegex(v22.v2.EvidenceError, "bytes|sha256"):
            fixture.validate()

    def test_mutated_transfer_bytes_are_rejected(self):
        fixture = self.fixture()
        fixture.mutate_bundle_role(
            f"model.{fixture.model['model_id']}.route_transfer"
        )
        with self.assertRaisesRegex(v22.v2.EvidenceError, "bytes|sha256"):
            fixture.validate()

    def test_mutated_mechanics_bytes_are_rejected(self):
        fixture = self.fixture()
        fixture.mutate_bundle_role(
            f"model.{fixture.model['model_id']}.mechanics.phone"
        )
        with self.assertRaisesRegex(v22.v2.EvidenceError, "bytes|sha256"):
            fixture.validate()

    def test_mutated_placement_bytes_are_rejected(self):
        fixture = self.fixture()
        fixture.mutate_bundle_role(
            f"model.{fixture.model['model_id']}.placement.op15"
        )
        with self.assertRaisesRegex(v22.v2.EvidenceError, "bytes|sha256"):
            fixture.validate()

    def test_mutated_manifest_digest_is_rejected(self):
        fixture = self.fixture()
        fixture.mutate_manifest_digest(
            f"model.{fixture.model['model_id']}.route_transfer"
        )
        with self.assertRaisesRegex(v22.v2.EvidenceError, "sha256"):
            fixture.validate()

    def test_artifact_digest_mismatch_is_rejected(self):
        fixture = self.fixture()
        fixture.artifact["artifacts"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(common.ReadinessError, "ARTIFACT_SHA256"):
            fixture.validate()

    def test_changed_artifact_stat_is_rejected(self):
        fixture = self.fixture()
        fixture.fresh["artifact_stats"][0]["stat"]["mtime_ns"] += 1
        with self.assertRaisesRegex(common.ReadinessError, "ARTIFACT_CHANGED"):
            fixture.validate()

    def test_same_second_artifact_mutation_is_rejected(self):
        fixture = self.fixture()
        fixture.fresh["artifact_stats"][0]["stat"]["mtime_ns"] += 1
        self.assertEqual(
            fixture.fresh["artifact_stats"][0]["stat"]["mtime_ns"] // 1_000_000_000,
            fixture.artifact["artifacts"][0]["stat"]["mtime_ns"] // 1_000_000_000,
        )
        with self.assertRaisesRegex(common.ReadinessError, "ARTIFACT_CHANGED"):
            fixture.validate()

    def test_fresh_snapshot_before_lock_is_rejected(self):
        fixture = self.fixture()
        fixture.fresh["started_ns"] = fixture.lock["event_ns"] - 1
        with self.assertRaisesRegex(common.ReadinessError, "FRESH_PRELOCK"):
            fixture.validate()

    def test_stale_fresh_snapshot_is_rejected(self):
        fixture = self.fixture()
        age = fixture.contract["readiness_v2_3"][
            "fresh_snapshot_maximum_age_ns"
        ]
        fixture.fresh["completed_ns"] = fixture.phase.started_ns - age - 1
        fixture.fresh["started_ns"] = fixture.fresh["completed_ns"] - 1
        with self.assertRaisesRegex(common.ReadinessError, "FRESH_STALE"):
            fixture.validate()

    def test_wrong_gpu_is_rejected(self):
        fixture = self.fixture()
        fixture.fresh["cuda"]["uuid"] = "GPU-wrong"
        with self.assertRaisesRegex(common.ReadinessError, "CUDA_IDENTITY"):
            fixture.validate()

    def test_runtime_boot_change_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["executors"][1]["boot_id"] = (
            "41111111-1111-4111-8111-111111111111"
        )
        with self.assertRaisesRegex(common.ReadinessError, "RUNTIME_BOOT"):
            fixture.validate()

    def test_runtime_mechanics_mismatch_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["executors"][1]["mechanics_sha256"] = "0" * 64
        with self.assertRaisesRegex(common.ReadinessError, "MECHANICS_LINK"):
            fixture.validate()

    def test_runtime_gpu_artifact_mismatch_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["executors"][0]["artifact_sha256"] = "0" * 64
        with self.assertRaisesRegex(common.ReadinessError, "GPU_ARTIFACT"):
            fixture.validate()

    def test_runtime_shard_digest_mismatch_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["executors"][1]["loaded_shard_sha256"] = "0" * 64
        with self.assertRaisesRegex(common.ReadinessError, "SHARD_DIGEST"):
            fixture.validate()

    def test_runtime_worker_model_mismatch_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["executors"][1]["worker_model_sha256"] = "0" * 64
        with self.assertRaisesRegex(common.ReadinessError, "WORKER_MODEL"):
            fixture.validate()

    def test_runtime_worker_digest_mismatch_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["executors"][1]["worker_executable_sha256"] = "0" * 64
        with self.assertRaisesRegex(common.ReadinessError, "WORKER_DIGEST"):
            fixture.validate()

    def test_runtime_worker_stat_mismatch_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["executors"][1]["worker_executable_stat"][
            "mtime_ns"
        ] += 1
        with self.assertRaisesRegex(common.ReadinessError, "WORKER_STAT"):
            fixture.validate()

    def test_runtime_worker_start_ticks_are_required(self):
        fixture = self.fixture()
        fixture.runtime["executors"][1]["worker_start_ticks"] = 0
        with self.assertRaisesRegex(common.ReadinessError, "worker_start_ticks"):
            fixture.validate()

    def test_runtime_session_protocol_is_exact(self):
        fixture = self.fixture()
        fixture.runtime["executors"][1]["session_protocol_version"] = 1
        with self.assertRaisesRegex(common.ReadinessError, "SESSION_PROTOCOL"):
            fixture.validate()

    def test_cleanup_leak_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["executors"][2]["active_sequences_after_cleanup"] = 1
        with self.assertRaisesRegex(common.ReadinessError, "RUNTIME_CLEANUP"):
            fixture.validate()

    def test_low_memory_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["executors"][1]["available_bytes"] = 1
        with self.assertRaisesRegex(common.ReadinessError, "RUNTIME_HEADROOM"):
            fixture.validate()

    def test_process_swap_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["executors"][1]["process_swap_bytes"] = 4096
        with self.assertRaisesRegex(common.ReadinessError, "RUNTIME_PROCESS_SWAP"):
            fixture.validate()

    def test_wrong_direct_peer_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["executors"][1]["direct_peer"]["peer_ipv4"] = "192.0.2.99"
        with self.assertRaisesRegex(common.ReadinessError, "DIRECT_PEER"):
            fixture.validate()

    def test_forged_local_ip_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["executors"][1]["direct_peer"]["local_ipv4"] = "192.0.2.99"
        with self.assertRaisesRegex(common.ReadinessError, "FRESH_LOCAL_IP"):
            fixture.validate()

    def test_forged_interface_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["executors"][1]["interface_before"]["interface"] = "rmnet0"
        fixture.runtime["executors"][1]["interface_after"]["interface"] = "rmnet0"
        fixture.runtime["executors"][1]["direct_peer"]["interface"] = "rmnet0"
        with self.assertRaisesRegex(common.ReadinessError, "FRESH_INTERFACE"):
            fixture.validate()

    def test_transfer_digest_mismatch_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["executors"][1]["route_transfer_sha256"] = "0" * 64
        with self.assertRaisesRegex(common.ReadinessError, "TRANSFER_LINK"):
            fixture.validate()

    def test_interface_counter_reset_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["executors"][1]["interface_after"]["tx_bytes"] = 1
        with self.assertRaisesRegex(common.ReadinessError, "COUNTER_RESET"):
            fixture.validate()

    def test_interface_delta_below_payload_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["executors"][1]["interface_after"]["tx_bytes"] = (
            fixture.runtime["executors"][1]["interface_before"]["tx_bytes"]
            + fixture.direct_payload_bytes
            - 1
        )
        with self.assertRaisesRegex(
            common.ReadinessError,
            "INTERFACE_TRANSFER_BYTES",
        ):
            fixture.validate()

    def test_thermal_limit_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["executors"][1]["gpu_max_millic"] = 90000
        with self.assertRaisesRegex(common.ReadinessError, "RUNTIME_THERMAL"):
            fixture.validate()

    def test_pre_acquisition_runtime_is_rejected(self):
        fixture = self.fixture()
        fixture.runtime["started_ns"] = fixture.phase.started_ns - 1
        with self.assertRaisesRegex(common.ReadinessError, "RUNTIME_INTERVAL"):
            fixture.validate()


if __name__ == "__main__":
    unittest.main()
