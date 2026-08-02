import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


S39 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S39))
sys.path.insert(0, str(S39 / "tests"))

import build_cp0_r1_v21 as builder
import cp0_r1_evidence_v2 as v2
import cp0_r1_evidence_v21 as v21
from test_cp0_r1_evidence_v2 import BundleFixture as V2BundleFixture


def marker(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class PhaseFixture:
    def __init__(
        self,
        root,
        phase,
        contract,
        contract_raw,
        parent,
        candidate,
        candidate_raw,
        prior_results=None,
        prior_locks=None,
    ):
        self.root = root
        self.phase = phase
        self.contract = contract
        self.contract_raw = contract_raw
        self.parent = parent
        self.candidate = candidate
        self.candidate_raw = candidate_raw
        self.prior_results = prior_results or []
        self.prior_locks = prior_locks or []
        phase_index = {"A_ONLY": 0, "B_ONLY": 1, "PAIR": 2}[phase]
        self.phase_id = f"cp0-r1-v21-{phase.lower()}-test"
        self.opened_ns = 1_000_000_000 + phase_index * 20_000_000_000
        self.started_ns = self.opened_ns + 10_000_000_000
        self.closed_ns = self.started_ns + 5_000_000_000
        self.rows = {}
        self.artifact_digests = {}
        self.lock = None
        base_root = root / "v2-base"
        self.base = V2BundleFixture(
            base_root,
            parent,
            (S39 / "CP0_R1_EVIDENCE_CONTRACT_V2.json").read_bytes(),
            candidate,
            candidate_raw,
        )
        self._build()

    def common(self, role, kind, event_ns):
        return {
            "acquisition_id": self.phase_id,
            "event_ns": event_ns,
            "kind": kind,
            "phase": self.phase,
            "phase_id": self.phase_id,
            "role": role,
        }

    def add_phase_fields(self, rows, base_event):
        result = []
        for index, source in enumerate(rows):
            row = copy.deepcopy(source)
            row["acquisition_id"] = self.phase_id
            row["phase"] = self.phase
            row["phase_id"] = self.phase_id
            row["event_ns"] = base_event + index
            result.append(row)
        return result

    def build_model(self, slot):
        model = next(model for model in self.candidate["models"] if model["slot"] == slot)
        prefix = f"model.{model['model_id']}"
        geometry = self.contract["model_geometry"][model["model_id"]]

        route_role = f"{prefix}.route_lock"
        route = self.add_phase_fields(
            self.base.rows[route_role],
            self.opened_ns + 100,
        )
        route[0]["frozen_ns"] = route[0]["event_ns"]
        route[0].update(
            {
                "activation_dtype": geometry["activation_dtype"],
                "activation_element_bytes": geometry["activation_element_bytes"],
                "cuda_model_path": geometry["cuda_model_path"],
                "hidden_size": geometry["hidden_size"],
            }
        )
        known = geometry.get("known_shards")
        for phone in ("op15", "op12"):
            if known is not None:
                shard = known[phone]
            else:
                shard = {
                    "bytes": 4_100_000_000 if phone == "op15" else 3_300_000_000,
                    "path": geometry["planned_phone_path"],
                    "sha256": marker(f"{model['model_id']}-{phone}-shard"),
                }
            route[0][f"{phone}_shard_sha256"] = shard["sha256"]
            route[0][f"{phone}_shard_bytes"] = shard["bytes"]
            route[0][f"{phone}_shard_path"] = shard["path"]
        self.rows[route_role] = route
        self.lock = route[0]

        corpus_role = "quality.corpus"
        corpus = self.add_phase_fields(
            self.base.rows[corpus_role],
            self.opened_ns + 200,
        )
        self.rows[corpus_role] = corpus

        for suffix, offset in (
            ("mechanics.phone", 100_000),
            ("oracle.cuda_route", 200_000),
            ("oracle.cuda_monolithic", 300_000),
        ):
            role = f"{prefix}.{suffix}"
            rows = self.add_phase_fields(
                self.base.rows[role],
                self.started_ns + offset,
            )
            rows[0]["call_shapes"] = [
                {"call_index": 0, "n_seqs": 8, "n_tokens": 16, "phase": "prefill"},
                {"call_index": 1, "n_seqs": 8, "n_tokens": 8, "phase": "decode"},
                {"call_index": 2, "n_seqs": 8, "n_tokens": 8, "phase": "decode"},
            ]
            self.rows[role] = rows

        memory_role = f"{prefix}.cuda_memory"
        memory = self.add_phase_fields(
            self.base.rows[memory_role],
            self.started_ns + 10_000,
        )
        memory_times = [
            self.started_ns + 10_000,
            self.started_ns + 30_000,
            self.started_ns + 50_000,
        ]
        for index, row in enumerate(memory):
            row["event_ns"] = memory_times[index]
            row["timestamp_ns"] = memory_times[index]
            row["sample_id"] = f"{slot}-{row['kind']}"
            row["sampler_sha256"] = marker(f"sampler-{slot}")
            if row["kind"] == "ready":
                row["process_pid"] = 1000 + (0 if slot == "A" else 1)
                row["process_used_bytes"] = (
                    row["model_buffer_bytes"] + row["kv_buffer_bytes"]
                )
            else:
                row["process_pid"] = 0
                row["process_used_bytes"] = 0
            row["free_bytes"] = row["memory_total_bytes"] - row["used_bytes"]
        self.rows[memory_role] = memory

        corpus_raw = b"".join(v2.canonical_line(row) for row in corpus)
        corpus_sha = v2.sha256_bytes(corpus_raw)
        normalized_corpus = v21._normalize_rows(corpus)
        for suffix, offset in (("cuda", 400_000), ("phone", 500_000)):
            role = f"{prefix}.quality.{suffix}"
            rows = self.add_phase_fields(
                self.base.rows[role],
                self.started_ns + offset,
            )
            for index, row in enumerate(rows):
                row["corpus_sha256"] = corpus_sha
                row["corpus_item_sha256"] = v2.digest_json(normalized_corpus[index])
            self.rows[role] = rows

        phone_role = f"{prefix}.mechanics.phone"
        phone_execution = v2.validate_execution(
            v21._normalize_rows(self.rows[phone_role]),
            phone_role,
            self.phase_id,
            model,
            "PHONE_COLLECTIVE",
        )
        normalized_phone_requests = {
            row["request_id"]: row
            for row in v21._normalize_rows(self.rows[phone_role])
            if row["kind"] == "request"
        }
        ready_normalized = v21._normalize_rows(
            [memory[1]],
            {
                "ready": {
                    "process_pid",
                    "process_used_bytes",
                    "sample_id",
                    "sampler_sha256",
                }
            },
        )[0]
        bridge_role = f"{prefix}.bridge"
        bridge = self.add_phase_fields(
            self.base.rows[bridge_role],
            self.started_ns + 20_000,
        )
        bridge[0]["timestamp_ns"] = self.started_ns + 20_000
        bridge[0]["event_ns"] = bridge[0]["timestamp_ns"]
        for request_id, row in enumerate(bridge[1:-1]):
            request = phone_execution["requests"][request_id]
            row["timestamp_ns"] = self.started_ns + 21_000 + request_id
            row["event_ns"] = row["timestamp_ns"]
            row["token_ids"] = request["continuation_tokens"]
            row["phone_request_sha256"] = v2.digest_json(
                normalized_phone_requests[request_id]
            )
        bridge[-1]["timestamp_ns"] = memory[1]["timestamp_ns"]
        bridge[-1]["event_ns"] = bridge[-1]["timestamp_ns"]
        bridge[-1]["cuda_memory_ready_sha256"] = v2.digest_json(ready_normalized)
        self.rows[bridge_role] = bridge

        for phone, offset in (("op15", 600_000), ("op12", 700_000)):
            role = f"{prefix}.placement.{phone}"
            self.rows[role] = self.add_phase_fields(
                self.base.rows[role],
                self.started_ns + offset,
            )
            meta = self.rows[role][0]
            meta["shard_sha256"] = route[0][f"{phone}_shard_sha256"]
            meta["stored_layers"] = route[0][f"{phone}_stored_layers"]

        transfer_role = f"{prefix}.route_transfer"
        source = self.base.rows[transfer_role][0]
        meta = {
            **copy.deepcopy(source),
            "acquisition_id": self.phase_id,
            "event_ns": self.started_ns + 800_000,
            "phase": self.phase,
            "phase_id": self.phase_id,
        }
        transfers = [meta]
        for index, call in enumerate(phone_execution["call_shapes"]):
            payload = (
                call["n_tokens"]
                * route[0]["hidden_size"]
                * route[0]["activation_element_bytes"]
            )
            transfers.append(
                {
                    **self.common(
                        transfer_role,
                        "transfer",
                        self.started_ns + 800_001 + index,
                    ),
                    "call_index": index,
                    "host_payload_bytes": 0,
                    "path": "WIFI_TCP_DIRECT",
                    "payload_bytes": payload,
                    "payload_sha256": marker(f"{slot}-payload-{index}"),
                    "receiver": "op12",
                    "row_count": call["n_tokens"],
                    "sender": "op15",
                }
            )
        self.rows[transfer_role] = transfers

    def build_pair(self):
        pair_role = "pair.cuda_memory"
        pair = self.add_phase_fields(
            self.base.rows[pair_role],
            self.started_ns + 10_000,
        )
        prior_hashes = [v21._phase_result_sha256(result) for result in self.prior_results]
        for index, row in enumerate(pair):
            timestamp = self.started_ns + 10_000 + index
            row["event_ns"] = timestamp
            row["timestamp_ns"] = timestamp
            row["sample_id"] = f"pair-{row['kind']}"
            row["sampler_sha256"] = marker("pair-sampler")
            row["used_bytes"] = row["memory_total_bytes"] - row["free_bytes"]
            if row["kind"] == "attempt":
                row["model_phase_result_sha256s"] = prior_hashes
                row["process_pids"] = [2001, 2002]
                row["process_used_bytes"] = [8_800_000_000, 8_000_000_000]
        self.rows[pair_role] = pair

        for direction, offset, to_lock in (
            ("A_to_B", 100_000, self.prior_locks[1]),
            ("B_to_A", 200_000, self.prior_locks[0]),
        ):
            role = f"reprepare.{direction}"
            rows = self.add_phase_fields(
                self.base.rows[role],
                self.started_ns + offset,
            )
            start = self.started_ns + offset
            rows[0]["timestamp_ns"] = start
            rows[0]["event_ns"] = start
            before_by_phone = {}
            for index, row in enumerate(rows[1:3], start=1):
                row["event_ns"] = start + index
                phone = row["phone"]
                expected = to_lock[f"{phone}_shard_bytes"]
                row["local_shard_bytes"] = expected
                row["local_ufs_read_bytes"] = 1_000
                before_by_phone[phone] = row
            for index, row in enumerate(rows[3:5], start=3):
                phone = row["phone"]
                expected = to_lock[f"{phone}_shard_bytes"]
                received = start + 1_000_000 + index
                row["event_ns"] = received
                row["host_received_ns"] = received
                row["local_shard_bytes"] = expected
                row["verified_shard_bytes_read"] = expected
                row["local_ufs_read_bytes"] = (
                    before_by_phone[phone]["local_ufs_read_bytes"] + expected
                )
                row["shard_sha256"] = to_lock[f"{phone}_shard_sha256"]
            rows[-1]["timestamp_ns"] = start + 2_000_000
            rows[-1]["event_ns"] = rows[-1]["timestamp_ns"]
            self.rows[role] = rows

    def preflight_rows(self, model_locks):
        role = "phase.preflight"
        commands = v21.expected_preflight_commands(
            self.contract,
            model_locks,
        )
        rows = []
        event = self.started_ns - 10_000
        adb_stdout = (
            "List of devices attached\n"
            "3C15AU002CL00000 device usb:8-3 product:CPH2749 "
            "model:CPH2749 device:OP611FL1 transport_id:1\n"
            "5ae7a43d device usb:6-2 product:CPH2583 "
            "model:CPH2583 device:OP595DL1 transport_id:2\n"
        )
        lock_by_slot = {model["slot"]: lock for model, lock in model_locks}
        model_by_slot = {model["slot"]: model for model, _ in model_locks}
        for index, label in enumerate(sorted(commands)):
            stdout = ""
            if label == "adb_5038":
                stdout = adb_stdout
            elif label.startswith("cuda_"):
                slot = label[-1]
                model = model_by_slot[slot]
                cuda = self.contract["devices"]["cuda"]
                stdout = (
                    f"{cuda['host']}\n"
                    f"{cuda['name']}, {cuda['uuid']}, 16380\n"
                    f"MODEL_BYTES={model['artifact']['bytes']}\n"
                    f"MODEL_SHA256={model['artifact']['sha256']}\n"
                )
            elif label.startswith("op"):
                phone, slot = label.split("_")
                device = self.contract["devices"][phone]
                lock = lock_by_slot[slot]
                stdout = (
                    f"{device['model']}\n{device['product']}\n{device['device']}\n"
                    f"{'1' if phone == 'op15' else '2'}1111111-1111-4111-"
                    "8111-111111111111\n"
                    f"SHARD_BYTES={lock[f'{phone}_shard_bytes']}\n"
                    f"SHARD_SHA256={lock[f'{phone}_shard_sha256']}\n"
                )
            rows.append(
                {
                    **self.common(role, "probe", event + index),
                    "argv": commands[label],
                    "label": label,
                    "returncode": 0,
                    "started_ns": event + index - 1,
                    "stderr": "",
                    "stdout": stdout,
                    "timed_out": False,
                }
            )
        completed = event + len(commands)
        rows.append(
            {
                **self.common(role, "meta", completed),
                "completed_ns": completed,
                "forbidden_work_executed": False,
                "probe_labels": sorted(commands),
            }
        )
        return rows

    def _raw(self, role):
        return b"".join(v2.canonical_line(row) for row in self.rows[role])

    def build_phase_lock(self):
        route_sha = "NONE"
        corpus_sha = "NONE"
        slot = "PAIR"
        if self.phase in v21.PHASE_SLOT:
            slot = v21.PHASE_SLOT[self.phase]
            model = next(
                model for model in self.candidate["models"] if model["slot"] == slot
            )
            route_sha = v2.sha256_bytes(self._raw(f"model.{model['model_id']}.route_lock"))
            corpus_sha = v2.sha256_bytes(self._raw("quality.corpus"))
        prior_hashes = [v21._phase_result_sha256(result) for result in self.prior_results]
        self.rows["phase.lock"] = [
            {
                **self.common("phase.lock", "phase_lock", self.opened_ns + 300),
                "candidate_sha256": v2.sha256_bytes(self.candidate_raw),
                "clock_id": "HOST_MONOTONIC_RAW",
                "contract_sha256": v2.sha256_bytes(self.contract_raw),
                "model_slot": slot,
                "prior_phase_result_sha256s": prior_hashes,
                "quality_corpus_sha256": corpus_sha,
                "route_lock_sha256": route_sha,
            }
        ]

    def _build(self):
        if self.phase == "A_ONLY":
            self.build_model("A")
            model = self.candidate["models"][0]
            model_locks = [(model, self.lock)]
        elif self.phase == "B_ONLY":
            self.build_model("B")
            model = self.candidate["models"][1]
            model_locks = [(model, self.lock)]
        else:
            self.build_pair()
            model_locks = list(zip(self.candidate["models"], self.prior_locks))
        self.rows["phase.preflight"] = self.preflight_rows(model_locks)
        self.build_phase_lock()
        self.write()

    def write(self):
        raw_dir = self.root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for index, role in enumerate(
            self.contract["phase_protocol"]["phase_roles"][self.phase]
        ):
            raw = self._raw(role)
            path = f"raw/{index:02d}.jsonl"
            (self.root / path).write_bytes(raw)
            artifacts.append(
                {
                    "bytes": len(raw),
                    "format": "CANONICAL_ASCII_JSONL",
                    "path": path,
                    "role": role,
                    "sha256": v2.sha256_bytes(raw),
                }
            )
        manifest = {
            "acquisition_started_ns": self.started_ns,
            "artifacts": artifacts,
            "candidate_sha256": v2.sha256_bytes(self.candidate_raw),
            "clock_id": "HOST_MONOTONIC_RAW",
            "contract_sha256": v2.sha256_bytes(self.contract_raw),
            "phase": self.phase,
            "phase_closed_ns": self.closed_ns,
            "phase_id": self.phase_id,
            "phase_opened_ns": self.opened_ns,
            "schema": "s39-cp0-r1-evidence-bundle-v2.1",
        }
        (self.root / "EVIDENCE_BUNDLE.json").write_bytes(v2.canonical_bytes(manifest))

    def mutate(self, role, callback):
        callback(self.rows[role])
        if role in ("quality.corpus",) or role.endswith(".route_lock"):
            self.build_phase_lock()
        self.write()


class Cp0R1EvidenceV21Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract, cls.contract_raw, cls.candidate, cls.candidate_raw, cls.parent = (
            v21.validate_inputs(
                S39 / "CP0_R1_EVIDENCE_CONTRACT_V2_1.json",
                S39 / "CP0_R1_CANDIDATE.json",
            )
        )

    def temp_root(self, name):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name) / name

    def evaluate(self, fixture, prior=None):
        return v21.evaluate_root(
            fixture.root,
            "EVIDENCE_BUNDLE.json",
            self.contract,
            self.contract_raw,
            self.candidate,
            self.candidate_raw,
            self.parent,
            prior or [],
        )

    def chain(self):
        a = PhaseFixture(
            self.temp_root("a"),
            "A_ONLY",
            self.contract,
            self.contract_raw,
            self.parent,
            self.candidate,
            self.candidate_raw,
        )
        a_result = self.evaluate(a)
        b = PhaseFixture(
            self.temp_root("b"),
            "B_ONLY",
            self.contract,
            self.contract_raw,
            self.parent,
            self.candidate,
            self.candidate_raw,
            [a_result],
            [a.lock],
        )
        b_result = self.evaluate(b, [a_result])
        pair = PhaseFixture(
            self.temp_root("pair"),
            "PAIR",
            self.contract,
            self.contract_raw,
            self.parent,
            self.candidate,
            self.candidate_raw,
            [a_result, b_result],
            [a.lock, b.lock],
        )
        return a, a_result, b, b_result, pair

    def test_contract_is_byte_deterministic(self):
        self.assertEqual(
            (S39 / "CP0_R1_EVIDENCE_CONTRACT_V2_1.json").read_bytes(),
            builder.canonical_bytes(builder.build_contract()),
        )

    def test_a_only_passes_without_b_or_pair(self):
        a, a_result, _, _, _ = self.chain()
        self.assertEqual(a_result["status"], "MODEL_A_QUALIFICATION_PASS")
        self.assertNotEqual(a_result["status"], "TWO_ROUTE_ELIGIBILITY_PASS")
        self.assertEqual(a_result["derived"]["model"]["quality"]["phone_correct"], 64)
        self.assertTrue(a.root.exists())

    def test_b_requires_rederived_a(self):
        a, a_result, b, _, _ = self.chain()
        with self.assertRaisesRegex(v2.EvidenceError, "B requires A"):
            self.evaluate(b)
        self.assertEqual(self.evaluate(b, [a_result])["status"], "MODEL_B_QUALIFICATION_PASS")

    def test_complete_phase_chain_passes(self):
        _, a_result, _, b_result, pair = self.chain()
        result = self.evaluate(pair, [a_result, b_result])
        self.assertEqual(result["status"], "TWO_ROUTE_ELIGIBILITY_PASS")
        self.assertGreater(result["derived"]["pair_cuda_memory"]["lower_bound_used_bytes"], 0)

    def test_event_outside_phase_is_rejected(self):
        a, _, _, _, _ = self.chain()
        role = f"model.{self.candidate['models'][0]['model_id']}.quality.cuda"
        a.rows[role][0]["event_ns"] = a.closed_ns + 1
        a.write()
        with self.assertRaisesRegex(v2.EvidenceError, "PHASE_INTERVAL"):
            self.evaluate(a)

    def test_stale_preflight_is_rejected(self):
        a, _, _, _, _ = self.chain()
        delta = self.contract["gates"]["phase_preflight_maximum_age_ns"] + 1
        for row in a.rows["phase.preflight"]:
            row["event_ns"] = a.started_ns - delta
            if row["kind"] == "probe":
                row["started_ns"] = row["event_ns"]
            if row["kind"] == "meta":
                row["completed_ns"] = row["event_ns"]
        a.write()
        with self.assertRaisesRegex(v2.EvidenceError, "PREFLIGHT_STALE"):
            self.evaluate(a)

    def test_phase_lock_must_precede_preflight(self):
        a, _, _, _, _ = self.chain()
        a.rows["phase.lock"][0]["event_ns"] = a.started_ns - 1
        a.write()
        with self.assertRaisesRegex(v2.EvidenceError, "LOCK_ORDER"):
            self.evaluate(a)

    def test_preflight_argv_is_exact(self):
        a, _, _, _, _ = self.chain()
        a.rows["phase.preflight"][0]["argv"].append("--unexpected")
        a.write()
        with self.assertRaisesRegex(v2.EvidenceError, "PREFLIGHT_ARGV"):
            self.evaluate(a)

    def test_quality_corpus_item_link_is_required(self):
        a, _, _, _, _ = self.chain()
        role = f"model.{self.candidate['models'][0]['model_id']}.quality.phone"
        a.rows[role][0]["corpus_item_sha256"] = marker("fabricated-corpus")
        a.write()
        with self.assertRaisesRegex(v2.EvidenceError, "corpus_item"):
            self.evaluate(a)

    def test_exact_memory_accounting_is_required(self):
        a, _, _, _, _ = self.chain()
        role = f"model.{self.candidate['models'][0]['model_id']}.cuda_memory"
        a.rows[role][1]["free_bytes"] -= 1
        a.write()
        with self.assertRaisesRegex(v2.EvidenceError, "MEMORY_EXACT"):
            self.evaluate(a)

    def test_process_memory_cannot_understate_buffers(self):
        a, _, _, _, _ = self.chain()
        role = f"model.{self.candidate['models'][0]['model_id']}.cuda_memory"
        a.rows[role][1]["process_used_bytes"] = 1
        a.write()
        with self.assertRaisesRegex(v2.EvidenceError, "PROCESS_MEMORY"):
            self.evaluate(a)

    def test_short_phone_continuation_cannot_hide_in_zip(self):
        a, _, _, _, _ = self.chain()
        role = f"model.{self.candidate['models'][0]['model_id']}.mechanics.phone"
        a.rows[role][1]["continuation_tokens"].pop()
        a.write()
        with self.assertRaisesRegex(v2.EvidenceError, "B8_DECODE_ROWS|ORACLE_LENGTH"):
            self.evaluate(a)

    def test_non_b8_call_geometry_is_rejected(self):
        a, _, _, _, _ = self.chain()
        role = f"model.{self.candidate['models'][0]['model_id']}.oracle.cuda_route"
        a.rows[role][0]["call_shapes"][0]["n_seqs"] = 7
        a.write()
        with self.assertRaisesRegex(v2.EvidenceError, "B8_GEOMETRY"):
            self.evaluate(a)

    def test_bridge_must_link_phone_tokens(self):
        a, _, _, _, _ = self.chain()
        role = f"model.{self.candidate['models'][0]['model_id']}.bridge"
        a.rows[role][1]["token_ids"][0] += 1
        a.write()
        with self.assertRaisesRegex(v2.EvidenceError, "BRIDGE_TOKENS"):
            self.evaluate(a)

    def test_bridge_must_link_cuda_ready_sample(self):
        a, _, _, _, _ = self.chain()
        role = f"model.{self.candidate['models'][0]['model_id']}.bridge"
        a.rows[role][-1]["cuda_memory_ready_sha256"] = marker("other-ready")
        a.write()
        with self.assertRaisesRegex(v2.EvidenceError, "BRIDGE_READY_LINK"):
            self.evaluate(a)

    def test_transfer_size_is_exact(self):
        a, _, _, _, _ = self.chain()
        role = f"model.{self.candidate['models'][0]['model_id']}.route_transfer"
        a.rows[role][1]["payload_bytes"] += 4
        a.write()
        with self.assertRaisesRegex(v2.EvidenceError, "TRANSFER_SIZE"):
            self.evaluate(a)

    def test_partial_shard_ufs_read_is_rejected(self):
        _, a_result, _, b_result, pair = self.chain()
        pair.rows["reprepare.A_to_B"][3]["verified_shard_bytes_read"] -= 1
        pair.write()
        with self.assertRaisesRegex(v2.EvidenceError, "FULL_SHARD_UFS"):
            self.evaluate(pair, [a_result, b_result])

    def test_ufs_counter_must_cover_exact_full_shard(self):
        _, a_result, _, b_result, pair = self.chain()
        pair.rows["reprepare.A_to_B"][3]["local_ufs_read_bytes"] -= 1
        pair.write()
        with self.assertRaisesRegex(v2.EvidenceError, "UFS_COUNTER"):
            self.evaluate(pair, [a_result, b_result])

    def test_pair_lock_must_bind_both_prior_results(self):
        _, a_result, _, b_result, pair = self.chain()
        pair.rows["phase.lock"][0]["prior_phase_result_sha256s"].pop()
        pair.write()
        with self.assertRaisesRegex(v2.EvidenceError, "prior_phase_result"):
            self.evaluate(pair, [a_result, b_result])


if __name__ == "__main__":
    unittest.main()
