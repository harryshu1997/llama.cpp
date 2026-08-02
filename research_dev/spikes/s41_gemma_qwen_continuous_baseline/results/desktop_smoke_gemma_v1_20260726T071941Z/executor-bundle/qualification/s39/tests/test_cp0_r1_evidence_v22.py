import copy
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


S39 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S39))
sys.path.insert(0, str(S39 / "tests"))

import build_cp0_r1_mmlu64_v22 as mmlu
import build_cp0_r1_v22 as builder
import cp0_r1_evidence_v2 as v2
import cp0_r1_evidence_v21 as v21
import cp0_r1_evidence_v22 as v22
from test_cp0_r1_evidence_v21 import PhaseFixture, marker


class V22PhaseFixture(PhaseFixture):
    def __init__(self, *args, frozen_corpus, **kwargs):
        self.frozen_corpus = frozen_corpus
        super().__init__(*args, **kwargs)
        self.set_phase_id(self.phase_id.replace("v21", "v22"), write=False)
        if self.phase in ("A_ONLY", "B_ONLY"):
            self.repair_model()
        self.build_phase_lock()
        self.write()

    def set_phase_id(self, phase_id, write=True):
        self.phase_id = phase_id
        for rows in self.rows.values():
            for row in rows:
                row["acquisition_id"] = phase_id
                row["phase_id"] = phase_id
        if write:
            self.build_phase_lock()
            self.write()

    def repair_quality(self, install_frozen=True):
        corpus = self.rows["quality.corpus"]
        if install_frozen:
            for index, row in enumerate(corpus):
                for key, value in self.frozen_corpus[index].items():
                    row[key] = copy.deepcopy(value)
        corpus_raw = b"".join(v2.canonical_line(row) for row in corpus)
        corpus_sha = v2.sha256_bytes(corpus_raw)
        normalized = v21._normalize_rows(corpus)
        model = next(
            model
            for model in self.candidate["models"]
            if model["slot"] == v21.PHASE_SLOT[self.phase]
        )
        prefix = f"model.{model['model_id']}"
        prompt_format = self.candidate["task_suite"]["prompt_format"]
        for suffix in ("cuda", "phone"):
            role = f"{prefix}.quality.{suffix}"
            for index, row in enumerate(self.rows[role]):
                item = corpus[index]
                prompt = prompt_format.format(
                    question=item["question"],
                    choice0=item["choices"][0],
                    choice1=item["choices"][1],
                    choice2=item["choices"][2],
                    choice3=item["choices"][3],
                )
                row["corpus_sha256"] = corpus_sha
                row["corpus_item_sha256"] = v2.digest_json(normalized[index])
                row["prompt_sha256"] = hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest()
                row["raw_output"] = item["expected_answer"]

    def repair_executions(self, continuation_tokens):
        model = next(
            model
            for model in self.candidate["models"]
            if model["slot"] == v21.PHASE_SLOT[self.phase]
        )
        prefix = f"model.{model['model_id']}"
        call_shapes = [
            {"call_index": 0, "n_seqs": 8, "n_tokens": 16, "phase": "prefill"}
        ]
        call_shapes.extend(
            {
                "call_index": index + 1,
                "n_seqs": 8,
                "n_tokens": 8,
                "phase": "decode",
            }
            for index in range(continuation_tokens)
        )
        roles = (
            f"{prefix}.mechanics.phone",
            f"{prefix}.oracle.cuda_route",
            f"{prefix}.oracle.cuda_monolithic",
        )
        for role in roles:
            self.rows[role][0]["call_shapes"] = copy.deepcopy(call_shapes)
            for row in self.rows[role][1:]:
                request_id = row["request_id"]
                row["continuation_tokens"] = [
                    300 + request_id + 100 * index
                    for index in range(continuation_tokens)
                ]

        transfer_role = f"{prefix}.route_transfer"
        meta = self.rows[transfer_role][0]
        transfers = [meta]
        for index, call in enumerate(call_shapes):
            payload = (
                call["n_tokens"]
                * self.lock["hidden_size"]
                * self.lock["activation_element_bytes"]
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
                    "payload_sha256": marker(
                        f"{self.phase}-payload-{continuation_tokens}-{index}"
                    ),
                    "receiver": "op12",
                    "row_count": call["n_tokens"],
                    "sender": "op15",
                }
            )
        self.rows[transfer_role] = transfers

    def repair_memory_and_bridge(self):
        model = next(
            model
            for model in self.candidate["models"]
            if model["slot"] == v21.PHASE_SLOT[self.phase]
        )
        prefix = f"model.{model['model_id']}"
        memory_role = f"{prefix}.cuda_memory"
        memory = self.rows[memory_role]
        times = (
            self.started_ns + 10_000,
            self.started_ns + 300_000,
            self.started_ns + 400_000,
        )
        for row, timestamp in zip(memory, times):
            row["event_ns"] = timestamp
            row["timestamp_ns"] = timestamp

        mechanics_role = f"{prefix}.mechanics.phone"
        phone = v2.validate_execution(
            v21._normalize_rows(self.rows[mechanics_role]),
            mechanics_role,
            self.phase_id,
            model,
            "PHONE_COLLECTIVE",
        )
        normalized_requests = {
            row["request_id"]: row
            for row in v21._normalize_rows(self.rows[mechanics_role])
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
        bridge = self.rows[bridge_role]
        bridge[0]["timestamp_ns"] = self.started_ns + 20_000
        bridge[0]["event_ns"] = bridge[0]["timestamp_ns"]
        for request_id, row in enumerate(bridge[1:-1]):
            timestamp = self.started_ns + 200_000 + request_id
            row["timestamp_ns"] = timestamp
            row["event_ns"] = timestamp
            row["token_ids"] = phone["requests"][request_id]["continuation_tokens"]
            row["phone_request_sha256"] = v2.digest_json(
                normalized_requests[request_id]
            )
        bridge[-1]["timestamp_ns"] = memory[1]["timestamp_ns"]
        bridge[-1]["event_ns"] = bridge[-1]["timestamp_ns"]
        bridge[-1]["cuda_memory_ready_sha256"] = v2.digest_json(ready_normalized)

    def repair_model(self, continuation_tokens=8):
        if self.phase == "A_ONLY":
            expected = self.contract["incumbent_route_lock"]
            for key in (
                "backend",
                "cut_layer",
                "op12_shard_sha256",
                "op12_stored_layers",
                "op15_shard_sha256",
                "op15_stored_layers",
            ):
                self.lock[key] = copy.deepcopy(expected[key])
            model = self.candidate["models"][0]
            prefix = f"model.{model['model_id']}"
            for phone in ("op15", "op12"):
                meta = self.rows[f"{prefix}.placement.{phone}"][0]
                meta["stored_layers"] = copy.deepcopy(
                    self.lock[f"{phone}_stored_layers"]
                )
                meta["shard_sha256"] = self.lock[f"{phone}_shard_sha256"]
                meta["executed_layers"] = (
                    [0, self.lock["cut_layer"]]
                    if phone == "op15"
                    else [self.lock["cut_layer"], model["n_layer"]]
                )
        self.repair_quality()
        self.repair_executions(continuation_tokens)
        self.repair_memory_and_bridge()
        self.build_phase_lock()
        self.write()


class Cp0R1EvidenceV22Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.contract,
            cls.contract_raw,
            cls.candidate,
            cls.candidate_raw,
            cls.parent,
            cls.frozen_corpus,
        ) = v22.validate_inputs(
            S39 / "CP0_R1_EVIDENCE_CONTRACT_V2_2.json",
            S39 / "CP0_R1_CANDIDATE.json",
        )

    def temp_root(self, name):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name) / name

    def fixture(self, phase, prior=None, locks=None):
        return V22PhaseFixture(
            self.temp_root(phase.lower()),
            phase,
            self.contract,
            self.contract_raw,
            self.parent,
            self.candidate,
            self.candidate_raw,
            prior or [],
            locks or [],
            frozen_corpus=self.frozen_corpus,
        )

    def evaluate(self, fixture, prior=None):
        return v22.evaluate_root(
            fixture.root,
            v22.MANIFEST_NAME,
            self.contract,
            self.contract_raw,
            self.candidate,
            self.candidate_raw,
            self.parent,
            self.frozen_corpus,
            prior or [],
        )

    def chain(self):
        a = self.fixture("A_ONLY")
        a_result = self.evaluate(a)
        b = self.fixture("B_ONLY", [a_result], [a.lock])
        b_result = self.evaluate(b, [a_result])
        pair = self.fixture(
            "PAIR",
            [a_result, b_result],
            [a.lock, b.lock],
        )
        return a, a_result, b, b_result, pair

    def test_contract_and_corpus_are_byte_deterministic(self):
        self.assertEqual(
            (S39 / "CP0_R1_EVIDENCE_CONTRACT_V2_2.json").read_bytes(),
            builder.canonical_bytes(builder.build_contract()),
        )
        self.assertEqual(len(self.frozen_corpus), 64)
        self.assertEqual(self.frozen_corpus[0]["subject"], "abstract_algebra")
        self.assertEqual(self.frozen_corpus[57]["source_row"], 1)
        sources = mmlu.load_source_manifest(
            S39 / "CP0_R1_MMLU64_SOURCES_V2_2.json"
        )
        self.assertEqual(len(sources["files"]), 57)
        self.assertEqual(sources["revision"], self.candidate["task_suite"]["revision"])

    def test_a_only_and_full_cycle_authorization_pass(self):
        a, a_result, b, b_result, pair = self.chain()
        self.assertEqual(a_result["status"], "MODEL_A_QUALIFICATION_PASS")
        self.assertGreaterEqual(
            a_result["derived"]["model"]["quality"]["cuda_correct"],
            25,
        )
        authorization = v22.authorize_cycle(
            a.root,
            b.root,
            pair.root,
            v22.MANIFEST_NAME,
            self.contract,
            self.contract_raw,
            self.candidate,
            self.candidate_raw,
            self.parent,
            self.frozen_corpus,
        )
        self.assertEqual(
            authorization["status"],
            "ONE_REDUCED_A_TO_B_TO_A_CYCLE_AUTHORIZED",
        )
        self.assertEqual(b_result["status"], "MODEL_B_QUALIFICATION_PASS")

    def test_rebound_noncanonical_corpus_is_rejected(self):
        a = self.fixture("A_ONLY")
        a.rows["quality.corpus"][0]["question"] += " forged"
        a.repair_quality(install_frozen=False)
        a.build_phase_lock()
        a.write()
        with self.assertRaisesRegex(v2.EvidenceError, "CANONICAL_CORPUS"):
            self.evaluate(a)

    def test_cuda_above_chance_floor_is_required(self):
        a = self.fixture("A_ONLY")
        model_id = self.candidate["models"][0]["model_id"]
        for suffix in ("cuda", "phone"):
            rows = a.rows[f"model.{model_id}.quality.{suffix}"]
            for index, row in enumerate(rows[:40]):
                expected = self.frozen_corpus[index]["expected_answer"]
                row["raw_output"] = "ABCD"[("ABCD".index(expected) + 1) % 4]
        a.write()
        with self.assertRaisesRegex(v2.EvidenceError, "CUDA_QUALITY_FLOOR"):
            self.evaluate(a)

    def test_incumbent_route_fields_are_exact_bound(self):
        expected = copy.deepcopy(self.contract["incumbent_route_lock"])
        model = self.candidate["models"][0]
        mutations = {
            "backend": "HTP0",
            "cut_layer": 29,
            "op12_shard_sha256": marker("wrong-op12"),
            "op12_stored_layers": [23, 40],
            "op15_shard_sha256": marker("wrong-op15"),
            "op15_stored_layers": [0, 31],
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                lock = copy.deepcopy(expected)
                lock[key] = value
                with self.assertRaisesRegex(v2.EvidenceError, "INCUMBENT_ROUTE"):
                    v22.validate_incumbent_route(lock, self.contract, model)

    def test_equal_seven_token_paths_are_rejected(self):
        a = self.fixture("A_ONLY")
        a.repair_model(continuation_tokens=7)
        with self.assertRaisesRegex(v2.EvidenceError, "CONTINUATION_LENGTH"):
            self.evaluate(a)

    def test_publication_before_linked_completion_is_rejected(self):
        a = self.fixture("A_ONLY")
        model_id = self.candidate["models"][0]["model_id"]
        role = f"model.{model_id}.mechanics.phone"
        for index, row in enumerate(a.rows[role][1:]):
            row["event_ns"] = a.started_ns + 250_000 + index
        a.write()
        with self.assertRaisesRegex(v2.EvidenceError, "BRIDGE_CAUSAL_ORDER"):
            self.evaluate(a)

    def test_bridge_timestamp_must_equal_phase_event(self):
        a = self.fixture("A_ONLY")
        model_id = self.candidate["models"][0]["model_id"]
        role = f"model.{model_id}.bridge"
        a.rows[role][1]["event_ns"] -= 1
        a.write()
        with self.assertRaisesRegex(v2.EvidenceError, "BRIDGE_EVENT_TIME"):
            self.evaluate(a)

    def test_a_and_b_phase_ids_must_be_distinct(self):
        a, a_result, b, _, _ = self.chain()
        b.set_phase_id(a.phase_id)
        with self.assertRaisesRegex(v2.EvidenceError, "PHASE_ID_REUSE"):
            self.evaluate(b, [a_result])

    def test_pair_phase_id_must_be_distinct(self):
        _, a_result, b, b_result, pair = self.chain()
        pair.set_phase_id(b.phase_id)
        with self.assertRaisesRegex(v2.EvidenceError, "PHASE_ID_REUSE"):
            self.evaluate(pair, [a_result, b_result])

    def test_authorization_reopens_tampered_a_root(self):
        a, _, b, _, pair = self.chain()
        a.rows["quality.corpus"][0]["question"] += " tampered"
        a.repair_quality(install_frozen=False)
        a.build_phase_lock()
        a.write()
        with self.assertRaises(v2.EvidenceError):
            v22.authorize_cycle(
                a.root,
                b.root,
                pair.root,
                v22.MANIFEST_NAME,
                self.contract,
                self.contract_raw,
                self.candidate,
                self.candidate_raw,
                self.parent,
                self.frozen_corpus,
            )

    def test_legacy_status_only_cycle_result_is_rejected(self):
        root = self.temp_root("legacy")
        root.mkdir(parents=True)
        legacy = root / "result.json"
        legacy.write_bytes(
            v2.canonical_bytes({"status": "TWO_ROUTE_ELIGIBILITY_PASS"})
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(S39 / "cp0_r1_evidence_v22.py"),
                "--authorize-cycle",
                "--legacy-result",
                str(legacy),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertNotIn("Traceback", completed.stderr + completed.stdout)

    def test_parent_manifests_remain_unchanged(self):
        expected = {
            "CP0_R1_SHA256SUMS.txt": (
                "480cf8836e0a83ea9102c19e82bf5ec01e2f78b0ea2a3d9061be71bed4b31275"
            ),
            "CP0_R1_V2_SHA256SUMS.txt": (
                "517c23a049f3378ca10d71d5397abbe4dd7730bdc251a179fd84e96968d15d8c"
            ),
            "CP0_R1_V2_1_SHA256SUMS.txt": (
                "4cc9f08b85793f3d5f93f3eff97879102e11f79c6b669dc21af234607e8a893f"
            ),
        }
        for name, digest in expected.items():
            with self.subTest(name=name):
                self.assertEqual(v2.sha256_bytes((S39 / name).read_bytes()), digest)


if __name__ == "__main__":
    unittest.main()
