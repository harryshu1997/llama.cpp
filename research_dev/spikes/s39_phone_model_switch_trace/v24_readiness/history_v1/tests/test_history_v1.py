#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
HISTORY = HERE.parent
if str(HISTORY) not in sys.path:
    sys.path.insert(0, str(HISTORY))

import capture_tokenizer_plan_v1 as capture
import history_common_v1 as common


FAKE_CODEC = """#!/usr/bin/python3
import json
import pathlib
import sys

mode = {mode!r}
model = pathlib.Path(sys.argv[2])
prompt = pathlib.Path(sys.argv[5]).read_text(encoding="utf-8")
request_id = int(pathlib.Path(sys.argv[5]).stem.split("-")[1])
count = 10 + request_id
tokens = [(request_id * 19 + index) % 100 for index in range(count)]
if mode == "negative":
    tokens[0] = -1
elif mode == "out_of_range":
    tokens[0] = 100
elif mode == "too_long":
    tokens = [1] * 505
if mode == "missing_key":
    print('{{"not_tokens":true}}')
elif mode == "noncanonical":
    print(json.dumps(tokens, separators=(",", ":")))
else:
    print("[" + ", ".join(str(token) for token in tokens) + "]")
if mode == "mutate_model":
    with model.open("ab") as output:
        output.write(b"x")
"""


def canonical(value) -> bytes:
    return common.canonical_bytes(value)


class Fixture:
    def __init__(self, root: Path, mode: str = "normal"):
        self.root = root
        self.corpus_path = root / "corpus.jsonl"
        self.candidate_path = root / "candidate.json"
        self.model_path = root / "model.gguf"
        self.codec_path = root / "fake-codec"
        self.plan_path = root / "plan.json"
        self.history_path = root / "history.json"

        corpus = []
        for index in range(64):
            corpus.append({
                "choices": [f"A{index}", f"B{index}", f"C{index}", f"D{index}"],
                "dataset": "cais/mmlu",
                "dataset_revision": "revision-test",
                "expected_answer": "ABCD"[index % 4],
                "item_index": index,
                "question": f"Question {index} \\u2192 value?",
                "source_row": index,
                "subject": f"subject-{index:02d}",
            })
        self.corpus = corpus
        self.corpus_path.write_bytes(b"".join(canonical(row) for row in corpus))
        candidate = {
            "candidate_attempt": 1,
            "candidate_attempt_limit": 1,
            "contract_sha256": "0" * 64,
            "historical_routes": {},
            "models": [],
            "schema": "s39-cp0-r1-candidate-v1",
            "status": "TEST",
            "task_suite": {
                "chat_template": "NONE_RAW_COMPLETION",
                "dataset": "cais/mmlu",
                "items": 64,
                "maximum_output_tokens": 8,
                "prompt_format": (
                    "Question: {question}\\nA. {choice0}\\nB. {choice1}\\n"
                    "C. {choice2}\\nD. {choice3}\\nAnswer:"
                ),
                "revision": "revision-test",
            },
        }
        self.candidate_path.write_bytes(canonical(candidate))
        self.model_path.write_bytes(b"fake-model-v1")
        self.codec_path.write_text(FAKE_CODEC.format(mode=mode), encoding="ascii")
        self.codec_path.chmod(
            self.codec_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP
        )
        self.locks = common.Locks(
            model_sha256=hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
            model_bytes=self.model_path.stat().st_size,
            vocab_size=100,
            candidate_sha256=hashlib.sha256(
                self.candidate_path.read_bytes()
            ).hexdigest(),
            candidate_bytes=self.candidate_path.stat().st_size,
            corpus_sha256=hashlib.sha256(
                self.corpus_path.read_bytes()
            ).hexdigest(),
            corpus_bytes=self.corpus_path.stat().st_size,
            corpus_items=64,
            corpus_revision="revision-test",
        )
        plan = capture.build_plan(
            self.codec_path.resolve(),
            self.model_path.resolve(),
            "tokenizer.test",
            self.locks,
        )
        self.plan_path.write_bytes(canonical(plan))

    def build(self):
        return common.build_history(
            self.corpus_path.resolve(),
            self.candidate_path.resolve(),
            self.plan_path.resolve(),
            self.locks,
        )


class HistoryTests(unittest.TestCase):
    def test_happy_path_and_independent_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            result = fixture.build()
            self.assertEqual(len(result["requests"]), 64)
            self.assertEqual(
                [row["item_index"] for row in result["requests"]],
                list(range(64)),
            )
            self.assertEqual(
                [len(row["token_ids"]) for row in result["requests"][:8]],
                list(range(10, 18)),
            )
            self.assertEqual(
                [row["request_id"] for row in result["requests"]],
                [index % 8 + 1 for index in range(64)],
            )
            self.assertEqual(
                [row["seq_id"] for row in result["requests"]],
                [index % 8 for index in range(64)],
            )
            mechanics = result["mechanics_b8"]
            self.assertEqual(mechanics, result["quality_groups"][0])
            self.assertEqual(mechanics["item_indices"], list(range(8)))
            self.assertEqual(
                [row["group_index"] for row in result["quality_groups"]],
                list(range(8)),
            )
            self.assertEqual(len(mechanics["prefill_partitions"]), 2)
            self.assertTrue(
                all(
                    len(row["rows"]) <= 64
                    for group in result["quality_groups"]
                    for row in group["prefill_partitions"]
                )
            )
            positions_by_partition = {}
            for partition_index, partition in enumerate(
                mechanics["prefill_partitions"]
            ):
                for row in partition["rows"]:
                    previous = positions_by_partition.setdefault(
                        row["position"], partition_index
                    )
                    self.assertEqual(previous, partition_index)
            flattened = [
                row
                for call in mechanics["prefill_partitions"]
                for row in call["rows"]
            ]
            self.assertEqual(
                [(row["position"], row["request_id"]) for row in flattened],
                sorted(
                    (row["position"], row["request_id"]) for row in flattened
                ),
            )
            self.assertEqual(len(mechanics["decode_calls"]), 7)
            self.assertTrue(
                all(len(call["rows"]) == 8 for call in mechanics["decode_calls"])
            )
            for decode_index, call in enumerate(mechanics["decode_calls"]):
                self.assertEqual(call["continuation_input_ordinal"], decode_index)
                self.assertEqual(
                    call["continuation_output_ordinal"], decode_index + 1
                )
                self.assertEqual(
                    [row["request_id"] for row in call["rows"]],
                    list(range(1, 9)),
                )
                self.assertEqual(
                    [row["seq_id"] for row in call["rows"]],
                    list(range(8)),
                )
                self.assertEqual(
                    [row["position"] for row in call["rows"]],
                    [
                        len(request["token_ids"]) + decode_index
                        for request in result["requests"][:8]
                    ],
                )
            self.assertEqual(
                set(result),
                {
                    "batch",
                    "candidate_sha256",
                    "continuation_tokens_per_request",
                    "corpus_sha256",
                    "mechanics_b8",
                    "model_id",
                    "model_sha256",
                    "n_batch",
                    "n_ctx_seq",
                    "n_ubatch",
                    "prefill_chunking",
                    "prefill_row_order",
                    "quality_groups",
                    "requests",
                    "schema",
                    "tokenizer",
                },
            )
            common.validate_history(
                result,
                fixture.corpus_path.resolve(),
                fixture.candidate_path.resolve(),
                fixture.plan_path.resolve(),
                fixture.locks,
            )

    def test_duplicate_corpus_id_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            rows = copy.deepcopy(fixture.corpus)
            rows[1]["item_index"] = 0
            fixture.corpus_path.write_bytes(
                b"".join(canonical(row) for row in rows)
            )
            fixture.locks = copy.copy(fixture.locks)
            fixture.locks = common.Locks(
                **{
                    **fixture.locks.__dict__,
                    "corpus_sha256": hashlib.sha256(
                        fixture.corpus_path.read_bytes()
                    ).hexdigest(),
                    "corpus_bytes": fixture.corpus_path.stat().st_size,
                }
            )
            with self.assertRaisesRegex(common.HistoryError, "DUPLICATE"):
                fixture.build()

    def test_missing_or_reordered_corpus_id_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            rows = copy.deepcopy(fixture.corpus)
            rows[0], rows[1] = rows[1], rows[0]
            fixture.corpus_path.write_bytes(
                b"".join(canonical(row) for row in rows)
            )
            fixture.locks = common.Locks(
                **{
                    **fixture.locks.__dict__,
                    "corpus_sha256": hashlib.sha256(
                        fixture.corpus_path.read_bytes()
                    ).hexdigest(),
                    "corpus_bytes": fixture.corpus_path.stat().st_size,
                }
            )
            with self.assertRaisesRegex(common.HistoryError, "REORDERED"):
                fixture.build()

    def test_malformed_tokenizer_output_rejected(self):
        for mode, message in (
            ("missing_key", "TOKEN_COUNT"),
            ("noncanonical", "OUTPUT_CANONICAL"),
        ):
            with self.subTest(mode=mode):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = Fixture(Path(temporary), mode)
                    with self.assertRaisesRegex(common.HistoryError, message):
                        fixture.build()

    def test_token_range_rejected(self):
        for mode in ("negative", "out_of_range"):
            with self.subTest(mode=mode):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = Fixture(Path(temporary), mode)
                    with self.assertRaisesRegex(common.HistoryError, "TOKEN_RANGE"):
                        fixture.build()

    def test_prompt_over_context_bound_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), "too_long")
            with self.assertRaisesRegex(common.HistoryError, "TOKEN_COUNT"):
                fixture.build()

    def test_noncanonical_plan_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            plan = json.loads(fixture.plan_path.read_text(encoding="ascii"))
            fixture.plan_path.write_text(json.dumps(plan, indent=2), encoding="ascii")
            with self.assertRaisesRegex(common.HistoryError, "E_CANONICAL"):
                fixture.build()

    def test_row_gap_and_reorder_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            result = fixture.build()
            for mutation in ("gap", "reorder"):
                changed = copy.deepcopy(result)
                if mutation == "gap":
                    changed["mechanics_b8"]["prefill_partitions"][0]["rows"][0][
                        "position"
                    ] = 1
                else:
                    rows = changed["mechanics_b8"]["prefill_partitions"][0][
                        "rows"
                    ]
                    rows[0], rows[1] = rows[1], rows[0]
                with self.subTest(mutation=mutation):
                    with self.assertRaisesRegex(
                        common.HistoryError, "HISTORY_MISMATCH"
                    ):
                        common.validate_history(
                            changed,
                            fixture.corpus_path.resolve(),
                            fixture.candidate_path.resolve(),
                            fixture.plan_path.resolve(),
                            fixture.locks,
                        )

    def test_prompt_bytes_and_wire_mapping_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            result = fixture.build()
            mutations = []
            prompt = copy.deepcopy(result)
            prompt["requests"][0]["prompt_utf8_base64"] = "eA=="
            mutations.append(prompt)
            wire = copy.deepcopy(result)
            wire["mechanics_b8"]["prefill_partitions"][0]["rows"][0][
                "request_id"
            ] = 0
            mutations.append(wire)
            decode = copy.deepcopy(result)
            decode["mechanics_b8"]["decode_calls"][0]["rows"][0]["position"] += 1
            mutations.append(decode)
            for index, changed in enumerate(mutations):
                with self.subTest(index=index):
                    with self.assertRaisesRegex(
                        common.HistoryError, "HISTORY_MISMATCH"
                    ):
                        common.validate_history(
                            changed,
                            fixture.corpus_path.resolve(),
                            fixture.candidate_path.resolve(),
                            fixture.plan_path.resolve(),
                            fixture.locks,
                        )

    def test_source_mutation_during_tokenization_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), "mutate_model")
            with self.assertRaisesRegex(common.HistoryError, "MODEL_MUTATED"):
                fixture.build()


if __name__ == "__main__":
    unittest.main()
