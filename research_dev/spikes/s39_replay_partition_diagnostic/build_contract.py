#!/usr/bin/env python3
"""Freeze exact W9 inputs and the CUDA replay-partition contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import replay_partition as rp


W9 = rp.S39 / "W9_PROFILED_CUTOVER_CONTRACT.json"
W8_REPORT = (
    rp.S39
    / "results/w8_live_promotion_r1/run_20260725T013802Z/treatment_report.json"
)
W9_RUN = (
    rp.S39
    / "results/w9_profiled_cutover/run_20260725T_W9"
)
W9_MANIFEST = W9_RUN / "SHA256SUMS.txt"
W9_LEDGER = W9_RUN / "P1/treatment/publication_ledger"
MODEL = Path(
    "/home/myid/zs89458/Documents/models/"
    "Qwen2.5-14B-Instruct-Q8_0.gguf"
)
HOST_WORKER = rp.ROOT / "build-cuda/bin/llama-layersplit"
HOST_RELAY = rp.ROOT / "build-cuda/bin/llama-stage-direct-relay"

W9_SHA256 = "85d3fd2ebce18741586c3a8131d7d614dd701ae23251b7642792ad981076a575"
W8_REPORT_SHA256 = "32cd83d3964b956ce4bfb71f850d97a55c7d0e0668d7b0e2d8aa006138691601"
W9_MANIFEST_SHA256 = "895470a6edf4f7be79ca5015f357246469abbc475725d17512987cacbd81b1a6"
W9_FINAL_LEDGER_SHA256 = "ae8349fbaea4a47503cf0d9e4192dfd8b4f078ba845a7ba66bd133cbf11473d6"
MODEL_SHA256 = "23ca481b8226b2492ba8f3eb7af41e0f99d8605c16fb6dec7bc5cf6716b673cf"
HOST_WORKER_SHA256 = "833a7ed615a5402153a490433a773efe9b02d4447f1f9a8c72efd95f112fed70"
HOST_RELAY_SHA256 = "3b616b4f1372f138d9ca56976618a11523a90b29fb734638a8585a184d06761e"
F0_SHA256 = "fd0000745ad7b5ddefcb7a4645769a920536934bafbfb14df52e1847e40e6f67"
F1_SHA256 = "03d09124220ce94fa865708f8d53c860e889c507484b0881adea896b90cb5243"
W9_CONTINUATION_SHA256 = "01d21f1435e314a3047184cbf375cfec44e6ded1f3ec2938e634a02718453b1a"
W9_RUN_ID = "c2823ff96e366b5fa2c8c3f5f3e1ea9384d4ca2e1f033c0936c6b8f91d69ba5c"
W9_TRANSACTION_ID = "605153a948bf1ebe1fd23ba103158179e99fca8ec804a586920f6c2c257f11ae"


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(rp.ROOT))


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest() -> None:
    rp.require(rp.sha256(W9_MANIFEST.read_bytes()) == W9_MANIFEST_SHA256,
               "W9 root manifest digest mismatch")
    seen: set[str] = set()
    for line in W9_MANIFEST.read_text(encoding="ascii").splitlines():
        digest, separator, name = line.partition("  ")
        rp.require(
            separator == "  "
            and rp.HEX64.fullmatch(digest) is not None
            and name not in seen,
            "W9 root manifest line is invalid",
        )
        seen.add(name)
        target = (W9_RUN / name).resolve()
        rp.require(
            W9_RUN.resolve() in target.parents and target.is_file(),
            f"W9 root manifest target is invalid: {name}",
        )
        rp.require(
            digest_file(target) == digest,
            f"W9 root manifest target changed: {name}",
        )


def load_ledger() -> list[dict[str, Any]]:
    paths = sorted(W9_LEDGER.glob("*.json"))
    rp.require(
        len(paths) == 114
        and [path.name for path in paths]
        == [f"{index:06d}.json" for index in range(114)],
        "W9 ledger file set changed",
    )
    records: list[dict[str, Any]] = []
    previous = "0" * 64
    for index, path in enumerate(paths):
        value, raw = rp.read_canonical(path, f"W9 ledger {index}")
        rp.require(
            value.get("schema") == "s39-profiled-cutover-ledger-record-v1"
            and value.get("record_index") == index
            and value.get("previous_record_sha256") == previous
            and value.get("run_id") == W9_RUN_ID
            and value.get("transaction_id") == W9_TRANSACTION_ID,
            f"W9 ledger record {index} is invalid",
        )
        previous = rp.sha256(raw)
        records.append(value)
    rp.require(previous == W9_FINAL_LEDGER_SHA256, "W9 final ledger digest mismatch")
    return records


def published_tokens(
    records: list[dict[str, Any]],
    classification: str,
    positions: range,
) -> list[list[int]]:
    selected = [
        record["payload"]
        for record in records
        if record["event"] == "TOKEN_PUBLISHED"
        and record["payload"].get("classification") == classification
    ]
    output: list[list[tuple[int, int]]] = [[] for _ in range(8)]
    for item in selected:
        request_id = item["request_id"]
        rp.require(
            rp.is_int(request_id)
            and 0 <= request_id < 8
            and rp.is_int(item["position"])
            and item["position"] in positions
            and rp.is_int(item["token"])
            and item["token"] >= 0,
            f"W9 {classification} token is invalid",
        )
        output[request_id].append((item["position"], item["token"]))
    expected_width = len(positions)
    rp.require(
        all(
            len(row) == expected_width
            and [position for position, _ in sorted(row)] == list(positions)
            for row in output
        ),
        f"W9 {classification} geometry changed",
    )
    return [
        [token for _, token in sorted(row)]
        for row in output
    ]


def build_inputs() -> dict[str, object]:
    rp.require(rp.sha256(W9.read_bytes()) == W9_SHA256, "W9 contract changed")
    rp.require(
        rp.sha256(W8_REPORT.read_bytes()) == W8_REPORT_SHA256,
        "W8 treatment report changed",
    )
    verify_manifest()
    records = load_ledger()
    w8, _ = rp.read_canonical(W8_REPORT, "W8 treatment report")
    sequences = sorted(w8["sequences"], key=lambda item: item["sequence_index"])
    rp.require(
        len(sequences) == 8
        and [item["sequence_index"] for item in sequences] == list(range(8)),
        "W8 sequence set changed",
    )
    f0_tokens = published_tokens(records, "PHONE_F0", range(10, 11))
    delta_tokens = published_tokens(records, "PHONE_INFLIGHT", range(11, 12))
    continuation = published_tokens(
        records,
        "CUDA_CONTINUATION",
        range(12, 23),
    )
    f0_histories = []
    f1_histories = []
    output_sequences = []
    for index, sequence in enumerate(sequences):
        prompt = list(sequence["prompt_tokens"])
        preexisting = list(sequence["preexisting_tokens"])
        rp.require(
            len(prompt) == 8
            and len(preexisting) == 2
            and all(rp.is_int(token) and token >= 0 for token in prompt + preexisting),
            f"W8 sequence {index} is invalid",
        )
        f0 = prompt + preexisting + f0_tokens[index]
        f1 = f0 + delta_tokens[index]
        f0_histories.append(f0)
        f1_histories.append(f1)
        output_sequences.append({
            "f0_phone_token": f0_tokens[index][0],
            "f1_delta_token": delta_tokens[index][0],
            "preexisting_tokens": preexisting,
            "prompt_id": sequence["prompt_id"],
            "prompt_tokens": prompt,
            "sequence_index": index,
            "w9_continuation": continuation[index],
        })
    rp.require(
        rp.w5.histories_digest(f0_histories) == F0_SHA256,
        "reconstructed F0 does not match W9",
    )
    rp.require(
        rp.w5.histories_digest(f1_histories) == F1_SHA256,
        "reconstructed F1 does not match W9",
    )
    rp.require(
        rp.sha256(rp.canonical(continuation)) == W9_CONTINUATION_SHA256,
        "reconstructed W9 continuation changed",
    )
    snapshots = {
        record["event"]: record["payload"]
        for record in records
        if record["event"] in {"F0_SNAPSHOT", "F1_ACK"}
    }
    rp.require(
        snapshots["F0_SNAPSHOT"]["history_sha256"] == F0_SHA256
        and snapshots["F0_SNAPSHOT"]["positions"] == [10] * 8
        and snapshots["F1_ACK"]["history_sha256"] == F1_SHA256
        and snapshots["F1_ACK"]["positions"] == [11] * 8
        and snapshots["F1_ACK"]["d_actual"] == 1,
        "W9 frontier records changed",
    )
    return {
        "batch": 8,
        "f0_histories": f0_histories,
        "f0_history_sha256": F0_SHA256,
        "f1_histories": f1_histories,
        "f1_history_sha256": F1_SHA256,
        "schema": rp.INPUT_SCHEMA,
        "sequences": output_sequences,
        "source": {
            "w8_treatment_report_sha256": W8_REPORT_SHA256,
            "w9_final_ledger_record_sha256": W9_FINAL_LEDGER_SHA256,
            "w9_root_manifest_sha256": W9_MANIFEST_SHA256,
            "w9_run_id": W9_RUN_ID,
            "w9_transaction_id": W9_TRANSACTION_ID,
        },
        "w9_continuation": continuation,
        "w9_continuation_sha256": W9_CONTINUATION_SHA256,
    }


def path(name: str, identity_base: int) -> dict[str, object]:
    if name == "incremental_8_3_plus_1":
        return {
            "delta_chunk": 1,
            "history": "F0_PLUS_DELTA",
            "identity_base": identity_base,
            "name": name,
            "replay_chunk": 8,
        }
    if name == "full_f1_chunk2":
        return {
            "delta_chunk": 0,
            "history": "F1",
            "identity_base": identity_base,
            "name": name,
            "replay_chunk": 2,
        }
    rp.require(name == "full_f1_8_4", "unknown path")
    return {
        "delta_chunk": 0,
        "history": "F1",
        "identity_base": identity_base,
        "name": name,
        "replay_chunk": 8,
    }


def run(name: str, paths: list[dict[str, object]]) -> dict[str, object]:
    return {"fresh_process": True, "name": name, "paths": paths}


def build_contract(inputs_sha256: str) -> dict[str, object]:
    sources = [
        rp.HERE / "build_contract.py",
        rp.HERE / "replay_partition.py",
        rp.HERE / "replay_partition_probe.py",
        rp.HERE / "run_replay_partition_diagnostic.py",
        rp.HERE / "validate_replay_partition_diagnostic.py",
        rp.S39 / "phone_cuda_delta_probe.py",
        rp.S39 / "phone_cuda_handoff_probe.py",
        rp.S22 / "stage_v3_client.py",
    ]
    return {
        "artifacts": {
            "cuda_uuid": "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f",
            "host_relay": {
                "path": relative(HOST_RELAY),
                "sha256": HOST_RELAY_SHA256,
            },
            "host_worker": {
                "path": relative(HOST_WORKER),
                "sha256": HOST_WORKER_SHA256,
            },
            "model": {
                "path": str(MODEL),
                "sha256": MODEL_SHA256,
            },
        },
        "claim_boundary": {
            "controller_integration": False,
            "energy": False,
            "performance": False,
            "scheduler_eligible": False,
            "task_quality": False,
        },
        "execution": {
            "batch": 8,
            "continuation_tokens": 11,
            "include_optional_w5_geometry": True,
            "layer_split": [30, 48],
            "n_ctx_seq": 64,
            "runs": [
                run("fresh_incremental_r1", [path("incremental_8_3_plus_1", 10000)]),
                run("fresh_incremental_r2", [path("incremental_8_3_plus_1", 10000)]),
                run("fresh_full_chunk2_r1", [path("full_f1_chunk2", 20000)]),
                run("fresh_full_chunk2_r2", [path("full_f1_chunk2", 20000)]),
                run(
                    "same_process_incremental_then_full",
                    [
                        path("incremental_8_3_plus_1", 30000),
                        path("full_f1_chunk2", 40000),
                    ],
                ),
                run("fresh_full_8_4_r1", [path("full_f1_8_4", 50000)]),
                run("fresh_full_8_4_r2", [path("full_f1_8_4", 50000)]),
            ],
        },
        "inputs": {
            "path": relative(rp.HERE / "W9_HISTORIES.json"),
            "sha256": inputs_sha256,
        },
        "requirements": [
            "EXACT_W9_F0_F1_HISTORY_DIGESTS",
            "TWO_FRESH_INCREMENTAL_REPETITIONS",
            "TWO_FRESH_FULL_F1_CHUNK2_REPETITIONS",
            "ONE_INCREMENTAL_REMOVE_FULL_F1_SAME_PROCESS_SEQUENCE",
            "RAW_ROWS_AND_VECTORS_PERSIST_BEFORE_COMPARISON",
            "ZERO_SEQUENCE_STATE_AFTER_EACH_PATH",
            "CUDA0_ONLY_REALIZED_PLACEMENT",
            "NO_W9_MUTATION_OR_REPLACEMENT",
        ],
        "schema": rp.CONTRACT_SCHEMA,
        "scope": "CUDA_ONLY_DIAGNOSTIC",
        "source_evidence": {
            "f0_history_sha256": F0_SHA256,
            "f1_history_sha256": F1_SHA256,
            "w8_treatment_report": {
                "path": relative(W8_REPORT),
                "sha256": W8_REPORT_SHA256,
            },
            "w9_contract": {
                "path": relative(W9),
                "sha256": W9_SHA256,
            },
            "w9_final_ledger_record": {
                "path": relative(W9_LEDGER / "000113.json"),
                "sha256": W9_FINAL_LEDGER_SHA256,
            },
            "w9_root_manifest": {
                "path": relative(W9_MANIFEST),
                "sha256": W9_MANIFEST_SHA256,
            },
            "w9_run_id": W9_RUN_ID,
            "w9_transaction_id": W9_TRANSACTION_ID,
            "w9_continuation_sha256": W9_CONTINUATION_SHA256,
        },
        "source_sha256": {
            relative(source): digest_file(source)
            for source in sources
        },
        "status": "FROZEN_BEFORE_CUDA_ACQUISITION",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=rp.HERE)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    rp.require(output_dir == rp.HERE, "contract must be frozen in its source directory")
    inputs_path = output_dir / "W9_HISTORIES.json"
    contract_path = output_dir / "REPLAY_PARTITION_DIAGNOSTIC.json"
    rp.require(
        not inputs_path.exists() and not contract_path.exists(),
        "frozen inputs or contract already exist",
    )
    inputs = build_inputs()
    inputs_raw = rp.canonical(inputs)
    rp.write_atomic(inputs_path, inputs)
    try:
        contract = build_contract(rp.sha256(inputs_raw))
        rp.write_atomic(contract_path, contract)
        rp.load_contract(contract_path)
        rp.load_inputs(inputs_path, contract)
    except BaseException:
        inputs_path.unlink(missing_ok=True)
        contract_path.unlink(missing_ok=True)
        raise
    print(rp.sha256(contract_path.read_bytes()))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except rp.DiagnosticError as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
