#!/usr/bin/env python3

import argparse
import copy
from pathlib import Path
from typing import Any

import build_cp0_r1_mmlu64_v22 as mmlu
import build_cp0_r1_v22 as builder
import cp0_r1_evidence_v2 as v2
import cp0_r1_evidence_v21 as v21


HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2_2.json"
DEFAULT_CANDIDATE = HERE / "CP0_R1_CANDIDATE.json"
PARENT_CONTRACT = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2_1.json"
MANIFEST_NAME = "EVIDENCE_BUNDLE.json"


def load_frozen_corpus(contract: dict[str, Any]) -> list[dict[str, Any]]:
    frozen = contract["quality_corpus"]
    raw = v2.secure_read(HERE, frozen["path"])
    v2.exact(len(raw), frozen["bytes"], "quality_corpus.bytes")
    v2.exact(v2.sha256_bytes(raw), frozen["sha256"], "quality_corpus.sha256")
    rows = []
    for index, line in enumerate(raw.splitlines(keepends=True)):
        row = v2.parse_json(line, f"quality_corpus[{index}]")
        v2.require(v2.canonical_line(row) == line, f"E_CANONICAL: corpus[{index}]")
        v2.exact_keys(
            row,
            {
                "choices",
                "dataset",
                "dataset_revision",
                "expected_answer",
                "item_index",
                "question",
                "source_row",
                "subject",
            },
            f"quality_corpus[{index}]",
        )
        v2.exact(row["item_index"], index, f"quality_corpus[{index}].item_index")
        rows.append(row)
    v2.exact(len(rows), frozen["items"], "quality_corpus.items")
    source_raw = v2.secure_read(HERE, frozen["source_manifest_path"])
    v2.exact(
        v2.sha256_bytes(source_raw),
        frozen["source_manifest_sha256"],
        "quality_corpus.source_manifest_sha256",
    )
    source_manifest = mmlu.load_source_manifest(HERE / frozen["source_manifest_path"])
    v2.exact(source_manifest["dataset"], frozen["dataset"], "quality_corpus.source")
    v2.exact(
        source_manifest["revision"],
        frozen["revision"],
        "quality_corpus.source_revision",
    )
    v2.exact(len(source_manifest["files"]), 57, "quality_corpus.source_files")
    v2.exact(
        builder.sha256_file(HERE / "build_cp0_r1_mmlu64_v22.py"),
        frozen["builder_sha256"],
        "quality_corpus.builder_sha256",
    )
    return rows


def validate_inputs(
    contract_path: Path,
    candidate_path: Path,
) -> tuple[
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
    dict[str, Any],
    list[dict[str, Any]],
]:
    contract, contract_raw = v2.load_canonical_path(contract_path)
    v2.exact(contract, builder.build_contract(), "contract")
    (
        parent,
        parent_raw,
        candidate,
        candidate_raw,
        v2_parent,
    ) = v21.validate_inputs(PARENT_CONTRACT, candidate_path)
    v2.exact(
        v2.sha256_bytes(parent_raw),
        contract["parent"]["contract_sha256"],
        "contract.parent.contract_sha256",
    )
    v2.exact(
        v2.sha256_bytes(candidate_raw),
        contract["parent"]["candidate_sha256"],
        "contract.parent.candidate_sha256",
    )
    del parent
    corpus = load_frozen_corpus(contract)
    task = candidate["task_suite"]
    v2.exact(task["dataset"], contract["quality_corpus"]["dataset"], "corpus.dataset")
    v2.exact(
        task["revision"],
        contract["quality_corpus"]["revision"],
        "corpus.revision",
    )
    v2.exact(task["items"], len(corpus), "corpus.items")
    return contract, contract_raw, candidate, candidate_raw, v2_parent, corpus


def phase_result_sha256(result: dict[str, Any]) -> str:
    return v2.sha256_bytes(v2.canonical_bytes(result))


def corpus_content(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dynamic = {
        "acquisition_id",
        "event_ns",
        "kind",
        "phase",
        "phase_id",
        "role",
    }
    return [
        {key: value for key, value in row.items() if key not in dynamic}
        for row in rows
    ]


def validate_exact_corpus(
    rows: list[dict[str, Any]],
    frozen: list[dict[str, Any]],
) -> None:
    v2.exact(corpus_content(rows), frozen, "E_CANONICAL_CORPUS")


def validate_incumbent_route(
    lock: dict[str, Any],
    contract: dict[str, Any],
    model: dict[str, Any],
) -> None:
    expected = contract["incumbent_route_lock"]
    v2.exact(model["slot"], "A", "incumbent.slot")
    v2.exact(lock["model_id"], expected["model_id"], "E_INCUMBENT_ROUTE: model")
    for key in (
        "backend",
        "cut_layer",
        "op12_shard_sha256",
        "op12_stored_layers",
        "op15_shard_sha256",
        "op15_stored_layers",
    ):
        v2.exact(lock[key], expected[key], f"E_INCUMBENT_ROUTE: {key}")


def validate_continuations(
    rows_by_role: dict[str, list[dict[str, Any]]],
    model: dict[str, Any],
    phase_id: str,
    expected_tokens: int,
) -> dict[str, int]:
    executions = v21._validate_execution_geometry(rows_by_role, model, phase_id)
    paths = ("phone", "cuda_route", "cuda_monolithic")
    for path, execution in zip(paths, executions):
        for request_id in range(8):
            v2.exact(
                len(execution["requests"][request_id]["continuation_tokens"]),
                expected_tokens,
                f"E_CONTINUATION_LENGTH: {path}[{request_id}]",
            )
    return {"paths": 3, "requests": 8, "tokens_per_request": expected_tokens}


def validate_bridge_causality(
    rows_by_role: dict[str, list[dict[str, Any]]],
    model: dict[str, Any],
) -> dict[str, int]:
    prefix = f"model.{model['model_id']}"
    mechanics_role = f"{prefix}.mechanics.phone"
    bridge_role = f"{prefix}.bridge"
    completed = {
        row["request_id"]: row["event_ns"]
        for row in rows_by_role[mechanics_role]
        if row["kind"] == "request"
    }
    publications = [
        row
        for row in rows_by_role[bridge_role]
        if row["kind"] == "phone_publication_received"
    ]
    ready = next(row for row in rows_by_role[bridge_role] if row["kind"] == "cuda_ready")
    for index, row in enumerate(rows_by_role[bridge_role]):
        v2.exact(
            row["event_ns"],
            row["timestamp_ns"],
            f"E_BRIDGE_EVENT_TIME: {bridge_role}[{index}]",
        )
    for row in publications:
        request_id = row["request_id"]
        v2.require(request_id in completed, f"E_BRIDGE_COMPLETION: {request_id}")
        v2.require(
            completed[request_id] <= row["timestamp_ns"] < ready["timestamp_ns"],
            f"E_BRIDGE_CAUSAL_ORDER: {request_id}",
        )
    v2.exact(len(publications), 8, "E_BRIDGE_PUBLICATIONS")
    return {
        "cuda_ready_ns": ready["timestamp_ns"],
        "linked_publications": len(publications),
    }


def validate_prior_result(
    result: dict[str, Any],
    contract_raw: bytes,
    candidate_raw: bytes,
    phase: str,
    status: str,
) -> None:
    v2.exact(result.get("schema"), "s39-cp0-r1-evidence-result-v2.2", "E_LEGACY_RESULT")
    v2.exact(result.get("phase"), phase, "E_PHASE_CHAIN")
    v2.exact(result.get("status"), status, "E_PHASE_CHAIN")
    v2.exact(
        result.get("contract_sha256"),
        v2.sha256_bytes(contract_raw),
        "E_RESULT_CONTRACT",
    )
    v2.exact(
        result.get("candidate_sha256"),
        v2.sha256_bytes(candidate_raw),
        "E_RESULT_CANDIDATE",
    )


def evaluate_model_phase(
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate: dict[str, Any],
    candidate_raw: bytes,
    parent: dict[str, Any],
    frozen_corpus: list[dict[str, Any]],
    manifest: dict[str, Any],
    manifest_raw: bytes,
    rows_by_role: dict[str, list[dict[str, Any]]],
    artifact_digests: dict[str, str],
    prior_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if manifest["phase"] == "B_ONLY":
        v2.require(len(prior_results) == 1, "E_PHASE_CHAIN: B requires A")
        validate_prior_result(
            prior_results[0],
            contract_raw,
            candidate_raw,
            "A_ONLY",
            "MODEL_A_QUALIFICATION_PASS",
        )
        v2.require(
            prior_results[0]["phase_id"] != manifest["phase_id"],
            "E_PHASE_ID_REUSE: A/B",
        )
    result = v21.evaluate_model_phase(
        contract,
        contract_raw,
        candidate,
        candidate_raw,
        parent,
        manifest,
        manifest_raw,
        rows_by_role,
        artifact_digests,
        prior_results,
    )
    slot = v21.PHASE_SLOT[manifest["phase"]]
    model = next(model for model in candidate["models"] if model["slot"] == slot)
    validate_exact_corpus(rows_by_role["quality.corpus"], frozen_corpus)
    quality = result["derived"]["model"]["quality"]
    v2.require(
        quality["cuda_correct"]
        >= contract["gates"]["cuda_quality_minimum_correct_items"],
        f"E_CUDA_QUALITY_FLOOR: {model['model_id']}",
    )
    if slot == "A":
        validate_incumbent_route(
            result["derived"]["model"]["route_lock"],
            contract,
            model,
        )
    continuation = validate_continuations(
        rows_by_role,
        model,
        manifest["phase_id"],
        contract["gates"]["continuation_tokens_per_request"],
    )
    causality = validate_bridge_causality(rows_by_role, model)
    result = copy.deepcopy(result)
    result["schema"] = "s39-cp0-r1-evidence-result-v2.2"
    result["derived"]["v2_2"] = {
        "canonical_corpus_sha256": contract["quality_corpus"]["sha256"],
        "continuation": continuation,
        "publication_causality": causality,
    }
    return result


def evaluate_pair_phase(
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate: dict[str, Any],
    candidate_raw: bytes,
    parent: dict[str, Any],
    manifest: dict[str, Any],
    manifest_raw: bytes,
    rows_by_role: dict[str, list[dict[str, Any]]],
    artifact_digests: dict[str, str],
    prior_results: list[dict[str, Any]],
) -> dict[str, Any]:
    v2.require(len(prior_results) == 2, "E_PHASE_CHAIN: pair requires A and B")
    validate_prior_result(
        prior_results[0],
        contract_raw,
        candidate_raw,
        "A_ONLY",
        "MODEL_A_QUALIFICATION_PASS",
    )
    validate_prior_result(
        prior_results[1],
        contract_raw,
        candidate_raw,
        "B_ONLY",
        "MODEL_B_QUALIFICATION_PASS",
    )
    phase_ids = [
        prior_results[0]["phase_id"],
        prior_results[1]["phase_id"],
        manifest["phase_id"],
    ]
    v2.require(len(set(phase_ids)) == 3, "E_PHASE_ID_REUSE: A/B/PAIR")
    result = v21.evaluate_pair_phase(
        contract,
        contract_raw,
        candidate,
        candidate_raw,
        parent,
        manifest,
        manifest_raw,
        rows_by_role,
        artifact_digests,
        prior_results,
    )
    result = copy.deepcopy(result)
    result["schema"] = "s39-cp0-r1-evidence-result-v2.2"
    result["derived"]["phase_ids"] = phase_ids
    return result


def evaluate_root(
    root: Path,
    manifest_name: str,
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate: dict[str, Any],
    candidate_raw: bytes,
    parent: dict[str, Any],
    frozen_corpus: list[dict[str, Any]],
    prior_results: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest, manifest_raw, rows, digests = v21.load_bundle(
        root,
        manifest_name,
        contract,
        contract_raw,
        candidate_raw,
    )
    if manifest["phase"] == "PAIR":
        return evaluate_pair_phase(
            contract,
            contract_raw,
            candidate,
            candidate_raw,
            parent,
            manifest,
            manifest_raw,
            rows,
            digests,
            prior_results,
        )
    return evaluate_model_phase(
        contract,
        contract_raw,
        candidate,
        candidate_raw,
        parent,
        frozen_corpus,
        manifest,
        manifest_raw,
        rows,
        digests,
        prior_results,
    )


def evaluate_chain(
    a_root: Path,
    b_root: Path,
    pair_root: Path,
    manifest_name: str,
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate: dict[str, Any],
    candidate_raw: bytes,
    parent: dict[str, Any],
    frozen_corpus: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    a_result = evaluate_root(
        a_root,
        manifest_name,
        contract,
        contract_raw,
        candidate,
        candidate_raw,
        parent,
        frozen_corpus,
        [],
    )
    b_result = evaluate_root(
        b_root,
        manifest_name,
        contract,
        contract_raw,
        candidate,
        candidate_raw,
        parent,
        frozen_corpus,
        [a_result],
    )
    pair_result = evaluate_root(
        pair_root,
        manifest_name,
        contract,
        contract_raw,
        candidate,
        candidate_raw,
        parent,
        frozen_corpus,
        [a_result, b_result],
    )
    return a_result, b_result, pair_result


def authorize_cycle(
    a_root: Path,
    b_root: Path,
    pair_root: Path,
    manifest_name: str,
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate: dict[str, Any],
    candidate_raw: bytes,
    parent: dict[str, Any],
    frozen_corpus: list[dict[str, Any]],
) -> dict[str, Any]:
    a_result, b_result, pair_result = evaluate_chain(
        a_root,
        b_root,
        pair_root,
        manifest_name,
        contract,
        contract_raw,
        candidate,
        candidate_raw,
        parent,
        frozen_corpus,
    )
    v2.exact(
        pair_result["status"],
        "TWO_ROUTE_ELIGIBILITY_PASS",
        "E_CYCLE_PAIR_STATUS",
    )
    return {
        "candidate_sha256": v2.sha256_bytes(candidate_raw),
        "contract_sha256": v2.sha256_bytes(contract_raw),
        "phase_result_sha256s": [
            phase_result_sha256(a_result),
            phase_result_sha256(b_result),
            phase_result_sha256(pair_result),
        ],
        "schema": "s39-cp0-r1-cycle-authorization-v2.2",
        "status": contract["claim_boundary"]["cycle_authorization"]["status"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive V2.2 qualification or cycle authorization from raw roots"
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--a-bundle-root", type=Path)
    parser.add_argument("--b-bundle-root", type=Path)
    parser.add_argument("--manifest", default=MANIFEST_NAME)
    parser.add_argument("--authorize-cycle", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        (
            contract,
            contract_raw,
            candidate,
            candidate_raw,
            parent,
            frozen_corpus,
        ) = validate_inputs(args.contract, args.candidate)
        if args.authorize_cycle:
            v2.require(args.a_bundle_root is not None, "E_CYCLE_ROOT: A")
            v2.require(args.b_bundle_root is not None, "E_CYCLE_ROOT: B")
            v2.require(args.bundle_root is not None, "E_CYCLE_ROOT: PAIR")
            result = authorize_cycle(
                args.a_bundle_root,
                args.b_bundle_root,
                args.bundle_root,
                args.manifest,
                contract,
                contract_raw,
                candidate,
                candidate_raw,
                parent,
                frozen_corpus,
            )
        elif args.bundle_root is None:
            result = {
                "candidate_sha256": v2.sha256_bytes(candidate_raw),
                "contract_sha256": v2.sha256_bytes(contract_raw),
                "schema": "s39-cp0-r1-evidence-contract-check-v2.2",
                "status": "V2_2_EVIDENCE_READY_ACQUISITION_NOT_RUN",
            }
        else:
            manifest_raw = v2.secure_read(args.bundle_root, args.manifest)
            manifest = v2.parse_json(manifest_raw, args.manifest)
            phase = manifest.get("phase")
            prior = []
            if phase in ("B_ONLY", "PAIR"):
                v2.require(args.a_bundle_root is not None, "E_PHASE_CHAIN: missing A")
                prior.append(
                    evaluate_root(
                        args.a_bundle_root,
                        args.manifest,
                        contract,
                        contract_raw,
                        candidate,
                        candidate_raw,
                        parent,
                        frozen_corpus,
                        [],
                    )
                )
            if phase == "PAIR":
                v2.require(args.b_bundle_root is not None, "E_PHASE_CHAIN: missing B")
                prior.append(
                    evaluate_root(
                        args.b_bundle_root,
                        args.manifest,
                        contract,
                        contract_raw,
                        candidate,
                        candidate_raw,
                        parent,
                        frozen_corpus,
                        prior[:1],
                    )
                )
            result = evaluate_root(
                args.bundle_root,
                args.manifest,
                contract,
                contract_raw,
                candidate,
                candidate_raw,
                parent,
                frozen_corpus,
                prior,
            )
        print(v2.canonical_bytes(result).decode("ascii"), end="")
        return 0
    except (v2.EvidenceError, OSError, KeyError, ValueError) as exc:
        print(f"CP0_R1_V2_2_REFUSED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
