#!/usr/bin/env python3
"""Independently reduce persisted replay-partition captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import replay_partition as rp


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_prefix_json(path: Path, prefix: str) -> dict[str, Any]:
    values = []
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if line.startswith(prefix):
            try:
                value = json.loads(line[len(prefix):], object_pairs_hook=rp.strict_object)
            except json.JSONDecodeError as exc:
                raise rp.DiagnosticError(
                    f"{path.name}: malformed {prefix.strip()}"
                ) from exc
            rp.require(type(value) is dict, f"{path.name}: invalid certificate")
            values.append(value)
    rp.require(len(values) == 1, f"{path.name}: expected one {prefix.strip()}")
    return values[0]


def validate_placement(
    path: Path,
    role: str,
    process: dict[str, Any],
    layer_start: int,
    layer_end: int,
) -> dict[str, Any]:
    placement = load_prefix_json(path, "PLACEMENTCERT ")
    session = load_prefix_json(path, "SESSIONCERT ")
    rp.require(
        placement.get("schema") == "layersplit-scheduled-placement-v2"
        and placement.get("pid") == process["pid"]
        and placement.get("run_rc") == 0
        and placement.get("status") == "SCHEDULED_PLACEMENT_OK"
        and placement.get("missing_buffer_compute_nodes") == 0
        and placement.get("compute_nodes", 0) > 0
        and placement.get("layer_start") == layer_start
        and placement.get("layer_end") == layer_end
        and placement.get("n_layer") == 48,
        f"{path.name}: invalid placement certificate",
    )
    rp.require(
        session.get("schema") == "ls-stagenet-session-v2"
        and session.get("worker_pid") == process["pid"]
        and session.get("session_end") == "STOP"
        and session.get("placement_status") == "SCHEDULED_PLACEMENT_OK"
        and session.get("missing_buffer_compute_nodes") == 0
        and session.get("layer_start") == layer_start
        and session.get("layer_end") == layer_end
        and session.get("n_layer") == 48,
        f"{path.name}: invalid session certificate",
    )
    buffers = placement.get("compute_by_buffer_type")
    by_op = placement.get("compute_by_op_and_buffer")
    rp.require(
        type(buffers) is dict and type(by_op) is dict,
        f"{path.name}: invalid placement tallies",
    )
    if role == "tail":
        rp.require(
            set(buffers) == {"CUDA0"}
            and all(set(per_buffer) == {"CUDA0"} for per_buffer in by_op.values()),
            f"{path.name}: tail fallback",
        )
    else:
        rp.require(
            set(buffers).issubset({"CUDA0", "CUDA_Host"})
            and "CUDA0" in buffers,
            f"{path.name}: head fallback",
        )
        for op, per_buffer in by_op.items():
            rp.require(
                set(per_buffer).issubset({"CUDA0", "CUDA_Host"})
                and ("CUDA_Host" not in per_buffer or op == "GET_ROWS"),
                f"{path.name}: undeclared host operation",
            )
    return {
        "compute_by_buffer_type": buffers,
        "compute_nodes": placement["compute_nodes"],
        "layer_range": [layer_start, layer_end],
        "pid": process["pid"],
        "role": role,
        "status": placement["status"],
    }


def expected_rows(
    histories: Sequence[Sequence[int]],
    identity_base: int,
    start: int,
    end: int,
) -> list[dict[str, object]]:
    return [
        {
            "hidden": None,
            "position": position,
            "request_id": identity_base + sequence,
            "route_epoch": identity_base + sequence,
            "seq_id": sequence,
            "token": histories[sequence][position],
        }
        for sequence in range(len(histories))
        for position in range(start, end)
    ]


def output_vector(call: dict[str, Any], batch: int) -> list[int]:
    output = call["output_rows"]
    by_sequence: dict[int, list[dict[str, Any]]] = {}
    for row in output:
        by_sequence.setdefault(row["seq_id"], []).append(row)
    rp.require(set(by_sequence) == set(range(batch)), "call: missing output sequence")
    vector = []
    for sequence in range(batch):
        last = by_sequence[sequence][-1]
        rp.require(
            rp.is_int(last["token"]) and last["token"] >= 0,
            "call: invalid output token",
        )
        vector.append(last["token"])
    return vector


def validate_call(
    call: dict[str, Any],
    index: int,
    phase: str,
    rows: list[dict[str, object]],
) -> None:
    rp.require(
        type(call) is dict
        and set(call)
        == {
            "call_index",
            "ended_ns",
            "input_rows",
            "output_rows",
            "phase",
            "shape",
            "started_ns",
        },
        "call: key mismatch",
    )
    rp.require(
        call["call_index"] == index
        and call["phase"] == phase
        and rp.is_int(call["started_ns"])
        and rp.is_int(call["ended_ns"])
        and call["ended_ns"] >= call["started_ns"]
        and call["input_rows"] == rows,
        "call: identity, phase, timing, or input rows mismatch",
    )
    outputs = call["output_rows"]
    rp.require(
        type(outputs) is list and len(outputs) == len(rows),
        "call: output row count mismatch",
    )
    for source, result in zip(rows, outputs):
        rp.require(
            type(result) is dict
            and set(result)
            == {
                "hidden",
                "position",
                "request_id",
                "route_epoch",
                "seq_id",
                "token",
            }
            and result["hidden"] is None
            and all(
                result[field] == source[field]
                for field in ("position", "request_id", "route_epoch", "seq_id")
            )
            and rp.is_int(result["token"])
            and result["token"] >= 0,
            "call: output lineage mismatch",
        )
    positions = sorted({row["position"] for row in rows})
    counts: dict[int, int] = {}
    for row in rows:
        counts[row["seq_id"]] = counts.get(row["seq_id"], 0) + 1
    rp.require(
        call["shape"]
        == {
            "positions": positions,
            "row_count": len(rows),
            "rows_per_sequence": [counts[index] for index in sorted(counts)],
            "sequence_count": len(counts),
        },
        "call: shape mismatch",
    )


def validate_path(
    report: dict[str, Any],
    path: dict[str, Any],
    spec: dict[str, Any],
    inputs: dict[str, Any],
    contract: dict[str, Any],
) -> list[list[int]]:
    rp.require(
        type(path) is dict
        and set(path)
        == {
            "call_end",
            "call_start",
            "continuation",
            "continuation_metrics",
            "delta_metrics",
            "history",
            "history_sha256",
            "identity_base",
            "initial_prediction",
            "name",
            "replay_metrics",
            "state_counts",
        },
        "path: key mismatch",
    )
    rp.require(
        path["name"] == spec["name"]
        and path["history"] == spec["history"]
        and path["identity_base"] == spec["identity_base"]
        and path["state_counts"]
        == {
            "after_remove": 0,
            "before": 0,
            "before_remove": contract["execution"]["batch"],
        },
        "path: identity or state counts mismatch",
    )
    calls = report["calls"]
    cursor = path["call_start"]
    rp.require(rp.is_int(cursor) and cursor >= 0, "path: invalid call start")
    identity_base = spec["identity_base"]
    batch = contract["execution"]["batch"]
    if spec["history"] == "F0_PLUS_DELTA":
        histories = inputs["f0_histories"]
        for start, end in ((0, 8), (8, 11)):
            rows = expected_rows(histories, identity_base, start, end)
            validate_call(calls[cursor], cursor, "REPLAY_F0", rows)
            cursor += 1
        delta_rows = expected_rows(
            inputs["f1_histories"],
            identity_base,
            11,
            12,
        )
        validate_call(
            calls[cursor],
            cursor,
            "INGEST_F1_MINUS_F0",
            delta_rows,
        )
        cursor += 1
    else:
        histories = inputs["f1_histories"]
        chunk = spec["replay_chunk"]
        for start in range(0, 12, chunk):
            end = min(start + chunk, 12)
            rows = expected_rows(histories, identity_base, start, end)
            validate_call(calls[cursor], cursor, "REPLAY_F1", rows)
            cursor += 1
    rp.require(cursor > path["call_start"], "path: no replay calls")
    prediction = output_vector(calls[cursor - 1], batch)
    rp.require(
        path["initial_prediction"] == prediction,
        "path: initial prediction mismatch",
    )
    continuation = [[token] for token in prediction]
    for offset in range(1, contract["execution"]["continuation_tokens"]):
        position = 12 + offset - 1
        rows = [
            {
                "hidden": None,
                "position": position,
                "request_id": identity_base + sequence,
                "route_epoch": identity_base + sequence,
                "seq_id": sequence,
                "token": prediction[sequence],
            }
            for sequence in range(batch)
        ]
        validate_call(
            calls[cursor],
            cursor,
            "AUTONOMOUS_CONTINUATION",
            rows,
        )
        prediction = output_vector(calls[cursor], batch)
        for sequence, token in enumerate(prediction):
            continuation[sequence].append(token)
        cursor += 1
    rp.require(
        cursor == path["call_end"] and path["continuation"] == continuation,
        "path: continuation or call boundary mismatch",
    )
    expected_history_sha = (
        inputs["f0_history_sha256"]
        if spec["history"] == "F0_PLUS_DELTA"
        else inputs["f1_history_sha256"]
    )
    rp.require(path["history_sha256"] == expected_history_sha, "path: history digest")
    return continuation


def validate_raw(
    path: Path,
    run_spec: dict[str, Any],
    contract: dict[str, Any],
    contract_sha256: str,
    inputs: dict[str, Any],
) -> dict[str, list[list[int]]]:
    report, _ = rp.read_canonical(path, f"raw report {run_spec['name']}")
    rp.require(
        set(report)
        == {
            "calls",
            "capture_order",
            "contract_sha256",
            "ended_ns",
            "hello",
            "inputs_sha256",
            "paths",
            "probe",
            "run_name",
            "schema",
            "scope",
            "started_ns",
            "state_counts",
            "status",
            "top2_logit_margins",
        },
        "raw report: key mismatch",
    )
    rp.require(
        report["schema"] == rp.RAW_SCHEMA
        and report["scope"] == "CUDA_ONLY_DIAGNOSTIC"
        and report["status"] == "RAW_CAPTURE_COMPLETE_NO_EQUALITY_EVALUATED"
        and report["run_name"] == run_spec["name"]
        and report["contract_sha256"] == contract_sha256
        and report["inputs_sha256"] == contract["inputs"]["sha256"]
        and report["state_counts"]
        == {"after_all_paths": 0, "before_all_paths": 0}
        and report["capture_order"][-1]
        == "POST_RUN_COMPARISON_BY_SEPARATE_VALIDATOR"
        and report["top2_logit_margins"]["captured"] is False,
        "raw report: contract or capture status mismatch",
    )
    hello = report["hello"]
    rp.require(
        hello["layer_start"] == 0
        and hello["layer_end"] == 48
        and hello["n_layer"] == 48
        and hello["model_sha256"] == contract["artifacts"]["model"]["sha256"],
        "raw report: route hello mismatch",
    )
    rp.require(
        type(report["calls"]) is list
        and type(report["paths"]) is list
        and len(report["paths"]) == len(run_spec["paths"]),
        "raw report: path count mismatch",
    )
    vectors = {}
    previous_end = 0
    for path_value, path_spec in zip(report["paths"], run_spec["paths"]):
        rp.require(path_value["call_start"] == previous_end, "raw report: call gap")
        vector = validate_path(
            report,
            path_value,
            path_spec,
            inputs,
            contract,
        )
        vectors[path_spec["name"]] = vector
        previous_end = path_value["call_end"]
    rp.require(previous_end == len(report["calls"]), "raw report: trailing calls")
    return vectors


def comparison(
    name: str,
    left_name: str,
    left: Sequence[Sequence[int]],
    right_name: str,
    right: Sequence[Sequence[int]],
) -> dict[str, object]:
    mismatch = rp.first_mismatch(left, right)
    return {
        "equal": mismatch is None,
        "first_mismatch": mismatch,
        "left": left_name,
        "name": name,
        "right": right_name,
    }


def verify_acquisition_manifest(root: Path) -> str:
    manifest = root / "SHA256SUMS_ACQUISITION.txt"
    raw = manifest.read_bytes()
    seen = set()
    for line in raw.decode("ascii").splitlines():
        digest, separator, name = line.partition("  ")
        rp.require(
            separator == "  "
            and rp.HEX64.fullmatch(digest) is not None
            and name not in seen,
            "acquisition manifest line is invalid",
        )
        seen.add(name)
        target = (root / name).resolve()
        rp.require(
            root.resolve() in target.parents
            and target.is_file()
            and digest_file(target) == digest,
            f"acquisition artifact changed: {name}",
        )
    required = {"ACQUISITION.json", "ACQUISITION_CONTEXT.json"}
    rp.require(required.issubset(seen), "acquisition manifest is incomplete")
    return rp.sha256(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=rp.HERE / "REPLAY_PARTITION_DIAGNOSTIC.json",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")

    contract, contract_sha256 = rp.load_contract(args.contract)
    inputs_path = rp.repo_path(contract["inputs"]["path"], "contract.inputs.path")
    inputs = rp.load_inputs(inputs_path, contract)
    root = args.run_dir.resolve()
    manifest_sha256 = verify_acquisition_manifest(root)
    acquisition, _ = rp.read_canonical(root / "ACQUISITION.json", "acquisition")
    rp.require(
        acquisition["status"] == "RAW_CAPTURE_COMPLETE_NO_EQUALITY_EVALUATED"
        and acquisition["comparison_evaluated"] is False
        and acquisition["contract_sha256"] == contract_sha256
        and acquisition["inputs_sha256"] == contract["inputs"]["sha256"],
        "acquisition status mismatch",
    )

    run_specs = contract["execution"]["runs"]
    expected_names = [run["name"] for run in run_specs]
    rp.require(
        [Path(item["path"]).parent.name for item in acquisition["raw_reports"]]
        == expected_names
        and [Path(item["path"]).parent.name for item in acquisition["run_records"]]
        == expected_names,
        "acquisition run order mismatch",
    )
    vectors: dict[str, dict[str, list[list[int]]]] = {}
    placements = []
    process_identities = set()
    for index, run_spec in enumerate(run_specs):
        directory = root / run_spec["name"]
        record, record_raw = rp.read_canonical(
            directory / "RUN_RECORD.json",
            f"run record {run_spec['name']}",
        )
        rp.require(
            record["schema"] == rp.RUN_RECORD_SCHEMA
            and record["run_index"] == index
            and record["run_name"] == run_spec["name"]
            and record["contract_sha256"] == contract_sha256
            and record["inputs_sha256"] == contract["inputs"]["sha256"]
            and record["probe_returncode"] == 0
            and record["route_returncodes"] == {"head": 0, "relay": 0, "tail": 0}
            and record["fresh_process"] is True,
            "run record mismatch",
        )
        expected_record = acquisition["run_records"][index]
        rp.require(
            expected_record["sha256"] == rp.sha256(record_raw),
            "run record digest mismatch",
        )
        for name, digest in record["files"].items():
            rp.require(
                digest_file(directory / name) == digest,
                f"run artifact changed: {run_spec['name']}/{name}",
            )
        raw_entry = acquisition["raw_reports"][index]
        rp.require(
            record["files"]["raw_report.json"] == raw_entry["sha256"],
            "raw report binding mismatch",
        )
        processes = {item["role"]: item for item in record["processes"]}
        rp.require(set(processes) == {"head", "relay", "tail"}, "process roles mismatch")
        for process in processes.values():
            identity = (process["pid"], process["start_time_ticks"])
            rp.require(identity not in process_identities, "route process was reused")
            process_identities.add(identity)
        placements.extend([
            validate_placement(
                directory / "cuda_head.log",
                "head",
                processes["head"],
                0,
                30,
            ),
            validate_placement(
                directory / "cuda_tail.log",
                "tail",
                processes["tail"],
                30,
                48,
            ),
        ])
        vectors[run_spec["name"]] = validate_raw(
            directory / "raw_report.json",
            run_spec,
            contract,
            contract_sha256,
            inputs,
        )

    inc1 = vectors["fresh_incremental_r1"]["incremental_8_3_plus_1"]
    inc2 = vectors["fresh_incremental_r2"]["incremental_8_3_plus_1"]
    full1 = vectors["fresh_full_chunk2_r1"]["full_f1_chunk2"]
    full2 = vectors["fresh_full_chunk2_r2"]["full_f1_chunk2"]
    same_inc = vectors["same_process_incremental_then_full"][
        "incremental_8_3_plus_1"
    ]
    same_full = vectors["same_process_incremental_then_full"]["full_f1_chunk2"]
    w5_1 = vectors["fresh_full_8_4_r1"]["full_f1_8_4"]
    w5_2 = vectors["fresh_full_8_4_r2"]["full_f1_8_4"]
    ledger = inputs["w9_continuation"]
    comparisons = [
        comparison("incremental_fresh_repeat", "incremental_r1", inc1, "incremental_r2", inc2),
        comparison("full_chunk2_fresh_repeat", "full_chunk2_r1", full1, "full_chunk2_r2", full2),
        comparison("full_8_4_fresh_repeat", "full_8_4_r1", w5_1, "full_8_4_r2", w5_2),
        comparison("incremental_vs_full_chunk2", "incremental_r1", inc1, "full_chunk2_r1", full1),
        comparison("incremental_vs_full_8_4", "incremental_r1", inc1, "full_8_4_r1", w5_1),
        comparison("full_chunk2_vs_full_8_4", "full_chunk2_r1", full1, "full_8_4_r1", w5_1),
        comparison("same_process_incremental", "same_incremental", same_inc, "fresh_incremental_r1", inc1),
        comparison("same_process_full_chunk2", "same_full_chunk2", same_full, "fresh_full_chunk2_r1", full1),
        comparison("w9_ledger_incremental_r1", "incremental_r1", inc1, "w9_ledger", ledger),
        comparison("w9_ledger_incremental_r2", "incremental_r2", inc2, "w9_ledger", ledger),
        comparison("w9_ledger_same_incremental", "same_incremental", same_inc, "w9_ledger", ledger),
    ]
    by_name = {item["name"]: item for item in comparisons}
    repeatable = all(
        by_name[name]["equal"]
        for name in (
            "incremental_fresh_repeat",
            "full_chunk2_fresh_repeat",
            "full_8_4_fresh_repeat",
        )
    )
    cleanup_exact = all(
        by_name[name]["equal"]
        for name in ("same_process_incremental", "same_process_full_chunk2")
    )
    w9_path_exact = all(
        by_name[name]["equal"]
        for name in (
            "w9_ledger_incremental_r1",
            "w9_ledger_incremental_r2",
            "w9_ledger_same_incremental",
        )
    )
    cross_geometry_exact = all(
        by_name[name]["equal"]
        for name in (
            "incremental_vs_full_chunk2",
            "incremental_vs_full_8_4",
            "full_chunk2_vs_full_8_4",
        )
    )
    findings = []
    if not repeatable:
        findings.append("IDENTICAL_GEOMETRY_NOT_REPEATABLE")
    if repeatable and not cleanup_exact:
        findings.append("SAME_PROCESS_CLEANUP_RESET_BUG")
    if repeatable and not w9_path_exact:
        findings.append("PATH_MATCHED_REPLAY_DIFFERS_FROM_W9_LEDGER")
    if repeatable and not cross_geometry_exact:
        findings.append("BATCH_SHAPE_NUMERICAL_SENSITIVITY")
    if not findings:
        findings.append("NO_MISMATCH_REPRODUCED")
    if "IDENTICAL_GEOMETRY_NOT_REPEATABLE" in findings:
        primary = "BACKEND_OR_KV_STATE_BUG_STOP_INTEGRATION"
    elif "SAME_PROCESS_CLEANUP_RESET_BUG" in findings:
        primary = "CLEANUP_RESET_BUG"
    elif "PATH_MATCHED_REPLAY_DIFFERS_FROM_W9_LEDGER" in findings:
        primary = "REPLAY_DELTA_IMPLEMENTATION_OR_PROVENANCE_BUG"
    elif "BATCH_SHAPE_NUMERICAL_SENSITIVITY" in findings:
        primary = "BATCH_SHAPE_NUMERICAL_SENSITIVITY"
    else:
        primary = "NO_MISMATCH_REPRODUCED"

    analysis = {
        "acquisition_manifest_sha256": manifest_sha256,
        "comparisons": comparisons,
        "contract_sha256": contract_sha256,
        "findings": findings,
        "inputs_sha256": contract["inputs"]["sha256"],
        "interpretation": {
            "cross_geometry_exact": cross_geometry_exact,
            "fresh_geometry_repeatable": repeatable,
            "path_matched_w9_ledger_exact": w9_path_exact,
            "same_process_cleanup_exact": cleanup_exact,
        },
        "placements": placements,
        "primary_diagnosis": primary,
        "qwen_q8_route": "STOP_TASK_QUALITY_FAILED",
        "schema": rp.ANALYSIS_SCHEMA,
        "scope": "CUDA_ONLY_DIAGNOSTIC",
        "status": "REPLAY_PARTITION_DIAGNOSTIC_COMPLETE",
        "top2_logit_margins": "NOT_CAPTURED_PROTOCOL_RETURNS_TOKEN_IDS_ONLY",
    }
    rp.write_atomic(args.output, analysis)
    print(primary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, rp.DiagnosticError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
