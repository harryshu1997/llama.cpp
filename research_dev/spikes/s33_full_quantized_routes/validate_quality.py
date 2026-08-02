#!/usr/bin/env python3
"""Independently bind S33 quantized-route quality and placement evidence."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
S32 = HERE.parent / "s32_quantized_overlap_residency"
if str(S32) not in sys.path:
    sys.path.insert(0, str(S32))

from validate_chain import ChainError, canonical, sha256, validate_worker


SCHEMA = "s33-quantized-quality-case-v1"
PROBE_SCHEMA = "s33-quantized-quality-probe-v1"
CORPUS_SCHEMA = "s33-wikitext-corpus-manifest-v1"
HASH_RE = re.compile(rb"^([0-9a-f]{64})  ([^\r\n]+)\r?\n$")
HEX_RE = re.compile(r"^[0-9a-f]{64}$")
CONFIGS = {
    "Q4_0": {
        "model_sha256": "494518c2262a26e2a607af0e40bca11c4de5a0b108e7c21308906dbbfb1c6f8c",
        "route": [["OP12", 0, 4], ["OP15", 4, 24], ["CUDA", 24, 48]],
        "reference_route": [["CUDA", 0, 24], ["CUDA", 24, 48]],
    },
    "Q8_0": {
        "model_sha256": "7b56cbd0e0d96d5c8d7df9c21b39d67264e58cbabf7a96d4681eff8b3d492848",
        "route": [["OP12", 0, 2], ["OP15", 2, 16], ["CUDA", 16, 48]],
        "reference_route": [["CUDA", 0, 16], ["CUDA", 16, 48]],
    },
}


def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ChainError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode_json(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw, object_pairs_hook=strict_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ChainError(f"invalid {label} JSON") from exc
    if type(value) is not dict:
        raise ChainError(f"{label} must be an object")
    return value


def require_int(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ChainError(f"{label} must be an integer >= {minimum}")
    return value


def parse_hash_record(raw: bytes, digest: str, expected_path: str, label: str) -> None:
    match = HASH_RE.fullmatch(raw)
    if match is None:
        raise ChainError(f"{label} hash record is malformed")
    if match.group(1).decode("ascii") != digest:
        raise ChainError(f"{label} model digest mismatch")
    if match.group(2).decode("ascii") != expected_path:
        raise ChainError(f"{label} model path mismatch")


def validate_tokens(value: object, label: str) -> list[list[int]]:
    if type(value) is not list or len(value) != 128:
        raise ChainError(f"{label} request count mismatch")
    for row in value:
        if type(row) is not list or len(row) != 8:
            raise ChainError(f"{label} output length mismatch")
        if any(type(token) is not int or token < 0 for token in row):
            raise ChainError(f"{label} contains an invalid token")
    return value


def validate_memory(value: object) -> tuple[list[str], bool]:
    if type(value) is not dict or set(value) != {
        "head_before", "head_after", "middle_before", "middle_after",
    }:
        raise ChainError("phone memory evidence set mismatch")
    swap_failures: list[str] = []
    identity_bound = True
    for device in ("head", "middle"):
        identities: set[tuple[object, object]] = set()
        for phase in ("before", "after"):
            snapshot = value[f"{device}_{phase}"]
            if type(snapshot) is not dict or not HEX_RE.fullmatch(str(snapshot.get("raw_sha256"))):
                raise ChainError("phone memory snapshot is malformed")
            process = snapshot.get("process_kib")
            system = snapshot.get("system_kib")
            if type(process) is not dict or type(system) is not dict:
                raise ChainError("phone memory fields are malformed")
            swap = require_int(process.get("VmSwap"), "VmSwap")
            require_int(process.get("VmHWM"), "VmHWM", 1)
            require_int(process.get("VmRSS"), "VmRSS", 1)
            require_int(system.get("MemAvailable"), "MemAvailable", 1)
            if swap:
                swap_failures.append(f"{device.upper()}_{phase.upper()}_SWAP_NONZERO")
            pid = snapshot.get("pid")
            serial = snapshot.get("adb_serial")
            if type(pid) is not int or pid <= 0 or type(serial) is not str or not serial:
                identity_bound = False
            else:
                if process.get("Pid") != pid:
                    raise ChainError("phone memory PID differs from parsed status")
                identities.add((serial, pid))
        if identities and len(identities) != 1:
            raise ChainError("phone memory identity changed during execution")
    return swap_failures, identity_bound


def validate_hello(value: object, start: int, end: int, terminal: bool, label: str) -> None:
    if type(value) is not dict:
        raise ChainError(f"{label} hello is malformed")
    if (value.get("layer_start"), value.get("layer_end")) != (start, end):
        raise ChainError(f"{label} hello range mismatch")
    if value.get("n_layer") != 48 or value.get("n_embd") != 3840:
        raise ChainError(f"{label} hello model mismatch")
    if require_int(value.get("max_streams"), f"{label} max_streams", 32) < 32:
        raise ChainError(f"{label} cannot execute B32")
    if min(
        require_int(value.get("n_batch"), f"{label} n_batch", 32),
        require_int(value.get("n_ubatch"), f"{label} n_ubatch", 32),
    ) < 32:
        raise ChainError(f"{label} token capacity is too small")
    capabilities = require_int(value.get("capabilities"), f"{label} capabilities")
    if bool(capabilities & 0x10) != terminal:
        raise ChainError(f"{label} terminal capability mismatch")


def validate_probe(
    raw: bytes, quantization: str, corpus_raw: bytes, manifest_raw: bytes,
) -> tuple[dict[str, object], bool, bool]:
    config = CONFIGS[quantization]
    probe = decode_json(raw, "probe")
    if canonical(probe) != raw:
        raise ChainError("probe is not canonical JSONL")
    if probe.get("schema") != PROBE_SCHEMA or probe.get("scheduler_eligible") is not False:
        raise ChainError("raw probe identity or authorization mismatch")
    if probe.get("quantization") != quantization:
        raise ChainError("probe quantization mismatch")
    if probe.get("model_sha256") != config["model_sha256"]:
        raise ChainError("probe model digest mismatch")
    if probe.get("route") != config["route"] or probe.get("reference_route") != config["reference_route"]:
        raise ChainError("probe route mismatch")
    if (
        probe.get("batch") != 32 or probe.get("cohorts") != 4
        or probe.get("prompt_tokens") != 8 or probe.get("output_tokens") != 8
    ):
        raise ChainError("probe shape mismatch")

    manifest = decode_json(manifest_raw, "corpus manifest")
    if canonical(manifest) != manifest_raw or manifest.get("schema") != CORPUS_SCHEMA:
        raise ChainError("corpus manifest is not canonical or has the wrong schema")
    if manifest.get("model_sha256") != config["model_sha256"]:
        raise ChainError("corpus model digest mismatch")
    if manifest.get("output_sha256") != sha256(corpus_raw) or manifest.get("output_records") != 128:
        raise ChainError("corpus output binding mismatch")
    if (
        probe.get("corpus_sha256") != sha256(corpus_raw)
        or probe.get("corpus_manifest_sha256") != sha256(manifest_raw)
    ):
        raise ChainError("probe corpus binding mismatch")

    physical = validate_tokens(probe.get("physical_tokens"), "physical")
    reference = validate_tokens(probe.get("reference_tokens"), "reference")
    if probe.get("physical_tokens_sha256") != sha256(canonical(physical)):
        raise ChainError("physical token digest mismatch")
    if probe.get("reference_tokens_sha256") != sha256(canonical(reference)):
        raise ChainError("reference token digest mismatch")
    first = sum(left[0] == right[0] for left, right in zip(physical, reference))
    exact = sum(left == right for left, right in zip(physical, reference))
    decisions = sum(
        left == right
        for left_row, right_row in zip(physical, reference)
        for left, right in zip(left_row, right_row)
    )
    quality = {
        "first_token_matches": first,
        "first_token_agreement": first / 128,
        "token_decision_matches": decisions,
        "token_decision_agreement": decisions / 1024,
        "exact_sequence_matches": exact,
        "exact_sequence_agreement": exact / 128,
    }
    if probe.get("quality") != quality:
        raise ChainError("quality summary was not recomputed correctly")
    thresholds = {
        "min_first_token_agreement": 0.95,
        "min_token_decision_agreement": 0.95,
        "min_exact_sequence_agreement": 0.80,
    }
    if probe.get("thresholds") != thresholds:
        raise ChainError("quality thresholds differ from the frozen contract")
    quality_pass = (
        quality["first_token_agreement"] >= 0.95
        and quality["token_decision_agreement"] >= 0.95
        and quality["exact_sequence_agreement"] >= 0.80
    )
    if probe.get("quality_gate_pass") is not quality_pass:
        raise ChainError("quality gate label is inconsistent")

    swap_failures, identity_bound = validate_memory(probe.get("memory"))
    if probe.get("resource_failures") != swap_failures:
        raise ChainError("resource failure list is inconsistent")
    resource_pass = not swap_failures
    if probe.get("resource_gate_pass") is not resource_pass:
        raise ChainError("resource gate label is inconsistent")
    expected_status = (
        "QUALITY_PASS" if quality_pass and resource_pass
        else "QUALITY_FAIL" if not quality_pass
        else "RESOURCE_FAIL"
    )
    if probe.get("status") != expected_status:
        raise ChainError("probe status is inconsistent")

    timings = probe.get("timings")
    if type(timings) is not list or len(timings) != 4:
        raise ChainError("timing cohort count mismatch")
    physical_totals: list[int] = []
    reference_totals: list[int] = []
    for cohort, row in enumerate(timings):
        if type(row) is not dict or row.get("cohort") != cohort:
            raise ChainError("timing cohort identity mismatch")
        physical_us = row.get("physical_us")
        reference_us = row.get("reference_us")
        if type(physical_us) is not dict or set(physical_us) != {"head_us", "middle_us", "tail_us"}:
            raise ChainError("physical timing fields mismatch")
        if type(reference_us) is not dict or set(reference_us) != {"head_us", "tail_us"}:
            raise ChainError("reference timing fields mismatch")
        physical_totals.append(sum(require_int(value, "physical time", 1) for value in physical_us.values()))
        reference_totals.append(sum(require_int(value, "reference time", 1) for value in reference_us.values()))
    if probe.get("physical_cohort_median_us") != statistics.median(physical_totals):
        raise ChainError("physical median was not recomputed correctly")
    if probe.get("reference_cohort_median_us") != statistics.median(reference_totals):
        raise ChainError("reference median was not recomputed correctly")

    route = config["route"]
    hellos = probe.get("hellos")
    if type(hellos) is not dict or set(hellos) != {"head", "middle", "tail", "reference"}:
        raise ChainError("worker hello set mismatch")
    validate_hello(hellos["head"], route[0][1], route[0][2], False, "head")
    validate_hello(hellos["middle"], route[1][1], route[1][2], False, "middle")
    validate_hello(hellos["tail"], route[2][1], route[2][2], True, "tail")
    validate_hello(hellos["reference"], 0, route[2][1], False, "reference")
    return probe, quality_pass, identity_bound


def memory_matches_workers(probe: dict[str, object], worker_pids: dict[str, int]) -> bool:
    memory = probe["memory"]
    assert isinstance(memory, dict)
    for device in ("head", "middle"):
        expected_pid = worker_pids[device]
        for phase in ("before", "after"):
            snapshot = memory[f"{device}_{phase}"]
            if type(snapshot) is not dict or snapshot.get("pid") != expected_pid:
                return False
    return True


def validate_case(
    quantization: str, result_dir: Path, corpus_path: Path, manifest_path: Path,
    host_model_path: str, remote_model_path: str,
) -> dict[str, object]:
    config = CONFIGS[quantization]
    paths = {
        "probe": result_dir / "probe.json",
        "op12_log": result_dir / "op12.log",
        "op15_log": result_dir / "op15.log",
        "tail_log": result_dir / "cuda_tail.log",
        "reference_log": result_dir / "cuda_reference.log",
        "host_model_hash": result_dir / "host_model.sha256",
        "op12_model_hash": result_dir / "op12_model.sha256",
        "op15_model_hash": result_dir / "op15_model.sha256",
        "corpus": corpus_path,
        "corpus_manifest": manifest_path,
    }
    raw = {name: path.read_bytes() for name, path in paths.items()}
    probe, quality_pass, identity_bound = validate_probe(
        raw["probe"], quantization, raw["corpus"], raw["corpus_manifest"],
    )
    digest = config["model_sha256"]
    parse_hash_record(raw["host_model_hash"], digest, host_model_path, "host")
    parse_hash_record(raw["op12_model_hash"], digest, remote_model_path, "OP12")
    parse_hash_record(raw["op15_model_hash"], digest, remote_model_path, "OP15")

    route = config["route"]
    head_session, _, _ = validate_worker(
        raw["op12_log"], None, route[0][1], route[0][2], "HTP0", 1920, "OP12",
    )
    middle_session, _, _ = validate_worker(
        raw["op15_log"], None, route[1][1], route[1][2], "HTP0", 1920, "OP15",
    )
    validate_worker(raw["tail_log"], None, route[2][1], 48, "CUDA0", 3840, "CUDA tail")
    validate_worker(raw["reference_log"], None, 0, route[2][1], "CUDA0", 1920, "CUDA reference")
    worker_pids = {
        "head": require_int(head_session.get("worker_pid"), "OP12 PID", 1),
        "middle": require_int(middle_session.get("worker_pid"), "OP15 PID", 1),
    }
    resource_bound = identity_bound and memory_matches_workers(probe, worker_pids)
    resource_pass = probe["resource_gate_pass"] is True and resource_bound
    eligible = quality_pass and resource_pass
    status = (
        "QUALITY_ROUTE_PASS" if eligible
        else "QUALITY_FAIL" if not quality_pass
        else "RESOURCE_EVIDENCE_FAIL"
    )
    return {
        "schema": SCHEMA,
        "status": status,
        "scheduler_eligible": eligible,
        "quantization": quantization,
        "model_sha256": digest,
        "route": route,
        "quality": probe["quality"],
        "quality_gate_pass": quality_pass,
        "resource_claim_from_probe": probe["resource_gate_pass"],
        "resource_identity_bound": resource_bound,
        "resource_gate_pass": resource_pass,
        "artifacts": {name: sha256(value) for name, value in sorted(raw.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--host-model", required=True)
    parser.add_argument("--remote-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    try:
        result = validate_case(
            args.quantization, args.result_dir, args.corpus, args.corpus_manifest,
            args.host_model, args.remote_model,
        )
    except (ChainError, OSError, ValueError) as exc:
        print(f"S33_QUALITY_BIND_FAIL: {exc}", file=sys.stderr)
        return 2
    args.output.write_bytes(canonical(result))
    print(canonical(result).decode("ascii"), end="")
    return 0 if result["scheduler_eligible"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
