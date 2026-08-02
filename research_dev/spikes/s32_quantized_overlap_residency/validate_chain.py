#!/usr/bin/env python3
"""Bind the S32 Q8 chain probe to model, placement, and process evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


SCHEMA = "s32-q8-chain-case-v1"
PROBE_SCHEMA = "s32-q8-chain-token-proof-v2"
MODEL_SHA256 = "7b56cbd0e0d96d5c8d7df9c21b39d67264e58cbabf7a96d4681eff8b3d492848"
HASH_RE = re.compile(rb"^([0-9a-f]{64})  ([^\r\n]+)\r?\n$")


class ChainError(ValueError):
    pass


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


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require_int(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ChainError(f"{label} must be an integer >= {minimum}")
    return value


def one_record(log: bytes, prefix: bytes, label: str) -> dict[str, object]:
    rows = [line[len(prefix):] for line in log.splitlines() if line.startswith(prefix)]
    if len(rows) != 1:
        raise ChainError(f"expected exactly one {label}")
    return decode_json(rows[0], label)


def parse_hash_record(raw: bytes, expected_path: str, label: str) -> None:
    match = HASH_RE.fullmatch(raw)
    if match is None:
        raise ChainError(f"{label} hash record is malformed")
    if match.group(1).decode("ascii") != MODEL_SHA256:
        raise ChainError(f"{label} model digest mismatch")
    if match.group(2).decode("ascii") != expected_path:
        raise ChainError(f"{label} model path mismatch")


def parse_status(raw: bytes, label: str) -> dict[str, int]:
    try:
        text = raw.decode("ascii")
    except UnicodeError as exc:
        raise ChainError(f"{label} process status is not ASCII") from exc
    result: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, rest = line.partition(":")
        if not separator or key not in {"Pid", "VmHWM", "VmRSS", "VmSwap"}:
            continue
        pieces = rest.split()
        if not pieces or not pieces[0].isdigit():
            raise ChainError(f"{label} has malformed {key}")
        if key != "Pid" and (len(pieces) != 2 or pieces[1] != "kB"):
            raise ChainError(f"{label} has malformed {key} units")
        result[key] = int(pieces[0])
    if set(result) != {"Pid", "VmHWM", "VmRSS", "VmSwap"}:
        raise ChainError(f"{label} process status is incomplete")
    return result


def validate_compute_map(value: object, expected_backend: str) -> int:
    if type(value) is not dict or not value:
        raise ChainError("compute placement map must be a nonempty object")
    total = 0
    for op, buffers in value.items():
        if type(op) is not str or not op or type(buffers) is not dict or not buffers:
            raise ChainError("compute placement map is malformed")
        for backend, count in buffers.items():
            require_int(count, f"{op}/{backend} count", 1)
            total += count
            if backend == expected_backend:
                continue
            allowed_host = expected_backend == "CUDA0" and backend == "CUDA_Host"
            allowed_cpu = expected_backend == "HTP0" and backend == "CPU"
            if not (op == "GET_ROWS" and (allowed_host or allowed_cpu)):
                raise ChainError(f"unexpected placement {op}/{backend}")
    return total


def validate_worker(
    log_raw: bytes,
    status_raw: bytes | None,
    start: int,
    end: int,
    expected_backend: str,
    expected_steps: int,
    label: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, int] | None]:
    session = one_record(log_raw, b"SESSIONCERT ", f"{label} SESSIONCERT")
    placement = one_record(log_raw, b"PLACEMENTCERT ", f"{label} PLACEMENTCERT")
    for record in (session, placement):
        if record.get("layer_start") != start or record.get("layer_end") != end:
            raise ChainError(f"{label} layer range mismatch")
        if record.get("n_layer") != 48 or record.get("missing_buffer_compute_nodes") != 0:
            raise ChainError(f"{label} model or buffer evidence mismatch")
    if session.get("expected_backend") != expected_backend:
        raise ChainError(f"{label} expected backend mismatch")
    if session.get("session_end") != "STOP" or session.get("placement_status") != "SCHEDULED_PLACEMENT_OK":
        raise ChainError(f"{label} session did not stop cleanly")
    if session.get("steps_session") != expected_steps:
        raise ChainError(f"{label} session row count mismatch")
    if placement.get("status") != "SCHEDULED_PLACEMENT_OK" or placement.get("run_rc") != 0:
        raise ChainError(f"{label} placement did not pass")
    if session.get("compute_by_op_and_buffer") != placement.get("compute_by_op_and_buffer"):
        raise ChainError(f"{label} session and placement maps differ")
    total = validate_compute_map(placement.get("compute_by_op_and_buffer"), expected_backend)
    if placement.get("compute_nodes") != total:
        raise ChainError(f"{label} compute count does not balance")
    status = None
    if status_raw is not None:
        status = parse_status(status_raw, label)
        if status["Pid"] != session.get("worker_pid") or status["Pid"] != placement.get("pid"):
            raise ChainError(f"{label} process identity mismatch")
    return session, placement, status


def validate_timing(rows: object, keys: tuple[str, ...], steps: int, label: str) -> list[int]:
    if type(rows) is not list or len(rows) != steps:
        raise ChainError(f"{label} timing length mismatch")
    totals: list[int] = []
    for row in rows:
        if type(row) is not dict or set(row) != set(keys):
            raise ChainError(f"{label} timing row mismatch")
        values = [require_int(row[key], f"{label}/{key}", 1) for key in keys]
        totals.append(sum(values))
    return totals


def validate_probe(raw: bytes) -> dict[str, object]:
    probe = decode_json(raw, "probe")
    if raw != canonical(probe):
        raise ChainError("probe is not canonical JSONL")
    if probe.get("schema") != PROBE_SCHEMA or probe.get("status") not in {
        "TOKEN_EXACT_PASS", "TOKEN_EXACT_FAIL",
    }:
        raise ChainError("probe did not complete the v2 token gate")
    if probe.get("scheduler_eligible") is not False:
        raise ChainError("raw probe must not self-authorize scheduler eligibility")
    if probe.get("model_sha256") != MODEL_SHA256:
        raise ChainError("probe model digest mismatch")
    if probe.get("batch") != 32 or probe.get("steps") != 4:
        raise ChainError("probe shape mismatch")
    if probe.get("initial_tokens") != list(range(2, 34)):
        raise ChainError("probe does not contain 32 distinct frozen inputs")
    if probe.get("route") != [["OP12", 0, 4], ["OP15", 4, 16], ["CUDA", 16, 48]]:
        raise ChainError("physical route mismatch")
    if probe.get("reference_route") != [["CUDA", 0, 16], ["CUDA", 16, 48]]:
        raise ChainError("reference route mismatch")
    chain = probe.get("chain_tokens")
    reference = probe.get("reference_tokens")
    if type(chain) is not list or len(chain) != 32 or type(reference) is not list or len(reference) != 32:
        raise ChainError("token output count is invalid")
    for rows in (chain, reference):
        for row in rows:
            if type(row) is not list or len(row) != 4 or any(type(token) is not int or token < 0 for token in row):
                raise ChainError("token output dimensions are invalid")
    matching_requests = sum(left == right for left, right in zip(chain, reference))
    if probe.get("matching_requests") != matching_requests:
        raise ChainError("matching request count mismatch")
    token_pass = matching_requests == 32
    if (probe.get("status") == "TOKEN_EXACT_PASS") != token_pass:
        raise ChainError("token status is inconsistent with the outputs")
    chain_digest = sha256(canonical(chain))
    reference_digest = sha256(canonical(reference))
    if probe.get("chain_tokens_sha256") != chain_digest or probe.get("reference_tokens_sha256") != reference_digest:
        raise ChainError("token output digest mismatch")
    chain_totals = validate_timing(probe.get("chain_stage_us"), ("head_us", "middle_us", "tail_us"), 4, "chain")
    reference_totals = validate_timing(probe.get("reference_stage_us"), ("head_us", "tail_us"), 4, "reference")
    if probe.get("chain_step_median_us") != statistics.median(chain_totals):
        raise ChainError("chain median was not recomputed correctly")
    if probe.get("reference_step_median_us") != statistics.median(reference_totals):
        raise ChainError("reference median was not recomputed correctly")
    hellos = probe.get("hellos")
    expected = {
        "head": (0, 4, False),
        "middle": (4, 16, False),
        "tail": (16, 48, True),
        "reference": (0, 16, False),
    }
    if type(hellos) is not dict or set(hellos) != set(expected):
        raise ChainError("worker hello set mismatch")
    for label, (start, end, terminal) in expected.items():
        hello = hellos[label]
        if type(hello) is not dict:
            raise ChainError(f"{label} hello is malformed")
        if (hello.get("layer_start"), hello.get("layer_end")) != (start, end):
            raise ChainError(f"{label} hello range mismatch")
        if hello.get("n_layer") != 48 or hello.get("n_embd") != 3840:
            raise ChainError(f"{label} hello model mismatch")
        if require_int(hello.get("max_streams"), f"{label} max_streams", 32) < 32:
            raise ChainError(f"{label} has too few streams")
        capabilities = require_int(hello.get("capabilities"), f"{label} capabilities")
        if bool(capabilities & 0x10) != terminal:
            raise ChainError(f"{label} terminal capability mismatch")
    return probe


def validate_chain(
    probe_path: Path,
    op12_log_path: Path,
    op15_log_path: Path,
    tail_log_path: Path,
    reference_log_path: Path,
    op12_status_path: Path,
    op15_status_path: Path,
    op12_hash_path: Path,
    op15_hash_path: Path,
    host_hash_path: Path,
    remote_model_path: str,
    host_model_path: str,
) -> dict[str, object]:
    paths = {
        "probe": probe_path,
        "op12_log": op12_log_path,
        "op15_log": op15_log_path,
        "tail_log": tail_log_path,
        "reference_log": reference_log_path,
        "op12_status": op12_status_path,
        "op15_status": op15_status_path,
        "op12_hash": op12_hash_path,
        "op15_hash": op15_hash_path,
        "host_hash": host_hash_path,
    }
    raw = {name: path.read_bytes() for name, path in paths.items()}
    probe = validate_probe(raw["probe"])
    parse_hash_record(raw["op12_hash"], remote_model_path, "OP12")
    parse_hash_record(raw["op15_hash"], remote_model_path, "OP15")
    parse_hash_record(raw["host_hash"], host_model_path, "host")
    _, _, op12_status = validate_worker(
        raw["op12_log"], raw["op12_status"], 0, 4, "HTP0", 128, "OP12",
    )
    _, _, op15_status = validate_worker(
        raw["op15_log"], raw["op15_status"], 4, 16, "HTP0", 128, "OP15",
    )
    validate_worker(raw["tail_log"], None, 16, 48, "CUDA0", 256, "CUDA tail")
    validate_worker(raw["reference_log"], None, 0, 16, "CUDA0", 128, "CUDA reference")
    assert op12_status is not None and op15_status is not None
    resource_failures = [
        label for label, status in (("OP12_SWAP_NONZERO", op12_status), ("OP15_SWAP_NONZERO", op15_status))
        if status["VmSwap"] != 0
    ]
    token_pass = probe["status"] == "TOKEN_EXACT_PASS"
    chain_tokens = probe["chain_tokens"]
    reference_tokens = probe["reference_tokens"]
    matching_token_decisions = sum(
        left == right
        for chain_row, reference_row in zip(chain_tokens, reference_tokens)
        for left, right in zip(chain_row, reference_row)
    )
    if not token_pass:
        status = "TOKEN_EXACT_FAIL"
    elif resource_failures:
        status = "TOKEN_EXACT_PASS_RESOURCE_BLOCKED"
    else:
        status = "TOKEN_EXACT_PASS"
    return {
        "schema": SCHEMA,
        "status": status,
        "scheduler_eligible": False,
        "scope": "B32_FOUR_STEP_TOKEN_CORRECTNESS_ONLY",
        "model_sha256": MODEL_SHA256,
        "batch": 32,
        "steps": 4,
        "matching_requests": probe["matching_requests"],
        "matching_token_decisions": matching_token_decisions,
        "chain_step_median_us": probe["chain_step_median_us"],
        "reference_step_median_us": probe["reference_step_median_us"],
        "resource_failures": resource_failures,
        "phone_process_kib": {"OP12": op12_status, "OP15": op15_status},
        "artifacts": {name: sha256(value) for name, value in raw.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--op12-log", type=Path, required=True)
    parser.add_argument("--op15-log", type=Path, required=True)
    parser.add_argument("--tail-log", type=Path, required=True)
    parser.add_argument("--reference-log", type=Path, required=True)
    parser.add_argument("--op12-status", type=Path, required=True)
    parser.add_argument("--op15-status", type=Path, required=True)
    parser.add_argument("--op12-hash", type=Path, required=True)
    parser.add_argument("--op15-hash", type=Path, required=True)
    parser.add_argument("--host-hash", type=Path, required=True)
    parser.add_argument("--remote-model", required=True)
    parser.add_argument("--host-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    try:
        result = validate_chain(
            args.probe,
            args.op12_log,
            args.op15_log,
            args.tail_log,
            args.reference_log,
            args.op12_status,
            args.op15_status,
            args.op12_hash,
            args.op15_hash,
            args.host_hash,
            args.remote_model,
            args.host_model,
        )
    except (OSError, ChainError) as exc:
        print(f"S32_CHAIN_FAIL: {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(result))
    print(canonical(result).decode("ascii"), end="")
    return 0 if result["status"] != "TOKEN_EXACT_FAIL" else 3


if __name__ == "__main__":
    raise SystemExit(main())
