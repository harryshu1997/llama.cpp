#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_EVIDENCE = HERE / "results" / "w0_route_screen"
DEFAULT_BATCH_EVIDENCE = HERE / "results" / "w0_qwen_batch_wifi"
DEFAULT_SHARDS = HERE / "SHARD_MANIFEST.json"
DEFAULT_OUTPUT = HERE / "CURRENT_ROUTE_READINESS.json"

SHA256_RE = re.compile(r"[0-9a-f]{64}")

MODEL_SPECS = {
    "gemma-4-12b-it-q4_0": {
        "cuda": "gemma_cuda_control.err",
        "head": "op15_gemma_headnet.log",
        "n_layer": 48,
        "tail": "op12_gemma_tailnet.log",
    },
    "qwen3-14b-q4_k_m": {
        "cuda": "qwen_cuda_control.err",
        "head": "op15_qwen_headnet.log",
        "n_layer": 40,
        "tail": "op12_qwen_tailnet.log",
    },
}

EXPECTED_ARTIFACTS = {
    "gemma_cuda_control.err",
    "gemma_cuda_control.out",
    "op12_gemma_tailnet.log",
    "op12_qwen_tailnet.log",
    "op15_gemma_headnet.log",
    "op15_qwen_headnet.log",
    "qwen_cuda_control.err",
    "qwen_cuda_control.out",
}


class ReadinessError(RuntimeError):
    pass


SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "s39_qwen_batch_summary",
    HERE / "summarize_qwen_batch_wifi.py",
)
if SUMMARY_SPEC is None or SUMMARY_SPEC.loader is None:
    raise RuntimeError("cannot load S39 batch evidence reducer")
BATCH_SUMMARY = importlib.util.module_from_spec(SUMMARY_SPEC)
SUMMARY_SPEC.loader.exec_module(BATCH_SUMMARY)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessError(message)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_int(value: Any, field: str, minimum: int = 0) -> int:
    require(is_int(value), f"{field}: expected integer")
    require(value >= minimum, f"{field}: expected >= {minimum}")
    return value


def require_string(value: Any, field: str) -> str:
    require(isinstance(value, str) and bool(value), f"{field}: expected string")
    return value


def require_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field}: expected object")
    actual = set(value)
    require(
        actual == expected,
        f"{field}: keys differ; missing={sorted(expected - actual)}, "
        f"unknown={sorted(actual - expected)}",
    )
    return value


def reject_constant(value: str) -> None:
    raise ReadinessError(f"invalid JSON constant {value}")


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_json(raw: bytes, field: str) -> Any:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ReadinessError(f"{field}: expected ASCII") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ReadinessError(f"{field}: invalid JSON: {error}") from error


def read_bytes(path: Path, field: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ReadinessError(f"{field}: cannot read {path}: {error}") from error


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def load_artifacts(evidence_dir: Path) -> tuple[dict[str, str], dict[str, bytes]]:
    raw = read_bytes(evidence_dir / "SHA256SUMS.txt", "SHA256SUMS")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ReadinessError("SHA256SUMS: expected ASCII") from error
    require(bool(lines), "SHA256SUMS: empty")

    result = {}
    for line_number, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        require(match is not None, f"SHA256SUMS:{line_number}: invalid record")
        digest, name = match.groups()
        require(name not in result, f"SHA256SUMS:{line_number}: duplicate artifact")
        result[name] = digest
    require(set(result) == EXPECTED_ARTIFACTS, "SHA256SUMS: artifact set mismatch")

    artifacts = {}
    for name, expected in result.items():
        raw = read_bytes(evidence_dir / name, f"artifact.{name}")
        actual = sha256(raw)
        require(actual == expected, f"artifact.{name}: SHA-256 mismatch")
        artifacts[name] = raw
    return result, artifacts


def extract_record(raw: bytes, prefix: str, field: str) -> dict[str, Any]:
    prefix_raw = prefix.encode("ascii")
    matches = [line[len(prefix_raw):] for line in raw.splitlines() if line.startswith(prefix_raw)]
    require(len(matches) == 1, f"{field}: expected exactly one {prefix.strip()}")
    value = parse_json(matches[0], field)
    require(isinstance(value, dict), f"{field}: expected object")
    return value


def placement_problems(
    record: dict[str, Any],
    *,
    role: str,
    mode: str,
    layer_start: int,
    layer_end: int,
    n_layer: int,
) -> list[str]:
    problems = []
    expected_scalars = {
        "schema": "layersplit-scheduled-placement-v2",
        "role": role,
        "mode": mode,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "n_layer": n_layer,
        "run_rc": 0,
        "status": "SCHEDULED_PLACEMENT_OK",
        "missing_buffer_compute_nodes": 0,
    }
    for field, expected in expected_scalars.items():
        if record.get(field) != expected or type(record.get(field)) is not type(expected):
            problems.append(f"{role}.{field}")

    compute_nodes = record.get("compute_nodes")
    if not is_int(compute_nodes) or compute_nodes <= 0:
        problems.append(f"{role}.compute_nodes")

    by_buffer = record.get("compute_by_buffer_type")
    if not isinstance(by_buffer, dict) or not is_int(by_buffer.get("OpenCL")) or by_buffer["OpenCL"] <= 0:
        problems.append(f"{role}.opencl_compute")
    if isinstance(by_buffer, dict):
        unexpected = set(by_buffer) - {"CPU", "OpenCL"}
        if unexpected:
            problems.append(f"{role}.unexpected_backend")
        if (
            any(not is_int(value) or value < 0 for value in by_buffer.values())
            or sum(by_buffer.values()) != compute_nodes
        ):
            problems.append(f"{role}.backend_tally")

    by_op = record.get("compute_by_op")
    if (
        not isinstance(by_op, dict)
        or any(not is_int(value) or value < 0 for value in by_op.values())
        or sum(by_op.values()) != compute_nodes
    ):
        problems.append(f"{role}.op_tally")

    by_op_buffer = record.get("compute_by_op_and_buffer")
    if not isinstance(by_op_buffer, dict):
        problems.append(f"{role}.op_buffer_map")
    else:
        op_buffer_total = 0
        for op, backends in by_op_buffer.items():
            if not isinstance(backends, dict) or any(
                not is_int(value) or value < 0 for value in backends.values()
            ):
                problems.append(f"{role}.op_buffer_tally")
                continue
            op_buffer_total += sum(backends.values())
            if "CPU" in backends and op != "GET_ROWS":
                problems.append(f"{role}.cpu_fallback.{op}")
        if op_buffer_total != compute_nodes:
            problems.append(f"{role}.op_buffer_tally")
    return sorted(set(problems))


def validate_result(record: dict[str, Any], field: str) -> list[int]:
    require(record.get("schema") == "layersplit-headnet-result-v1", f"{field}: schema")
    require(record.get("status") == "completed", f"{field}: incomplete")
    prompt_tokens = require_int(record.get("prompt_tokens"), f"{field}.prompt_tokens", 1)
    output_tokens = require_int(record.get("output_tokens"), f"{field}.output_tokens", 1)
    token_ids = record.get("token_ids")
    require(
        isinstance(token_ids, list)
        and len(token_ids) == output_tokens
        and all(is_int(token) and token >= 0 for token in token_ids),
        f"{field}.token_ids",
    )
    decode_steps = record.get("decode_step_us")
    require(
        isinstance(decode_steps, list)
        and len(decode_steps) == output_tokens - 1
        and all(is_int(value) and value > 0 for value in decode_steps),
        f"{field}.decode_step_us",
    )
    for name in ("ttft_us", "service_wall_us", "exchange_count", "activation_bytes", "control_bytes"):
        require_int(record.get(name), f"{field}.{name}", 1)
    require(
        record["exchange_count"] == prompt_tokens + output_tokens - 1,
        f"{field}.exchange_count",
    )
    return token_ids


def validate_cuda(record: dict[str, Any], field: str) -> list[int]:
    expected = {
        "status": "ok",
        "route": "SERVER_ONLY",
        "request_index": 0,
        "batch_index": 0,
        "batch_size": 1,
        "stream_index": 0,
    }
    for name, value in expected.items():
        require(
            record.get(name) == value and type(record.get(name)) is type(value),
            f"{field}.{name}",
        )
    generated = require_int(record.get("generated_tokens"), f"{field}.generated_tokens", 1)
    requested = require_int(record.get("requested_tokens"), f"{field}.requested_tokens", 1)
    require(
        requested == generated,
        f"{field}: incomplete generation",
    )
    tokens = record.get("token_ids")
    require(
        isinstance(tokens, list)
        and len(tokens) == generated
        and all(is_int(token) and token >= 0 for token in tokens),
        f"{field}.token_ids",
    )
    for name in ("prefill_us", "decode_us", "request_wall_us"):
        require_int(record.get(name), f"{field}.{name}", 1)
    return tokens


def load_shards(path: Path) -> dict[str, Any]:
    root = parse_json(read_bytes(path, "shard_manifest"), "shard_manifest")
    require(isinstance(root, dict), "shard_manifest: expected object")
    require(root.get("schema_version") == 1, "shard_manifest: unsupported version")
    models = root.get("models")
    require(isinstance(models, dict), "shard_manifest.models: expected object")
    return models


def load_batch_certificate(
    batch_evidence_dir: Path,
    cuda_path: Path,
    shard_manifest_path: Path,
) -> tuple[dict[str, Any], str]:
    certificate_path = batch_evidence_dir / "qwen_batch_certificate.json"
    raw = read_bytes(certificate_path, "batch_certificate")
    try:
        derived = BATCH_SUMMARY.build(
            batch_evidence_dir,
            cuda_path,
            shard_manifest_path,
        )
    except BATCH_SUMMARY.BatchEvidenceError as error:
        raise ReadinessError(f"batch_certificate: {error}") from error
    require(
        BATCH_SUMMARY.canonical_bytes(derived) == raw,
        "batch_certificate: does not match raw evidence",
    )
    require(
        derived.get("schema") == "s39-qwen-batch-certificate-v1",
        "batch_certificate: schema",
    )
    require(
        derived.get("status") == "PROVISIONAL_BATCH",
        "batch_certificate: status",
    )
    return derived, sha256(raw)


def build(
    evidence_dir: Path,
    shard_manifest_path: Path,
    batch_evidence_dir: Path = DEFAULT_BATCH_EVIDENCE,
) -> dict[str, Any]:
    hashes, artifacts = load_artifacts(evidence_dir)
    shards = load_shards(shard_manifest_path)
    require(set(MODEL_SPECS) <= set(shards), "shard_manifest: missing model")
    batch_certificate, batch_digest = load_batch_certificate(
        batch_evidence_dir,
        evidence_dir / MODEL_SPECS["qwen3-14b-q4_k_m"]["cuda"],
        shard_manifest_path,
    )

    routes = {}
    for model_id, spec in MODEL_SPECS.items():
        model = shards[model_id]
        source = model.get("source")
        placements = model.get("placements")
        require(isinstance(source, dict), f"shard_manifest.{model_id}.source")
        require(isinstance(placements, dict), f"shard_manifest.{model_id}.placements")
        model_sha = require_string(source.get("sha256"), f"shard_manifest.{model_id}.sha256")
        require(SHA256_RE.fullmatch(model_sha) is not None, f"shard_manifest.{model_id}.sha256")
        require(
            is_int(source.get("block_count"))
            and source["block_count"] == spec["n_layer"],
            f"shard_manifest.{model_id}.layers",
        )

        head_end = placements.get("op15", {}).get("layer_end")
        tail_start = placements.get("op12", {}).get("layer_start")
        cut = 30
        require(
            is_int(head_end) and is_int(tail_start) and tail_start <= cut <= head_end,
            f"shard_manifest.{model_id}: cut is not resident",
        )

        head_raw = artifacts[spec["head"]]
        tail_raw = artifacts[spec["tail"]]
        cuda_raw = artifacts[spec["cuda"]]
        head_result = extract_record(head_raw, "HEADNET_RESULT ", f"{model_id}.result")
        head_cert = extract_record(head_raw, "PLACEMENTCERT ", f"{model_id}.head_cert")
        tail_cert = extract_record(tail_raw, "PLACEMENTCERT ", f"{model_id}.tail_cert")
        cuda_result = extract_record(cuda_raw, "ROUTEJSON ", f"{model_id}.cuda_result")

        phone_tokens = validate_result(head_result, f"{model_id}.result")
        cuda_tokens = validate_cuda(cuda_result, f"{model_id}.cuda_result")
        cuda_prompt_tokens = require_int(
            cuda_result.get("prompt_tokens"),
            f"{model_id}.cuda_result.prompt_tokens",
            1,
        )
        require(head_result["prompt_tokens"] == cuda_prompt_tokens, f"{model_id}: prompt count mismatch")
        require(
            head_result["output_tokens"] == cuda_result.get("generated_tokens"),
            f"{model_id}: output count mismatch",
        )

        problems = placement_problems(
            head_cert,
            role="phone_head",
            mode="headnet",
            layer_start=0,
            layer_end=cut,
            n_layer=spec["n_layer"],
        )
        problems += placement_problems(
            tail_cert,
            role="phone_tail",
            mode="tailnet",
            layer_start=cut,
            layer_end=spec["n_layer"],
            n_layer=spec["n_layer"],
        )
        problems = sorted(set(problems))

        if problems:
            status = "BLOCKED"
            reason = "placement gate failed: " + ",".join(problems)
        elif phone_tokens != cuda_tokens:
            status = "FAIL_CORRECTNESS"
            reason = "phone token IDs differ from the same-artifact CUDA control"
        elif model_id == "qwen3-14b-q4_k_m":
            require(
                batch_certificate["model_sha256"] == model_sha,
                f"{model_id}: batch model mismatch",
            )
            require(
                batch_certificate["cut_layer"] == cut,
                f"{model_id}: batch cut mismatch",
            )
            status = "PROVISIONAL_BATCH"
            reason = (
                "B1/B8/B32 single-prompt cohorts match CUDA with persistent "
                "workers; corpus, repeated-process, and measured-path gates remain"
            )
        else:
            status = "PROVISIONAL_B1"
            reason = (
                "one prompt and eight output tokens match CUDA; "
                "corpus and repeated-process gates remain"
            )

        evidence = {
            "cuda_control": hashes[spec["cuda"]],
            "op12_tail": hashes[spec["tail"]],
            "op15_head": hashes[spec["head"]],
        }
        if model_id == "qwen3-14b-q4_k_m":
            evidence["batch_wifi"] = batch_digest
        routes[model_id] = {
            "backend": "GPUOpenCL",
            "cut_layer": cut,
            "evidence": evidence,
            "model_sha256": model_sha,
            "reason": reason,
            "status": status,
        }
    return {"routes": routes, "schema_version": 1}


def atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise ReadinessError(f"cannot write {path}: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive S39 route readiness")
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument(
        "--batch-evidence",
        type=Path,
        default=DEFAULT_BATCH_EVIDENCE,
    )
    parser.add_argument("--shard-manifest", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = build(args.evidence, args.shard_manifest, args.batch_evidence)
        atomic_write(args.output, canonical_bytes(result))
    except ReadinessError as error:
        print(f"S39_READINESS_ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps({key: value["status"] for key, value in result["routes"].items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
