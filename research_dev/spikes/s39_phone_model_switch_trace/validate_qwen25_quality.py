#!/usr/bin/env python3
"""Independently validate the Qwen2.5 corpus-quality route evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, NamedTuple


MODEL_SHA256 = "924a4c39ef9fc6c139875ab6771c2e8172a3b40ffec5c720eca69ad7a0edfae7"
SOURCE_SHA256 = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"
OP15_SHARD_SHA256 = "4eb56f404eb8e4e64b63ba0870db3a41016486cc137ce7d65d51969902970a16"
OP12_SHARD_SHA256 = "f741cd302150adb0be9349d9ae257e77f4e7e18be8414e98797b32445ed42737"
REPORT_SCHEMA = "s39-qwen25-route-quality-v1"
CORPUS_MANIFEST_SCHEMA = "s33-wikitext-corpus-manifest-v1"
N_LAYER = 48
N_EMBD = 5120
BATCH = 32
PROMPT_TOKENS = 8
OUTPUT_TOKENS = 8
PREFILL_CHUNK = 2
THRESHOLDS = {
    "min_exact_sequence_agreement": 0.80,
    "min_first_token_agreement": 0.95,
    "min_token_decision_agreement": 0.95,
}
FILE_TYPE_LABELS = {
    2: "Q4_0",
    7: "Q8_0",
}


class RouteSpec(NamedTuple):
    model_sha256: str
    file_type: int
    cut_layer: int
    op15_shard_sha256: str
    op12_shard_sha256: str
    op15_adb_target: str
    op12_adb_target: str


DEFAULT_SPEC = RouteSpec(
    model_sha256=MODEL_SHA256,
    file_type=2,
    cut_layer=32,
    op15_shard_sha256=OP15_SHARD_SHA256,
    op12_shard_sha256=OP12_SHARD_SHA256,
    op15_adb_target="172.20.173.218:5555",
    op12_adb_target="5ae7a43d",
)


class EvidenceError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def load_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=strict_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{path}: invalid JSON") from exc
    require(type(value) is dict, f"{path}: expected object")
    require(canonical(value) == raw, f"{path}: noncanonical JSON")
    return value, raw


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def exact(value: Any, expected: Any, label: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"{label}: expected {expected!r}, got {value!r}",
    )


def validate_spec(spec: RouteSpec) -> None:
    require(
        type(spec.model_sha256) is str
        and re.fullmatch(r"[0-9a-f]{64}", spec.model_sha256) is not None,
        "route_spec.model_sha256",
    )
    require(
        type(spec.file_type) is int and spec.file_type in FILE_TYPE_LABELS,
        "route_spec.file_type",
    )
    require(
        type(spec.cut_layer) is int and 0 < spec.cut_layer < N_LAYER,
        "route_spec.cut_layer",
    )
    for label, digest in (
        ("op15_shard_sha256", spec.op15_shard_sha256),
        ("op12_shard_sha256", spec.op12_shard_sha256),
    ):
        require(
            type(digest) is str
            and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            f"route_spec.{label}",
        )
    require(
        type(spec.op15_adb_target) is str and bool(spec.op15_adb_target),
        "route_spec.op15_adb_target",
    )
    require(
        type(spec.op12_adb_target) is str and bool(spec.op12_adb_target),
        "route_spec.op12_adb_target",
    )


def parse_prefixed(path: Path, prefix: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.startswith(prefix):
            continue
        try:
            value = json.loads(
                line[len(prefix):],
                object_pairs_hook=strict_object,
            )
        except (json.JSONDecodeError, EvidenceError) as exc:
            raise EvidenceError(
                f"{path}:{line_number}: invalid {prefix.strip()} JSON"
            ) from exc
        require(
            type(value) is dict,
            f"{path}:{line_number}: expected {prefix.strip()} object",
        )
        records.append(value)
    return records


def extract_one(path: Path, prefix: str) -> dict[str, Any]:
    records = parse_prefixed(path, prefix)
    require(
        len(records) == 1,
        f"{path}: expected exactly one {prefix.strip()} record",
    )
    return records[0]


def validate_corpus(
    corpus_path: Path,
    manifest_path: Path,
    report: dict[str, Any],
    spec: RouteSpec,
) -> None:
    corpus_raw = corpus_path.read_bytes()
    manifest, manifest_raw = load_canonical(manifest_path)
    exact(
        manifest.get("schema"),
        CORPUS_MANIFEST_SCHEMA,
        "manifest.schema",
    )
    exact(manifest.get("source_sha256"), SOURCE_SHA256, "manifest.source_sha256")
    exact(
        manifest.get("model_sha256"),
        spec.model_sha256,
        "manifest.model_sha256",
    )
    exact(
        manifest.get("output_sha256"),
        sha256(corpus_raw),
        "manifest.output_sha256",
    )
    exact(report.get("corpus_sha256"), sha256(corpus_raw), "report.corpus_sha256")
    exact(
        report.get("corpus_manifest_sha256"),
        sha256(manifest_raw),
        "report.corpus_manifest_sha256",
    )


def validate_tokens(
    value: Any,
    prompts: int,
    label: str,
) -> list[list[int]]:
    require(type(value) is list and len(value) == prompts, f"{label}: row count")
    rows: list[list[int]] = []
    for index, row in enumerate(value):
        require(
            type(row) is list and len(row) == OUTPUT_TOKENS,
            f"{label}[{index}]: token count",
        )
        require(
            all(type(token) is int and token >= 0 for token in row),
            f"{label}[{index}]: invalid token",
        )
        rows.append(row)
    return rows


def derive_quality(
    physical: list[list[int]],
    reference: list[list[int]],
) -> dict[str, Any]:
    prompts = len(physical)
    require(prompts > 0 and len(reference) == prompts, "token row mismatch")
    first = sum(left[0] == right[0] for left, right in zip(physical, reference))
    exact_rows = sum(left == right for left, right in zip(physical, reference))
    decisions = sum(
        left == right
        for left_row, right_row in zip(physical, reference)
        for left, right in zip(left_row, right_row)
    )
    return {
        "exact_sequence_agreement": exact_rows / prompts,
        "exact_sequence_matches": exact_rows,
        "first_token_agreement": first / prompts,
        "first_token_matches": first,
        "token_decision_agreement": decisions / (prompts * OUTPUT_TOKENS),
        "token_decision_matches": decisions,
    }


def validate_hello(value: Any, label: str, spec: RouteSpec) -> None:
    require(type(value) is dict, f"{label}: expected object")
    exact(
        value.get("model_sha256"),
        spec.model_sha256,
        f"{label}.model_sha256",
    )
    exact(value.get("file_type"), spec.file_type, f"{label}.file_type")
    exact(value.get("layer_start"), 0, f"{label}.layer_start")
    exact(value.get("layer_end"), N_LAYER, f"{label}.layer_end")
    exact(value.get("n_layer"), N_LAYER, f"{label}.n_layer")
    exact(value.get("n_embd"), N_EMBD, f"{label}.n_embd")
    for key, minimum in (
        ("max_streams", BATCH),
        ("n_batch", BATCH * PREFILL_CHUNK),
        ("n_ubatch", BATCH * PREFILL_CHUNK),
        ("n_ctx_seq", PROMPT_TOKENS + OUTPUT_TOKENS - 1),
    ):
        field = value.get(key)
        require(
            type(field) is int and field >= minimum,
            f"{label}.{key}: insufficient capacity",
        )


def cpu_ops(placement: dict[str, Any]) -> set[str]:
    by_op = placement.get("compute_by_op_and_buffer")
    require(type(by_op) is dict, "placement.compute_by_op_and_buffer")
    result: set[str] = set()
    for op, buffers in by_op.items():
        require(type(op) is str and type(buffers) is dict, "placement op entry")
        for buffer, count in buffers.items():
            require(
                type(buffer) is str and type(count) is int and count >= 0,
                "placement buffer entry",
            )
            if count > 0 and buffer in {"CPU", "CPU_Mapped", "CUDA_Host"}:
                result.add(op)
    return result


def validate_stage_log(
    path: Path,
    expected_backend: str,
    buffer_backend: str,
    layer_start: int,
    layer_end: int,
    rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    session = extract_one(path, "SESSIONCERT ")
    placement = extract_one(path, "PLACEMENTCERT ")
    exact(session.get("schema"), "ls-stagenet-session-v2", f"{path}.session.schema")
    exact(session.get("session_end"), "STOP", f"{path}.session.session_end")
    exact(
        session.get("expected_backend"),
        expected_backend,
        f"{path}.session.backend",
    )
    exact(session.get("layer_start"), layer_start, f"{path}.session.layer_start")
    exact(session.get("layer_end"), layer_end, f"{path}.session.layer_end")
    exact(session.get("n_layer"), N_LAYER, f"{path}.session.n_layer")
    exact(session.get("steps_session"), rows, f"{path}.session.steps_session")
    exact(
        session.get("missing_buffer_compute_nodes"),
        0,
        f"{path}.session.missing_buffer_compute_nodes",
    )
    exact(
        session.get("placement_status"),
        "SCHEDULED_PLACEMENT_OK",
        f"{path}.session.placement_status",
    )

    exact(
        placement.get("schema"),
        "layersplit-scheduled-placement-v2",
        f"{path}.placement.schema",
    )
    exact(
        placement.get("status"),
        "SCHEDULED_PLACEMENT_OK",
        f"{path}.placement.status",
    )
    exact(placement.get("layer_start"), layer_start, f"{path}.placement.layer_start")
    exact(placement.get("layer_end"), layer_end, f"{path}.placement.layer_end")
    exact(placement.get("n_layer"), N_LAYER, f"{path}.placement.n_layer")
    exact(
        placement.get("missing_buffer_compute_nodes"),
        0,
        f"{path}.placement.missing_buffer_compute_nodes",
    )
    compute_nodes = placement.get("compute_nodes")
    require(
        type(compute_nodes) is int and compute_nodes > 0,
        f"{path}.placement.compute_nodes",
    )
    by_buffer = placement.get("compute_by_buffer_type")
    require(type(by_buffer) is dict, f"{path}.placement.compute_by_buffer_type")
    require(
        type(by_buffer.get(buffer_backend)) is int
        and by_buffer[buffer_backend] > 0,
        f"{path}: expected backend did no work",
    )
    require(cpu_ops(placement) <= {"GET_ROWS"}, f"{path}: undeclared host compute")
    return session, placement


def validate_post_run_context(
    path: Path,
    head_session: dict[str, Any],
    tail_session: dict[str, Any],
    spec: RouteSpec,
) -> None:
    context, _ = load_canonical(path)
    exact(
        context.get("schema"),
        "s39-qwen25-post-run-context-v1",
        "post_context.schema",
    )
    exact(
        context.get("scope"),
        "POST_RUN_SAME_BOOT_FILE_IDENTITY",
        "post_context.scope",
    )
    exact(
        context.get("model_sha256"),
        spec.model_sha256,
        "post_context.model_sha256",
    )
    acquired = context.get("acquisition_unix_s")
    require(
        type(acquired) is int and acquired > 0,
        "post_context.acquisition_unix_s",
    )
    route = context.get("route")
    exact(
        route,
        {
            "backend": "GPUOpenCL",
            "cut_layer": spec.cut_layer,
            "op12_layers": [spec.cut_layer, N_LAYER],
            "op15_layers": [0, spec.cut_layer],
        },
        "post_context.route",
    )
    op15 = context.get("op15")
    op12 = context.get("op12")
    require(type(op15) is dict and type(op12) is dict, "post_context.devices")
    exact(
        op15.get("adb_target"),
        spec.op15_adb_target,
        "post_context.op15.target",
    )
    exact(
        op12.get("adb_target"),
        spec.op12_adb_target,
        "post_context.op12.target",
    )
    exact(
        op15.get("boot_id"),
        head_session.get("device_boot_id"),
        "post_context.op15.boot_id",
    )
    exact(
        op12.get("boot_id"),
        tail_session.get("device_boot_id"),
        "post_context.op12.boot_id",
    )
    exact(
        op15.get("shard_sha256"),
        spec.op15_shard_sha256,
        "post_context.op15.shard_sha256",
    )
    exact(
        op12.get("shard_sha256"),
        spec.op12_shard_sha256,
        "post_context.op12.shard_sha256",
    )
    runtime_sha = op15.get("runtime_sha256")
    require(
        type(runtime_sha) is str
        and re.fullmatch(r"[0-9a-f]{64}", runtime_sha) is not None,
        "post_context.op15.runtime_sha256",
    )
    exact(
        op12.get("runtime_sha256"),
        runtime_sha,
        "post_context.op12.runtime_sha256",
    )
    require(
        type(op15.get("relay_sha256")) is str
        and re.fullmatch(r"[0-9a-f]{64}", op15["relay_sha256"]) is not None,
        "post_context.op15.relay_sha256",
    )


def validate_relay(
    path: Path,
    batches: int,
    rows: int,
    spec: RouteSpec,
) -> None:
    record = extract_one(path, "DIRECTCERT ")
    exact(record.get("schema"), "ls-stage-direct-relay-v1", f"{path}.schema")
    exact(record.get("status"), "DIRECT_RELAY_OK", f"{path}.status")
    exact(record.get("run_rc"), 0, f"{path}.run_rc")
    exact(record.get("layer_start"), 0, f"{path}.layer_start")
    exact(record.get("cut_layer"), spec.cut_layer, f"{path}.cut_layer")
    exact(record.get("layer_end"), N_LAYER, f"{path}.layer_end")
    exact(record.get("n_layer"), N_LAYER, f"{path}.n_layer")
    exact(record.get("n_embd"), N_EMBD, f"{path}.n_embd")
    exact(record.get("file_type"), spec.file_type, f"{path}.file_type")
    exact(
        record.get("model_sha256"),
        spec.model_sha256,
        f"{path}.model_sha256",
    )
    exact(record.get("batches"), batches, f"{path}.batches")
    exact(record.get("rows"), rows, f"{path}.rows")
    exact(
        record.get("activation_payload_bytes"),
        rows * N_EMBD * 4,
        f"{path}.activation_payload_bytes",
    )
    exact(
        record.get("host_activation_payload_bytes"),
        0,
        f"{path}.host_activation_payload_bytes",
    )


def validate_repeat(
    evidence_dir: Path,
    report: dict[str, Any],
    report_raw: bytes,
) -> dict[str, str]:
    repeat_path = evidence_dir / "quality_repeat_report.json"
    check_path = evidence_dir / "REPEAT_CHECK.json"
    if not repeat_path.exists() and not check_path.exists():
        return {}
    require(
        repeat_path.exists() and check_path.exists(),
        "repeat evidence is incomplete",
    )
    repeat, repeat_raw = load_canonical(repeat_path)
    for key in (
        "batch",
        "cohorts",
        "corpus_manifest_sha256",
        "corpus_sha256",
        "file_type",
        "model_sha256",
        "output_tokens",
        "phone_attention",
        "physical_tokens",
        "physical_tokens_sha256",
        "prefill_chunk",
        "prompt_tokens",
        "prompts",
        "quality",
        "quality_gate_pass",
        "reference_tokens",
        "reference_tokens_sha256",
        "scope",
        "status",
        "thresholds",
    ):
        exact(repeat.get(key), report.get(key), f"repeat.{key}")
    check, check_raw = load_canonical(check_path)
    expected = {
        "first_report_sha256": sha256(repeat_raw),
        "physical_tokens_sha256": report["physical_tokens_sha256"],
        "reference_tokens_sha256": report["reference_tokens_sha256"],
        "schema": "s39-qwen25-quality-repeat-v1",
        "second_report_sha256": sha256(report_raw),
        "status": "TOKEN_HASH_REPEAT_PASS",
    }
    exact(check, expected, "repeat_check")
    return {
        check_path.name: sha256(check_raw),
        repeat_path.name: sha256(repeat_raw),
    }


def validate_report(
    report_path: Path,
    corpus_path: Path,
    manifest_path: Path,
    evidence_dir: Path,
    spec: RouteSpec = DEFAULT_SPEC,
) -> dict[str, Any]:
    validate_spec(spec)
    report, report_raw = load_canonical(report_path)
    exact(report.get("schema"), REPORT_SCHEMA, "report.schema")
    exact(
        report.get("model_sha256"),
        spec.model_sha256,
        "report.model_sha256",
    )
    exact(report.get("file_type"), spec.file_type, "report.file_type")
    exact(report.get("batch"), BATCH, "report.batch")
    exact(report.get("prefill_chunk"), PREFILL_CHUNK, "report.prefill_chunk")
    exact(report.get("prompt_tokens"), PROMPT_TOKENS, "report.prompt_tokens")
    exact(report.get("output_tokens"), OUTPUT_TOKENS, "report.output_tokens")
    exact(report.get("thresholds"), THRESHOLDS, "report.thresholds")
    exact(report.get("phone_session_end"), "STOP", "report.phone_session_end")
    exact(report.get("phone_attention"), "fused", "report.phone_attention")

    cohorts = report.get("cohorts")
    prompts = report.get("prompts")
    require(type(cohorts) is int and cohorts > 0, "report.cohorts")
    exact(prompts, cohorts * BATCH, "report.prompts")
    exact(
        report.get("scope"),
        (
            f"WIKITEXT_{prompts}_PROMPTS_QWEN25_"
            f"{FILE_TYPE_LABELS[spec.file_type]}_GREEDY_AGREEMENT"
        ),
        "report.scope",
    )
    validate_corpus(corpus_path, manifest_path, report, spec)

    physical = validate_tokens(report.get("physical_tokens"), prompts, "physical")
    reference = validate_tokens(report.get("reference_tokens"), prompts, "reference")
    exact(
        report.get("physical_tokens_sha256"),
        sha256(canonical(physical)),
        "report.physical_tokens_sha256",
    )
    exact(
        report.get("reference_tokens_sha256"),
        sha256(canonical(reference)),
        "report.reference_tokens_sha256",
    )
    quality = derive_quality(physical, reference)
    exact(report.get("quality"), quality, "report.quality")
    gate_pass = (
        quality["first_token_agreement"]
        >= THRESHOLDS["min_first_token_agreement"]
        and quality["token_decision_agreement"]
        >= THRESHOLDS["min_token_decision_agreement"]
        and quality["exact_sequence_agreement"]
        >= THRESHOLDS["min_exact_sequence_agreement"]
    )
    exact(report.get("quality_gate_pass"), gate_pass, "report.quality_gate_pass")
    exact(
        report.get("status"),
        "QUALITY_PASS" if gate_pass else "QUALITY_FAIL",
        "report.status",
    )
    exact(report.get("scheduler_eligible"), False, "report.scheduler_eligible")

    hellos = report.get("hellos")
    require(type(hellos) is dict, "report.hellos")
    validate_hello(hellos.get("phone"), "report.hellos.phone", spec)
    validate_hello(hellos.get("cuda"), "report.hellos.cuda", spec)

    timings = report.get("timings")
    require(type(timings) is list and len(timings) == cohorts, "report.timings")
    for index, timing in enumerate(timings):
        require(type(timing) is dict, f"report.timings[{index}]")
        exact(timing.get("cohort"), index, f"report.timings[{index}].cohort")
        for key in ("cuda_us", "phone_us"):
            value = timing.get(key)
            require(
                type(value) is int and value > 0,
                f"report.timings[{index}].{key}",
            )

    batches = cohorts * (
        PROMPT_TOKENS // PREFILL_CHUNK + OUTPUT_TOKENS - 1
    )
    rows = cohorts * (
        BATCH * PROMPT_TOKENS
        + BATCH * (OUTPUT_TOKENS - 1)
    )
    phone_head_session, phone_head = validate_stage_log(
        evidence_dir / "op15_head.log",
        "GPUOpenCL",
        "OpenCL",
        0,
        spec.cut_layer,
        rows,
    )
    phone_tail_session, phone_tail = validate_stage_log(
        evidence_dir / "op12_tail.log",
        "GPUOpenCL",
        "OpenCL",
        spec.cut_layer,
        N_LAYER,
        rows,
    )
    _, cuda_head = validate_stage_log(
        evidence_dir / "cuda_head.log",
        "CUDA0",
        "CUDA0",
        0,
        spec.cut_layer,
        rows,
    )
    _, cuda_tail = validate_stage_log(
        evidence_dir / "cuda_tail.log",
        "CUDA0",
        "CUDA0",
        spec.cut_layer,
        N_LAYER,
        rows,
    )
    validate_relay(evidence_dir / "phone_relay.log", batches, rows, spec)
    validate_relay(evidence_dir / "cuda_relay.log", batches, rows, spec)
    validate_post_run_context(
        evidence_dir / "POST_RUN_CONTEXT.json",
        phone_head_session,
        phone_tail_session,
        spec,
    )
    repeat_artifacts = validate_repeat(evidence_dir, report, report_raw)

    artifact_names = (
        "POST_RUN_CONTEXT.json",
        "cuda_head.log",
        "cuda_relay.log",
        "cuda_tail.log",
        "op12_tail.log",
        "op15_head.log",
        "phone_relay.log",
    )
    return {
        "artifacts": {
            **{
                path.name: sha256_file(path)
                for path in (
                    report_path,
                    corpus_path,
                    manifest_path,
                    *(evidence_dir / name for name in artifact_names),
                )
            },
            **repeat_artifacts,
        },
        "known_limits": {
            "fresh_process_repeats": (
                "TOKEN_HASH_REPEAT_PASS_PLACEMENT_REPEAT_NOT_BOUND"
                if repeat_artifacts
                else "NOT_RUN"
            ),
            "op12_optional_kernel_compile_failure": (
                "sub_group_shuffle_xor"
                in (evidence_dir / "op12_tail.log").read_text(
                    encoding="utf-8"
                )
            ),
            "phone_energy": "NOT_MEASURED",
        },
        "placement": {
            "cuda_compute_nodes": (
                cuda_head["compute_nodes"] + cuda_tail["compute_nodes"]
            ),
            "op12_compute_nodes": phone_tail["compute_nodes"],
            "op15_compute_nodes": phone_head["compute_nodes"],
        },
        "quality": quality,
        "report_sha256": sha256(report_raw),
        "scheduler_eligible": False,
        "schema": "s39-qwen25-quality-certificate-v1",
        "status": "QUALITY_FAIL" if not gate_pass else "QUALITY_PASS_REPEATS_PENDING",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--model-sha256", default=DEFAULT_SPEC.model_sha256)
    parser.add_argument(
        "--file-type",
        type=int,
        choices=tuple(FILE_TYPE_LABELS),
        default=DEFAULT_SPEC.file_type,
    )
    parser.add_argument("--cut-layer", type=int, default=DEFAULT_SPEC.cut_layer)
    parser.add_argument(
        "--op15-shard-sha256",
        default=DEFAULT_SPEC.op15_shard_sha256,
    )
    parser.add_argument(
        "--op12-shard-sha256",
        default=DEFAULT_SPEC.op12_shard_sha256,
    )
    parser.add_argument(
        "--op15-adb-target",
        default=DEFAULT_SPEC.op15_adb_target,
    )
    parser.add_argument(
        "--op12-adb-target",
        default=DEFAULT_SPEC.op12_adb_target,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    digests = (
        args.model_sha256,
        args.op15_shard_sha256,
        args.op12_shard_sha256,
    )
    if (
        any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in digests)
        or not 0 < args.cut_layer < N_LAYER
        or not args.op15_adb_target
        or not args.op12_adb_target
    ):
        parser.error("invalid route specification")
    spec = RouteSpec(
        model_sha256=args.model_sha256,
        file_type=args.file_type,
        cut_layer=args.cut_layer,
        op15_shard_sha256=args.op15_shard_sha256,
        op12_shard_sha256=args.op12_shard_sha256,
        op15_adb_target=args.op15_adb_target,
        op12_adb_target=args.op12_adb_target,
    )
    try:
        certificate = validate_report(
            args.report,
            args.corpus,
            args.manifest,
            args.evidence_dir,
            spec,
        )
        raw = canonical(certificate)
        if args.output is not None:
            args.output.write_bytes(raw)
        print(raw.decode("ascii"), end="")
        return 0 if certificate["status"] != "QUALITY_FAIL" else 3
    except (EvidenceError, OSError, UnicodeError) as exc:
        print(
            canonical({"error": str(exc), "status": "EVIDENCE_FAIL"}).decode(
                "ascii"
            ),
            end="",
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
