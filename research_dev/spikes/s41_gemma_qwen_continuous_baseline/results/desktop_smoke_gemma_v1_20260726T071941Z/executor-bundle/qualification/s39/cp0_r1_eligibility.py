#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = HERE / "CP0_R1_TWO_ROUTE_ELIGIBILITY_CONTRACT.json"
DEFAULT_CANDIDATE = HERE / "CP0_R1_CANDIDATE.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class EligibilityError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EligibilityError(message)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return (text + "\n").encode("ascii")
    except (TypeError, UnicodeEncodeError) as exc:
        raise EligibilityError("value is not canonical ASCII JSON") from exc


def load_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=strict_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EligibilityError(f"{path}: invalid JSON") from exc
    require(type(value) is dict, f"{path}: expected object")
    require(canonical_bytes(value) == raw, f"{path}: noncanonical JSON")
    return value, raw


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def is_int(value: Any) -> bool:
    return type(value) is int


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(is_int(value), f"{field}: expected integer")
    require(value >= minimum, f"{field}: expected >= {minimum}")
    return value


def string(value: Any, field: str) -> str:
    require(type(value) is str and bool(value), f"{field}: expected string")
    return value


def digest(value: Any, field: str) -> str:
    value = string(value, field)
    require(SHA256_RE.fullmatch(value) is not None, f"{field}: invalid SHA-256")
    return value


def exact_keys(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"{field}: expected object")
    actual = set(value)
    require(
        actual == keys,
        f"{field}: keys differ; missing={sorted(keys - actual)}, "
        f"unknown={sorted(actual - keys)}",
    )
    return value


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"{field}: expected {expected!r}, got {value!r}",
    )


def digest_list(value: Any, field: str) -> list[str]:
    require(type(value) is list and bool(value), f"{field}: expected nonempty list")
    result = [digest(item, f"{field}[{index}]") for index, item in enumerate(value)]
    require(len(set(result)) == len(result), f"{field}: duplicate digest")
    return result


def validate_contract(contract: dict[str, Any]) -> None:
    exact_keys(
        contract,
        {
            "candidate_search",
            "claim_boundary",
            "live_bridge",
            "pair_gates",
            "phone_memory",
            "reprepare",
            "route_gates",
            "schema",
            "scope",
            "serving_envelope",
            "status",
            "target_server",
            "task_quality",
        },
        "contract",
    )
    exact(contract["schema"], "s39-two-route-eligibility-contract-v1", "contract.schema")
    exact(
        contract["status"],
        "FROZEN_BEFORE_CP0_R1_ACQUISITION",
        "contract.status",
    )
    exact(
        contract["scope"],
        "CP0_R1_TWO_ROUTE_ELIGIBILITY_ONLY",
        "contract.scope",
    )

    search = exact_keys(
        contract["candidate_search"],
        {
            "cycle_authorized_only_after_two_route_pass",
            "maximum_new_candidates",
            "stop_after_first_candidate_failure",
        },
        "contract.candidate_search",
    )
    exact(search["maximum_new_candidates"], 1, "candidate_search.maximum")
    exact(search["stop_after_first_candidate_failure"], True, "candidate_search.stop")
    exact(
        search["cycle_authorized_only_after_two_route_pass"],
        True,
        "candidate_search.cycle",
    )

    target = exact_keys(
        contract["target_server"],
        {
            "device_name",
            "device_uuid",
            "memory_total_bytes",
            "minimum_free_vram_bytes",
        },
        "contract.target_server",
    )
    string(target["device_name"], "target_server.device_name")
    require(
        re.fullmatch(r"GPU-[0-9a-f-]{36}", string(target["device_uuid"], "target_server.uuid"))
        is not None,
        "target_server.device_uuid",
    )
    total = integer(target["memory_total_bytes"], "target_server.memory_total_bytes", 1)
    free = integer(
        target["minimum_free_vram_bytes"],
        "target_server.minimum_free_vram_bytes",
        1,
    )
    require(free < total, "target_server: headroom must be below capacity")

    envelope = exact_keys(
        contract["serving_envelope"],
        {
            "batch",
            "kv_type_k",
            "kv_type_v",
            "max_streams",
            "n_batch",
            "n_ctx_seq",
            "n_ubatch",
            "sampler",
        },
        "contract.serving_envelope",
    )
    for field in ("batch", "max_streams", "n_batch", "n_ubatch", "n_ctx_seq"):
        integer(envelope[field], f"serving_envelope.{field}", 1)
    exact(envelope["batch"], 8, "serving_envelope.batch")
    require(
        envelope["max_streams"] >= envelope["batch"],
        "serving_envelope.max_streams",
    )
    require(
        envelope["n_batch"] >= envelope["batch"]
        and envelope["n_ubatch"] >= envelope["batch"],
        "serving_envelope: insufficient batch capacity",
    )
    for field in ("kv_type_k", "kv_type_v", "sampler"):
        string(envelope[field], f"serving_envelope.{field}")

    phone_memory = exact_keys(
        contract["phone_memory"],
        {
            "maximum_process_swap_bytes",
            "maximum_system_swap_growth_bytes",
            "minimum_available_bytes_per_phone",
        },
        "contract.phone_memory",
    )
    integer(
        phone_memory["minimum_available_bytes_per_phone"],
        "phone_memory.minimum_available",
        1,
    )
    exact(phone_memory["maximum_process_swap_bytes"], 0, "phone_memory.process_swap")
    exact(
        phone_memory["maximum_system_swap_growth_bytes"],
        0,
        "phone_memory.system_swap_growth",
    )

    quality = exact_keys(
        contract["task_quality"],
        {
            "cross_backend_greedy_agreement",
            "cross_geometry_greedy_agreement",
            "items",
            "maximum_new_errors",
            "maximum_score_regression_items",
            "require_all_outputs_parseable",
        },
        "contract.task_quality",
    )
    integer(quality["items"], "task_quality.items", 1)
    integer(quality["maximum_new_errors"], "task_quality.maximum_new_errors")
    integer(
        quality["maximum_score_regression_items"],
        "task_quality.maximum_score_regression_items",
    )
    exact(quality["require_all_outputs_parseable"], True, "task_quality.parseable")
    exact(
        quality["cross_backend_greedy_agreement"],
        "DIAGNOSTIC_ONLY",
        "task_quality.cross_backend",
    )
    exact(
        quality["cross_geometry_greedy_agreement"],
        "DIAGNOSTIC_ONLY",
        "task_quality.cross_geometry",
    )

    bridge = exact_keys(
        contract["live_bridge"],
        {
            "minimum_requests_published_before_cuda_ready",
            "minimum_useful_phone_tokens",
        },
        "contract.live_bridge",
    )
    for field in bridge:
        integer(bridge[field], f"live_bridge.{field}", 1)

    reprepare = exact_keys(
        contract["reprepare"],
        {
            "maximum_elapsed_us",
            "maximum_network_weight_bytes",
            "maximum_usb_weight_bytes",
            "source",
        },
        "contract.reprepare",
    )
    integer(reprepare["maximum_elapsed_us"], "reprepare.maximum_elapsed_us", 1)
    exact(reprepare["maximum_network_weight_bytes"], 0, "reprepare.network_bytes")
    exact(reprepare["maximum_usb_weight_bytes"], 0, "reprepare.usb_bytes")
    exact(reprepare["source"], "PHONE_LOCAL_UFS_ONLY", "reprepare.source")

    exact(
        contract["route_gates"],
        [
            "ARTIFACT_AND_CONFIG_IDENTITY",
            "CUDA_B8_SERVE_WITH_HEADROOM",
            "FULL_TWO_PHONE_COVERAGE",
            "DIRECT_PHONE_ACTIVATION",
            "REALIZED_BACKEND_PLACEMENT",
            "POSITIVE_PHONE_MEMORY_HEADROOM",
            "ZERO_SWAP_GROWTH",
            "EXACT_HISTORY_POSITION_OWNERSHIP_CLEANUP",
            "INDEPENDENT_PATH_MATCHED_CUDA_ORACLE",
            "TASK_QUALITY_NONINFERIORITY",
            "USEFUL_PHONE_PUBLICATION_BEFORE_CUDA_READY",
        ],
        "contract.route_gates",
    )
    exact(
        contract["pair_gates"],
        [
            "MEASURED_CUDA_NON_CORESIDENCY",
            "LOCAL_UFS_REPREPARE_A_TO_B",
            "LOCAL_UFS_REPREPARE_B_TO_A",
        ],
        "contract.pair_gates",
    )

    claim = exact_keys(
        contract["claim_boundary"],
        {"pass_authorizes", "pass_does_not_authorize", "pass_status"},
        "contract.claim_boundary",
    )
    exact(claim["pass_status"], "TWO_ROUTE_ELIGIBILITY_PASS", "claim.pass_status")
    exact(
        claim["pass_authorizes"],
        "ONE_REDUCED_A_TO_B_TO_A_CYCLE",
        "claim.pass_authorizes",
    )
    exact(
        claim["pass_does_not_authorize"],
        [
            "TRACE_REPLAY",
            "CONTROLLER_INTEGRATION",
            "ENERGY_ACQUISITION",
            "ARCHITECTURE_DIVERSITY_CLAIM",
        ],
        "claim.pass_does_not_authorize",
    )


def validate_artifact(value: Any, field: str, require_origin: bool) -> dict[str, Any]:
    keys = {"bytes", "file_name", "sha256"}
    if require_origin:
        keys |= {"repository", "revision", "upstream_model"}
    artifact = exact_keys(value, keys, field)
    integer(artifact["bytes"], f"{field}.bytes", 1)
    string(artifact["file_name"], f"{field}.file_name")
    digest(artifact["sha256"], f"{field}.sha256")
    if require_origin:
        string(artifact["repository"], f"{field}.repository")
        require(
            re.fullmatch(r"[0-9a-f]{40}", string(artifact["revision"], f"{field}.revision"))
            is not None,
            f"{field}.revision",
        )
        string(artifact["upstream_model"], f"{field}.upstream_model")
    return artifact


def validate_route_binding(
    value: Any,
    field: str,
    n_layer: int,
) -> dict[str, Any] | None:
    if value is None:
        return None
    binding = exact_keys(
        value,
        {
            "backend",
            "executed_cut_layer",
            "op12_shard_sha256",
            "op12_stored_layers",
            "op15_shard_sha256",
            "op15_stored_layers",
        },
        field,
    )
    exact(binding["backend"], "GPUOpenCL", f"{field}.backend")
    cut = integer(binding["executed_cut_layer"], f"{field}.cut", 1)
    require(cut < n_layer, f"{field}.cut")
    for phone in ("op15", "op12"):
        layers = binding[f"{phone}_stored_layers"]
        require(
            type(layers) is list
            and len(layers) == 2
            and all(is_int(item) for item in layers)
            and 0 <= layers[0] < layers[1] <= n_layer,
            f"{field}.{phone}_stored_layers",
        )
        digest(binding[f"{phone}_shard_sha256"], f"{field}.{phone}_shard_sha256")
    require(
        binding["op15_stored_layers"][0] == 0
        and binding["op15_stored_layers"][1] >= cut,
        f"{field}: OP15 does not cover cut",
    )
    require(
        binding["op12_stored_layers"][0] <= cut
        and binding["op12_stored_layers"][1] == n_layer,
        f"{field}: OP12 does not cover cut",
    )
    return binding


def validate_candidate(
    candidate: dict[str, Any],
    contract_raw: bytes,
) -> list[dict[str, Any]]:
    exact_keys(
        candidate,
        {
            "candidate_attempt",
            "candidate_attempt_limit",
            "contract_sha256",
            "historical_routes",
            "models",
            "schema",
            "status",
            "task_suite",
        },
        "candidate",
    )
    exact(candidate["schema"], "s39-cp0-r1-candidate-v1", "candidate.schema")
    require(
        type(candidate["status"]) is str
        and candidate["status"]
        in {
            "CANDIDATE_SELECTED_ARTIFACT_NOT_ACQUIRED",
            "PAIR_FROZEN_BEFORE_PAID_ACQUISITION",
        },
        "candidate.status",
    )
    exact(candidate["contract_sha256"], sha256(contract_raw), "candidate.contract_sha256")
    exact(candidate["candidate_attempt"], 1, "candidate.attempt")
    exact(candidate["candidate_attempt_limit"], 1, "candidate.attempt_limit")

    models = candidate["models"]
    require(type(models) is list and len(models) == 2, "candidate.models")
    result = []
    for index, model in enumerate(models):
        field = f"candidate.models[{index}]"
        exact_keys(
            model,
            {
                "architecture",
                "artifact",
                "model_id",
                "n_layer",
                "quantization",
                "readiness_at_freeze",
                "route_binding",
                "selection",
                "slot",
            },
            field,
        )
        slot = string(model["slot"], f"{field}.slot")
        exact(slot, "AB"[index], f"{field}.slot")
        string(model["selection"], f"{field}.selection")
        string(model["model_id"], f"{field}.model_id")
        exact(model["architecture"], "qwen3", f"{field}.architecture")
        string(model["quantization"], f"{field}.quantization")
        n_layer = integer(model["n_layer"], f"{field}.n_layer", 1)
        validate_artifact(model["artifact"], f"{field}.artifact", slot == "B")
        binding = validate_route_binding(
            model["route_binding"],
            f"{field}.route_binding",
            n_layer,
        )
        if slot == "A":
            require(binding is not None, f"{field}: incumbent route binding missing")
            exact(
                model["readiness_at_freeze"],
                "PROVISIONAL_BATCH",
                f"{field}.readiness",
            )
        else:
            if candidate["status"] == "CANDIDATE_SELECTED_ARTIFACT_NOT_ACQUIRED":
                exact(binding, None, f"{field}.route_binding")
                exact(
                    model["readiness_at_freeze"],
                    "NOT_ACQUIRED",
                    f"{field}.readiness",
                )
            else:
                require(binding is not None, f"{field}: candidate route binding missing")
                exact(
                    model["readiness_at_freeze"],
                    "ROUTE_BINDING_FROZEN_EVIDENCE_PENDING",
                    f"{field}.readiness",
                )
        result.append(model)
    require(
        models[0]["artifact"]["sha256"] != models[1]["artifact"]["sha256"],
        "candidate.models: duplicate artifact",
    )

    suite = exact_keys(
        candidate["task_suite"],
        {
            "answer_mapping",
            "answer_parser",
            "chat_template",
            "dataset",
            "few_shot_examples",
            "items",
            "maximum_output_tokens",
            "prompt_format",
            "revision",
            "row_order",
            "selection",
            "split",
            "subject_order",
            "text_normalization",
        },
        "candidate.task_suite",
    )
    for field in (
        "answer_mapping",
        "answer_parser",
        "chat_template",
        "dataset",
        "prompt_format",
        "row_order",
        "selection",
        "split",
        "subject_order",
        "text_normalization",
    ):
        string(suite[field], f"candidate.task_suite.{field}")
    require(
        re.fullmatch(r"[0-9a-f]{40}", string(suite["revision"], "task_suite.revision"))
        is not None,
        "candidate.task_suite.revision",
    )
    exact(suite["items"], 64, "candidate.task_suite.items")
    integer(suite["maximum_output_tokens"], "task_suite.maximum_output_tokens", 1)
    exact(suite["few_shot_examples"], 0, "task_suite.few_shot_examples")

    historical = exact_keys(
        candidate["historical_routes"],
        {
            "gemma-4-12b-it-q4_0",
            "qwen2.5-14b-q4_0",
            "qwen2.5-14b-q8_0",
        },
        "candidate.historical_routes",
    )
    require(
        all(type(value) is str and value.endswith("_UNCHANGED") for value in historical.values()),
        "candidate.historical_routes",
    )
    return result


def validate_config(value: Any, contract: dict[str, Any], field: str) -> None:
    expected = contract["serving_envelope"]
    config = exact_keys(value, set(expected), field)
    for key, expected_value in expected.items():
        exact(config[key], expected_value, f"{field}.{key}")


def validate_cuda(
    value: Any,
    model: dict[str, Any],
    contract: dict[str, Any],
    field: str,
) -> dict[str, int]:
    record = exact_keys(
        value,
        {
            "artifacts",
            "batch",
            "completed_requests",
            "config",
            "device_name",
            "device_uuid",
            "free_vram_bytes",
            "host_swap_used_after_bytes",
            "host_swap_used_before_bytes",
            "kv_buffer_bytes",
            "model_buffer_bytes",
            "model_sha256",
            "peak_used_vram_bytes",
            "placement_compute_nodes",
            "placement_status",
            "state_count_after",
        },
        field,
    )
    digest_list(record["artifacts"], f"{field}.artifacts")
    exact(record["model_sha256"], model["artifact"]["sha256"], f"{field}.model_sha256")
    target = contract["target_server"]
    exact(record["device_name"], target["device_name"], f"{field}.device_name")
    exact(record["device_uuid"], target["device_uuid"], f"{field}.device_uuid")
    exact(record["batch"], contract["serving_envelope"]["batch"], f"{field}.batch")
    require(
        integer(record["completed_requests"], f"{field}.completed_requests", 1)
        == record["batch"],
        f"{field}.completed_requests",
    )
    validate_config(record["config"], contract, f"{field}.config")
    exact(record["placement_status"], "SCHEDULED_PLACEMENT_OK", f"{field}.placement")
    integer(record["placement_compute_nodes"], f"{field}.compute_nodes", 1)
    exact(record["state_count_after"], 0, f"{field}.state_count_after")
    for key in (
        "free_vram_bytes",
        "host_swap_used_after_bytes",
        "host_swap_used_before_bytes",
        "kv_buffer_bytes",
        "model_buffer_bytes",
        "peak_used_vram_bytes",
    ):
        integer(record[key], f"{field}.{key}", 0)
    require(record["model_buffer_bytes"] > 0, f"{field}.model_buffer_bytes")
    require(record["kv_buffer_bytes"] > 0, f"{field}.kv_buffer_bytes")
    require(
        record["free_vram_bytes"] >= target["minimum_free_vram_bytes"],
        f"{field}: insufficient GPU headroom",
    )
    require(
        record["peak_used_vram_bytes"] + target["minimum_free_vram_bytes"]
        <= target["memory_total_bytes"],
        f"{field}: individual model does not fit with headroom",
    )
    require(
        record["host_swap_used_after_bytes"]
        == record["host_swap_used_before_bytes"],
        f"{field}: host swap grew",
    )
    return {
        "kv_buffer_bytes": record["kv_buffer_bytes"],
        "model_buffer_bytes": record["model_buffer_bytes"],
    }


def validate_phone_memory(
    value: Any,
    contract: dict[str, Any],
    field: str,
) -> None:
    record = exact_keys(
        value,
        {
            "available_bytes",
            "process_swap_bytes",
            "system_swap_used_after_bytes",
            "system_swap_used_before_bytes",
        },
        field,
    )
    for key in record:
        integer(record[key], f"{field}.{key}")
    gates = contract["phone_memory"]
    require(
        record["available_bytes"] >= gates["minimum_available_bytes_per_phone"],
        f"{field}: insufficient memory headroom",
    )
    require(
        record["process_swap_bytes"] <= gates["maximum_process_swap_bytes"],
        f"{field}: process swap",
    )
    require(
        record["system_swap_used_after_bytes"]
        - record["system_swap_used_before_bytes"]
        <= gates["maximum_system_swap_growth_bytes"],
        f"{field}: system swap grew",
    )


def validate_phone_placement(value: Any, field: str) -> None:
    record = exact_keys(
        value,
        {"backend", "compute_nodes", "cpu_ops", "status"},
        field,
    )
    exact(record["backend"], "GPUOpenCL", f"{field}.backend")
    exact(record["status"], "SCHEDULED_PLACEMENT_OK", f"{field}.status")
    integer(record["compute_nodes"], f"{field}.compute_nodes", 1)
    require(
        type(record["cpu_ops"]) is list
        and len(set(record["cpu_ops"])) == len(record["cpu_ops"])
        and all(type(op) is str for op in record["cpu_ops"])
        and set(record["cpu_ops"]) <= {"GET_ROWS"},
        f"{field}.cpu_ops",
    )


def validate_mechanics(value: Any, field: str) -> None:
    record = exact_keys(
        value,
        {
            "cross_backend_greedy_matches",
            "cross_backend_greedy_total",
            "cross_geometry_exact",
            "duplicate_tokens",
            "expected_history_sha256",
            "expected_positions_sha256",
            "missing_tokens",
            "observed_history_sha256",
            "observed_positions_sha256",
            "oracle_expected_sha256",
            "oracle_backend",
            "oracle_call_shapes_sha256",
            "oracle_observed_sha256",
            "oracle_program_sha256",
            "ownership_transition_count",
            "route_program_sha256",
            "route_call_shapes_sha256",
            "stale_tokens",
            "terminal_state_counts",
        },
        field,
    )
    for key in (
        "expected_history_sha256",
        "expected_positions_sha256",
        "observed_history_sha256",
        "observed_positions_sha256",
        "oracle_expected_sha256",
        "oracle_call_shapes_sha256",
        "oracle_observed_sha256",
        "oracle_program_sha256",
        "route_call_shapes_sha256",
        "route_program_sha256",
    ):
        digest(record[key], f"{field}.{key}")
    require(
        record["expected_history_sha256"] == record["observed_history_sha256"],
        f"{field}: history mismatch",
    )
    require(
        record["expected_positions_sha256"] == record["observed_positions_sha256"],
        f"{field}: position mismatch",
    )
    require(
        record["oracle_expected_sha256"] == record["oracle_observed_sha256"],
        f"{field}: path-matched CUDA oracle mismatch",
    )
    exact(record["oracle_backend"], "CUDA0", f"{field}.oracle_backend")
    require(
        record["route_call_shapes_sha256"] == record["oracle_call_shapes_sha256"],
        f"{field}: CUDA oracle is not path-matched",
    )
    require(
        record["oracle_program_sha256"] != record["route_program_sha256"],
        f"{field}: oracle is not independent",
    )
    exact(record["ownership_transition_count"], 1, f"{field}.ownership_transition_count")
    for key in ("duplicate_tokens", "missing_tokens", "stale_tokens"):
        exact(record[key], 0, f"{field}.{key}")
    states = exact_keys(
        record["terminal_state_counts"],
        {"cuda", "op12", "op15"},
        f"{field}.terminal_state_counts",
    )
    for device, count in states.items():
        exact(count, 0, f"{field}.terminal_state_counts.{device}")
    total = integer(
        record["cross_backend_greedy_total"],
        f"{field}.cross_backend_greedy_total",
        1,
    )
    matches = integer(
        record["cross_backend_greedy_matches"],
        f"{field}.cross_backend_greedy_matches",
    )
    require(matches <= total, f"{field}: invalid diagnostic count")
    require(type(record["cross_geometry_exact"]) is bool, f"{field}.cross_geometry_exact")


def validate_quality(
    value: Any,
    contract: dict[str, Any],
    candidate: dict[str, Any],
    field: str,
) -> None:
    record = exact_keys(
        value,
        {
            "artifacts",
            "cross_backend_greedy_matches",
            "cross_backend_greedy_total",
            "cuda_correct",
            "cuda_parsed",
            "dataset",
            "dataset_revision",
            "phone_correct",
            "phone_new_errors",
            "phone_parsed",
            "phone_recovered_errors",
            "total",
        },
        field,
    )
    digest_list(record["artifacts"], f"{field}.artifacts")
    suite = candidate["task_suite"]
    exact(record["dataset"], suite["dataset"], f"{field}.dataset")
    exact(record["dataset_revision"], suite["revision"], f"{field}.dataset_revision")
    gates = contract["task_quality"]
    exact(record["total"], gates["items"], f"{field}.total")
    for key in (
        "cross_backend_greedy_matches",
        "cross_backend_greedy_total",
        "cuda_correct",
        "cuda_parsed",
        "phone_correct",
        "phone_new_errors",
        "phone_parsed",
        "phone_recovered_errors",
    ):
        integer(record[key], f"{field}.{key}")
        require(record[key] <= record["total"], f"{field}.{key}: exceeds total")
    require(
        record["cuda_parsed"] == record["total"]
        and record["phone_parsed"] == record["total"],
        f"{field}: unparseable task output",
    )
    require(
        record["phone_new_errors"] <= gates["maximum_new_errors"],
        f"{field}: too many new task errors",
    )
    require(
        record["phone_correct"] + gates["maximum_score_regression_items"]
        >= record["cuda_correct"],
        f"{field}: task score regression",
    )
    require(
        record["phone_correct"]
        == record["cuda_correct"]
        - record["phone_new_errors"]
        + record["phone_recovered_errors"],
        f"{field}: paired task counts are inconsistent",
    )
    require(
        record["cross_backend_greedy_total"] > 0
        and record["cross_backend_greedy_matches"]
        <= record["cross_backend_greedy_total"],
        f"{field}: invalid greedy diagnostic",
    )


def validate_bridge(
    value: Any,
    contract: dict[str, Any],
    field: str,
) -> None:
    record = exact_keys(
        value,
        {
            "artifacts",
            "cuda_ready_ns",
            "live_phone_publication_ns",
            "requests_published_before_cuda_ready",
            "useful_phone_tokens",
        },
        field,
    )
    digest_list(record["artifacts"], f"{field}.artifacts")
    for key in (
        "cuda_ready_ns",
        "live_phone_publication_ns",
        "requests_published_before_cuda_ready",
        "useful_phone_tokens",
    ):
        integer(record[key], f"{field}.{key}", 1)
    require(
        record["live_phone_publication_ns"] < record["cuda_ready_ns"],
        f"{field}: phone publication is not before CUDA readiness",
    )
    gates = contract["live_bridge"]
    require(
        record["useful_phone_tokens"] >= gates["minimum_useful_phone_tokens"],
        f"{field}: insufficient useful phone tokens",
    )
    require(
        record["requests_published_before_cuda_ready"]
        >= gates["minimum_requests_published_before_cuda_ready"],
        f"{field}: insufficient requests published before CUDA readiness",
    )


def validate_phone_route(
    value: Any,
    model: dict[str, Any],
    contract: dict[str, Any],
    candidate: dict[str, Any],
    field: str,
) -> None:
    record = exact_keys(
        value,
        {
            "activation",
            "artifacts",
            "batch",
            "coverage",
            "mechanics",
            "memory",
            "model_sha256",
            "placement",
            "quality",
        },
        field,
    )
    digest_list(record["artifacts"], f"{field}.artifacts")
    exact(record["model_sha256"], model["artifact"]["sha256"], f"{field}.model_sha256")
    exact(record["batch"], contract["serving_envelope"]["batch"], f"{field}.batch")

    binding = model["route_binding"]
    require(type(binding) is dict, f"{field}: route binding is not frozen")
    coverage = exact_keys(
        record["coverage"],
        {"cut_layer", "n_layer", "op12_layers", "op15_layers"},
        f"{field}.coverage",
    )
    exact(coverage["n_layer"], model["n_layer"], f"{field}.coverage.n_layer")
    exact(
        coverage["cut_layer"],
        binding["executed_cut_layer"],
        f"{field}.coverage.cut_layer",
    )
    exact(coverage["op15_layers"], [0, coverage["cut_layer"]], f"{field}.coverage.op15")
    exact(
        coverage["op12_layers"],
        [coverage["cut_layer"], model["n_layer"]],
        f"{field}.coverage.op12",
    )

    activation = exact_keys(
        record["activation"],
        {"direct_payload_bytes", "host_payload_bytes", "path"},
        f"{field}.activation",
    )
    exact(activation["path"], "OP15_TO_OP12_WIFI_TCP", f"{field}.activation.path")
    integer(activation["direct_payload_bytes"], f"{field}.activation.direct", 1)
    exact(activation["host_payload_bytes"], 0, f"{field}.activation.host")

    placement = exact_keys(
        record["placement"],
        {"op12", "op15"},
        f"{field}.placement",
    )
    memory = exact_keys(record["memory"], {"op12", "op15"}, f"{field}.memory")
    for phone in ("op15", "op12"):
        validate_phone_placement(placement[phone], f"{field}.placement.{phone}")
        validate_phone_memory(memory[phone], contract, f"{field}.memory.{phone}")

    validate_mechanics(record["mechanics"], f"{field}.mechanics")
    validate_quality(record["quality"], contract, candidate, f"{field}.quality")


def validate_pair_capacity(
    value: Any,
    models: list[dict[str, Any]],
    cuda_allocations: dict[str, dict[str, int]],
    contract: dict[str, Any],
) -> None:
    field = "evidence.pair_capacity"
    record = exact_keys(
        value,
        {
            "artifacts",
            "idle_used_vram_bytes",
            "lower_bound_used_vram_bytes",
            "models",
            "required_headroom_bytes",
            "target_device_uuid",
            "total_vram_bytes",
        },
        field,
    )
    digest_list(record["artifacts"], f"{field}.artifacts")
    target = contract["target_server"]
    exact(record["target_device_uuid"], target["device_uuid"], f"{field}.device_uuid")
    exact(record["total_vram_bytes"], target["memory_total_bytes"], f"{field}.total")
    exact(
        record["required_headroom_bytes"],
        target["minimum_free_vram_bytes"],
        f"{field}.headroom",
    )
    idle = integer(record["idle_used_vram_bytes"], f"{field}.idle")
    allocations = exact_keys(
        record["models"],
        {model["model_id"] for model in models},
        f"{field}.models",
    )
    computed = idle + record["required_headroom_bytes"]
    for model in models:
        model_id = model["model_id"]
        item = exact_keys(
            allocations[model_id],
            {"kv_buffer_bytes", "model_buffer_bytes"},
            f"{field}.models.{model_id}",
        )
        for key in ("kv_buffer_bytes", "model_buffer_bytes"):
            exact(
                item[key],
                cuda_allocations[model_id][key],
                f"{field}.models.{model_id}.{key}",
            )
            computed += item[key]
    exact(record["lower_bound_used_vram_bytes"], computed, f"{field}.lower_bound")
    require(
        computed > target["memory_total_bytes"],
        f"{field}: models can coexist with required headroom",
    )


def validate_reprepare(
    value: Any,
    models: list[dict[str, Any]],
    contract: dict[str, Any],
) -> None:
    field = "evidence.reprepare"
    require(type(value) is list and len(value) == 2, f"{field}: expected two directions")
    expected = [
        (models[0]["model_id"], models[1]["model_id"]),
        (models[1]["model_id"], models[0]["model_id"]),
    ]
    for index, (record, direction) in enumerate(zip(value, expected)):
        item_field = f"{field}[{index}]"
        record = exact_keys(
            record,
            {
                "artifacts",
                "ended_ns",
                "from_model_id",
                "local_ufs_bytes_read",
                "network_weight_bytes",
                "op12_shard_sha256",
                "op15_shard_sha256",
                "ready_generation_after",
                "ready_generation_before",
                "ready_model_sha256",
                "released_state_count",
                "source",
                "started_ns",
                "to_model_id",
                "usb_weight_bytes",
            },
            item_field,
        )
        digest_list(record["artifacts"], f"{item_field}.artifacts")
        exact(record["from_model_id"], direction[0], f"{item_field}.from")
        exact(record["to_model_id"], direction[1], f"{item_field}.to")
        to_model = models[1] if index == 0 else models[0]
        exact(
            record["ready_model_sha256"],
            to_model["artifact"]["sha256"],
            f"{item_field}.ready_model_sha256",
        )
        binding = to_model["route_binding"]
        require(type(binding) is dict, f"{item_field}: target route binding missing")
        exact(
            record["op15_shard_sha256"],
            binding["op15_shard_sha256"],
            f"{item_field}.op15_shard_sha256",
        )
        exact(
            record["op12_shard_sha256"],
            binding["op12_shard_sha256"],
            f"{item_field}.op12_shard_sha256",
        )
        exact(record["source"], "PHONE_LOCAL_UFS_ONLY", f"{item_field}.source")
        exact(record["released_state_count"], 0, f"{item_field}.released_state_count")
        started = integer(record["started_ns"], f"{item_field}.started_ns", 1)
        ended = integer(record["ended_ns"], f"{item_field}.ended_ns", 1)
        require(ended > started, f"{item_field}: invalid time interval")
        require(
            (ended - started + 999) // 1000 <= contract["reprepare"]["maximum_elapsed_us"],
            f"{item_field}: dwell bound exceeded",
        )
        integer(record["local_ufs_bytes_read"], f"{item_field}.local_ufs_bytes_read", 1)
        exact(record["usb_weight_bytes"], 0, f"{item_field}.usb_weight_bytes")
        exact(record["network_weight_bytes"], 0, f"{item_field}.network_weight_bytes")
        before = integer(
            record["ready_generation_before"],
            f"{item_field}.ready_generation_before",
        )
        after = integer(
            record["ready_generation_after"],
            f"{item_field}.ready_generation_after",
            1,
        )
        require(after > before, f"{item_field}: readiness generation did not advance")


def evaluate(
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate: dict[str, Any],
    candidate_raw: bytes,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    validate_contract(contract)
    models = validate_candidate(candidate, contract_raw)
    require(
        all(type(model["route_binding"]) is dict for model in models),
        "candidate route binding is incomplete",
    )
    exact_keys(
        evidence,
        {
            "candidate_attempts",
            "candidate_sha256",
            "contract_sha256",
            "models",
            "pair_capacity",
            "reported_status",
            "reprepare",
            "schema",
        },
        "evidence",
    )
    exact(evidence["schema"], "s39-two-route-eligibility-evidence-v1", "evidence.schema")
    exact(evidence["contract_sha256"], sha256(contract_raw), "evidence.contract_sha256")
    exact(evidence["candidate_sha256"], sha256(candidate_raw), "evidence.candidate_sha256")
    exact(evidence["candidate_attempts"], 1, "evidence.candidate_attempts")

    model_records = exact_keys(
        evidence["models"],
        {model["model_id"] for model in models},
        "evidence.models",
    )
    cuda_allocations = {}
    for model in models:
        model_id = model["model_id"]
        field = f"evidence.models.{model_id}"
        record = exact_keys(
            model_records[model_id],
            {"bridge", "cuda_b8", "phone_route"},
            field,
        )
        cuda_allocations[model_id] = validate_cuda(
            record["cuda_b8"],
            model,
            contract,
            f"{field}.cuda_b8",
        )
        validate_phone_route(
            record["phone_route"],
            model,
            contract,
            candidate,
            f"{field}.phone_route",
        )
        validate_bridge(record["bridge"], contract, f"{field}.bridge")

    validate_pair_capacity(
        evidence["pair_capacity"],
        models,
        cuda_allocations,
        contract,
    )
    validate_reprepare(evidence["reprepare"], models, contract)
    expected_status = contract["claim_boundary"]["pass_status"]
    exact(evidence["reported_status"], expected_status, "evidence.reported_status")
    return {
        "schema": "s39-two-route-eligibility-result-v1",
        "contract_sha256": sha256(contract_raw),
        "candidate_sha256": sha256(candidate_raw),
        "status": expected_status,
        "cycle_authorized": True,
        "trace_authorized": False,
        "energy_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the frozen S39 CP0-R1 eligibility contract"
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--evidence", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract, contract_raw = load_canonical(args.contract)
        candidate, candidate_raw = load_canonical(args.candidate)
        validate_contract(contract)
        validate_candidate(candidate, contract_raw)
        if args.evidence is None:
            result = {
                "schema": "s39-two-route-eligibility-contract-check-v1",
                "contract_sha256": sha256(contract_raw),
                "candidate_sha256": sha256(candidate_raw),
                "status": "CONTRACT_VALID_CANDIDATE_NOT_ACQUIRED",
            }
        else:
            evidence, _ = load_canonical(args.evidence)
            result = evaluate(
                contract,
                contract_raw,
                candidate,
                candidate_raw,
                evidence,
            )
        print(canonical_bytes(result).decode("ascii"), end="")
        return 0
    except (EligibilityError, OSError) as exc:
        print(f"CP0_R1_REFUSED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
