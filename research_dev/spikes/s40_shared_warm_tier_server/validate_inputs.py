#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from evidence_common import (
    canonical_bytes,
    EvidenceError,
    digest_file,
    read_json,
    read_jsonl,
    require,
    require_int,
    require_string,
    validate_digest,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = HERE / "EXPERIMENT_CONTRACT.json"

REQUEST_KEYS = {
    "arrival_us",
    "event_id",
    "input_tokens",
    "model_id",
    "output_tokens",
    "prompt_tokens",
    "request_index",
    "schema",
    "slo_us",
    "source_input_tokens",
    "source_model",
    "source_output_tokens",
    "source_t_us",
}
SWITCH_KEYS = {
    "from_model_id",
    "intent_index",
    "kind",
    "schema",
    "source_event_id",
    "source_intent_id",
    "source_t_us",
    "t_us",
    "to_model_id",
}
CONTRACT_KEYS = {
    "baselines",
    "energy",
    "matrix",
    "policy",
    "run_manifest_requirements",
    "runtime_source",
    "schema",
    "schema_version",
    "workload",
}
WORKLOAD_KEYS = {
    "campaign_horizon_us",
    "drain_bound_us",
    "greedy_sampling",
    "input_manifest_path",
    "input_manifest_sha256",
    "models",
    "output_tokens_per_request",
    "request_count",
    "requests_path",
    "requests_sha256",
    "slo_us",
    "switch_count",
    "switches_path",
    "switches_role",
    "switches_sha256",
    "trace_start_lead_us",
}


def resolve(contract_path: Path, relative: str) -> Path:
    path = (contract_path.parent / relative).resolve()
    require(path.is_file(), f"missing bound file {path}")
    return path


def check_bound_file(
        contract_path: Path,
        record: dict[str, Any],
        path_key: str,
        digest_key: str) -> Path:
    path = resolve(contract_path, require_string(record[path_key], path_key))
    expected = validate_digest(record[digest_key], digest_key)
    require(digest_file(path) == expected, f"{path_key}: SHA-256 mismatch")
    return path


def validate_requests(
        rows: list[dict[str, Any]],
        workload: dict[str, Any]) -> dict[str, int]:
    expected_count = require_int(workload["request_count"], "request_count", 1)
    expected_output = require_int(
        workload["output_tokens_per_request"],
        "output_tokens_per_request",
        1,
    )
    expected_slo = require_int(workload["slo_us"], "slo_us", 1)
    models = workload["models"]
    require(
        isinstance(models, list) and len(models) == 2
        and all(isinstance(item, str) and item for item in models)
        and len(set(models)) == 2,
        "models: expected two unique strings",
    )
    require(len(rows) == expected_count, "requests: record count mismatch")

    counts = {model_id: 0 for model_id in models}
    event_ids: set[str] = set()
    previous_arrival = -1
    for index, row in enumerate(rows):
        require(set(row) == REQUEST_KEYS, f"request[{index}]: key set mismatch")
        require(
            row["schema"] == "s39-cp0d-desktop-request-v1",
            f"request[{index}]: schema mismatch",
        )
        require_int(row["request_index"], f"request[{index}].request_index")
        require(row["request_index"] == index, f"request[{index}]: index mismatch")
        arrival = require_int(row["arrival_us"], f"request[{index}].arrival_us")
        require(arrival >= previous_arrival, f"request[{index}]: arrival order")
        previous_arrival = arrival
        event_id = require_string(row["event_id"], f"request[{index}].event_id")
        require(event_id not in event_ids, f"request[{index}]: duplicate event")
        event_ids.add(event_id)
        model_id = require_string(row["model_id"], f"request[{index}].model_id")
        require(model_id in counts, f"request[{index}]: unknown model")
        counts[model_id] += 1
        prompt = row["prompt_tokens"]
        require(
            isinstance(prompt, list)
            and all(isinstance(token, int) and not isinstance(token, bool)
                    and token >= 0 for token in prompt),
            f"request[{index}]: invalid prompt tokens",
        )
        input_tokens = require_int(
            row["input_tokens"], f"request[{index}].input_tokens", 1)
        require(len(prompt) == input_tokens, f"request[{index}]: prompt length")
        require(
            require_int(
                row["output_tokens"],
                f"request[{index}].output_tokens",
                1,
            ) == expected_output,
            f"request[{index}]: output token budget mismatch",
        )
        require(
            require_int(row["slo_us"], f"request[{index}].slo_us", 1)
            == expected_slo,
            f"request[{index}]: SLO mismatch",
        )
        require_int(
            row["source_input_tokens"],
            f"request[{index}].source_input_tokens",
            1,
        )
        require_int(
            row["source_output_tokens"],
            f"request[{index}].source_output_tokens",
            1,
        )
        require_int(row["source_t_us"], f"request[{index}].source_t_us")
        require_string(row["source_model"], f"request[{index}].source_model")
    require(sorted(counts.values()) == [17, 57], "requests: model mix mismatch")
    return counts


def validate_switches(
        rows: list[dict[str, Any]],
        workload: dict[str, Any],
        requests: list[dict[str, Any]]) -> None:
    expected = require_int(workload["switch_count"], "switch_count", 1)
    models = set(workload["models"])
    request_by_event = {row["event_id"]: row for row in requests}
    require(len(rows) == expected, "switches: record count mismatch")
    previous_t = -1
    for index, row in enumerate(rows):
        require(set(row) == SWITCH_KEYS, f"switch[{index}]: key set mismatch")
        require(
            row["schema"] == "s39-cp0d-desktop-switch-v1"
            and row["kind"] == "TARGET_CHANGE",
            f"switch[{index}]: identity mismatch",
        )
        require_int(row["intent_index"], f"switch[{index}].intent_index")
        require(row["intent_index"] == index, f"switch[{index}]: index mismatch")
        t_us = require_int(row["t_us"], f"switch[{index}].t_us")
        require(t_us > previous_t, f"switch[{index}]: nonmonotonic time")
        previous_t = t_us
        source = require_string(
            row["from_model_id"], f"switch[{index}].from_model_id")
        target = require_string(
            row["to_model_id"], f"switch[{index}].to_model_id")
        require(
            {source, target} == models and source != target,
            f"switch[{index}]: model pair mismatch",
        )
        if index:
            require(
                source == rows[index - 1]["to_model_id"],
                f"switch[{index}]: transition chain mismatch",
            )
        source_event = require_string(
            row["source_event_id"], f"switch[{index}].source_event_id")
        require(source_event in request_by_event, f"switch[{index}]: source event")
        request = request_by_event[source_event]
        require(
            request["model_id"] == target and request["arrival_us"] == t_us,
            f"switch[{index}]: trigger request mismatch",
        )
        require_string(
            row["source_intent_id"], f"switch[{index}].source_intent_id")
        require_int(row["source_t_us"], f"switch[{index}].source_t_us")


def validate_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    raw = path.read_bytes()
    contract = read_json(path, "contract")
    require(canonical_bytes(contract) == raw, "contract: not canonical JSON")
    require(set(contract) == CONTRACT_KEYS, "contract: key set mismatch")
    require(
        contract.get("schema") == "s40-shared-warm-tier-experiment-v1"
        and contract.get("schema_version") == 1,
        "contract: unsupported identity",
    )
    workload = contract.get("workload")
    require(isinstance(workload, dict), "contract.workload: expected object")
    require(set(workload) == WORKLOAD_KEYS, "contract.workload: key set mismatch")
    require(workload["greedy_sampling"] is True,
            "contract.workload: greedy sampling required")
    require(
        require_int(
            workload["campaign_horizon_us"],
            "contract.workload.campaign_horizon_us",
            1,
        ) == 900_000_000,
        "contract.workload: campaign horizon mismatch",
    )
    require(
        require_int(
            workload["drain_bound_us"],
            "contract.workload.drain_bound_us",
            1,
        ) == 120_000_000,
        "contract.workload: drain bound mismatch",
    )
    require(
        require_int(
            workload["trace_start_lead_us"],
            "contract.workload.trace_start_lead_us",
            1,
        ) == 1_000_000,
        "contract.workload: trace start lead mismatch",
    )
    require(
        workload["campaign_horizon_us"] > workload["slo_us"],
        "contract.workload: SLO must not define the campaign horizon",
    )
    require(
        workload["models"] == [
            "qwen3-8b-q8_0",
            "qwen3-14b-q4_k_m",
        ],
        "contract.workload: exact model order mismatch",
    )

    policy = contract.get("policy")
    require(
        policy == {
            "coalesce_only_before_transition_start": True,
            "dispatch_when_compatible_ready_credit_exists": True,
            "gpu_admission_closes_at": "SWITCH_START",
            "historical_switch_rows_drive_new_modes": False,
            "id": "s40-oldest-demand-work-conserving-v1",
            "model_queue_order": "FIFO",
            "new_mode_switch_timing": "DEMAND_DERIVED",
            "next_gpu_target":
                "MODEL_OF_OLDEST_NONTERMINAL_NON_GPU_DEMAND",
            "proposal_is_advisory": True,
            "run_end_terminal_states": ["COMPLETED", "STRANDED"],
        },
        "contract.policy: mismatch",
    )
    require(
        workload["switches_role"] == "HISTORICAL_C0_PROVENANCE_ONLY",
        "contract.workload: switch role mismatch",
    )
    require(
        contract.get("energy") == {
            "allowed_scope": "SELECTED_GPU_BOARD_DEVELOPMENT_ONLY",
            "claim_status": "BLOCKED_UNTIL_E2_INSTRUMENT_QUALITY_GATE",
            "phone_energy": "UNKNOWN",
            "server_wall_energy": "UNKNOWN",
            "total_system_energy": "UNKNOWN",
        },
        "contract.energy: boundary mismatch",
    )
    require(
        contract.get("run_manifest_requirements") == {
            "exact_command_array_required": True,
            "experiment_contract_sha256_required": True,
            "raw_event_ledger_required": True,
            "raw_executor_evidence_required": True,
            "raw_resource_samples_required": True,
            "required_http_threads": 128,
        },
        "contract.run_manifest_requirements: mismatch",
    )
    runtime_source = contract.get("runtime_source")
    require(isinstance(runtime_source, dict),
            "contract.runtime_source: expected object")
    runtime_contract_value = dict(runtime_source)
    runtime_contract_value["desktop_contract_path"] = (
        "../s39_desktop_swap_baseline/DESKTOP_BASELINE_CONTRACT.json"
    )
    require(
        runtime_contract_value == {
            "desktop_contract_path":
                "../s39_desktop_swap_baseline/"
                "DESKTOP_BASELINE_CONTRACT.json",
            "desktop_contract_sha256":
                "b2e87de77c86d2ff880882dea8210c9cff314b2b63ff673c93836d9c86a77d2f",
            "expected_gpu_name": "NVIDIA GeForce RTX 4060 Ti",
            "expected_gpu_uuid":
                "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
            "models": {
                "qwen3-14b-q4_k_m": {
                    "bytes": 9001752960,
                    "sha256":
                        "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0",
                },
                "qwen3-8b-q8_0": {
                    "bytes": 8709518112,
                    "sha256":
                        "408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6",
                },
            },
            "serving": {
                "batch_size": 2048,
                "cache_type_k": "f16",
                "cache_type_v": "f16",
                "context_size": 4096,
                "continuous_batching": True,
                "flash_attention": True,
                "maximum_active_requests": 8,
                "parallel_slots": 8,
                "physical_ubatch_size": 512,
                "split_mode": "none",
            },
        },
        "contract.runtime_source: mismatch",
    )
    desktop_contract_path = check_bound_file(
        path,
        runtime_source,
        "desktop_contract_path",
        "desktop_contract_sha256",
    )
    desktop_contract = read_json(desktop_contract_path, "desktop_contract")
    require(
        desktop_contract.get("device", {}).get("gpu_name")
        == runtime_source["expected_gpu_name"]
        and desktop_contract.get("device", {}).get("gpu_uuid")
        == runtime_source["expected_gpu_uuid"],
        "desktop_contract: GPU identity mismatch",
    )
    for model_id, expected_model in runtime_source["models"].items():
        source_model = desktop_contract.get("models", {}).get(model_id)
        require(
            isinstance(source_model, dict)
            and source_model.get("bytes") == expected_model["bytes"]
            and source_model.get("sha256") == expected_model["sha256"],
            f"desktop_contract: model mismatch {model_id}",
        )

    matrix = contract.get("matrix")
    require(isinstance(matrix, dict), "contract.matrix: expected object")
    require(
        set(matrix) == {
            "C1_GPU_ONLY_OPTIMIZED",
            "C2_GPU_PLUS_CPU_WARM_EXECUTOR",
            "C3_DUAL_PARTIAL_OFFLOAD",
            "C4_TWO_GPU_ORACLE",
            "T1_PHONE_WARM_TIER",
            "T2_PHONE_NO_PROMOTION",
        },
        "contract.matrix: mode set mismatch",
    )
    require(
        matrix["C1_GPU_ONLY_OPTIMIZED"] == {
            "cache_regimes": ["WARM_HOST_CACHE", "COLD_NVME"],
            "executor_placement": "GPU_ONE_MODEL_AT_A_TIME",
            "primary": True,
            "promotion_enabled": True,
            "warm_executor": None,
        },
        "contract.matrix.C1: mismatch",
    )
    require(
        matrix["C2_GPU_PLUS_CPU_WARM_EXECUTOR"] == {
            "cache_regimes": ["WARM_HOST_CACHE"],
            "executor_placement": "GPU_HOT_CPU_ALTERNATE",
            "primary": True,
            "promotion_enabled": True,
            "warm_executor": "CPU_RAM",
        },
        "contract.matrix.C2: mismatch",
    )
    require(
        matrix["C3_DUAL_PARTIAL_OFFLOAD"] == {
            "cache_regimes": ["WARM_HOST_CACHE"],
            "executor_placement":
                "BOTH_MODELS_PARTIAL_ON_ONE_GPU_PLUS_CPU",
            "minimum_gpu_headroom_bytes": 536870912,
            "profile_lock_required_before_trace": True,
            "promotion_enabled": False,
            "warm_executor": None,
        },
        "contract.matrix.C3: mismatch",
    )
    require(
        matrix["C4_TWO_GPU_ORACLE"] == {
            "cache_regimes": ["WARM_HOST_CACHE"],
            "executor_placement": "ONE_MODEL_PER_GPU",
            "optional": True,
            "resource_matched": False,
            "warm_executor": "GPU1",
        },
        "contract.matrix.C4: mismatch",
    )
    require(
        matrix["T1_PHONE_WARM_TIER"] == {
            "cache_regimes": ["WARM_HOST_CACHE"],
            "executor_placement": "GPU_HOT_OP15_OP12_ALTERNATE",
            "k_extra": 0,
            "primary": True,
            "promotion_enabled": True,
            "replay": "PATH_MATCHED_TOKEN_HISTORY",
            "warm_executor": "OP15_OP12",
        },
        "contract.matrix.T1: mismatch",
    )
    require(
        matrix["T2_PHONE_NO_PROMOTION"] == {
            "cache_regimes": ["WARM_HOST_CACHE"],
            "executor_placement":
                "GPU_HOT_OP15_OP12_ALTERNATE_NO_PROMOTION",
            "k_extra": 0,
            "promotion_enabled": False,
            "warm_executor": "OP15_OP12",
        },
        "contract.matrix.T2: mismatch",
    )

    manifest_path = check_bound_file(
        path, workload, "input_manifest_path", "input_manifest_sha256")
    requests_path = check_bound_file(
        path, workload, "requests_path", "requests_sha256")
    switches_path = check_bound_file(
        path, workload, "switches_path", "switches_sha256")

    manifest = read_json(manifest_path, "input_manifest")
    files = manifest.get("files")
    require(isinstance(files, dict), "input_manifest.files: expected object")
    for name, target, digest in (
        ("DESKTOP_REQUESTS.jsonl", requests_path, workload["requests_sha256"]),
        ("DESKTOP_SWITCHES.jsonl", switches_path, workload["switches_sha256"]),
    ):
        record = files.get(name)
        require(isinstance(record, dict), f"input_manifest: missing {name}")
        require(record.get("sha256") == digest, f"input_manifest: {name} digest")
        require(record.get("bytes") == target.stat().st_size,
                f"input_manifest: {name} bytes")

    requests = read_jsonl(requests_path, "requests")
    counts = validate_requests(requests, workload)
    switches = read_jsonl(switches_path, "switches")
    validate_switches(switches, workload, requests)

    baselines = contract.get("baselines")
    require(isinstance(baselines, dict), "contract.baselines: expected object")
    c0 = baselines.get("C0_EXISTING_GPU_SWITCH")
    require(isinstance(c0, dict) and c0.get("read_only") is True,
            "contract: historical C0 must be read-only")
    require(
        set(baselines) == {"C0_EXISTING_GPU_SWITCH"}
        and c0.get("policy") == "historical_non_coalescing",
        "contract: historical C0 policy mismatch",
    )
    for prefix in ("evidence", "campaign", "analysis"):
        check_bound_file(
            path,
            c0,
            f"{prefix}_manifest_path",
            f"{prefix}_manifest_sha256",
        )

    return {
        "models": counts,
        "request_count": len(requests),
        "status": "S40_INPUTS_VALID",
        "switch_count": len(switches),
    }


SERVING_VALUE_FLAGS = (
    ("context_size", "--ctx-size", ("-c", "--ctx-size")),
    ("parallel_slots", "--parallel", ("-np", "--parallel")),
    ("batch_size", "--batch-size", ("-b", "--batch-size")),
    (
        "physical_ubatch_size",
        "--ubatch-size",
        ("-ub", "--ubatch-size"),
    ),
    (
        "cache_type_k",
        "--cache-type-k",
        ("-ctk", "--cache-type-k"),
    ),
    (
        "cache_type_v",
        "--cache-type-v",
        ("-ctv", "--cache-type-v"),
    ),
    ("split_mode", "--split-mode", ("-sm", "--split-mode")),
)

SERVING_FORBIDDEN_FLAGS = (
    "--api-key",
    "--api-key-file",
    "--ui-mcp-proxy",
    "--ui_mcp_proxy",
    "--webui-mcp-proxy",
    "--webui_mcp_proxy",
    "--tools",
    "-ag",
    "--agent",
)


def validate_serving_argv(
        argv: Any,
        serving: Any,
        field: str) -> dict[str, str | bool]:
    require(
        isinstance(argv, list)
        and argv
        and all(isinstance(item, str) and item for item in argv),
        f"{field}: expected command array",
    )
    require(
        isinstance(serving, dict)
        and serving.get("maximum_active_requests")
        == serving.get("parallel_slots"),
        f"{field}: invalid serving contract",
    )
    require(
        not any(
            item in SERVING_FORBIDDEN_FLAGS
            or any(
                item.startswith(flag + "=")
                for flag in SERVING_FORBIDDEN_FLAGS
            )
            for item in argv
        ),
        f"{field}: forbidden server feature",
    )
    normalized: dict[str, str | bool] = {}
    for key, canonical, aliases in SERVING_VALUE_FLAGS:
        positions = [
            index for index, item in enumerate(argv)
            if item in aliases
            or any(item.startswith(alias + "=") for alias in aliases)
        ]
        require(
            len(positions) == 1
            and argv[positions[0]] == canonical
            and positions[0] + 1 < len(argv)
            and argv[positions[0] + 1] == str(serving[key]),
            f"{field}: {canonical} must match the serving contract",
        )
        normalized[key] = argv[positions[0] + 1]

    flash_aliases = ("-fa", "--flash-attn")
    flash_positions = [
        index for index, item in enumerate(argv)
        if item in flash_aliases
        or any(item.startswith(alias + "=") for alias in flash_aliases)
    ]
    expected_flash = "on" if serving["flash_attention"] else "off"
    require(
        len(flash_positions) == 1
        and argv[flash_positions[0]] == "--flash-attn"
        and flash_positions[0] + 1 < len(argv)
        and argv[flash_positions[0] + 1] == expected_flash,
        f"{field}: --flash-attn must match the serving contract",
    )
    normalized["flash_attention"] = expected_flash

    positive = [
        index for index, item in enumerate(argv)
        if item in {"-cb", "--cont-batching"}
        or item.startswith("-cb=")
        or item.startswith("--cont-batching=")
    ]
    negative = [
        index for index, item in enumerate(argv)
        if item == "--no-cont-batching"
        or item.startswith("--no-cont-batching=")
    ]
    if serving["continuous_batching"]:
        require(
            len(positive) == 1
            and argv[positive[0]] == "--cont-batching"
            and not negative,
            f"{field}: --cont-batching must match the serving contract",
        )
    else:
        require(
            not positive
            and len(negative) == 1
            and argv[negative[0]] == "--no-cont-batching",
            f"{field}: --cont-batching must match the serving contract",
        )
    normalized["continuous_batching"] = serving["continuous_batching"]
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    try:
        result = validate_contract(args.contract)
        print(result)
        return 0
    except EvidenceError as error:
        print(f"validate_inputs: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
