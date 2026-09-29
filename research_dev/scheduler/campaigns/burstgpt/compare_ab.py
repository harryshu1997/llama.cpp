#!/usr/bin/env python3
"""Fail-closed comparison for matched BurstGPT A/B runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import statistics
import sqlite3
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler._internal.adaptive_decode_contracts import (  # noqa: E402
    AdaptiveDecodeError,
    AdaptiveDecodeGroupedObservation,
)
from research_dev.scheduler.adapters.llama_server import parse_llama_server_ffn_call  # noqa: E402


EXPECTED_COUNTS = {"gemma": 17, "llama": 10, "qwen": 57}
PHONE_MARKERS = ("adreno", "functionfs", "htp", "op15", "phone", "usb")
GPU_MARKERS = ("cuda", "gpu")
TARGET_SAVING_PPM = 250_000


class ComparisonError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


def canonical(value: object) -> bytes:
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


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, "result must be an object: " + str(path))
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def cuda_graph_evidence(path: Path) -> dict[str, Any]:
    """Read real CUDA API and GPU graph activity from a read-only Nsight export."""
    require(path.is_file(), "CUDA graph trace is absent")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        require({"StringIds", "CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"}
                <= tables, "CUDA graph activity is absent")
        raw = connection.execute(
            "SELECT s.value,r.returnValue,COUNT(*),SUM(r.end-r.start) "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
            "WHERE s.value LIKE 'cuda%Graph%' OR s.value LIKE 'cuda%Capture%' "
            "GROUP BY s.value,r.returnValue ORDER BY s.value,r.returnValue"
        ).fetchall()
        activities, gpu_ns, instances = connection.execute(
            "SELECT COUNT(*),SUM(end-start),COUNT(DISTINCT graphExecId) "
            "FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
        ).fetchone()
        api = {name: {"calls": count, "return_value": status, "host_duration_ns": duration}
               for name, status, count, duration in raw}
        require(raw and all(row[1] == 0 for row in raw), "CUDA graph API failed")

        def count(prefix):
            return sum(row[2] for row in raw if row[0].split("_v", 1)[0] == prefix)

        captures = count("cudaStreamEndCapture")
        launches = count("cudaGraphLaunch")
        require(captures > 0 and launches > 0 and activities == launches,
                "CUDA graph capture/replay is not physically demonstrated")
        instantiations = count("cudaGraphInstantiate") + count("cudaGraphInstantiateWithFlags")
        updates = count("cudaGraphExecUpdate")
        return {
            "schema": "research-cuda-graph-evidence-v1",
            "trace_sha256": file_sha256(path), "api": api,
            "captures": captures, "instantiations": instantiations,
            "executable_updates": updates,
            "recaptures_reusing_executable": max(0, updates - instantiations),
            "recapture_definition": "Executable updates beyond initial instantiation; all raw counters retained.",
            "launches": launches, "gpu_graph_activities": activities,
            "gpu_graph_instances": instances, "gpu_graph_duration_ns": gpu_ns,
        }


def frozen_reference_identity(result: dict[str, Any]) -> dict[str, Any]:
    """Stable comparison contract; expected values must be frozen before a run."""
    large = {result["model_roles"][name] for name in ("gemma", "qwen")}
    parents = {}
    for row in request_rows(result):
        if row["model_id"] in large:
            parent = _desktop_parent_identity(row)
            require(parents.setdefault(row["model_id"], parent) == parent,
                    "reference changed its desktop parent during execution")
    fields = (
        "catalog_sha256", "execution_identity", "model_artifacts", "model_roles",
        "trace_identity", "replay_schedule", "maximum_latency_ppm",
        "initial_observation_inputs", "adaptive_controller_configuration",
        "preparation_accounting", "fixed_phone_residency", "selection_mode",
    )
    require(all(key in result for key in fields), "frozen reference contract is incomplete")
    return {
        **{key: result[key] for key in fields},
        "workload": [list(row) for row in workload_identity(result)],
        "desktop_parents": parents,
        "energy_boundary": {
            key: result["trace_energy"].get(key) for key in (
                "energy_boundary_id", "attribution_kind",
            )
        } | {"measurement_evidence_ids": sorted(result["trace_energy"]["measurement_evidence_ids"])},
    }


def validate_frozen_reference(
    result: dict[str, Any], expected: dict[str, Any], *, expected_replay: dict[str, Any],
) -> None:
    require(result.get("status") == "PASS", "reference did not pass")
    result_counts(result, expected_replay)
    require_clean_execution(result, "reference")
    energy_domains(result)
    identity = frozen_reference_identity(result)
    expected = {**expected, "energy_boundary": {
        **expected["energy_boundary"],
        "measurement_evidence_ids": sorted(expected["energy_boundary"]["measurement_evidence_ids"]),
    }}
    require(identity == expected, "reference differs from its frozen experiment specification")
    modes = execution_identity(result)["cuda_graph_mode_by_artifact"]
    require(set(modes.values()) == {"default"}, "reference disabled CUDA graphs")
    fixed = result["fixed_phone_residency"]
    if fixed is None:
        require(result["selection_mode"] == "desktop-baseline"
                and all(not plan_uses_phone(row) for row in request_rows(result)),
                "desktop reference used phone execution")
        return
    plan = result.get("offline_phone_residency") or {}
    require(plan.get("state") == "READY", "fixed reference residency was not fully READY")
    target = plan.get("target_layout") or {}
    assignments = fixed["assignments"]
    keys = ("session_id", "artifact_sha256", "layer_mask", "maximum_columns")
    require(sorted(tuple(row[key] for key in keys) for row in target.get("shards", []))
            == sorted(tuple(row[key] for key in keys) for row in assignments),
            "fixed reference changed its assignment")
    require(len(plan.get("stages", [])) == len(assignments)
            and all(stage["state"] == "READY" for stage in plan["stages"]),
            "fixed reference reloaded or failed a session")
    physical = result.get("phone_residency_at_completion") or {}
    require(physical.get("load_count_by_session") == {
        row["session_id"]: 1 for row in assignments
    }, "fixed reference physical reload count differs")
    require({row["session_id"]: row["session_generation"]
             for row in physical.get("phone_shards", [])}
            == target.get("session_generation_by_id"),
            "fixed reference physical generation differs")


def reference_phone_power_sensitivity(result: dict[str, Any]) -> dict[str, Any]:
    domains = energy_domains(result)
    metadata = result["trace_energy"].get("estimation_metadata") or {}
    active = metadata.get("phone_active_time_ns")
    idle = metadata.get("phone_idle_time_ns")
    require(type(active) is int and active >= 0 and type(idle) is int and idle >= 0,
            "reference lacks phone active/idle accounting")
    require(active + idle == result["paid_end_ns"] - result["paid_start_ns"],
            "phone accounting differs from the paid boundary")
    require(metadata.get("phone_idle_power_mw") == 875
            and metadata.get("phone_active_power_mw") == 4500,
            "reference changed its assumed phone power policy")
    desktop = sum(value for key, value in domains.items()
                  if not any(marker in key for marker in ("phone", "op15")))
    nominal = sum(value for key, value in domains.items()
                  if any(marker in key for marker in ("phone", "op15")))
    require(abs(nominal - (active * 4500 + idle * 875) / 1_000_000) <= 1,
            "nominal phone estimate differs from its recorded durations")
    return {
        str(power): {
            "phone_energy_uj": round((active * power + idle * 875) / 1_000_000),
            "fleet_energy_uj": desktop + round((active * power + idle * 875) / 1_000_000),
        }
        for power in (3000, 4500, 6000)
    }


def reference_assistance_summary(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    by_request = defaultdict(list)
    for window in load_adaptive_windows(path, result):
        by_request[window["request_id"]].append(window)
    summary = {}
    for request in request_rows(result):
        model = request["model_id"]
        row = summary.setdefault(model, {
            "output_tokens": 0, "assisted_tokens": 0, "fraction_weighted_token_equivalents": 0.0,
            "calls_by_session": Counter(), "layer_column_contracts": [], "fractions_ppm": Counter(),
        })
        row["output_tokens"] += request["output_tokens"]
        proof = request.get("physical_execution_proof") or {}
        contracts = _native_fraction_counts(path, request, by_request[request["request_id"]])
        assisted = sum(count for _layers, _columns, _fraction, count in contracts)
        require(assisted <= request["output_tokens"], "assisted tokens exceed decode tokens")
        row["assisted_tokens"] += assisted
        row["fractions_ppm"]["0"] += request["output_tokens"] - assisted
        for layers, columns, fraction, count in contracts:
            row["fractions_ppm"][str(fraction)] += count
            row["fraction_weighted_token_equivalents"] += count * fraction / 1_000_000
            row["layer_column_contracts"].append({
                "layers": list(layers), "columns": columns, "split_fraction_ppm": fraction,
                "physical_decode_tokens": count,
            })
        for session in proof.get("phone_calls_by_session") or []:
            row["calls_by_session"][session["session_id"]] += session["calls"]
    for row in summary.values():
        row["calls_by_session"] = dict(sorted(row["calls_by_session"].items()))
        row["fractions_ppm"] = dict(sorted(row["fractions_ppm"].items()))
        row["assisted_decode_percentage"] = row["assisted_tokens"] * 100 / row["output_tokens"]
        row["weighted_all_decode_percentage"] = row["fraction_weighted_token_equivalents"] * 100 / row["output_tokens"]
    return summary


def _native_fraction_counts(path: Path, request, windows):
    policies = {}
    for window in windows:
        ack = window.get("applied_ack")
        if ack is None:
            continue
        policy = window["policy"]
        identity = (tuple(policy["layer_indices"]), policy["columns"], policy["split_fraction_ppm"])
        generation = ack["plan_generation"]
        require(policies.setdefault(generation, identity) == identity,
                "one acknowledged generation has conflicting fraction contracts")
    calls = defaultdict(Counter)
    actual_by_layer = Counter()
    for stderr in sorted(path.parent.glob("*.stderr")):
        for line in stderr.read_text(encoding="utf-8").splitlines():
            call = parse_llama_server_ffn_call(line)
            if call is None:
                continue
            context = [row for row in call.contexts if row.scheduler_request_id == request["request_id"]]
            if not context:
                continue
            require(len(context) == 1 and len(call.contexts) == 1
                    and context[0].rows == call.tokens == 1,
                    "reference coverage requires singleton decode calls")
            generation = context[0].plan_generation
            require(generation in policies, "native call lacks an acknowledged policy generation")
            layers, columns, fraction = policies[generation]
            require(fraction > 0 and call.layer in layers and call.columns == columns,
                    "native call differs from its fraction/layer/column contract")
            calls[generation][call.layer] += 1
            actual_by_layer[call.layer] += 1
    proof = request.get("physical_execution_proof") or {}
    require(sum(actual_by_layer.values()) == proof.get("phone_call_count", 0),
            "native coverage and terminal phone calls differ")
    require(dict(actual_by_layer) == {row["layer"]: row["calls"] for row in proof.get("phone_calls_by_layer", [])},
            "native coverage and per-layer terminal proof differ")
    counts = Counter()
    for generation, layers_count in calls.items():
        layers, columns, fraction = policies[generation]
        require(set(layers_count) == set(layers) and len(set(layers_count.values())) == 1,
                "native generation has incomplete per-token layer coverage")
        counts[(layers, columns, fraction)] += next(iter(layers_count.values()))
    return [(*key, count) for key, count in sorted(counts.items())]


def summarize_frozen_reference_set(
    specification: dict[str, Any], result_paths: dict[str, Path],
) -> dict[str, Any]:
    """Validate each predeclared arm; only the upstream native implementation may differ."""
    contracts = specification["contracts"]
    require(set(result_paths) == set(contracts), "frozen reference set is incomplete")
    results = {name: load_object(path) for name, path in result_paths.items()}
    default = results["desktop-default-cuda"]
    matched = results["desktop-matched-cuda"]
    common = ("model_artifacts", "model_roles", "trace_identity", "replay_schedule",
              "maximum_latency_ppm", "initial_observation_inputs", "adaptive_controller_configuration",
              "preparation_accounting", "energy_boundary", "workload")
    rows = {}
    phone_identity = phone_preflight_identity(result_paths["desktop-matched-cuda"])
    for name, result in results.items():
        path = result_paths[name]
        validate_frozen_reference(result, contracts[name], expected_replay=specification["expected_replay"])
        require(all(contracts[name][key] == contracts["desktop-matched-cuda"][key] for key in common),
                "frozen references differ in workload, evidence, policy, or accounting")
        require(phone_preflight_identity(path) == phone_identity, "frozen phone artifacts differ")
        for request in request_rows(result):
            expected = specification["desktop_placement_sha256_by_reference"][name][request["model_id"]]
            actual = (request["execution_command"].get("operator_plan") or {}).get("desktop_placement_sha256")
            require(actual == expected, "reference did not execute its frozen desktop placement")
            launch = specification["desktop_launch_parameters_by_reference"][name][request["model_id"]]
            actual_launch = request["execution_command"]["adapter_parameters"]
            for key in ("model_alias", "gpu_layers", "context_size", "parallel", "batch_size",
                        "ubatch_size", "threads", "threads_batch", "cpu_affinity", "cuda_graph_mode",
                        "desktop_launch_mode", "cpu_device_id", "gpu_device_id"):
                require(actual_launch.get(key) == launch.get(key), "reference desktop launch differs: " + key)
        if name.startswith("fixed-"):
            validate_matched(matched, result, expected_replay=specification["expected_replay"])
        rows[name] = {
            "result_sha256": file_sha256(path), "duration_us": result["duration_us"],
            "energy_uj_by_domain": energy_domains(result),
            "phone_power_sensitivity": reference_phone_power_sensitivity(result),
            "requests": [{"request_id": row["request_id"], "model_id": row["model_id"],
                           "service_latency_us": row["actual_latency_us"],
                           "arrival_us": row["replay_arrival_us"],
                           "completion_us": row["terminal_ticket"]["execution_receipt"]["finished_us"]}
                          for row in request_rows(result)],
            "assistance": reference_assistance_summary(path, result),
            "transport": transport_summary(result),
            "cuda_graphs": cuda_graph_evidence(path.parent.parent / "CUDA.sqlite"),
            "physical_residency": result.get("phone_residency_at_completion"),
        }
    native_keys = {"server", "resident_server"}
    require(specification["desktop_placement_sha256_by_reference"]["desktop-default-cuda"]
            == specification["desktop_placement_sha256_by_reference"]["desktop-matched-cuda"],
            "upstream comparison changed desktop placement")
    for model, parent in contracts["desktop-default-cuda"]["desktop_parents"].items():
        matched_parent = contracts["desktop-matched-cuda"]["desktop_parents"][model]
        require({k: v for k, v in parent.items() if not k.startswith("capacity_parent_")}
                == {k: v for k, v in matched_parent.items() if not k.startswith("capacity_parent_")},
                "upstream comparison changed parent launch settings")
    a, b = execution_identity(default), execution_identity(matched)
    require({k: v for k, v in a.items() if k != "binaries"}
            == {k: v for k, v in b.items() if k != "binaries"},
            "upstream comparison changed harness source or graph mode")
    require({k: v for k, v in a["binaries"].items() if k not in native_keys and not k.startswith("host_dependency:")}
            == {k: v for k, v in b["binaries"].items() if k not in native_keys and not k.startswith("host_dependency:")},
            "upstream comparison changed unrelated binaries")
    best = min((name for name in rows if name.startswith("fixed-")),
               key=lambda name: rows[name]["phone_power_sensitivity"]["4500"]["fleet_energy_uj"])
    for name, row in rows.items():
        row["saving_ppm_vs_reference"] = {
            parent: {
                power: round((1 - estimate["fleet_energy_uj"] /
                              rows[parent]["phone_power_sensitivity"][power]["fleet_energy_uj"]) * 1_000_000)
                for power, estimate in row["phone_power_sensitivity"].items()
            } for parent in ("desktop-default-cuda", "desktop-matched-cuda", best)
        }
    return {"schema": "cuda-graph-fixed-reference-comparison-v1", "comparison_valid": True,
            "best_tested_fixed_layout_nominal": best, "references": rows,
            "scope": "One preparation-inclusive small run per reference; not a steady-state or replicated result."}


def _check_object_hash(value: dict[str, Any], key: str) -> None:
    body = {name: item for name, item in value.items() if name != key}
    require(value.get(key) == "sha256:" + hashlib.sha256(canonical(body)[:-1]).hexdigest(),
            key + " differs")


def _is_sha256(value: object) -> bool:
    return (type(value) is str and value.startswith("sha256:")
            and len(value) == 71 and all(c in "0123456789abcdef" for c in value[7:]))


def phone_preflight_identity(result_path: Path) -> dict[str, Any]:
    receipt = load_object(result_path.parent / "DIRECT_PHONE_PREFLIGHT.json")
    keys = ("remote_hashes", "ffn_shard_indexes", "phone_kernel_release", "usb_close_sha256")
    require(all(key in receipt for key in keys), "phone preflight identity is incomplete")
    for key in ("remote_hashes", "ffn_shard_indexes"):
        hashes = receipt[key]
        require(type(hashes) is dict and (key != "remote_hashes" or bool(hashes)),
                "phone preflight hash map is invalid")
        require(all(_is_sha256(value) for value in hashes.values()),
                "phone preflight hash is invalid")
    require(_is_sha256(receipt["usb_close_sha256"]), "phone close identity is invalid")
    require(type(receipt["phone_kernel_release"]) is str and bool(receipt["phone_kernel_release"]),
            "phone preflight kernel identity is invalid")
    return {key: receipt[key] for key in keys}


def percentile(values: Iterable[float], fraction: float) -> float | None:
    rows = sorted(float(value) for value in values)
    if not rows:
        return None
    index = (len(rows) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return rows[low]
    return rows[low] + (rows[high] - rows[low]) * (index - low)


def request_rows(result: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = result.get("request_results")
    require(type(rows) is list, "request_results must be a list")
    require(all(type(row) is dict for row in rows), "invalid request result")
    return tuple(rows)


def terminal_ticket(row: dict[str, Any]) -> dict[str, Any]:
    ticket = row.get("terminal_ticket")
    require(type(ticket) is dict, "request lacks terminal ticket")
    require(
        ticket.get("dispatch_state") == "COMPLETED",
        "request terminal state is not COMPLETED",
    )
    return ticket


def execution_plan(row: dict[str, Any]) -> dict[str, Any]:
    plan = terminal_ticket(row).get("execution_plan")
    require(type(plan) is dict, "request lacks execution plan")
    return plan


def _text_values(value: object) -> Iterable[str]:
    if type(value) is str:
        yield value.lower()
    elif type(value) is list:
        for row in value:
            yield from _text_values(row)
    elif type(value) is dict:
        for key, row in value.items():
            yield str(key).lower()
            yield from _text_values(row)


def uses_marker(value: object, markers: tuple[str, ...]) -> bool:
    return any(
        marker in text
        for text in _text_values(value)
        for marker in markers
    )


def plan_uses_phone(row: dict[str, Any]) -> bool:
    plan = execution_plan(row)
    contract = plan.get("execution_contract") or {}
    proof = row.get("physical_execution_proof") or {}
    return bool(
        contract.get("phone_shards")
        or proof.get("phone_call_count", 0)
        or uses_marker(plan.get("device_ids", []), PHONE_MARKERS)
    )


def plan_uses_gpu(row: dict[str, Any]) -> bool:
    plan = execution_plan(row)
    binding = terminal_ticket(row).get("binding") or {}
    participants = binding.get("participants") or []
    return (
        uses_marker(plan.get("device_ids", []), GPU_MARKERS)
        or uses_marker(participants, GPU_MARKERS)
        or uses_marker(
            (row.get("execution_command") or {}).get(
                "adapter_parameters", {}
            ).get("gpu_device_id", ""),
            GPU_MARKERS,
        )
    )


def _named_replay_counts(
    result: dict[str, Any], expected_replay: dict[str, Any]
) -> dict[str, int]:
    schema = "research-scheduler-burstgpt-replay-v1"
    require(expected_replay.get("schema") == schema, "expected replay schema differs")
    arrivals = expected_replay.get("arrivals")
    require(type(arrivals) is list and bool(arrivals), "expected replay is empty")
    expected = []
    for row in arrivals:
        require(type(row) is dict, "expected replay arrival is invalid")
        index, arrival = row.get("combined_request_index"), row.get("replay_arrival_us")
        require(type(index) is int and index >= 0 and type(arrival) is int
                and arrival >= 0, "expected replay arrival is invalid")
        expected.append((index, arrival))
    require(len({row[0] for row in expected}) == len(expected)
            and [row[1] for row in expected] == sorted(row[1] for row in expected),
            "expected replay is duplicated or unordered")
    replay = result.get("replay_schedule") or {}
    rows = replay.get("schedule")
    require(replay.get("schema") == schema
            and type(expected_replay.get("trace_name")) is str
            and bool(expected_replay["trace_name"])
            and replay.get("trace_name") == expected_replay["trace_name"]
            and type(rows) is list and all(type(row) is dict for row in rows),
            "result named replay identity differs")
    require([(row.get("combined_request_index"), row.get("replay_arrival_us"))
             for row in rows] == expected, "result named replay arrivals differ")
    require(replay.get("schedule_sha256") == "sha256:" + hashlib.sha256(
                canonical(rows)).hexdigest(), "result replay hash differs")
    requests = request_rows(result)
    require(len(requests) == len(expected)
            and len({row.get("request_id") for row in requests}) == len(expected)
            and len({row.get("combined_request_index") for row in requests}) == len(expected),
            "result is not a complete named replay")
    require(sorted((row.get("combined_request_index"), row.get("replay_arrival_us"),
                    row.get("request_id")) for row in requests)
            == sorted((row["combined_request_index"], row["replay_arrival_us"],
                       row.get("request_id")) for row in rows),
            "result requests differ from named replay")
    roles = result.get("model_roles") or {}
    require(set(roles) == set(EXPECTED_COUNTS) and len(set(roles.values())) == len(roles),
            "model role identity is invalid")
    by_model = Counter(row.get("model_id") for row in requests)
    require(set(by_model) <= set(roles.values()), "request model role is invalid")
    return {role: by_model[model_id] for role, model_id in roles.items()}


def result_counts(
    result: dict[str, Any], expected_replay: dict[str, Any] | None = None
) -> dict[str, int]:
    counts = result.get("counts")
    require(type(counts) is dict, "result counts are absent")
    actual = {
        role: counts.get(role)
        for role in EXPECTED_COUNTS
    }
    expected = (EXPECTED_COUNTS if expected_replay is None
                else _named_replay_counts(result, expected_replay))
    require(actual == expected, "model request counts differ")
    expected_total = sum(expected.values())
    require(
        counts.get("requests") == expected_total
        and counts.get("terminals") == expected_total
        and len(request_rows(result)) == expected_total,
        "result is not a complete " + str(expected_total) + "-request run",
    )
    return actual


def workload_identity(result: dict[str, Any]) -> tuple[tuple[object, ...], ...]:
    artifacts = result.get("model_artifacts")
    require(type(artifacts) is dict, "model artifacts are absent")
    rows = []
    for row in request_rows(result):
        model_id = row.get("model_id")
        artifact = artifacts.get(model_id)
        require(type(artifact) is dict, "request model artifact is absent")
        rows.append((
            row.get("combined_request_index"),
            row.get("request_id"),
            model_id,
            artifact.get("artifact_sha256"),
            row.get("input_tokens"),
            row.get("output_tokens"),
            row.get("prompt_sha256"),
            row.get("seed"),
        ))
    require(
        all(
            type(row[0]) is int
            and type(row[1]) is str
            and type(row[2]) is str
            and type(row[3]) is str
            and type(row[4]) is int
            and type(row[5]) is int
            and type(row[6]) is str
            and type(row[7]) is int
            for row in rows
        ),
        "request workload identity is incomplete",
    )
    return tuple(sorted(rows))


def execution_identity(result: dict[str, Any]) -> dict[str, Any]:
    identity = result.get("execution_identity")
    require(type(identity) is dict, "execution identity is absent")
    require(
        type(identity.get("source_manifest_sha256")) is str,
        "source manifest identity is absent",
    )
    binaries = identity.get("binaries")
    require(type(binaries) is dict and binaries, "binary identities are absent")
    require(
        all(type(value) is str for value in binaries.values()),
        "binary identity is invalid",
    )
    graph_modes = identity.get("cuda_graph_mode_by_artifact")
    require(
        type(graph_modes) is dict and bool(graph_modes)
        and all(
            type(artifact) is str and artifact.startswith("sha256:")
            and mode in ("default", "disabled")
            for artifact, mode in graph_modes.items()
        ),
        "CUDA graph mode identity is absent or invalid",
    )
    return identity


def energy_domains(result: dict[str, Any]) -> dict[str, int]:
    trace_energy = result.get("trace_energy")
    require(type(trace_energy) is dict, "trace fleet energy is absent")
    domains = trace_energy.get("fleet_energy_uj_by_domain")
    require(type(domains) is dict, "trace energy domains are absent")
    require(
        all(type(value) is int and value >= 0 for value in domains.values()),
        "trace energy domain is invalid",
    )
    categories = {
        "cpu": sum(value for key, value in domains.items() if "cpu" in key),
        "gpu": sum(
            value for key, value in domains.items()
            if "gpu" in key or "cuda" in key
        ),
        "phone": sum(
            value for key, value in domains.items()
            if "phone" in key or "op15" in key
        ),
    }
    require(all(value > 0 for value in categories.values()), "fleet boundary is incomplete")
    return {str(key): int(value) for key, value in sorted(domains.items())}


def recursive_counter(value: object, key: str) -> int:
    if isinstance(value, dict):
        return int(value.get(key, 0) or 0) + sum(
            recursive_counter(row, key) for row in value.values()
        )
    if isinstance(value, list):
        return sum(recursive_counter(row, key) for row in value)
    return 0


def require_clean_execution(result: dict[str, Any], arm: str) -> None:
    for row in request_rows(result):
        quality = row.get("output_quality") or {}
        require(
            quality.get("accepted") is True,
            arm + " output quality was not accepted",
        )
        require(
            not row.get("recoveries"),
            arm + " contains an unexplained fallback",
        )
        require(
            terminal_ticket(row).get("selection_mode")
                == result.get("selection_mode"),
            arm + " ticket selection mode differs",
        )
    receipts = (
        result.get("direct_phone_receipts") or [],
        result.get("transport_receipts") or [],
    )
    require(
        sum(recursive_counter(rows, "reset_recoveries") for rows in receipts)
            == 0,
        arm + " contains a USB reset",
    )


def _desktop_parent_identity(row: dict[str, Any]) -> dict[str, Any]:
    params = (row.get("execution_command") or {}).get("adapter_parameters") or {}
    keys = (
        "capacity_parent_source_placement_sha256",
        "capacity_parent_qualification_sha256",
        "capacity_parent_calibration_live_free_bytes",
        "capacity_parent_peak_vram_bytes",
        "capacity_parent_required_with_reserve_bytes",
        "capacity_parent_maximum_gpu_layers",
        "gpu_layers", "gpu_device_id", "cpu_device_id", "cuda_graph_mode",
        "context_size", "parallel", "batch_size", "ubatch_size",
    )
    require(all(key in params for key in keys), "desktop parent qualification is incomplete")
    for key in keys[:2]:
        value = params[key]
        require(type(value) is str and value.startswith("sha256:") and len(value) == 71,
                "desktop parent qualification hash is invalid")
    return {**{key: params[key] for key in keys},
            "desktop_launch_mode": params.get("desktop_launch_mode", "canonical")}


def source_manifest_for(result_path: Path) -> tuple[str, dict[str, Any]] | None:
    """The launch manifest written next to a result's run/ directory, with its file digest."""
    candidate = result_path.resolve().parent.parent / "SOURCE_MANIFEST.json"
    if not candidate.is_file():
        return None
    return file_sha256(candidate), load_object(candidate)


def source_identity_matches(
    baseline: dict[str, Any], adaptive: dict[str, Any],
    source_manifests: tuple[tuple[str, dict[str, Any]], tuple[str, dict[str, Any]]] | None,
) -> str:
    """Accept equal manifest digests, or two launch manifests that pin identical
    source files and commit for each arm; everything else fails closed.

    Every launch writes its own SOURCE_MANIFEST.json embedding the resolved
    campaign (ids, output paths, selection mode), so two arms of one deploy
    tree carry different manifest digests while pinning byte-identical
    source. Content identity is what the comparison needs; the manifest
    file must still hash to the digest the result recorded.
    """
    identity_a = execution_identity(baseline)
    identity_b = execution_identity(adaptive)
    if identity_a == identity_b:
        return "manifest-digest-equal"
    stripped_a = {k: v for k, v in identity_a.items() if k != "source_manifest_sha256"}
    stripped_b = {k: v for k, v in identity_b.items() if k != "source_manifest_sha256"}
    require(stripped_a == stripped_b, "A/B source or binary identity differs")
    require(source_manifests is not None, "A/B source or binary identity differs")
    (digest_a, manifest_a), (digest_b, manifest_b) = source_manifests
    require(
        digest_a == identity_a["source_manifest_sha256"]
        and digest_b == identity_b["source_manifest_sha256"],
        "A/B source manifest does not match its result",
    )
    require(
        manifest_a.get("schema") == manifest_b.get("schema")
        and manifest_a.get("head") == manifest_b.get("head")
        and manifest_a.get("branch") == manifest_b.get("branch")
        and manifest_a.get("files") == manifest_b.get("files")
        and bool(manifest_a.get("files")),
        "A/B source or binary identity differs",
    )
    return "source-files-equal"


def validate_matched(
    baseline: dict[str, Any], adaptive: dict[str, Any],
    *, expected_replay: dict[str, Any] | None = None,
    source_manifests: tuple[tuple[str, dict[str, Any]], tuple[str, dict[str, Any]]] | None = None,
) -> str:
    require(
        baseline.get("status") == adaptive.get("status") == "PASS",
        "both arms must pass",
    )
    result_counts(baseline, expected_replay)
    result_counts(adaptive, expected_replay)
    require(
        baseline.get("selection_mode") == "desktop-baseline",
        "arm A is not desktop-baseline",
    )
    require(
        adaptive.get("selection_mode") == "energy-aware",
        "arm B is not normal energy-aware mode",
    )
    for field in (
        "catalog_sha256",
        "model_artifacts",
        "model_roles",
        "trace_identity",
        "replay_schedule",
        "preparation_accounting",
        "initial_observation_inputs",
        "adaptive_controller_configuration",
    ):
        require(
            baseline.get(field) == adaptive.get(field),
            "A/B identity differs: " + field,
        )
    source_identity = source_identity_matches(baseline, adaptive, source_manifests)
    require(
        workload_identity(baseline) == workload_identity(adaptive),
        "A/B prompts, seeds, or token work differ",
    )
    maximum_latency_ppm = baseline.get("maximum_latency_ppm")
    require(
        type(maximum_latency_ppm) is int
        and maximum_latency_ppm >= 1_000_000
        and adaptive.get("maximum_latency_ppm") == maximum_latency_ppm,
        "A/B configured latency allowance differs",
    )
    roles = baseline["model_roles"]
    require(
        set(roles) == set(EXPECTED_COUNTS),
        "model role identity is invalid",
    )
    large_models = {roles["qwen"], roles["gemma"]}
    adaptive_by_index = {row["combined_request_index"]: row for row in request_rows(adaptive)}
    for row in request_rows(baseline):
        require(not plan_uses_phone(row), "baseline executed a phone route")
        if row.get("model_id") in large_models:
            require(
                plan_uses_gpu(row),
                "large-model baseline did not physically use CUDA",
            )
            require(_desktop_parent_identity(row) == _desktop_parent_identity(
                        adaptive_by_index[row["combined_request_index"]]),
                    "A/B desktop parent placement or qualification differs")
    boundary_a, boundary_b = baseline.get("trace_energy") or {}, adaptive.get("trace_energy") or {}
    for field in ("energy_boundary_id", "attribution_kind", "measurement_evidence_ids"):
        require(boundary_a.get(field) is not None and boundary_a.get(field) == boundary_b.get(field),
                "A/B energy boundary differs: " + field)
    energy_domains(baseline)
    energy_domains(adaptive)
    require_clean_execution(baseline, "baseline")
    require_clean_execution(adaptive, "adaptive")
    return source_identity


def route_summary(result: dict[str, Any]) -> dict[str, dict[str, int]]:
    summary: dict[str, Counter[str]] = defaultdict(Counter)
    for row in request_rows(result):
        plan = execution_plan(row)
        summary[str(row["model_id"])][str(plan.get("route_family"))] += 1
    return {
        model: dict(sorted(counts.items()))
        for model, counts in sorted(summary.items())
    }


def session_summary(result: dict[str, Any]) -> dict[str, dict[str, int]]:
    calls: Counter[str] = Counter()
    bytes_by_session: Counter[str] = Counter()
    for row in request_rows(result):
        proof = row.get("physical_execution_proof") or {}
        for session in proof.get("phone_calls_by_session") or []:
            session_id = str(session.get("session_id"))
            calls[session_id] += int(session.get("calls", 0))
            bytes_by_session[session_id] += int(session.get("payload_bytes", 0))
    return {
        session_id: {
            "calls": calls[session_id],
            "payload_bytes": bytes_by_session[session_id],
        }
        for session_id in sorted(set(calls) | set(bytes_by_session))
    }


def load_adaptive_windows(
    result_path: Path, result: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    requests = request_rows(result)
    referenced = [row for row in requests
                  if (row.get("fraction_history") or {}).get("adaptive_grouped_observation_sha256")]
    if not referenced:
        return ()
    relative = result.get("adaptive_observation_store_path")
    require(type(relative) is str, "current adaptive observation store is absent")
    path = result_path.parent / relative
    require(path.is_file(), "current adaptive observation store is absent")
    value = load_object(path)
    groups = value.get("groups")
    require(type(groups) is list and all(type(group) is dict for group in groups),
            "adaptive observation groups are invalid")
    by_hash = defaultdict(list)
    for group in groups:
        by_hash[group.get("grouped_observation_sha256")].append(group)
    current = {}
    by_request = {row["request_id"]: row for row in requests}
    for row in referenced:
        history = row["fraction_history"]
        identity = history["adaptive_grouped_observation_sha256"]
        matches = by_hash[identity]
        require(len(matches) == 1, "current adaptive observation is missing or ambiguous")
        try:
            group = AdaptiveDecodeGroupedObservation.from_json(matches[0])
        except AdaptiveDecodeError as error:
            raise ComparisonError(str(error)) from error
        owner = by_request.get(group.request_id)
        require(owner is not None and group.terminal_status == "COMPLETED",
                "current adaptive observation owner is absent or incomplete")
        proof = row.get("physical_execution_proof") or {}
        _check_object_hash(proof, "proof_sha256")
        owner_ticket = terminal_ticket(owner).get("ticket_id")
        require(group.ticket_id == owner_ticket == history.get("adaptive_observation_ticket_id")
                and proof.get("ticket_id") == terminal_ticket(row).get("ticket_id")
                and proof.get("adaptive_grouped_observation_sha256") == identity
                and proof.get("adaptive_group_owner_request_id") == group.request_id,
                "current adaptive observation execution identity differs")
        require(group.model_artifact_sha256 == proof.get("artifact_sha256")
                == result["model_artifacts"][row["model_id"]]["artifact_sha256"],
                "current adaptive observation artifact differs")
        _desktop_parent_identity(row)
        executed_parent = (row["execution_command"].get("operator_plan") or {}).get(
            "desktop_placement_sha256"
        )
        require(_is_sha256(executed_parent) and group.desktop_placement_sha256 == executed_parent,
                "current adaptive observation desktop parent differs")
        if history.get("adaptive_observation_scope") == "decode_cohort_shared":
            require(any(row["request_id"] in window.cohort_member_request_ids
                        and window.energy_owner_request_id == group.request_id
                        for window in group.windows), "adaptive cohort membership differs")
        else:
            require(group.request_id == row["request_id"], "current adaptive request differs")
        current[identity] = group
    windows = []
    for identity in sorted(current):
        windows.extend(row.to_json() for row in current[identity].windows)
    return tuple(windows)


def _session_proof_key(row: dict[str, Any], *, terminal: bool) -> tuple[object, ...]:
    generation = row.get("session_generation")
    require(type(generation) is int and generation > 0, "phone proof generation is invalid")
    fields = ("session_id", "artifact_sha256", "resident_geometry_sha256", "operator_plan_sha256")
    require(all(type(row.get(key)) is str and bool(row[key]) for key in fields),
            "phone session proof identity is incomplete")
    require(all(_is_sha256(row[key]) for key in fields[1:]),
            "phone session proof hash is invalid")
    if not terminal:
        require(type(row.get("endpoint")) is str and bool(row["endpoint"])
                and row["endpoint"].isascii(), "phone endpoint proof is absent")
    endpoint = row.get("endpoint_sha256") if terminal else (
        "sha256:" + hashlib.sha256(str(row.get("endpoint", "")).encode("ascii")).hexdigest()
    )
    require(_is_sha256(endpoint), "phone endpoint proof is absent")
    return tuple(row[key] for key in fields) + (generation, endpoint)


def transport_summary(result: dict[str, Any]) -> dict[str, object]:
    receipts = result.get("direct_phone_receipts") or []
    terminals = [row["terminal"] for row in receipts if "terminal" in row]
    request_counts, terminal_counts = Counter(), Counter()
    request_bytes, terminal_bytes = Counter(), Counter()
    calls = payload = 0
    for row in request_rows(result):
        proof = row.get("physical_execution_proof") or {}
        count = proof.get("phone_call_count", 0)
        require(type(count) is int and count >= 0, "request phone calls are invalid")
        if not count:
            continue
        _check_object_hash(proof, "proof_sha256")
        require(proof.get("ticket_id") == terminal_ticket(row).get("ticket_id"),
                "phone execution ticket differs")
        calls += count
        require(type(proof.get("phone_payload_bytes")) is int
                and proof["phone_payload_bytes"] >= 0, "request phone bytes are invalid")
        payload += proof["phone_payload_bytes"]
        sessions = proof.get("phone_calls_by_session") or []
        require(all(type(s.get(key)) is int and s[key] >= 0
                    for s in sessions for key in ("calls", "payload_bytes")),
                "request phone session counters are invalid")
        require(sum(s["calls"] for s in sessions) == count,
                "request and session phone calls differ")
        require(sum(s["payload_bytes"] for s in sessions) == proof["phone_payload_bytes"],
                "request and session phone bytes differ")
        for session in sessions:
            require(session["artifact_sha256"] == proof["artifact_sha256"],
                    "phone session artifact differs")
            key = _session_proof_key(session, terminal=False)
            request_counts[key] += session["calls"]
            request_bytes[key] += session["payload_bytes"]
    for terminal in terminals:
        require(terminal.get("status") == 0, "phone terminal did not succeed")
        sessions = terminal.get("session_proofs") or []
        require(all(type(s.get(key)) is int and s[key] >= 0
                    for s in sessions for key in ("calls", "h2d_bytes")),
                "native phone session counters are invalid")
        require(sum(s["calls"] for s in sessions) == terminal.get("requests"),
                "native terminal and session calls differ")
        for session in sessions:
            key = _session_proof_key(session, terminal=True)
            terminal_counts[key] += session["calls"]
            terminal_bytes[key] += session["h2d_bytes"]
    require(request_counts == terminal_counts and request_bytes == terminal_bytes,
            "request and native generation-keyed phone proofs differ")
    return {
        "calls": calls, "payload_bytes": payload,
        "fallbacks": recursive_counter(receipts, "fallbacks"),
        "resets": recursive_counter(receipts, "reset_recoveries"),
        "maximum_active_slots": max([row.get("maximum_active_slots", 0) for row in receipts] or [0]) or None,
        "maximum_queue_depth": max([row.get("queue_depth", 0) for row in terminals] or [0]),
        "call_count_source": "request-and-generation-keyed-native-terminal-proofs",
    }


def estimator_errors(result: dict[str, Any]) -> dict[str, object]:
    latency = []
    energy = []
    for row in request_rows(result):
        ticket = terminal_ticket(row)
        decision = ticket.get("decision") or {}
        actual_latency = row.get("actual_latency_us")
        start = decision.get("start_us")
        finish = decision.get("finish_us")
        if (
            type(actual_latency) is int
            and type(start) is int
            and type(finish) is int
            and finish > start
        ):
            latency.append(abs(finish - start - actual_latency) / actual_latency)
        route = (decision.get("energy_breakdown") or {}).get("route") or {}
        predicted_energy = route.get("fleet_energy_uj")
        receipt = ticket.get("execution_receipt") or {}
        actual_energy = receipt.get("whole_fleet_energy_uj")
        if (
            type(predicted_energy) is int
            and type(actual_energy) is int
            and actual_energy > 0
        ):
            energy.append(abs(predicted_energy - actual_energy) / actual_energy)
    return {
        "energy_mape_ppm": (
            None if not energy else round(statistics.fmean(energy) * 1_000_000)
        ),
        "energy_request_windows_are_diagnostic": True,
        "latency_mape_ppm": (
            None if not latency else round(statistics.fmean(latency) * 1_000_000)
        ),
        "samples": {"energy": len(energy), "latency": len(latency)},
    }


def summarize(
    baseline_path: Path,
    adaptive_path: Path,
    baseline: dict[str, Any],
    adaptive: dict[str, Any],
    *, expected_replay: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_a = source_manifest_for(baseline_path)
    manifest_b = source_manifest_for(adaptive_path)
    source_identity = validate_matched(
        baseline, adaptive, expected_replay=expected_replay,
        source_manifests=(
            None if manifest_a is None or manifest_b is None
            else (manifest_a, manifest_b)
        ),
    )
    phone_identity = phone_preflight_identity(baseline_path)
    require(phone_identity == phone_preflight_identity(adaptive_path),
            "A/B phone binary or shard identity differs")
    energy_a = energy_domains(baseline)
    energy_b = energy_domains(adaptive)
    total_a = sum(energy_a.values())
    total_b = sum(energy_b.values())
    require(total_a > 0, "baseline fleet energy is zero")
    duration_a = int(baseline["duration_us"])
    duration_b = int(adaptive["duration_us"])
    by_index_a = {
        row["combined_request_index"]: row for row in request_rows(baseline)
    }
    ratios = [
        request["actual_latency_us"]
        / by_index_a[index]["actual_latency_us"]
        for index, request in sorted(
            (row["combined_request_index"], row)
            for row in request_rows(adaptive)
        )
    ]
    timing_us = [
        int(row["total_us"])
        for row in adaptive.get("scheduler_decision_timings") or []
        if type(row) is dict and type(row.get("total_us")) is int
    ]
    windows = load_adaptive_windows(adaptive_path, adaptive)
    fractions: Counter[str] = Counter()
    for window in windows:
        policy = window.get("policy") or {}
        fraction = policy.get("split_fraction_ppm")
        if type(fraction) is int:
            fractions[str(fraction)] += max(
                0,
                window.get("accounting_token_count") or (
                    int(window.get("token_end", 0)) - int(window.get("token_start", 0))
                ),
            )
    transport = transport_summary(adaptive)
    transport["maximum_active_slots"] = max(
        [transport["maximum_active_slots"] or 0]
        + [window.get("maximum_active_slots", 0) for window in windows]
    ) or None
    transport_summary(baseline)
    roles = adaptive["model_roles"]
    llama_phone_only = sum(
        row.get("model_id") == roles["llama"]
        and execution_plan(row).get("route_family") == "whole_model"
        and plan_uses_phone(row)
        for row in request_rows(adaptive)
    )
    placement_a = baseline.get("placement_summary") or {}
    placement_b = adaptive.get("placement_summary") or {}
    saving_ppm = round((1 - total_b / total_a) * 1_000_000)
    maximum_latency_ppm = int(baseline["maximum_latency_ppm"])
    maximum_latency_ratio = max(ratios)
    target_met = (
        saving_ppm >= TARGET_SAVING_PPM
        and maximum_latency_ratio * 1_000_000 <= maximum_latency_ppm
    )
    sessions = session_summary(adaptive)
    comparison = {
        "source_identity": source_identity,
        "adaptive": {
            "duration_us": duration_b,
            "energy_uj_by_domain": energy_b,
            "fleet_energy_uj": total_b,
            "result_sha256": file_sha256(adaptive_path),
            "throughput_tokens_per_s": (
                sum(row["output_tokens"] for row in request_rows(adaptive))
                * 1_000_000
                / duration_b
            ),
        },
        "adaptive_fraction_token_counts": dict(sorted(fractions.items())),
        "adaptive_fraction_token_count_scope": "current-execution-recorded-windows-excluding-unmeasured-tail",
        "baseline": {
            "duration_us": duration_a,
            "energy_uj_by_domain": energy_a,
            "fleet_energy_uj": total_a,
            "result_sha256": file_sha256(baseline_path),
            "throughput_tokens_per_s": (
                sum(row["output_tokens"] for row in request_rows(baseline))
                * 1_000_000
                / duration_a
            ),
        },
        "catalog_sha256": baseline["catalog_sha256"],
        "phone_preflight_identity": phone_identity,
        "comparison_valid": True,
        "request_count": len(request_rows(baseline)),
        "expected_replay_sha256": (None if expected_replay is None else
            "sha256:" + hashlib.sha256(canonical(expected_replay)).hexdigest()),
        "estimator_error": {
            "adaptive": estimator_errors(adaptive),
            "baseline": estimator_errors(baseline),
        },
        "htp_sessions": sessions,
        "htp_sessions_physically_used": {
            session_id: sessions.get(session_id, {}).get("calls", 0) > 0
            for session_id in ("HTP0", "HTP1", "HTP2")
        },
        "latency_ratio": {
            "configured_maximum_ppm": maximum_latency_ppm,
            "maximum": maximum_latency_ratio,
            "mean": statistics.fmean(ratios),
            "p50": percentile(ratios, 0.50),
            "p95": percentile(ratios, 0.95),
        },
        "llama_phone_only_requests": llama_phone_only,
        "placement": {
            "adaptive": route_summary(adaptive),
            "baseline": route_summary(baseline),
        },
        "reloads": {
            "adaptive": int(placement_b.get("model_reload_count", 0)),
            "baseline": int(placement_a.get("model_reload_count", 0)),
        },
        "saving_ppm": saving_ppm,
        "scheduler_timing_us": {
            "maximum": max(timing_us) if timing_us else None,
            "p50": percentile(timing_us, 0.50),
            "p95": percentile(timing_us, 0.95),
        },
        "schema": "burstgpt-matched-ab-comparison-v1",
        "source_manifest_sha256": execution_identity(baseline)[
            "source_manifest_sha256"
        ],
        "target": {
            "energy_saving_ppm": TARGET_SAVING_PPM,
            "latency_ratio_ppm": maximum_latency_ppm,
        },
        "target_met": target_met,
        "transitions": {
            "adaptive": int(placement_b.get("transition_count", 0)),
            "baseline": int(placement_a.get("transition_count", 0)),
        },
        "transport": transport,
    }
    comparison["comparison_sha256"] = "sha256:" + hashlib.sha256(
        canonical(comparison)
    ).hexdigest()
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("adaptive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-schedule", type=Path,
                        help="require this exact named replay instead of the full 84 requests")
    args = parser.parse_args()
    for path in (args.baseline, args.adaptive, *(() if args.replay_schedule is None
                                                else (args.replay_schedule,))):
        require(path.is_file(), "result is absent: " + str(path))
    if args.output is not None:
        require(not args.output.exists(), "comparison output already exists")
    return args


def main() -> int:
    try:
        args = parse_args()
        comparison = summarize(
            args.baseline,
            args.adaptive,
            load_object(args.baseline),
            load_object(args.adaptive),
            expected_replay=(None if args.replay_schedule is None
                             else load_object(args.replay_schedule)),
        )
        payload = canonical(comparison)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(payload)
        print(payload.decode("ascii"), end="")
        return 0
    except (ComparisonError, OSError, ValueError) as error:
        print("comparison blocked: " + str(error), file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
