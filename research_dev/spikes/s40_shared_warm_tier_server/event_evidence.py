#!/usr/bin/env python3
"""Validate and reduce canonical events from the shared warm-tier controller."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import struct
from typing import Any

from evidence_common import (
    EvidenceError,
    canonical_bytes,
    digest_bytes,
    percentile,
    read_jsonl,
    read_json,
    require,
    require_int,
    require_string,
    validate_digest,
)


EVENT_KEYS = {
    "command_id",
    "command_disposition",
    "command_kind",
    "controller_epoch",
    "detail",
    "executor_id",
    "history_sha256",
    "kind",
    "model_id",
    "new_owner",
    "new_ownership_epoch",
    "old_owner",
    "old_ownership_epoch",
    "publication_index",
    "request",
    "request_id",
    "result_publications",
    "result_request_complete",
    "run_id",
    "runtime_config_sha256",
    "schema",
    "schema_version",
    "sequence",
    "state_after",
    "state_before",
    "success",
    "t_ns",
}
RESULT_PUBLICATION_KEYS = {
    "owner_id",
    "ownership_epoch",
    "position",
    "publication_index",
    "token",
}
REQUEST_KEYS = {
    "committed_output_tokens",
    "model_id",
    "owner_id",
    "ownership_epoch",
    "position",
    "prompt_tokens",
    "publication_index",
    "request_id",
    "state",
}
RESOURCE_KEYS = {
    "controller_pid",
    "controller_process_cpu_ticks",
    "controller_process_cpu_utilization_milli_pct",
    "controller_process_rss_bytes",
    "controller_process_start_ticks",
    "controller_process_swap_bytes",
    "cpu_utilization_milli_pct",
    "gpu_memory_free_bytes",
    "gpu_memory_used_bytes",
    "gpu_power_mw",
    "gpu_uuid",
    "host_boot_id",
    "process_metric_scope",
    "run_id",
    "schema",
    "sequence",
    "system_mem_available_bytes",
    "system_swap_free_bytes",
    "system_swap_total_bytes",
    "t_ns",
}

KINDS = {
    "run_start",
    "run_end",
    "request_arrived",
    "request_dispatched",
    "execute_end",
    "token_committed",
    "request_completed",
    "request_stranded",
    "model_state_changed",
    "switch_intent_submitted",
    "switch_intent_queued",
    "switch_intent_coalesced",
    "drain_begin",
    "drain_end",
    "unload_begin",
    "unload_end",
    "load_begin",
    "load_end",
    "replay_begin",
    "replay_end",
    "ownership_commit",
    "ownership_commit_complete",
    "discard_begin",
    "discard_end",
    "cleanup_begin",
    "cleanup_end",
    "executor_failed",
    "resource_sample",
    "phone_telemetry",
}
COMMAND_KINDS = {
    0: ("request_dispatched", "execute_end"),
    1: ("drain_begin", "drain_end"),
    2: ("unload_begin", "unload_end"),
    3: ("load_begin", "load_end"),
    4: ("replay_begin", "replay_end"),
    5: ("discard_begin", "discard_end"),
    6: ("cleanup_begin", "cleanup_end"),
}
COMMAND_EVENT_KINDS = {
    event_kind
    for pair in COMMAND_KINDS.values()
    for event_kind in pair
} | {
    "executor_failed",
    "request_completed",
    "token_committed",
}
MODEL_STATES = {
    "ABSENT",
    "LOADING",
    "READY",
    "DRAINING",
    "REPLAYING",
    "FAILED",
}
REQUEST_STATES = {
    "QUEUED",
    "ACTIVE",
    "COMPLETED",
    "STRANDED",
}
ALLOWED_MODEL_TRANSITIONS = {
    ("ABSENT", "LOADING"),
    ("ABSENT", "READY"),
    ("ABSENT", "ABSENT"),
    ("ABSENT", "FAILED"),
    ("LOADING", "READY"),
    ("LOADING", "REPLAYING"),
    ("LOADING", "FAILED"),
    ("READY", "DRAINING"),
    ("READY", "REPLAYING"),
    ("READY", "FAILED"),
    ("DRAINING", "ABSENT"),
    ("DRAINING", "FAILED"),
    ("REPLAYING", "READY"),
    ("REPLAYING", "FAILED"),
    ("FAILED", "ABSENT"),
    ("FAILED", "LOADING"),
}
REQUEST_KINDS = {
    "request_arrived",
    "request_dispatched",
    "token_committed",
    "request_completed",
    "request_stranded",
    "ownership_commit",
}
PHASES = {
    "drain": ("drain_begin", "drain_end"),
    "unload": ("unload_begin", "unload_end"),
    "load": ("load_begin", "load_end"),
    "replay": ("replay_begin", "replay_end"),
    "cleanup": ("cleanup_begin", "cleanup_end"),
    "discard": ("discard_begin", "discard_end"),
}
TIMELINE_STEP_NS = 1_000_000_000
THROUGHPUT_WINDOW_NS = 5_000_000_000
TRACE_START_KEYS = {
    "active_drain_deadline_ns",
    "campaign_horizon_ns",
    "campaign_horizon_us",
    "created_ns",
    "drain_bound_us",
    "event_log_run_start_ns",
    "experiment_contract_sha256",
    "host_boot_id",
    "requests_sha256",
    "run_id",
    "runtime_config_sha256",
    "schema",
    "trace_origin_ns",
    "trace_start_lead_us",
}


def nullable_string(value: Any, field: str) -> str | None:
    require(
        value is None or isinstance(value, str),
        f"{field}: expected string or null",
    )
    return value


def nullable_int(value: Any, field: str) -> int | None:
    require(
        value is None or (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        ),
        f"{field}: expected nonnegative integer or null",
    )
    return value


def token_history_digest(prompt: list[int], committed: list[int]) -> str:
    raw = bytearray(b"s40-token-history-v1")
    raw.extend(struct.pack("<Q", len(prompt)))
    for token in prompt:
        raw.extend(struct.pack("<I", token & 0xffffffff))
    raw.extend(struct.pack("<Q", len(committed)))
    for token in committed:
        raw.extend(struct.pack("<I", token & 0xffffffff))
    return digest_bytes(bytes(raw))


def validate_tokens(value: Any, field: str) -> list[int]:
    require(
        isinstance(value, list)
        and all(isinstance(token, int) and not isinstance(token, bool)
                and 0 <= token < (1 << 31) for token in value),
        f"{field}: expected nonnegative int32 token array",
    )
    return value


def validate_request_snapshot(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field}: expected object")
    require(set(value) == REQUEST_KEYS, f"{field}: key set mismatch")
    require_string(value["request_id"], f"{field}.request_id")
    require_string(value["model_id"], f"{field}.model_id")
    validate_tokens(value["prompt_tokens"], f"{field}.prompt_tokens")
    validate_tokens(
        value["committed_output_tokens"],
        f"{field}.committed_output_tokens",
    )
    require_int(value["position"], f"{field}.position")
    nullable_string(value["owner_id"], f"{field}.owner_id")
    require_int(value["ownership_epoch"], f"{field}.ownership_epoch")
    require_int(value["publication_index"], f"{field}.publication_index")
    require(value["state"] in REQUEST_STATES, f"{field}: invalid state")
    require(
        value["publication_index"] == len(value["committed_output_tokens"]),
        f"{field}: publication count mismatch",
    )
    require(
        value["position"]
        == len(value["prompt_tokens"]) + len(value["committed_output_tokens"]),
        f"{field}: position mismatch",
    )
    return value


def validate_event_shape(
        event: dict[str, Any],
        index: int,
        run_id: str,
        previous_t_ns: int,
        previous_epoch: int) -> tuple[int, int]:
    field = f"event[{index}]"
    require(set(event) == EVENT_KEYS, f"{field}: key set mismatch")
    require(
        event["schema"] == "s40-warm-tier-event-v3"
        and event["schema_version"] == 3,
        f"{field}: schema mismatch",
    )
    require(event["run_id"] == run_id, f"{field}: run ID mismatch")
    validate_digest(
        event["runtime_config_sha256"],
        f"{field}.runtime_config_sha256",
    )
    require_int(event["sequence"], f"{field}.sequence")
    require(event["sequence"] == index, f"{field}: sequence mismatch")
    t_ns = require_int(event["t_ns"], f"{field}.t_ns")
    require(t_ns >= previous_t_ns, f"{field}: nonmonotonic time")
    epoch = require_int(event["controller_epoch"], f"{field}.controller_epoch")
    require(epoch >= previous_epoch, f"{field}: stale controller epoch")
    require(event["kind"] in KINDS, f"{field}: invalid kind")
    command_id = nullable_int(event["command_id"], f"{field}.command_id")
    command_kind = nullable_int(
        event["command_kind"], f"{field}.command_kind")
    disposition = nullable_string(
        event["command_disposition"],
        f"{field}.command_disposition",
    )
    if event["kind"] in COMMAND_EVENT_KINDS:
        require(
            command_id is not None
            and command_id > 0
            and command_kind in COMMAND_KINDS,
            f"{field}: missing or invalid command identity",
        )
    else:
        require(
            command_id is None and command_kind is None,
            f"{field}: unexpected command identity",
        )
    result_publications = event["result_publications"]
    require(
        isinstance(result_publications, list),
        f"{field}.result_publications: expected array",
    )
    for result_index, publication in enumerate(result_publications):
        result_field = f"{field}.result_publications[{result_index}]"
        require(
            isinstance(publication, dict)
            and set(publication) == RESULT_PUBLICATION_KEYS,
            f"{result_field}: key set mismatch",
        )
        owner_id = require_string(
            publication["owner_id"], f"{result_field}.owner_id")
        require(
            owner_id == event["executor_id"],
            f"{result_field}: executor ownership mismatch",
        )
        require_int(
            publication["ownership_epoch"],
            f"{result_field}.ownership_epoch",
            1,
        )
        require_int(publication["position"], f"{result_field}.position")
        require_int(
            publication["publication_index"],
            f"{result_field}.publication_index",
        )
        require_int(publication["token"], f"{result_field}.token")
    require(
        isinstance(event["result_request_complete"], bool),
        f"{field}.result_request_complete: expected bool",
    )
    command_end_kinds = {
        end_kind for _, end_kind in COMMAND_KINDS.values()
    }
    if event["kind"] in command_end_kinds:
        require(
            disposition in {"RECEIVED", "QUARANTINED"},
            f"{field}: invalid command disposition",
        )
        if command_kind != 0:
            require(
                not result_publications
                and event["result_request_complete"] is False,
                f"{field}: lifecycle command returned publications",
            )
    else:
        require(
            disposition is None
            and not result_publications
            and event["result_request_complete"] is False,
            f"{field}: unexpected raw command result",
        )
    for name in (
        "model_id",
        "request_id",
        "executor_id",
        "old_owner",
        "new_owner",
        "history_sha256",
    ):
        nullable_string(event[name], f"{field}.{name}")
    for name in (
        "old_ownership_epoch",
        "new_ownership_epoch",
        "publication_index",
    ):
        nullable_int(event[name], f"{field}.{name}")
    require(isinstance(event["success"], bool), f"{field}.success: expected bool")
    require(isinstance(event["detail"], str), f"{field}.detail: expected string")
    if event["kind"] == "model_state_changed":
        require(
            event["state_before"] in MODEL_STATES
            and event["state_after"] in MODEL_STATES,
            f"{field}: invalid model state",
        )
    else:
        require(
            event["state_before"] is None and event["state_after"] is None,
            f"{field}: unexpected model state",
        )
    if event["request"] is not None:
        snapshot = validate_request_snapshot(event["request"], f"{field}.request")
        require(
            event["request_id"] == snapshot["request_id"]
            and event["model_id"] == snapshot["model_id"],
            f"{field}: request envelope mismatch",
        )
        expected_digest = token_history_digest(
            snapshot["prompt_tokens"],
            snapshot["committed_output_tokens"],
        )
        require(
            event["history_sha256"] == expected_digest,
            f"{field}: request history digest mismatch",
        )
    elif event["kind"] in REQUEST_KINDS:
        require(False, f"{field}: missing request snapshot")
    else:
        require(
            event["history_sha256"] is None,
            f"{field}: unexpected history digest",
        )
    if event["history_sha256"] is not None:
        validate_digest(event["history_sha256"], f"{field}.history_sha256")
    return t_ns, epoch


def extract_command_ledger(
        events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    commands: dict[int, dict[str, Any]] = {}
    failed_events: set[int] = set()
    for index, event in enumerate(events):
        kind = event["kind"]
        command_id = event["command_id"]
        command_kind = event["command_kind"]
        if command_id is None:
            continue
        field = f"event[{index}]"
        begin_kind, end_kind = COMMAND_KINDS[command_kind]
        identity = {
            "executor_id": event["executor_id"],
            "model_id": event["model_id"],
            "request_id": event["request_id"],
        }
        require(
            identity["executor_id"] is not None
            and identity["model_id"] is not None,
            f"{field}: incomplete command identity",
        )
        if kind == begin_kind:
            require(
                command_id not in commands,
                f"{field}: duplicate command begin",
            )
            commands[command_id] = {
                **identity,
                "begin_sequence": event["sequence"],
                "command_id": command_id,
                "command_kind": command_kind,
                "controller_epoch": event["controller_epoch"],
                "end_sequence": None,
                "committed_publications": [],
                "disposition": None,
                "publications": None,
                "request_complete": None,
                "success": None,
            }
            continue
        command = commands.get(command_id)
        require(command is not None, f"{field}: command has no begin")
        require(
            command["command_kind"] == command_kind
            and all(command[name] == value for name, value in identity.items()),
            f"{field}: command identity changed",
        )
        require(
            event["controller_epoch"] >= command["controller_epoch"],
            f"{field}: command event precedes issue epoch",
        )
        if kind == end_kind:
            require(
                command["end_sequence"] is None,
                f"{field}: duplicate command end",
            )
            command["end_sequence"] = event["sequence"]
            command["disposition"] = event["command_disposition"]
            command["publications"] = event["result_publications"]
            command["request_complete"] = event["result_request_complete"]
            command["success"] = event["success"]
        elif kind == "token_committed":
            require(
                command_kind == 0
                and command["end_sequence"] is not None
                and command["disposition"] == "RECEIVED"
                and command["success"] is True,
                f"{field}: publication is not from successful EXECUTE",
            )
            snapshot = event["request"]
            publication_index = event["publication_index"]
            require(
                publication_index is not None
                and publication_index < len(
                    snapshot["committed_output_tokens"]),
                f"{field}: publication index is outside history",
            )
            command["committed_publications"].append({
                "owner_id": snapshot["owner_id"],
                "ownership_epoch": snapshot["ownership_epoch"],
                "position": snapshot["position"] - 1,
                "publication_index": publication_index,
                "token": snapshot["committed_output_tokens"][
                    publication_index],
            })
        elif kind == "request_completed":
            require(
                command_kind == 0
                and command["end_sequence"] is not None
                and command["disposition"] == "RECEIVED"
                and command["success"] is True,
                f"{field}: completion is not from successful EXECUTE",
            )
            require(
                command["request_complete"] is True,
                f"{field}: completion missing from raw result",
            )
            require(
                not command.get("completion_seen", False),
                f"{field}: duplicate completion",
            )
            command["completion_seen"] = True
        elif kind == "executor_failed":
            require(
                command["end_sequence"] is not None
                and command["success"] is False
                and command_id not in failed_events,
                f"{field}: invalid executor failure linkage",
            )
            failed_events.add(command_id)
        else:
            require(False, f"{field}: command event kind mismatch")

    for command_id, command in commands.items():
        require(
            command["end_sequence"] is not None,
            f"command[{command_id}]: missing end",
        )
        require(
            command["disposition"] in {"RECEIVED", "QUARANTINED"},
            f"command[{command_id}]: missing disposition",
        )
        if command["disposition"] == "QUARANTINED":
            require(
                not command["committed_publications"]
                and not command.get("completion_seen", False),
                f"command[{command_id}]: quarantined result was committed",
            )
        elif command["command_kind"] == 0 and command["success"] is True:
            require(
                command["committed_publications"] == command["publications"],
                f"command[{command_id}]: committed/raw publication mismatch",
            )
            require(
                command.get("completion_seen", False)
                is command["request_complete"],
                f"command[{command_id}]: committed/raw completion mismatch",
            )
        else:
            require(
                not command["committed_publications"]
                and not command.get("completion_seen", False),
                f"command[{command_id}]: failed/lifecycle command committed output",
            )
    return [
        {
            "command_id": command["command_id"],
            "command_kind": command["command_kind"],
            "controller_epoch": command["controller_epoch"],
            "disposition": command["disposition"],
            "executor_id": command["executor_id"],
            "model_id": command["model_id"],
            "publications": command["publications"],
            "request_id": command["request_id"],
            "request_complete": command["request_complete"],
            "success": command["success"],
        }
        for _, command in sorted(commands.items())
    ]


def validate_resources(
        rows: list[dict[str, Any]],
        run_id: str,
        start_ns: int,
        end_ns: int) -> dict[str, Any]:
    require(bool(rows), "resources: empty")
    previous_t = -1
    previous_sequence = -1
    gpu_uuid: str | None = None
    host_boot_id: str | None = None
    controller_pid: int | None = None
    controller_start_ticks: int | None = None
    previous_process_ticks = -1
    for index, row in enumerate(rows):
        field = f"resource[{index}]"
        require(set(row) == RESOURCE_KEYS, f"{field}: key set mismatch")
        require(
            row["schema"] == "s40-selected-gpu-resource-v3"
            and row["run_id"] == run_id,
            f"{field}: identity mismatch",
        )
        sequence = require_int(row["sequence"], f"{field}.sequence")
        require(sequence == previous_sequence + 1, f"{field}: sequence mismatch")
        previous_sequence = sequence
        t_ns = require_int(row["t_ns"], f"{field}.t_ns")
        require(t_ns > previous_t, f"{field}: nonmonotonic time")
        previous_t = t_ns
        current_uuid = require_string(row["gpu_uuid"], f"{field}.gpu_uuid")
        if gpu_uuid is None:
            gpu_uuid = current_uuid
        require(current_uuid == gpu_uuid, f"{field}: GPU identity changed")
        current_boot_id = require_string(
            row["host_boot_id"], f"{field}.host_boot_id")
        if host_boot_id is None:
            host_boot_id = current_boot_id
        require(current_boot_id == host_boot_id,
                f"{field}: host boot identity changed")
        current_pid = require_int(
            row["controller_pid"], f"{field}.controller_pid", 2)
        if controller_pid is None:
            controller_pid = current_pid
        require(
            current_pid == controller_pid,
            f"{field}: controller PID changed",
        )
        current_start_ticks = require_int(
            row["controller_process_start_ticks"],
            f"{field}.controller_process_start_ticks",
            1,
        )
        if controller_start_ticks is None:
            controller_start_ticks = current_start_ticks
        require(
            current_start_ticks == controller_start_ticks,
            f"{field}: controller process identity changed",
        )
        current_process_ticks = require_int(
            row["controller_process_cpu_ticks"],
            f"{field}.controller_process_cpu_ticks",
        )
        require(
            current_process_ticks >= previous_process_ticks,
            f"{field}: process CPU counter regressed",
        )
        previous_process_ticks = current_process_ticks
        for name in RESOURCE_KEYS - {
                "schema", "run_id", "gpu_uuid", "host_boot_id",
                "process_metric_scope"}:
            if name not in {"sequence", "t_ns"}:
                require_int(row[name], f"{field}.{name}")
        require(
            row["process_metric_scope"] == "CONTROLLER_PROCESS_ONLY",
            f"{field}: process metric scope mismatch",
        )
        require(
            row["cpu_utilization_milli_pct"] <= 100_000
            and row[
                "controller_process_cpu_utilization_milli_pct"
            ] <= 100_000,
            f"{field}: utilization exceeds host capacity",
        )
        require(
            row["system_swap_free_bytes"] <= row["system_swap_total_bytes"],
            f"{field}: invalid swap counters",
        )
    require(rows[0]["t_ns"] <= start_ns, "resources: missing start bracket")
    require(rows[-1]["t_ns"] >= end_ns, "resources: missing end bracket")
    for left, right in zip(rows, rows[1:]):
        require(
            right["t_ns"] - left["t_ns"] <= 500_000_000,
            "resources: sample gap exceeds 500 ms",
        )

    energy_nj = 0
    for left, right in zip(rows, rows[1:]):
        interval_start = max(start_ns, left["t_ns"])
        interval_end = min(end_ns, right["t_ns"])
        if interval_end > interval_start:
            energy_nj += (
                left["gpu_power_mw"] * (interval_end - interval_start)
                // 1_000
            )
    swap_used = [
        row["system_swap_total_bytes"] - row["system_swap_free_bytes"]
        for row in rows
    ]
    points = [start_ns]
    points.extend(
        row["t_ns"] for row in rows
        if start_ns < row["t_ns"] < end_ns)
    points.append(end_ns)
    resource_index = max(
        index for index, row in enumerate(rows)
        if row["t_ns"] <= start_ns)
    energy_timeline = [{
        "cumulative_gpu_energy_nj": 0,
        "gpu_power_mw": rows[resource_index]["gpu_power_mw"],
        "t_offset_ns": 0,
    }]
    cumulative_nj = 0
    previous_point = start_ns
    for point in points[1:]:
        cumulative_nj += (
            rows[resource_index]["gpu_power_mw"]
            * (point - previous_point)
            // 1_000
        )
        while resource_index + 1 < len(rows) \
                and rows[resource_index + 1]["t_ns"] <= point:
            resource_index += 1
        energy_timeline.append({
            "cumulative_gpu_energy_nj": cumulative_nj,
            "gpu_power_mw": rows[resource_index]["gpu_power_mw"],
            "t_offset_ns": point - start_ns,
        })
        previous_point = point
    require(
        cumulative_nj == energy_nj,
        "resources: timeline energy mismatch",
    )
    return {
        "cpu_utilization_milli_pct_p50": percentile(
            [row["cpu_utilization_milli_pct"] for row in rows], 1, 2),
        "gpu_energy_nj": energy_nj,
        "energy_claim_authorized": False,
        "gpu_energy_scope": "SELECTED_GPU_BOARD_DEVELOPMENT_ONLY",
        "gpu_uuid": gpu_uuid,
        "gpu_power_update_count": sum(
            left["gpu_power_mw"] != right["gpu_power_mw"]
            for left, right in zip(rows, rows[1:])
        ),
        "host_boot_id": host_boot_id,
        "timeline": energy_timeline,
        "peak_gpu_memory_used_bytes": max(
            row["gpu_memory_used_bytes"] for row in rows),
        "controller_process_metric_scope": "CONTROLLER_PROCESS_ONLY",
        "peak_controller_process_rss_bytes": max(
            row["controller_process_rss_bytes"] for row in rows),
        "minimum_system_mem_available_bytes": min(
            row["system_mem_available_bytes"] for row in rows),
        "controller_process_swap_growth_bytes": (
            max(row["controller_process_swap_bytes"] for row in rows)
            - rows[0]["controller_process_swap_bytes"]
        ),
        "controller_process_cpu_utilization_milli_pct_p50": percentile(
            [
                row["controller_process_cpu_utilization_milli_pct"]
                for row in rows
            ],
            1,
            2,
        ),
        "controller_pid": controller_pid,
        "controller_start_ticks": controller_start_ticks,
        "system_swap_growth_bytes": max(swap_used) - swap_used[0],
    }


def build_throughput_timeline(
        run_start_ns: int,
        run_end_ns: int,
        model_ids: list[str],
        token_events: list[tuple[int, str]]) -> list[dict[str, Any]]:
    offsets = list(range(
        0,
        run_end_ns - run_start_ns + 1,
        TIMELINE_STEP_NS,
    ))
    final_offset = run_end_ns - run_start_ns
    if offsets[-1] != final_offset:
        offsets.append(final_offset)
    rows = []
    for offset in offsets:
        point = run_start_ns + offset
        window_start = max(run_start_ns, point - THROUGHPUT_WINDOW_NS)
        denominator_ns = point - window_start
        counts = Counter(
            model_id for t_ns, model_id in token_events
            if window_start < t_ns <= point)
        rows.append({
            "model_milli_tokens_per_second": {
                model_id: (
                    counts[model_id] * 1_000_000_000_000
                    // denominator_ns
                    if denominator_ns > 0 else 0
                )
                for model_id in model_ids
            },
            "t_offset_ns": offset,
        })
    return rows


def validate_trace_start(
        value: Any,
        *,
        run_id: str,
        runtime_config_sha256: str,
        event_log_run_start_ns: int) -> dict[str, Any]:
    require(isinstance(value, dict), "trace_start: expected object")
    require(set(value) == TRACE_START_KEYS,
            "trace_start: key set mismatch")
    require(value["schema"] == "s40-trace-start-v1",
            "trace_start: schema mismatch")
    require(value["run_id"] == run_id, "trace_start: run ID mismatch")
    require(
        value["runtime_config_sha256"] == runtime_config_sha256,
        "trace_start: runtime config mismatch",
    )
    validate_digest(
        value["experiment_contract_sha256"],
        "trace_start.experiment_contract_sha256",
    )
    validate_digest(
        value["requests_sha256"], "trace_start.requests_sha256")
    require_string(value["host_boot_id"], "trace_start.host_boot_id")
    run_start_ns = require_int(
        value["event_log_run_start_ns"],
        "trace_start.event_log_run_start_ns",
    )
    require(run_start_ns == event_log_run_start_ns,
            "trace_start: RUN_START timestamp mismatch")
    created_ns = require_int(value["created_ns"], "trace_start.created_ns")
    origin_ns = require_int(
        value["trace_origin_ns"], "trace_start.trace_origin_ns")
    lead_us = require_int(
        value["trace_start_lead_us"], "trace_start.trace_start_lead_us", 1)
    require(origin_ns == created_ns + lead_us * 1000,
            "trace_start: lead arithmetic mismatch")
    require(run_start_ns <= created_ns < origin_ns,
            "trace_start: invalid run/start ordering")
    campaign_us = require_int(
        value["campaign_horizon_us"],
        "trace_start.campaign_horizon_us",
        1,
    )
    drain_us = require_int(
        value["drain_bound_us"], "trace_start.drain_bound_us", 1)
    campaign_ns = require_int(
        value["campaign_horizon_ns"],
        "trace_start.campaign_horizon_ns",
    )
    drain_deadline_ns = require_int(
        value["active_drain_deadline_ns"],
        "trace_start.active_drain_deadline_ns",
    )
    require(campaign_ns == origin_ns + campaign_us * 1000,
            "trace_start: campaign horizon arithmetic mismatch")
    require(
        drain_deadline_ns == campaign_ns + drain_us * 1000,
        "trace_start: active-drain arithmetic mismatch",
    )
    return value


def reduce_events(
        events: list[dict[str, Any]],
        expected_requests: list[dict[str, Any]],
        resources: list[dict[str, Any]] | None = None,
        expected_runtime_config_sha256: str | None = None,
        trace_start: dict[str, Any] | None = None) -> dict[str, Any]:
    require(bool(events), "events: empty")
    run_id = require_string(events[0].get("run_id"), "events.run_id")
    runtime_config_sha256 = validate_digest(
        events[0].get("runtime_config_sha256"),
        "events.runtime_config_sha256",
    )
    if expected_runtime_config_sha256 is not None:
        require(
            runtime_config_sha256
            == validate_digest(
                expected_runtime_config_sha256,
                "expected_runtime_config_sha256",
            ),
            "events: runtime config digest mismatch",
        )
    previous_t = -1
    previous_epoch = 0
    for index, event in enumerate(events):
        require(
            event.get("runtime_config_sha256") == runtime_config_sha256,
            f"event[{index}]: runtime config digest changed",
        )
        previous_t, previous_epoch = validate_event_shape(
            event, index, run_id, previous_t, previous_epoch)
    require(
        events[0]["kind"] == "run_start"
        and events[-1]["kind"] == "run_end",
        "events: missing run bounds",
    )
    require(
        sum(event["kind"] == "run_start" for event in events) == 1
        and sum(event["kind"] == "run_end" for event in events) == 1,
        "events: duplicate run bounds",
    )

    expected = {row["event_id"]: row for row in expected_requests}
    require(len(expected) == len(expected_requests), "expected requests: duplicate")
    request_state: dict[str, dict[str, Any]] = {}
    model_state: dict[tuple[str, str], str] = {}
    phase_begin: dict[
        tuple[str, int, str, str, str],
        dict[str, Any],
    ] = {}
    phase_durations: dict[str, list[int]] = {
        name: [] for name in PHASES
    }
    failed_executors: set[str] = set()
    cleanup_failed = False
    executor_failed = False
    transition_intents: dict[int, dict[str, Any]] = {}
    queued_switch_intent_count = 0
    model_ready_times: dict[int, int] = {}
    ownership_commit_times: dict[int, list[int]] = {}
    ownership_commit_requests: dict[int, set[str]] = {}
    ownership_commit_complete: dict[int, dict[str, Any]] = {}
    pending_commit_epoch: int | None = None
    rollback_restores: list[dict[str, Any]] = []
    target_failures: dict[int, list[dict[str, Any]]] = {}
    discard_times: dict[int, int] = {}
    unexplained_failed_events: list[str] = []

    arrivals: dict[str, int] = {}
    arrival_order: list[str] = []
    dispatches: dict[str, int] = {}
    first_tokens: dict[str, int] = {}
    completions: dict[str, int] = {}
    stranded: dict[str, int] = {}
    token_times: dict[str, list[int]] = {}
    token_events: list[tuple[int, str]] = []

    for event in events:
        kind = event["kind"]
        request_id = event["request_id"]
        executor_id = event["executor_id"]
        epoch = event["controller_epoch"]
        if pending_commit_epoch is not None:
            require(
                kind in {
                    "ownership_commit",
                    "ownership_commit_complete",
                } and epoch == pending_commit_epoch,
                "ownership: event interleaved before commit complete",
            )
        if not event["success"] \
                and kind not in {
                    "request_stranded",
                    "cleanup_end",
                    "executor_failed",
                }:
            unexplained_failed_events.append(kind)
        if kind == "model_state_changed":
            require(event["model_id"] is not None and executor_id is not None,
                    "model state: missing identity")
            transition = (event["state_before"], event["state_after"])
            if transition == ("DRAINING", "READY"):
                rollback_restores.append({
                    "epoch": epoch,
                    "executor_id": executor_id,
                    "model_id": event["model_id"],
                    "t_ns": event["t_ns"],
                })
            else:
                require(transition in ALLOWED_MODEL_TRANSITIONS,
                        "model state: invalid transition")
            key = (event["model_id"], executor_id)
            if key in model_state:
                require(model_state[key] == event["state_before"],
                        "model state: broken chain")
            model_state[key] = event["state_after"]
            intent = transition_intents.get(epoch)
            if intent is not None \
                    and event["state_after"] == "READY" \
                    and event["model_id"] == intent["model_id"] \
                    and executor_id == intent["executor_id"]:
                model_ready_times[epoch] = event["t_ns"]
            if event["state_after"] == "FAILED":
                target_failures.setdefault(epoch, []).append({
                    "executor_id": executor_id,
                    "model_id": event["model_id"],
                    "t_ns": event["t_ns"],
                })

        if kind == "switch_intent_submitted":
            require(epoch not in transition_intents,
                    "switch intent: duplicate epoch")
            require(
                event["model_id"] is not None
                and event["executor_id"] is not None,
                "switch intent: missing target identity",
            )
            transition_intents[epoch] = {
                "executor_id": event["executor_id"],
                "model_id": event["model_id"],
                "t_ns": event["t_ns"],
            }
        elif kind in {"switch_intent_queued", "switch_intent_coalesced"}:
            require(
                event["model_id"] is not None
                and event["executor_id"] is not None,
                "switch queue/coalesce: missing replacement identity",
            )
            if kind == "switch_intent_queued":
                queued_switch_intent_count += 1
        elif kind == "ownership_commit_complete":
            intent = transition_intents.get(epoch)
            require(intent is not None,
                    "ownership complete: missing switch intent")
            require(
                event["model_id"] == intent["model_id"]
                and executor_id == intent["executor_id"],
                "ownership complete: target identity mismatch",
            )
            require(epoch not in ownership_commit_complete,
                    "ownership complete: duplicate epoch")
            require(
                epoch in model_ready_times,
                "ownership complete: target is not ready",
            )
            require(
                bool(event["detail"])
                and event["detail"].isascii()
                and event["detail"].isdigit()
                and str(int(event["detail"])) == event["detail"],
                "ownership complete: invalid commit count",
            )
            count = int(event["detail"])
            require(count > 0, "ownership complete: empty commit set")
            require(
                len(ownership_commit_times.get(epoch, [])) == count,
                "ownership complete: commit count mismatch",
            )
            ownership_commit_complete[epoch] = {
                "count": count,
                "t_ns": event["t_ns"],
            }
            require(
                pending_commit_epoch == epoch,
                "ownership complete: no pending commit group",
            )
            pending_commit_epoch = None

        for phase_name, (begin_kind, end_kind) in PHASES.items():
            key = (
                phase_name,
                epoch,
                event["model_id"] or "",
                event["request_id"] or "",
                executor_id or "",
            )
            if kind == begin_kind:
                if phase_name == "cleanup":
                    require(
                        request_id is not None
                        and epoch in ownership_commit_complete,
                        "cleanup: ownership commit is not complete",
                    )
                elif phase_name == "discard":
                    require(
                        request_id is None,
                        "discard: unexpected request identity",
                    )
                    require(epoch not in discard_times,
                            "discard: duplicate epoch")
                    discard_times[epoch] = event["t_ns"]
                require(key not in phase_begin, f"{phase_name}: duplicate begin")
                phase_begin[key] = {
                    "request": event["request"],
                    "t_ns": event["t_ns"],
                }
            elif kind == end_kind:
                require(key in phase_begin, f"{phase_name}: missing begin")
                begin = phase_begin.pop(key)
                require(
                    event["request"] == begin["request"],
                    f"{phase_name}: command frontier changed",
                )
                phase_durations[phase_name].append(
                    event["t_ns"] - begin["t_ns"])
                if phase_name == "cleanup" and not event["success"]:
                    cleanup_failed = True

        if kind == "executor_failed":
            require(executor_id is not None, "executor failure: missing executor")
            failed_executors.add(executor_id)
            executor_failed = True

        phase_end_kinds = {pair[1] for pair in PHASES.values()}
        if event["request"] is not None \
                and kind not in REQUEST_KINDS \
                and kind not in phase_end_kinds:
            require(request_id is not None,
                    "request phase: missing request ID")
            current = request_state.get(request_id)
            require(current is not None,
                    "request phase: missing prior request state")
            snapshot = event["request"]
            require(
                snapshot["prompt_tokens"] == current["prompt_tokens"]
                and snapshot["committed_output_tokens"]
                == current["committed_output_tokens"]
                and snapshot["owner_id"] == current["owner_id"]
                and snapshot["ownership_epoch"]
                == current["ownership_epoch"],
                "request phase: frontier mismatch",
            )
        if kind not in REQUEST_KINDS:
            continue
        require(request_id is not None, "request event: missing request ID")
        snapshot = event["request"]
        expected_row = expected.get(request_id)
        require(expected_row is not None, "request event: unknown request")
        require(
            snapshot["model_id"] == expected_row["model_id"]
            and snapshot["prompt_tokens"] == expected_row["prompt_tokens"],
            "request event: frozen input mismatch",
        )

        previous = request_state.get(request_id)
        if previous is None:
            require(kind == "request_arrived", "request: first event not arrival")
            require(
                snapshot["state"] == "QUEUED"
                and not snapshot["committed_output_tokens"]
                and snapshot["publication_index"] == 0,
                "request arrival: invalid initial state",
            )
            require(
                snapshot["owner_id"] is None
                and snapshot["ownership_epoch"] == 0,
                "request arrival: unexpected owner",
            )
            arrivals[request_id] = event["t_ns"]
            arrival_order.append(request_id)
            token_times[request_id] = []
        else:
            require(
                snapshot["prompt_tokens"] == previous["prompt_tokens"],
                "request: prompt mutated",
            )
            old_committed = previous["committed_output_tokens"]
            new_committed = snapshot["committed_output_tokens"]
            require(
                new_committed[:len(old_committed)] == old_committed,
                "request: committed history changed",
            )
            if kind not in {"ownership_commit", "request_dispatched"}:
                require(
                    snapshot["owner_id"] == previous["owner_id"]
                    and snapshot["ownership_epoch"]
                    == previous["ownership_epoch"],
                    "request: owner changed without atomic commit",
                )

        if kind == "request_dispatched":
            require(previous is not None, "dispatch: missing request state")
            if previous["state"] == "QUEUED":
                require(
                    request_id not in dispatches
                    and previous["owner_id"] is None
                    and previous["ownership_epoch"] == 0
                    and snapshot["state"] == "ACTIVE"
                    and snapshot["ownership_epoch"] == 1,
                    "dispatch: invalid initial owner assignment",
                )
                dispatches[request_id] = event["t_ns"]
            else:
                require(
                    previous["state"] == "ACTIVE"
                    and snapshot == previous,
                    "dispatch: active frontier changed",
                )
            require(executor_id == snapshot["owner_id"],
                    "dispatch: executor is not owner")
        elif kind == "token_committed":
            require(previous is not None, "token: missing previous request state")
            require(
                snapshot["state"] == "ACTIVE"
                and len(snapshot["committed_output_tokens"])
                == len(previous["committed_output_tokens"]) + 1
                and snapshot["publication_index"]
                == previous["publication_index"] + 1
                and event["publication_index"]
                == snapshot["publication_index"] - 1,
                "token: duplicate or skipped publication",
            )
            require(executor_id == snapshot["owner_id"],
                    "token: executor is not owner")
            if request_id not in first_tokens:
                first_tokens[request_id] = event["t_ns"]
            token_times[request_id].append(event["t_ns"])
            token_events.append((event["t_ns"], snapshot["model_id"]))
        elif kind == "ownership_commit":
            require(previous is not None, "ownership: missing previous state")
            intent = transition_intents.get(epoch)
            require(
                intent is not None
                and event["model_id"] == intent["model_id"]
                and executor_id == intent["executor_id"],
                "ownership: target identity mismatch",
            )
            require(
                epoch not in ownership_commit_complete,
                "ownership: commit after complete",
            )
            if pending_commit_epoch is None:
                pending_commit_epoch = epoch
            require(
                pending_commit_epoch == epoch,
                "ownership: overlapping commit groups",
            )
            committed_requests = ownership_commit_requests.setdefault(
                epoch, set())
            require(
                request_id not in committed_requests,
                "ownership: request committed twice",
            )
            committed_requests.add(request_id)
            require(
                event["old_owner"] == previous["owner_id"]
                and event["old_ownership_epoch"]
                == previous["ownership_epoch"]
                and event["new_owner"] == snapshot["owner_id"]
                and event["new_ownership_epoch"]
                == snapshot["ownership_epoch"]
                and snapshot["ownership_epoch"]
                == previous["ownership_epoch"] + 1,
                "ownership: stale or skipped epoch",
            )
            require(
                event["publication_index"] == snapshot["publication_index"],
                "ownership: publication frontier mismatch",
            )
            require(
                snapshot["committed_output_tokens"]
                == previous["committed_output_tokens"],
                "ownership: history changed during commit",
            )
            require(
                event["new_owner"] not in failed_executors,
                "ownership: target executor already failed",
            )
            expected_digest = token_history_digest(
                snapshot["prompt_tokens"],
                snapshot["committed_output_tokens"],
            )
            require(
                event["history_sha256"] == expected_digest,
                "ownership: history digest mismatch",
            )
            ownership_commit_times.setdefault(epoch, []).append(event["t_ns"])
        elif kind == "request_completed":
            require(request_id not in completions and request_id not in stranded,
                    "request: duplicate terminal")
            require(
                snapshot["state"] == "COMPLETED"
                and previous is not None
                and snapshot["committed_output_tokens"]
                == previous["committed_output_tokens"]
                and len(snapshot["committed_output_tokens"])
                == expected_row["output_tokens"],
                "completion: token count or publication mismatch",
            )
            completions[request_id] = event["t_ns"]
        elif kind == "request_stranded":
            require(request_id not in completions and request_id not in stranded,
                    "request: duplicate terminal")
            require(
                snapshot["state"] == "STRANDED"
                and previous is not None
                and snapshot["committed_output_tokens"]
                == previous["committed_output_tokens"],
                "stranded: invalid state or publication",
            )
            stranded[request_id] = event["t_ns"]
        request_state[request_id] = snapshot

    require(not phase_begin, "events: unterminated phase")
    require(
        set(ownership_commit_times) == set(ownership_commit_complete),
        "ownership: missing or orphan commit-complete marker",
    )
    require(pending_commit_epoch is None,
            "ownership: pending commit group at run end")
    for restore in rollback_restores:
        epoch = restore["epoch"]
        intent = transition_intents.get(epoch)
        require(intent is not None,
                "rollback restore: missing switch intent")
        failures = target_failures.get(epoch, [])
        matching_failure = [
            failure for failure in failures
            if failure["model_id"] == intent["model_id"]
            and failure["executor_id"] == intent["executor_id"]
            and failure["t_ns"] >= restore["t_ns"]
        ]
        require(
            restore["model_id"] == intent["model_id"]
            and restore["executor_id"] != intent["executor_id"]
            and bool(matching_failure)
            and epoch in discard_times
            and any(
                failure["t_ns"] <= discard_times[epoch]
                for failure in matching_failure
            )
            and epoch not in ownership_commit_times
            and epoch not in ownership_commit_complete,
            "rollback restore: not a failed precommit epoch",
        )
    require(set(arrivals) == set(expected), "requests: arrival conservation")
    require(
        arrival_order == [row["event_id"] for row in expected_requests],
        "requests: frozen arrival order mismatch",
    )
    require(
        set(completions) | set(stranded) == set(expected)
        and not (set(completions) & set(stranded)),
        "requests: terminal conservation",
    )
    command_ledger = extract_command_ledger(events)
    controller_run_start_ns = events[0]["t_ns"]
    run_end_ns = events[-1]["t_ns"]
    run_start_ns = controller_run_start_ns
    if trace_start is not None:
        validate_trace_start(
            trace_start,
            run_id=run_id,
            runtime_config_sha256=runtime_config_sha256,
            event_log_run_start_ns=controller_run_start_ns,
        )
        run_start_ns = trace_start["trace_origin_ns"]
        require(
            run_end_ns <= trace_start["active_drain_deadline_ns"],
            "events: run exceeds active-drain deadline",
        )
        reason = events[-1]["detail"]
        require(
            reason in {"TRACE_COMPLETE", "HORIZON_REACHED"},
            "events: invalid physical finalization reason",
        )
        if reason == "HORIZON_REACHED":
            require(
                run_end_ns >= trace_start["campaign_horizon_ns"],
                "events: horizon finalization precedes horizon",
            )
    require(run_end_ns > run_start_ns, "events: empty measured run")
    request_metrics: list[dict[str, Any]] = []
    intra_request_publication_gaps: list[int] = []
    for request_id, row in expected.items():
        scheduled_ns = run_start_ns + row["arrival_us"] * 1000
        require(arrivals[request_id] >= scheduled_ns,
                "request: arrival precedes frozen schedule")
        if request_id in completions:
            require(
                request_id in dispatches and request_id in first_tokens,
                "completed request: missing dispatch or first token",
            )
            times = token_times[request_id]
            require(
                len(times) == row["output_tokens"],
                "completed request: expected one event per token quantum",
            )
            intra_request_publication_gaps.extend(
                right - left for left, right in zip(times, times[1:]))
            completion_ns = completions[request_id]
            latency_ns = completion_ns - scheduled_ns
            request_metrics.append({
                "completion_latency_ns": latency_ns,
                "completion_ns": completion_ns,
                "model_id": row["model_id"],
                "queue_ns": dispatches[request_id] - scheduled_ns,
                "request_id": request_id,
                "slo_met": latency_ns <= row["slo_us"] * 1000,
                "ttft_ns": first_tokens[request_id] - scheduled_ns,
            })

    duration_ns = run_end_ns - run_start_ns
    demand_intervals = []
    for request_id, row in expected.items():
        start_ns = run_start_ns + row["arrival_us"] * 1000
        end_ns = completions.get(request_id, stranded.get(request_id))
        require(end_ns is not None and end_ns >= start_ns,
                "request: invalid demand interval")
        demand_intervals.append((start_ns, end_ns))
    merged_intervals: list[list[int]] = []
    for start_ns, end_ns in sorted(demand_intervals):
        if not merged_intervals or start_ns > merged_intervals[-1][1]:
            merged_intervals.append([start_ns, end_ns])
        else:
            merged_intervals[-1][1] = max(
                merged_intervals[-1][1], end_ns)
    all_token_times = sorted(
        value for values in token_times.values() for value in values)
    global_publication_gaps: list[int] = []
    for start_ns, end_ns in merged_intervals:
        points = [start_ns]
        points.extend(
            value for value in all_token_times
            if start_ns <= value <= end_ns)
        points.append(end_ns)
        global_publication_gaps.extend(
            right - left for left, right in zip(points, points[1:]))
    completed_count = len(completions)
    slo_count = sum(row["slo_met"] for row in request_metrics)
    counts = Counter(row["model_id"] for row in request_metrics)
    publication_times = {
        epoch: (
            ownership_commit_complete[epoch]["t_ns"]
            if epoch in ownership_commit_complete else ready_ns
        )
        for epoch, ready_ns in model_ready_times.items()
    }
    publication_delays = [
        publication_ns - transition_intents[epoch]["t_ns"]
        for epoch, publication_ns in publication_times.items()
        if epoch in transition_intents
        and publication_ns >= transition_intents[epoch]["t_ns"]
    ]
    commit_delays = [
        complete["t_ns"] - model_ready_times[epoch]
        for epoch, complete in ownership_commit_complete.items()
        if epoch in model_ready_times
        and complete["t_ns"] >= model_ready_times[epoch]
    ]
    switch_timeline = [
        {
            "controller_epoch": epoch,
            "intent_t_offset_ns": intent["t_ns"] - run_start_ns,
            "model_id": intent["model_id"],
            "publication_t_offset_ns": (
                publication_times[epoch] - run_start_ns
                if epoch in publication_times else None
            ),
        }
        for epoch, intent in sorted(transition_intents.items())
    ]

    result: dict[str, Any] = {
        "completed_request_count": completed_count,
        "command_ledger": command_ledger,
        "controller_run_start_ns": controller_run_start_ns,
        "completion_latency_p50_ns": (
            percentile(
                [row["completion_latency_ns"] for row in request_metrics],
                1,
                2,
            ) if request_metrics else None
        ),
        "completion_latency_p95_ns": (
            percentile(
                [row["completion_latency_ns"] for row in request_metrics],
                95,
                100,
            ) if request_metrics else None
        ),
        "completion_latency_p99_ns": (
            percentile(
                [row["completion_latency_ns"] for row in request_metrics],
                99,
                100,
            ) if request_metrics else None
        ),
        "executor_failed": executor_failed,
        "cleanup_failed": cleanup_failed,
        "maximum_model_publication_gap_ns": (
            max(publication_delays) if publication_delays else 0
        ),
        "maximum_global_token_publication_gap_ns": (
            max(global_publication_gaps) if global_publication_gaps else 0
        ),
        "maximum_intra_request_token_publication_gap_ns": (
            max(intra_request_publication_gaps)
            if intra_request_publication_gaps else 0
        ),
        "model_completed_counts": dict(sorted(counts.items())),
        "model_throughput_milli_rps": {
            model_id: count * 1_000_000_000_000 // duration_ns
            for model_id, count in sorted(counts.items())
        },
        "ownership_commit_count": len(ownership_commit_complete),
        "ownership_commit_latency_p95_ns": (
            percentile(commit_delays, 95, 100) if commit_delays else None
        ),
        "phase_duration_ns": {
            name: {
                "count": len(values),
                "p50": percentile(values, 1, 2) if values else None,
                "p95": percentile(values, 95, 100) if values else None,
            }
            for name, values in sorted(phase_durations.items())
        },
        "precommit_rollback_count": len(rollback_restores),
        "queue_p50_ns": (
            percentile([row["queue_ns"] for row in request_metrics], 1, 2)
            if request_metrics else None
        ),
        "queue_p95_ns": (
            percentile([row["queue_ns"] for row in request_metrics], 95, 100)
            if request_metrics else None
        ),
        "queue_p99_ns": (
            percentile([row["queue_ns"] for row in request_metrics], 99, 100)
            if request_metrics else None
        ),
        "request_count": len(expected),
        "request_metrics": request_metrics,
        "run_duration_ns": duration_ns,
        "run_id": run_id,
        "runtime_config_sha256": runtime_config_sha256,
        "trace_origin_ns": run_start_ns,
        "slo_goodput_milli_rps": slo_count * 1_000_000_000_000 // duration_ns,
        "slo_met_count": slo_count,
        "stranded_request_count": len(stranded),
        "switch_intent_queued_count": queued_switch_intent_count,
        "timeline": {
            "model_token_throughput": build_throughput_timeline(
                run_start_ns,
                run_end_ns,
                sorted({row["model_id"] for row in expected.values()}),
                token_events,
            ),
            "sample_step_ns": TIMELINE_STEP_NS,
            "schema": "s40-reduced-timeline-v1",
            "switches": switch_timeline,
            "throughput_window_ns": THROUGHPUT_WINDOW_NS,
        },
        "throughput_milli_rps": completed_count * 1_000_000_000_000 // duration_ns,
        "tokens_per_second_milli": (
            sum(len(request_state[request_id]["committed_output_tokens"])
                for request_id in completions)
            * 1_000_000_000_000
            // duration_ns
        ),
        "ttft_p50_ns": (
            percentile([row["ttft_ns"] for row in request_metrics], 1, 2)
            if request_metrics else None
        ),
        "ttft_p95_ns": (
            percentile([row["ttft_ns"] for row in request_metrics], 95, 100)
            if request_metrics else None
        ),
        "ttft_p99_ns": (
            percentile([row["ttft_ns"] for row in request_metrics], 99, 100)
            if request_metrics else None
        ),
        "verdict": (
            "FAIL_CLEANUP" if cleanup_failed
            else "FAIL_EXECUTOR" if executor_failed
            else "FAIL_EVENT" if unexplained_failed_events
            else "PASS"
        ),
    }
    if resources is None:
        result["energy"] = {
            "gpu_energy_scope": "UNKNOWN",
            "energy_claim_authorized": False,
            "phone_energy": "UNKNOWN",
            "server_wall_energy": "UNKNOWN",
            "total_system_energy": "UNKNOWN",
        }
    else:
        resource_result = validate_resources(
            resources, run_id, run_start_ns, run_end_ns)
        result["energy"] = {
            **resource_result,
            "phone_energy": "UNKNOWN",
            "server_wall_energy": "UNKNOWN",
            "total_system_energy": "UNKNOWN",
        }
    return result


def reduce_paths(
        event_path: Path,
        expected_path: Path,
        resource_path: Path | None = None,
        trace_start_path: Path | None = None) -> dict[str, Any]:
    events = read_jsonl(event_path, "events")
    expected = read_jsonl(expected_path, "expected_requests")
    resources = (
        read_jsonl(resource_path, "resources")
        if resource_path is not None else None
    )
    trace_start = (
        read_json(trace_start_path, "trace_start")
        if trace_start_path is not None else None
    )
    if trace_start is not None:
        require(
            trace_start_path.read_bytes() == canonical_bytes(trace_start),
            "trace_start: not canonical JSON",
        )
    return reduce_events(
        events, expected, resources, trace_start=trace_start)
