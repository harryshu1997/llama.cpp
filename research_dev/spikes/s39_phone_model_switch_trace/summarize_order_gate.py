#!/usr/bin/env python3
"""Reduce the matched W3 sorted-versus-shuffled physical evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import summarize_qwen_batch_wifi as common
from direct_order_probe import (
    FROZEN_REQUESTS,
    FROZEN_STEPS,
    REQUEST_BASE,
    sequence_order,
)


DEFAULT_EVIDENCE = HERE / "results" / "w3_order_gate"
DEFAULT_OUTPUT = DEFAULT_EVIDENCE / "order_gate_certificate.json"

MODEL_SHA256 = "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"
PROMPT_TOKENS = [785, 6722, 315, 9625, 374]
GENERATED_TOKENS = [12095, 13, 3555, 374, 279, 6722, 315, 279]
N_EMBD = 5120
N_LAYER = 40
CUT_LAYER = 30
ROWS = FROZEN_REQUESTS * (len(PROMPT_TOKENS) + FROZEN_STEPS - 1)
ACTIVATION_BYTES = ROWS * N_EMBD * 4
BATCH_SIZES = [FROZEN_REQUESTS * len(PROMPT_TOKENS)] + [
    FROZEN_REQUESTS
] * (FROZEN_STEPS - 1)
SPEEDUP_GATE_MILLI = 1200
THERMAL_MEAN_DELTA_MILLIC = 3000
RUNS = (
    ("sorted_ab", "sorted", "ab", 0),
    ("shuffled_ab", "shuffled", "ab", 1),
    ("shuffled_ba", "shuffled", "ba", 0),
    ("sorted_ba", "sorted", "ba", 1),
)

REPORT_KEYS = {
    "batch_events",
    "batch_summary",
    "configuration",
    "latency_ms",
    "order",
    "permutation",
    "requests",
    "route_id",
    "schema",
    "scope",
    "status",
    "token_checks",
    "tokens_exact",
    "transport",
    "verdict",
    "worker",
}
CONFIG_KEYS = {
    "batch_knee",
    "expected_tokens",
    "gather_us",
    "prompt_tokens",
    "queue_depth",
    "requests",
    "slo_ms",
    "steps",
}
EVENT_KEYS = {
    "batch_size",
    "compute_us",
    "decode_rows",
    "max_queue_us",
    "mixed_phase",
    "phases",
    "positions",
    "prefill_rows",
    "priorities",
    "release_reason",
    "request_ids",
    "route_epochs",
    "sequence_ids",
}
REQUEST_KEYS = {
    "elapsed_ms",
    "request_id",
    "route_epoch",
    "sequence_id",
    "slo_met",
    "tokens",
}
DIRECT_CERT_KEYS = {
    "activation_payload_bytes",
    "batches",
    "cut_layer",
    "file_type",
    "head_endpoint",
    "host_activation_payload_bytes",
    "layer_end",
    "layer_start",
    "model_sha256",
    "n_embd",
    "n_layer",
    "rows",
    "run_rc",
    "schema",
    "status",
    "tail_endpoint",
}
THERMAL_KEYS = {"captured_utc_ns", "label", "samples", "schema"}
THERMAL_SAMPLE_KEYS = {
    "device_boot_id",
    "gpu_max_millic",
    "gpu_zones_millic",
    "serial",
}
CONTEXT_KEYS = {
    "acquisition_utc",
    "gate",
    "model",
    "op12",
    "op15",
    "runtime",
    "schema",
    "source",
}


def exact(value: Any, expected: Any, field: str) -> None:
    common.require(
        value == expected and type(value) is type(expected),
        field,
    )


def expected_event(
    index: int,
    order: list[int],
) -> dict[str, Any]:
    prefill = index == 0
    if prefill:
        sequence_ids = [
            sequence_id
            for sequence_id in order
            for _ in PROMPT_TOKENS
        ]
        request_ids = [
            REQUEST_BASE + sequence_id
            for sequence_id in order
            for _ in PROMPT_TOKENS
        ]
        positions = list(range(len(PROMPT_TOKENS))) * FROZEN_REQUESTS
    else:
        sequence_ids = order
        request_ids = [
            REQUEST_BASE + sequence_id for sequence_id in order
        ]
        positions = [len(PROMPT_TOKENS) + index - 1] * FROZEN_REQUESTS
    return {
        "decode_rows": 0 if prefill else FROZEN_REQUESTS,
        "mixed_phase": False,
        "phases": [
            "prefill" if prefill else "decode"
        ] * BATCH_SIZES[index],
        "positions": positions,
        "prefill_rows": BATCH_SIZES[index] if prefill else 0,
        "priorities": [0] * BATCH_SIZES[index],
        "release_reason": "BATCH_KNEE" if prefill else "DEADLINE",
        "request_ids": request_ids,
        "route_epochs": [1] * BATCH_SIZES[index],
        "sequence_ids": sequence_ids,
    }


def validate_report(raw: bytes, name: str, order_name: str) -> dict[str, Any]:
    report = common.require_keys(
        common.parse_json(raw, name),
        REPORT_KEYS,
        name,
    )
    expected_scalars = {
        "order": order_name,
        "permutation": sequence_order(order_name),
        "route_id": f"s39-qwen-order-{name}",
        "schema": "s39-direct-order-route-v1",
        "status": "DIRECT_ORDER_MECHANICS_PASS",
        "token_checks": FROZEN_REQUESTS * FROZEN_STEPS,
        "tokens_exact": True,
        "verdict": "PASS",
    }
    for key, value in expected_scalars.items():
        exact(report[key], value, f"{name}.{key}")

    config = common.require_keys(
        report["configuration"],
        CONFIG_KEYS,
        f"{name}.configuration",
    )
    expected_config = {
        "batch_knee": BATCH_SIZES[0],
        "expected_tokens": GENERATED_TOKENS,
        "gather_us": 50000,
        "prompt_tokens": PROMPT_TOKENS,
        "queue_depth": 256,
        "requests": FROZEN_REQUESTS,
        "steps": FROZEN_STEPS,
    }
    for key, value in expected_config.items():
        exact(config[key], value, f"{name}.configuration.{key}")
    common.require_number(
        config["slo_ms"],
        f"{name}.configuration.slo_ms",
        Decimal(1),
    )

    worker = common.require_keys(
        report["worker"],
        common.WORKER_KEYS,
        f"{name}.worker",
    )
    expected_worker = {
        "file_type": 15,
        "layer_end": N_LAYER,
        "layer_start": 0,
        "max_streams": FROZEN_REQUESTS,
        "model_sha256": MODEL_SHA256,
        "n_embd": N_EMBD,
        "n_layer": N_LAYER,
    }
    for key, value in expected_worker.items():
        exact(worker[key], value, f"{name}.worker.{key}")
    for key in ("capabilities", "n_batch", "n_ctx_seq", "n_ubatch"):
        common.require_int(worker[key], f"{name}.worker.{key}", 1)
    common.require(
        worker["n_batch"] >= BATCH_SIZES[0]
        and worker["n_ubatch"] >= BATCH_SIZES[0]
        and worker["capabilities"] & 16,
        f"{name}.worker.capacity",
    )

    events_value = report["batch_events"]
    common.require(
        isinstance(events_value, list)
        and len(events_value) == len(BATCH_SIZES),
        f"{name}.batch_events",
    )
    events = []
    order = sequence_order(order_name)
    for index, candidate in enumerate(events_value):
        field = f"{name}.batch_events[{index}]"
        event = common.require_keys(candidate, EVENT_KEYS, field)
        exact(event["batch_size"], BATCH_SIZES[index], f"{field}.batch_size")
        common.require_int(event["compute_us"], f"{field}.compute_us", 1)
        common.require_int(event["max_queue_us"], f"{field}.max_queue_us")
        for key, value in expected_event(index, order).items():
            exact(event[key], value, f"{field}.{key}")
        common.require(
            event["decode_rows"] + event["prefill_rows"]
            == event["batch_size"],
            f"{field}.row_conservation",
        )
        events.append(event)

    summary = common.require_keys(
        report["batch_summary"],
        {"batch_count", "batch_sizes", "compute_us_total", "max_batch"},
        f"{name}.batch_summary",
    )
    exact(
        summary["batch_count"],
        len(BATCH_SIZES),
        f"{name}.batch_summary.batch_count",
    )
    exact(
        summary["batch_sizes"],
        BATCH_SIZES,
        f"{name}.batch_summary.batch_sizes",
    )
    exact(
        summary["max_batch"],
        max(BATCH_SIZES),
        f"{name}.batch_summary.max_batch",
    )
    compute_us = sum(event["compute_us"] for event in events)
    exact(
        summary["compute_us_total"],
        compute_us,
        f"{name}.batch_summary.compute_us_total",
    )

    requests = report["requests"]
    common.require(
        isinstance(requests, list) and len(requests) == FROZEN_REQUESTS,
        f"{name}.requests",
    )
    elapsed = []
    for sequence_id, candidate in enumerate(requests):
        field = f"{name}.requests[{sequence_id}]"
        request = common.require_keys(candidate, REQUEST_KEYS, field)
        expected_request = {
            "request_id": REQUEST_BASE + sequence_id,
            "route_epoch": 1,
            "sequence_id": sequence_id,
            "slo_met": True,
            "tokens": GENERATED_TOKENS,
        }
        for key, value in expected_request.items():
            exact(request[key], value, f"{field}.{key}")
        elapsed.append(
            common.require_number(
                request["elapsed_ms"],
                f"{field}.elapsed_ms",
                Decimal(0),
            )
        )

    latency = common.require_keys(
        report["latency_ms"],
        {"max", "p50", "p95"},
        f"{name}.latency_ms",
    )
    expected_latency = {
        "max": max(elapsed),
        "p50": common.nearest_rank(elapsed, 1, 2),
        "p95": common.nearest_rank(elapsed, 19, 20),
    }
    for key, value in expected_latency.items():
        common.require(
            common.require_number(
                latency[key],
                f"{name}.latency_ms.{key}",
                Decimal(0),
            )
            == value,
            f"{name}.latency_ms.{key}",
        )

    transport = common.require_keys(
        report["transport"],
        {
            "direct_activation_payload_bytes",
            "host_activation_payload_bytes",
            "mode",
            "weight_provisioning",
        },
        f"{name}.transport",
    )
    expected_transport = {
        "direct_activation_payload_bytes": ACTIVATION_BYTES,
        "host_activation_payload_bytes": 0,
        "mode": "OP15_TO_OP12_DIRECT_WIFI",
        "weight_provisioning": "USB_BEFORE_SERVICE",
    }
    for key, value in expected_transport.items():
        exact(transport[key], value, f"{name}.transport.{key}")
    exact(
        report["scope"],
        {
            "energy": "NOT_MEASURED",
            "inter_stage_overlap": "NOT_IMPLEMENTED",
            "variable": "ROW_ORDER_ONLY",
        },
        f"{name}.scope",
    )
    return {
        "artifact_sha256": common.sha256(raw),
        "compute_us": compute_us,
        "latency_p50_ns": common.decimal_ms_to_ns(
            latency["p50"],
            f"{name}.latency_ms.p50",
        ),
    }


def validate_relay(raw: bytes, name: str) -> dict[str, Any]:
    records = common.extract_prefixed_records(
        raw,
        b"DIRECTCERT ",
        f"{name}.relay",
    )
    common.require(len(records) == 1, f"{name}.relay: expected one cert")
    cert = common.require_keys(
        records[0],
        DIRECT_CERT_KEYS,
        f"{name}.relay",
    )
    expected = {
        "activation_payload_bytes": ACTIVATION_BYTES,
        "batches": len(BATCH_SIZES),
        "cut_layer": CUT_LAYER,
        "file_type": 15,
        "head_endpoint": "127.0.0.1:39315",
        "host_activation_payload_bytes": 0,
        "layer_end": N_LAYER,
        "layer_start": 0,
        "model_sha256": MODEL_SHA256,
        "n_embd": N_EMBD,
        "n_layer": N_LAYER,
        "rows": ROWS,
        "run_rc": 0,
        "schema": "ls-stage-direct-relay-v1",
        "status": "DIRECT_RELAY_OK",
        "tail_endpoint": "172.20.59.72:39312",
    }
    for key, value in expected.items():
        exact(cert[key], value, f"{name}.relay.{key}")
    return {"artifact_sha256": common.sha256(raw)}


def validate_worker(
    raw: bytes,
    *,
    field: str,
    layer_start: int,
    layer_end: int,
    role: str,
    mode: str,
    allow_cpu_get_rows: bool,
) -> dict[str, Any]:
    sessions = common.extract_prefixed_records(
        raw,
        b"SESSIONCERT ",
        f"{field}.session",
    )
    common.require(len(sessions) == 2, f"{field}: expected two sessions")
    pid = None
    nonce = None
    boot = None
    nodes = []
    for index, candidate in enumerate(sessions):
        session = common.require_keys(
            candidate,
            common.SESSION_KEYS,
            f"{field}.session[{index}]",
        )
        expected = {
            "expected_backend": "GPUOpenCL",
            "layer_end": layer_end,
            "layer_start": layer_start,
            "missing_buffer_compute_nodes": 0,
            "n_layer": N_LAYER,
            "placement_status": "SCHEDULED_PLACEMENT_OK",
            "proto_version": 2,
            "reset_applied": index == 0,
            "schema": "ls-stagenet-session-v2",
            "session_end": "DETACH" if index == 0 else "STOP",
            "session_id": index + 1,
            "steps_session": ROWS,
            "steps_total": ROWS * (index + 1),
        }
        for key, value in expected.items():
            exact(session[key], value, f"{field}.session[{index}].{key}")
        current_pid = common.require_int(
            session["worker_pid"],
            f"{field}.session[{index}].worker_pid",
            1,
        )
        current_nonce = session["worker_boot_nonce"]
        current_boot = session["device_boot_id"]
        common.require(
            isinstance(current_nonce, str) and bool(current_nonce),
            f"{field}.session[{index}].worker_boot_nonce",
        )
        common.require(
            isinstance(current_boot, str) and bool(current_boot),
            f"{field}.session[{index}].device_boot_id",
        )
        pid = current_pid if pid is None else pid
        nonce = current_nonce if nonce is None else nonce
        boot = current_boot if boot is None else boot
        exact(current_pid, pid, f"{field}.session[{index}].worker_pid")
        exact(current_nonce, nonce, f"{field}.session[{index}].worker_boot_nonce")
        exact(current_boot, boot, f"{field}.session[{index}].device_boot_id")
        node_count = common.validate_op_map(
            session["compute_by_op_and_buffer"],
            f"{field}.session[{index}].ops",
            allow_cpu_get_rows,
        )
        common.require(node_count > 0, f"{field}.session[{index}].ops")
        nodes.append(node_count)

    placements = common.extract_prefixed_records(
        raw,
        b"PLACEMENTCERT ",
        f"{field}.placement",
    )
    common.require(len(placements) == 1, f"{field}: expected one placement")
    placement = common.require_keys(
        placements[0],
        common.PLACEMENT_KEYS,
        f"{field}.placement",
    )
    expected_placement = {
        "layer_end": layer_end,
        "layer_start": layer_start,
        "missing_buffer_compute_nodes": 0,
        "mode": mode,
        "n_layer": N_LAYER,
        "pid": pid,
        "role": role,
        "run_rc": 0,
        "schema": "layersplit-scheduled-placement-v2",
        "status": "SCHEDULED_PLACEMENT_OK",
    }
    for key, value in expected_placement.items():
        exact(placement[key], value, f"{field}.placement.{key}")
    placement_nodes = common.require_int(
        placement["compute_nodes"],
        f"{field}.placement.compute_nodes",
        1,
    )
    derived_nodes = common.validate_op_map(
        placement["compute_by_op_and_buffer"],
        f"{field}.placement.ops",
        allow_cpu_get_rows,
    )
    exact(placement_nodes, derived_nodes, f"{field}.placement.compute_nodes")
    exact(
        placement["compute_by_op_and_buffer"],
        sessions[-1]["compute_by_op_and_buffer"],
        f"{field}.placement.session_match",
    )
    exact(nodes[-1], placement_nodes, f"{field}.placement.session_nodes")
    return {
        "artifact_sha256": common.sha256(raw),
        "device_boot_id": boot,
        "session_compute_nodes": nodes,
        "worker_boot_nonce": nonce,
        "worker_pid": pid,
    }


def validate_thermal(
    raw: bytes,
    expected_label: str,
    expected_boots: dict[str, str],
) -> dict[str, Any]:
    value = common.require_keys(
        common.parse_json(raw, expected_label),
        THERMAL_KEYS,
        expected_label,
    )
    exact(
        value["schema"],
        "s39-phone-thermal-bracket-v1",
        f"{expected_label}.schema",
    )
    exact(value["label"], expected_label, f"{expected_label}.label")
    common.require_int(
        value["captured_utc_ns"],
        f"{expected_label}.captured_utc_ns",
        1,
    )
    samples = value["samples"]
    common.require(
        isinstance(samples, dict) and set(samples) == {"OP12", "OP15"},
        f"{expected_label}.samples",
    )
    maxima = {}
    serials = {
        "OP12": "5ae7a43d",
        "OP15": "3C15AU002CL00000",
    }
    for device in sorted(samples):
        field = f"{expected_label}.samples.{device}"
        sample = common.require_keys(
            samples[device],
            THERMAL_SAMPLE_KEYS,
            field,
        )
        exact(
            sample["device_boot_id"],
            expected_boots[device],
            f"{field}.device_boot_id",
        )
        exact(sample["serial"], serials[device], f"{field}.serial")
        maximum = common.require_int(
            sample["gpu_max_millic"],
            f"{field}.gpu_max_millic",
            1,
        )
        zones = sample["gpu_zones_millic"]
        common.require(
            isinstance(zones, dict)
            and bool(zones)
            and all(
                isinstance(name, str)
                and name.startswith("gpuss-")
                and common.is_int(temp)
                and 0 < temp < 200000
                for name, temp in zones.items()
            ),
            f"{field}.gpu_zones_millic",
        )
        exact(maximum, max(zones.values()), f"{field}.gpu_max_millic")
        maxima[device] = maximum
    return {
        "artifact_sha256": common.sha256(raw),
        "gpu_max_millic": maxima,
    }


def validate_context(raw: bytes) -> dict[str, Any]:
    value = common.require_keys(
        common.parse_json(raw, "run_context"),
        CONTEXT_KEYS,
        "run_context",
    )
    exact(
        value["schema"],
        "s39-direct-order-run-context-v1",
        "run_context.schema",
    )
    common.require(
        isinstance(value["acquisition_utc"], str)
        and bool(value["acquisition_utc"]),
        "run_context.acquisition_utc",
    )
    exact(
        value["gate"],
        {
            "minimum_speedup_milli": SPEEDUP_GATE_MILLI,
            "thermal_mean_delta_limit_millic": THERMAL_MEAN_DELTA_MILLIC,
        },
        "run_context.gate",
    )
    exact(
        value["model"],
        {
            "file_type": 15,
            "model_sha256": MODEL_SHA256,
            "route": "qwen3-14b-q4_k_m",
        },
        "run_context.model",
    )
    exact(
        value["op12"],
        {
            "boot_id": "3ba01077-fa7c-4a69-859d-705eee5041f9",
            "layer_range": [30, 40],
            "layersplit_sha256": (
                "94160b3ccb731474b4aba77333ab1c95e"
                "b2651ae26bb9ce0f3377e9db24cc299"
            ),
            "model_shard_sha256": (
                "72e312af745160dc33a0ba39ba94fbbce"
                "6112950d0409d39c42ddc3b25e756ab"
            ),
            "serial": "5ae7a43d",
            "wifi": "172.20.59.72",
        },
        "run_context.op12",
    )
    exact(
        value["op15"],
        {
            "boot_id": "65582fc7-4801-4d81-b171-3cb37e93b7fb",
            "layer_range": [0, 30],
            "layersplit_sha256": (
                "94160b3ccb731474b4aba77333ab1c95e"
                "b2651ae26bb9ce0f3377e9db24cc299"
            ),
            "model_shard_sha256": (
                "ba56b9c5e19b3a4512777e6a47803cc0"
                "3261c2d3c2734965cd5ec96b7c6c59fb"
            ),
            "relay_sha256": (
                "1c809cb50cae6aa86869d61068a05173c"
                "719e4542c851e478ee1e033c5456929"
            ),
            "serial": "3C15AU002CL00000",
            "wifi": "172.20.173.218",
        },
        "run_context.op15",
    )
    exact(
        value["runtime"],
        {
            "activation_path": "OP15_TO_OP12_WIFI_TCP",
            "host_activation_payload_bytes": 0,
            "host_role": "ADMISSION_AND_RESULT_OWNERSHIP",
            "inter_stage_overlap": "NOT_IMPLEMENTED",
            "relay_endpoint": "172.20.173.218:39415",
            "tail_endpoint": "172.20.59.72:39312",
            "treatment_orders": [
                ["sorted", "shuffled"],
                ["shuffled", "sorted"],
            ],
            "weight_path": "USB_ADB_BEFORE_SERVICE",
        },
        "run_context.runtime",
    )
    exact(
        value["source"],
        {
            "git_commit": "7d1926dffc0e9666ec5aa507e688727048827676",
            "mixed_phase_batcher_sha256": (
                "f8eb2b440a494b91f4c42f4ae156fd885"
                "af7a8481305f9490f8943ecb251ef4f"
            ),
        },
        "run_context.source",
    )
    return {
        "artifact_sha256": common.sha256(raw),
        "acquisition_utc": value["acquisition_utc"],
    }


def build(evidence: Path = DEFAULT_EVIDENCE) -> dict[str, Any]:
    context = validate_context(
        common.read_bytes(evidence / "RUN_CONTEXT.json", "run_context")
    )
    reports = {}
    relays = {}
    for name, order_name, _pair, _session_index in RUNS:
        reports[name] = validate_report(
            common.read_bytes(evidence / f"{name}.json", name),
            name,
            order_name,
        )
        relays[name] = validate_relay(
            common.read_bytes(
                evidence / f"relay_{name}.log",
                f"relay_{name}",
            ),
            name,
        )

    workers = {}
    for pair in ("ab", "ba"):
        workers[f"OP15_{pair}"] = validate_worker(
            common.read_bytes(
                evidence / f"op15_{pair}.log",
                f"op15_{pair}",
            ),
            field=f"op15_{pair}",
            layer_start=0,
            layer_end=CUT_LAYER,
            role="phone_stage",
            mode="stagenet",
            allow_cpu_get_rows=True,
        )
        workers[f"OP12_{pair}"] = validate_worker(
            common.read_bytes(
                evidence / f"op12_{pair}.log",
                f"op12_{pair}",
            ),
            field=f"op12_{pair}",
            layer_start=CUT_LAYER,
            layer_end=N_LAYER,
            role="host_tail_v3",
            mode="tailv3",
            allow_cpu_get_rows=False,
        )

    session_nodes = {}
    for name, _order_name, pair, session_index in RUNS:
        session_nodes[name] = {
            "OP12": workers[f"OP12_{pair}"]["session_compute_nodes"][
                session_index
            ],
            "OP15": workers[f"OP15_{pair}"]["session_compute_nodes"][
                session_index
            ],
        }

    thermal = {}
    expected_boots = {
        "OP12": workers["OP12_ab"]["device_boot_id"],
        "OP15": workers["OP15_ab"]["device_boot_id"],
    }
    exact(
        workers["OP12_ba"]["device_boot_id"],
        expected_boots["OP12"],
        "OP12 boot changed across pairs",
    )
    exact(
        workers["OP15_ba"]["device_boot_id"],
        expected_boots["OP15"],
        "OP15 boot changed across pairs",
    )
    for name, _order_name, _pair, _session_index in RUNS:
        for edge in ("before", "after"):
            label = f"{name}_{edge}"
            thermal[label] = validate_thermal(
                common.read_bytes(
                    evidence / f"thermal_{label}.json",
                    f"thermal_{label}",
                ),
                label,
                expected_boots,
            )

    sorted_runs = ["sorted_ab", "sorted_ba"]
    shuffled_runs = ["shuffled_ab", "shuffled_ba"]
    sorted_compute = sum(reports[name]["compute_us"] for name in sorted_runs)
    shuffled_compute = sum(
        reports[name]["compute_us"] for name in shuffled_runs
    )
    pair_speedup_milli = {}
    pair_direction = True
    for pair in ("ab", "ba"):
        sorted_us = reports[f"sorted_{pair}"]["compute_us"]
        shuffled_us = reports[f"shuffled_{pair}"]["compute_us"]
        speedup = shuffled_us * 1000 // sorted_us
        pair_speedup_milli[pair] = speedup
        pair_direction = pair_direction and speedup >= SPEEDUP_GATE_MILLI
    aggregate_speedup_milli = shuffled_compute * 1000 // sorted_compute

    thermal_balance = {}
    thermal_matched = True
    for device in ("OP12", "OP15"):
        sorted_start_sum = sum(
            thermal[f"{name}_before"]["gpu_max_millic"][device]
            for name in sorted_runs
        )
        shuffled_start_sum = sum(
            thermal[f"{name}_before"]["gpu_max_millic"][device]
            for name in shuffled_runs
        )
        mean_delta = abs(sorted_start_sum - shuffled_start_sum) // 2
        thermal_balance[device] = {
            "mean_start_delta_millic": mean_delta,
            "shuffled_start_sum_millic": shuffled_start_sum,
            "sorted_start_sum_millic": sorted_start_sum,
        }
        thermal_matched = (
            thermal_matched
            and mean_delta <= THERMAL_MEAN_DELTA_MILLIC
        )

    effect_pass = (
        thermal_matched
        and pair_direction
        and aggregate_speedup_milli >= SPEEDUP_GATE_MILLI
    )
    status = (
        "ROW_ORDER_EFFECT_PASS"
        if effect_pass
        else (
            "ROW_ORDER_EFFECT_THERMAL_UNMATCHED"
            if not thermal_matched
            else "ROW_ORDER_EFFECT_NOT_CONFIRMED"
        )
    )
    return {
        "schema": "s39-direct-order-gate-v1",
        "status": status,
        "gate": {
            "aggregate_speedup_milli": aggregate_speedup_milli,
            "minimum_speedup_milli": SPEEDUP_GATE_MILLI,
            "pair_direction_pass": pair_direction,
            "pair_speedup_milli": pair_speedup_milli,
            "thermal_matched": thermal_matched,
            "thermal_mean_delta_limit_millic": THERMAL_MEAN_DELTA_MILLIC,
        },
        "execution": {
            "activation_bytes_per_run": ACTIVATION_BYTES,
            "batch_sizes": BATCH_SIZES,
            "rows_per_run": ROWS,
            "session_compute_nodes": session_nodes,
            "shuffled_compute_us_total": shuffled_compute,
            "sorted_compute_us_total": sorted_compute,
        },
        "correctness": {
            "generated_tokens": GENERATED_TOKENS,
            "model_sha256": MODEL_SHA256,
            "token_checks": FROZEN_REQUESTS * FROZEN_STEPS * len(RUNS),
            "tokens_exact": True,
        },
        "thermal": thermal_balance,
        "evidence": {
            "run_context": context,
            "relays": relays,
            "reports": reports,
            "thermal": thermal,
            "workers": workers,
        },
        "scope": {
            "energy": "NOT_MEASURED",
            "inter_stage_overlap": "NOT_IMPLEMENTED",
            "variable": "ROW_ORDER_ONLY",
        },
    }


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    raw = common.canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.evidence)
    write_atomic(args.output, result)
    print(common.canonical_bytes(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (common.BatchEvidenceError, OSError, ValueError) as error:
        print(f"S39_ORDER_GATE_EVIDENCE_ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
