#!/usr/bin/env python3
"""Versioned S12 mechanics replay for independent WiFi ingress and USB egress."""

from __future__ import annotations

import argparse
import heapq
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from policies import background_hbm_at, causal_batch_size
from profile_coverage import prepare_trace
from s12lib import (
    MAX_SAFE_INT,
    MIB,
    S12Error,
    canonical_json,
    load_json,
    nearest_rank,
    profile_rows_by_batch,
    read_bytes_once,
    read_jsonl_snapshot,
    require_bool,
    require_exact_keys,
    require_int,
    require_str,
    sha256_object,
    sha256_bytes,
    strict_json_loads,
    validate_profile,
    write_canonical,
)
from vq_sim import RequestState, TERMINALS, run_offline


US_PER_SECOND = 1000000
DUAL_PATH_CLAIM_SCOPE = "SYNTHETIC_ONLY_DUAL_PATH_MECHANICS_NO_PERFORMANCE_CLAIM"
DUAL_PATH_POLICIES = (
    "server_only_optimized",
    "causal_server_batch",
    "fixed_phone",
)
REPO_ROOT = Path(__file__).resolve().parents[3]
PR_COMPLETE = 0
PR_HBM = 1
PR_ARRIVAL = 2
PR_FLUSH = 3


@dataclass
class PhoneGroup:
    group_id: str
    request_ids: list[str]
    batch_size: int
    row: dict[str, Any]
    admitted_us: int
    reason: str
    server_admission_feasible: bool
    wifi_input_bytes_by_step: list[int]
    usb_result_bytes_by_step: list[int]
    compute_us_by_step: list[int]
    tail_us_by_step: list[int]
    phase_us_by_step: list[dict[str, int | None]]
    step_index: int = 0
    complete: bool = False

    @property
    def wifi_input_bytes(self) -> int:
        return self.wifi_input_bytes_by_step[self.step_index]

    @property
    def usb_result_bytes(self) -> int:
        return self.usb_result_bytes_by_step[self.step_index]

    @property
    def compute_us(self) -> int:
        return self.compute_us_by_step[self.step_index]

    @property
    def tail_us(self) -> int:
        return self.tail_us_by_step[self.step_index]

    @property
    def phase_us(self) -> dict[str, int | None]:
        return self.phase_us_by_step[self.step_index]


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _validate_path(name: str, value: Any, direction: str) -> dict[str, Any]:
    path = require_exact_keys(
        name,
        value,
        {"bytes_per_s", "direction", "domain_id", "fixed_latency_us", "path_id"},
    )
    require_int(f"{name}.bytes_per_s", path["bytes_per_s"], 1)
    require_str(f"{name}.direction", path["direction"], {direction})
    require_str(f"{name}.domain_id", path["domain_id"])
    require_int(f"{name}.fixed_latency_us", path["fixed_latency_us"], 0)
    require_str(f"{name}.path_id", path["path_id"])
    return path


def validate_dual_path_config(config: Any) -> dict[str, Any]:
    cfg = require_exact_keys(
        "config",
        config,
        {
            "background_hbm_mib_timeline",
            "batch_hold_us",
            "horizon_us",
            "offline_max_requests",
            "phone_queue_limit",
            "phone_residency",
            "policies",
            "profile_path",
            "schema",
            "server_hbm_capacity_mib",
            "server_queue_limit",
            "trace_mode",
            "trace_path",
            "transport",
        },
    )
    require_str("config.schema", cfg["schema"], {"s12-dual-path-config-v1"})
    require_str(
        "config.trace_mode",
        cfg["trace_mode"],
        {"strict_real", "semi_synthetic_shape_shadow"},
    )
    for key in (
        "batch_hold_us",
        "horizon_us",
        "phone_queue_limit",
        "server_hbm_capacity_mib",
        "server_queue_limit",
    ):
        require_int(f"config.{key}", cfg[key], 1)
    require_int("config.offline_max_requests", cfg["offline_max_requests"], 1, 24)
    require_str("config.profile_path", cfg["profile_path"])
    require_str("config.trace_path", cfg["trace_path"])

    policies = cfg["policies"]
    if type(policies) is not list or not policies:
        raise S12Error("config.policies: expected non-empty array")
    if any(type(policy) is not str or policy not in DUAL_PATH_POLICIES for policy in policies):
        raise S12Error("config.policies: unsupported policy")
    if len(policies) != len(set(policies)):
        raise S12Error("config.policies: duplicate policy")

    residency = require_exact_keys(
        "config.phone_residency",
        cfg["phone_residency"],
        {"ready", "weight_bytes"},
    )
    require_bool("config.phone_residency.ready", residency["ready"])
    require_int("config.phone_residency.weight_bytes", residency["weight_bytes"], 1)

    timeline = cfg["background_hbm_mib_timeline"]
    if type(timeline) is not list or not timeline:
        raise S12Error("config.background_hbm_mib_timeline: expected non-empty array")
    previous = -1
    for index, raw_point in enumerate(timeline):
        point = require_exact_keys(
            f"config.background_hbm_mib_timeline[{index}]",
            raw_point,
            {"t_us", "used_mib"},
        )
        t_us = require_int(f"config.background_hbm_mib_timeline[{index}].t_us", point["t_us"])
        require_int(
            f"config.background_hbm_mib_timeline[{index}].used_mib",
            point["used_mib"],
            0,
            cfg["server_hbm_capacity_mib"],
        )
        if t_us <= previous:
            raise S12Error("config.background_hbm_mib_timeline: timestamps must increase")
        if index == 0 and t_us != 0:
            raise S12Error("config.background_hbm_mib_timeline: first timestamp must be zero")
        if t_us > cfg["horizon_us"]:
            raise S12Error("config.background_hbm_mib_timeline: timestamp exceeds horizon")
        previous = t_us

    transport = require_exact_keys(
        "config.transport",
        cfg["transport"],
        {
            "compute_link_overlap",
            "host_result_buffer_bytes",
            "phone_inflight_group_limit",
            "phone_ingress_buffer_bytes",
            "phone_result_buffer_bytes",
            "scope",
            "usb_p2h",
            "wifi_h2p",
            "wifi_usb_independent",
        },
    )
    require_str(
        "config.transport.scope",
        transport["scope"],
        {"ASSUMED_MECHANICS_ONLY_UNMEASURED"},
    )
    require_str(
        "config.transport.compute_link_overlap",
        transport["compute_link_overlap"],
        {"SERIALIZED_UNTIL_MEASURED"},
    )
    require_bool("config.transport.wifi_usb_independent", transport["wifi_usb_independent"])
    wifi = _validate_path("config.transport.wifi_h2p", transport["wifi_h2p"], "HOST_TO_PHONE")
    usb = _validate_path("config.transport.usb_p2h", transport["usb_p2h"], "PHONE_TO_HOST")
    if wifi["path_id"] == usb["path_id"] or wifi["domain_id"] == usb["domain_id"]:
        raise S12Error("config.transport: WiFi and USB must bind distinct paths and domains")
    for key in (
        "host_result_buffer_bytes",
        "phone_ingress_buffer_bytes",
        "phone_result_buffer_bytes",
    ):
        require_int(f"config.transport.{key}", transport[key], 1)
    require_int(
        "config.transport.phone_inflight_group_limit",
        transport["phone_inflight_group_limit"],
        1,
        1,
    )
    return cfg


def transfer_us(nbytes: int, path: dict[str, Any]) -> int:
    require_int("transfer bytes", nbytes, 1)
    scaled = nbytes * US_PER_SECOND
    if scaled > MAX_SAFE_INT:
        raise S12Error("transfer duration: integer overflow")
    return path["fixed_latency_us"] + (scaled + path["bytes_per_s"] - 1) // path["bytes_per_s"]


def s11_head_wifi_input_bytes_by_step(batch_size: int, model: dict[str, Any]) -> list[int]:
    """Causal application payload for prefill followed by decode steps."""
    decode_steps = model["generated_tokens"] - 1
    prefill = 16 + batch_size * model["prompt_tokens"] * 4
    return [prefill] + [12 + batch_size * 12] * decode_steps


def s11_head_usb_result_bytes_by_step(
    batch_size: int,
    model: dict[str, Any],
) -> list[int]:
    row_bytes = batch_size * model["embedding_width"] * model["activation_element_bytes"]
    prefill = 8 + model["prompt_tokens"] * row_bytes
    return [prefill] + [8 + row_bytes] * (model["generated_tokens"] - 1)


def s11_head_wifi_input_bytes(batch_size: int, model: dict[str, Any]) -> int:
    return sum(s11_head_wifi_input_bytes_by_step(batch_size, model))


def s11_head_usb_result_bytes(row: dict[str, Any], model: dict[str, Any]) -> int:
    values = s11_head_usb_result_bytes_by_step(row["batch_size"], model)
    if sum(values) != row["activation_bytes"] + model["generated_tokens"] * 8:
        raise S12Error("USB result payload does not match activation row")
    return sum(values)


def split_phase_us(total_us: int, model: dict[str, Any]) -> list[int]:
    """Preserve an aggregate proxy while keeping every autoregressive step causal."""
    weights = [model["prompt_tokens"]] + [1] * (model["generated_tokens"] - 1)
    total_weight = sum(weights)
    boundaries = [0]
    cumulative = 0
    for weight in weights:
        cumulative += weight
        boundaries.append(total_us * cumulative // total_weight)
    values = [boundaries[index + 1] - boundaries[index] for index in range(len(weights))]
    if any(value <= 0 for value in values) or sum(values) != total_us:
        raise S12Error("phase proxy cannot be split into positive causal steps")
    return values


def new_phase_records(count: int) -> list[dict[str, int | None]]:
    return [
        {
            "wifi_start": None,
            "wifi_finish": None,
            "compute_start": None,
            "compute_finish": None,
            "usb_start": None,
            "usb_finish": None,
            "tail_start": None,
            "tail_finish": None,
        }
        for _ in range(count)
    ]


def _same_evidence_value(name: str, actual: Any, expected: Any) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise S12Error(f"profile evidence mismatch for {name}")


def validate_dual_profile_evidence(
    profile: dict[str, Any],
    repo_root: str | Path = REPO_ROOT,
) -> None:
    """Derive every scheduled S11 row field from its hashed plan and summary."""
    root = Path(repo_root)
    functional_bindings = [
        binding
        for binding in profile["source_bindings"]
        if binding["kind"] == "S11_FUNCTIONAL_RESULT"
    ]
    if len(functional_bindings) != 1:
        raise S12Error("profile evidence requires exactly one S11_FUNCTIONAL_RESULT")
    functional_binding = functional_bindings[0]
    functional_bytes = read_bytes_once(root / functional_binding["path"])
    if sha256_bytes(functional_bytes) != functional_binding["sha256"]:
        raise S12Error("profile functional-result snapshot digest mismatch")
    try:
        functional = strict_json_loads(functional_bytes.decode("ascii"))
    except UnicodeError as exc:
        raise S12Error("profile functional-result is not ASCII JSON") from exc
    model = profile["model"]
    try:
        identity_checks = {
            "functional.schema": (
                functional["schema"],
                "s11-fixed-route-functional-result-v1",
            ),
            "functional.verdict": (functional["result"]["functional_verdict"], "PASS"),
            "functional.energy": (functional["result"]["energy_verdict"], "NOT_RUN"),
            "functional.host_model_sha256": (
                "sha256:" + functional["artifacts"]["host_model"]["sha256"],
                model["host_model_sha256"],
            ),
            "functional.phone_weight_bytes": (
                functional["artifacts"]["op15_shard"]["bytes"],
                model["phone_weight_bytes"],
            ),
            "functional.prompt_sha256": (
                "sha256:" + functional["workload"]["prompt_sha256"],
                model["prompt_sha256"],
            ),
            "functional.prompt_tokens": (
                functional["workload"]["prompt_tokens"],
                model["prompt_tokens"],
            ),
            "functional.generated_tokens": (
                functional["workload"]["generated_tokens_per_request"],
                model["generated_tokens"],
            ),
            "functional.route": (
                functional["treatment"]["route"],
                profile["route"]["phone"],
            ),
            "functional.backend": (
                functional["treatment"]["op15_backend"],
                profile["route"]["phone_backend"],
            ),
            "functional.layer_range": (
                functional["treatment"]["op15_layer_range"],
                profile["route"]["phone_layer_range"],
            ),
            "frozen.model_id": (model["model_id"], "gemma-4-12b-it-f16"),
            "frozen.embedding_width": (model["embedding_width"], 3840),
            "frozen.activation_element_bytes": (model["activation_element_bytes"], 4),
        }
    except (KeyError, TypeError) as exc:
        raise S12Error("profile functional-result evidence is malformed") from exc
    for name, (actual, expected) in identity_checks.items():
        _same_evidence_value(name, actual, expected)

    for index, row in enumerate(profile["rows"]):
        plan_bytes = read_bytes_once(root / row["plan_path"])
        summary_bytes = read_bytes_once(root / row["summary_path"])
        if sha256_bytes(plan_bytes) != row["plan_sha256"]:
            raise S12Error(f"profile.rows[{index}].plan_sha256: snapshot digest mismatch")
        if sha256_bytes(summary_bytes) != row["summary_sha256"]:
            raise S12Error(f"profile.rows[{index}].summary_sha256: snapshot digest mismatch")
        try:
            plan = strict_json_loads(plan_bytes.decode("ascii"))
            summary = strict_json_loads(summary_bytes.decode("ascii"))
        except UnicodeError as exc:
            raise S12Error(f"profile.rows[{index}]: evidence is not ASCII JSON") from exc
        try:
            pairs = summary["pairs"]
            if type(pairs) is not list or len(pairs) != 1:
                raise S12Error(f"profile.rows[{index}]: expected one evidence pair")
            pair = pairs[0]
            control = pair["control_batch_metrics"]
            treatment = pair["treatment_batch_metrics"]
            control_records = control["group_records"]
            treatment_records = treatment["group_records"]
            if (
                type(control_records) is not list
                or len(control_records) != 1
                or type(treatment_records) is not list
                or len(treatment_records) != 1
            ):
                raise S12Error(f"profile.rows[{index}]: expected one group record per route")
            control_record = control_records[0]
            treatment_record = treatment_records[0]
            checks = {
                "plan.schema": (plan["schema"], "s11-fixed-route-poc-v1"),
                "plan.scope": (plan["scope"], "MECHANICS_ONLY"),
                "plan.treatment_route": (plan["treatment_route"], profile["route"]["phone"]),
                "plan.workload.batch_size": (plan["workload"]["batch_size"], row["batch_size"]),
                "plan.workload.n_gen": (
                    plan["workload"]["n_gen"],
                    profile["model"]["generated_tokens"],
                ),
                "plan.workload.context": (
                    plan["workload"]["driver_context_per_sequence"],
                    profile["model"]["context_tokens"],
                ),
                "plan.workload.chat": (plan["workload"]["chat"], True),
                "plan.workload.sampling": (
                    plan["workload"]["sampling"],
                    "greedy_argmax",
                ),
                "plan.workload.prompt_sha256": (
                    sha256_bytes(plan["workload"]["prompt"].encode("ascii")),
                    model["prompt_sha256"],
                ),
                "plan.host.model_id": (
                    Path(plan["host"]["model"]).stem.lower(),
                    model["model_id"],
                ),
                "plan.host.model_bytes": (
                    plan["host"]["model_bytes"],
                    functional["artifacts"]["host_model"]["bytes"],
                ),
                "plan.phone.backend": (
                    plan["phones"]["backend"],
                    profile["route"]["phone_backend"],
                ),
                "plan.phone.layer_range": (
                    plan["phones"]["op15"]["layer_range"],
                    profile["route"]["phone_layer_range"],
                ),
                "plan.phone.weight_bytes": (
                    plan["phones"]["op15"]["model"]["bytes"],
                    model["phone_weight_bytes"],
                ),
                "summary.schema": (summary["schema"], "s11-fixed-route-poc-result-v1"),
                "summary.treatment_route": (
                    summary["treatment_route"],
                    profile["route"]["phone"],
                ),
                "summary.plan_sha256": (summary["plan_sha256"], plan["plan_sha256"]),
                "summary.aggregate.exact": (summary["aggregate"]["exact_work_all_pairs"], True),
                "summary.aggregate.measurement": (
                    summary["aggregate"]["measurement_status"],
                    "NOT_RUN",
                ),
                "pair.exact_work": (pair["exact_work"], row["exact_work"]),
                "pair.control_hbm": (
                    pair["control_gpu_ready_memory_mib"],
                    row["control_ready_hbm_mib"],
                ),
                "pair.phone_hbm": (
                    pair["treatment_gpu_ready_memory_mib"],
                    row["phone_route_ready_hbm_mib"],
                ),
                "pair.hbm_relief": (
                    pair["gpu_ready_memory_relief_mib"],
                    row["server_hbm_relief_mib"],
                ),
                "control.batch_size": (control["batch_size"], row["batch_size"]),
                "control.group_count": (control["group_count"], 1),
                "control.group_us": (control["median_group_wall_us"], row["server_group_us"]),
                "control.request_wall_us": (
                    control_record["request_wall_us"],
                    row["server_group_us"],
                ),
                "treatment.batch_size": (treatment["batch_size"], row["batch_size"]),
                "treatment.group_count": (treatment["group_count"], 1),
                "treatment.group_us": (
                    treatment["median_group_wall_us"],
                    row["phone_route_group_us"],
                ),
                "treatment.request_wall_us": (
                    treatment_record["request_wall_us"],
                    row["phone_route_group_us"],
                ),
                "treatment.phone_stage_us": (
                    treatment_record["stage_a_us"],
                    row["phone_stage_us"],
                ),
                "treatment.server_tail_us": (
                    treatment_record["host_us"],
                    row["server_tail_us"],
                ),
            }
        except (KeyError, TypeError, IndexError, UnicodeError) as exc:
            raise S12Error(f"profile.rows[{index}]: malformed evidence artifact") from exc
        for name, (actual, expected) in checks.items():
            _same_evidence_value(f"rows[{index}].{name}", actual, expected)


def _merged(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for start, finish in sorted(intervals):
        if finish <= start:
            continue
        if not out or start >= out[-1][1]:
            out.append([start, finish])
        else:
            out[-1][1] = max(out[-1][1], finish)
    return [(start, finish) for start, finish in out]


def overlap_us(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> int:
    lhs = _merged(left)
    rhs = _merged(right)
    i = 0
    j = 0
    total = 0
    while i < len(lhs) and j < len(rhs):
        total += max(0, min(lhs[i][1], rhs[j][1]) - max(lhs[i][0], rhs[j][0]))
        if lhs[i][1] <= rhs[j][1]:
            i += 1
        else:
            j += 1
    return total


class DualPathReplay:
    def __init__(
        self,
        policy: str,
        cfg: dict[str, Any],
        coverage: dict[str, Any],
        rows: dict[int, dict[str, Any]],
        model: dict[str, Any],
    ):
        if policy not in DUAL_PATH_POLICIES[1:]:
            raise S12Error("dual-path replay requires a causal runtime policy")
        self.policy = policy
        self.cfg = cfg
        self.coverage = coverage
        self.rows = rows
        self.model = model
        self.supported = tuple(sorted(rows))
        self.host_residency_mode = "TAIL_ONLY" if policy == "fixed_phone" else "FULL_MODEL"
        hbm_field = (
            "phone_route_ready_hbm_mib"
            if self.host_residency_mode == "TAIL_ONLY"
            else "control_ready_hbm_mib"
        )
        self.host_resident_hbm_mib = max(row[hbm_field] for row in rows.values())
        self.tail_vs_full_residency_delta_mib = (
            max(row["control_ready_hbm_mib"] for row in rows.values())
            - max(row["phone_route_ready_hbm_mib"] for row in rows.values())
        )
        self.host_resident_hbm_relief_mib = (
            self.tail_vs_full_residency_delta_mib
            if self.host_residency_mode == "TAIL_ONLY"
            else 0
        )
        if (
            self.host_resident_hbm_mib
            + max(point["used_mib"] for point in cfg["background_hbm_mib_timeline"])
            > cfg["server_hbm_capacity_mib"]
        ):
            raise S12Error("static host residency exceeds A6000 HBM capacity")
        self.now = 0
        self.sequence = 0
        self.heap: list[tuple[int, int, int, str, Any]] = []
        self.requests = [
            RequestState(
                event_id=req["event_id"],
                arrival_us=req["arrival_us"],
                claim_scope=req["claim_scope"],
                profile_eligible=req["profile_eligible"],
            )
            for req in coverage["prepared_requests"]
        ]
        if any(req.arrival_us > cfg["horizon_us"] for req in self.requests):
            raise S12Error("trace arrival exceeds replay horizon")
        self.by_id = {req.event_id: req for req in self.requests}
        self.queue: list[RequestState] = []
        self.groups: dict[str, PhoneGroup] = {}
        self.wifi_wait: list[str] = []
        self.compute_wait: list[str] = []
        self.usb_wait: list[str] = []
        self.tail_wait: list[str] = []
        self.wifi_busy: str | None = None
        self.compute_busy: str | None = None
        self.usb_busy: str | None = None
        self.server_busy: tuple[str, str] | None = None
        self.phone_inflight_groups = 0
        self.ingress_buffer_bytes = 0
        self.phone_result_buffer_bytes = 0
        self.host_result_buffer_bytes = 0
        self.peak_ingress_buffer_bytes = 0
        self.peak_phone_result_buffer_bytes = 0
        self.peak_host_result_buffer_bytes = 0
        self.wifi_bytes_completed = 0
        self.usb_bytes_completed = 0
        self.wifi_intervals: list[tuple[int, int]] = []
        self.compute_intervals: list[tuple[int, int]] = []
        self.usb_intervals: list[tuple[int, int]] = []
        self.server_intervals: list[tuple[int, int]] = []
        self.server_route_batches = 0
        self.server_decisions: list[dict[str, Any]] = []
        self.phone_route_batches = 0
        self.queue_depth_max = 0
        self.peak_a6000_hbm_mib = self.host_resident_hbm_mib + max(
            point["used_mib"] for point in cfg["background_hbm_mib_timeline"]
        )
        for point in cfg["background_hbm_mib_timeline"]:
            self.push(point["t_us"], PR_HBM, "hbm_change", point)
        for req in self.requests:
            self.push(req.arrival_us, PR_ARRIVAL, "arrival", req.event_id)

    @property
    def transport(self) -> dict[str, Any]:
        return self.cfg["transport"]

    def push(self, t_us: int, priority: int, kind: str, payload: Any) -> None:
        require_int("event timestamp", t_us)
        heapq.heappush(self.heap, (t_us, priority, self.sequence, kind, payload))
        self.sequence += 1

    def available_hbm_mib(self) -> int:
        return self.cfg["server_hbm_capacity_mib"] - background_hbm_at(
            self.now,
            self.cfg["background_hbm_mib_timeline"],
        )

    def _residency_fits_now(self) -> bool:
        return self.host_resident_hbm_mib <= self.available_hbm_mib()

    def _finish(self, req: RequestState, terminal: str) -> None:
        if terminal not in TERMINALS:
            raise AssertionError(f"unknown terminal {terminal}")
        if req.terminal is not None:
            raise AssertionError(f"request {req.event_id} terminalized twice")
        req.terminal = terminal
        req.finish_us = self.now

    def on_arrival(self, request_id: str) -> None:
        req = self.by_id[request_id]
        if not req.profile_eligible:
            self._finish(req, "unprofiled")
            return
        limit = (
            self.cfg["phone_queue_limit"]
            if self.policy == "fixed_phone"
            else self.cfg["server_queue_limit"]
        )
        if len(self.queue) >= limit:
            self._finish(req, "rejected_queue_full")
            return
        self.queue.append(req)
        self.queue_depth_max = max(self.queue_depth_max, len(self.queue))
        self.push(req.arrival_us + self.cfg["batch_hold_us"], PR_FLUSH, "flush", req.event_id)

    def _oldest_ready(self) -> bool:
        if not self.queue:
            return False
        return (
            self.now >= self.queue[0].arrival_us + self.cfg["batch_hold_us"]
            or len(self.queue) >= self.supported[-1]
        )

    def _candidate_batches(self, requested: int) -> list[int]:
        return list(reversed([value for value in self.supported if value <= requested]))

    def _server_executable(self, batch: int, available_hbm_mib: int) -> bool:
        if self.host_residency_mode != "FULL_MODEL":
            return False
        row = self.rows[batch]
        return (
            self.host_resident_hbm_mib <= available_hbm_mib
            and self._can_finish_before_horizon(row["server_group_us"])
        )

    def _phone_route_parts(self, batch: int) -> dict[str, list[int]]:
        row = self.rows[batch]
        wifi = s11_head_wifi_input_bytes_by_step(batch, self.model)
        usb = s11_head_usb_result_bytes_by_step(batch, self.model)
        if sum(usb) != s11_head_usb_result_bytes(row, self.model):
            raise AssertionError("directional USB bytes do not conserve the profile activation")
        return {
            "compute": split_phase_us(row["phone_stage_us"], self.model),
            "tail": split_phase_us(row["server_tail_us"], self.model),
            "usb": usb,
            "wifi": wifi,
        }

    def _phone_executable(self, batch: int, available_hbm_mib: int) -> bool:
        if self.host_residency_mode != "TAIL_ONLY":
            return False
        if self.phone_inflight_groups >= self.transport["phone_inflight_group_limit"]:
            return False
        if self.host_resident_hbm_mib > available_hbm_mib:
            return False
        parts = self._phone_route_parts(batch)
        if (
            max(parts["wifi"]) > self.transport["phone_ingress_buffer_bytes"]
            or max(parts["usb"]) > self.transport["phone_result_buffer_bytes"]
            or max(parts["usb"]) > self.transport["host_result_buffer_bytes"]
        ):
            return False
        lower_bound = sum(parts["compute"]) + sum(parts["tail"])
        lower_bound += sum(
            transfer_us(value, self.transport["wifi_h2p"]) for value in parts["wifi"]
        )
        lower_bound += sum(
            transfer_us(value, self.transport["usb_p2h"]) for value in parts["usb"]
        )
        return self._can_finish_before_horizon(lower_bound)

    def _route_choice(self) -> tuple[str, int, bool, str] | None:
        requested = causal_batch_size(len(self.queue), self._oldest_ready(), self.supported)
        if requested is None:
            return None
        available = self.available_hbm_mib()
        candidates = self._candidate_batches(requested)
        if self.policy == "causal_server_batch":
            for batch in candidates:
                if self._server_executable(batch, available):
                    return "SERVER_ONLY", batch, True, "CAUSAL_SERVER_BATCH"
            return None
        paths_ready = self.cfg["phone_residency"]["ready"]
        if self.policy == "fixed_phone":
            if not paths_ready:
                return None
            for batch in candidates:
                if self._phone_executable(batch, available):
                    server_feasible = self._server_executable(batch, available)
                    return "A0_OP15", batch, server_feasible, "FIXED_PHONE"
            return None
        raise AssertionError(f"unsupported dual-path policy {self.policy}")

    def _can_finish_before_horizon(self, duration_us: int) -> bool:
        return self.now + duration_us <= self.cfg["horizon_us"]

    def _start_wifi(self) -> bool:
        if self.wifi_busy is not None or not self.wifi_wait:
            return False
        if self.compute_busy is not None:
            return False
        if not self.transport["wifi_usb_independent"] and self.usb_busy is not None:
            return False
        group = self.groups[self.wifi_wait[0]]
        if self.ingress_buffer_bytes + group.wifi_input_bytes > self.transport["phone_ingress_buffer_bytes"]:
            return False
        duration = transfer_us(group.wifi_input_bytes, self.transport["wifi_h2p"])
        if not self._can_finish_before_horizon(duration):
            return False
        self.wifi_wait.pop(0)
        self.wifi_busy = group.group_id
        self.ingress_buffer_bytes += group.wifi_input_bytes
        self.peak_ingress_buffer_bytes = max(self.peak_ingress_buffer_bytes, self.ingress_buffer_bytes)
        finish = self.now + duration
        group.phase_us["wifi_start"] = self.now
        group.phase_us["wifi_finish"] = finish
        self.wifi_intervals.append((self.now, finish))
        self.push(finish, PR_COMPLETE, "wifi_done", group.group_id)
        return True

    def _start_compute(self) -> bool:
        if self.compute_busy is not None or not self.compute_wait:
            return False
        if self.wifi_busy is not None or self.usb_busy is not None:
            return False
        group = self.groups[self.compute_wait[0]]
        if (
            self.phone_result_buffer_bytes + group.usb_result_bytes
            > self.transport["phone_result_buffer_bytes"]
        ):
            return False
        duration = group.compute_us
        if not self._can_finish_before_horizon(duration):
            return False
        self.compute_wait.pop(0)
        self.compute_busy = group.group_id
        self.ingress_buffer_bytes -= group.wifi_input_bytes
        self.phone_result_buffer_bytes += group.usb_result_bytes
        self.peak_phone_result_buffer_bytes = max(
            self.peak_phone_result_buffer_bytes,
            self.phone_result_buffer_bytes,
        )
        finish = self.now + duration
        group.phase_us["compute_start"] = self.now
        group.phase_us["compute_finish"] = finish
        self.compute_intervals.append((self.now, finish))
        self.push(finish, PR_COMPLETE, "compute_done", group.group_id)
        return True

    def _start_usb(self) -> bool:
        if self.usb_busy is not None or not self.usb_wait:
            return False
        if self.compute_busy is not None:
            return False
        if not self.transport["wifi_usb_independent"] and self.wifi_busy is not None:
            return False
        group = self.groups[self.usb_wait[0]]
        if self.host_result_buffer_bytes + group.usb_result_bytes > self.transport["host_result_buffer_bytes"]:
            return False
        duration = transfer_us(group.usb_result_bytes, self.transport["usb_p2h"])
        if not self._can_finish_before_horizon(duration):
            return False
        self.usb_wait.pop(0)
        self.usb_busy = group.group_id
        self.host_result_buffer_bytes += group.usb_result_bytes
        self.peak_host_result_buffer_bytes = max(
            self.peak_host_result_buffer_bytes,
            self.host_result_buffer_bytes,
        )
        finish = self.now + duration
        group.phase_us["usb_start"] = self.now
        group.phase_us["usb_finish"] = finish
        self.usb_intervals.append((self.now, finish))
        self.push(finish, PR_COMPLETE, "usb_done", group.group_id)
        return True

    def _start_tail(self) -> bool:
        if self.server_busy is not None or not self.tail_wait:
            return False
        if self.host_residency_mode != "TAIL_ONLY":
            raise AssertionError("phone tail attempted without tail-only host residency")
        group = self.groups[self.tail_wait[0]]
        if not self._residency_fits_now():
            return False
        duration = group.tail_us
        if not self._can_finish_before_horizon(duration):
            return False
        self.tail_wait.pop(0)
        self.server_busy = ("phone_tail", group.group_id)
        finish = self.now + duration
        group.phase_us["tail_start"] = self.now
        group.phase_us["tail_finish"] = finish
        self.server_intervals.append((self.now, finish))
        self.peak_a6000_hbm_mib = max(
            self.peak_a6000_hbm_mib,
            background_hbm_at(self.now, self.cfg["background_hbm_mib_timeline"])
            + self.host_resident_hbm_mib,
        )
        self.push(finish, PR_COMPLETE, "server_done", {"kind": "phone_tail", "id": group.group_id})
        return True

    def _start_server_group(self, batch: int, reason: str) -> bool:
        if self.host_residency_mode != "FULL_MODEL":
            raise AssertionError("server route attempted without full-model host residency")
        if self.server_busy is not None or self.tail_wait:
            return False
        row = self.rows[batch]
        if not self._residency_fits_now():
            return False
        duration = row["server_group_us"]
        if not self._can_finish_before_horizon(duration):
            return False
        selected = self.queue[:batch]
        del self.queue[:batch]
        request_ids = []
        for req in selected:
            req.start_us = self.now
            req.route = "SERVER_ONLY"
            request_ids.append(req.event_id)
        batch_id = f"server-{self.server_route_batches:06d}"
        self.server_route_batches += 1
        self.server_busy = ("server", batch_id)
        finish = self.now + duration
        self.server_intervals.append((self.now, finish))
        self.peak_a6000_hbm_mib = max(
            self.peak_a6000_hbm_mib,
            background_hbm_at(self.now, self.cfg["background_hbm_mib_timeline"])
            + self.host_resident_hbm_mib,
        )
        self.push(
            finish,
            PR_COMPLETE,
            "server_done",
            {"id": batch_id, "kind": "server", "request_ids": request_ids},
        )
        self.server_decisions.append(
            {
                "batch_size": batch,
                "finish_us": finish,
                "group_id": batch_id,
                "reason": reason,
                "request_ids": request_ids,
                "route": "SERVER_ONLY",
                "server_admission_feasible": True,
                "start_us": self.now,
            }
        )
        return True

    def _admit_phone_group(
        self,
        batch: int,
        server_feasible: bool,
        reason: str,
    ) -> bool:
        if self.phone_inflight_groups >= self.transport["phone_inflight_group_limit"]:
            return False
        row = self.rows[batch]
        if not self._phone_executable(batch, self.available_hbm_mib()):
            return False
        parts = self._phone_route_parts(batch)
        selected = self.queue[:batch]
        del self.queue[:batch]
        group_id = f"phone-{self.phone_route_batches:06d}"
        group = PhoneGroup(
            group_id=group_id,
            request_ids=[req.event_id for req in selected],
            batch_size=batch,
            row=row,
            admitted_us=self.now,
            reason=reason,
            server_admission_feasible=server_feasible,
            wifi_input_bytes_by_step=parts["wifi"],
            usb_result_bytes_by_step=parts["usb"],
            compute_us_by_step=parts["compute"],
            tail_us_by_step=parts["tail"],
            phase_us_by_step=new_phase_records(len(parts["wifi"])),
        )
        for req in selected:
            req.start_us = self.now
            req.route = "A0_OP15"
        self.groups[group_id] = group
        self.wifi_wait.append(group_id)
        self.phone_inflight_groups += 1
        self.phone_route_batches += 1
        return True

    def try_schedule(self) -> None:
        while True:
            changed = False
            changed = self._start_tail() or changed
            changed = self._start_usb() or changed
            changed = self._start_compute() or changed
            changed = self._start_wifi() or changed

            if self.queue:
                choice = self._route_choice()
                if choice is not None:
                    route, batch, server_feasible, reason = choice
                    if route == "SERVER_ONLY":
                        changed = self._start_server_group(batch, reason) or changed
                    else:
                        changed = self._admit_phone_group(batch, server_feasible, reason) or changed
            if not changed:
                return

    def _on_wifi_done(self, group_id: str) -> None:
        if self.wifi_busy != group_id:
            raise AssertionError("WiFi completion does not own WiFi lane")
        self.wifi_busy = None
        self.wifi_bytes_completed += self.groups[group_id].wifi_input_bytes
        self.compute_wait.append(group_id)

    def _on_compute_done(self, group_id: str) -> None:
        if self.compute_busy != group_id:
            raise AssertionError("compute completion does not own phone lane")
        self.compute_busy = None
        self.usb_wait.append(group_id)

    def _on_usb_done(self, group_id: str) -> None:
        if self.usb_busy != group_id:
            raise AssertionError("USB completion does not own USB lane")
        self.usb_busy = None
        group = self.groups[group_id]
        self.phone_result_buffer_bytes -= group.usb_result_bytes
        self.usb_bytes_completed += group.usb_result_bytes
        self.tail_wait.append(group_id)

    def _on_server_done(self, payload: dict[str, Any]) -> None:
        if self.server_busy != (payload["kind"], payload["id"]):
            raise AssertionError("server completion does not own A6000 lane")
        self.server_busy = None
        if payload["kind"] == "server":
            for request_id in payload["request_ids"]:
                self._finish(self.by_id[request_id], "completed_server")
            return
        group = self.groups[payload["id"]]
        self.host_result_buffer_bytes -= group.usb_result_bytes
        if group.step_index + 1 < len(group.phase_us_by_step):
            group.step_index += 1
            self.wifi_wait.append(group.group_id)
            return
        group.complete = True
        self.phone_inflight_groups -= 1
        for request_id in group.request_ids:
            self._finish(self.by_id[request_id], "completed_phone")

    def _check_invariants(self) -> None:
        limits = self.transport
        values = (
            ("phone ingress", self.ingress_buffer_bytes, limits["phone_ingress_buffer_bytes"]),
            ("phone result", self.phone_result_buffer_bytes, limits["phone_result_buffer_bytes"]),
            ("host result", self.host_result_buffer_bytes, limits["host_result_buffer_bytes"]),
        )
        for name, used, limit in values:
            if not 0 <= used <= limit:
                raise AssertionError(f"{name} buffer ledger out of range")
        if not 0 <= self.phone_inflight_groups <= limits["phone_inflight_group_limit"]:
            raise AssertionError("phone in-flight group ledger out of range")
        if self.compute_busy is not None and (self.wifi_busy is not None or self.usb_busy is not None):
            raise AssertionError("unmeasured compute/link overlap occurred")
        total = (
            background_hbm_at(self.now, self.cfg["background_hbm_mib_timeline"])
            + self.host_resident_hbm_mib
        )
        if total > self.cfg["server_hbm_capacity_mib"]:
            raise S12Error("background HBM change oversubscribed static host residency")
        if len(self.queue) > max(self.cfg["server_queue_limit"], self.cfg["phone_queue_limit"]):
            raise AssertionError("host queue bound exceeded")

    def run(self) -> dict[str, Any]:
        while self.heap:
            t_us = self.heap[0][0]
            if t_us > self.cfg["horizon_us"]:
                break
            self.now = t_us
            while self.heap and self.heap[0][0] == t_us:
                _, _, _, kind, payload = heapq.heappop(self.heap)
                if kind == "arrival":
                    self.on_arrival(payload)
                elif kind == "wifi_done":
                    self._on_wifi_done(payload)
                elif kind == "compute_done":
                    self._on_compute_done(payload)
                elif kind == "usb_done":
                    self._on_usb_done(payload)
                elif kind == "server_done":
                    self._on_server_done(payload)
                elif kind == "hbm_change":
                    self.peak_a6000_hbm_mib = max(
                        self.peak_a6000_hbm_mib,
                        payload["used_mib"] + self.host_resident_hbm_mib,
                    )
                elif kind == "flush":
                    pass
                else:
                    raise AssertionError(f"unknown event {kind}")
                self._check_invariants()
            self.try_schedule()
            self._check_invariants()

        self.now = self.cfg["horizon_us"]
        for req in self.requests:
            if req.terminal is None:
                self._finish(req, "timed_out")
        discarded = sum(1 for group in self.groups.values() if not group.complete)
        if discarded != self.phone_inflight_groups:
            raise AssertionError("discarded group count does not match in-flight ledger")
        released_at_horizon = {
            "compute_wait_groups": len(self.compute_wait),
            "host_result_buffer_bytes": self.host_result_buffer_bytes,
            "host_queued_requests": len(self.queue),
            "phone_inflight_groups": self.phone_inflight_groups,
            "phone_ingress_buffer_bytes": self.ingress_buffer_bytes,
            "phone_result_buffer_bytes": self.phone_result_buffer_bytes,
            "tail_wait_groups": len(self.tail_wait),
            "usb_wait_groups": len(self.usb_wait),
            "wifi_wait_groups": len(self.wifi_wait),
        }
        self.queue.clear()
        self.wifi_wait.clear()
        self.compute_wait.clear()
        self.usb_wait.clear()
        self.tail_wait.clear()
        self.wifi_busy = None
        self.compute_busy = None
        self.usb_busy = None
        self.server_busy = None
        self.ingress_buffer_bytes = 0
        self.phone_result_buffer_bytes = 0
        self.host_result_buffer_bytes = 0
        self.phone_inflight_groups = 0
        self._check_invariants()
        return self._build_result(discarded, released_at_horizon)

    def _build_result(
        self,
        discarded_inflight_groups: int,
        released_at_horizon: dict[str, int],
    ) -> dict[str, Any]:
        counts = {terminal: 0 for terminal in TERMINALS}
        queue_delays = []
        completion_latencies = []
        outcomes = []
        for req in self.requests:
            if req.terminal is None:
                raise AssertionError(f"request {req.event_id} has no terminal")
            counts[req.terminal] += 1
            if req.start_us is not None:
                queue_delays.append(req.start_us - req.arrival_us)
            if req.finish_us is not None and req.terminal.startswith("completed_"):
                completion_latencies.append(req.finish_us - req.arrival_us)
            outcomes.append(
                {
                    "arrival_us": req.arrival_us,
                    "event_id": req.event_id,
                    "finish_us": req.finish_us,
                    "route": req.route,
                    "start_us": req.start_us,
                    "terminal": req.terminal,
                }
            )
        if sum(counts.values()) != len(self.requests):
            raise AssertionError("terminal conservation failed")

        decisions = list(self.server_decisions)
        for group in self.groups.values():
            phases = [dict(record) for record in group.phase_us_by_step]
            if group.complete:
                previous_tail = None
                for step in phases:
                    ordered = [step[key] for key in (
                        "wifi_start", "wifi_finish", "compute_start", "compute_finish",
                        "usb_start", "usb_finish", "tail_start", "tail_finish",
                    )]
                    if any(value is None for value in ordered) or ordered != sorted(ordered):
                        raise AssertionError("completed phone group has broken phase causality")
                    if previous_tail is not None and step["wifi_start"] < previous_tail:
                        raise AssertionError("decode input became ready before its preceding tail")
                    previous_tail = step["tail_finish"]
            decisions.append(
                {
                    "batch_size": group.batch_size,
                    "group_id": group.group_id,
                    "phase_us_by_step": phases,
                    "reason": group.reason,
                    "request_ids": group.request_ids,
                    "route": "A0_OP15",
                    "server_admission_feasible": group.server_admission_feasible,
                    "profile_row_hbm_delta_mib": group.row["server_hbm_relief_mib"],
                    "usb_result_bytes": sum(group.usb_result_bytes_by_step),
                    "usb_result_bytes_by_step": group.usb_result_bytes_by_step,
                    "wifi_input_bytes": sum(group.wifi_input_bytes_by_step),
                    "wifi_input_bytes_by_step": group.wifi_input_bytes_by_step,
                }
            )
        decisions.sort(
            key=lambda decision: (
                decision.get(
                    "start_us",
                    decision.get("phase_us_by_step", [{}])[0].get("wifi_start"),
                )
                if decision.get(
                    "start_us",
                    decision.get("phase_us_by_step", [{}])[0].get("wifi_start"),
                ) is not None
                else self.cfg["horizon_us"],
                decision["group_id"],
            )
        )
        routes = {decision["route"] for decision in decisions}
        expected_route = "A0_OP15" if self.host_residency_mode == "TAIL_ONLY" else "SERVER_ONLY"
        if routes - {expected_route}:
            raise AssertionError("result mixes routes across static host residencies")
        completed = counts["completed_server"] + counts["completed_phone"]
        completed_requests = [
            req for req in self.requests if req.terminal is not None and req.terminal.startswith("completed_")
        ]
        makespan = (
            max(req.finish_us for req in completed_requests if req.finish_us is not None)
            - min(req.arrival_us for req in completed_requests)
            if completed_requests
            else None
        )
        phone_active = self.wifi_intervals + self.compute_intervals + self.usb_intervals
        useful_wifi_bytes = sum(
            sum(group.wifi_input_bytes_by_step) for group in self.groups.values() if group.complete
        )
        useful_usb_bytes = sum(
            sum(group.usb_result_bytes_by_step) for group in self.groups.values() if group.complete
        )
        if useful_wifi_bytes > self.wifi_bytes_completed or useful_usb_bytes > self.usb_bytes_completed:
            raise AssertionError("useful path bytes exceed completed transfer ledger")
        result = {
            "claim_scope": DUAL_PATH_CLAIM_SCOPE,
            "clairvoyant": False,
            "completed_generated_tokens": completed * self.model["generated_tokens"],
            "completed_phone": counts["completed_phone"],
            "completed_prompt_tokens": completed * self.model["prompt_tokens"],
            "completed_requests": completed,
            "completed_server": counts["completed_server"],
            "completion_latency_p50_us": nearest_rank(completion_latencies, 1, 2),
            "completion_latency_p95_us": nearest_rank(completion_latencies, 19, 20),
            "compute_link_overlap_policy": self.transport["compute_link_overlap"],
            "decisions": decisions,
            "discarded_inflight_groups": discarded_inflight_groups,
            "dual_path_overlap_us": overlap_us(self.wifi_intervals, self.usb_intervals),
            "energy": {
                "phone_nj": None,
                "server_gpu_board_nj": None,
                "status": "NOT_RUN",
                "total_wall_nj": None,
            },
            "host_result_buffer_peak_bytes": self.peak_host_result_buffer_bytes,
            "host_residency_mode": self.host_residency_mode,
            "host_residency_transition_status": "NONE_STATIC_FOR_REPLAY",
            "host_resident_hbm_mib": self.host_resident_hbm_mib,
            "host_resident_hbm_relief_mib": self.host_resident_hbm_relief_mib,
            "horizon_release": released_at_horizon,
            "makespan_us": makespan,
            "outcomes": outcomes,
            "path_independence_status": (
                "TOPOLOGY_ASSUMPTION_UNMEASURED"
                if self.transport["wifi_usb_independent"]
                else "SERIALIZED_ABLATION_CONTROL"
            ),
            "peak_a6000_hbm_bytes": self.peak_a6000_hbm_mib * MIB,
            "peak_a6000_hbm_mib": self.peak_a6000_hbm_mib,
            "phase_time_status": "PHONE_STAGE_WALL_PROXY_NOT_DECOMPOSED",
            "phase_proxy_split": "ACTIVATION_ROW_PROPORTIONAL_PRESERVES_AGGREGATE",
            "phone_quanta_per_group": self.model["generated_tokens"],
            "phone_inflight_group_limit": self.transport["phone_inflight_group_limit"],
            "phone_ingress_buffer_peak_bytes": self.peak_ingress_buffer_bytes,
            "phone_result_buffer_peak_bytes": self.peak_phone_result_buffer_bytes,
            "phone_route_batches": self.phone_route_batches,
            "policy": self.policy,
            "profiled_requests": self.coverage["profiled_requests"],
            "queue_delay_p50_us": nearest_rank(queue_delays, 1, 2),
            "queue_delay_p95_us": nearest_rank(queue_delays, 19, 20),
            "queue_depth_max": self.queue_depth_max,
            "rejected_queue_full": counts["rejected_queue_full"],
            "requests": len(self.requests),
            "schema": "s12-dual-path-policy-result-v1",
            "server_phone_overlap_us": overlap_us(self.server_intervals, phone_active),
            "server_route_batches": self.server_route_batches,
            "terminal_conservation": sum(counts.values()),
            "tail_vs_full_residency_delta_mib": self.tail_vs_full_residency_delta_mib,
            "timed_out": counts["timed_out"],
            "trace_coverage_scope": self.coverage["claim_scope"],
            "transport_scope": self.transport["scope"],
            "unprofiled": counts["unprofiled"],
            "usb_p2h": {
                "busy_us": sum(finish - start for start, finish in self.usb_intervals),
                "domain_id": self.transport["usb_p2h"]["domain_id"],
                "path_id": self.transport["usb_p2h"]["path_id"],
                "payload_bytes_completed": self.usb_bytes_completed,
                "payload_bytes_useful": useful_usb_bytes,
                "payload_bytes_wasted": self.usb_bytes_completed - useful_usb_bytes,
            },
            "wifi_h2p": {
                "busy_us": sum(finish - start for start, finish in self.wifi_intervals),
                "domain_id": self.transport["wifi_h2p"]["domain_id"],
                "path_id": self.transport["wifi_h2p"]["path_id"],
                "payload_bytes_completed": self.wifi_bytes_completed,
                "payload_bytes_useful": useful_wifi_bytes,
                "payload_bytes_wasted": self.wifi_bytes_completed - useful_wifi_bytes,
            },
            "wifi_usb_independent": self.transport["wifi_usb_independent"],
        }
        result["result_digest"] = sha256_object(result)
        return result


def _adapt_offline_result(
    result: dict[str, Any],
    cfg: dict[str, Any],
    rows: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    result = dict(result)
    result.pop("result_digest", None)
    for key in (
        "activation_bytes_total",
        "activation_formula",
        "completed_throughput",
        "is_offline_upper_bound",
        "memory_phone_selections",
        "memory_server_infeasible_events",
        "phone_residency_ready",
        "phone_resident_weight_bytes",
        "phone_route_relief_mib_observations",
        "rejected_requests",
        "server_hbm_capacity_bytes",
        "server_hbm_capacity_mib",
        "upper_bound_scope",
    ):
        result.pop(key, None)
    trace_coverage_scope = result["claim_scope"]
    resident_hbm_mib = max(row["control_ready_hbm_mib"] for row in rows.values())
    resident_relief_mib = resident_hbm_mib - max(
        row["phone_route_ready_hbm_mib"] for row in rows.values()
    )
    peak_hbm_mib = resident_hbm_mib + max(
        point["used_mib"] for point in cfg["background_hbm_mib_timeline"]
    )
    if peak_hbm_mib > cfg["server_hbm_capacity_mib"]:
        raise S12Error("static host residency exceeds A6000 HBM capacity")
    result.update(
        {
            "claim_scope": DUAL_PATH_CLAIM_SCOPE,
            "compute_link_overlap_policy": cfg["transport"]["compute_link_overlap"],
            "discarded_inflight_groups": 0,
            "dual_path_overlap_us": 0,
            "host_result_buffer_peak_bytes": 0,
            "host_residency_mode": "FULL_MODEL",
            "host_residency_transition_status": "NONE_STATIC_FOR_REPLAY",
            "host_resident_hbm_mib": resident_hbm_mib,
            "host_resident_hbm_relief_mib": 0,
            "horizon_release": {
                "compute_wait_groups": 0,
                "host_result_buffer_bytes": 0,
                "host_queued_requests": 0,
                "phone_inflight_groups": 0,
                "phone_ingress_buffer_bytes": 0,
                "phone_result_buffer_bytes": 0,
                "tail_wait_groups": 0,
                "usb_wait_groups": 0,
                "wifi_wait_groups": 0,
            },
            "path_independence_status": (
                "TOPOLOGY_ASSUMPTION_UNMEASURED"
                if cfg["transport"]["wifi_usb_independent"]
                else "SERIALIZED_ABLATION_CONTROL"
            ),
            "phase_time_status": "PHONE_STAGE_WALL_PROXY_NOT_DECOMPOSED",
            "phase_proxy_split": "ACTIVATION_ROW_PROPORTIONAL_PRESERVES_AGGREGATE",
            "peak_a6000_hbm_bytes": peak_hbm_mib * MIB,
            "peak_a6000_hbm_mib": peak_hbm_mib,
            "phone_quanta_per_group": 0,
            "phone_inflight_group_limit": cfg["transport"]["phone_inflight_group_limit"],
            "phone_ingress_buffer_peak_bytes": 0,
            "phone_result_buffer_peak_bytes": 0,
            "schema": "s12-dual-path-policy-result-v1",
            "server_phone_overlap_us": 0,
            "tail_vs_full_residency_delta_mib": resident_relief_mib,
            "transport_scope": cfg["transport"]["scope"],
            "trace_coverage_scope": trace_coverage_scope,
            "usb_p2h": {
                "busy_us": 0,
                "domain_id": cfg["transport"]["usb_p2h"]["domain_id"],
                "path_id": cfg["transport"]["usb_p2h"]["path_id"],
                "payload_bytes_completed": 0,
                "payload_bytes_useful": 0,
                "payload_bytes_wasted": 0,
            },
            "wifi_h2p": {
                "busy_us": 0,
                "domain_id": cfg["transport"]["wifi_h2p"]["domain_id"],
                "path_id": cfg["transport"]["wifi_h2p"]["path_id"],
                "payload_bytes_completed": 0,
                "payload_bytes_useful": 0,
                "payload_bytes_wasted": 0,
            },
            "wifi_usb_independent": cfg["transport"]["wifi_usb_independent"],
        }
    )
    result["result_digest"] = sha256_object(result)
    return result


def run_dual_path_config(
    config_path: str | Path,
    selected_policies: list[str] | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    cfg = validate_dual_path_config(load_json(config_file))
    profile_file = _resolve(config_file, cfg["profile_path"])
    trace_file = _resolve(config_file, cfg["trace_path"])
    profile = validate_profile(
        load_json(profile_file),
        base_dir=REPO_ROOT,
        verify_artifacts=True,
    )
    validate_dual_profile_evidence(profile, REPO_ROOT)
    if cfg["phone_residency"]["weight_bytes"] != profile["model"]["phone_weight_bytes"]:
        raise S12Error("config.phone_residency.weight_bytes does not match profile")
    trace_records, trace_digest = read_jsonl_snapshot(trace_file)
    coverage = prepare_trace(trace_records, profile, cfg["trace_mode"])
    if any(
        request["arrival_us"] > cfg["horizon_us"]
        for request in coverage["prepared_requests"]
    ):
        raise S12Error("trace arrival exceeds replay horizon")
    rows = profile_rows_by_batch(profile)
    policies = cfg["policies"] if selected_policies is None else selected_policies
    if not policies or len(policies) != len(set(policies)):
        raise S12Error("selected policies must be non-empty and unique")
    if any(policy not in cfg["policies"] for policy in policies):
        raise S12Error("selected policy is not frozen in config")

    results = []
    for policy in policies:
        if policy == "server_only_optimized":
            results.append(_adapt_offline_result(run_offline(cfg, coverage, rows), cfg, rows))
        else:
            results.append(DualPathReplay(policy, cfg, coverage, rows, profile["model"]).run())
    manifest = {
        "claim_scope": DUAL_PATH_CLAIM_SCOPE,
        "config_digest": sha256_object(cfg),
        "config_path": str(config_file),
        "coverage_digest": coverage["coverage_digest"],
        "energy_status": "NOT_RUN",
        "evidence_artifacts_verified": True,
        "policy_results": results,
        "profile_digest": sha256_object(profile),
        "profile_path": str(profile_file),
        "schema": "s12-dual-path-replay-v1",
        "trace_mode": cfg["trace_mode"],
        "trace_coverage_scope": coverage["claim_scope"],
        "trace_path": str(trace_file),
        "trace_sha256": trace_digest,
        "transport_scope": cfg["transport"]["scope"],
    }
    manifest["deterministic_replay_sha256"] = sha256_object(results)
    return manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out")
    parser.add_argument("--policy", action="append", choices=DUAL_PATH_POLICIES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = run_dual_path_config(args.config, args.policy)
        if args.out:
            write_canonical(args.out, result)
        else:
            print(canonical_json(result))
        return 0
    except (S12Error, AssertionError) as exc:
        print(f"DUAL_PATH_VQ_FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
