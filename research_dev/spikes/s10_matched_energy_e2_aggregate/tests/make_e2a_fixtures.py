#!/usr/bin/env python3
"""Deterministic E2A fixtures. No clock, no randomness, no network.

Builds a complete, internally consistent 8-pair bundle: a PreRunPlan, both anchor
receipts, an AttemptLedger, a RequestSetManifest, and for each of the 16 planned
slots a RealizedTimeline (with its raw + execution artifacts), a LifecycleRecord,
and a RequestOutcomeRecord.

Everything here is SYNTHETIC and says so in every record. Synthetic evidence
exercises mechanics and can never carry a physical result: the gate refuses it
explicitly, and the tests assert that refusal. These fixtures exist to prove the
machinery runs, not to measure anything.

Timeline records reuse E2's frozen raw/execution artifact formats and its
integrator, so a fixture that E2 would reject is not accidentally accepted here.
The one thing not reused is E2's own `timeline()` helper, which pins every window
to [0, window_us]: E2A DERIVES first_executed_role from the realized windows
rather than reading a declared field, so the slots need real, distinct, ordered
windows for that derivation to have anything to bite on.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import e2a_canon as canon      # noqa: E402
import resolver                # noqa: E402
from e2a_e2 import comparator  # noqa: E402
from e2a_e2 import integrator  # noqa: E402

OUT = ROOT / "fixtures"

WORKLOAD = canon.digest({"workload": "s10_e2a_synthetic"})
TRACE = canon.digest({"trace": "s10_e2a_synthetic"})
SLO = canon.digest({"slo_policy": "s10_e2a_synthetic"})
POLICY_CONTROL = canon.digest({"policy": "OPTIMIZED_SERVER_ONLY_CONTROL"})
POLICY_TREATMENT = canon.digest({"policy": "Q_PIM_TREATMENT"})
NORMALIZER = canon.digest({"normalizer": "e2.zoh.left_edge.v1"})
MODEL = canon.digest({"model": "gemma-4-12b-f16-synthetic"})
TOKENIZER = canon.digest({"tokenizer": "gemma-4-synthetic"})
SAMPLER = canon.digest({"sampler": "greedy"})
STOP_SET = canon.digest({"stop": ["<eos>"]})
DECODE_PARAMS = canon.digest({"decode": "greedy-synthetic"})
CLOCK_EPOCH = "boot-synthetic-e2a-0000"
BOARD = "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f"
PHONE = "phone.op15"
USB_LINK = "usb.bus006"
CONTROL_DEVICE = "host.control"
SET_ID = "set.s10.e2a.synthetic"
CHAIN_ID = "chain.s10.e2a.synthetic"
COMMITMENT_NAMESPACE = "lazyllm.s10.e2a.synthetic"

N_PAIRS = 8
WINDOW_US = 20_000_000
SLOT_STRIDE_US = 25_000_000
PERIOD_US = 100_000
CONTROL_MW = 100
TREATMENT_MW = 80
UNCERTAINTY_MW = 1
N_REQUESTS = 3

CONTROL_ROLE = "OPTIMIZED_SERVER_ONLY_CONTROL"
TREATMENT_ROLE = "Q_PIM_TREATMENT"


def route_schedule(man, role):
    request_ids = [entry["request_id"] for entry in man["entries"]]
    treatment = role == TREATMENT_ROLE
    compute_domain = "PHONE" if treatment else "SERVER"
    compute_device = PHONE if treatment else BOARD
    compute_backend = "HTP" if treatment else "CUDA"
    transfer_backend = "USB_BULK" if treatment else "CUDA"
    island = canon.digest({
        "island": "gemma.synthetic.decode",
        "role": role,
    })

    def node(action_id, action_kind, domain, device, backend, requests,
             input_requirement="ZERO", output_requirement="ZERO",
             duration_requirement="POSITIVE", lease_requirement="REQUIRED",
             operator_island_digest=None, model_digest=None):
        return {
            "action_id": action_id,
            "action_kind": action_kind,
            "execution_domain": domain,
            "device_identity": device,
            "backend_kind": backend,
            "operator_island_digest": operator_island_digest,
            "model_digest": model_digest,
            "request_ids": requests,
            "input_requirement": input_requirement,
            "output_requirement": output_requirement,
            "duration_requirement": duration_requirement,
            "lease_requirement": lease_requirement,
        }

    nodes = [
        node("a.lease", "LEASE_ACQUIRE", "CONTROL", CONTROL_DEVICE,
             "CONTROL", []),
        node("a.prefetch", "PREFETCH", compute_domain, compute_device,
             compute_backend, []),
        node("a.h2d", "H2D", compute_domain, compute_device,
             transfer_backend, request_ids, input_requirement="POSITIVE"),
    ]
    for index, request_id in enumerate(request_ids):
        nodes.append(
            node(f"a.submit.{index}", "QUEUE_SUBMIT", "CONTROL",
                 CONTROL_DEVICE, "CONTROL", [request_id],
                 duration_requirement="NONNEGATIVE"))
    nodes.extend([
        node("a.exec", "EXEC", compute_domain, compute_device, compute_backend,
             request_ids, input_requirement="POSITIVE",
             output_requirement="POSITIVE",
             operator_island_digest=island, model_digest=MODEL),
        node("a.d2h", "D2H", compute_domain, compute_device, transfer_backend,
             request_ids, output_requirement="POSITIVE"),
        node("a.result", "RESULT_EMIT", "CONTROL", CONTROL_DEVICE, "CONTROL",
             request_ids),
        node("a.cleanup", "CLEANUP", compute_domain, compute_device,
             compute_backend, []),
        node("a.release", "LEASE_RELEASE", "CONTROL", CONTROL_DEVICE,
             "CONTROL", []),
        node("a.drain", "QUEUE_DRAIN", "CONTROL", CONTROL_DEVICE, "CONTROL",
             [], lease_requirement="FORBIDDEN"),
    ])

    def edge(edge_id, source, target, kind, requests=None, payload_id=None):
        return {
            "edge_id": edge_id,
            "from_action_id": source,
            "to_action_id": target,
            "edge_kind": kind,
            "payload_id": payload_id,
            "request_ids": requests or [],
        }

    route_prefix = "phone" if treatment else "server"
    edges = [
        edge("e.lease.prefetch", "a.lease", "a.prefetch", "CONTROL"),
        edge("e.prefetch.h2d", "a.prefetch", "a.h2d", "CONTROL"),
        edge("e.h2d.exec", "a.h2d", "a.exec", "DATA",
             request_ids, f"{route_prefix}.input"),
    ]
    for index, request_id in enumerate(request_ids):
        edges.append(
            edge(f"e.lease.submit.{index}", "a.lease", f"a.submit.{index}",
                 "CONTROL", [request_id]))
        edges.append(
            edge(f"e.submit.{index}.exec", f"a.submit.{index}", "a.exec",
                 "CONTROL", [request_id]))
    edges.extend([
        edge("e.exec.d2h", "a.exec", "a.d2h", "DATA",
             request_ids, f"{route_prefix}.output"),
        edge("e.d2h.result", "a.d2h", "a.result", "DATA",
             request_ids, f"{route_prefix}.result"),
        edge("e.result.cleanup", "a.result", "a.cleanup", "CONTROL"),
        edge("e.cleanup.release", "a.cleanup", "a.release", "CONTROL"),
        edge("e.release.drain", "a.release", "a.drain", "CONTROL"),
    ])
    record = {
        "schema_version": 4,
        "kind": "RouteSchedule",
        "route_id": ("route.treatment.synthetic" if treatment
                     else "route.control.synthetic"),
        "role": role,
        "request_set_manifest_sha256": man["record_sha256"],
        "model_digest": MODEL,
        "server_device_ids": [BOARD],
        "phone_device_ids": [PHONE],
        "phone_assisted_request_ids": request_ids if treatment else [],
        "nodes": nodes,
        "edges": edges,
        "record_sha256": "",
    }
    return canon.seal(record)


def role_for(slot_index):
    """The frozen ABBA rotation, by slot."""
    pair_index, ordinal = divmod(slot_index, 2)
    first = CONTROL_ROLE if pair_index % 2 == 0 else TREATMENT_ROLE
    second = TREATMENT_ROLE if pair_index % 2 == 0 else CONTROL_ROLE
    return first if ordinal == 0 else second


def make_samples(power_mw, start_us, window_us, period_us):
    """A sample stream whose zero-order-hold integral is exactly power*window.

    Power alternates +/-1 mW each period so the stream carries genuine value
    changes; the pairs cancel exactly, so the integral stays known in closed form
    and the tests can check the arithmetic by hand.
    """
    samples = []
    timestamp = start_us
    toggle = 0
    while timestamp < start_us + window_us:
        delta = 1 if toggle % 2 == 0 else -1
        samples.append([timestamp, power_mw + delta])
        timestamp += period_us
        toggle += 1
    samples.append([start_us + window_us, power_mw])
    return samples


def timeline(slot_index, power_mw, route, uncertainty_mw=UNCERTAINTY_MW,
             provenance="SYNTHETIC", instrument_kind="SYNTHETIC"):
    role = role_for(slot_index)
    # Slots start at stride 1, not 0: the ledger records a SLOT_OPEN before each
    # window opens, and a monotonic clock has no negative timestamps.
    start = (slot_index + 1) * SLOT_STRIDE_US
    end = start + WINDOW_US
    tid = f"tl.slot{slot_index:02d}"
    samples = make_samples(power_mw, start, WINDOW_US, PERIOD_US)
    normalized = integrator.normalize_samples(samples)
    energy = integrator.integrate(normalized, start, end)
    updates, gap = integrator.window_quality(normalized, start, end)
    pstates = ["P0"] * len(samples)
    record = {
        "schema_version": 1,
        "kind": "RealizedTimeline",
        "timeline_id": tid,
        "run_nonce": f"run.{tid}",
        "role": role,
        "provenance": provenance,
        "policy_digest": (POLICY_CONTROL if role == CONTROL_ROLE
                          else POLICY_TREATMENT),
        "workload_digest": WORKLOAD,
        "trace_digest": TRACE,
        "slo_policy_digest": SLO,
        "route_schedule_digest": route["record_sha256"],
        "offered_work": N_REQUESTS,
        "completed_work": N_REQUESTS,
        "outcomes": {"met": N_REQUESTS, "tardy": 0, "rejected": 0, "canceled": 0},
        "clock_epoch_id": CLOCK_EPOCH,
        "marker_start_us": start,
        "marker_end_us": end,
        "window_start_us": start,
        "window_end_us": end,
        "scope": "GPU_BOARD",
        "instrument_kind": instrument_kind,
        "instrument_identity": "synthetic-meter-0",
        "instrument_label": "synthetic mechanics meter",
        "synchronization": "MONOTONIC_SHARED",
        "board_uuids": [BOARD],
        "included_rails": [f"{BOARD}/BOARD"],
        "excluded_rails": ["SERVER/USB_VBUS", "PHONE/BATTERY"],
        "execution_artifact_id": f"execution.{tid}",
        "execution_artifact_sha256": "",
        "execution_artifact_path": f"execution/{tid}.json",
        "raw_artifact_id": f"raw.{tid}",
        "raw_artifact_sha256": "",
        "raw_artifact_path": f"raw/{tid}.json",
        "paid_payload_sha256": comparator.paid_payload_digest(
            normalized, pstates, start, end),
        "normalizer_digest": NORMALIZER,
        "sample_count": len(normalized),
        "independent_updates": updates,
        "max_gap_us": gap,
        "energy_nj": energy,
        "uncertainty_nj": uncertainty_mw * WINDOW_US,
        "source_revision": "933c722f6",
        "build_revision": "synthetic-e2a-build-0",
        "device_identity": BOARD,
        "status": "OK",
        "reason_code": "SYNTHETIC_MECHANICS_FIXTURE",
        "record_sha256": "",
    }
    raw = {"schema": "e2.raw.v2", "pstates": pstates, "samples": samples}
    raw.update({field: record[field] for field in comparator.RAW_BOUND_FIELDS})
    raw_text = json.dumps(raw, indent=2, sort_keys=True) + "\n"
    record["raw_artifact_sha256"] = canon.sha256_bytes(raw_text.encode("ascii"))

    execution = {"schema": "e2.execution.v2"}
    execution.update({field: record[field]
                      for field in comparator.EXECUTION_BOUND_FIELDS})
    execution_text = json.dumps(execution, indent=2, sort_keys=True) + "\n"
    record["execution_artifact_sha256"] = canon.sha256_bytes(
        execution_text.encode("ascii"))
    return canon.seal(record), raw_text, execution_text


def manifest():
    entries = []
    for index in range(N_REQUESTS):
        entry = {
            "request_id": f"req.{index}",
            "input_sha256": canon.digest({"input": index}),
            "prompt_token_ids_sha256": canon.digest({"prompt_ids": index}),
            "prompt_tokens": 128,
            "arrival_us": 920_000 + index * 10_000,
            "model_digest": MODEL,
            "tokenizer_digest": TOKENIZER,
            "sampling_mode": "GREEDY",
            "seed": 1234,
            "max_tokens": 64,
            "stop_set_digest": STOP_SET,
            "decode_params_digest": DECODE_PARAMS,
            "slo_class": "interactive",
            "slo_deadline_us": 5_000_000,
            "length_tolerance_tokens": 0,
            "min_prefix_tokens": 32,
            "entry_sha256": "",
        }
        entry["entry_sha256"] = canon.digest(
            {k: v for k, v in entry.items() if k != "entry_sha256"})
        entries.append(entry)
    record = {
        "schema_version": 4,
        "kind": "RequestSetManifest",
        "manifest_id": "manifest.s10.e2a.synthetic",
        "set_id": SET_ID,
        "corpus_sha256": canon.digest({"corpus": "s10_e2a_synthetic"}),
        "sampler_digest": SAMPLER,
        "entries": entries,
        "work_vector": {
            "requests": N_REQUESTS,
            "prompt_tokens": 128 * N_REQUESTS,
            "max_generated_tokens": 64 * N_REQUESTS,
        },
        "record_sha256": "",
    }
    return canon.seal(record)


def outcomes_for(slot_index, tl, man):
    items = []
    artifacts = {}
    start = tl["window_start_us"]
    for index, entry in enumerate(man["entries"]):
        rid = entry["request_id"]
        arrival_us = start + entry["arrival_us"]
        dispatched_us = start + 1_000_000 + index * 1_000_000
        last_token_us = start + 1_500_000 + index * 1_000_000
        token_ids = [index * 1000 + token for token in range(48)]
        token_events = [
            {"token_id": token_id,
             "emitted_us": dispatched_us + ((position + 1) * 500_000) // 48}
            for position, token_id in enumerate(token_ids)
        ]
        output = {
            "schema": "e2a.output.v4",
            "timeline_id": tl["timeline_id"],
            "run_nonce": tl["run_nonce"],
            "request_id": rid,
            "model_digest": entry["model_digest"],
            "tokenizer_digest": entry["tokenizer_digest"],
            "sampling_mode": entry["sampling_mode"],
            "seed": entry["seed"],
            "input_sha256": entry["input_sha256"],
            "prompt_token_ids_sha256": entry["prompt_token_ids_sha256"],
            "decode_params_digest": entry["decode_params_digest"],
            "stop_set_digest": entry["stop_set_digest"],
            "arrival_us": arrival_us,
            "dispatched_us": dispatched_us,
            "token_events": token_events,
            "stop_reason": "EOS",
        }
        output_path = f"outputs/slot{slot_index:02d}/{rid}.json"
        output_text = json.dumps(output, indent=2, sort_keys=True) + "\n"
        artifacts[output_path] = output_text
        items.append({
            "request_id": rid,
            "manifest_entry_sha256": entry["entry_sha256"],
            "realized_model_digest": MODEL,
            "weights_transform_id": "IDENTITY",
            "realized_seed": entry["seed"],
            "realized_sampling_mode": "GREEDY",
            "realized_input_sha256": entry["input_sha256"],
            "realized_prompt_token_ids_sha256":
                entry["prompt_token_ids_sha256"],
            "realized_decode_params_digest": entry["decode_params_digest"],
            "realized_stop_set_digest": entry["stop_set_digest"],
            "realized_output_tokens": 48,
            "stop_reason": "EOS",
            "terminal_outcome": "met",
            "arrival_us": arrival_us,
            "dispatched_us": dispatched_us,
            "last_token_us": last_token_us,
            "output_artifact_path": output_path,
            "output_artifact_sha256": canon.sha256_bytes(
                output_text.encode("ascii")),
            "output_token_ids_sha256": canon.digest(token_ids),
            "certificate": {
                "certificate_kind": "GREEDY_PREFIX_AGREEMENT",
                "prefix_agreement_tokens": 48,
                "first_divergence_index": -1,
                "referee_verdict": "AGREE",
            },
        })
    record = {
        "schema_version": 4,
        "kind": "RequestOutcomeRecord",
        "outcome_set_id": f"outcomes.slot{slot_index:02d}",
        "set_id": SET_ID,
        "timeline_id": tl["timeline_id"],
        "run_nonce": tl["run_nonce"],
        "pair_index": slot_index // 2,
        "role": tl["role"],
        "manifest_sha256": man["record_sha256"],
        "outcomes": items,
        "record_sha256": "",
    }
    return canon.seal(record), artifacts


def lifecycle_for(slot_index, tl, man, route):
    start = tl["window_start_us"]
    end = tl["window_end_us"]
    actions = [
        {"action_id": "a.lease", "action_kind": "LEASE_ACQUIRE",
         "state": "COMPLETE", "enqueue_us": start, "start_us": start,
         "end_us": start + 1000, "ack_us": start + 1000,
         "lease_id": "lease.0", "parent_action_id": None},
        {"action_id": "a.prefetch", "action_kind": "PREFETCH",
         "state": "COMPLETE", "enqueue_us": start, "start_us": start + 1000,
         "end_us": start + 500_000, "ack_us": start + 500_000,
         "lease_id": None, "parent_action_id": None},
        {"action_id": "a.h2d", "action_kind": "H2D", "state": "COMPLETE",
         "enqueue_us": start + 500_000, "start_us": start + 500_000,
         "end_us": start + 900_000, "ack_us": start + 900_000,
         "lease_id": None, "parent_action_id": None},
    ]
    for index, entry in enumerate(man["entries"]):
        arrival = start + entry["arrival_us"]
        actions.append({
            "action_id": f"a.submit.{index}",
            "action_kind": "QUEUE_SUBMIT", "state": "COMPLETE",
            "enqueue_us": arrival, "start_us": arrival,
            "end_us": arrival, "ack_us": arrival,
            "lease_id": None, "parent_action_id": None,
        })
    actions.extend([
        {"action_id": "a.exec", "action_kind": "EXEC", "state": "COMPLETE",
         "enqueue_us": start + 900_000, "start_us": start + 1_000_000,
         "end_us": end - 2_000_000, "ack_us": end - 2_000_000,
         "lease_id": None, "parent_action_id": None},
        {"action_id": "a.d2h", "action_kind": "D2H", "state": "COMPLETE",
         "enqueue_us": end - 2_000_000, "start_us": end - 2_000_000,
         "end_us": end - 1_500_000, "ack_us": end - 1_500_000,
         "lease_id": None, "parent_action_id": None},
        {"action_id": "a.result", "action_kind": "RESULT_EMIT",
         "state": "COMPLETE", "enqueue_us": end - 1_500_000,
         "start_us": end - 1_400_000, "end_us": end - 1_300_000,
         "ack_us": end - 1_300_000, "lease_id": None,
         "parent_action_id": "a.exec"},
        {"action_id": "a.cleanup", "action_kind": "CLEANUP",
         "state": "COMPLETE", "enqueue_us": end - 1_300_000,
         "start_us": end - 1_200_000, "end_us": end - 900_000,
         "ack_us": end - 900_000, "lease_id": None, "parent_action_id": None},
        {"action_id": "a.release", "action_kind": "LEASE_RELEASE",
         "state": "COMPLETE", "enqueue_us": end - 900_000,
         "start_us": end - 900_000, "end_us": end - 800_000,
         "ack_us": end - 800_000, "lease_id": "lease.0",
         "parent_action_id": None},
        {"action_id": "a.drain", "action_kind": "QUEUE_DRAIN",
         "state": "COMPLETE", "enqueue_us": end - 800_000,
         "start_us": end - 700_000, "end_us": end - 600_000,
         "ack_us": end - 600_000, "lease_id": None, "parent_action_id": None},
    ])
    request_ids = [f"req.{index}" for index in range(N_REQUESTS)]
    nodes = {node["action_id"]: node for node in route["nodes"]}
    incoming = {}
    for edge in route["edges"]:
        incoming.setdefault(edge["to_action_id"], []).append(
            edge["from_action_id"])
    for action in actions:
        if action["action_kind"] in ("H2D", "EXEC", "D2H", "RESULT_EMIT"):
            action["request_ids"] = request_ids
        elif action["action_kind"] == "QUEUE_SUBMIT":
            index = int(action["action_id"].rsplit(".", 1)[1])
            action["request_ids"] = [request_ids[index]]
        else:
            action["request_ids"] = []
        node = nodes[action["action_id"]]
        action["parent_action_id"] = (
            sorted(incoming.get(action["action_id"], []))[0]
            if incoming.get(action["action_id"]) else None)
        action["route_schedule_digest"] = route["record_sha256"]
        action["execution_domain"] = node["execution_domain"]
        action["device_identity"] = node["device_identity"]
        action["backend_kind"] = node["backend_kind"]
        action["operator_island_digest"] = node["operator_island_digest"]
        action["model_digest"] = node["model_digest"]
        action["input_bytes"] = (
            8 * 1024 * 1024
            if action["action_kind"] == "H2D"
            else 4096 if node["input_requirement"] == "POSITIVE" else 0)
        action["output_bytes"] = (
            4096 if node["output_requirement"] == "POSITIVE" else 0)
        action["lease_id"] = (
            "lease.0" if node["lease_requirement"] == "REQUIRED" else None)
    record = {
        "schema_version": 4,
        "kind": "LifecycleRecord",
        "lifecycle_id": f"lc.slot{slot_index:02d}",
        "set_id": SET_ID,
        "timeline_id": tl["timeline_id"],
        "run_nonce": tl["run_nonce"],
        "clock_epoch_id": tl["clock_epoch_id"],
        "window_start_us": start,
        "window_end_us": end,
        "entry_state_digest": resolver.lifecycle_state_digest(actions, start),
        "exit_state_digest": resolver.lifecycle_state_digest(actions, end),
        "drain_acknowledged_us": end - 600_000,
        "actions": actions,
        "record_sha256": "",
    }
    return canon.seal(record)


def plan(man, timelines, lifecycles, routes):
    slots = []
    for slot_index in range(2 * N_PAIRS):
        slots.append({
            "slot_index": slot_index,
            "pair_index": slot_index // 2,
            "role": role_for(slot_index),
            "ordinal_in_pair": slot_index % 2,
            "run_nonce": timelines[slot_index]["run_nonce"],
            "expected_entry_state_digest":
                lifecycles[slot_index]["entry_state_digest"],
            "expected_exit_state_digest":
                lifecycles[slot_index]["exit_state_digest"],
        })
    declared_order = [role_for(index) for index in range(2 * N_PAIRS)]
    record = {
        "schema_version": 4,
        "kind": "PreRunPlan",
        "plan_id": "plan.s10.e2a.synthetic",
        "chain_id": CHAIN_ID,
        "set_id": SET_ID,
        "experiment_identity": "s10.e2a.mechanics",
        "n_pairs": N_PAIRS,
        "slots": slots,
        "declared_order": declared_order,
        "order_digest": canon.digest(declared_order),
        "aggregate_method": "SUM_ALL_PAIRS_V1",
        "max_attempts_per_slot": 1,
        "warmup_count": 0,
        "gate_constants_digest": resolver.GATE_CONSTANTS_DIGEST,
        "request_set_manifest_sha256": man["record_sha256"],
        "workload_digest": WORKLOAD,
        "trace_digest": TRACE,
        "slo_policy_digest": SLO,
        "policy_digest_control": POLICY_CONTROL,
        "policy_digest_treatment": POLICY_TREATMENT,
        "route_schedule_digest_control":
            routes[CONTROL_ROLE]["record_sha256"],
        "route_schedule_digest_treatment":
            routes[TREATMENT_ROLE]["record_sha256"],
        "server_device_ids": [BOARD],
        "phone_device_ids": [PHONE],
        "commitment_namespace": COMMITMENT_NAMESPACE,
        "model_digest": MODEL,
        "tokenizer_digest": TOKENIZER,
        "sampler_digest": SAMPLER,
        "source_revision": "933c722f6",
        "build_revision": "synthetic-e2a-build-0",
        "scope": "GPU_BOARD",
        "instrument_kind": "SYNTHETIC",
        "instrument_identity": "synthetic-meter-0",
        "device_identity": BOARD,
        "board_uuids": [BOARD],
        "included_rails": [f"{BOARD}/BOARD"],
        "excluded_rails": ["SERVER/USB_VBUS", "PHONE/BATTERY"],
        "server_wall_capability_record_sha256": None,
        "record_sha256": "",
    }
    return canon.seal(record)


def ledger(pln, plan_anchor, timelines, lifecycles, outcomes):
    entries = []
    previous = "0" * 64

    def append(entry_kind, slot_index, monotonic_us, run_nonce=None,
               plan_anchor_sha=None, timeline_id=None, timeline_sha=None,
               lifecycle_sha=None, outcomes_sha=None, window_start_us=None,
               window_end_us=None, drain_acknowledged_us=None):
        nonlocal previous
        body = {
            "seq": len(entries),
            "prev_entry_sha256": previous,
            "entry_kind": entry_kind,
            "slot_index": slot_index,
            "attempt_ordinal": 0,
            "monotonic_us": monotonic_us,
            "clock_epoch_id": CLOCK_EPOCH,
            "status": "OK",
            "reason_code": "SYNTHETIC_MECHANICS_FIXTURE",
            "run_nonce": run_nonce,
            "timeline_id": timeline_id,
            "plan_anchor_record_sha256": plan_anchor_sha,
            "timeline_record_sha256": timeline_sha,
            "lifecycle_record_sha256": lifecycle_sha,
            "request_outcome_record_sha256": outcomes_sha,
            "window_start_us": window_start_us,
            "window_end_us": window_end_us,
            "drain_acknowledged_us": drain_acknowledged_us,
        }
        body["entry_sha256"] = canon.digest(body)
        entries.append(body)
        previous = body["entry_sha256"]

    append("PLAN_ANCHOR", -1, 0,
           plan_anchor_sha=plan_anchor["record_sha256"])
    for slot_index in range(2 * N_PAIRS):
        tl = timelines[slot_index]
        append("SLOT_OPEN", slot_index, tl["window_start_us"] - 1000,
               run_nonce=tl["run_nonce"])
        append("SLOT_ATTEMPT_START", slot_index, tl["window_start_us"],
               run_nonce=tl["run_nonce"],
               window_start_us=tl["window_start_us"])
        append("SLOT_ATTEMPT_END", slot_index, tl["window_end_us"],
               run_nonce=tl["run_nonce"], timeline_id=tl["timeline_id"],
               timeline_sha=tl["record_sha256"],
               lifecycle_sha=lifecycles[slot_index]["record_sha256"],
               outcomes_sha=outcomes[slot_index]["record_sha256"],
               window_start_us=tl["window_start_us"],
               window_end_us=tl["window_end_us"],
               drain_acknowledged_us=
               lifecycles[slot_index]["drain_acknowledged_us"])
    append("SET_SEAL", -1, (2 * N_PAIRS + 2) * SLOT_STRIDE_US)
    record = {
        "schema_version": 4,
        "kind": "AttemptLedger",
        "ledger_id": "ledger.s10.e2a.synthetic",
        "chain_id": CHAIN_ID,
        "set_id": SET_ID,
        "plan_sha256": pln["record_sha256"],
        "entries": entries,
        "head_sha256": previous,
        "record_sha256": "",
    }
    return canon.seal(record)


def anchor_receipt(pln, proof_digest, anchor_kind="RFC3161_TSA",
                   trust_root_custodian="THIRD_PARTY_CA",
                   pin_location="E2A_FROZEN_CONSTANT"):
    """A receipt that is as strong as this host can make one.

    It is deliberately NOT weakened: anchor_kind is the real third-party TSA,
    the custodian is a third-party CA, the pin location is the frozen constant.
    The gate still refuses it, and that is the finding -- the refusal is
    E_ANCHOR_UNENUMERABLE, about a property RFC3161 does not have, not
    E_ANCHOR_NOT_INDEPENDENT, about one it does.
    """
    record = {
        "schema_version": 4,
        "kind": "PlanAnchorReceipt",
        "anchor_id": "anchor.plan.s10.e2a.synthetic",
        "anchor_kind": anchor_kind,
        "chain_id": CHAIN_ID,
        "set_id": SET_ID,
        "plan_id": pln["plan_id"],
        "experiment_identity": pln["experiment_identity"],
        "commitment_namespace": pln["commitment_namespace"],
        "anchored_digest": pln["record_sha256"],
        "message_imprint_sha256": pln["record_sha256"],
        "token_hash_algorithm": "sha256",
        "token_der_path": "anchors/plan_token.der",
        "token_der_sha256": TOKEN_DIGESTS["plan"],
        "anchor_time_utc_us": 1_784_000_000_000_000,
        "serial_number_hex": "0x0640067A",
        "tsa_subject_dn": "CN=freetsa.org,O=Free TSA,C=DE",
        "tsa_leaf_cert_sha256": canon.digest({"leaf": "synthetic-tsa"}),
        "tsa_policy_oid": "1.2.3.4.1",
        "trust_root_id": "synthetic-tsa-root",
        "trust_root_sha256": canon.digest({"root": "synthetic-tsa"}),
        "trust_root_pin_location": pin_location,
        "trust_root_custodian": trust_root_custodian,
        "log_identity": "synthetic-log",
        "log_checkpoint_sha256": canon.digest({"checkpoint": 1}),
        "identity_binding_sha256": canon.digest({
            "identity": pln["experiment_identity"],
        }),
        "commitment_proof_path": "anchors/commitment_proof.json",
        "commitment_proof_sha256": proof_digest,
        "enumeration_status": "COMPLETE",
        "committed_plan_count": 1,
        "committed_plan_set_sha256": canon.digest([pln["record_sha256"]]),
        "verifier_kind": "OPENSSL_TS_VERIFY",
        "provenance": "MEASURED",
        "status": "OK",
        "reason_code": "SYNTHETIC_MECHANICS_FIXTURE",
        "record_sha256": "",
    }
    return canon.seal(record)


def close_receipt(pln, anchor, led, proof_digest,
                  anchor_kind="RFC3161_TSA"):
    record = {
        "schema_version": 4,
        "kind": "LedgerCloseReceipt",
        "close_id": "anchor.close.s10.e2a.synthetic",
        "anchor_kind": anchor_kind,
        "chain_id": CHAIN_ID,
        "set_id": SET_ID,
        "experiment_identity": pln["experiment_identity"],
        "commitment_namespace": pln["commitment_namespace"],
        "plan_anchor_id": anchor["anchor_id"],
        "plan_anchor_record_sha256": anchor["record_sha256"],
        "attempt_ledger_sha256": led["record_sha256"],
        "ledger_head_sha256": led["head_sha256"],
        "anchored_digest": led["record_sha256"],
        "message_imprint_sha256": led["record_sha256"],
        "token_hash_algorithm": "sha256",
        "token_der_path": "anchors/close_token.der",
        "token_der_sha256": TOKEN_DIGESTS["close"],
        "anchor_time_utc_us": 1_784_000_600_000_000,
        "serial_number_hex": "0x0640067B",
        "tsa_subject_dn": "CN=freetsa.org,O=Free TSA,C=DE",
        "tsa_leaf_cert_sha256": canon.digest({"leaf": "synthetic-tsa"}),
        "tsa_policy_oid": "1.2.3.4.1",
        "trust_root_id": "synthetic-tsa-root",
        "trust_root_sha256": canon.digest({"root": "synthetic-tsa"}),
        "trust_root_pin_location": "E2A_FROZEN_CONSTANT",
        "trust_root_custodian": "THIRD_PARTY_CA",
        "log_identity": anchor["log_identity"],
        "log_checkpoint_sha256": anchor["log_checkpoint_sha256"],
        "identity_binding_sha256": anchor["identity_binding_sha256"],
        "commitment_proof_path": anchor["commitment_proof_path"],
        "commitment_proof_sha256": proof_digest,
        "enumeration_status": anchor["enumeration_status"],
        "committed_plan_count": anchor["committed_plan_count"],
        "committed_plan_set_sha256": anchor["committed_plan_set_sha256"],
        "verifier_kind": "OPENSSL_TS_VERIFY",
        "provenance": "MEASURED",
        "status": "OK",
        "reason_code": "SYNTHETIC_MECHANICS_FIXTURE",
        "record_sha256": "",
    }
    return canon.seal(record)


TOKEN_DIGESTS = {}


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")


def generate():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # Anchor tokens are inputs to their receipt records.
    for name in ("plan", "close"):
        # A stand-in for a DER token. It is NOT a real RFC3161 token and no test
        # pretends otherwise: the gate refuses this anchor on its KIND, long
        # before any signature would be parsed.
        blob = f"SYNTHETIC-NOT-A-REAL-RFC3161-TOKEN-{name}\n".encode("ascii")
        path = OUT / "anchors" / f"{name}_token.der"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        TOKEN_DIGESTS[name] = canon.sha256_bytes(blob)

    man = manifest()
    manifest_text = json.dumps(man, indent=2, sort_keys=True) + "\n"
    _write(OUT / "manifest.json", manifest_text)

    routes = {
        CONTROL_ROLE: route_schedule(man, CONTROL_ROLE),
        TREATMENT_ROLE: route_schedule(man, TREATMENT_ROLE),
    }
    route_texts = {}
    for role, name in ((CONTROL_ROLE, "route_control.json"),
                       (TREATMENT_ROLE, "route_treatment.json")):
        route_text = json.dumps(
            routes[role], indent=2, sort_keys=True) + "\n"
        route_texts[role] = route_text
        _write(OUT / "routes" / name, route_text)

    timelines = {}
    lifecycles = {}
    outcome_records = {}
    slot_index_entries = []
    for slot_index in range(2 * N_PAIRS):
        role = role_for(slot_index)
        power = CONTROL_MW if role == CONTROL_ROLE else TREATMENT_MW
        route = routes[role]
        tl, raw_text, execution_text = timeline(slot_index, power, route)
        timelines[slot_index] = tl
        _write(OUT / "raw" / f"{tl['timeline_id']}.json", raw_text)
        _write(OUT / "execution" / f"{tl['timeline_id']}.json", execution_text)
        tl_text = json.dumps(tl, indent=2, sort_keys=True) + "\n"
        tl_path = OUT / "timelines" / f"{tl['timeline_id']}.json"
        _write(tl_path, tl_text)

        lc = lifecycle_for(slot_index, tl, man, route)
        lifecycles[slot_index] = lc
        lc_text = json.dumps(lc, indent=2, sort_keys=True) + "\n"
        _write(OUT / "lifecycle" / f"{lc['lifecycle_id']}.json", lc_text)

        ro, output_artifacts = outcomes_for(slot_index, tl, man)
        outcome_records[slot_index] = ro
        for relative, output_text in output_artifacts.items():
            _write(OUT / relative, output_text)
        ro_text = json.dumps(ro, indent=2, sort_keys=True) + "\n"
        _write(OUT / "outcomes" / f"{ro['outcome_set_id']}.json", ro_text)

        slot_index_entries.append({
            "slot_index": slot_index,
            "timeline_path": f"timelines/{tl['timeline_id']}.json",
            "timeline_sha256": canon.sha256_bytes(tl_text.encode("ascii")),
            "lifecycle_path": f"lifecycle/{lc['lifecycle_id']}.json",
            "lifecycle_sha256": canon.sha256_bytes(lc_text.encode("ascii")),
            "outcomes_path": f"outcomes/{ro['outcome_set_id']}.json",
            "outcomes_sha256": canon.sha256_bytes(ro_text.encode("ascii")),
        })

    pln = plan(man, timelines, lifecycles, routes)
    plan_text = json.dumps(pln, indent=2, sort_keys=True) + "\n"
    _write(OUT / "plan.json", plan_text)

    proof = {
        "schema": "e2a.commitment-proof.v4",
        "experiment_identity": pln["experiment_identity"],
        "commitment_namespace": pln["commitment_namespace"],
        "log_identity": "synthetic-log",
        "log_checkpoint_sha256": canon.digest({"checkpoint": 1}),
        "identity_binding_sha256": canon.digest({
            "identity": pln["experiment_identity"],
        }),
        "enumeration_status": "COMPLETE",
        "committed_plan_count": 1,
        "committed_plan_sha256s": [pln["record_sha256"]],
        "committed_plan_set_sha256": canon.digest([pln["record_sha256"]]),
    }
    proof_text = json.dumps(proof, indent=2, sort_keys=True) + "\n"
    _write(OUT / "anchors" / "commitment_proof.json", proof_text)
    proof_digest = canon.sha256_bytes(proof_text.encode("ascii"))

    anchor = anchor_receipt(pln, proof_digest)
    anchor_text = json.dumps(anchor, indent=2, sort_keys=True) + "\n"
    _write(OUT / "plan_anchor.json", anchor_text)

    led = ledger(pln, anchor, timelines, lifecycles, outcome_records)
    ledger_text = json.dumps(led, indent=2, sort_keys=True) + "\n"
    _write(OUT / "ledger.json", ledger_text)

    close = close_receipt(pln, anchor, led, proof_digest)
    close_text = json.dumps(close, indent=2, sort_keys=True) + "\n"
    _write(OUT / "ledger_close.json", close_text)

    index = {
        "schema_version": 4,
        "kind": "E2ABundle",
        "plan_path": "plan.json",
        "plan_sha256": canon.sha256_bytes(plan_text.encode("ascii")),
        "plan_anchor_path": "plan_anchor.json",
        "plan_anchor_sha256": canon.sha256_bytes(anchor_text.encode("ascii")),
        "ledger_path": "ledger.json",
        "ledger_sha256": canon.sha256_bytes(ledger_text.encode("ascii")),
        "ledger_close_path": "ledger_close.json",
        "ledger_close_sha256": canon.sha256_bytes(close_text.encode("ascii")),
        "manifest_path": "manifest.json",
        "manifest_sha256": canon.sha256_bytes(manifest_text.encode("ascii")),
        "route_control_path": "routes/route_control.json",
        "route_control_sha256": canon.sha256_bytes(
            route_texts[CONTROL_ROLE].encode("ascii")),
        "route_treatment_path": "routes/route_treatment.json",
        "route_treatment_sha256": canon.sha256_bytes(
            route_texts[TREATMENT_ROLE].encode("ascii")),
        "server_wall_capability_path": None,
        "server_wall_capability_sha256": None,
        "slots": slot_index_entries,
    }
    _write(OUT / "bundle.json",
           json.dumps(index, indent=2, sort_keys=True) + "\n")
    return {"plan": pln, "ledger": led, "manifest": man, "anchor": anchor,
            "close": close, "timelines": timelines, "routes": routes,
            "index": index}


def main(argv=None):
    generate()
    print("S10_E2A_FIXTURES_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
