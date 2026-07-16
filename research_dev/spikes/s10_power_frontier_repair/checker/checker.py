#!/usr/bin/env python3
"""Standalone S10-V0-R feasibility, accounting, and optimality checker.

This module imports no oracle, policy, simulator, or candidate-generator code. It
validates the signed record and rederives schedule feasibility, activation
lifetime, outcomes, transition intervals, and energy.

Two visibly separate modes:

  default (exact certificate): every feasibility/accounting check above PLUS an
    independent optimality proof. Solver counters are not part of the signed
    certificate. The proof is a checker-owned recomputation (reference.py) with
    separate configuration, start-time, legality, and objective enumeration and
    no oracle imports. A feasible but suboptimal certificate is REJECTED. If the
    instance falls outside the reference's declared tiny domain the checker fails
    closed and certifies nothing.

  --feasibility-only: accounting validation ONLY. It never claims validity or
    optimality and always reports optimality_verified=false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import defaultdict

import reference


SCHEMAS = pathlib.Path(__file__).resolve().parents[1] / "schemas"
SCHEMA_PATH = str(SCHEMAS)
sys.path[:] = [entry for entry in sys.path if entry != SCHEMA_PATH]
sys.path.insert(0, SCHEMA_PATH)
from foundation_instance_gate import instance_schema_errors


class DuplicateKeyError(ValueError):
    pass


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def load_strict(path):
    with open(path, encoding="ascii") as handle:
        return json.load(handle, object_pairs_hook=_unique_object,
                         parse_constant=_reject_constant)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _is_int(value):
    return type(value) is int


def _server_energy(inst, actions):
    power = inst["server_power"]
    windows = []
    for action in actions:
        if action["device"] != "SERVER":
            continue
        start = max(0, action["start_us"] - power["wake_us"])
        end = min(inst["horizon_us"], action["finish_us"] + power["idle_entry_us"])
        windows.append((start, end))
    windows.sort()
    merged = []
    for start, end in windows:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    active_us = sum(end - start for start, end in merged)
    energy = power["p8_mw"] * inst["horizon_us"]
    energy += (power["p0_mw"] - power["p8_mw"]) * active_us
    energy += power["transition_nj"] * len(merged)
    return energy, merged


def check(inst, cert, require_complete=True):
    failures = []
    schema_failures = []

    def need(condition, message):
        if not condition:
            failures.append(message)

    try:
        schema_failures.extend(
            f"instance schema violation: {error}"
            for error in instance_schema_errors(inst)
        )
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        schema_failures.append(f"instance schema gate failed closed: {exc}")
    if not isinstance(inst, dict):
        return schema_failures + failures

    instance_fields = {
        "schema_version", "instance_id", "horizon_us",
        "activation_mem_bound_bytes", "server_power", "devices",
        "batch_profiles", "requests", "nodes", "evidence",
    }
    certificate_fields = {
        "schema_version", "instance_id", "instance_sha256", "actions",
        "request_outcomes", "activation_peak_bytes", "energy", "objective",
        "search", "certificate_sha256",
    }
    need(isinstance(inst, dict) and set(inst) == instance_fields,
         "instance fields do not match schema v2")
    need(isinstance(cert, dict) and set(cert) == certificate_fields,
         "certificate fields do not match schema v2")
    if failures:
        return schema_failures + failures
    need(inst["schema_version"] == 2 and cert["schema_version"] == 2,
         "schema_version must be 2")
    need(isinstance(inst["instance_id"], str) and bool(inst["instance_id"]),
         "invalid instance_id")
    need(cert["instance_id"] == inst["instance_id"], "instance_id mismatch")
    need(cert["instance_sha256"] == digest(inst), "instance_sha256 mismatch")
    body = {key: value for key, value in cert.items() if key != "certificate_sha256"}
    need(cert["certificate_sha256"] == digest(body), "certificate_sha256 mismatch")

    power = inst["server_power"]
    need(isinstance(power, dict)
         and set(power) == {"p8_mw", "p0_mw", "wake_us", "idle_entry_us", "transition_nj"},
         "invalid server_power fields")
    if not isinstance(power, dict):
        return schema_failures + failures
    for key in ("p8_mw", "p0_mw", "wake_us", "idle_entry_us", "transition_nj"):
        need(_is_int(power.get(key)) and power.get(key, -1) >= 0,
             f"invalid server_power.{key}")
    if _is_int(power.get("p0_mw")) and _is_int(power.get("p8_mw")):
        need(power["p0_mw"] >= power["p8_mw"], "P0 power below P8")
    need(_is_int(inst["horizon_us"]) and inst["horizon_us"] > 0, "invalid horizon")
    need(_is_int(inst["activation_mem_bound_bytes"])
         and inst["activation_mem_bound_bytes"] > 0, "invalid activation bound")

    device_fields = {"kind", "active_mw"}
    devices = inst["devices"]
    need(isinstance(devices, dict) and "SERVER" in devices, "SERVER device missing")
    if isinstance(devices, dict):
        for name, device in devices.items():
            need(isinstance(name, str) and bool(name), "invalid device name")
            need(isinstance(device, dict) and set(device) == device_fields,
                 f"invalid device fields: {name}")
            if isinstance(device, dict):
                need(device.get("kind") in ("server", "phone"), f"invalid device kind: {name}")
                need(_is_int(device.get("active_mw")) and device.get("active_mw", -1) >= 0,
                     f"invalid device power: {name}")
    if (isinstance(devices, dict) and "SERVER" in devices
            and isinstance(devices["SERVER"], dict)):
        need(devices["SERVER"].get("kind") == "server", "SERVER must have server kind")
        need(devices["SERVER"].get("active_mw") == power.get("p0_mw"),
             "SERVER active_mw must equal server_power.p0_mw")
        need({name for name, device in devices.items()
              if isinstance(device, dict) and device.get("kind") == "server"} == {"SERVER"},
             "SERVER must be the only server-kind device")
    evidence = inst["evidence"]
    need(isinstance(evidence, dict)
         and evidence.get("scope") == "MECHANICS_ONLY"
         and all(isinstance(key, str) and isinstance(value, str)
                 for key, value in evidence.items()),
         "evidence scope must be MECHANICS_ONLY")
    if failures:
        return schema_failures + failures

    profiles = inst["batch_profiles"]
    need(isinstance(profiles, dict), "batch_profiles must be an object")
    if isinstance(profiles, dict):
        for key, profile in profiles.items():
            need(isinstance(key, str) and bool(key) and isinstance(profile, dict) and bool(profile),
                 f"invalid batch profile {key}")
            if isinstance(profile, dict):
                for tokens, duration in profile.items():
                    need(isinstance(tokens, str) and tokens.isdigit() and int(tokens) > 0
                         and _is_int(duration) and duration > 0,
                         f"invalid batch profile entry {key}/{tokens}")
    need(isinstance(inst["requests"], list) and 0 < len(inst["requests"]) <= 8,
         "requests must contain one to eight entries")
    need(isinstance(inst["nodes"], list) and 0 < len(inst["nodes"]) <= 8,
         "nodes must contain one to eight entries")
    if failures:
        return schema_failures + failures

    request_fields = {"id", "arrival_us", "terminal_node", "deadline_us", "priority"}
    node_fields = {
        "id", "request_id", "model_id", "weight_set_id", "predecessors",
        "release_us", "routes", "batch_key", "tokens", "output_bytes",
    }
    route_fields = {"duration_us", "extra_energy_nj"}
    req_by_id = {}
    for request in inst["requests"]:
        need(isinstance(request, dict) and set(request) == request_fields,
             f"invalid request fields: {request.get('id') if isinstance(request, dict) else '?'}")
        if not isinstance(request, dict):
            continue
        rid = request.get("id")
        need(isinstance(rid, str) and rid and rid not in req_by_id, "duplicate/invalid request id")
        if isinstance(rid, str) and rid not in req_by_id:
            req_by_id[rid] = request

    node_by_id = {}
    for node in inst["nodes"]:
        need(isinstance(node, dict) and set(node) == node_fields,
             f"invalid node fields: {node.get('id') if isinstance(node, dict) else '?'}")
        if not isinstance(node, dict):
            continue
        nid = node.get("id")
        need(isinstance(nid, str) and nid and nid not in node_by_id, "duplicate/invalid node id")
        request_id = node.get("request_id")
        need(isinstance(request_id, str) and request_id in req_by_id,
             f"unknown request for node {nid}")
        need(isinstance(node.get("model_id"), str) and bool(node.get("model_id")),
             f"invalid model_id for {nid}")
        need(isinstance(node.get("weight_set_id"), str) and bool(node.get("weight_set_id")),
             f"invalid weight_set_id for {nid}")
        need(isinstance(node.get("predecessors"), list)
             and all(isinstance(pred, str) and pred for pred in node.get("predecessors", []))
             and len(node.get("predecessors", [])) == len(set(node.get("predecessors", []))),
             f"invalid predecessors for {nid}")
        routes = node.get("routes")
        need(isinstance(routes, dict) and bool(routes),
             f"invalid routes for {nid}")
        for device, route in routes.items() if isinstance(routes, dict) else ():
            need(device in devices, f"unknown device {device} for {nid}")
            need(isinstance(route, dict) and set(route) == route_fields,
                 f"invalid route fields for {nid}/{device}")
            if isinstance(route, dict):
                need(_is_int(route.get("duration_us")) and route.get("duration_us", 0) > 0,
                     f"invalid duration for {nid}/{device}")
                need(_is_int(route.get("extra_energy_nj"))
                     and route.get("extra_energy_nj", -1) >= 0,
                     f"invalid extra energy for {nid}/{device}")
                if device == "SERVER":
                    need(route.get("extra_energy_nj") == 0,
                         f"SERVER route extra energy must be zero for {nid}")
        if isinstance(nid, str) and nid not in node_by_id:
            node_by_id[nid] = node
    if failures:
        return schema_failures + failures

    for rid, request in req_by_id.items():
        arrival = request["arrival_us"]
        deadline = request["deadline_us"]
        valid_arrival = _is_int(arrival) and arrival >= 0
        valid_deadline = (_is_int(deadline) and valid_arrival
                          and arrival <= deadline <= inst["horizon_us"])
        need(valid_arrival,
             f"invalid arrival for {rid}")
        need(valid_deadline,
             f"invalid deadline for {rid}")
        need(_is_int(request["priority"]) and request["priority"] == 0,
             f"foundation priority must be zero for {rid}")
        need(isinstance(request["terminal_node"], str) and bool(request["terminal_node"]),
             f"invalid terminal node for {rid}")
        terminal = request["terminal_node"]
        need(isinstance(terminal, str) and terminal in node_by_id,
             f"unknown terminal node for {rid}")
        if isinstance(terminal, str) and terminal in node_by_id:
            need(node_by_id[terminal]["request_id"] == rid,
                 f"terminal node belongs to another request: {rid}")
    if failures:
        return schema_failures + failures
    for nid, node in node_by_id.items():
        request = req_by_id[node["request_id"]]
        need(_is_int(node["release_us"]) and node["release_us"] >= request["arrival_us"],
             f"invalid release for {nid}")
        need(_is_int(node["tokens"]) and node["tokens"] > 0, f"invalid tokens for {nid}")
        need(_is_int(node["output_bytes"]) and node["output_bytes"] >= 0,
             f"invalid output bytes for {nid}")
        need(node["batch_key"] is None or isinstance(node["batch_key"], str),
             f"invalid batch_key for {nid}")
        for pred in node["predecessors"]:
            need(pred in node_by_id and pred != nid, f"invalid predecessor {pred} for {nid}")
            if pred in node_by_id:
                need(node_by_id[pred]["request_id"] == node["request_id"],
                     f"unsupported cross-request dependency {pred}->{nid}")
        if isinstance(node["batch_key"], str) and "SERVER" in node["routes"]:
            profile = inst["batch_profiles"].get(node["batch_key"])
            need(isinstance(profile, dict), f"missing batch profile for {nid}")
            if isinstance(profile, dict):
                expected = profile.get(str(node["tokens"]))
                need(expected == node["routes"]["SERVER"]["duration_us"],
                     f"singleton batch duration mismatch for {nid}")
    if failures:
        return schema_failures + failures

    successors = defaultdict(list)
    indegree = {nid: 0 for nid in node_by_id}
    for nid, node in node_by_id.items():
        for pred in node["predecessors"]:
            successors[pred].append(nid)
            indegree[nid] += 1
    ready = sorted(nid for nid, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        nid = ready.pop(0)
        visited += 1
        for succ in successors[nid]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                ready.append(succ)
                ready.sort()
    need(visited == len(node_by_id), "node graph contains a cycle")
    for rid, request in req_by_id.items():
        terminal = request["terminal_node"]
        need(not successors[terminal], f"terminal node is not a sink: {rid}/{terminal}")
        ancestors = set()
        stack = [terminal]
        while stack:
            nid = stack.pop()
            if nid in ancestors:
                continue
            ancestors.add(nid)
            stack.extend(node_by_id[nid]["predecessors"])
        request_nodes = {nid for nid, node in node_by_id.items() if node["request_id"] == rid}
        need(ancestors == request_nodes, f"request DAG is not closed by terminal node: {rid}")
    if failures:
        return schema_failures + failures

    action_fields = {"id", "device", "members", "start_us", "finish_us"}
    actions = cert["actions"]
    need(isinstance(actions, list) and actions, "actions must be nonempty")
    if not isinstance(actions, list):
        return schema_failures + failures
    action_by_id = {}
    node_action = {}
    device_intervals = defaultdict(list)
    for action in actions:
        need(isinstance(action, dict) and set(action) == action_fields,
             f"invalid action fields: {action.get('id') if isinstance(action, dict) else '?'}")
        if not isinstance(action, dict):
            continue
        aid = action.get("id")
        need(isinstance(aid, str) and aid and aid not in action_by_id, "duplicate/invalid action id")
        device = action.get("device")
        need(isinstance(device, str) and bool(device) and device in devices,
             f"unknown action device {device}")
        members = action.get("members")
        valid_members = (isinstance(members, list) and bool(members)
                         and all(isinstance(nid, str) and nid for nid in members))
        need(valid_members and len(members) == len(set(members)),
             f"invalid members for action {aid}")
        start_us = action.get("start_us")
        finish_us = action.get("finish_us")
        need(_is_int(start_us) and start_us >= 0,
             f"invalid start for action {aid}")
        need(_is_int(finish_us) and _is_int(start_us)
             and finish_us > start_us and finish_us <= inst["horizon_us"],
             f"invalid finish for action {aid}")
        for nid in members if valid_members else ():
            need(nid in node_by_id, f"unknown node {nid} in action {aid}")
            need(nid not in node_action, f"node {nid} executes more than once")
            if nid in node_by_id and nid not in node_action:
                node_action[nid] = action
        if isinstance(aid, str) and aid not in action_by_id:
            action_by_id[aid] = action
        if isinstance(device, str) and device in devices and _is_int(start_us) and _is_int(finish_us):
            device_intervals[device].append((action["start_us"], action["finish_us"], aid))
    for nid in node_by_id:
        need(nid in node_action, f"node {nid} omitted from actions")
    if failures:
        return schema_failures + failures

    for action in actions:
        device = action["device"]
        members = action["members"]
        member_nodes = [node_by_id[nid] for nid in members]
        for node in member_nodes:
            need(device in node["routes"], f"uncertified route {device} for {node['id']}")
            need(action["start_us"] >= node["release_us"], f"node {node['id']} starts before release")
            for pred in node["predecessors"]:
                need(node_action[pred]["finish_us"] <= action["start_us"],
                     f"node {node['id']} starts before predecessor {pred} completes")
        if len(members) > 1:
            keys = {node["batch_key"] for node in member_nodes}
            models = {node["model_id"] for node in member_nodes}
            weights = {node["weight_set_id"] for node in member_nodes}
            need(device == "SERVER" and len(keys) == 1 and None not in keys
                 and len(models) == 1 and len(weights) == 1,
                 f"illegal batch identity in action {action['id']}")
            member_set = set(members)
            need(not any(member_set.intersection(node["predecessors"]) for node in member_nodes),
                 f"dependent nodes batched together in action {action['id']}")
        duration = None
        if device == "SERVER" and member_nodes[0]["batch_key"] is not None:
            key = member_nodes[0]["batch_key"]
            tokens = sum(node["tokens"] for node in member_nodes)
            duration = inst["batch_profiles"].get(key, {}).get(str(tokens))
            need(_is_int(duration), f"missing batch duration {key}/{tokens}")
        else:
            need(len(members) == 1, f"non-server action {action['id']} is batched")
            route = member_nodes[0]["routes"].get(device)
            duration = route.get("duration_us") if isinstance(route, dict) else None
            need(_is_int(duration),
                 f"missing route duration for action {action['id']}")
        if _is_int(duration):
            need(action["finish_us"] - action["start_us"] == duration,
                 f"wrong duration for action {action['id']}")

    if failures:
        return schema_failures + failures

    for device, intervals in device_intervals.items():
        intervals.sort()
        for left, right in zip(intervals, intervals[1:]):
            need(left[1] <= right[0], f"device overlap on {device}: {left[2]}/{right[2]}")
    if device_intervals.get("SERVER"):
        first_server_start = min(interval[0] for interval in device_intervals["SERVER"])
        need(first_server_start >= power["wake_us"],
             "first SERVER action starts before initial wake completes")

    live_intervals = []
    for nid, node in node_by_id.items():
        if node["output_bytes"] == 0:
            continue
        start = node_action[nid]["start_us"]
        end = (max(node_action[succ]["finish_us"] for succ in successors[nid])
               if successors[nid] else node_action[nid]["finish_us"])
        if end > start:
            live_intervals.append((start, end, node["output_bytes"], nid))
    points = sorted({point for start, end, _, _ in live_intervals for point in (start, end)})
    peak = 0
    for point in points:
        peak = max(peak, sum(size for start, end, size, _ in live_intervals
                             if start <= point < end))
    need(peak <= inst["activation_mem_bound_bytes"],
         f"activation peak {peak} exceeds bound {inst['activation_mem_bound_bytes']}")
    need(_is_int(cert["activation_peak_bytes"]), "invalid activation_peak_bytes")
    need(cert["activation_peak_bytes"] == peak, "activation_peak_bytes mismatch")

    outcome_fields = {"request_id", "terminal_finish_us", "outcome", "lateness_us"}
    outcome_by_id = {}
    misses = 0
    lateness = 0
    need(isinstance(cert["request_outcomes"], list), "request_outcomes must be an array")
    if not isinstance(cert["request_outcomes"], list):
        return schema_failures + failures
    for outcome in cert["request_outcomes"]:
        need(isinstance(outcome, dict) and set(outcome) == outcome_fields,
             "invalid request outcome fields")
        if not isinstance(outcome, dict):
            continue
        rid = outcome.get("request_id")
        valid_rid = isinstance(rid, str) and rid in req_by_id
        need(valid_rid and rid not in outcome_by_id, f"duplicate/unknown outcome {rid}")
        need(_is_int(outcome.get("terminal_finish_us"))
             and outcome.get("terminal_finish_us", -1) >= 0,
             f"invalid terminal finish for {rid}")
        need(outcome.get("outcome") in ("MET", "TARDY"), f"invalid outcome for {rid}")
        need(_is_int(outcome.get("lateness_us")) and outcome.get("lateness_us", -1) >= 0,
             f"invalid lateness for {rid}")
        if valid_rid and rid not in outcome_by_id:
            outcome_by_id[rid] = outcome
    for rid, request in req_by_id.items():
        need(rid in outcome_by_id, f"missing outcome for {rid}")
        if rid not in outcome_by_id:
            continue
        finish = node_action[request["terminal_node"]]["finish_us"]
        late = max(0, finish - request["deadline_us"])
        expected = "MET" if late == 0 else "TARDY"
        outcome = outcome_by_id[rid]
        need(outcome["terminal_finish_us"] == finish, f"terminal finish mismatch for {rid}")
        need(outcome["lateness_us"] == late and outcome["outcome"] == expected,
             f"outcome mismatch for {rid}")
        misses += int(late > 0)
        lateness += late

    server_energy, server_intervals = _server_energy(inst, actions)
    phone_energy = 0
    phone_by_device = defaultdict(int)
    for action in actions:
        device = action["device"]
        if device == "SERVER":
            continue
        duration = action["finish_us"] - action["start_us"]
        part = devices[device]["active_mw"] * duration
        part += sum(node_by_id[nid]["routes"][device]["extra_energy_nj"]
                    for nid in action["members"])
        phone_energy += part
        phone_by_device[device] += part
    total_energy = server_energy + phone_energy
    expected_energy = {
        "server_nj": server_energy,
        "phone_nj": phone_energy,
        "phone_by_device_nj": dict(sorted(phone_by_device.items())),
        "total_nj": total_energy,
        "server_p0_intervals": server_intervals,
    }
    energy = cert["energy"]
    energy_fields = {
        "server_nj", "phone_nj", "phone_by_device_nj", "total_nj",
        "server_p0_intervals",
    }
    need(isinstance(energy, dict) and set(energy) == energy_fields,
         "invalid energy fields")
    if isinstance(energy, dict):
        for key in ("server_nj", "phone_nj", "total_nj"):
            need(_is_int(energy.get(key)) and energy.get(key, -1) >= 0,
                 f"invalid energy.{key}")
        by_device = energy.get("phone_by_device_nj")
        need(isinstance(by_device, dict)
             and all(isinstance(name, str) and _is_int(value) and value >= 0
                     for name, value in by_device.items()),
             "invalid energy.phone_by_device_nj")
        intervals = energy.get("server_p0_intervals")
        need(isinstance(intervals, list)
             and all(isinstance(interval, list) and len(interval) == 2
                     and all(_is_int(value) and value >= 0 for value in interval)
                     for interval in intervals),
             "invalid energy.server_p0_intervals")
    need(cert["energy"] == expected_energy, "energy decomposition mismatch")
    objective = [misses, lateness, -(len(req_by_id) - misses), total_energy]
    need(isinstance(cert["objective"], list) and len(cert["objective"]) == 4
         and all(_is_int(value) for value in cert["objective"]),
         "invalid objective")
    need(cert["objective"] == objective, "objective mismatch")

    search = cert["search"]
    need(isinstance(search, dict)
         and set(search) == {"complete"},
         "invalid search record fields")
    if isinstance(search, dict):
        need(isinstance(search.get("complete"), bool), "invalid search.complete")
        if require_complete:
            need(search.get("complete") is True, "search.complete must be true")
    if require_complete and not schema_failures and not failures:
        # The signed completeness marker is not proof. reference.py re-derives the
        # optimum from the instance alone.
        try:
            proven = reference.optimum(inst)
        except reference.ReferenceOutOfDomain as exc:
            failures.append(
                "optimality is not independently verifiable inside the reference domain "
                f"({exc}); no optimum is certified")
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            failures.append(f"independent optimum reference failed: {exc}")
        else:
            claimed = tuple(cert["objective"])
            if claimed > proven:
                failures.append(
                    f"certificate is feasible but SUBOPTIMAL: objective {list(claimed)} is worse "
                    f"than the independently proven optimum {list(proven)}")
            elif claimed < proven:
                failures.append(
                    f"certificate objective {list(claimed)} is better than the independently "
                    f"proven optimum {list(proven)}; accounting is inconsistent")
    return schema_failures + failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--certificate", required=True)
    parser.add_argument("--feasibility-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        inst = load_strict(args.instance)
        cert = load_strict(args.certificate)
        failures = check(inst, cert, require_complete=not args.feasibility_only)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, DuplicateKeyError,
            TypeError, KeyError, AttributeError) as exc:
        print(f"CHECK_FAIL: {exc}", file=sys.stderr)
        return 2
    if failures:
        for failure in failures:
            print(f"CHECK_FAIL: {failure}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(json.dumps({"feasible": True, "instance_id": inst["instance_id"],
                          "objective": cert["objective"],
                          "mode": "feasibility_only" if args.feasibility_only
                                  else "exact_certificate",
                          "optimality_verified": not args.feasibility_only},
                         sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
