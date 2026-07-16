#!/usr/bin/env python3
"""Exact enumerator for bounded S10-V0-R DAGs.

Two exact modes, selected by the instance, never by convenience:

EARLIEST mode (wake_us == idle_entry_us == transition_nj == 0 and every
output_bytes == 0). The solver enumerates route assignments, every compatible
server batch partition, and every per-device action order, then constructs the
earliest schedule implied by each order and the DAG. This is exact, and the
argument is frozen in PLAN.md ("Exact temporal domain"): with zero
wake/idle/transition the server P0 windows are exactly the actions, one lane
forbids overlap, so
active_us == sum(server durations) and the merged-window count is multiplied by
transition_nj == 0. Server energy is therefore identical for every start-time
placement of a given order, and phone energy never depends on start times. The
remaining objective terms (misses, lateness, -met) are non-decreasing in every
finish time, and earliest-start simultaneously minimises every finish time of a
fixed order. Earliest-start is thus lexicographically optimal for that order.

TEMPORAL mode (any nonzero wake/idle/transition or any nonzero output_bytes).
Earliest-start is NOT exact there: intentional delay can merge two P0 windows or
move an activation lifetime out of another one's way. The solver then enumerates
every legal integer start time inside each action's feasibility window, in
addition to routes, partitions, and device orders. Feasibility windows are
derived from releases, DAG and lane precedence, and the HORIZON only -- never
from deadlines, because TARDY is a legal outcome and a deadline-derived bound
would silently discard feasible schedules.

Instances are intentionally tiny. Exceeding the declared state bound raises and
never yields an approximate result or a complete=true certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pathlib
import sys
from collections import defaultdict


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


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _is_int(value):
    return type(value) is int


def validate_instance(inst):
    required = {
        "schema_version", "instance_id", "horizon_us",
        "activation_mem_bound_bytes", "server_power", "devices",
        "batch_profiles", "requests", "nodes", "evidence",
    }
    _require(set(inst) == required, "instance fields do not match schema v2")
    _require(inst["schema_version"] == 2, "schema_version must be 2")
    _require(isinstance(inst["instance_id"], str) and inst["instance_id"], "invalid instance_id")
    _require(_is_int(inst["horizon_us"]) and inst["horizon_us"] > 0, "invalid horizon_us")
    _require(_is_int(inst["activation_mem_bound_bytes"])
             and inst["activation_mem_bound_bytes"] > 0, "invalid activation bound")

    power_fields = {"p8_mw", "p0_mw", "wake_us", "idle_entry_us", "transition_nj"}
    power = inst["server_power"]
    _require(set(power) == power_fields, "invalid server_power fields")
    for key in power_fields:
        _require(_is_int(power[key]) and power[key] >= 0, f"invalid server_power.{key}")
    _require(power["p0_mw"] >= power["p8_mw"], "P0 power must be at least P8 power")

    devices = inst["devices"]
    _require(isinstance(devices, dict) and "SERVER" in devices, "SERVER device is required")
    for name, dev in devices.items():
        _require(isinstance(name, str) and name, "invalid device name")
        _require(set(dev) == {"kind", "active_mw"}, f"invalid fields for device {name}")
        _require(dev["kind"] in ("server", "phone"), f"invalid device kind for {name}")
        _require(_is_int(dev["active_mw"]) and dev["active_mw"] >= 0,
                 f"invalid active power for {name}")
    _require(devices["SERVER"]["kind"] == "server", "SERVER must have server kind")
    _require({name for name, dev in devices.items() if dev["kind"] == "server"} == {"SERVER"},
             "SERVER must be the only server-kind device")
    _require(devices["SERVER"]["active_mw"] == power["p0_mw"],
             "SERVER active_mw must equal server_power.p0_mw")
    evidence = inst["evidence"]
    _require(isinstance(evidence, dict)
             and evidence.get("scope") == "MECHANICS_ONLY"
             and all(isinstance(key, str) and isinstance(value, str)
                     for key, value in evidence.items()),
             "evidence scope must be MECHANICS_ONLY")

    requests = inst["requests"]
    nodes = inst["nodes"]
    _require(isinstance(requests, list) and requests, "requests must be nonempty")
    _require(isinstance(nodes, list) and nodes, "nodes must be nonempty")
    _require(len(requests) <= 8 and len(nodes) <= 8,
             "foundation instances are limited to eight requests and nodes")
    request_fields = {"id", "arrival_us", "terminal_node", "deadline_us", "priority"}
    node_fields = {
        "id", "request_id", "model_id", "weight_set_id", "predecessors",
        "release_us", "routes", "batch_key", "tokens", "output_bytes",
    }
    req_by_id = {}
    for req in requests:
        _require(set(req) == request_fields, f"invalid request fields: {req.get('id')}")
        rid = req["id"]
        _require(isinstance(rid, str) and rid and rid not in req_by_id, "duplicate/invalid request id")
        _require(_is_int(req["arrival_us"]) and req["arrival_us"] >= 0,
                 f"invalid arrival for {rid}")
        _require(_is_int(req["deadline_us"])
                 and req["arrival_us"] <= req["deadline_us"] <= inst["horizon_us"],
                 f"invalid deadline for {rid}")
        _require(_is_int(req["priority"]) and req["priority"] == 0,
                 f"foundation priority must be zero for {rid}")
        req_by_id[rid] = req

    node_by_id = {}
    for node in nodes:
        _require(set(node) == node_fields, f"invalid node fields: {node.get('id')}")
        nid = node["id"]
        _require(isinstance(nid, str) and nid and nid not in node_by_id, "duplicate/invalid node id")
        _require(isinstance(node["request_id"], str) and node["request_id"] in req_by_id,
                 f"unknown request for node {nid}")
        _require(isinstance(node["model_id"], str) and node["model_id"], f"invalid model for {nid}")
        _require(isinstance(node["weight_set_id"], str) and node["weight_set_id"],
                 f"invalid weight set for {nid}")
        _require(isinstance(node["predecessors"], list)
                 and all(isinstance(pred, str) and pred for pred in node["predecessors"])
                 and len(node["predecessors"]) == len(set(node["predecessors"])),
                 f"invalid predecessors for {nid}")
        _require(_is_int(node["release_us"])
                 and node["release_us"] >= req_by_id[node["request_id"]]["arrival_us"],
                 f"invalid release for {nid}")
        _require(_is_int(node["tokens"]) and node["tokens"] > 0, f"invalid tokens for {nid}")
        _require(_is_int(node["output_bytes"]) and node["output_bytes"] >= 0,
                 f"invalid output bytes for {nid}")
        _require(node["batch_key"] is None or isinstance(node["batch_key"], str),
                 f"invalid batch key for {nid}")
        routes = node["routes"]
        _require(isinstance(routes, dict) and routes, f"node {nid} has no routes")
        for device, route in routes.items():
            _require(device in devices, f"node {nid} references unknown device {device}")
            _require(set(route) == {"duration_us", "extra_energy_nj"},
                     f"invalid route fields for {nid}/{device}")
            _require(_is_int(route["duration_us"]) and route["duration_us"] > 0,
                     f"invalid route duration for {nid}/{device}")
            _require(_is_int(route["extra_energy_nj"]) and route["extra_energy_nj"] >= 0,
                     f"invalid route energy for {nid}/{device}")
            if device == "SERVER":
                _require(route["extra_energy_nj"] == 0,
                         f"SERVER route extra energy must be zero for {nid}")
        node_by_id[nid] = node

    for node in nodes:
        nid = node["id"]
        for pred in node["predecessors"]:
            _require(pred in node_by_id and pred != nid, f"invalid predecessor {pred} for {nid}")
            _require(node_by_id[pred]["request_id"] == node["request_id"],
                     f"cross-request dependency is unsupported in foundation: {pred}->{nid}")
        if node["batch_key"] is not None and "SERVER" in node["routes"]:
            _require(node["batch_key"] in inst["batch_profiles"],
                     f"missing batch profile for {nid}")
            profile = inst["batch_profiles"][node["batch_key"]]
            _require(str(node["tokens"]) in profile, f"missing singleton batch latency for {nid}")
            _require(profile[str(node["tokens"])] == node["routes"]["SERVER"]["duration_us"],
                     f"singleton batch latency differs from SERVER route for {nid}")

    for key, profile in inst["batch_profiles"].items():
        _require(isinstance(key, str) and key and isinstance(profile, dict) and profile,
                 "invalid batch profile")
        for tokens, duration in profile.items():
            _require(tokens.isdigit() and int(tokens) > 0 and _is_int(duration) and duration > 0,
                     f"invalid batch profile entry {key}/{tokens}")

    for req in requests:
        terminal = req["terminal_node"]
        _require(isinstance(terminal, str) and terminal in node_by_id,
                 f"unknown terminal node for {req['id']}")
        _require(node_by_id[terminal]["request_id"] == req["id"],
                 f"terminal node belongs to another request: {req['id']}")

    # DAG check independent of any schedule.
    indegree = {nid: 0 for nid in node_by_id}
    successors = defaultdict(list)
    for node in nodes:
        for pred in node["predecessors"]:
            indegree[node["id"]] += 1
            successors[pred].append(node["id"])
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
    _require(visited == len(nodes), "node graph contains a cycle")

    for req in requests:
        rid = req["id"]
        terminal = req["terminal_node"]
        _require(not successors[terminal], f"terminal node is not a sink: {rid}/{terminal}")
        ancestors = set()
        stack = [terminal]
        while stack:
            nid = stack.pop()
            if nid in ancestors:
                continue
            ancestors.add(nid)
            stack.extend(node_by_id[nid]["predecessors"])
        request_nodes = {nid for nid, node in node_by_id.items() if node["request_id"] == rid}
        _require(ancestors == request_nodes,
                 f"request DAG is not closed by terminal node: {rid}")
    schema_errors = instance_schema_errors(inst)
    if schema_errors:
        raise ValueError(f"instance schema violation: {schema_errors[0]}")
    return req_by_id, node_by_id


def set_partitions(items):
    """Yield every set partition once, with canonical block ordering."""
    items = tuple(sorted(items))
    if not items:
        yield ()
        return

    def rec(index, blocks):
        if index == len(items):
            yield tuple(tuple(block) for block in blocks)
            return
        item = items[index]
        for pos in range(len(blocks)):
            blocks[pos].append(item)
            yield from rec(index + 1, blocks)
            blocks[pos].pop()
        blocks.append([item])
        yield from rec(index + 1, blocks)
        blocks.pop()

    yield from rec(0, [])


def _action_duration(inst, node_by_id, device, members):
    if device == "SERVER" and node_by_id[members[0]]["batch_key"] is not None:
        key = node_by_id[members[0]]["batch_key"]
        tokens = sum(node_by_id[nid]["tokens"] for nid in members)
        try:
            return inst["batch_profiles"][key][str(tokens)]
        except KeyError as exc:
            raise ValueError(f"missing batch duration {key}/{tokens}") from exc
    _require(len(members) == 1, "only SERVER batch-profile nodes may form a batch")
    return node_by_id[members[0]]["routes"][device]["duration_us"]


def _layouts(inst, node_by_id, placement):
    grouped = defaultdict(list)
    fixed = []
    for nid, device in placement.items():
        node = node_by_id[nid]
        if device == "SERVER" and node["batch_key"] is not None:
            grouped[node["batch_key"]].append(nid)
        else:
            fixed.append((device, (nid,)))

    partition_sets = [list(set_partitions(members)) for _, members in sorted(grouped.items())]
    if not partition_sets:
        partition_sets = [[()]]
    for choice in itertools.product(*partition_sets):
        actions = list(fixed)
        for blocks in choice:
            for block in blocks:
                if block:
                    actions.append(("SERVER", tuple(sorted(block))))
        actions.sort(key=lambda item: (item[0], item[1]))
        # Nodes in one batch must be independent and identity-compatible.
        valid = True
        for device, members in actions:
            if len(members) <= 1:
                continue
            first = node_by_id[members[0]]
            for nid in members[1:]:
                node = node_by_id[nid]
                if (device != "SERVER" or node["batch_key"] != first["batch_key"]
                        or node["model_id"] != first["model_id"]
                        or node["weight_set_id"] != first["weight_set_id"]):
                    valid = False
            member_set = set(members)
            if any(member_set.intersection(node_by_id[nid]["predecessors"]) for nid in members):
                valid = False
            try:
                _action_duration(inst, node_by_id, device, members)
            except ValueError:
                valid = False
        if valid:
            yield actions


def _schedule_layout(inst, node_by_id, actions, orders):
    action_ids = [f"a{i:02d}" for i in range(len(actions))]
    records = {}
    node_action = {}
    for aid, (device, members) in zip(action_ids, actions):
        duration = _action_duration(inst, node_by_id, device, members)
        records[aid] = {"id": aid, "device": device, "members": list(members),
                        "duration_us": duration}
        for nid in members:
            node_action[nid] = aid

    predecessors = {aid: set() for aid in action_ids}
    for aid, record in records.items():
        for nid in record["members"]:
            for pred in node_by_id[nid]["predecessors"]:
                paid = node_action[pred]
                if paid == aid:
                    return None
                predecessors[aid].add(paid)
    for device, order in orders.items():
        del device
        for before, after in zip(order, order[1:]):
            predecessors[after].add(before)

    indegree = {aid: len(preds) for aid, preds in predecessors.items()}
    successors = defaultdict(list)
    for aid, preds in predecessors.items():
        for pred in preds:
            successors[pred].append(aid)
    ready = sorted(aid for aid, degree in indegree.items() if degree == 0)
    first_server = orders.get("SERVER", (None,))[0] if orders.get("SERVER") else None
    finish = {}
    scheduled = []
    while ready:
        aid = ready.pop(0)
        record = records[aid]
        release = max(node_by_id[nid]["release_us"] for nid in record["members"])
        start = max([release] + [finish[pred] for pred in predecessors[aid]])
        if aid == first_server:
            start = max(start, inst["server_power"]["wake_us"])
        end = start + record["duration_us"]
        if end > inst["horizon_us"]:
            return None
        finish[aid] = end
        scheduled.append({"id": aid, "device": record["device"],
                          "members": record["members"], "start_us": start,
                          "finish_us": end})
        for succ in successors[aid]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                ready.append(succ)
                ready.sort()
    if len(scheduled) != len(actions):
        return None
    scheduled.sort(key=lambda a: (a["start_us"], a["finish_us"], a["device"], a["members"]))
    return scheduled


def _server_energy(inst, actions):
    power = inst["server_power"]
    horizon = inst["horizon_us"]
    windows = []
    for action in actions:
        if action["device"] != "SERVER":
            continue
        start = max(0, action["start_us"] - power["wake_us"])
        end = min(horizon, action["finish_us"] + power["idle_entry_us"])
        windows.append((start, end))
    windows.sort()
    merged = []
    for start, end in windows:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    active_us = sum(end - start for start, end in merged)
    energy = power["p8_mw"] * horizon
    energy += (power["p0_mw"] - power["p8_mw"]) * active_us
    energy += power["transition_nj"] * len(merged)
    return energy, merged


def _phone_energy(inst, node_by_id, actions):
    energy = 0
    by_device = defaultdict(int)
    for action in actions:
        if action["device"] == "SERVER":
            continue
        duration = action["finish_us"] - action["start_us"]
        device = action["device"]
        part = inst["devices"][device]["active_mw"] * duration
        part += sum(node_by_id[nid]["routes"][device]["extra_energy_nj"]
                    for nid in action["members"])
        energy += part
        by_device[device] += part
    return energy, dict(sorted(by_device.items()))


def _activation_peak(inst, node_by_id, actions):
    node_action = {nid: action for action in actions for nid in action["members"]}
    successors = defaultdict(list)
    for node in node_by_id.values():
        for pred in node["predecessors"]:
            successors[pred].append(node["id"])
    intervals = []
    for nid, node in node_by_id.items():
        if node["output_bytes"] == 0:
            continue
        # output_bytes is a whole allocation, not a streaming tensor. Count it
        # from producer execution through the last direct consumer completion.
        start = node_action[nid]["start_us"]
        end = (max(node_action[succ]["finish_us"] for succ in successors[nid])
               if successors[nid] else node_action[nid]["finish_us"])
        if end > start:
            intervals.append((start, end, node["output_bytes"], nid))
    points = sorted({value for start, end, _, _ in intervals for value in (start, end)})
    peak = 0
    for point in points:
        live = sum(size for start, end, size, _ in intervals if start <= point < end)
        peak = max(peak, live)
    return peak


def evaluate(inst, req_by_id, node_by_id, actions):
    peak = _activation_peak(inst, node_by_id, actions)
    if peak > inst["activation_mem_bound_bytes"]:
        return None
    node_action = {nid: action for action in actions for nid in action["members"]}
    misses = 0
    lateness = 0
    outcomes = []
    for rid, req in sorted(req_by_id.items()):
        finish = node_action[req["terminal_node"]]["finish_us"]
        late = max(0, finish - req["deadline_us"])
        outcome = "MET" if late == 0 else "TARDY"
        misses += int(late > 0)
        lateness += late
        outcomes.append({"request_id": rid, "terminal_finish_us": finish,
                         "outcome": outcome, "lateness_us": late})
    server_energy, intervals = _server_energy(inst, actions)
    phone_energy, phone_by_device = _phone_energy(inst, node_by_id, actions)
    total_energy = server_energy + phone_energy
    objective = [misses, lateness, -(len(req_by_id) - misses), total_energy]
    return {
        "request_outcomes": outcomes,
        "activation_peak_bytes": peak,
        "energy": {
            "server_nj": server_energy,
            "phone_nj": phone_energy,
            "phone_by_device_nj": phone_by_device,
            "total_nj": total_energy,
            "server_p0_intervals": intervals,
        },
        "objective": objective,
    }


# --- declared finite temporal-search domain (CP1) -----------------------------
# The exact temporal domain is deliberately much smaller than the schema maximum.
# Anything outside it raises; it never yields an approximate or complete=true
# result. See PLAN.md for the frozen contract.
TEMPORAL_MAX_NODES = 6
TEMPORAL_MAX_ACTIONS = 6
# An instance is inside the exact temporal domain only if the total UNPRUNED
# start-time product over every layout and device order is at most this bound.
# The bound is evaluated before any search, so an unsupported instance fails
# closed immediately instead of consuming the state cap. Branch-and-bound only
# reduces the states actually visited; it never widens the declared domain.
TEMPORAL_MAX_WINDOW_PRODUCT = 8_000_000


def temporal_mode(inst, node_by_id):
    """Return 'EARLIEST' when earliest-start is provably exact, else 'TEMPORAL'."""
    power = inst["server_power"]
    zero_transition = not (power["wake_us"] or power["idle_entry_us"]
                           or power["transition_nj"])
    zero_activation = not any(node["output_bytes"] for node in node_by_id.values())
    return "EARLIEST" if (zero_transition and zero_activation) else "TEMPORAL"


def _topological(preds, count):
    indegree = [len(preds[i]) for i in range(count)]
    succs = [[] for _ in range(count)]
    for i in range(count):
        for j in preds[i]:
            succs[j].append(i)
    ready = sorted(i for i in range(count) if indegree[i] == 0)
    order = []
    while ready:
        i = ready.pop(0)
        order.append(i)
        for s in succs[i]:
            indegree[s] -= 1
            if indegree[s] == 0:
                ready.append(s)
                ready.sort()
    if len(order) != count:
        return None, None
    return order, succs


def _latest_starts(inst, recs, succs, count, order):
    """Latest legal start per action from the HORIZON and successor chains only.

    Deadlines are deliberately excluded: TARDY is a legal terminal outcome, so a
    deadline-derived bound would discard feasible schedules and break exactness.
    Relaxed in reverse topological order so every successor is bounded first.
    """
    horizon = inst["horizon_us"]
    lst = [None] * count
    for i in reversed(order):
        bound = horizon - recs[i]["duration_us"]
        for s in succs[i]:
            bound = min(bound, lst[s] - recs[i]["duration_us"])
        lst[i] = bound
    return lst


def _temporal_layout_search(inst, req_by_id, node_by_id, recs, node_action, preds,
                            state, prune=True):
    """Exhaustive integer start-time enumeration for one (layout, order).

    Documented pruning rules (both are lexicographic LOWER bounds, so pruning can
    never remove an optimum; `prune=False` disables them for differential tests):

      R1 lateness/miss bound: placed terminal actions contribute their exact
         lateness; unplaced terminals contribute the lateness of their earliest
         still-legal finish. Delay never reduces lateness, so this bounds below.
      R2 energy bound: p8*horizon + (p0-p8)*sum(server durations) + transition_nj
         * (1 if any server action else 0) + the layout's fixed phone energy.
         Merged P0 windows always cover at least the server actions themselves and
         number at least one, and phone energy never depends on start times.
    """
    count = len(recs)
    horizon = inst["horizon_us"]
    power = inst["server_power"]
    order, succs = _topological(preds, count)
    if order is None:
        return
    lst = _latest_starts(inst, recs, succs, count, order)
    for i in range(count):
        if lst[i] < 0:
            return

    # static per-layout facts
    durations = [rec["duration_us"] for rec in recs]
    is_server = [rec["device"] == "SERVER" for rec in recs]
    release = []
    for rec in recs:
        release.append(max(node_by_id[nid]["release_us"] for nid in rec["members"]))
    # every SERVER action must leave room for its P0 wake ramp from time zero
    lb0 = []
    for i, rec in enumerate(recs):
        base = release[i]
        if is_server[i]:
            base = max(base, power["wake_us"])
        lb0.append(base)

    # Declared-domain pre-check: the unpruned start-time product for this layout
    # and order. Accumulated across the whole instance and compared before any
    # recursion, so an out-of-domain instance fails closed immediately.
    potential = 1
    for i in range(count):
        span = lst[i] - lb0[i] + 1
        if span <= 0:
            return
        potential *= span
        if potential > TEMPORAL_MAX_WINDOW_PRODUCT:
            break
    state["potential"] += potential
    if state["potential"] > TEMPORAL_MAX_WINDOW_PRODUCT:
        raise RuntimeError(
            "temporal start-time domain exceeds the declared bound "
            f"({TEMPORAL_MAX_WINDOW_PRODUCT}); refusing to emit an approximate result")

    phone_fixed = 0
    for i, rec in enumerate(recs):
        if is_server[i]:
            continue
        device = rec["device"]
        phone_fixed += inst["devices"][device]["active_mw"] * durations[i]
        phone_fixed += sum(node_by_id[nid]["routes"][device]["extra_energy_nj"]
                           for nid in rec["members"])
    server_busy = sum(durations[i] for i in range(count) if is_server[i])
    min_windows = 1 if any(is_server) else 0
    energy_floor = (power["p8_mw"] * horizon
                    + (power["p0_mw"] - power["p8_mw"]) * server_busy
                    + power["transition_nj"] * min_windows
                    + phone_fixed)

    terminal_action = {}
    for rid, req in req_by_id.items():
        terminal_action[rid] = node_action[req["terminal_node"]]
    total_requests = len(req_by_id)

    # activation intervals: producer action -> last direct consumer action
    node_succ_actions = defaultdict(set)
    for nid, node in node_by_id.items():
        for pred in node["predecessors"]:
            node_succ_actions[pred].add(node_action[nid])
    activation = []
    for nid, node in node_by_id.items():
        if node["output_bytes"] == 0:
            continue
        activation.append((node_action[nid], sorted(node_succ_actions[nid]),
                           node["output_bytes"]))
    bound_bytes = inst["activation_mem_bound_bytes"]

    starts = [None] * count

    def finish(i):
        return starts[i] + durations[i]

    def leaf():
        # activation peak
        if activation:
            intervals = []
            for prod, cons, size in activation:
                begin = starts[prod]
                end = (max(finish(c) for c in cons) if cons else finish(prod))
                if end > begin:
                    intervals.append((begin, end, size))
            peak = 0
            if intervals:
                points = sorted({p for begin, end, _ in intervals for p in (begin, end)})
                for p in points:
                    live = sum(size for begin, end, size in intervals if begin <= p < end)
                    if live > peak:
                        peak = live
            if peak > bound_bytes:
                return None
        else:
            peak = 0
        misses = 0
        lateness = 0
        for rid, req in req_by_id.items():
            late = finish(terminal_action[rid]) - req["deadline_us"]
            if late > 0:
                misses += 1
                lateness += late
        windows = []
        for i in range(count):
            if is_server[i]:
                windows.append((max(0, starts[i] - power["wake_us"]),
                                min(horizon, finish(i) + power["idle_entry_us"])))
        windows.sort()
        merged = []
        for begin, end in windows:
            if not merged or begin > merged[-1][1]:
                merged.append([begin, end])
            elif end > merged[-1][1]:
                merged[-1][1] = end
        active_us = sum(end - begin for begin, end in merged)
        energy = (power["p8_mw"] * horizon
                  + (power["p0_mw"] - power["p8_mw"]) * active_us
                  + power["transition_nj"] * len(merged)
                  + phone_fixed)
        objective = (misses, lateness, misses - total_requests, energy)
        return objective, peak

    def bound(depth):
        """Lexicographic lower bound for the partial assignment (rules R1, R2)."""
        misses_lb = 0
        lateness_lb = 0
        for rid, req in req_by_id.items():
            i = terminal_action[rid]
            pos = order.index(i)
            if pos < depth:
                end = finish(i)
            else:
                end = lb0[i] + durations[i]     # earliest still-legal finish
            late = end - req["deadline_us"]
            if late > 0:
                misses_lb += 1
                lateness_lb += late
        return (misses_lb, lateness_lb, misses_lb - total_requests, energy_floor)

    def recurse(depth):
        if depth == count:
            state["states"] += 1
            if state["states"] > state["max_states"]:
                raise RuntimeError("exact state bound exceeded; no approximate result emitted")
            got = leaf()
            if got is None:
                return
            objective, peak = got
            schedule = [{"id": recs[i]["id"], "device": recs[i]["device"],
                         "members": list(recs[i]["members"]),
                         "start_us": starts[i], "finish_us": finish(i)}
                        for i in range(count)]
            schedule.sort(key=lambda a: (a["start_us"], a["finish_us"], a["device"],
                                         a["members"]))
            key = objective + (canonical(schedule),)
            if state["best_key"] is None or key < state["best_key"]:
                state["best_key"] = key
                state["best"] = (schedule, objective, peak)
            return
        i = order[depth]
        low = lb0[i]
        for j in preds[i]:
            low = max(low, finish(j))
        high = lst[i]
        for t in range(low, high + 1):
            starts[i] = t
            if prune and state["best_key"] is not None:
                if bound(depth + 1) > state["best_key"][:4]:
                    starts[i] = None
                    continue
            recurse(depth + 1)
        starts[i] = None

    recurse(0)


def _action_records(inst, node_by_id, actions):
    recs = []
    node_action = {}
    for index, (device, members) in enumerate(actions):
        duration = _action_duration(inst, node_by_id, device, members)
        recs.append({"id": f"a{index:02d}", "device": device,
                     "members": list(members), "duration_us": duration})
        for nid in members:
            node_action[nid] = index
    return recs, node_action


def _action_precedence(node_by_id, recs, node_action, orders):
    count = len(recs)
    preds = [set() for _ in range(count)]
    for index, rec in enumerate(recs):
        for nid in rec["members"]:
            for pred in node_by_id[nid]["predecessors"]:
                source = node_action[pred]
                if source == index:
                    return None
                preds[index].add(source)
    for device, order in orders.items():
        del device
        for before, after in zip(order, order[1:]):
            preds[after].add(before)
    return preds


def _solve_temporal(inst, req_by_id, node_by_id, max_states, prune):
    if len(node_by_id) > TEMPORAL_MAX_NODES:
        raise RuntimeError(
            f"temporal exact domain is limited to {TEMPORAL_MAX_NODES} nodes; "
            "refusing to emit an approximate result")
    node_ids = sorted(node_by_id)
    route_choices = [sorted(node_by_id[nid]["routes"]) for nid in node_ids]
    state = {"best": None, "best_key": None, "states": 0, "max_states": max_states,
             "potential": 0}
    for devices in itertools.product(*route_choices):
        placement = dict(zip(node_ids, devices))
        for actions in _layouts(inst, node_by_id, placement):
            if len(actions) > TEMPORAL_MAX_ACTIONS:
                raise RuntimeError(
                    f"temporal exact domain is limited to {TEMPORAL_MAX_ACTIONS} actions; "
                    "refusing to emit an approximate result")
            recs, node_action = _action_records(inst, node_by_id, actions)
            by_device = defaultdict(list)
            for index, rec in enumerate(recs):
                by_device[rec["device"]].append(index)
            devices_sorted = sorted(by_device)
            permutation_sets = [list(itertools.permutations(by_device[device]))
                                for device in devices_sorted]
            for permutation_choice in itertools.product(*permutation_sets):
                # Count every (layout, device order) search state that is entered,
                # not only the leaves. Pruning may legitimately leave a layout with
                # zero leaves, so leaf-only counting would understate the internal
                # limit used to fail closed.
                state["states"] += 1
                if state["states"] > state["max_states"]:
                    raise RuntimeError("exact state bound exceeded; no approximate result emitted")
                orders = dict(zip(devices_sorted, permutation_choice))
                preds = _action_precedence(node_by_id, recs, node_action, orders)
                if preds is None:
                    continue
                _temporal_layout_search(inst, req_by_id, node_by_id, recs,
                                        node_action, preds, state, prune)
    if state["best"] is None:
        raise RuntimeError("no feasible complete schedule")
    schedule, objective, _peak = state["best"]
    result = evaluate(inst, req_by_id, node_by_id, schedule)
    if result is None or tuple(result["objective"]) != objective:
        raise RuntimeError("internal disagreement between temporal search and evaluator")
    cert = {
        "schema_version": 2,
        "instance_id": inst["instance_id"],
        "instance_sha256": digest(inst),
        "actions": schedule,
        "request_outcomes": result["request_outcomes"],
        "activation_peak_bytes": result["activation_peak_bytes"],
        "energy": result["energy"],
        "objective": result["objective"],
        "search": {"complete": True},
    }
    cert["certificate_sha256"] = digest(cert)
    return cert


def solve(inst, max_states=2_000_000, prune=True):
    req_by_id, node_by_id = validate_instance(inst)
    if not _is_int(max_states) or max_states <= 0:
        raise ValueError("max_states must be a positive integer")
    if len(node_by_id) > 8:
        raise RuntimeError("foundation exact solver is limited to eight nodes")
    mode = temporal_mode(inst, node_by_id)
    if mode == "TEMPORAL":
        return _solve_temporal(inst, req_by_id, node_by_id, max_states, prune)
    node_ids = sorted(node_by_id)
    route_choices = [sorted(node_by_id[nid]["routes"]) for nid in node_ids]
    best = None
    best_key = None
    states = 0
    for devices in itertools.product(*route_choices):
        placement = dict(zip(node_ids, devices))
        for actions in _layouts(inst, node_by_id, placement):
            by_device = defaultdict(list)
            for index, (device, members) in enumerate(actions):
                by_device[device].append(f"a{index:02d}")
            devices_sorted = sorted(by_device)
            permutation_sets = [list(itertools.permutations(by_device[device]))
                                for device in devices_sorted]
            for permutation_choice in itertools.product(*permutation_sets):
                states += 1
                if states > max_states:
                    raise RuntimeError("exact state bound exceeded; no approximate result emitted")
                orders = dict(zip(devices_sorted, permutation_choice))
                schedule = _schedule_layout(inst, node_by_id, actions, orders)
                if schedule is None:
                    continue
                result = evaluate(inst, req_by_id, node_by_id, schedule)
                if result is None:
                    continue
                tie = canonical(schedule)
                key = tuple(result["objective"]) + (tie,)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (schedule, result)
    if best is None:
        raise RuntimeError("no feasible complete schedule")
    schedule, result = best
    cert = {
        "schema_version": 2,
        "instance_id": inst["instance_id"],
        "instance_sha256": digest(inst),
        "actions": schedule,
        "request_outcomes": result["request_outcomes"],
        "activation_peak_bytes": result["activation_peak_bytes"],
        "energy": result["energy"],
        "objective": result["objective"],
        "search": {"complete": True},
    }
    cert["certificate_sha256"] = digest(cert)
    return cert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--out")
    parser.add_argument("--max-states", type=int, default=2_000_000)
    args = parser.parse_args()
    try:
        inst = load_strict(args.instance)
        cert = solve(inst, args.max_states)
    except (OSError, UnicodeError, ValueError, RuntimeError, TypeError, KeyError,
            AttributeError, json.JSONDecodeError) as exc:
        print(f"ORACLE_FAIL: {exc}", file=sys.stderr)
        return 2
    output = json.dumps(cert, indent=2, sort_keys=True) + "\n"
    if args.out:
        with open(args.out, "w", encoding="ascii") as handle:
            handle.write(output)
    print(json.dumps({"instance_id": inst["instance_id"], "objective": cert["objective"],
                      "certificate_sha256": cert["certificate_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
