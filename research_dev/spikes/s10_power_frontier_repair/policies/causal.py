#!/usr/bin/env python3
"""Causal snapshot helper for the repaired S10 foundation.

Only requests admitted by `now_us` are copied into the planning instance. The
helper intentionally has no API through which later requests can influence the
decision. This is a foundation invariant, not the final bounded C5 policy.
"""

from __future__ import annotations

import copy
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "oracle"))
import exact  # noqa: E402


def visible_instance(inst, now_us):
    if type(now_us) is not int or not 0 <= now_us <= inst["horizon_us"]:
        raise ValueError("now_us must be an integer within the instance horizon")
    visible_requests = [request for request in inst["requests"]
                        if request["arrival_us"] <= now_us]
    if not visible_requests:
        return None
    request_ids = {request["id"] for request in visible_requests}
    visible_nodes = [node for node in inst["nodes"] if node["request_id"] in request_ids]
    batch_keys = {node["batch_key"] for node in visible_nodes if node["batch_key"] is not None}
    out = copy.deepcopy(inst)
    out["instance_id"] = f"{inst['instance_id']}@{now_us}"
    out["requests"] = copy.deepcopy(visible_requests)
    out["nodes"] = copy.deepcopy(visible_nodes)
    for node in out["nodes"]:
        node["release_us"] = max(node["release_us"], now_us)
    out["batch_profiles"] = {key: copy.deepcopy(inst["batch_profiles"][key])
                             for key in sorted(batch_keys)}
    return out


def first_action(inst, now_us, max_states=2_000_000):
    snapshot = visible_instance(inst, now_us)
    if snapshot is None:
        return None
    certificate = exact.solve(snapshot, max_states=max_states)
    action = min(certificate["actions"],
                 key=lambda item: (item["start_us"], item["finish_us"],
                                   item["device"], item["members"]))
    if action["start_us"] < now_us:
        raise RuntimeError("causal snapshot produced an action before its decision epoch")
    return {
        "decision_time_us": now_us,
        "device": action["device"],
        "members": action["members"],
        "planned_start_us": action["start_us"],
        "planned_finish_us": action["finish_us"],
    }
