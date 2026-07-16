#!/usr/bin/env python3
"""Small independent reference for unbatched zero-transition fixtures."""

from __future__ import annotations

import itertools
from collections import defaultdict


def solve_objective(inst):
    power = inst["server_power"]
    if power["wake_us"] or power["idle_entry_us"] or power["transition_nj"]:
        raise ValueError("reference supports zero-transition fixtures only")
    nodes = {node["id"]: node for node in inst["nodes"]}
    requests = {request["id"]: request for request in inst["requests"]}
    if any(node["batch_key"] is not None or node["output_bytes"] for node in nodes.values()):
        raise ValueError("reference supports unbatched fixtures without activation storage")

    node_ids = sorted(nodes)
    best = None
    for route_vector in itertools.product(*(sorted(nodes[nid]["routes"]) for nid in node_ids)):
        actions = [(device, nid) for nid, device in zip(node_ids, route_vector)]
        by_device = defaultdict(list)
        for index, (device, _) in enumerate(actions):
            by_device[device].append(index)
        devices = sorted(by_device)
        order_sets = [itertools.permutations(by_device[device]) for device in devices]
        for order_vector in itertools.product(*order_sets):
            orders = dict(zip(devices, order_vector))
            action_for_node = {nid: index for index, (_, nid) in enumerate(actions)}
            predecessors = {index: set() for index in range(len(actions))}
            for index, (_, nid) in enumerate(actions):
                predecessors[index].update(action_for_node[pred]
                                           for pred in nodes[nid]["predecessors"])
            for order in orders.values():
                for before, after in zip(order, order[1:]):
                    predecessors[after].add(before)

            finish = {}
            remaining = set(predecessors)
            while remaining:
                ready = sorted(index for index in remaining
                               if predecessors[index].issubset(finish))
                if not ready:
                    break
                index = ready[0]
                device, nid = actions[index]
                node = nodes[nid]
                start = max([node["release_us"]]
                            + [finish[pred] for pred in predecessors[index]])
                end = start + node["routes"][device]["duration_us"]
                if end > inst["horizon_us"]:
                    break
                finish[index] = end
                remaining.remove(index)
            if remaining:
                continue

            misses = 0
            lateness = 0
            for request in requests.values():
                end = finish[action_for_node[request["terminal_node"]]]
                late = max(0, end - request["deadline_us"])
                misses += int(late > 0)
                lateness += late

            server_busy = sum(nodes[nid]["routes"][device]["duration_us"]
                              for device, nid in actions if device == "SERVER")
            server_energy = power["p8_mw"] * inst["horizon_us"]
            server_energy += (power["p0_mw"] - power["p8_mw"]) * server_busy
            phone_energy = 0
            for device, nid in actions:
                if device == "SERVER":
                    continue
                route = nodes[nid]["routes"][device]
                phone_energy += inst["devices"][device]["active_mw"] * route["duration_us"]
                phone_energy += route["extra_energy_nj"]
            objective = (
                misses,
                lateness,
                -(len(requests) - misses),
                server_energy + phone_energy,
            )
            if best is None or objective < best:
                best = objective
    if best is None:
        raise RuntimeError("reference found no feasible schedule")
    return best


def solve_single_server_batch_objective(inst):
    """Independent reference for one-node requests on one batchable server."""
    power = inst["server_power"]
    if power["wake_us"] or power["idle_entry_us"] or power["transition_nj"]:
        raise ValueError("reference supports zero-transition fixtures only")
    if set(inst["devices"]) != {"SERVER"}:
        raise ValueError("reference supports one server device only")

    nodes = {node["id"]: node for node in inst["nodes"]}
    requests = {request["id"]: request for request in inst["requests"]}
    if any(node["predecessors"] or set(node["routes"]) != {"SERVER"}
           or node["batch_key"] is None or node["output_bytes"]
           for node in nodes.values()):
        raise ValueError("unsupported batched reference fixture")
    identities = {(node["model_id"], node["weight_set_id"], node["batch_key"])
                  for node in nodes.values()}
    if len(identities) != 1:
        raise ValueError("reference requires one batch identity")
    batch_key = next(iter(identities))[2]
    profile = inst["batch_profiles"][batch_key]

    def partitions(items):
        if not items:
            yield ()
            return
        first, *rest = items
        for mask in range(1 << len(rest)):
            block = (first,) + tuple(rest[index] for index in range(len(rest))
                                     if mask & (1 << index))
            remaining = [item for index, item in enumerate(rest)
                         if not mask & (1 << index)]
            for suffix in partitions(remaining):
                yield (block,) + suffix

    best = None
    for partition in partitions(sorted(nodes)):
        durations = []
        valid = True
        for block in partition:
            tokens = sum(nodes[nid]["tokens"] for nid in block)
            duration = profile.get(str(tokens))
            if duration is None:
                valid = False
                break
            durations.append(duration)
        if not valid:
            continue
        for order in itertools.permutations(range(len(partition))):
            now = 0
            finish = {}
            busy_us = 0
            for block_index in order:
                block = partition[block_index]
                now = max([now] + [nodes[nid]["release_us"] for nid in block])
                now += durations[block_index]
                busy_us += durations[block_index]
                for nid in block:
                    finish[nid] = now
            if now > inst["horizon_us"]:
                continue

            misses = 0
            lateness = 0
            for request in requests.values():
                end = finish[request["terminal_node"]]
                late = max(0, end - request["deadline_us"])
                misses += int(late > 0)
                lateness += late
            energy = power["p8_mw"] * inst["horizon_us"]
            energy += (power["p0_mw"] - power["p8_mw"]) * busy_us
            objective = (misses, lateness, -(len(requests) - misses), energy)
            if best is None or objective < best:
                best = objective
    if best is None:
        raise RuntimeError("batched reference found no feasible schedule")
    return best
