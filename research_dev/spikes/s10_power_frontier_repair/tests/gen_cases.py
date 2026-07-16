#!/usr/bin/env python3
"""Deterministic generator for tiny S10-V0-R temporal differential cases.

Frozen BEFORE results were read. Every case is derived only from its integer seed,
so the corpus is reproducible across processes and PYTHONHASHSEED values. Cases are
deliberately tiny so the independent reference can brute-force them.

Shapes covered (required by the CP4 contract):
  * unbatched single-node requests and compatible same-identity server batches;
  * multiple routes (SERVER and PHONE) so route choice matters;
  * DAG chains and a join (diamond) fork;
  * nonzero wake/idle-entry/transition costs; and
  * nonzero output_bytes with a binding activation bound.
"""

from __future__ import annotations

import random


def _power(rng):
    p8 = rng.randint(1, 3)
    p0 = p8 + rng.randint(2, 9)
    return {
        "p8_mw": p8,
        "p0_mw": p0,
        "wake_us": rng.choice([0, 0, 1, 2]),
        "idle_entry_us": rng.choice([0, 0, 1, 2]),
        "transition_nj": rng.choice([0, 5, 17, 40]),
    }


def _wrap(instance_id, horizon, power, devices, profiles, requests, nodes, bound):
    return {
        "schema_version": 2,
        "instance_id": instance_id,
        "horizon_us": horizon,
        "activation_mem_bound_bytes": bound,
        "server_power": power,
        "devices": devices,
        "batch_profiles": profiles,
        "requests": requests,
        "nodes": nodes,
        "evidence": {"scope": "MECHANICS_ONLY",
                     "purpose": "deterministic generated temporal differential case"},
    }


def _devices(rng, power, with_phone):
    devices = {"SERVER": {"kind": "server", "active_mw": power["p0_mw"]}}
    if with_phone:
        devices["PHONE"] = {"kind": "phone", "active_mw": rng.randint(1, 4)}
    return devices


def make_case(seed):
    """Return a tiny, valid schema-v2 instance for this seed."""
    rng = random.Random(seed)
    kind = seed % 4
    power = _power(rng)
    horizon = rng.randint(12, 18)
    if kind == 0:
        return _batched(seed, rng, power, horizon)
    if kind == 1:
        return _chain(seed, rng, power, horizon)
    if kind == 2:
        return _activation(seed, rng, power, horizon)
    return _diamond(seed, rng, power, horizon)


def _batched(seed, rng, power, horizon):
    """Two or three same-identity single-node server requests: compatible batches."""
    count = rng.randint(2, 3)
    unit = rng.randint(1, 3)
    profile = {}
    for tokens in range(1, count + 1):
        # sublinear batching so merging blocks can pay off
        profile[str(tokens)] = unit + (tokens - 1) * rng.randint(0, 1) + rng.randint(0, 1)
    profile["1"] = unit
    requests = []
    nodes = []
    for index in range(count):
        rid, nid = f"r{index}", f"n{index}"
        requests.append({"id": rid, "arrival_us": 0, "terminal_node": nid,
                         "deadline_us": rng.randint(4, horizon), "priority": 0})
        nodes.append({"id": nid, "request_id": rid, "model_id": "m0",
                      "weight_set_id": "w0", "predecessors": [],
                      "release_us": rng.randint(0, 2),
                      "routes": {"SERVER": {"duration_us": unit, "extra_energy_nj": 0}},
                      "batch_key": "op:w0", "tokens": 1, "output_bytes": 0})
    devices = _devices(rng, power, False)
    return _wrap(f"gen_batched_{seed}", horizon, power, devices,
                 {"op:w0": profile}, requests, nodes, 64)


def _chain(seed, rng, power, horizon):
    """One or two chains with a real route choice on each node."""
    requests = []
    nodes = []
    for index in range(rng.randint(1, 2)):
        rid = f"r{index}"
        previous = None
        length = rng.randint(1, 2)
        for depth in range(length):
            nid = f"n{index}_{depth}"
            routes = {"SERVER": {"duration_us": rng.randint(1, 3), "extra_energy_nj": 0}}
            if rng.random() < 0.7:
                routes["PHONE"] = {"duration_us": rng.randint(2, 5),
                                   "extra_energy_nj": rng.randint(0, 4)}
            nodes.append({"id": nid, "request_id": rid, "model_id": "m0",
                          "weight_set_id": f"w{depth}",
                          "predecessors": [] if previous is None else [previous],
                          "release_us": rng.randint(0, 2), "routes": routes,
                          "batch_key": None, "tokens": 1, "output_bytes": 0})
            previous = nid
        requests.append({"id": rid, "arrival_us": 0, "terminal_node": previous,
                         "deadline_us": rng.randint(4, horizon), "priority": 0})
    devices = _devices(rng, power, True)
    return _wrap(f"gen_chain_{seed}", horizon, power, devices, {}, requests, nodes, 64)


def _activation(seed, rng, power, horizon):
    """Two producer/consumer pairs with a binding activation bound."""
    size = rng.randint(40, 100)
    bound = rng.choice([size, 2 * size - 1, 2 * size])
    requests = []
    nodes = []
    for index in range(2):
        rid = f"r{index}"
        producer, consumer = f"p{index}", f"c{index}"
        nodes.append({"id": producer, "request_id": rid, "model_id": f"m{index}",
                      "weight_set_id": "wp", "predecessors": [],
                      "release_us": rng.randint(0, 1),
                      "routes": {"SERVER": {"duration_us": rng.randint(1, 3),
                                            "extra_energy_nj": 0}},
                      "batch_key": None, "tokens": 1, "output_bytes": size})
        nodes.append({"id": consumer, "request_id": rid, "model_id": f"m{index}",
                      "weight_set_id": f"w{index}", "predecessors": [producer],
                      "release_us": 0,
                      "routes": {"PHONE": {"duration_us": rng.randint(2, 4),
                                           "extra_energy_nj": rng.randint(0, 3)}},
                      "batch_key": None, "tokens": 1, "output_bytes": 0})
        requests.append({"id": rid, "arrival_us": 0, "terminal_node": consumer,
                         "deadline_us": rng.randint(6, horizon), "priority": 0})
    devices = _devices(rng, power, True)
    return _wrap(f"gen_activation_{seed}", horizon, power, devices, {},
                 requests, nodes, bound)


def _diamond(seed, rng, power, horizon):
    """A fork/join DAG: a -> {b, c} -> d, terminal d closes the request."""
    rid = "r0"
    nodes = [
        {"id": "a", "request_id": rid, "model_id": "m0", "weight_set_id": "wa",
         "predecessors": [], "release_us": 0,
         "routes": {"SERVER": {"duration_us": rng.randint(1, 2), "extra_energy_nj": 0}},
         "batch_key": None, "tokens": 1, "output_bytes": rng.choice([0, 30])},
        {"id": "b", "request_id": rid, "model_id": "m0", "weight_set_id": "wb",
         "predecessors": ["a"], "release_us": 0,
         "routes": {"SERVER": {"duration_us": rng.randint(1, 2), "extra_energy_nj": 0},
                    "PHONE": {"duration_us": rng.randint(2, 4),
                              "extra_energy_nj": rng.randint(0, 3)}},
         "batch_key": None, "tokens": 1, "output_bytes": 0},
        {"id": "c", "request_id": rid, "model_id": "m0", "weight_set_id": "wc",
         "predecessors": ["a"], "release_us": 0,
         "routes": {"SERVER": {"duration_us": rng.randint(1, 2), "extra_energy_nj": 0}},
         "batch_key": None, "tokens": 1, "output_bytes": 0},
        {"id": "d", "request_id": rid, "model_id": "m0", "weight_set_id": "wd",
         "predecessors": ["b", "c"], "release_us": 0,
         "routes": {"SERVER": {"duration_us": rng.randint(1, 2), "extra_energy_nj": 0}},
         "batch_key": None, "tokens": 1, "output_bytes": 0},
    ]
    requests = [{"id": rid, "arrival_us": 0, "terminal_node": "d",
                 "deadline_us": rng.randint(5, horizon), "priority": 0}]
    devices = _devices(rng, power, True)
    return _wrap(f"gen_diamond_{seed}", horizon, power, devices, {}, requests, nodes, 64)
