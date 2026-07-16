#!/usr/bin/env python3
"""Deterministically generate the synthetic MECHANICS_ONLY evidence fixtures.

EVERY artifact here has provenance=SYNTHETIC, so this bundle can only ever
support a MECHANICS_ONLY instance. No number below is a measurement and none may
be promoted to a physical claim; the contract enforces that mechanically
(E_PROVENANCE), not by convention.

The mechanics deliberately mirror the two frozen v2 counterexamples, so the
evidence-bound v3 path must reproduce the same optima:

  transition_v3 -> [0,0,-2,147250000]  (the delayed, window-merging placement)
  activation_v3 -> [0,0,-2,2974000]    (earliest is infeasible at peak 200 > 150)

Run with --check to assert the on-disk fixtures are byte-identical to a fresh
generation, which keeps the fixtures deterministic across processes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evidence")]

import canon  # noqa: E402
import validator  # noqa: E402

OUT = ROOT / "fixtures" / "evidence"

SYNTH_SHA = "0" * 63 + "1"
REV = "933c722f6"
BUILD = "synthetic-mechanics-build-0"


def artifact(aid, device, tool="s10_v0r_mechanics_generator"):
    return {
        "artifact_id": aid,
        "kind": "JSON",
        "path": f"synthetic/{aid}.json",
        "sha256": canon.digest({"synthetic_artifact": aid}),
        "producing_tool": tool,
        "command": f"tests/make_evidence_fixtures.py --emit {aid}",
        "source_revision": REV,
        "build_revision": BUILD,
        "device_id": device,
        "backend_build": BUILD,
        "timestamp_utc": "20260715T000000Z",
        "validity_us": 3600000000,
        "provenance": "SYNTHETIC",
    }


def seal(record):
    record["record_sha256"] = canon.record_digest(record)
    return record


def correctness(rid, island, graph, model, weights, device, artifacts):
    return seal({
        "record_id": rid,
        "kind": "CorrectnessCertificate",
        "record_version": 3,
        "island_id": island,
        "graph_id": graph,
        "model_id": model,
        "weight_set_id": weights,
        "shape_envelope": {"tokens_min": 1, "tokens_max": 8, "kv_min": 0,
                           "kv_max": 4096},
        "device_id": device,
        "backend_build": BUILD,
        "kernel_id": f"synthetic_kernel_{device.lower()}",
        "route_kind": "ACCELERATED",
        "reference_route": {"reference_kind": "CPU_FP32",
                            "device_id": "REFERENCE_CPU",
                            "backend_build": BUILD,
                            "kernel_id": "synthetic_reference_kernel",
                            "artifact_id": "art.reference.cpu"},
        "metric": "REL_L2",
        "threshold_ppm": 5000,
        "observed_ppm": 291,
        "no_fallback_proof": {"method": "OP_TRACE",
                              "artifact_id": artifacts[0],
                              "fallback_ops_observed": 0},
        "verdict": "PASS",
        "artifacts": artifacts + ["art.reference.cpu"],
        "status": "PASS",
        "reason_code": "SYNTHETIC_MECHANICS_FIXTURE",
        "record_sha256": "",
    })


def thermal(rid, device, artifacts):
    return seal({
        "record_id": rid,
        "kind": "ThermalInterferenceProfile",
        "record_version": 3,
        "device_id": device,
        "backend_build": BUILD,
        "thermal_state": "STEADY",
        "envelope": {"temp_min_c": 30, "temp_max_c": 60},
        "duration_us": 60000000,
        "co_runners": [],
        "slowdown_ppm": 0,
        "validity_us": 3600000000,
        "artifacts": artifacts,
        "status": "PASS",
        "reason_code": "SYNTHETIC_MECHANICS_FIXTURE",
        "record_sha256": "",
    })


def route(rid, island, graph, model, weights, device, p95, batch_size,
          correctness_id, thermal_id, boundary_id, artifacts, memory_bytes=1,
          layer_class="FFN", attention_class="NONE"):
    return seal({
        "record_id": rid,
        "kind": "RouteProfile",
        "record_version": 3,
        "island_id": island,
        "model_id": model,
        "weight_set_id": weights,
        "graph_id": graph,
        "layer_class": layer_class,
        "attention_class": attention_class,
        "device_id": device,
        "backend_build": BUILD,
        "kernel_id": f"synthetic_kernel_{device.lower()}",
        "boundary_id": boundary_id,
        "shape_envelope": {"tokens_min": 1, "tokens_max": 8, "kv_min": 0,
                           "kv_max": 4096},
        "batch_size": batch_size,
        "latency_scope": "END_TO_END",
        "latency": {"p50_us": max(1, p95 - 1), "p95_us": p95, "max_us": p95 + 1},
        "process_count": 8,
        "sample_count": 64,
        "memory_bytes": memory_bytes,
        "h2d_bytes": 0,
        "d2h_bytes": 0,
        "correctness_id": correctness_id,
        "thermal_id": thermal_id,
        "artifacts": artifacts,
        "status": "PASS",
        "reason_code": "SYNTHETIC_MECHANICS_FIXTURE",
        "record_sha256": "",
    })


def bound(rid, output_bytes, energy_nj, artifacts, island, graph, model,
          weights, device, direction="INTRA_HOST", transport="NONE",
          energy_scope="SERVER_WALL"):
    return seal({
        "record_id": rid,
        "kind": "BoundaryProfile",
        "record_version": 3,
        "island_id": island,
        "model_id": model,
        "weight_set_id": weights,
        "graph_id": graph,
        "device_id": device,
        "backend_build": BUILD,
        "direction": direction,
        "transport_domain": transport,
        "h2d_bytes": 0,
        "d2h_bytes": 0,
        "output_bytes": output_bytes,
        "transfer_us": 0,
        "verification_us": 0,
        "materialization_us": 0,
        "prepare_us": 0,
        "warmup_us": 0,
        "boundary_wall_us": 0,
        "contention_state": "ISOLATED",
        "energy_nj": energy_nj,
        # A zero is only evidence if it was measured AT a boundary. energy_scope
        # NONE would make this an unmeasured zero, which the contract refuses for
        # any MEASURED instance (E_UNKNOWN_AS_ZERO).
        "energy_scope": energy_scope,
        "process_count": 8,
        "sample_count": 64,
        "artifacts": artifacts,
        "status": "PASS",
        "reason_code": "SYNTHETIC_MECHANICS_FIXTURE",
        "record_sha256": "",
    })


def power(rid, device, scope, idle_mw, active_mw, wake_us, idle_entry_us,
          transition_nj, artifacts, included=None, excluded=None,
          instrument="synthetic_meter", instrument_kind="SYNTHETIC",
          sync="SHARED_CLOCK"):
    return seal({
        "record_id": rid,
        "kind": "PowerProfile",
        "record_version": 3,
        "scope": scope,
        "accounting_model": "SERVER_PSTATE" if device == "SERVER" else "ACTIVE_ONLY",
        "device_id": device,
        "instrument_kind": instrument_kind,
        "instrument": instrument,
        "synchronization": sync,
        "sample_rate_hz": 1000,
        "sample_count": 4096,
        "idle_mw": idle_mw,
        "active_mw": active_mw,
        "wake_us": wake_us,
        "idle_entry_us": idle_entry_us,
        "transition_nj": transition_nj,
        "uncertainty_mw": 0,
        "included_rails": included if included is not None else
                          [f"{device}/CPU", f"{device}/DRAM"],
        "excluded_rails": excluded if excluded is not None else
                          ["SERVER/USB_VBUS"],
        "validity_us": 3600000000,
        "artifacts": artifacts,
        "status": "PASS",
        "reason_code": "SYNTHETIC_MECHANICS_FIXTURE",
        "record_sha256": "",
    })


def build_bundle():
    artifacts = [
        artifact("art.server.trace", "SERVER"),
        artifact("art.server.power", "SERVER"),
        artifact("art.op15.trace", "OP15"),
        artifact("art.op15.power", "OP15"),
        artifact("art.boundary", "SERVER"),
        artifact("art.boundary.op15", "OP15"),
        artifact("art.reference.cpu", "REFERENCE_CPU"),
    ]
    correctness_records = [
        correctness("corr.m0.server", "island.ffn", "graph.g0", "m0", "w0",
                    "SERVER", ["art.server.trace"]),
        correctness("corr.m1.server", "island.ffn", "graph.g1", "m1", "w1",
                    "SERVER", ["art.server.trace"]),
        correctness("corr.cap.server", "island.capacity", "graph.cap", "mcap",
                    "wcap", "SERVER", ["art.server.trace"]),
        correctness("corr.a.p0.server", "island.ffn", "graph.a", "m0", "wp",
                    "SERVER", ["art.server.trace"]),
        correctness("corr.a.p1.server", "island.ffn", "graph.a", "m1", "wp",
                    "SERVER", ["art.server.trace"]),
        correctness("corr.a.c0.op15", "island.ffn", "graph.a", "m0", "w0",
                    "OP15", ["art.op15.trace"]),
        correctness("corr.a.c1.op15", "island.ffn", "graph.a", "m1", "w1",
                    "OP15", ["art.op15.trace"]),
    ]
    thermal_records = [
        thermal("thr.server", "SERVER", ["art.server.trace"]),
        thermal("thr.op15", "OP15", ["art.op15.trace"]),
    ]
    routes = [
        route("route.n0.server", "island.ffn", "graph.g0", "m0", "w0", "SERVER",
              100, 1, "corr.m0.server", "thr.server", "bnd.n0",
              ["art.server.trace"]),
        route("route.n1.server", "island.ffn", "graph.g1", "m1", "w1", "SERVER",
              100, 1, "corr.m1.server", "thr.server", "bnd.n1",
              ["art.server.trace"]),
        # The activation bound binds a designated capacity-probe route. See
        # EVIDENCE_CONTRACT.md section 2: this is an acknowledged proxy for a
        # device activation-memory capacity record, which does not exist yet.
        route("route.capacity.1", "island.capacity", "graph.cap", "mcap", "wcap",
              "SERVER", 1, 1, "corr.cap.server", "thr.server",
              "bnd.capacity", ["art.server.trace"], memory_bytes=1,
              layer_class="CAPACITY_PROBE"),
        route("route.capacity.150", "island.capacity", "graph.cap", "mcap", "wcap",
              "SERVER", 1, 1, "corr.cap.server", "thr.server",
              "bnd.capacity", ["art.server.trace"], memory_bytes=150,
              layer_class="CAPACITY_PROBE"),
        route("route.a.p0.server", "island.ffn", "graph.a", "m0", "wp", "SERVER",
              4, 1, "corr.a.p0.server", "thr.server", "bnd.a.p0",
              ["art.server.trace"]),
        route("route.a.p1.server", "island.ffn", "graph.a", "m1", "wp", "SERVER",
              4, 1, "corr.a.p1.server", "thr.server", "bnd.a.p1",
              ["art.server.trace"]),
        route("route.a.c0.op15", "island.ffn", "graph.a", "m0", "w0", "OP15",
              6, 1, "corr.a.c0.op15", "thr.op15", "bnd.a.c0",
              ["art.op15.trace"]),
        route("route.a.c1.op15", "island.ffn", "graph.a", "m1", "w1", "OP15",
              6, 1, "corr.a.c1.op15", "thr.op15", "bnd.a.c1",
              ["art.op15.trace"]),
    ]
    # One boundary record per node. A shared record cannot carry two nodes'
    # identities, and identity is now enforced for boundary bindings too.
    boundaries = [
        bound("bnd.n0", 0, 0, ["art.boundary"], "island.ffn", "graph.g0", "m0",
              "w0", "SERVER"),
        bound("bnd.n1", 0, 0, ["art.boundary"], "island.ffn", "graph.g1", "m1",
              "w1", "SERVER"),
        bound("bnd.capacity", 0, 0, ["art.boundary"], "island.capacity",
              "graph.cap", "mcap", "wcap", "SERVER"),
        bound("bnd.a.p0", 100, 0, ["art.boundary"], "island.ffn", "graph.a", "m0",
              "wp", "SERVER"),
        bound("bnd.a.p1", 100, 0, ["art.boundary"], "island.ffn", "graph.a", "m1",
              "wp", "SERVER"),
        bound("bnd.a.c0", 0, 0, ["art.boundary.op15"], "island.ffn", "graph.a",
              "m0", "w0", "OP15", direction="HOST_TO_PHONE", transport="USB3"),
        bound("bnd.a.c1", 0, 0, ["art.boundary.op15"], "island.ffn", "graph.a",
              "m1", "w1", "OP15", direction="HOST_TO_PHONE", transport="USB3"),
    ]
    power_records = [
        power("pwr.server", "SERVER", "SERVER_WALL", 25000, 300000, 50, 50,
              1000000, ["art.server.power"]),
        power("pwr.server.zero", "SERVER", "SERVER_WALL", 25000, 300000, 0, 0, 0,
              ["art.server.power"]),
        power("pwr.op15", "OP15", "SERVER_WALL", 0, 2000, 0, 0, 0,
              ["art.op15.power"], included=["OP15/SOC", "OP15/DRAM"],
              excluded=["SERVER/USB_VBUS"]),
    ]
    bundle = {
        "schema_version": 3,
        "bundle_id": "s10_v0r_mechanics_only",
        "provenance": "MECHANICS_ONLY",
        "evaluation_timestamp_utc": "20260715T000000Z",
        "artifacts": artifacts,
        "correctness": correctness_records,
        "routes": routes,
        "boundaries": boundaries,
        "thermal": thermal_records,
        "power": power_records,
        "bundle_sha256": "",
    }
    bundle["bundle_sha256"] = canon.bundle_digest(bundle)
    return bundle


def _records(bundle):
    out = {}
    for section, kind in validator.RECORD_SECTIONS:
        if kind is None:
            continue
        for record in bundle[section]:
            out[record["record_id"]] = record
    return out


def make_binding(bundle, target, record_id, field, value):
    record = _records(bundle)[record_id]
    return {"target": target, "record_id": record_id,
            "record_sha256": record["record_sha256"], "field": field,
            "value": value}


def transition_instance(bundle):
    """Mirrors fixtures/transition_delay_counterexample.json mechanics exactly."""
    inst = {
        "schema_version": 3,
        "instance_id": "transition_delay_counterexample_v3",
        "horizon_us": 2000,
        "activation_mem_bound_bytes": 1,
        "server_power": {"p8_mw": 25000, "p0_mw": 300000, "wake_us": 50,
                         "idle_entry_us": 50, "transition_nj": 1000000},
        "devices": {"SERVER": {"kind": "server", "active_mw": 300000}},
        "batch_profiles": {},
        "requests": [
            {"id": "r0", "arrival_us": 0, "terminal_node": "n0",
             "deadline_us": 950, "priority": 0},
            {"id": "r1", "arrival_us": 0, "terminal_node": "n1",
             "deadline_us": 1500, "priority": 0},
        ],
        "nodes": [
            {"id": "n0", "request_id": "r0", "model_id": "m0", "weight_set_id": "w0",
             "graph_id": "graph.g0", "island_id": "island.ffn", "predecessors": [],
             "release_us": 0,
             "routes": {"SERVER": {"duration_us": 100, "extra_energy_nj": 0}},
             "batch_key": None, "tokens": 1, "kv_tokens": 0, "output_bytes": 0},
            {"id": "n1", "request_id": "r1", "model_id": "m1", "weight_set_id": "w1",
             "graph_id": "graph.g1", "island_id": "island.ffn", "predecessors": [],
             "release_us": 1000,
             "routes": {"SERVER": {"duration_us": 100, "extra_energy_nj": 0}},
             "batch_key": None, "tokens": 1, "kv_tokens": 0, "output_bytes": 0},
        ],
        "evidence": {"schema_version": 3, "scope": "MECHANICS_ONLY",
                     "bundle_sha256": bundle["bundle_sha256"], "bindings": []},
    }
    plan = [
        ("activation_mem_bound_bytes", "route.capacity.1", "memory_bytes", 1),
        ("server_power.p8_mw", "pwr.server", "idle_mw", 25000),
        ("server_power.p0_mw", "pwr.server", "active_mw", 300000),
        ("server_power.wake_us", "pwr.server", "wake_us", 50),
        ("server_power.idle_entry_us", "pwr.server", "idle_entry_us", 50),
        ("server_power.transition_nj", "pwr.server", "transition_nj", 1000000),
        ("devices.SERVER.active_mw", "pwr.server", "active_mw", 300000),
        ("nodes.n0.output_bytes", "bnd.n0", "output_bytes", 0),
        ("nodes.n0.routes.SERVER.duration_us", "route.n0.server",
         "latency.p95_us", 100),
        ("nodes.n0.routes.SERVER.extra_energy_nj", "bnd.n0", "energy_nj", 0),
        ("nodes.n1.output_bytes", "bnd.n1", "output_bytes", 0),
        ("nodes.n1.routes.SERVER.duration_us", "route.n1.server",
         "latency.p95_us", 100),
        ("nodes.n1.routes.SERVER.extra_energy_nj", "bnd.n1", "energy_nj", 0),
    ]
    inst["evidence"]["bindings"] = [make_binding(bundle, *entry) for entry in plan]
    return inst


def activation_instance(bundle):
    """Mirrors fixtures/activation_delay_counterexample.json mechanics exactly."""
    inst = {
        "schema_version": 3,
        "instance_id": "activation_delay_counterexample_v3",
        "horizon_us": 30,
        "activation_mem_bound_bytes": 150,
        "server_power": {"p8_mw": 25000, "p0_mw": 300000, "wake_us": 0,
                         "idle_entry_us": 0, "transition_nj": 0},
        "devices": {"SERVER": {"kind": "server", "active_mw": 300000},
                    "OP15": {"kind": "phone", "active_mw": 2000}},
        "batch_profiles": {},
        "requests": [
            {"id": "r0", "arrival_us": 0, "terminal_node": "c0",
             "deadline_us": 16, "priority": 0},
            {"id": "r1", "arrival_us": 0, "terminal_node": "c1",
             "deadline_us": 24, "priority": 0},
        ],
        "nodes": [
            {"id": "p0", "request_id": "r0", "model_id": "m0", "weight_set_id": "wp",
             "graph_id": "graph.a", "island_id": "island.ffn", "predecessors": [],
             "release_us": 0,
             "routes": {"SERVER": {"duration_us": 4, "extra_energy_nj": 0}},
             "batch_key": None, "tokens": 1, "kv_tokens": 0,
             "output_bytes": 100},
            {"id": "c0", "request_id": "r0", "model_id": "m0", "weight_set_id": "w0",
             "graph_id": "graph.a", "island_id": "island.ffn",
             "predecessors": ["p0"], "release_us": 0,
             "routes": {"OP15": {"duration_us": 6, "extra_energy_nj": 0}},
             "batch_key": None, "tokens": 1, "kv_tokens": 0, "output_bytes": 0},
            {"id": "p1", "request_id": "r1", "model_id": "m1", "weight_set_id": "wp",
             "graph_id": "graph.a", "island_id": "island.ffn", "predecessors": [],
             "release_us": 0,
             "routes": {"SERVER": {"duration_us": 4, "extra_energy_nj": 0}},
             "batch_key": None, "tokens": 1, "kv_tokens": 0,
             "output_bytes": 100},
            {"id": "c1", "request_id": "r1", "model_id": "m1", "weight_set_id": "w1",
             "graph_id": "graph.a", "island_id": "island.ffn",
             "predecessors": ["p1"], "release_us": 0,
             "routes": {"OP15": {"duration_us": 6, "extra_energy_nj": 0}},
             "batch_key": None, "tokens": 1, "kv_tokens": 0, "output_bytes": 0},
        ],
        "evidence": {"schema_version": 3, "scope": "MECHANICS_ONLY",
                     "bundle_sha256": bundle["bundle_sha256"], "bindings": []},
    }
    plan = [
        ("activation_mem_bound_bytes", "route.capacity.150", "memory_bytes", 150),
        ("server_power.p8_mw", "pwr.server.zero", "idle_mw", 25000),
        ("server_power.p0_mw", "pwr.server.zero", "active_mw", 300000),
        ("server_power.wake_us", "pwr.server.zero", "wake_us", 0),
        ("server_power.idle_entry_us", "pwr.server.zero", "idle_entry_us", 0),
        ("server_power.transition_nj", "pwr.server.zero", "transition_nj", 0),
        ("devices.SERVER.active_mw", "pwr.server.zero", "active_mw", 300000),
        ("devices.OP15.active_mw", "pwr.op15", "active_mw", 2000),
    ]
    for nid, device, duration, out_bytes, record, boundary_record in (
            ("p0", "SERVER", 4, 100, "route.a.p0.server", "bnd.a.p0"),
            ("c0", "OP15", 6, 0, "route.a.c0.op15", "bnd.a.c0"),
            ("p1", "SERVER", 4, 100, "route.a.p1.server", "bnd.a.p1"),
            ("c1", "OP15", 6, 0, "route.a.c1.op15", "bnd.a.c1")):
        plan.append((f"nodes.{nid}.output_bytes", boundary_record,
                     "output_bytes", out_bytes))
        plan.append((f"nodes.{nid}.routes.{device}.duration_us", record,
                     "latency.p95_us", duration))
        plan.append((f"nodes.{nid}.routes.{device}.extra_energy_nj",
                     boundary_record, "energy_nj", 0))
    inst["evidence"]["bindings"] = [make_binding(bundle, *entry) for entry in plan]
    return inst


def emit(path, payload):
    return path, json.dumps(payload, indent=2, sort_keys=True) + "\n"


def generate():
    bundle = build_bundle()
    outputs = [
        emit(OUT / "mechanics_bundle.json", bundle),
        emit(OUT / "transition_v3.json", transition_instance(bundle)),
        emit(OUT / "activation_v3.json", activation_instance(bundle)),
    ]
    for descriptor in bundle["artifacts"]:
        payload = {"synthetic_artifact": descriptor["artifact_id"]}
        outputs.append((OUT / descriptor["path"],
                        canon.canonical(payload).decode("ascii")))
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="assert on-disk fixtures match a fresh generation")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for path, text in generate():
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.check:
            current = path.read_text(encoding="ascii")
            if current != text:
                print(f"FIXTURE_DRIFT: {path} differs from a fresh generation",
                      file=sys.stderr)
                return 1
        else:
            path.write_text(text, encoding="ascii")
    print("S10_V0R_EVIDENCE_FIXTURES_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
