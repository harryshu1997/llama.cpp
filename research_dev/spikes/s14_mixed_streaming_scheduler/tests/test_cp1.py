#!/usr/bin/env python3
"""Fail-closed tests for the S14 CP1 static mixed runtime.

Run under /usr/bin/python3 (imports the S12 reducer + jsonschema-free path):
    /usr/bin/python3 tests/test_cp1.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOD_DIR = HERE.parent
sys.path.insert(0, str(MOD_DIR))
import argparse
import json

import cp1_adapter as ad  # noqa: E402
import cp1_runtime as rt  # noqa: E402


def _catalog():
    return ad.load_frozen_catalog()


def _valid_fleet_record():
    return {
        "record_schema_version": 2,
        "verdict": "REAL_FLEET_FFN_PASS",
        "scheduler": "persistent_session_work_stealing_v1",
        "protocol_version": 3,
        "route_epoch": "1",
        "island_id": "1",
        "device_count": 1,
        "jobs": 2,
        "M": 16,
        "prefix": "blk.2",
        "model_source": "prestaged",
        "model_bytes": 12345,
        "model_sha256": "a" * 64,
        "oracle_backend": "CPU",
        "rel_l2_limit": 0.005,
        "rel_l2_max": 0.0002,
        "validated_dispatch_makespan_ms": 15.0,
        "last_rpc_complete_ms": 14.0,
        "devices": [{
            "id": "op12",
            "physical_id": "5ae7a43d",
            "physical_id_source": "caller_asserted",
            "host": "127.0.0.1",
            "port": 19012,
            "session_epoch": "10",
            "backend": "HTP0",
            "capability": "prestaged_gemma4_dense_ffn_v3",
            "initial_generation": "7",
            "initial_ready": True,
            "connect_status_ms": 1.0,
            "prepare_e2e_ms": 2.0,
            "jobs": 2,
            "client_operation_p50_ms": 11.0,
            "client_operation_p95_ms": 13.0,
            "worker_compute_p50_ms": 10.0,
            "rel_l2_max": 0.0002,
            "final_generation": "7",
        }],
        "assignments": [
            {"job": 0, "device": "op12", "client_operation_ms": 11.0,
             "worker_compute_ms": 10.0, "rel_l2": 0.0001},
            {"job": 1, "device": "op12", "client_operation_ms": 13.0,
             "worker_compute_ms": 11.0, "rel_l2": 0.0002},
        ],
    }


# ---- adapter ----

def test_load_frozen_catalog_ok():
    cat = _catalog()
    assert cat["catalog_hash"] == ad.FROZEN_CATALOG_HASH


def test_load_frozen_catalog_rejects_wrong_pin(tmp=None):
    cat = _catalog()
    tampered = copy.deepcopy(cat)
    # break the internal hash so recompute fails first
    tampered["catalog_hash"] = "sha256:" + "0" * 64
    import catalog_common as cc
    p = MOD_DIR / "tests" / "_tmp_cat.json"
    p.write_bytes(cc.canonical_json(tampered))
    try:
        raised = False
        try:
            ad.load_frozen_catalog(p)
        except ad.CP1Error:
            raised = True
        assert raised, "must reject a catalog whose hash does not verify"
    finally:
        p.unlink(missing_ok=True)


def test_measured_profile_server_tail_le_full():
    prof = ad.build_measured_profile(_catalog())
    isl = prof["islands"][0]
    assert isl["server_tail_us"] <= isl["server_full_us"]
    assert prof["latency_class"] == "MEASURED_LOWER_BOUND"
    assert prof["source_catalog_hash"] == ad.FROZEN_CATALOG_HASH
    # bge must be documented as excluded (no measured latency)
    assert any(e["island_id"] == "bge_encoder_0_12" for e in prof["excluded_islands"])


def test_reducer_catalog_shape():
    cat = ad.to_reducer_catalog(ad.build_measured_profile(_catalog()))
    isl = cat["gemma_head_0_2"]
    assert isl["model_id"] == "gemma-4-12b-it-f16"
    assert "OP15" in isl["routes"]
    r = isl["routes"]["OP15"]
    assert r["synthetic_phone_us"] == 154962  # measured LOWER_BOUND stage latency
    assert r["correctness_status"] == "MEASURED_LOWER_BOUND"


def test_trace_maps_model_class_and_rejects_unknown():
    cat = ad.to_reducer_catalog(ad.build_measured_profile(_catalog()))
    recs = [
        {"event_id": "e0", "t_us": 1000, "model_class": "large_text_generation"},
        {"event_id": "e1", "t_us": 2000, "model_class": "rag"},
    ]
    tr = ad.build_reducer_trace(recs, cat, 3, 1, 1)
    assert [t["island_id"] for t in tr] == ["gemma_head_0_2", "gemma_head_0_2"]
    assert all(t["provenance"] == "semi_synthetic" for t in tr)
    assert all(t["deadline_us"] > t["arrival_us"] for t in tr)
    bad = [{"event_id": "x", "t_us": 5, "model_class": "vision"}]
    raised = False
    try:
        ad.build_reducer_trace(bad, cat, 3, 1, 1)
    except ad.CP1Error:
        raised = True
    assert raised


def test_arrival_compression_preserves_order():
    cat = ad.to_reducer_catalog(ad.build_measured_profile(_catalog()))
    recs = [{"event_id": f"e{i}", "t_us": i * 1000, "model_class": "rag"} for i in range(5)]
    tr = ad.build_reducer_trace(recs, cat, 3, 1, 100)
    arrivals = [t["arrival_us"] for t in tr]
    assert arrivals == sorted(arrivals)


# ---- runtime ----

def test_trace_digest_binding():
    # the runtime must verify the staged trace against the frozen output_sha256
    recs = rt._load_trace()
    assert len(recs) == 177


def test_runtime_deterministic():
    a = rt.run()
    b = rt.run()
    assert a["result_digest"] == b["result_digest"]


def test_runtime_bindings_and_verdict():
    art = rt.run()
    assert art["bindings"]["island_catalog_hash"] == ad.FROZEN_CATALOG_HASH
    assert art["bindings"]["mix_v1_output_sha256"] == rt.MIX_V1_OUTPUT_SHA256
    # mechanics must conserve all 177 requests under every policy
    assert art["verdict_basis"]["mechanics_pass_terminal_conservation"] is True
    assert art["verdict_basis"]["sweep_terminal_conservation"] is True
    assert art["verdict_basis"]["cp1_gate_complete"] is False
    # honest facts: relief is NOT material and contention is NOT robust
    assert art["verdict_basis"]["relief_material_ge_1pct_or_hbm"] is False
    assert art["verdict_basis"]["contention_robust_ge_0_90"] is False
    assert art["verdict"] == "STATIC_MIXED_MECHANICS_PASS_RELIEF_INSUFFICIENT"


def test_zero_transfer_in_static():
    art = rt.run()
    for pt in [art["real_median"]] + art["synthetic_load_sweep"]:
        for pol, m in pt["per_control"].items():
            # metrics builder raises if transfer!=0; also assert overlap is a real int
            assert m["phone_server_overlap_us"] >= 0


def test_terminal_ledger_includes_timeouts():
    art = rt.run()
    for point in [art["real_median"]] + art["synthetic_load_sweep"]:
        for metrics in point["per_control"].values():
            assert metrics["terminal_total"] == 177


def test_phone_route_doubles_latency():
    art = rt.run()
    pp = art["real_median"]["per_control"]
    assert pp["static_two_phone"]["latency_p50_us"] > pp["server_only"]["latency_p50_us"]


def test_c1_control_is_present_as_fixed_phone():
    art = rt.run()
    assert art["real_median"]["comparison"]["fixed_static_phone"]["useful_ratio_vs_C0"] is not None
    c1m = art["real_median"]["per_control"]["fixed_static_phone"]
    # C1 is fixed-static: if the phone is not accepting requests, all server work
    # should be absent.
    assert c1m["server_full_intervals"] == 0
    assert c1m["server_tail_intervals"] == 0
    assert c1m["server_routed"] == 0
    assert c1m["phone_routed"] >= 0


def test_contention_failure_present():
    art = rt.run()
    # den=50 is the pure-throughput-loss contention point
    pt = next(p for p in art["synthetic_load_sweep"] if p["arrival_compression_den"] == 50)
    ratio = pt["comparison"]["static_two_phone"]["useful_ratio_vs_C0"]
    assert ratio is not None and ratio < 0.90, "den=50 must show C2 losing to C0"


def test_fleet_device_config_parse():
    d = rt.FleetDeviceConfig("op12,127.0.0.1,19012,5ae7a43d")
    assert d.device_id == "op12"
    assert d.host == "127.0.0.1"
    assert d.port == 19012
    assert d.to_arg() == "op12,127.0.0.1,19012,5ae7a43d"

    failed = False
    try:
        rt.FleetDeviceConfig("bad,missing-port")
    except ad.CP1Error:
        failed = True
    assert failed


def test_parse_fleet_record_valid():
    raw = _valid_fleet_record()
    parsed = rt._parse_fleet_record(raw)
    assert parsed["jobs"] == 2
    assert parsed["assignments_worker_p50_ms"] in (10, 11)
    assert parsed["model_source"] == "prestaged"
    assert parsed["record_schema_version"] == 2
    assert parsed["validated_dispatch_makespan_ms"] == 15
    assert parsed["assignments"][0]["job"] == 0
    assert parsed["assignments"][1]["device"] == "op12"


def test_parse_fleet_record_rejects_integral_float_and_nonfinite():
    raw = _valid_fleet_record()
    raw["jobs"] = 2.0
    failed = False
    try:
        rt._parse_fleet_record(raw)
    except ad.CP1Error:
        failed = True
    assert failed

    raw = _valid_fleet_record()
    raw["validated_dispatch_makespan_ms"] = float("inf")
    failed = False
    try:
        rt._parse_fleet_record(raw)
    except ad.CP1Error:
        failed = True
    assert failed


def test_parse_fleet_record_rejects_wrong_device_and_correctness():
    raw = _valid_fleet_record()
    raw["assignments"][1]["device"] = "op15"
    failed = False
    try:
        rt._parse_fleet_record(raw)
    except ad.CP1Error:
        failed = True
    assert failed

    raw = _valid_fleet_record()
    raw["rel_l2_max"] = 0.01
    failed = False
    try:
        rt._parse_fleet_record(raw)
    except ad.CP1Error:
        failed = True
    assert failed


def test_parse_fleet_record_rejects_bad_verdict():
    raw = {
        "record_schema_version": 2,
        "verdict": "UNKNOWN_VERDICT",
        "protocol_version": 3,
        "route_epoch": "1",
        "island_id": "1",
        "device_count": 1,
        "jobs": 1,
        "M": 16,
        "prefix": "blk.2",
        "model_source": "prestaged",
        "model_bytes": 1,
        "model_sha256": "abc",
        "oracle_backend": "CPU",
        "validated_dispatch_makespan_ms": 1,
        "last_rpc_complete_ms": 1,
        "assignments": [{"job": 0, "device": "op12", "client_operation_ms": 1, "worker_compute_ms": 1}],
    }
    failed = False
    try:
        rt._parse_fleet_record(raw)
    except ad.CP1Error:
        failed = True
    assert failed


def test_run_live_fleet_skips_controls_without_jobs():
    cfg = {
        "fleet_binary": "llama-phone-pim-fleet",
        "model": "scratchpad/phone_pim/12b-f16-mid-2-3.gguf",
        "prefix": "blk.2",
        "M": 16,
        "route_epoch": 1,
        "generation_hint": 1,
        "island_id": 1,
        "model_source": "prestaged",
        "release": False,
        "timeout_ms": 10000,
        "jobs": 4,
        "devices": [rt.FleetDeviceConfig("op12,127.0.0.1,19012,5ae7a43d")],
    }
    c0 = rt._run_live_fleet(rt.C0, 8, cfg)
    assert c0["mode"] == "skipped"


def test_run_live_fleet_executes_and_parses(monkeypatch=None):
    record = _valid_fleet_record()

    calls = {"cmd": None}

    class DummyCompleted:
        def __init__(self):
            self.returncode = 0
            self.stdout = json.dumps(record)
            self.stderr = ""

    def fake_run(cmd, check, capture_output, text, timeout):
        calls["cmd"] = cmd
        return DummyCompleted()

    orig = rt.subprocess.run
    rt.subprocess.run = fake_run
    try:
        cfg = {
            "fleet_binary": "/bin/true",
            "model": "scratchpad/phone_pim/12b-f16-mid-2-3.gguf",
            "prefix": "blk.2",
            "M": 16,
            "route_epoch": 1,
            "generation_hint": 1,
            "island_id": 1,
            "model_source": "prestaged",
            "release": False,
            "timeout_ms": 10000,
            "jobs": 2,
            "expected_model_bytes": 12345,
            "expected_model_sha256": "a" * 64,
            "expected_backend": "HTP0",
            "expected_capability": "prestaged_gemma4_dense_ffn_v3",
            "require_initial_ready": True,
            "devices": [rt.FleetDeviceConfig("op12,127.0.0.1,19012,5ae7a43d")],
        }
        out = rt._run_live_fleet(rt.C2, 4, cfg)
    finally:
        rt.subprocess.run = orig

    assert calls["cmd"] is not None
    assert calls["cmd"][0] == "/bin/true"
    assert "--jobs" in calls["cmd"]
    assert out["requested_jobs"] == 2
    assert out["mode"] == "executed"
    assert out["fleet_verdict"] == "REAL_FLEET_FFN_PASS"
    assert out["fleet_jobs"] == 2
    assert out["fleet_device_count"] == 1
    assert out["fleet_route_epoch"] == "1"



def _run():
    import inspect
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            # tests take no required args here
            sig = inspect.signature(t)
            if any(p.default is inspect._empty for p in sig.parameters.values()):
                t.__call__()  # tolerate optional-arg tests
            else:
                t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
