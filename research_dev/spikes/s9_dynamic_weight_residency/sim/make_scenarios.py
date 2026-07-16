#!/usr/bin/env python3
"""Emit the committed baseline-sweep scenario (sim_config v2). Deterministic."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, "scenarios")
os.makedirs(SC, exist_ok=True)


def req(i):
    return {"arrival_us": i * 2000000, "rank": i, "request_id": f"r{i}", "service_class": "decode",
            "model_id": "gemma4-op15", "weight_set_id": "ws-op15", "input_bytes": 983040,
            "output_bytes": 983040, "state_bytes": 4096, "deadline_us": None}


cfg = {
    "schema_version": 3, "config_id": "baseline-sweep-r1", "seed": 1, "horizon_us": 120000000,
    "link_goodput_sweep_bytes_per_s": [41943040, 104857600, 262144000, 419430400, 576716800],
    "baselines": ["server_only", "per_request_fetch", "static_placement", "lru", "lfu_ttl",
                  "fastest_ready", "relief_predictive", "clairvoyant"],
    "interference": {"htp_gpu_slowdown_permille": 1000, "transfer_compute_slowdown_permille": 1000, "measured": False},
    "queues": {"prefetch_depth": 4, "server_depth": 16, "lane_depth": 8},
    "server": {"gpu_lane_slots": 4, "hbm_credit_bytes": 4000000000,
               "per_class_gpu_us": {"decode": 2000, "prefill": 8000, "embedding": 1500},
               "per_class_hbm_bytes": {"decode": 1000000, "prefill": 4000000, "embedding": 500000}},
    "phones": [{"device_id": "op15", "soc": "op15", "contention_domain": "op15-bus008", "boot_epoch": 7,
                "lpddr_total_bytes": 10000000000, "ufs_total_bytes": 128000000000,
                "ufs_write_bytes_per_s": 800000000, "verify_bytes_per_s": 1500000000,
                "materialize_bytes_per_s": 6000000000, "prepare_htp_bytes_per_s": 4000000000,
                "prepare_gpu_bytes_per_s": 3000000000, "warmup_us": 50000, "htp_lane_slots": 2,
                "gpu_lane_slots": 1, "thermal_eligible": True}],
    "models": [{"model_id": "gemma4-op15", "weight_set_id": "ws-op15", "canonical_bytes": 900000000,
                "derived_bytes_htp": 0, "derived_bytes_gpu": 900000000, "scratch_bytes": 100000000,
                "partial_load_supported": True, "eligible_backends": ["htp", "gpu"], "phone_compute_us": 3000,
                "state_policy": "stateless"}],
    "workload": {"requests": [req(i) for i in range(12)]},
    "failure_schedule": [],
    "policy_params": {"reuse_horizon_us": 120000000, "min_hold_us": 30000000, "ttl_us": 45000000,
                      "predictor_ewma_permille": 500},
}

with open(os.path.join(SC, "baseline_sweep.config.json"), "w") as f:
    f.write(json.dumps(cfg, indent=2) + "\n")
print("wrote scenarios/baseline_sweep.config.json (v3/R1; 12 requests, 5-point sweep, 8 baselines)")
