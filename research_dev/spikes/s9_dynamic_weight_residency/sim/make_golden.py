#!/usr/bin/env python3
"""Regenerate the V0-R golden artifact (golden/v0r/baseline_sweep.v0r.manifest.json).

Run ONLY after a deliberate, reviewed mechanics change. sim/test_golden_replay.py
asserts a fresh run reproduces the stored goodput_results byte-for-byte; a mutation of
the stored file must make that test FAIL. Deterministic.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import residency_sim as R  # noqa: E402

GOLD = os.path.join(SPIKE, "golden", "v0r1")
CFG = os.path.join(HERE, "scenarios", "baseline_sweep.config.json")


def main():
    os.makedirs(GOLD, exist_ok=True)
    cfg = json.load(open(CFG))
    results = R.run_sweep(cfg)
    manifest = R.build_manifest(cfg, CFG, results)
    out = os.path.join(GOLD, "baseline_sweep.v0r1.manifest.json")
    with open(out, "w") as f:
        f.write(R.canonical(manifest) + "\n")
    print("wrote", os.path.relpath(out, SPIKE))
    print("replay:", manifest["deterministic_replay_sha256"])
    print("code_version:", manifest["code_version"])


if __name__ == "__main__":
    main()
