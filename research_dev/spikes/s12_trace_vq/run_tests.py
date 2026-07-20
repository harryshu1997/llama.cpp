#!/usr/bin/env python3
"""Run unit, CLI-negative, and cross-hash-seed determinism checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run_unit_tests() -> tuple[int, int]:
    suite = unittest.defaultTestLoader.discover(str(HERE / "tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return result.testsRun, len(result.failures) + len(result.errors)


def run_cli_negatives() -> int:
    cases = [
        [
            sys.executable,
            str(HERE / "vq_sim.py"),
            "--config",
            str(HERE / "fixtures" / "varied_arrivals.jsonl"),
        ],
        [
            sys.executable,
            str(HERE / "profile_coverage.py"),
            "--trace",
            str(HERE / "fixtures" / "varied_arrivals.jsonl"),
            "--profile",
            str(HERE / "profiles" / "s11_batched_route.json"),
            "--mode",
            "strict_real",
            "--repo-root",
            str(HERE),
            "--verify-artifacts",
        ],
        [
            sys.executable,
            str(HERE / "dual_path_vq.py"),
            "--config",
            str(HERE / "fixtures" / "varied_arrivals.jsonl"),
        ],
        [
            sys.executable,
            str(HERE / "two_level_vq.py"),
            "--config",
            str(HERE / "fixtures" / "varied_arrivals.jsonl"),
        ],
    ]
    for command in cases:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode == 0 or "Traceback" in completed.stderr:
            raise RuntimeError(
                f"CLI negative failed returncode={completed.returncode}: {' '.join(command)}"
            )
    return len(cases)


def run_seed_determinism() -> tuple[int, dict[str, str]]:
    commands = {
        "v0": (HERE / "vq_sim.py", HERE / "configs" / "shadow_fixture.json"),
        "dual": (HERE / "dual_path_vq.py", HERE / "configs" / "dual_path_shadow_fixture.json"),
        "two_level": (HERE / "two_level_vq.py", HERE / "configs" / "two_level_fixture.json"),
    }
    digests = {}
    total = 0
    for name, (program, config) in commands.items():
        hashes = []
        for seed in ("0", "1", "42", "12345", "random"):
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                [sys.executable, str(program), "--config", str(config)],
                text=True,
                capture_output=True,
                check=False,
                env=env,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"{name} seed {seed}: {completed.stderr.strip()}")
            manifest = json.loads(completed.stdout)
            hashes.append(manifest["deterministic_replay_sha256"])
            total += 1
        if len(set(hashes)) != 1:
            raise RuntimeError(f"{name} cross-seed replay mismatch: {hashes}")
        digests[name] = hashes[0]
    return total, digests


def main() -> int:
    count, failures = run_unit_tests()
    if failures:
        print(f"RESULT: FAIL unit={count} failures={failures}")
        return 1
    try:
        negative_count = run_cli_negatives()
        seeds, digests = run_seed_determinism()
    except RuntimeError as exc:
        print(f"RESULT: FAIL {exc}")
        return 1
    print(
        f"RESULT: PASS unit={count} cli_negative={negative_count} "
        f"hash_seed_runs={seeds} replay_v0={digests['v0']} replay_dual={digests['dual']} "
        f"replay_two_level={digests['two_level']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
