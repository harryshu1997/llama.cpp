#!/usr/bin/env python3
"""Run scheduler and historical S42 tests in isolated Python processes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCHEDULER_TEST_ROOT = HERE
REPO_ROOT = HERE.parents[2]
S42_TEST_ROOT = (
    HERE.parents[2]
    / "research_dev/spikes/s42_general_energy_scheduler_v1/tests"
)
TEST_NAMES = (
    "test_adoption_energy_screen.py",
    "test_adoption_qualification.py",
    "test_mixed_model_trace.py",
    "test_calibrate_profiles.py",
    "test_live_probe.py",
    "test_physical_artifacts.py",
    "test_kernel_energy_tools.py",
    "test_ubatch_graph_adapter.py",
    "test_route_compiler.py",
    "test_runtime_gates.py",
    "test_stage6_physical_ab.py",
    "test_six_model_scheduler.py",
    "test_whole_task_phone.py",
    "test_small_model_phone.py",
    "test_small_model_overlay.py",
    "test_multi_session_phone.py",
    "test_dynamic_residency_shadow.py",
    "test_dynamic_residency_profile.py",
    "test_dynamic_residency_pair.py",
    "test_dynamic_residency_abba.py",
    "test_full_fp16_burstgpt.py",
    "test_full_fp16_small_overlay.py",
    "test_gpu_wavefront_gate.py",
    "test_phone_arbiter_wavefront.py",
    "../multi_session_phone_v1/test_phone_arbiter_probe_analysis.py",
    "../full_fp16_burstgpt_v1/shape_balance_v1/test_shape_balance.py",
    "test_gpu_tensor_manifest.py",
    "test_gpu_tensor_manifest_bundle.py",
    "test_prefetch_fence_analysis.py",
    "test_prefetch_qualification.py",
    "test_contract_adapters.py",
    "test_offload_executor.py",
)
TESTS = (
    *sorted(SCHEDULER_TEST_ROOT.glob("test_*.py")),
    *(S42_TEST_ROOT / name for name in TEST_NAMES),
)


def command(path: Path) -> list[str]:
    # Most tests import sibling test modules by bare name (script mode puts tests/ on sys.path);
    # the few with package-relative imports (from .tiny_llama_gguf) must run as modules instead.
    if path.parent == SCHEDULER_TEST_ROOT and "\nfrom ." in path.read_text():
        return [sys.executable, "-m", f"research_dev.scheduler.tests.{path.stem}"]
    return [sys.executable, str(path)]


def main() -> int:
    failed = []
    for path in TESTS:
        # Fixture modules (test_automated_runtime.py holds the shared base class) define no tests;
        # unittest exits 5 for them, which is not a failure.
        if "def test_" not in path.read_text():
            continue
        completed = subprocess.run(command(path), cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            failed.append(f"{path.name}: exit {completed.returncode}")
    for line in failed:
        print(f"FAILED {line}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
