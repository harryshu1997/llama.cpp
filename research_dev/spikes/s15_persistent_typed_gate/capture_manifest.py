#!/usr/bin/env python3
"""Capture post-run executable and device identities for the typed gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
REMOTE = "/data/local/tmp/ls-s15-persistent-typed"
SERIAL = "3C15AU002CL00000"
SHARD = "/data/local/tmp/ls-npu/12b-f16-head-0-8.gguf"
SHARD_SHA256 = "a74845294bc1a3cba6ac26c747615d64d9127c904aae04c43c7ea2ebc1ea25c8"
MODEL_SHA256 = "bac4e29337273d1cb41aa36f4209ebcf63baa1162e6ab4dc55f6034b3d69b35a"
REMOTE_FILES = (
    "llama-layersplit", "libc++_shared.so", "libggml-base.so", "libggml-cpu.so",
    "libggml-hexagon.so", "libggml-htp-v81.so", "libggml-opencl.so",
    "libggml.so", "libllama-common.so", "libllama.so",
)
LOCAL_FILES = (
    "build-cuda/bin/llama-layersplit",
    "npu-harness/build/llamacpp/android-arm64-hexagon-release-eafdc75e/bin/llama-layersplit",
    "examples/layersplit/layersplit.cpp",
    "research_dev/spikes/s15_persistent_typed_gate/run_gate.py",
    "research_dev/spikes/s15_persistent_typed_gate/physical_mux.py",
    "research_dev/spikes/s15_persistent_typed_gate/validate_report.py",
    "research_dev/spikes/s15_persistent_typed_gate/capture_manifest.py",
    "research_dev/spikes/s15_persistent_live_launcher/live_contract.py",
    "research_dev/spikes/s15_persistent_live_launcher/persistent_bridge.py",
    "research_dev/spikes/s15_persistent_live_launcher/live_adapter.py",
    "research_dev/spikes/s15_persistent_runtime/persistent_transport.py",
    "research_dev/spikes/s15_persistent_runtime/session_adapter.py",
    "research_dev/spikes/s15_runtime_dispatch/executor_contract.py",
    "research_dev/spikes/s15_runtime_dispatch/physical_executor.py",
    "research_dev/spikes/s14_mixed_streaming_scheduler/power_frontier_policy.py",
    "research_dev/spikes/s15_burst_cohort/cohort.json",
    "research_dev/spikes/s15_burst_cohort/input_manifest.json",
    "research_dev/spikes/s15_batch32_gate/results/gate_report.json",
    "research_dev/spikes/s15_persistent_host_tail/results/report.json",
)


class ManifestError(RuntimeError):
    pass


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def adb_shell(command: str, timeout: int = 180) -> str:
    process = subprocess.run(
        ["adb", "-s", SERIAL, "shell", command], capture_output=True,
        timeout=timeout, check=False,
    )
    if process.returncode != 0:
        raise ManifestError(process.stderr.decode("utf-8", errors="replace").strip())
    return process.stdout.decode("ascii").strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if not args.capture:
        raise ManifestError("manifest acquisition requires --capture")
    output = RESULTS / "run_manifest.json"
    report_path = RESULTS / "report.json"
    if output.exists() or not report_path.is_file():
        raise ManifestError("report is missing or run manifest already exists")
    report = json.loads(report_path.read_bytes())
    local = {}
    for relative in LOCAL_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise ManifestError(f"local artifact is missing: {relative}")
        local[relative] = digest(path)
    model = ROOT.parent / "models/gemma-4-12B-it-f16.gguf"
    model_digest = digest(model)
    if model_digest != "sha256:" + MODEL_SHA256:
        raise ManifestError("full model digest mismatch")
    command = "cd " + REMOTE + " && sha256sum " + " ".join(REMOTE_FILES)
    remote = {}
    for line in adb_shell(command).splitlines():
        value, name = line.split(None, 1)
        remote[name] = "sha256:" + value
    if set(remote) != set(REMOTE_FILES):
        raise ManifestError("remote runtime hash set is incomplete")
    boot_id = adb_shell("cat /proc/sys/kernel/random/boot_id")
    shard_digest = adb_shell(f"sha256sum {SHARD}", 240).split()[0]
    if report.get("worker_binary_sha256") != remote["llama-layersplit"] \
            or report.get("worker_binary_sha256") != local[LOCAL_FILES[1]]:
        raise ManifestError("local/deployed worker identity mismatch")
    if report.get("host_binary_sha256") != local[LOCAL_FILES[0]] \
            or report.get("route", {}).get("device_boot_id") != boot_id \
            or shard_digest != SHARD_SHA256:
        raise ManifestError("host, boot, or phone shard identity mismatch")
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True, check=True,
    ).stdout.strip()
    manifest = {
        "schema": "s15-persistent-typed-run-manifest-v1",
        "report_sha256": digest(report_path),
        "git_head": git_head,
        "selected_gpu_uuid": "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f",
        "phone_serial": SERIAL,
        "phone_boot_id": boot_id,
        "phone_shard_path": SHARD,
        "phone_shard_sha256": "sha256:" + shard_digest,
        "full_model_path": str(model),
        "full_model_sha256": model_digest,
        "local_artifacts": local,
        "remote_runtime": remote,
        "energy_scope": "UNKNOWN",
    }
    output.write_bytes(canonical(manifest))
    print(f"CAPTURED_RUN_MANIFEST files={len(local)} remote={len(remote)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ManifestError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"MANIFEST_ERROR {exc}")
        raise SystemExit(2)
