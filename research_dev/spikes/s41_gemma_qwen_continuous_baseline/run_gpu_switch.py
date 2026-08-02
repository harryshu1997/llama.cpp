#!/usr/bin/env python3
"""Run one S41 GPU-only qualification or model-switch phase."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "s39_desktop_swap_baseline"
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(HERE))

import validate_inputs  # noqa: E402
import run_desktop_baseline as legacy  # noqa: E402


CONTRACT = HERE / "SERVER_BASELINE_CONTRACT.json"
INPUT_MANIFEST = HERE / "INPUT_MANIFEST.json"
REQUESTS = HERE / "REQUESTS.jsonl"
SWITCHES = HERE / "SWITCHES.jsonl"


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Runner(legacy.Runner):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.server = args.server.resolve()
        self.cuda_lib_dir = args.cuda_lib_dir.resolve()
        self.output = args.output.resolve()
        self.gpu_index = args.gpu_index
        self.server_timeout_s = args.server_timeout_s
        self.serving_profile = "fixed_full_cuda"
        self.contract = json.loads(CONTRACT.read_text(encoding="ascii"))
        self.requests = legacy.read_jsonl(REQUESTS)
        self.switches = legacy.read_jsonl(SWITCHES)
        self.models = {
            model_id: {**record, "path": Path(record["path"])}
            for model_id, record in self.contract["models"].items()
        }
        self.server_counter = 0
        self.current_server = None
        import threading
        self.current_pid_lock = threading.Lock()

    def preflight(self) -> dict[str, Any]:
        validate_inputs.validate(HERE)
        if not self.server.is_file() or not os.access(self.server, os.X_OK):
            raise legacy.RunError("llama-server is missing or not executable")
        if not self.cuda_lib_dir.is_dir():
            raise legacy.RunError("CUDA library directory is missing")
        if self.output.exists():
            raise legacy.RunError(f"output already exists: {self.output}")
        self.output.mkdir(parents=True)
        self.events = legacy.JsonlWriter(self.output / "events.jsonl")
        model_records: dict[str, Any] = {}
        for model_id, record in self.models.items():
            path = record["path"]
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                raise legacy.RunError(f"{model_id}: model size mismatch")
            actual = digest_file(path)
            if actual != record["sha256"]:
                raise legacy.RunError(f"{model_id}: model digest mismatch")
            model_records[model_id] = {
                "bytes": path.stat().st_size,
                "path": str(path),
                "sha256": actual,
            }
        gpu = legacy.gpu_snapshot(self.gpu_index)
        device = self.contract["device"]
        hostname = socket.gethostname()
        if hostname != device["host"] \
                or gpu["gpu_name"] != device["gpu_name"] \
                or gpu["gpu_uuid"] != device["gpu_uuid"] \
                or gpu["gpu_memory_total_bytes"] \
                != device["gpu_memory_total_bytes"]:
            raise legacy.RunError("desktop or GPU identity mismatch")
        foreign = [
            row for row in legacy.compute_processes()
            if row["pid"] != os.getpid()
            and row["used_memory_bytes"] > 128 * 1024 * 1024
        ]
        if foreign:
            raise legacy.RunError(
                f"foreign CUDA compute processes present: {foreign}")
        report = {
            "contract_sha256": digest_file(CONTRACT),
            "gpu": gpu,
            "hostname": hostname,
            "input_manifest_sha256": digest_file(INPUT_MANIFEST),
            "models": model_records,
            "server_path": str(self.server),
            "server_sha256": digest_file(self.server),
            "serving_profile": self.serving_profile,
            "system_memory": legacy.system_memory(),
        }
        (self.output / "preflight.json").write_bytes(
            legacy.canonical(report))
        return report


def run_noncoresidency(runner: Runner) -> dict[str, Any]:
    preflight = runner.preflight()
    runner.warm_both()
    models = list(runner.models)
    orders = [models, list(reversed(models))]
    attempts = []
    for order_index, (first_id, second_id) in enumerate(orders):
        first = legacy.ServerProcess(
            runner, first_id, runner.args.port + order_index * 2,
            f"pair-{order_index}-first",
        )
        runner.set_server(first)
        first_ready = first.start()
        second = legacy.ServerProcess(
            runner, second_id, runner.args.port + order_index * 2 + 1,
            f"pair-{order_index}-second",
        )
        second_ready = None
        second_error = None
        try:
            second_ready = second.start()
        except Exception as exc:
            second_error = str(exc)
        first_healthy = first.healthy()
        snapshot = legacy.gpu_snapshot(runner.gpu_index)
        second_eligible = (
            second_ready is not None
            and snapshot["gpu_memory_free_bytes"] >= 536_870_912
        )
        second.stop()
        first.stop()
        runner.set_server(None)
        if not first_healthy or second_eligible:
            raise legacy.RunError(
                f"non-co-residency failed for {first_id},{second_id}")
        attempts.append({
            "first_healthy_after_attempt": first_healthy,
            "first_model_id": first_id,
            "first_ready": first_ready,
            "gpu_after_attempt": snapshot,
            "second_eligible": second_eligible,
            "second_error": second_error,
            "second_model_id": second_id,
            "second_ready": second_ready,
        })
    report = {
        "attempts": attempts,
        "preflight": preflight,
        "schema": "s41-server-non-coresidency-v1",
        "status": "PHYSICAL_NON_CORESIDENCY_PASS",
    }
    (runner.output / "non_coresidency.json").write_bytes(
        legacy.canonical(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", required=True,
        choices=["qualify", "noncoresidency", "replay"],
    )
    parser.add_argument("--model-id")
    parser.add_argument("--regime", choices=["WARM_CACHE", "COLD_NVME"])
    parser.add_argument("--repeat-index", type=int)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=18250)
    parser.add_argument("--server-timeout-s", type=float, default=240.0)
    parser.add_argument("--request-timeout-s", type=float, default=240.0)
    args = parser.parse_args()
    models = set(json.loads(CONTRACT.read_text(encoding="ascii"))["models"])
    if args.phase == "qualify" and args.model_id not in models:
        parser.error("--model-id must name one bound model")
    if args.phase == "replay" and (
            args.regime is None or args.repeat_index is None):
        parser.error("--regime and --repeat-index are required for replay")
    return args


def main() -> int:
    args = parse_args()
    runner = Runner(args)
    try:
        if args.phase == "qualify":
            legacy.run_qualification(runner, args.model_id)
        elif args.phase == "noncoresidency":
            run_noncoresidency(runner)
        else:
            legacy.run_replay(runner, args.regime, args.repeat_index)
        return 0
    finally:
        if hasattr(runner, "events"):
            runner.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        legacy.RunError,
        legacy.cache_control.CacheError,
        validate_inputs.ValidationError,
    ) as exc:
        print(f"S41_GPU_SWITCH_ERROR: {exc}", flush=True)
        raise SystemExit(2)
