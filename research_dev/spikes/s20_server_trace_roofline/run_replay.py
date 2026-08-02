#!/usr/bin/env python3
"""Replay an observed BurstGPT cohort through one llama-server GPU."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any
import urllib.error
import urllib.request


class ReplayError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def http_json(url: str, body: dict[str, Any] | None = None,
              timeout: float = 5.0) -> Any:
    data = None if body is None else canonical_json(body).encode("ascii")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_text(url: str, timeout: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def parse_prometheus(payload: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            continue
        name = fields[0].split("{", 1)[0]
        try:
            values[name] = float(fields[1])
        except ValueError:
            continue
    return values


def stream_completion(url: str, body: dict[str, Any], timeout: float,
                      on_first_token: Any) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=canonical_json(body).encode("ascii"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    final: dict[str, Any] | None = None
    saw_token = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if not isinstance(chunk, dict):
                raise ReplayError("stream chunk must be an object")
            if "error" in chunk:
                raise ReplayError(f"server stream error: {chunk['error']}")
            if chunk.get("stop", False):
                final = chunk
                continue
            has_token = bool(chunk.get("tokens")) or "content" in chunk
            if has_token and not saw_token:
                saw_token = True
                on_first_token()
    if not saw_token:
        raise ReplayError("stream returned no generated token")
    if final is None:
        raise ReplayError("stream returned no final result")
    return final


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self._stream = path.open("w", encoding="ascii", newline="\n")
        self._lock = threading.Lock()

    def write(self, value: dict[str, Any]) -> None:
        line = canonical_json(value)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            self._stream.close()


class Nvtx:
    def __init__(self) -> None:
        self._lib = ctypes.CDLL("libnvToolsExt.so")
        self._lib.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        self._lib.nvtxRangePushA.restype = ctypes.c_int
        self._lib.nvtxRangePop.argtypes = []
        self._lib.nvtxRangePop.restype = ctypes.c_int
        self._lib.nvtxMarkA.argtypes = [ctypes.c_char_p]
        self._lib.nvtxMarkA.restype = None

    def push(self, name: str) -> None:
        self._lib.nvtxRangePushA(name.encode("ascii"))

    def pop(self) -> None:
        self._lib.nvtxRangePop()

    def mark(self, name: str) -> None:
        self._lib.nvtxMarkA(name.encode("ascii"))


def load_cohort(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="ascii") as stream:
        cohort = json.load(stream)
    if cohort.get("schema") != "s15-burst-b32-cohort-v1":
        raise ReplayError("unexpected cohort schema")
    requests = cohort.get("requests")
    if not isinstance(requests, list) or len(requests) != 32:
        raise ReplayError("cohort must contain exactly 32 requests")
    first = min(item["observed_t_us"] for item in requests)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in requests:
        event_id = item.get("event_id")
        if not isinstance(event_id, str) or event_id in seen:
            raise ReplayError("invalid or duplicate event_id")
        seen.add(event_id)
        n_input = item.get("observed_input_tokens")
        n_output = item.get("observed_output_tokens")
        if type(n_input) is not int or n_input <= 0:
            raise ReplayError(f"invalid input length for {event_id}")
        if type(n_output) is not int or n_output <= 0:
            raise ReplayError(f"invalid output length for {event_id}")
        normalized.append({
            "event_id": event_id,
            "source_row_id": item["source_row_id"],
            "arrival_us": item["observed_t_us"] - first,
            "input_tokens": n_input,
            "output_tokens": n_output,
        })
    normalized.sort(key=lambda item: (item["arrival_us"], item["source_row_id"]))
    return cohort, normalized


def wait_for_server(base_url: str, process: subprocess.Popen[bytes],
                    timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ReplayError(f"llama-server exited during load with rc={process.returncode}")
        try:
            health = http_json(base_url + "/health", timeout=1.0)
            if health.get("status") == "ok":
                return
            last_error = f"health status: {health!r}"
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise ReplayError(f"server readiness timeout: {last_error}")


def summarize_slots(slots: Any) -> dict[str, int]:
    if not isinstance(slots, list):
        raise ReplayError("/slots did not return a list")
    active = 0
    prefill = 0
    decode = 0
    for slot in slots:
        if not isinstance(slot, dict) or not slot.get("is_processing", False):
            continue
        active += 1
        next_token = slot.get("next_token", {})
        if isinstance(next_token, list):
            if len(next_token) != 1 or not isinstance(next_token[0], dict):
                raise ReplayError("invalid next_token list in /slots")
            next_token = next_token[0]
        if not isinstance(next_token, dict):
            raise ReplayError("invalid next_token in /slots")
        n_decoded = next_token.get("n_decoded", 0)
        if type(n_decoded) is not int or n_decoded < 0:
            raise ReplayError("invalid n_decoded in /slots")
        if n_decoded == 0:
            prefill += 1
        else:
            decode += 1
    return {"active_slots": active, "prefill_slots": prefill, "decode_slots": decode}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="ascii")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--port", type=int, default=18120)
    parser.add_argument("--sample-ms", type=int, default=100)
    parser.add_argument("--server-timeout-s", type=float, default=180.0)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    args = parser.parse_args()

    if args.sample_ms < 50 or args.sample_ms > 1000:
        raise ReplayError("sample-ms must be in [50, 1000]")
    if not args.server.is_file() or not os.access(args.server, os.X_OK):
        raise ReplayError("llama-server is missing or not executable")
    if not args.model.is_file() or not args.cohort.is_file():
        raise ReplayError("model or cohort is missing")

    model_hash = file_sha256(args.model)
    if model_hash != args.model_sha256:
        raise ReplayError(f"model digest mismatch: {model_hash}")
    cohort, requests = load_cohort(args.cohort)
    cohort_hash = file_sha256(args.cohort)

    args.output.mkdir(parents=True, exist_ok=False)
    events = JsonlWriter(args.output / "events.jsonl")
    samples = JsonlWriter(args.output / "runtime_samples.jsonl")
    server_stdout = (args.output / "server.stdout.log").open("wb")
    server_stderr = (args.output / "server.stderr.log").open("wb")

    base_url = f"http://127.0.0.1:{args.port}"
    command = [
        str(args.server),
        "--model", str(args.model),
        "--n-gpu-layers", "all",
        "--split-mode", "none",
        "--main-gpu", "0",
        "--device", "CUDA0",
        "--fit", "off",
        "--ctx-size", "24576",
        "--parallel", "32",
        "--batch-size", "4096",
        "--ubatch-size", "512",
        "--flash-attn", "on",
        "--cont-batching",
        "--kv-unified",
        "--no-cache-idle-slots",
        "--cache-type-k", "f16",
        "--cache-type-v", "f16",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--metrics",
        "--slots",
        "--no-webui",
        "--log-timestamps",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    process = subprocess.Popen(
        command,
        stdout=server_stdout,
        stderr=server_stderr,
        env=environment,
        start_new_session=True,
    )

    stop_sampler = threading.Event()
    go = threading.Event()
    replay_start_ns = [0]
    sampler_errors: list[str] = []
    request_errors: list[str] = []
    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()
    nvtx = Nvtx()

    def relative_ns() -> int:
        start = replay_start_ns[0]
        return 0 if start == 0 else max(0, time.monotonic_ns() - start)

    def sampler() -> None:
        while not stop_sampler.is_set():
            started = time.monotonic_ns()
            try:
                slot_values = summarize_slots(http_json(base_url + "/slots", timeout=10.0))
                samples.write({
                    "schema": "s20-runtime-sample-v1",
                    "t_ns": relative_ns(),
                    "probe_elapsed_ns": time.monotonic_ns() - started,
                    **slot_values,
                })
            except Exception as exc:  # persisted and fail-closed below
                sampler_errors.append(str(exc))
                samples.write({
                    "schema": "s20-runtime-sample-error-v1",
                    "t_ns": relative_ns(),
                    "error": str(exc),
                })
            elapsed = (time.monotonic_ns() - started) / 1e9
            stop_sampler.wait(max(0.0, args.sample_ms / 1000.0 - elapsed))

    def request_worker(item: dict[str, Any], token_id: int) -> None:
        go.wait()
        target_ns = replay_start_ns[0] + item["arrival_us"] * 1000
        while True:
            remaining = target_ns - time.monotonic_ns()
            if remaining <= 0:
                break
            time.sleep(min(remaining / 1e9, 0.01))
        start_ns = relative_ns()
        events.write({
            "schema": "s20-request-event-v1",
            "kind": "request_start",
            "t_ns": start_ns,
            **item,
        })
        try:
            first_token_ns = [None]

            def on_first_token() -> None:
                first_token_ns[0] = relative_ns()
                events.write({
                    "schema": "s20-request-event-v1",
                    "kind": "first_token",
                    "t_ns": first_token_ns[0],
                    "event_id": item["event_id"],
                })

            response = stream_completion(
                base_url + "/completion",
                {
                    "prompt": [token_id] * item["input_tokens"],
                    "n_predict": item["output_tokens"],
                    "ignore_eos": True,
                    "cache_prompt": False,
                    "temperature": 0.0,
                    "seed": item["source_row_id"],
                    "timings_per_token": False,
                    "stream": True,
                },
                args.request_timeout_s,
                on_first_token,
            )
            timings = response.get("timings", {})
            prompt_n = timings.get("prompt_n")
            predicted_n = timings.get("predicted_n")
            if prompt_n != item["input_tokens"]:
                raise ReplayError(
                    f"{item['event_id']}: prompt_n={prompt_n}, expected={item['input_tokens']}"
                )
            if predicted_n != item["output_tokens"]:
                raise ReplayError(
                    f"{item['event_id']}: predicted_n={predicted_n}, expected={item['output_tokens']}"
                )
            result = {
                "event_id": item["event_id"],
                "status": "completed",
                "start_ns": start_ns,
                "first_token_ns": first_token_ns[0],
                "end_ns": relative_ns(),
                "prompt_n": prompt_n,
                "predicted_n": predicted_n,
                "prompt_ms": timings.get("prompt_ms"),
                "predicted_ms": timings.get("predicted_ms"),
            }
            with results_lock:
                results.append(result)
            events.write({
                "schema": "s20-request-event-v1",
                "kind": "request_end",
                "t_ns": result["end_ns"],
                **result,
            })
        except Exception as exc:
            message = f"{item['event_id']}: {exc}"
            request_errors.append(message)
            events.write({
                "schema": "s20-request-event-v1",
                "kind": "request_error",
                "t_ns": relative_ns(),
                "event_id": item["event_id"],
                "error": str(exc),
            })

    sampler_thread: threading.Thread | None = None
    range_open = False
    try:
        wait_for_server(base_url, process, args.server_timeout_s)
        props = http_json(base_url + "/props", timeout=5.0)
        tokenize = http_json(base_url + "/tokenize", {"content": " measurement"}, timeout=5.0)
        token_ids = tokenize.get("tokens")
        if not isinstance(token_ids, list) or not token_ids or type(token_ids[-1]) is not int:
            raise ReplayError("failed to acquire a valid deterministic payload token")
        token_id = token_ids[-1]

        sampler_thread = threading.Thread(target=sampler, name="runtime-sampler", daemon=True)
        sampler_thread.start()
        workers = [
            threading.Thread(target=request_worker, args=(item, token_id), daemon=True)
            for item in requests
        ]
        for worker in workers:
            worker.start()

        replay_start_ns[0] = time.monotonic_ns()
        nvtx.push("MEASURED_REPLAY")
        range_open = True
        events.write({
            "schema": "s20-replay-event-v1",
            "kind": "replay_start",
            "t_ns": 0,
            "token_id": token_id,
        })
        go.set()
        for worker in workers:
            worker.join(args.request_timeout_s + 30.0)
            if worker.is_alive():
                request_errors.append("request worker did not terminate")

        time.sleep(args.sample_ms / 1000.0 * 2)
        events.write({
            "schema": "s20-replay-event-v1",
            "kind": "replay_end",
            "t_ns": relative_ns(),
        })
        nvtx.pop()
        range_open = False
        stop_sampler.set()
        if sampler_thread is not None:
            sampler_thread.join(5.0)
            if sampler_thread.is_alive():
                raise ReplayError("sampler did not terminate")
        if request_errors:
            raise ReplayError("; ".join(request_errors))
        if len(results) != len(requests):
            raise ReplayError("not all requests produced completion evidence")

        manifest = {
            "schema": "s20-server-trace-run-v1",
            "verdict": "RAW_REPLAY_PASS_ANALYSIS_PENDING",
            "source_scope": "OBSERVED_ARRIVALS_AND_TOKEN_COUNTS_SYNTHETIC_TOKEN_VALUES",
            "cohort_path": str(args.cohort.resolve()),
            "cohort_sha256": cohort_hash,
            "cohort_declared_scope": cohort.get("scope"),
            "model_path": str(args.model.resolve()),
            "model_sha256": model_hash,
            "server_path": str(args.server.resolve()),
            "server_sha256": file_sha256(args.server),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "gpu_physical_index": args.gpu_index,
            "gpu_uuid": args.gpu_uuid,
            "server_command": command,
            "request_count": len(results),
            "input_tokens": sum(item["input_tokens"] for item in requests),
            "output_tokens": sum(item["output_tokens"] for item in requests),
            "replay_wall_ns": max(item["end_ns"] for item in results),
            "runtime_sampler_errors": sampler_errors,
            "server_props": props,
            "results": sorted(results, key=lambda item: item["event_id"]),
        }
        atomic_json(args.output / "run_manifest.json", manifest)
        return 0
    finally:
        go.set()
        stop_sampler.set()
        if range_open:
            nvtx.pop()
        if sampler_thread is not None and sampler_thread.is_alive():
            sampler_thread.join(2.0)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10.0)
        events.close()
        samples.close()
        server_stdout.close()
        server_stderr.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReplayError as exc:
        print(f"S20_REPLAY_ERROR: {exc}", flush=True)
        raise SystemExit(2)
