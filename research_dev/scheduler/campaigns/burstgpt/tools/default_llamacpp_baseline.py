#!/usr/bin/env python3
"""Out-of-the-box llama.cpp reference arm: ONE stock ``llama-server`` in router mode serving the campaign trace.

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.default_llamacpp_baseline run \\
        --campaign TEMPLATE/campaign.json --models TEMPLATE/models.json --rig TEMPLATE/rig.json --out RUN_DIR
    python3 -m ...default_llamacpp_baseline plan --campaign ... --out PLAN.json          (plan only, no server)
    python3 -m ...default_llamacpp_baseline plan --smoke-model small.gguf --campaign ... --out PLAN.json
    python3 -m ...default_llamacpp_baseline run --plan PLAN.json --server BIN --out RUN_DIR [--gpu-index 0 ...]

What a user gets by serving the same trace with stock llama-server and default settings. It is an ADDITIONAL
reference arm; the tuned dispatcher-only baseline stays the headline comparison.

Work (identical to the campaign client, ``adapters/http_backend.py:263-272`` + ``runner.py:1192-1203``): per trace
request ``POST /completion`` with the pre-tokenized prompt, ``n_predict`` = trace output tokens, ``ignore_eos``,
``temperature`` 0, ``seed`` = combined request index, ``stream``, ``return_tokens``, ``cache_prompt`` false, header
``X-Scheduler-Request-ID``; plus ``model`` = the preset name, which router mode needs to route a POST.

Server: ``llama-server --models-preset P --models-max 1 -lv 4 --host 127.0.0.1 --port N``; the preset only names
the three GGUF files. Every placement/performance parameter of every model instance is the llama.cpp default: GPU
layers and context fitted to free device memory (1 GiB margin, minimum context 4096), slots auto (4, unified KV),
batch 2048 / ubatch 512, flash attention auto, mmap, default threads, 8 GiB host prompt cache, CUDA graphs on.
Deviations (each one listed in RESULT ``deviations_from_defaults`` with its reason):
  * ``--models-max 1`` (default 4): the desktop has 30 GB RAM and two f16 models of 24-29.5 GB; with the default the
    first model loaded keeps the GPU for the whole run and the next ones run from mmapped host memory (see
    ``PURE_DEFAULTS_PREDICTION``).
  * client FIFO drain-before-switch gate (``--switch-gate fifo-drain``): stock router mode evicts the LRU model even
    while it streams (``server-models.cpp:834-865``), so an in-flight request of the evicted model ends silently
    (verified). Requests are released in arrival order; a request for another model than the one in use waits until
    every in-flight request has finished. ``--switch-gate none`` sends every request at its arrival (pure default).
  * ``-lv 4`` (default 3): the fitted placement (fit decisions, ``offloaded N/M layers to GPU``, ``n_ctx``) is only
    logged at trace level; no per-token output is added.

Energy: the campaign's host sampler (``adapters/host_runtime.HostEnergySampler``: nvidia-smi board power + RAPL
``package-0`` energy_uj every 0.2 s) integrated with the campaign's integrators (``_integrate_gpu`` trapezoid,
``_integrate_rapl`` counter delta) over the paid window [first arrival epoch, last completion]. Output: a RESULT.json
subset that ``tools/fleet_energy.py`` and ``tools/latency_report.py`` read unchanged, ``streams/request-NNN.raw``,
``resource-samples.jsonl``, ``server.log`` (router + every model instance) and ``MODEL_PLACEMENTS.json`` (the fitted
placement each model instance got, parsed from the log).
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import functools
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from research_dev.scheduler.adapters.host_runtime import (
    HostEnergySampler,
    HostMetricCallbacks,
    _integrate_gpu,
    _integrate_rapl,
    default_host_metric_callbacks,
    linux_host_activity,
    linux_system_memory,
    rapl_package_snapshot,
)
from research_dev.scheduler.adapters.output_quality import assess_semantic_output
from research_dev.scheduler.campaigns.burstgpt.common import canonical, digest, load_object
from research_dev.scheduler.campaigns.burstgpt.trace_inputs import (
    GEMMA_ROLE,
    QWEN_ROLE,
    apply_named_replay_schedule,
    merge_rows,
    validate_trace,
)

PLAN_SCHEMA = "ws7-default-baseline-plan-v1"
RESULT_SCHEMA = "ws7-default-llamacpp-baseline-result-v1"
PLACEMENT_SCHEMA = "ws7-default-baseline-placements-v1"
OVERLAY_ROLE = "overlay"
GATE_MODES = ("fifo-drain", "none")
DEFAULT_MODELS_MAX = 1
DEFAULT_LOG_VERBOSITY = 4
DEFAULT_PORT = 18600
MAX_SAMPLE_GAP_S = 5.0  # the campaign's RaplNvmlPhoneEnergyMeter maximum_gap_s
REQUEST_TIMEOUT_S = 3600.0  # the campaign's LlamaCppCompletionPayload.timeout_s
ENV_SCRUB_PREFIXES = ("LLAMA_ARG_", "GGML_", "S41_", "LLAMA_SERVER_", "LLAMA_APP_CMD")

# Every flag or behaviour that differs from what `llama-server` does with no arguments besides the model list.
DEVIATIONS = {
    "models_max": {
        "flag": "--models-max 1", "default": "4 (common/common.h:664)", "category": "necessity",
        "reason": "30 GB desktop RAM: with the default the first model loaded keeps its VRAM for the whole run (the "
                  "router never reaches 4 instances with 3 models), every later model is fitted into ~0 free VRAM and "
                  "runs from mmapped host memory, and Qwen f16 (29.5 GB) plus the CPU part of Gemma f16 exceed the RAM "
                  "-> weights paged from disk on every token. One resident model avoids both."},
    "switch_gate": {
        "flag": "client FIFO drain-before-switch (--switch-gate fifo-drain)", "default": "clients send at arrival",
        "category": "necessity",
        "reason": "stock router mode evicts the least recently used model without checking for in-flight requests "
                  "(tools/server/server-models.cpp:834-865); its stream ends without a final chunk and the instance is "
                  "force-killed after the 10 s stop-timeout (verified locally with --models-max 1). Requests are "
                  "released in arrival order; a request for another model waits until nothing is in flight. "
                  "Same-model requests go to the router at arrival and queue in the model instance's slots."},
    "log_verbosity": {
        "flag": "-lv 4", "default": "3", "category": "observability",
        "reason": "the fitted placement (fit decisions, 'offloaded N/M layers to GPU', n_ctx, slots) is logged at "
                  "trace level only (common/log.cpp:441-456); no per-token lines are added. Placement and performance "
                  "parameters are unchanged."},
    "request_model_field": {
        "flag": "request body field \"model\"", "default": "absent in the campaign body", "category": "necessity",
        "reason": "router mode routes every POST by its JSON field \"model\" (server-models.cpp:1888); every other "
                  "request field is the campaign's."},
}

DEFAULTS_RELIED_ON = (
    "n_gpu_layers -1 = auto: fitted to free device memory (common/common.h:472, common/fit.cpp)",
    "fit target 1024 MiB per device, minimum fitted context 4096 (common/common.h:476-480)",
    "n_ctx 0 = training context, reduced by fit only when the model does not fit (common/fit.cpp:280-340)",
    "-np -1 = auto -> 4 slots with a unified KV cache (common/arg.cpp:1171, tools/server/server.cpp:1213-1218)",
    "n_batch 2048, n_ubatch 512, flash_attn auto, use_mmap true, warmup on (common/common.h)",
    "threads -1 = llama.cpp default thread count",
    "--cache-ram 8192 MiB host prompt cache (common/common.h:627)",
    "CUDA graphs on (no GGML_CUDA_DISABLE_GRAPHS; the campaign's servers run with graphs disabled)",
    "--models-autoload on; preset stop-timeout 10 s; router HTTP timeouts 600 s",
)

INPUTS_NOT_DEVIATIONS = (
    "--models-preset: one section per model with its GGUF path only (router mode needs the model list)",
    "--host 127.0.0.1 (the default) and --port N (default 8080; the rig uses a free port)",
    "LD_LIBRARY_PATH: the deploy build's runtime libraries (rig.json library_directories + the server directory)",
    "environment scrub: LLAMA_ARG_*, GGML_*, S41_*, LLAMA_SERVER_*, LLAMA_APP_CMD and CUDA_VISIBLE_DEVICES are removed "
    "so no non-default setting leaks into the router or its model instances",
    "binary: the campaign's desktop llama-server build (rig.json binaries.server; research fork whose FFN-split hooks "
    "are inactive without S41_* variables), so build differences do not confound the configuration comparison",
    "request fields other than model: the campaign's (cache_prompt false disables the default prompt-prefix reuse, "
    "ignore_eos + n_predict fix the output length, temperature 0 + seed make the tokens deterministic)",
)

PURE_DEFAULTS_PREDICTION = (
    "Reasoned from code, not run (the 24-29.5 GB models are desktop-only). With --models-max 4 (default) and clients "
    "sending at arrival: request 000 (Gemma, t=1 s) loads Gemma alone on the GPU: fit keeps a 1 GiB margin, cuts "
    "the context to 4096 (Gemma f16 does not fit even at 4096) and fills about half of the layers (estimate ~24 of "
    "48 + output, like the tuned plan's 24); 4 slots, unified KV. "
    "Request 001 (Qwen, t=118 s) arrives while Gemma still decodes 000; the router has 1 < 4 instances, so nothing is "
    "evicted and Qwen is fitted into the ~1 GiB Gemma left free: context 4096 and ~0 GPU layers, i.e. all 29.5 GB "
    "on the CPU from mmap. Qwen's 29.5 GB plus Gemma's ~12 GB host part exceed the 30 GB RAM (fit does not look at "
    "host RAM when a GPU exists, common/fit.cpp:232-283), so the page cache thrashes and weights are re-read from "
    "disk every decode step. Llama-3.2-1B (t=1,497 s) also loads next to both, with ~0 free VRAM (CPU). The placement "
    "depends on arrival order (the first model keeps the GPU for the whole run). With --models-max 1 and no client "
    "gate, every cross-model arrival while a request streams (from t=118 s on) kills that request's stream: it ends "
    "without a final chunk, so the trace cannot complete with its output lengths."
)


# ---------------------------------------------------------------- plan (the work)

def _models_by_role(models_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = models_manifest.get("models")
    if not isinstance(rows, list) or not rows:
        raise ValueError("models manifest has no models")
    by_role: dict[str, dict[str, Any]] = {}
    for row in rows:
        role = row.get("trace_role")
        if role in by_role:
            raise ValueError("duplicate trace_role " + str(role))
        by_role[role] = row
    for role in (QWEN_ROLE, GEMMA_ROLE, OVERLAY_ROLE):
        if role not in by_role:
            raise ValueError("models manifest lacks trace_role " + role)
    return by_role


def completion_body(request: Mapping[str, Any], model_name: str) -> dict[str, Any]:
    """The campaign's /completion body (http_backend.py:263-272) plus the router's ``model`` field."""
    return {
        "cache_prompt": False,
        "ignore_eos": True,
        "model": model_name,
        "n_predict": request["output_tokens"],
        "prompt": list(request["prompt_tokens"]),
        "return_tokens": True,
        "seed": request["seed"],
        "stream": True,
        "temperature": 0.0,
    }


def _prompt_sha256(tokens: Sequence[int]) -> str:
    return "sha256:" + hashlib.sha256(canonical(list(tokens))).hexdigest()


def plan_from_trace(large_path: Path, overlay_path: Path, manifest_path: Path, schedule_path: Path,
                    models_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The campaign's replay (validate_trace + merge_rows + named schedule) with each request's executing model."""
    manifest = load_object(manifest_path)
    large, overlay = validate_trace(large_path, overlay_path, manifest)
    by_role = _models_by_role(models_manifest)
    merged = merge_rows(large, overlay, {QWEN_ROLE: by_role[QWEN_ROLE]["model_id"],
                                         GEMMA_ROLE: by_role[GEMMA_ROLE]["model_id"]})
    selected, schedule = apply_named_replay_schedule(merged, load_object(schedule_path))
    known = {row["model_id"] for row in by_role.values()}
    requests = []
    for item in selected:
        row = item["row"]
        if item["model_id"] not in known:
            raise ValueError("trace model is not in the models manifest: " + item["model_id"])
        requests.append({
            "combined_request_index": item["combined_index"],
            "input_tokens": row["input_tokens"],
            "model_id": item["model_id"],
            "output_tokens": row["output_tokens"],
            "prompt_sha256": _prompt_sha256(row["prompt_tokens"]),
            "prompt_tokens": list(row["prompt_tokens"]),
            "replay_arrival_us": row["replay_arrival_us"],
            "request_id": row["event_id"],
            "seed": item["combined_index"],
            "slo_us": row["slo_us"],
            "source": item["source"],
            "source_arrival_us": row["source_arrival_us"],
        })
    inventory = manifest.get("model_inventory", {})
    models = []
    for role in (QWEN_ROLE, GEMMA_ROLE, OVERLAY_ROLE):
        row = by_role[role]
        expected = inventory.get(row["model_id"], {})
        models.append({"artifact_bytes": expected.get("artifact_bytes"), "artifact_sha256": expected.get("artifact_sha256"),
                       "model_id": row["model_id"], "path": row["host_artifact_path"], "trace_role": role})
    return {
        "models": models,
        "replay_schedule": schedule,
        "requests": requests,
        "schema": PLAN_SCHEMA,
        "source": {
            "large_requests": {"path": str(large_path), "sha256": digest(large_path)},
            "overlay_requests": {"path": str(overlay_path), "sha256": digest(overlay_path)},
            "replay_schedule": {"path": str(schedule_path), "sha256": digest(schedule_path)},
            "trace_manifest": {"path": str(manifest_path), "sha256": digest(manifest_path)},
        },
        "trace_name": schedule.get("trace_name"),
    }


def smoke_plan(trace_plan: Mapping[str, Any], model_path: str, *, time_scale: float = 0.01,
               max_prompt_tokens: int = 48, max_output_tokens: int = 24) -> dict[str, Any]:
    """A local smoke version of a trace plan: same request order, models and seeds, arrivals compressed by
    ``time_scale``, every prompt replaced by a truncated Qwen-tokenized prompt of the trace (Qwen3 vocabulary, so a
    small Qwen3 GGUF can serve it) and outputs capped. The three models become aliases of ONE small GGUF."""
    if not 0 < time_scale <= 1 or max_prompt_tokens < 1 or max_output_tokens < 1:
        raise ValueError("smoke plan scale is invalid")
    roles = {row["model_id"]: row["trace_role"] for row in trace_plan["models"]}
    qwen_id = next(row["model_id"] for row in trace_plan["models"] if row["trace_role"] == QWEN_ROLE)
    prompts = [row["prompt_tokens"][:max_prompt_tokens] for row in trace_plan["requests"] if row["model_id"] == qwen_id]
    if not prompts:
        raise ValueError("trace plan has no Qwen-tokenized prompt")
    names = {role: "smoke-" + role for role in (QWEN_ROLE, GEMMA_ROLE, OVERLAY_ROLE)}
    first = min(row["replay_arrival_us"] for row in trace_plan["requests"])
    requests = []
    for number, row in enumerate(trace_plan["requests"]):
        prompt = list(prompts[number % len(prompts)])
        requests.append({**row,
                         "input_tokens": len(prompt),
                         "model_id": names[roles[row["model_id"]]],
                         "output_tokens": min(row["output_tokens"], max_output_tokens),
                         "prompt_sha256": _prompt_sha256(prompt),
                         "prompt_tokens": prompt,
                         "replay_arrival_us": first + round((row["replay_arrival_us"] - first) * time_scale)})
    schedule = dict(trace_plan["replay_schedule"])
    schedule["schedule"] = [{**entry, "replay_arrival_us": request["replay_arrival_us"]}
                            for entry, request in zip(schedule.get("schedule", []), requests)]
    schedule["smoke_time_scale"] = time_scale
    return {
        **trace_plan,
        "models": [{"artifact_bytes": os.path.getsize(model_path) if os.path.isfile(model_path) else None,
                    "artifact_sha256": None, "model_id": names[role], "path": model_path, "trace_role": role}
                   for role in (QWEN_ROLE, GEMMA_ROLE, OVERLAY_ROLE)],
        "replay_schedule": schedule,
        "requests": requests,
        "smoke": {"max_output_tokens": max_output_tokens, "max_prompt_tokens": max_prompt_tokens,
                  "model_path": model_path, "time_scale": time_scale},
        "trace_name": str(trace_plan.get("trace_name")) + "-smoke",
    }


def validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("plan schema")
    names = [row["model_id"] for row in plan["models"]]
    if len(set(names)) != len(names) or not names:
        raise ValueError("plan model names")
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise ValueError("plan model name is not a plain preset name: " + name)
    previous = -1
    seen = set()
    for row in plan["requests"]:
        if row["model_id"] not in names or row["request_id"] in seen:
            raise ValueError("plan request identity: " + str(row.get("request_id")))
        if len(row["prompt_tokens"]) != row["input_tokens"] or row["output_tokens"] <= 0:
            raise ValueError("plan request shape: " + row["request_id"])
        if row["replay_arrival_us"] < previous:
            raise ValueError("plan arrivals are not ordered")
        previous = row["replay_arrival_us"]
        seen.add(row["request_id"])


def preset_text(models: Sequence[Mapping[str, Any]]) -> str:
    """The router preset: one section per model with its GGUF path and nothing else."""
    lines = ["; ws7 default llama.cpp baseline: model name -> GGUF path only; every other option is the default", ""]
    for row in models:
        lines += ["[" + row["model_id"] + "]", "model = " + row["path"], ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- client-side release gate

class SwitchGate:
    """Release order of arrived requests (not thread-safe; the replay holds its lock around every call).

    ``fifo-drain``: FIFO; the head is released when nothing is in flight or when it uses the model of the requests in
    flight. ``none``: every arrival is released at once (what a client does without knowing the router evicts)."""

    def __init__(self, mode: str) -> None:
        if mode not in GATE_MODES:
            raise ValueError("switch gate mode must be one of " + ", ".join(GATE_MODES))
        self.mode = mode
        self._queue: deque[Mapping[str, Any]] = deque()
        self._in_flight = 0
        self._model: str | None = None

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def arrive(self, request: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        self._queue.append(request)
        return self._release()

    def complete(self, request: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        if self._in_flight <= 0:
            raise RuntimeError("gate completion without a request in flight")
        self._in_flight -= 1
        return self._release()

    def _release(self) -> list[Mapping[str, Any]]:
        released = []
        while self._queue:
            head = self._queue[0]
            if self.mode == "fifo-drain" and self._in_flight and head["model_id"] != self._model:
                break
            self._queue.popleft()
            self._model = head["model_id"]
            self._in_flight += 1
            released.append(head)
        return released


# ---------------------------------------------------------------- one streamed completion

def stream_completion(host: str, port: int, body: bytes, headers: Mapping[str, str], stream_path: Path,
                      timeout_s: float = REQUEST_TIMEOUT_S, clock: Callable[[], int] = time.monotonic_ns
                      ) -> dict[str, Any]:
    """POST /completion and read the SSE stream the way the campaign client does (http_backend.py:311-420)."""
    record: dict[str, Any] = {"error": None, "final": None, "first_token_ns": None, "http_status": None,
                              "sent_ns": None, "end_ns": None, "stream_sha256": None, "text": "", "tokens": []}
    tokens: list[int] = []
    parts: list[str] = []
    connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        connection.connect()
        record["sent_ns"] = clock()
        connection.request("POST", "/completion", body=body, headers=dict(headers))
        response = connection.getresponse()
        record["http_status"] = response.status
        with stream_path.open("xb") as raw:
            if response.status != 200:
                content = response.read()
                raw.write(content)
                record["error"] = "HTTP %d: %s" % (response.status, content[:400].decode("utf-8", "replace"))
            else:
                while line := response.readline():
                    raw.write(line)
                    text = line.decode("utf-8", "replace").strip()
                    if not text.startswith("data:"):
                        continue
                    encoded = text[5:].strip()
                    if not encoded or encoded == "[DONE]":
                        continue
                    value = json.loads(encoded)
                    if type(value) is not dict or "error" in value:
                        record["error"] = "stream error chunk: " + encoded[:400]
                        break
                    chunk = value.get("tokens", [])
                    if type(chunk) is not list or any(type(token) is not int for token in chunk):
                        record["error"] = "stream tokens are invalid"
                        break
                    predicted = value.get("tokens_predicted")
                    if record["first_token_ns"] is None and (
                            chunk or (type(predicted) is int and predicted > 0 and not value.get("stop", False))):
                        record["first_token_ns"] = clock()
                    tokens.extend(chunk)
                    content = value.get("content", "")
                    if type(content) is str:
                        parts.append(content)
                    if value.get("stop", False):
                        record["final"] = value
                        record["end_ns"] = clock()
        if record["final"] is None and record["error"] is None:
            record["error"] = "stream ended without a final chunk"
    except (OSError, http.client.HTTPException, ValueError) as error:
        record["error"] = "%s: %s" % (type(error).__name__, error)
    finally:
        connection.close()
        if record["end_ns"] is None:
            record["end_ns"] = clock()
    if stream_path.is_file():
        record["stream_sha256"] = digest(stream_path)
    record["tokens"] = tokens
    record["text"] = "".join(parts)
    return record


def accounting(request: Mapping[str, Any], record: Mapping[str, Any], model_name: str) -> dict[str, Any]:
    """The campaign's per-request accounting check (http_backend.py:445-455) as a verdict, not an exception."""
    final = record.get("final") or {}
    timings = final.get("timings") if isinstance(final.get("timings"), dict) else {}
    reasons = []
    if record.get("error"):
        reasons.append("CLIENT_ERROR")
    if final.get("model") != model_name:
        reasons.append("MODEL_ALIAS")
    if timings.get("prompt_n") != request["input_tokens"]:
        reasons.append("PROMPT_TOKENS")
    if timings.get("predicted_n") != request["output_tokens"]:
        reasons.append("PREDICTED_TOKENS")
    if len(record.get("tokens") or ()) != request["output_tokens"]:
        reasons.append("STREAMED_TOKENS")
    return {"exact": not reasons, "final_model": final.get("model"), "predicted_n": timings.get("predicted_n"),
            "prompt_n": timings.get("prompt_n"), "reasons": reasons, "streamed_tokens": len(record.get("tokens") or ())}


# ---------------------------------------------------------------- replay

@dataclass
class ReplayOutcome:
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    arrived_ns: dict[str, int] = field(default_factory=dict)
    released_ns: dict[str, int] = field(default_factory=dict)
    release_order: list[str] = field(default_factory=list)
    inflight_peak: dict[str, int] = field(default_factory=dict)
    aborted: str | None = None


def replay(plan: Mapping[str, Any], host: str, port: int, gate_mode: str, epoch_ns: int, streams: Path, *,
           timeout_s: float = REQUEST_TIMEOUT_S, deadline_ns: int | None = None,
           tick: Callable[[], None] | None = None, clock: Callable[[], int] = time.monotonic_ns,
           sender: Callable[..., dict[str, Any]] = stream_completion) -> ReplayOutcome:
    """Arrive every request at ``epoch_ns + replay_arrival_us``; release through the gate; one thread per request."""
    gate = SwitchGate(gate_mode)
    outcome = ReplayOutcome()
    lock = threading.Condition()
    threads: list[threading.Thread] = []
    active: dict[str, int] = {}

    def worker(request: Mapping[str, Any]) -> None:
        body = canonical(completion_body(request, request["model_id"]))
        headers = {"Content-Type": "application/json", "X-Scheduler-Request-ID": request["request_id"]}
        path = streams / ("request-%03d.raw" % request["combined_request_index"])
        try:
            record = sender(host, port, body, headers, path, timeout_s, clock)
        except BaseException as error:  # never lose a completion: the gate must advance
            record = {"error": "%s: %s" % (type(error).__name__, error), "end_ns": clock(), "tokens": [], "text": "",
                      "final": None, "first_token_ns": None, "http_status": None, "sent_ns": None,
                      "stream_sha256": None}
        with lock:
            outcome.records[request["request_id"]] = record
            active[request["model_id"]] -= 1
            start(gate.complete(request))
            lock.notify_all()

    def start(released: Iterable[Mapping[str, Any]]) -> None:
        for request in released:
            outcome.released_ns[request["request_id"]] = clock()
            outcome.release_order.append(request["request_id"])
            model = request["model_id"]
            active[model] = active.get(model, 0) + 1
            outcome.inflight_peak[model] = max(outcome.inflight_peak.get(model, 0), active[model])
            thread = threading.Thread(target=worker, args=(request,), name="req-" + request["request_id"],
                                      daemon=True)
            threads.append(thread)
            thread.start()

    def wait_until(target_ns: int) -> None:
        while True:
            now = clock()
            if now >= target_ns:
                return
            if deadline_ns is not None and now >= deadline_ns:
                raise TimeoutError("replay deadline")
            if tick is not None:
                tick()
            time.sleep(min(0.5, (target_ns - now) / 1e9))

    try:
        for request in sorted(plan["requests"], key=lambda row: (row["replay_arrival_us"],
                                                                   row["combined_request_index"])):
            wait_until(epoch_ns + request["replay_arrival_us"] * 1000)
            with lock:
                outcome.arrived_ns[request["request_id"]] = clock()
                start(gate.arrive(request))
        with lock:
            while len(outcome.records) < len(plan["requests"]):
                if deadline_ns is not None and clock() >= deadline_ns:
                    raise TimeoutError("replay deadline")
                lock.wait(timeout=0.5)
                if tick is not None:
                    lock.release()
                    try:
                        tick()
                    finally:
                        lock.acquire()
    except TimeoutError as error:
        outcome.aborted = str(error)
    for thread in threads:
        thread.join(timeout=0 if outcome.aborted else 30)
    return outcome


# ---------------------------------------------------------------- router process and its log

@dataclass
class LogLine:
    t_ns: int
    text: str


class RouterProcess:
    """The router ``llama-server``: merged stdout/stderr to ``server.log``, every line kept with its arrival time."""

    def __init__(self, command: Sequence[str], environment: Mapping[str, str], log_path: Path) -> None:
        self.command = list(command)
        self.environment = dict(environment)
        self.log_path = log_path
        self.lines: list[LogLine] = []
        self._lock = threading.Lock()
        self.process: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        self.process = subprocess.Popen(self.command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, env=self.environment, start_new_session=True)
        self._reader = threading.Thread(target=self._read, name="router-log", daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        with self.log_path.open("xb") as sink:
            for raw in self.process.stdout:
                now = time.monotonic_ns()
                sink.write(raw)
                sink.flush()
                with self._lock:
                    self.lines.append(LogLine(now, raw.decode("utf-8", "replace").rstrip("\r\n")))

    def snapshot(self) -> list[LogLine]:
        with self._lock:
            return list(self.lines)

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self, grace_s: float = 30.0) -> int | None:
        """SIGTERM the router (it unloads its instances), then SIGKILL its process group (instances included)."""
        process = self.process
        if process is None:
            return None
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        if self._reader is not None:
            self._reader.join(timeout=10)
        if process.stdout is not None and (self._reader is None or not self._reader.is_alive()):
            process.stdout.close()
        return process.returncode


def http_get_json(host: str, port: int, path: str, timeout_s: float = 5.0) -> Any:
    connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        content = response.read()
        if response.status != 200:
            raise OSError("GET %s -> %d" % (path, response.status))
        return json.loads(content)
    finally:
        connection.close()


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((host, port)) == 0


_SPAWN = re.compile(r"spawning server instance with name=(\S+) on port (\d+)")
_ARG = re.compile(r"\bload:\s{3}(\S.*)$")
_CHILD = re.compile(r"^\[\s*(\d+)\]\s?(.*)$")
_STATE = "cmd_child_to_router:state:"
_EXITED = re.compile(r"instance name=(\S+) exited with status (-?\d+)")
_LRU = re.compile(r"models_max limit reached, removing LRU name=(\S+)")
_FORCE = re.compile(r"force-killing model instance name=(\S+)")
_CHILD_PREFIX = re.compile(r"^\d+\.\d{2}\.\d{3}\.\d{3} [IWEDT] ")

_PLACEMENT_INTS = {
    "n_ctx": re.compile(r"llama_context: n_ctx\s+= (\d+)"),
    "n_ctx_seq": re.compile(r"llama_context: n_ctx_seq\s+= (\d+)"),
    "n_seq_max": re.compile(r"llama_context: n_seq_max\s+= (\d+)"),
    "n_batch": re.compile(r"llama_context: n_batch\s+= (\d+)"),
    "n_ubatch": re.compile(r"llama_context: n_ubatch\s+= (\d+)"),
    "n_ctx_train": re.compile(r"print_info: n_ctx_train\s+= (\d+)"),
    "n_layer": re.compile(r"print_info: n_layer\s+= (\d+)"),
    "n_parallel_auto": re.compile(r"n_parallel is set to auto, using n_parallel = (\d+)"),
}
_PLACEMENT_WORDS = {
    "flash_attn": re.compile(r"llama_context: flash_attn\s+= (\S+)"),
    "kv_unified": re.compile(r"llama_context: kv_unified\s+= (\S+)"),
}
_OFFLOADED = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
_SLOTS = re.compile(r"initializing, n_slots = (\d+), n_ctx_slot = (\d+), kv_unified = '(\w+)'")
_THREADS = re.compile(r"system_info: n_threads = (\d+) \(n_threads_batch = (\d+)\) / (\d+)")
_BUFFER = re.compile(r"(\S+) (model|KV|compute) buffer size =\s+([\d.]+) MiB")
_DEVICE = re.compile(r"-\s+(\S+)\s+:\s+(.+?) \((\d+) MiB, (\d+) MiB free\)")
_FIT = re.compile(r"(common_params_fit_impl|common_fit_params): (.*)$")
_FA_AUTO = re.compile(r"Flash Attention was auto, set to (\w+)")


def parse_placement(child_lines: Iterable[str]) -> dict[str, Any]:
    """The fitted placement one model instance logged (child lines without the router's ``[port]`` prefix)."""
    out: dict[str, Any] = {"buffers_mib": {}, "devices_at_start": [], "fit_log": []}
    for raw in child_lines:
        line = _CHILD_PREFIX.sub("", raw)
        for key, pattern in _PLACEMENT_INTS.items():  # last value wins: the real load follows fit's dry runs
            if match := pattern.search(line):
                out[key] = int(match.group(1))
        for key, pattern in _PLACEMENT_WORDS.items():
            if match := pattern.search(line):
                out[key] = match.group(1)
        if match := _OFFLOADED.search(line):
            out["gpu_layers"], out["gpu_layers_of"] = int(match.group(1)), int(match.group(2))
        if match := _SLOTS.search(line):
            out["n_slots"], out["n_ctx_slot"] = int(match.group(1)), int(match.group(2))
            out["slots_kv_unified"] = match.group(3) == "true"
        if (match := _THREADS.search(line)) and "n_threads" not in out:
            out["n_threads"], out["n_threads_batch"], out["n_cpu_threads_total"] = map(int, match.groups())
        if match := _BUFFER.search(line):
            kind = match.group(2).lower()
            out["buffers_mib"].setdefault(kind, {})[match.group(1)] = float(match.group(3))
        if (match := _DEVICE.search(line)) and "fitting params" not in line:
            out["devices_at_start"].append({"device": match.group(1), "description": match.group(2),
                                            "total_mib": int(match.group(3)), "free_mib": int(match.group(4))})
        if match := _FIT.search(line):
            out["fit_log"].append(match.group(2))
        if match := _FA_AUTO.search(line):
            out["flash_attn_resolved"] = match.group(1)
    fit_text = " | ".join(out["fit_log"])
    if "successfully fit params" in fit_text:
        out["fit_status"] = "fitted"
    elif "failed to fit params" in fit_text or "error while trying to fit" in fit_text:
        out["fit_status"] = "fit_failed"
    elif out["fit_log"]:
        out["fit_status"] = "unknown"
    reduced = re.search(r"context size reduced from (\d+) to (\d+)", fit_text)
    if reduced:
        out["fit_context_reduced_from"], out["fit_context_reduced_to"] = int(reduced.group(1)), int(reduced.group(2))
    if "no changes needed" in fit_text:
        out["fit_changes"] = "none"
    return out


def parse_instances(lines: Sequence[LogLine], epoch_ns: int) -> list[dict[str, Any]]:
    """Model instances from the router log: spawn, args, ready, eviction, exit (paid-clock microseconds) + placement."""
    instances: list[dict[str, Any]] = []
    by_port: dict[int, dict[str, Any]] = {}
    child_lines: dict[int, list[str]] = {}
    collecting: dict[str, Any] | None = None

    def us(t_ns: int) -> int:
        return (t_ns - epoch_ns) // 1000

    def latest(name: str, *, open_only: bool = True) -> dict[str, Any] | None:
        for instance in reversed(instances):
            if instance["model_id"] == name and (not open_only or instance["exit_us"] is None):
                return instance
        return None

    for entry in lines:
        text = entry.text
        child = _CHILD.match(text)
        if child:
            collecting = None
            port = int(child.group(1))
            content = child.group(2)
            if port in by_port:
                child_lines.setdefault(id(by_port[port]), []).append(content)
                if content.startswith(_STATE):
                    try:
                        state = json.loads(content[len(_STATE):])
                    except ValueError:
                        continue
                    instance = by_port[port]
                    if state.get("state") == "ready" and instance["ready_us"] is None:
                        instance["ready_us"] = us(entry.t_ns)
                        instance["ready_info"] = (state.get("payload") or {}).get("meta")
            continue
        if match := _SPAWN.search(text):
            instance = {"args": [], "evicted_lru_us": None, "exit_status": None, "exit_us": None,
                        "force_killed": False, "model_id": match.group(1), "port": int(match.group(2)),
                        "ready_info": None, "ready_us": None, "spawn_us": us(entry.t_ns)}
            instances.append(instance)
            by_port[instance["port"]] = instance
            collecting = None
            continue
        if "spawning server instance with args:" in text:
            collecting = instances[-1] if instances else None
            continue
        if collecting is not None:
            arg = _ARG.search(text)
            if arg:
                collecting["args"].append(arg.group(1))
                continue
            collecting = None
        if match := _LRU.search(text):
            instance = latest(match.group(1))
            if instance is not None:
                instance["evicted_lru_us"] = us(entry.t_ns)
        elif match := _FORCE.search(text):
            instance = latest(match.group(1))
            if instance is not None:
                instance["force_killed"] = True
        elif match := _EXITED.search(text):
            instance = latest(match.group(1))
            if instance is not None:
                instance["exit_us"] = us(entry.t_ns)
                instance["exit_status"] = int(match.group(2))
    for index, instance in enumerate(instances):
        instance["instance_index"] = index
        instance["load_s"] = (None if instance["ready_us"] is None
                              else (instance["ready_us"] - instance["spawn_us"]) / 1e6)
        instance["placement"] = parse_placement(child_lines.get(id(instance), []))
    return instances


def serving_instance(instances: Sequence[Mapping[str, Any]], model_id: str, at_us: int) -> Mapping[str, Any] | None:
    """The latest instance of ``model_id`` spawned at or before ``at_us``."""
    chosen = None
    for instance in instances:
        if instance["model_id"] == model_id and instance["spawn_us"] <= at_us:
            chosen = instance
    return chosen


# ---------------------------------------------------------------- host energy (campaign sampler + integrators)

def indexed_gpu_snapshot(index: int) -> dict[str, object]:
    """``nvidia_gpu_snapshot`` for one GPU of a multi-GPU host (local tests; the desktop has one GPU)."""
    completed = subprocess.run(
        ["nvidia-smi", "--id=%d" % index,
         "--query-gpu=name,uuid,memory.total,memory.used,memory.free,utilization.gpu,power.draw",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
    fields = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(fields) != 7:
        raise ValueError("NVIDIA GPU sample is invalid")
    return {"memory_free_bytes": int(fields[4]) * 1024 * 1024, "memory_total_bytes": int(fields[2]) * 1024 * 1024,
            "memory_used_bytes": int(fields[3]) * 1024 * 1024, "name": fields[0],
            "power_mw": int(round(float(fields[6]) * 1000)), "utilization_pct": int(fields[5]), "uuid": fields[1]}


def host_metric_callbacks(gpu_index: int | None) -> HostMetricCallbacks:
    if gpu_index is None:
        return default_host_metric_callbacks()
    return HostMetricCallbacks(gpu_snapshot=functools.partial(indexed_gpu_snapshot, gpu_index),
                               rapl_package_snapshot=rapl_package_snapshot, system_memory=linux_system_memory,
                               host_activity=linux_host_activity)


def rapl_probe() -> str | None:
    """None when RAPL package-0 is readable, else the reason (e.g. root-only energy_uj)."""
    try:
        rapl_package_snapshot()
    except Exception as error:  # noqa: BLE001 - any failure means absent
        return "%s: %s" % (type(error).__name__, error)
    return None


def _bracket(points: Sequence[tuple[int, Mapping[str, Any]]], start_ns: int, end_ns: int,
             max_gap_ns: int) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Rows from the last sample <= start to the first sample >= end, and their coverage (campaign rule: every gap
    <= 5 s, RaplNvmlPhoneEnergyMeter._require_contiguous_coverage)."""
    points = sorted(points, key=lambda item: item[0])
    before = [index for index, (t, _) in enumerate(points) if t <= start_ns]
    after = [index for index, (t, _) in enumerate(points) if t >= end_ns]
    coverage = {"samples": len(points), "covered": bool(before and after)}
    if not coverage["covered"]:
        return [], coverage
    chosen = points[before[-1]:after[0] + 1]
    gaps = [b[0] - a[0] for a, b in zip(chosen, chosen[1:])]
    coverage.update({"samples_in_window": len(chosen), "max_gap_s": max(gaps, default=0) / 1e9,
                     "contiguous": all(gap <= max_gap_ns for gap in gaps)})
    return [row for _, row in chosen], coverage


def host_energy(rows: Sequence[Mapping[str, Any]], start_ns: int, end_ns: int,
                max_gap_s: float = MAX_SAMPLE_GAP_S) -> dict[str, Any]:
    """CPU package (RAPL) + GPU board (nvidia-smi) energy over [start_ns, end_ns] with the campaign's integrators;
    a source that is absent or does not cover the window is reported, not integrated."""
    if end_ns <= start_ns:
        raise ValueError("energy window is empty")
    gap_ns = round(max_gap_s * 1e9)
    gpu_rows, gpu_cov = _bracket([(int(r["gpu"]["sample_t_ns"]), r) for r in rows if isinstance(r.get("gpu"), dict)],
                                 start_ns, end_ns, gap_ns)
    rapl_rows, rapl_cov = _bracket([(int(r["rapl_package"]["sample_t_ns"]), r) for r in rows
                                    if isinstance(r.get("rapl_package"), dict)], start_ns, end_ns, gap_ns)
    duration_s = (end_ns - start_ns) / 1e9
    out: dict[str, Any] = {"boundary": "paid_trace_interval", "coverage": {"gpu": gpu_cov, "rapl_package": rapl_cov},
                           "cpu_package_energy_j": None, "gpu_board_energy_j": None,
                           "method": "RAPL package delta plus trapezoidal GPU board power"}
    if gpu_cov.get("covered") and gpu_cov.get("contiguous"):
        out["gpu_board_energy_j"] = _integrate_gpu(gpu_rows, start_ns, end_ns)
        out["gpu_board_average_power_w"] = out["gpu_board_energy_j"] / duration_s
    if rapl_cov.get("covered") and rapl_cov.get("contiguous"):
        out["cpu_package_energy_j"] = _integrate_rapl(rapl_rows, start_ns, end_ns)
        out["cpu_package_average_power_w"] = out["cpu_package_energy_j"] / duration_s
    if out["gpu_board_energy_j"] is not None and out["cpu_package_energy_j"] is not None:
        out["server_compute_device_energy_j"] = out["gpu_board_energy_j"] + out["cpu_package_energy_j"]
    return out


def trace_energy_block(energy: Mapping[str, Any], rapl_absent_reason: str | None, gpu_sampler: str
                       ) -> tuple[dict[str, Any], str]:
    """RESULT ``trace_energy`` in the campaign's shape (runner.measured_energy) and the evidence class."""
    domains: dict[str, int] = {}
    evidence_ids = []
    if energy.get("cpu_package_energy_j") is not None:
        domains["cpu-package"] = round(energy["cpu_package_energy_j"] * 1_000_000)
        evidence_ids.append("physical:rapl-package-0")
    if energy.get("gpu_board_energy_j") is not None:
        domains["gpu-board"] = round(energy["gpu_board_energy_j"] * 1_000_000)
        evidence_ids.append("physical:nvml-board-power")
    evidence = ("MEASURED" if len(domains) == 2 else "PARTIAL" if domains else "ABSENT")
    metadata = {"energy_attribution_reason": "ENERGY_CAMPAIGN_WINDOW", "gpu_sampler": gpu_sampler,
                "host_sampler": "adapters.host_runtime.HostEnergySampler (0.2 s)",
                "rapl_package": "present" if rapl_absent_reason is None else "absent: " + rapl_absent_reason}
    return ({"attribution_kind": "diagnostic", "energy_boundary_id": "desktop-cpu-package-gpu-board-v1",
             "estimation_metadata": metadata, "fleet_energy_uj_by_domain": domains,
             "measurement_evidence_ids": evidence_ids, "transfer_energy_uj_by_link": {}}, evidence)


def wait_for_samples(rows: Callable[[], Sequence[Mapping[str, Any]]], target_ns: int, need_rapl: bool,
                     timeout_s: float = 60.0) -> bool:
    """True once a sample taken at or after ``target_ns`` exists (GPU, and RAPL when present)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for row in reversed(rows()):
            gpu = row.get("gpu")
            rapl = row.get("rapl_package")
            if isinstance(gpu, dict) and int(gpu["sample_t_ns"]) >= target_ns and (
                    not need_rapl or (isinstance(rapl, dict) and int(rapl["sample_t_ns"]) >= target_ns)):
                return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------- result

def _us(t_ns: int | None, epoch_ns: int) -> int | None:
    return None if t_ns is None else (t_ns - epoch_ns) // 1000


def request_result(request: Mapping[str, Any], record: Mapping[str, Any], outcome: ReplayOutcome,
                   instances: Sequence[Mapping[str, Any]], epoch_ns: int, endpoint: str) -> dict[str, Any]:
    request_id = request["request_id"]
    verdict = accounting(request, record, request["model_id"])
    sent_us = _us(record.get("sent_ns"), epoch_ns)
    end_us = _us(record.get("end_ns"), epoch_ns)
    first_us = _us(record.get("first_token_ns"), epoch_ns)
    probe_us = first_us if first_us is not None else end_us
    instance = None if probe_us is None else serving_instance(instances, request["model_id"], probe_us)
    ready_us = None if instance is None else instance.get("ready_us")
    started_us = sent_us if ready_us is None or sent_us is None else max(sent_us, ready_us)
    tokens = list(record.get("tokens") or [])
    quality = None
    if tokens:
        try:
            quality = assess_semantic_output(record.get("text") or "", tuple(tokens)).to_json()
        except ValueError:
            quality = None
    final = record.get("final") or {}
    return {
        "accounting": verdict,
        "actual_endpoint": endpoint,
        "actual_executor_id": "default-router:" + request["model_id"],
        "actual_latency_us": None if started_us is None or end_us is None else end_us - started_us,
        "attempt_ticket_ids": [request_id + ":default:0"],
        "client": {
            "arrived_us": _us(outcome.arrived_ns.get(request_id), epoch_ns),
            "error": record.get("error"),
            "gate_wait_us": (None if request_id not in outcome.released_ns
                             else (outcome.released_ns[request_id] - outcome.arrived_ns[request_id]) // 1000),
            "http_status": record.get("http_status"),
            "released_us": _us(outcome.released_ns.get(request_id), epoch_ns),
            "sent_us": sent_us,
        },
        "combined_request_index": request["combined_request_index"],
        "completion": {
            "actual_end_us": end_us,
            "error": record.get("error"),
            "execution_receipt": {"endpoint": endpoint, "executor_id": "default-router:" + request["model_id"],
                                  "finished_us": end_us, "started_us": started_us},
            "status": "completed" if verdict["exact"] else "failed",
        },
        "first_token_ns": record.get("first_token_ns"),
        "input_tokens": request["input_tokens"],
        "model_id": request["model_id"],
        "model_instance": None if instance is None else {"instance_index": instance["instance_index"],
                                                         "port": instance["port"]},
        "output_quality": quality,
        "output_sha256": None if record.get("stream_sha256") is None else "sha256:" + record["stream_sha256"],
        "output_tokens": request["output_tokens"],
        "output_tokens_sha256": "sha256:" + hashlib.sha256(canonical(tokens)).hexdigest(),
        "prompt_sha256": request["prompt_sha256"],
        "recoveries": [],
        "replay_arrival_us": request["replay_arrival_us"],
        "request_id": request_id,
        "seed": request["seed"],
        "server_timings": final.get("timings"),
        "source": request["source"],
        "source_arrival_us": request["source_arrival_us"],
        "source_slo_us": request["slo_us"],
        "streamed_output_tokens": len(tokens),
        "terminal_ticket": {"transition_receipts": []},
        "trace_arrival_us": request["source_arrival_us"],
    }


def attach_loads(results: list[dict[str, Any]], instances: Sequence[Mapping[str, Any]]) -> None:
    """Charge every model load to the request that triggered it: the earliest-sent request served by the instance,
    sent before the spawn (router autoload). ``latency_report`` reads it as ``load_s``."""
    by_instance: dict[int, list[dict[str, Any]]] = {}
    for row in results:
        if row["model_instance"] is not None and row["client"]["sent_us"] is not None:
            by_instance.setdefault(row["model_instance"]["instance_index"], []).append(row)
    for instance in instances:
        rows = sorted(by_instance.get(instance["instance_index"], []), key=lambda row: row["client"]["sent_us"])
        if not rows or rows[0]["client"]["sent_us"] > instance["spawn_us"]:
            continue
        rows[0]["terminal_ticket"]["transition_receipts"].append({
            "finished_us": instance["ready_us"], "instance_index": instance["instance_index"],
            "kind": "router_autoload", "model_id": instance["model_id"], "started_us": instance["spawn_us"]})


def _role(model_id: str) -> str:
    for role in ("qwen", "gemma", "llama"):
        if role in model_id.lower():
            return role
    return model_id


def placements_by_model(instances: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    keys = ("gpu_layers", "gpu_layers_of", "n_ctx", "n_ctx_seq", "n_seq_max", "n_slots", "n_ctx_slot",
            "slots_kv_unified", "kv_unified", "flash_attn", "flash_attn_resolved", "n_batch", "n_ubatch",
            "n_threads", "n_threads_batch", "fit_status", "fit_changes", "fit_context_reduced_from",
            "fit_context_reduced_to", "n_ctx_train", "n_layer")
    out: dict[str, list[dict[str, Any]]] = {}
    for instance in instances:
        summary = {key: instance["placement"].get(key) for key in keys if key in instance["placement"]}
        summary["buffers_mib"] = instance["placement"].get("buffers_mib", {})
        distinct = out.setdefault(instance["model_id"], [])
        for existing in distinct:
            if {k: v for k, v in existing.items() if k != "instances"} == summary:
                existing["instances"].append(instance["instance_index"])
                break
        else:
            distinct.append({**summary, "instances": [instance["instance_index"]]})
    return out


def build_result(plan: Mapping[str, Any], outcome: ReplayOutcome, instances: Sequence[Mapping[str, Any]],
                 energy: Mapping[str, Any], epoch_ns: int, paid_end_ns: int, *, endpoint: str,
                 rapl_absent_reason: str | None, gpu_sampler: str, server: Mapping[str, Any], gate_mode: str,
                 models_max: int, log_verbosity: int | None, host_diagnostics: Mapping[str, Any] | None = None,
                 harness: Mapping[str, Any] | None = None) -> dict[str, Any]:
    results = []
    for request in plan["requests"]:
        record = outcome.records.get(request["request_id"]) or {
            "error": "not completed before the replay ended", "end_ns": paid_end_ns, "tokens": [], "final": None}
        results.append(request_result(request, record, outcome, instances, epoch_ns, endpoint))
    attach_loads(results, instances)
    trace_energy, evidence = trace_energy_block(energy, rapl_absent_reason, gpu_sampler)
    reasons = []
    failed = [row["request_id"] for row in results if not row["accounting"]["exact"]]
    if failed:
        reasons.append("REQUESTS_NOT_EXACT:" + ",".join(failed))
    if outcome.aborted:
        reasons.append("REPLAY_ABORTED:" + outcome.aborted)
    if evidence != "MEASURED":
        reasons.append("ENERGY_" + evidence)
    counts: dict[str, int] = {"completed": len(results) - len(failed), "failed": len(failed), "requests": len(results)}
    for row in results:
        role = _role(row["model_id"])
        counts[role] = counts.get(role, 0) + 1
    deviations = []
    if models_max != 4:
        deviations.append({**DEVIATIONS["models_max"], "flag": "--models-max %d" % models_max})
    if gate_mode != "none":
        deviations.append(DEVIATIONS["switch_gate"])
    if log_verbosity is not None and log_verbosity != 3:
        deviations.append({**DEVIATIONS["log_verbosity"], "flag": "-lv %d" % log_verbosity})
    deviations.append(DEVIATIONS["request_model_field"])
    return {
        "arm": "default-llamacpp-router",
        "client_inflight_peak": dict(outcome.inflight_peak),
        "counts": counts,
        "defaults_relied_on": list(DEFAULTS_RELIED_ON),
        "deviations_from_defaults": deviations,
        "duration_us": (paid_end_ns - epoch_ns) // 1000,
        "energy_evidence": evidence,
        "energy_summary": dict(energy),
        "harness": dict(harness or {}),
        "inputs_not_deviations": list(INPUTS_NOT_DEVIATIONS),
        "host_power_diagnostics": dict(host_diagnostics or {}),
        "model_instances": list(instances),
        "model_placements": placements_by_model(instances),
        "models": list(plan["models"]),
        "models_max": models_max,
        "paid_end_ns": paid_end_ns,
        "paid_start_ns": epoch_ns,
        "pure_defaults_prediction": PURE_DEFAULTS_PREDICTION,
        "release_order": list(outcome.release_order),
        "replay_schedule": plan["replay_schedule"],
        "request_fields": {"endpoint": "POST /completion",
                           "body": "campaign body (adapters/http_backend.py:263-272) + model",
                           "cache_prompt": False, "ignore_eos": True, "n_predict": "trace output_tokens",
                           "prompt": "trace prompt_tokens", "return_tokens": True,
                           "seed": "combined_request_index", "stream": True, "temperature": 0.0,
                           "headers": ["Content-Type: application/json", "X-Scheduler-Request-ID"]},
        "request_results": results,
        "schema": RESULT_SCHEMA,
        "server": dict(server),
        "status": "PASS" if not reasons else "FAIL",
        "status_reasons": reasons,
        "switch_gate": gate_mode,
        "trace_energy": trace_energy,
        "trace_identity": plan.get("source"),
        "trace_name": plan.get("trace_name"),
    }


# ---------------------------------------------------------------- run

def _file_sha256(path: Path) -> str | None:
    return "sha256:" + digest(path) if path.is_file() else None


def server_identity(binary: Path) -> dict[str, Any]:
    libraries = {}
    for path in sorted(binary.parent.glob("lib*.so*")):
        if path.is_file():
            libraries[path.name] = _file_sha256(path)
    return {"binary": str(binary), "binary_sha256": _file_sha256(binary), "runtime_libraries_sha256": libraries}


def router_command(binary: Path, preset: Path, host: str, port: int, models_max: int,
                   log_verbosity: int | None) -> list[str]:
    command = [str(binary), "--models-preset", str(preset), "--models-max", str(models_max),
               "--host", host, "--port", str(port)]
    if log_verbosity is not None:
        command += ["-lv", str(log_verbosity)]
    return command


def router_environment(base: Mapping[str, str], library_dirs: Sequence[str], server_dir: Path,
                       cuda_visible_devices: str | None) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Base environment minus every variable that could change llama.cpp behaviour, plus the runtime libraries.
    Returns (environment, overrides, removed names)."""
    environment = dict(base)
    removed = sorted(name for name in environment
                     if name.startswith(ENV_SCRUB_PREFIXES) or name == "CUDA_VISIBLE_DEVICES")
    for name in removed:
        del environment[name]
    overrides: dict[str, str] = {}
    library_path = ":".join(dict.fromkeys([*library_dirs, str(server_dir)]))
    environment["LD_LIBRARY_PATH"] = library_path + (":" + base["LD_LIBRARY_PATH"]
                                                     if base.get("LD_LIBRARY_PATH") else "")
    overrides["LD_LIBRARY_PATH"] = environment["LD_LIBRARY_PATH"]
    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        overrides["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices + " (local test only)"
    return environment, overrides, removed


def check_router_support(binary: Path, environment: Mapping[str, str]) -> None:
    completed = subprocess.run([str(binary), "--help"], capture_output=True, text=True, env=dict(environment),
                               timeout=60, check=False)
    text = completed.stdout + completed.stderr
    missing = [flag for flag in ("--models-preset", "--models-max", "--fit") if flag not in text]
    if missing:
        raise SystemExit("server binary lacks router/fit support (%s): %s" % (", ".join(missing), binary))


def check_models(plan: Mapping[str, Any], check_sizes: bool) -> list[dict[str, Any]]:
    checks = []
    for row in plan["models"]:
        path = Path(row["path"])
        if not path.is_file():
            raise SystemExit("model file is missing: " + str(path))
        size = path.stat().st_size
        expected = row.get("artifact_bytes")
        if check_sizes and expected is not None and size != expected:
            raise SystemExit("model size differs from the trace inventory: %s (%d != %d)" % (path, size, expected))
        checks.append({"artifact_bytes": size, "model_id": row["model_id"], "path": str(path),
                       "size_matches_trace_inventory": None if expected is None else size == expected})
    return checks


def _load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """(plan, rig) from --plan or --campaign/--models/--rig/trace paths."""
    campaign = load_object(Path(args.campaign)) if args.campaign else {}
    rig_path = args.rig or campaign.get("rig_manifest_path")
    rig = load_object(Path(rig_path)) if rig_path else {}
    if getattr(args, "plan", None):
        return json.loads(Path(args.plan).read_text()), rig
    trace = campaign.get("trace", {})
    paths = {name: getattr(args, name) or trace.get(key) for name, key in (
        ("large_requests", "large_requests_path"), ("overlay_requests", "overlay_requests_path"),
        ("trace_manifest", "trace_manifest_path"), ("replay_schedule", "replay_schedule_path"))}
    missing = [name for name, value in paths.items() if not value]
    models_path = args.models or campaign.get("models_manifest_path")
    if missing or not models_path:
        raise SystemExit("trace inputs missing: " + ", ".join(missing + ([] if models_path else ["models"])))
    plan = plan_from_trace(Path(paths["large_requests"]), Path(paths["overlay_requests"]),
                           Path(paths["trace_manifest"]), Path(paths["replay_schedule"]),
                           load_object(Path(models_path)))
    plan["models_manifest"] = str(models_path)
    return plan, rig


class _Interrupted(Exception):
    pass


def run(args: argparse.Namespace, *, sampler_factory: Callable[[HostMetricCallbacks], Any] = HostEnergySampler,
        rapl_check: Callable[[], str | None] = rapl_probe) -> int:
    plan, rig = _load_inputs(args)
    validate_plan(plan)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("output directory is not empty: " + str(out))
    binary = Path(args.server or (rig.get("binaries") or {}).get("server") or "")
    if not binary.is_file():
        raise SystemExit("llama-server binary not found: " + str(binary))
    library_dirs = list(args.library_dir or (rig.get("library_directories") or {}).values())
    environment, overrides, removed = router_environment(os.environ, library_dirs, binary.parent,
                                                         args.cuda_visible_devices)
    check_router_support(binary, environment)
    model_checks = check_models(plan, not args.skip_model_size_check)
    host = "127.0.0.1"
    if port_in_use(host, args.port):
        raise SystemExit("port %d is in use" % args.port)
    log_verbosity = None if args.log_verbosity == "default" else int(args.log_verbosity)

    out.mkdir(parents=True, exist_ok=True)
    streams = out / "streams"
    streams.mkdir()
    (out / "REPLAY_PLAN.json").write_bytes(canonical(plan))
    (out / "REPLAY_SCHEDULE.json").write_bytes(canonical(plan["replay_schedule"]))
    preset = out / "models-preset.ini"
    preset.write_text(preset_text(plan["models"]))
    command = router_command(binary, preset, host, args.port, args.models_max, log_verbosity)
    rapl_absent = rapl_check()
    gpu_sampler = ("adapters.host_runtime.nvidia_gpu_snapshot" if args.gpu_index is None
                   else "nvidia-smi --id=%d (local multi-GPU host)" % args.gpu_index)
    sampler = sampler_factory(host_metric_callbacks(args.gpu_index))
    router = RouterProcess(command, environment, out / "server.log")
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def interrupted(signum, _frame):
        raise _Interrupted("signal %d" % signum)

    for sig in previous:
        signal.signal(sig, interrupted)
    samples: list[Mapping[str, Any]] = []
    epoch_ns = paid_end_ns = time.monotonic_ns()
    replay_started = False  # no paid window (and no energy) before the replay epoch
    outcome = ReplayOutcome()
    energy: dict[str, Any] = {"coverage": {}, "cpu_package_energy_j": None, "gpu_board_energy_j": None}
    diagnostics: dict[str, Any] = {}
    router_exit = None
    try:
        sampler.start()
        router.start()
        deadline = time.monotonic() + 120
        while True:
            if not router.alive():
                raise SystemExit("router exited during startup; see " + str(out / "server.log"))
            try:
                if http_get_json(host, args.port, "/health").get("status") == "ok":
                    break
            except (OSError, ValueError, http.client.HTTPException):
                pass
            if time.monotonic() > deadline:
                raise SystemExit("router did not become healthy")
            time.sleep(0.2)
        models_listing = http_get_json(host, args.port, "/models")
        (out / "ROUTER_MODELS.json").write_bytes(canonical(models_listing))
        if not wait_for_samples(sampler.latest_rows, time.monotonic_ns(), rapl_absent is None):
            raise SystemExit("host energy sampler is not producing samples")
        epoch_ns = time.monotonic_ns()
        replay_started = True
        max_ns = None if args.max_duration_s is None else epoch_ns + int(args.max_duration_s * 1e9)
        print("replay start: %d requests, gate %s, models-max %d, port %d" % (
            len(plan["requests"]), args.switch_gate, args.models_max, args.port), flush=True)
        outcome = replay(plan, host, args.port, args.switch_gate, epoch_ns, streams, deadline_ns=max_ns,
                         tick=sampler.latest_rows)
        paid_end_ns = time.monotonic_ns()
        if not wait_for_samples(sampler.latest_rows, paid_end_ns, rapl_absent is None):
            print("warning: no host sample after the paid window end", file=sys.stderr, flush=True)
    except _Interrupted as error:
        outcome.aborted = str(error)
        paid_end_ns = time.monotonic_ns()
    finally:
        router_exit = router.stop()
        try:
            sampler.stop()
        except Exception as error:  # noqa: BLE001 - recorded, the run artifacts are still written
            diagnostics["sampler_stop_error"] = "%s: %s" % (type(error).__name__, error)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        samples = list(sampler.rows())
        with (out / "resource-samples.jsonl").open("wb") as sink:
            for row in samples:
                sink.write(canonical(row))
    if replay_started and paid_end_ns > epoch_ns:
        energy = host_energy(samples, epoch_ns, paid_end_ns)
    diagnostics.update(sampler.diagnostics())
    (out / "host-power-diagnostics.json").write_bytes(canonical(diagnostics))
    instances = parse_instances(router.snapshot(), epoch_ns)
    server = {**server_identity(binary), "environment_overrides": overrides, "environment_removed": removed,
              "exit_code": router_exit, "host": host, "model_checks": model_checks,
              "models_preset": preset.read_text(), "port": args.port, "router_command": command}
    harness = {"argv": sys.argv, "hostname": platform.node(), "python": platform.python_version()}
    result = build_result(plan, outcome, instances, energy, epoch_ns, paid_end_ns,
                          endpoint="http://%s:%d" % (host, args.port), rapl_absent_reason=rapl_absent,
                          gpu_sampler=gpu_sampler, server=server, gate_mode=args.switch_gate,
                          models_max=args.models_max, log_verbosity=log_verbosity,
                          host_diagnostics={k: v for k, v in diagnostics.items() if k != "events"},
                          harness=harness)
    (out / "MODEL_PLACEMENTS.json").write_bytes(canonical({"instances": instances,
                                                           "by_model": result["model_placements"],
                                                           "schema": PLACEMENT_SCHEMA}))
    (out / "RESULT.json").write_bytes(canonical(result))
    print(json.dumps({"counts": result["counts"], "duration_us": result["duration_us"],
                      "energy_evidence": result["energy_evidence"], "status": result["status"],
                      "status_reasons": result["status_reasons"]}, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" or (args.allow_partial_energy and not [
        reason for reason in result["status_reasons"] if not reason.startswith("ENERGY_")]) else 1


# ---------------------------------------------------------------- CLI

def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign", help="campaign.json (trace paths; default models/rig manifests)")
    parser.add_argument("--models", help="models.json (research-scheduler-models-v1)")
    parser.add_argument("--rig", help="rig.json (binaries.server, library_directories)")
    parser.add_argument("--large-requests", dest="large_requests")
    parser.add_argument("--overlay-requests", dest="overlay_requests")
    parser.add_argument("--trace-manifest", dest="trace_manifest")
    parser.add_argument("--replay-schedule", dest="replay_schedule")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="write the replay plan (no server)")
    _add_input_arguments(plan)
    plan.add_argument("--out", required=True)
    plan.add_argument("--smoke-model", help="make a local smoke plan served by this small Qwen3 GGUF")
    plan.add_argument("--smoke-time-scale", type=float, default=0.01)
    plan.add_argument("--smoke-max-prompt", type=int, default=48)
    plan.add_argument("--smoke-max-output", type=int, default=24)
    runner = sub.add_parser("run", help="start the router, replay, measure, write RESULT.json")
    _add_input_arguments(runner)
    runner.add_argument("--plan", help="a plan written by 'plan' (instead of trace inputs)")
    runner.add_argument("--out", required=True, help="run directory (must be new or empty)")
    runner.add_argument("--server", help="llama-server binary (default: rig binaries.server)")
    runner.add_argument("--library-dir", action="append", help="runtime library directory (default: rig)")
    runner.add_argument("--port", type=int, default=DEFAULT_PORT)
    runner.add_argument("--models-max", type=int, default=DEFAULT_MODELS_MAX)
    runner.add_argument("--switch-gate", choices=GATE_MODES, default="fifo-drain")
    runner.add_argument("--log-verbosity", default=str(DEFAULT_LOG_VERBOSITY),
                        help="router/instance -lv (default 4 to log the fitted placement; 'default' = no flag)")
    runner.add_argument("--gpu-index", type=int, help="sample this GPU only (multi-GPU hosts; local tests)")
    runner.add_argument("--cuda-visible-devices", help="CUDA_VISIBLE_DEVICES for the server (local tests only)")
    runner.add_argument("--max-duration-s", type=float, default=14_400.0)
    runner.add_argument("--skip-model-size-check", action="store_true")
    runner.add_argument("--allow-partial-energy", action="store_true",
                        help="exit 0 when only the energy evidence is incomplete (e.g. RAPL root-only)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        args.plan = None
        plan, _ = _load_inputs(args)
        if args.smoke_model:
            plan = smoke_plan(plan, args.smoke_model, time_scale=args.smoke_time_scale,
                              max_prompt_tokens=args.smoke_max_prompt, max_output_tokens=args.smoke_max_output)
        validate_plan(plan)
        Path(args.out).write_bytes(canonical(plan))
        print("plan: %d requests, models %s -> %s" % (len(plan["requests"]),
                                                    ", ".join(row["model_id"] for row in plan["models"]), args.out))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
