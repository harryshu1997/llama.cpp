#!/usr/bin/env python3
"""WS16: matched per-device weight-stream microbenchmark (desktop CPU, desktop GPU, OP15, Pixel).

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.weight_stream_microbench plan    CONFIG.json
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.weight_stream_microbench prepare CONFIG.json OUT.json
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.weight_stream_microbench smoke   CONFIG.json OUT --leg NAME
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.weight_stream_microbench run     CONFIG.json OUT [--only a,b]
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.weight_stream_microbench analyze OUT --meter METER.json \\
        [--json R.json] [--md R.md] [--latex R.tex]

One operator group of a real GGUF (``ffn``, ``attn_proj``, ``output_head``) is streamed at the SAME serving cadence
on every device: ``calls_per_token`` calls ``gap_call_ms`` apart, then ``gap_token_ms`` (the head runs once per
token). Each leg has an idle window before and after its segments with its weights resident, and one measured
segment per batch-row count (1/2/4) after ``warmup_tokens`` warm-up tokens. ``rows`` takes up to 512, one prefill
micro-batch per call (the phone worker's ``max_tokens`` must cover the largest).

  desktop legs  ``weight-stream-bench`` (ws16 bench/, built against the deployed libggml*.so) on the CPU backend
                (the server's thread count) or CUDA0; per-call compute time = graph compute + synchronize.
  phone legs    the deployed FFN worker through the scheduler's own session classes: the OP15's HTP worker
                over ``adb forward`` (AdbTcpPhoneWorkerSession, transport ``adb-tcp``) or over the production
                FunctionFS DMA-BUF path (FunctionFsBridgePhoneWorkerSession, transport ``functionfs-bridge``), the
                Pixel's DVFS-fixed packed worker over the AOA bridge (AoaBridgePhoneWorkerSession, built from its
                qualified evidence bundle). Per call: host RPC time and the worker's own compute time (EXECUTE
                response). Inputs are deterministic synthetic f16 rows (GEMV time is data independent; outputs are
                hash-checked, not compared).

Energy, one convention for every device (``analyze``):
  domains   desktop CPU package (RAPL) and GPU board (NVML), sampled every 0.2 s by the campaign's
            HostEnergySampler in this process; each phone's terminal energy (USB input + net battery discharge,
            coulomb counter) from ``tools/rig/meter_phones.sh --charging controlled`` wrapped around ``run``.
  windows   CLOCK_MONOTONIC windows written by the legs (the clock the sampler and the meter anchors use).
  gross     energy of a domain over a measured segment / calls (contains that domain's idle power, so it depends
            on the cadence's duty cycle).
  incremental (the reported convention)  (segment energy - P_idle x segment duration) / calls, with P_idle the
            same domain's mean power over the leg's own idle windows (weights resident, no calls).
  device J  incremental energy of the executing device's domain per call; ``system J`` sums the incremental
            energy of all domains (for a phone leg it adds the desktop's cost of waiting in THIS harness, which is
            a lower bound of llama-server's waiting cost); nJ/byte = device incremental J / logical weight bytes
            (two-byte representation: FFN 534.8 MB, projections 125.8 MB, head 1,556 MB for Qwen3-14B).
Transport note: an ``adb-tcp`` OP15 leg's phone energy contains adbd's forwarding work (an upper bound of the
production path's phone cost). A ``functionfs-bridge`` leg runs the production path: the phone session script swaps
the Android gadget for the FunctionFS one (no adb until the leg ends) and the host's llama-ffn-split-usb-bridge
forwards each TCP call over FunctionFS with the DMA-BUF transfers of llama-server's USB client; the bridge's host
CPU time is in the desktop domains, as the USB client's is in a served run.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import socket
import statistics
import struct
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

SCHEMA = "ws16-weight-stream-microbench-v1"
RESULT_SCHEMA = "ws16-weight-stream-microbench-result-v1"
DESKTOP_DOMAINS = ("cpu", "gpu")
OPS = ("ffn", "attn_proj", "output_head")
PROTOCOL_MAGIC = 0x46534631
HELLO_V5 = struct.Struct("<IHHQIIHH4x")          # 32 B (FunctionFS-era workers)
HELLO_V5_RESPONSE_BYTES = 64
HELLO_V6 = struct.Struct("<IHHQIIHH32s4x")       # 64 B (ffn-split-protocol.h v6, padded)
HELLO_V6_RESPONSE_BYTES = 96
EXECUTE = struct.Struct("<IHHIiIIIII")           # 36 B, same in v5 and v6
EXECUTE_RESPONSE = struct.Struct("<IHHHHIiIIIIIQ")  # 48 B, same in v5 and v6
F16_IO_FLAG, SWIGLU_FLAG = 1, 2
PHONE_TRANSPORTS = ("adb-tcp", "aoa-bridge", "functionfs-bridge")
# a ``functionfs-bridge`` leg's ``functionfs`` section: the rig's OP15 ``phone`` block + the transport identity
FUNCTIONFS_KEYS = ("session_script", "session_script_sha256", "restore_script", "restore_script_sha256",
                   "session_root", "android_gadget_path", "functionfs_gadget_path", "functionfs_root_path",
                   "phone_usb_controller", "kernel_release", "bridge_path", "bridge_sha256", "bridge_port")


def fnv32(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------- configuration

def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    def need(condition: bool, message: str) -> None:
        if not condition:
            raise SystemExit("config: " + message)
    need(config.get("schema") == SCHEMA, "schema must be " + SCHEMA)
    cadence = config.get("cadence", {})
    for key in ("calls_per_token", "gap_call_ms", "gap_token_ms", "warmup_tokens"):
        need(key in cadence and cadence[key] >= 0, "cadence." + key + " is required")
    need(cadence["calls_per_token"] >= 1, "cadence.calls_per_token must be >= 1")
    rows = config.get("rows", [1, 2, 4])
    # Up to one prefill micro-batch (ubatch 512) per call: the 2026-10-08 phone-prefill screen.
    need(rows and all(type(r) is int and 1 <= r <= 512 for r in rows), "rows must be integers in [1, 512]")
    need(config.get("segment_seconds", 0) > 0 and config.get("idle_s", 0) > 0, "segment_seconds and idle_s > 0")
    names = [leg.get("name") for leg in config.get("legs", [])]
    need(names and len(set(names)) == len(names) and all(names), "legs need unique names")
    for leg in config["legs"]:
        need(leg.get("kind") in ("desktop", "phone"), leg["name"] + ": kind must be desktop or phone")
        need(leg.get("op") in OPS, leg["name"] + ": op must be one of " + ",".join(OPS))
        if leg["kind"] == "desktop":
            for key in ("bench", "model", "backend"):
                need(bool(leg.get(key)), f"{leg['name']}: {key} is required")
        else:
            need(leg["op"] == "ffn", leg["name"] + ": phone legs stream the FFN (the deployed workers' operator)")
            need(leg.get("device") in ("op15", "pixel"), leg["name"] + ": device must be op15 or pixel")
            need(leg.get("transport") in PHONE_TRANSPORTS, leg["name"] + ": transport " + "|".join(PHONE_TRANSPORTS))
            if leg["transport"] == "functionfs-bridge":
                need(leg["device"] == "op15" and "worker" in leg and leg.get("protocol_version") == 6,
                     leg["name"] + ": functionfs-bridge is the OP15's worker leg at protocol 6 (the bridge's)")
                missing = [key for key in FUNCTIONFS_KEYS if key not in leg.get("functionfs", {})]
                need(not missing, leg["name"] + ": functionfs needs " + ",".join(missing))
            need(leg.get("protocol_version") in (5, 6), leg["name"] + ": protocol_version must be 5 or 6")
            need(leg.get("activation") in ("swiglu", "geglu"), leg["name"] + ": activation swiglu|geglu")
            for key in ("layer", "n_embd", "n_ff", "est_call_ms"):
                need(key in leg, f"{leg['name']}: {key} is required")
            need(all(str(r) in {str(k) for k in leg["est_call_ms"]} for r in rows),
                 leg["name"] + ": est_call_ms needs every row count")
            need(("worker" in leg) != ("evidence" in leg), leg["name"] + ": exactly one of worker / evidence")


def token_cycle_s(config: Mapping[str, Any], call_ms: float, op: str) -> float:
    cadence = config["cadence"]
    calls = 1 if op == "output_head" else cadence["calls_per_token"]
    return (calls * call_ms + (calls - 1) * cadence["gap_call_ms"] + cadence["gap_token_ms"]) / 1000.0


def phone_tokens(config: Mapping[str, Any], leg: Mapping[str, Any], rows: int) -> int:
    """Measured tokens of a phone segment: enough for ``segment_seconds`` at the leg's estimated call time."""
    call_ms = float(leg["est_call_ms"][str(rows)])
    return max(1, math.ceil(config["segment_seconds"] / token_cycle_s(config, call_ms, "ffn")))


def phone_budget(config: Mapping[str, Any], leg: Mapping[str, Any]) -> int:
    """Worker budget = every warm-up and measured call + one closing call after the idle-after window (a
    ``functionfs-bridge`` worker runs without a budget and serves one call fewer: see drive_phone_leg)."""
    cadence = config["cadence"]
    calls = sum((cadence["warmup_tokens"] + phone_tokens(config, leg, rows)) * cadence["calls_per_token"]
                for rows in config.get("rows", [1, 2, 4]))
    return calls + 1


def plan(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = config.get("rows", [1, 2, 4])
    legs, total = [], 0.0
    for leg in config["legs"]:
        segments = len(rows) * (config["segment_seconds"] + config["cadence"]["warmup_tokens"]
                                * token_cycle_s(config, float(leg.get("est_call_ms", {}).get("1", 10)), leg["op"]))
        seconds = 2 * config["idle_s"] + segments + leg.get("setup_s", 30 if leg["kind"] == "desktop" else 90)
        entry = {"name": leg["name"], "kind": leg["kind"], "op": leg["op"], "seconds": round(seconds)}
        if leg["kind"] == "phone":
            entry["budget_calls"] = phone_budget(config, leg)
            entry["tokens_per_segment"] = {str(r): phone_tokens(config, leg, r) for r in rows}
        legs.append(entry)
        total += seconds + config.get("cooldown_s", 10)
    return {"legs": legs, "total_s": round(total), "total_min": round(total / 60, 1)}


# ---------------------------------------------------------------- desktop legs

def desktop_command(config: Mapping[str, Any], leg: Mapping[str, Any], out_path: Path) -> list[str]:
    cadence = config["cadence"]
    return [str(leg["bench"]), "--model", str(leg["model"]), "--op", leg["op"], "--layer", str(leg.get("layer", 0)),
            "--backend", leg["backend"], "--threads", str(leg.get("threads", 8)),
            "--rows", ",".join(str(r) for r in config.get("rows", [1, 2, 4])),
            "--seconds", str(config["segment_seconds"]), "--warmup-tokens", str(cadence["warmup_tokens"]),
            "--calls-per-token", str(cadence["calls_per_token"]), "--gap-call-ms", str(cadence["gap_call_ms"]),
            "--gap-token-ms", str(cadence["gap_token_ms"]), "--idle-before-s", str(config["idle_s"]),
            "--idle-after-s", str(config["idle_s"]), "--seed", str(leg.get("seed", 1234)), "--out", str(out_path)]


def run_desktop_leg(config: Mapping[str, Any], leg: Mapping[str, Any], directory: Path,
                    run: Callable = subprocess.run) -> dict[str, Any]:
    out_path = directory / "LEG.jsonl"
    command = desktop_command(config, leg, out_path)
    env = dict(os.environ)
    env.update({key: str(value) for key, value in leg.get("env", {}).items()})
    with (directory / "bench.stderr").open("w") as stderr:
        completed = run(command, stdin=subprocess.DEVNULL, stdout=stderr, stderr=subprocess.STDOUT, env=env,
                        timeout=leg.get("timeout_s", 3600))
    if completed.returncode != 0:
        raise RuntimeError(f"{leg['name']}: bench exited {completed.returncode} (see bench.stderr)")
    return {"command": command, "exit_code": completed.returncode}


# ---------------------------------------------------------------- phone legs

def synthetic_rows(n_embd: int, rows: int, seed: int) -> bytes:
    """Deterministic f16 activations ~N(0, 1) (the inputs' values do not change GEMV time)."""
    rng = random.Random(seed)
    values = [rng.gauss(0.0, 1.0) for _ in range(n_embd * rows)]
    return struct.pack("<%de" % len(values), *values)


def receive(stream: socket.socket, size: int) -> bytes:
    chunks, remaining = [], size
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            raise RuntimeError("worker closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def hello(stream: socket.socket, leg: Mapping[str, Any], artifact_sha256: str, max_tokens: int) -> None:
    flags = F16_IO_FLAG | (SWIGLU_FLAG if leg["activation"] == "swiglu" else 0)
    mask = 1 << int(leg["layer"])
    if leg["protocol_version"] == 6:
        stream.sendall(HELLO_V6.pack(PROTOCOL_MAGIC, 6, 1, mask, leg["n_embd"], leg["n_ff"], flags, max_tokens,
                                     bytes.fromhex(artifact_sha256[7:])))
        response = receive(stream, HELLO_V6_RESPONSE_BYTES)
    else:
        stream.sendall(HELLO_V5.pack(PROTOCOL_MAGIC, 5, 1, mask, leg["n_embd"], leg["n_ff"], flags, max_tokens))
        response = receive(stream, HELLO_V5_RESPONSE_BYTES)
    magic, version, message, status = struct.unpack_from("<IHHH", response)
    if (magic, version, message, status) != (PROTOCOL_MAGIC, leg["protocol_version"], 2, 0):
        raise RuntimeError(f"HELLO refused: magic {magic:#x} version {version} message {message} status {status}")


def execute(stream: socket.socket, leg: Mapping[str, Any], request_id: int, payload: bytes, rows: int,
            clock: Callable[[], int]) -> tuple[int, int, int]:
    version = leg["protocol_version"]
    t0 = clock()
    stream.sendall(EXECUTE.pack(PROTOCOL_MAGIC, version, 3, request_id, int(leg["layer"]), leg["n_embd"] * rows,
                                len(payload), fnv32(payload), leg["n_ff"], rows) + payload)
    response = EXECUTE_RESPONSE.unpack(receive(stream, EXECUTE_RESPONSE.size))
    output = receive(stream, len(payload))
    t1 = clock()
    if response[:6] != (PROTOCOL_MAGIC, version, 4, 0, 0, request_id) or response[9] != fnv32(output):
        raise RuntimeError(f"EXECUTE {request_id} failed: {response[:10]}")
    return t0, t1, int(response[-1])


def drive_phone_leg(port: int, config: Mapping[str, Any], leg: Mapping[str, Any], artifact_sha256: str,
                    max_tokens: int, out_path: Path, *, clock: Callable[[], int] = time.monotonic_ns,
                    sleep: Callable[[float], None] = time.sleep, connect: Callable = socket.create_connection,
                    meta: Mapping[str, Any] | None = None, close: str = "execute") -> int:
    """Idle-before, one segment per row count, idle-after, then the close; returns calls sent.

    ``close``: ``execute`` = a closing call that exhausts the worker's budget (TCP workers), ``shutdown`` = the
    USB bridge's local shutdown request (request id 0, layer -1), after which the bridge ends the phone session."""
    if close not in ("execute", "shutdown"):
        raise ValueError("close must be execute or shutdown")
    cadence, rows_list = config["cadence"], config.get("rows", [1, 2, 4])
    logical = 3 * leg["n_embd"] * leg["n_ff"] * 2
    sent = 0
    with out_path.open("w") as out, connect(("127.0.0.1", port), timeout=120) as stream:
        def emit(record: Mapping[str, Any]) -> None:
            out.write(json.dumps(record, sort_keys=True) + "\n")
            out.flush()
        stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        hello(stream, leg, artifact_sha256, max_tokens)
        emit({"kind": "meta", "schema": "ws16-phone-leg-v1", "op": "ffn", "device": leg["device"],
              "layer": leg["layer"], "n_embd": leg["n_embd"], "n_ff": leg["n_ff"], "logical_bytes": logical,
              "calls_per_token": cadence["calls_per_token"], "gap_call_ms": cadence["gap_call_ms"],
              "gap_token_ms": cadence["gap_token_ms"], "warmup_tokens": cadence["warmup_tokens"],
              "protocol_version": leg["protocol_version"], "transport": leg["transport"], **(meta or {})})
        t0 = clock()
        sleep(config["idle_s"])
        emit({"kind": "idle", "phase": "before", "t0_ns": t0, "t1_ns": clock()})
        payloads = {rows: synthetic_rows(leg["n_embd"], rows, 1000 + rows) for rows in rows_list}
        for rows in rows_list:
            tokens = phone_tokens(config, leg, rows)
            start, measured_start, measured = clock(), None, 0
            for token in range(-cadence["warmup_tokens"], tokens):
                if token == 0:
                    measured_start = clock()
                for position in range(cadence["calls_per_token"]):
                    sent += 1
                    c0, c1, compute_us = execute(stream, leg, sent, payloads[rows], rows, clock)
                    emit({"kind": "call", "rows": rows, "warmup": token < 0, "token": token, "position": position,
                          "t_start_ns": c0, "t_end_ns": c1, "rpc_us": (c1 - c0) / 1e3, "compute_us": compute_us})
                    measured += token >= 0
                    sleep((cadence["gap_call_ms"] if position + 1 < cadence["calls_per_token"]
                           else cadence["gap_token_ms"]) / 1000.0)
            emit({"kind": "segment", "rows": rows, "t0_ns": measured_start if measured_start is not None else start,
                  "t1_ns": clock(), "calls": measured, "warmup_t0_ns": start})
        t0 = clock()
        sleep(config["idle_s"])
        emit({"kind": "idle", "phase": "after", "t0_ns": t0, "t1_ns": clock()})
        if close == "shutdown":
            stream.sendall(EXECUTE.pack(PROTOCOL_MAGIC, leg["protocol_version"], 3, 0, -1, 0, 0, 0, 0, 0))
        else:
            sent += 1  # the closing call exhausts the budget: the worker exits normally, no drain needed
            execute(stream, leg, sent, payloads[rows_list[0]], rows_list[0], clock)
    return sent


def phone_worker_configuration(config: Mapping[str, Any], leg: Mapping[str, Any]):
    """AdbTcpWorkerConfiguration (+ AOA bridge configuration JSON) of a phone leg, with the leg's budget.

    ``evidence``: a qualified helper evidence bundle (pins, environment, transport) loaded with the scheduler's
    own loader; only the layer mask and the finite budget change. ``worker``: explicit fields (OP15 HTP worker in
    TCP mode), pinned by ``prepare``."""
    import dataclasses
    from research_dev.scheduler.adapters.phone_tcp_session import AdbTcpWorkerConfiguration
    # a budget exit would break the USB bridge (it expects the worker until its own shutdown request)
    budget = 0 if leg["transport"] == "functionfs-bridge" else phone_budget(config, leg)
    if "evidence" in leg:
        from research_dev.scheduler.campaigns.burstgpt.helper_phone_evidence import load_helper_evidence
        evidence = load_helper_evidence(Path(leg["evidence"]))
        worker = dataclasses.replace(evidence.worker, layer_mask=1 << int(leg["layer"]), max_requests=budget)
        if (worker.n_embd, worker.columns) != (int(leg["n_embd"]), int(leg["n_ff"])):
            raise SystemExit(leg["name"] + ": n_embd / n_ff differ from the evidence worker")
        return worker, (evidence.aoa_bridge.to_json() if evidence.aoa_bridge is not None else None)
    worker = dict(leg["worker"])
    worker.update(adb_path=Path(worker.get("adb_path", "/usr/bin/adb")),
                  library_directories=tuple(worker["library_directories"]), layer_mask=1 << int(leg["layer"]),
                  n_embd=int(leg["n_embd"]), columns=int(leg["n_ff"]), swiglu=leg["activation"] == "swiglu",
                  max_requests=budget)
    return AdbTcpWorkerConfiguration(**worker), leg.get("aoa_bridge")


def open_phone_session(config: Mapping[str, Any], leg: Mapping[str, Any]):
    worker, aoa = phone_worker_configuration(config, leg)
    if leg["transport"] == "adb-tcp":
        from research_dev.scheduler.adapters.phone_tcp_session import AdbTcpPhoneWorkerSession
        return AdbTcpPhoneWorkerSession(worker), worker
    if leg["transport"] == "functionfs-bridge":
        return FunctionFsBridgePhoneWorkerSession(worker, leg["functionfs"]), worker
    from research_dev.scheduler.adapters.phone_aoa_session import AoaBridgeConfiguration, AoaBridgePhoneWorkerSession
    from research_dev.scheduler.campaigns.burstgpt.tools import pixel_transport_ab
    if aoa is None:
        raise SystemExit(leg["name"] + ": an aoa-bridge leg needs the evidence's aoa_bridge section")
    harness = pixel_transport_ab.Harness({"phone": {"serial": worker.serial, "adb_port": worker.adb_port,
                                                    "adb_path": str(worker.adb_path), "as_root": worker.as_root},
                                          "aoa_bridge": aoa})
    harness.ensure_usb_mode("accessory")
    return AoaBridgePhoneWorkerSession(worker, AoaBridgeConfiguration.from_json(aoa)), worker


class FunctionFsBridgePhoneWorkerSession:
    """The OP15 leg over the production FunctionFS DMA-BUF path.

    The production phone session script (rig.json ``phone.session_script``) starts the pinned worker with
    ``--ffs-root``, swaps the Android gadget for the FunctionFS one (18d1:2d00, no adb while it runs) and restores
    Android USB when the worker exits. On the host, ``llama-ffn-split-usb-bridge`` accepts this harness's TCP v6
    connection and forwards every call over FunctionFS. The worker runs without a budget; the driver ends the leg
    with the bridge's local shutdown request, the bridge ends the phone session (layer -1) and exits with its
    FFNDMABUF receipt. ``stop`` then waits for adb, acknowledges the restoration (restore_android_usb.sh restarts
    adbd without it) and copies the phone session's logs. A start or stop that leaves the gadget up uses the
    production close tool (``usb_close_path``) or the restore script, so the phone never waits for the watchdog."""

    close_request = "shutdown"
    VENDOR_ID, PRODUCT_ID = "18d1", "2d00"
    # set by direct_phone_ffn_session.sh itself; a leg's worker_environment may repeat them, nothing else
    SCRIPT_FIXED_ENVIRONMENT = {"GGML_HEXAGON_MBUF": "4192", "GGML_HEXAGON_NDEV": "1", "S41_DISABLE_GRAPH_CACHE": "1"}

    def __init__(self, worker, functionfs: Mapping[str, Any], *, run: Callable = subprocess.run,
                 popen: Callable = subprocess.Popen, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, probe: Callable | None = None,
                 restored: Callable | None = None, listening: Callable[[int], bool] | None = None) -> None:
        from research_dev.scheduler.adapters import bridge
        missing = [key for key in FUNCTIONFS_KEYS if key not in functionfs]
        if missing:
            raise ValueError("functionfs section needs " + ",".join(missing))
        worker_dir = str(Path(worker.worker_path).parent)
        environment = dict(worker.worker_environment)
        fixed = dict(self.SCRIPT_FIXED_ENVIRONMENT, ADSP_LIBRARY_PATH=worker_dir)
        if tuple(worker.library_directories) != (worker_dir,) or any(
                key not in fixed and key not in ("GGML_HEXAGON_NHVX", "GGML_HEXAGON_VMEM")
                or key in fixed and value != fixed[key] for key, value in environment.items()):
            raise ValueError("the session script loads the worker's own directory with its fixed environment; "
                             "only GGML_HEXAGON_NHVX / GGML_HEXAGON_VMEM may differ")
        self.worker, self.functionfs = worker, dict(functionfs)
        self.nhvx = environment.get("GGML_HEXAGON_NHVX", "4")
        self.vmem = environment.get("GGML_HEXAGON_VMEM", "3328")
        self._run, self._popen, self._sleep, self._clock = run, popen, sleep, clock
        self._probe = probe or (lambda: bridge.probe_functionfs_usb_device(
            vendor_id=self.VENDOR_ID, product_id=self.PRODUCT_ID))
        self._restored = restored or (lambda timeout_s: bridge.verify_android_usb_restored(
            serial=worker.serial, adb_port=worker.adb_port,
            minimum_speed_mbps=int(self.functionfs.get("minimum_usb_speed_mbps", 5000)), timeout_s=timeout_s))
        self._listening = listening or self._tcp_listening
        self._parse_terminal = bridge.parse_functionfs_bridge_terminal
        self._bridge: subprocess.Popen | None = None
        self._bridge_log: Path | None = None
        self._remote_root: str | None = None
        self._log_dir: Path | None = None
        self._boot_id = ""
        self.restoration: dict[str, Any] | None = None
        self.phone_terminal_status: str | None = None
        self.orphans_killed: list[int] = []

    @property
    def active(self) -> bool:
        return self._bridge is not None or self._remote_root is not None

    def _root_shell(self, command: str, *, timeout_s: float = 60) -> str:
        worker = self.worker
        return self._run([str(worker.adb_path), "-P", str(worker.adb_port), "-s", worker.serial, "shell",
                          "su -c " + shlex.quote(command)], check=True, capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, timeout=timeout_s).stdout

    @staticmethod
    def _tcp_listening(port: int) -> bool:
        suffix = "0100007F:%04X" % port
        with open("/proc/net/tcp", encoding="ascii") as table:
            return any(len(columns) > 3 and columns[1] == suffix and columns[3] == "0A"
                       for columns in (line.split() for line in table))

    def _receipt(self, phase: str, details: Mapping[str, Any]):
        from research_dev.scheduler.adapters.phone_tcp_session import AdbTcpWorkerReceipt
        return AdbTcpWorkerReceipt(phase, self.worker.device_id, self.worker.serial, self._boot_id, dict(details))

    def _processes(self) -> list[tuple[int, int, str]]:
        rows = []
        for line in self._root_shell("ps -A -o PID,PPID,ARGS").splitlines():
            fields = line.split(None, 2)
            if len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit():
                rows.append((int(fields[0]), int(fields[1]), fields[2]))
        return rows

    def _worker_running(self) -> bool:
        """A process executing the worker (the script's own arguments name it too: not a worker)."""
        return any(args.split()[:1] == [self.worker.worker_path] for _, _, args in self._processes())

    def _kill_orphaned_watchdog(self) -> list[int]:
        """SIGKILL what is left of this session's script: its watchdog subshell outlives the script's cleanup
        (the plain kill does not end it while it waits on ``sleep session_timeout_s``; 2026-10-11, PID re-parented
        to init 20 min after the smoke) and still carries the script's arguments; then the subshell's sleep."""
        rows = self._processes()
        script, root = self.functionfs["session_script"], self._remote_root
        parents = [pid for pid, _, args in rows if script in args and root in args.split()]
        children = [pid for pid, ppid, _ in rows if ppid in parents]
        if parents:
            self._root_shell("kill -KILL " + " ".join(map(str, parents + children)) + " 2>/dev/null; true",
                             timeout_s=10)
        return parents + children

    def _gadget_up(self) -> bool:
        try:
            self._probe()
        except Exception:  # noqa: BLE001 - absent (or not unique): no live FunctionFS function to close
            return False
        return True

    def preflight(self):
        """Refuse a foreign kernel, a running worker, a live FunctionFS device, a busy bridge port and any
        phone or host file that differs from its pin."""
        worker, functionfs = self.worker, self.functionfs
        release = self._root_shell("uname -r").strip()
        if release != functionfs["kernel_release"]:
            raise RuntimeError(f"phone kernel {release!r} is not the FunctionFS kernel {functionfs['kernel_release']}")
        if self._worker_running():
            raise RuntimeError(f"{worker.serial} already runs {worker.worker_path}")
        if self._gadget_up():
            raise RuntimeError("a FunctionFS device (18d1:2d00) is already enumerated on this host")
        if self._listening(int(functionfs["bridge_port"])):
            raise RuntimeError(f"bridge port {functionfs['bridge_port']} is in use")
        pins = dict(worker.expected_sha256_by_path, **{functionfs["session_script"]: functionfs["session_script_sha256"],
                                                       functionfs["restore_script"]: functionfs["restore_script_sha256"]})
        observed = {}
        for line in self._root_shell("sha256sum " + " ".join(shlex.quote(path) for path in pins),
                                     timeout_s=900).splitlines():
            digest, _, path = line.strip().partition("  ")
            observed[path] = "sha256:" + digest
        mismatched = sorted(path for path, digest in pins.items() if observed.get(path) != digest)
        bridge_sha256 = "sha256:" + hashlib.sha256(Path(functionfs["bridge_path"]).read_bytes()).hexdigest()
        if bridge_sha256 != functionfs["bridge_sha256"]:
            mismatched.append(functionfs["bridge_path"])
        if mismatched:
            raise RuntimeError("files differ from their pinned sha256: " + ",".join(mismatched))
        self._boot_id = self._root_shell("cat /proc/sys/kernel/random/boot_id").strip()
        return self._receipt("preflight", {"kernel_release": release, "observed_sha256_by_path": observed,
                                           "bridge_sha256": bridge_sha256, "transport": "functionfs-bridge"})

    def _usbfs_bytes(self) -> int:
        override = self.functionfs.get("usbfs_available_bytes")
        if override is not None:
            return int(override)
        return int(Path("/sys/module/usbcore/parameters/usbfs_memory_mb").read_text().strip()) * 1024 * 1024

    def _remote_command(self, remote_root: str) -> str:
        worker, functionfs = self.worker, self.functionfs
        from research_dev.scheduler.adapters.phone_helpers import layer_spec
        words = [
            "SCHEDULER_ANDROID_GADGET=" + functionfs["android_gadget_path"],
            "SCHEDULER_FUNCTIONFS_GADGET=" + functionfs["functionfs_gadget_path"],
            "SCHEDULER_FUNCTIONFS_ROOT=" + functionfs["functionfs_root_path"],
            "SCHEDULER_PHONE_UDC=" + functionfs["phone_usb_controller"],
            "S41_FFN_F16_IO=1", "S41_FFN_STAGED_DMABUF=0", "S41_FFN_MAX_TOKENS=" + str(worker.max_tokens),
            "S41_FFN_COLUMN_QUANTUM=" + str(worker.column_quantum),
            "S41_FFN_QUEUE_DEPTH=" + str(functionfs.get("queue_depth", 1)),
            "S41_FFN_VMEM=" + self.vmem, "GGML_HEXAGON_NHVX=" + self.nhvx,
            "sh", functionfs["session_script"], worker.worker_path, worker.shard_path, layer_spec(worker.layer_mask),
            str(worker.columns), worker.backend, remote_root, functionfs["restore_script"],
            str(functionfs.get("session_timeout_s", 1800)), "0", worker.artifact_sha256,
        ]
        return ("mkdir -p " + shlex.quote(remote_root) + " && nohup env " + shlex.join(words)
                + " > " + shlex.quote(remote_root + "/launch.log") + " 2>&1 < /dev/null &")

    def start(self, log_path: Path):
        """Launch the phone session, wait for the FunctionFS device, then start the bridge (it accepts the
        driver's connection and opens USB on the driver's HELLO, so readiness here = its port listens)."""
        if self.active:
            raise RuntimeError("functionfs-bridge session is already active")
        if not self._boot_id:
            self.preflight()
        functionfs = self.functionfs
        token = hashlib.sha256(f"ws16:{os.getpid()}:{time.monotonic_ns()}".encode()).hexdigest()[:16]
        self._remote_root = functionfs["session_root"].rstrip("/") + "/" + token
        self._log_dir = log_path.parent
        try:
            self._root_shell(self._remote_command(self._remote_root), timeout_s=30)
            deadline = self._clock() + float(functionfs.get("launch_timeout_s", 240))
            next_check, usb, last = self._clock() + 2.0, None, "FunctionFS USB did not enumerate"
            minimum = int(functionfs.get("minimum_usb_speed_mbps", 5000))
            while usb is None:
                if self._clock() > deadline:
                    raise RuntimeError(last)
                try:
                    observed = self._probe()
                    if observed.negotiated_speed_mbps >= minimum:
                        usb = observed
                        break
                    last = f"FunctionFS USB at {observed.negotiated_speed_mbps} Mb/s (< {minimum})"
                except Exception as error:  # noqa: BLE001 - not enumerated yet
                    last = str(error)
                if self._clock() >= next_check:  # adb still answers until the gadget swap
                    next_check = self._clock() + 2.0
                    try:
                        status = self._root_shell("cat " + shlex.quote(self._remote_root + "/terminal.status")
                                                  + " 2>/dev/null || true", timeout_s=5).strip()
                    except Exception:  # noqa: BLE001 - adb is gone: the swap is under way
                        status = ""
                    if status:
                        raise RuntimeError("phone session ended before USB enumeration: status=" + status)
                self._sleep(0.25)
            usbfs_bytes = self._usbfs_bytes()
            command = [functionfs["bridge_path"], "127.0.0.1", str(functionfs["bridge_port"]),
                       functionfs.get("allocator", "devmem"), str(functionfs.get("queue_depth", 1)),
                       str(usbfs_bytes), functionfs.get("transport_generation", "functionfs-dmabuf-async-ring-v2")]
            environment = dict(os.environ)
            if functionfs.get("bridge_library_path"):
                environment["LD_LIBRARY_PATH"] = functionfs["bridge_library_path"]
            self._bridge_log = log_path.with_name("bridge.log")
            with self._bridge_log.open("x") as log:
                self._bridge = self._popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, env=environment)
            listen_deadline = self._clock() + 15
            while not self._listening(int(functionfs["bridge_port"])):
                if self._bridge.poll() is not None or self._clock() > listen_deadline:
                    raise RuntimeError("USB bridge did not listen: " + self._bridge_log.read_text()[-2000:])
                self._sleep(0.1)
        except BaseException as error:
            cleanup = self._release(force_close=True)
            if cleanup:
                raise RuntimeError(f"{error}; cleanup: {'; '.join(cleanup)}") from error
            raise
        return self._receipt("launch", {"remote_root": self._remote_root, "bridge_command": command,
                                        "bridge_pid": self._bridge.pid, "usbfs_available_bytes": usbfs_bytes,
                                        "usb": dataclasses.asdict(usb), "transport": "functionfs-bridge"})

    def transport_contract(self):
        return SimpleNamespace(control_port=int(self.functionfs["bridge_port"]))

    def _close_gadget(self) -> list[str]:
        """End a phone session whose bridge did not: the production close tool, else the restore script is
        out of reach (no adb while the gadget is up) and the watchdog (session_timeout_s) restores USB."""
        close_path = self.functionfs.get("usb_close_path")
        if not close_path:
            return ["FunctionFS gadget still up and no usb_close_path: the phone watchdog restores USB after "
                    + str(self.functionfs.get("session_timeout_s", 1800)) + " s"]
        worker = self.worker
        flags = F16_IO_FLAG | (SWIGLU_FLAG if worker.swiglu else 0)
        result = self._run([close_path, "0x" + self.VENDOR_ID, "0x" + self.PRODUCT_ID, hex(worker.layer_mask),
                            str(worker.n_embd), str(worker.columns), str(worker.max_tokens), str(flags),
                            str(self._usbfs_bytes()),
                            self.functionfs.get("transport_generation", "functionfs-dmabuf-async-ring-v2"),
                            worker.artifact_sha256], check=False, capture_output=True, text=True, timeout=30)
        if result.returncode != 0 or "FFNUSB_CLOSE status=ok" not in result.stdout:
            return ["USB close failed: " + (result.stderr.strip() or result.stdout.strip())[-500:]]
        return []

    def _release(self, *, force_close: bool) -> list[str]:
        """Bridge gone, gadget gone, Android USB back with the host's acknowledgement, phone logs copied."""
        errors: list[str] = []
        if self._bridge is not None and self._bridge.poll() is None:
            self._bridge.terminate()
            try:
                self._bridge.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._bridge.kill()
                self._bridge.wait(timeout=10)
        if self._remote_root is not None:
            deadline = self._clock() + (5 if force_close else 30)
            while self._gadget_up() and self._clock() < deadline:
                self._sleep(0.5)
            if self._gadget_up():
                errors += self._close_gadget()
            else:
                try:  # a start that failed before the gadget swap: the script's own cleanup restores USB
                    self._root_shell("test -f " + shlex.quote(self._remote_root + "/terminal.status")
                                     + " || sh " + shlex.quote(self.functionfs["restore_script"]) + " "
                                     + shlex.quote(self._remote_root), timeout_s=30)
                except Exception:  # noqa: BLE001 - adb not back yet; the restoration wait below decides
                    pass
            try:
                self.restoration = dataclasses.asdict(self._restored(90))
            except Exception as error:  # noqa: BLE001
                errors.append("Android USB not restored: " + str(error))
            else:
                try:
                    self._root_shell("touch " + shlex.quote(self._remote_root + "/host-ack"), timeout_s=10)
                    self.orphans_killed = self._kill_orphaned_watchdog()
                    self.phone_terminal_status = self._root_shell(
                        "cat " + shlex.quote(self._remote_root + "/terminal.status") + " 2>/dev/null || true",
                        timeout_s=10).strip()
                    for name in ("launch.log", "session.log", "worker.log", "restore-watchdog.log"):
                        text = self._root_shell("cat " + shlex.quote(self._remote_root + "/" + name)
                                                + " 2>/dev/null || true", timeout_s=30)
                        (self._log_dir / ("phone-" + name)).write_text(text)
                except Exception as error:  # noqa: BLE001
                    errors.append("phone session acknowledgement failed: " + str(error))
        self._bridge, self._remote_root = None, None
        return errors

    def stop(self, *, served_calls: int):
        """The bridge exits after the driver's shutdown request; its receipt must count every call."""
        errors, terminal, returncode = [], None, None
        self.restoration, self.phone_terminal_status = None, None
        if self._bridge is not None:
            try:
                returncode = self._bridge.wait(timeout=60)
            except subprocess.TimeoutExpired:
                errors.append("USB bridge did not exit after the shutdown request")
        if self._bridge_log is not None and self._bridge_log.exists():
            try:
                terminal = self._parse_terminal(self._bridge_log.read_text(errors="backslashreplace").splitlines())
            except Exception as error:  # noqa: BLE001
                errors.append(str(error))
        if returncode not in (None, 0):
            errors.append(f"USB bridge exited {returncode}")
        if terminal is not None and terminal.calls != served_calls:
            errors.append(f"USB bridge served {terminal.calls} calls, the driver sent {served_calls}")
        errors += self._release(force_close=bool(errors))
        if self.phone_terminal_status not in (None, "0"):
            errors.append("phone session exit status " + str(self.phone_terminal_status))
        if errors:
            raise RuntimeError("; ".join(errors))
        return self._receipt("stop", {"bridge_returncode": returncode, "bridge_terminal": dict(terminal.values),
                                      "restoration": self.restoration, "phone_terminal_status":
                                      self.phone_terminal_status, "orphans_killed": self.orphans_killed,
                                      "served_calls": served_calls})


def run_phone_leg(config: Mapping[str, Any], leg: Mapping[str, Any], directory: Path,
                  session_factory: Callable = open_phone_session, **drive_options: Any) -> dict[str, Any]:
    session, worker = session_factory(config, leg)
    receipts: dict[str, Any] = {"preflight": session.preflight().to_json()}
    receipts["launch"] = session.start(directory / "worker.log").to_json()
    sent = 0
    try:
        sent = drive_phone_leg(session.transport_contract().control_port, config, leg, worker.artifact_sha256,
                               worker.max_tokens, directory / "LEG.jsonl",
                               meta={"serial": worker.serial, "backend": worker.backend,
                                     "worker_path": worker.worker_path, "shard_path": worker.shard_path},
                               close=getattr(session, "close_request", "execute"), **drive_options)
    finally:
        if session.active:
            receipts["stop"] = session.stop(served_calls=sent).to_json()
    receipts["calls_sent"] = sent
    return receipts


# ---------------------------------------------------------------- run

def default_sampler(gpu_index: int | None = None):
    """The campaign's host sampler (0.2 s); ``gpu_index`` selects one GPU on a multi-GPU host (local tests)."""
    from research_dev.scheduler.adapters.host_runtime import HostEnergySampler
    from research_dev.scheduler.campaigns.burstgpt.tools.default_llamacpp_baseline import host_metric_callbacks
    return HostEnergySampler(host_metric_callbacks(gpu_index), interval_s=0.2)


def run(config: Mapping[str, Any], output: Path, *, only: set[str] | None = None,
        sampler_factory: Callable = default_sampler, desktop_runner: Callable = run_desktop_leg,
        phone_runner: Callable = run_phone_leg, sleep: Callable[[float], None] = time.sleep) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "CONFIG.json", config)
    write_json(output / "PLAN.json", plan(config))
    sampler = (sampler_factory(config.get("host", {}).get("gpu_index"))
               if sampler_factory is default_sampler else sampler_factory())
    sampler.start()
    results: list[dict[str, Any]] = []
    try:
        for leg in config["legs"]:
            if only and leg["name"] not in only:
                continue
            directory = output / leg["name"]
            directory.mkdir()
            write_json(directory / "LEG_CONFIG.json", leg)
            entry: dict[str, Any] = {"name": leg["name"], "kind": leg["kind"], "status": "FAILED"}
            try:
                runner = desktop_runner if leg["kind"] == "desktop" else phone_runner
                entry["receipts"] = runner(config, leg, directory)
                entry["status"] = "PASS"
            except Exception as error:  # noqa: BLE001 - one leg's failure is recorded; the others still run
                entry["error"] = f"{type(error).__name__}: {error}"
            results.append(entry)
            write_json(output / "RUN_RESULT.json", results)
            sleep(config.get("cooldown_s", 10))
    finally:
        sampler.stop()
        with (output / "HOST_SAMPLES.jsonl").open("w") as stream:
            for row in sampler.rows():
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        try:
            write_json(output / "HOST_SAMPLER_DIAGNOSTICS.json", sampler.diagnostics())
        except AttributeError:
            pass
    return results


# ---------------------------------------------------------------- analysis

def _desktop_energy(rows: Sequence[Mapping[str, Any]], t0_ns: int, t1_ns: int) -> dict[str, float | None]:
    from research_dev.scheduler.campaigns.burstgpt.tools.default_llamacpp_baseline import host_energy
    energy = host_energy(rows, t0_ns, t1_ns)
    return {"cpu": energy.get("cpu_package_energy_j"), "gpu": energy.get("gpu_board_energy_j")}


def _phone_energy(meter: Mapping[str, Any], logs: Mapping[str, list], t0_ns: int, t1_ns: int
                  ) -> dict[str, dict[str, Any]]:
    from research_dev.scheduler.campaigns.burstgpt.tools.fleet_energy import phone_energy
    out = {}
    for label, anchor in sorted(meter.items()):
        if not isinstance(anchor, dict) or label not in logs:
            continue
        out[label] = phone_energy(label, anchor, logs[label], t0_ns / 1e9, t1_ns / 1e9, 0.85)
    return out


def load_meter(meter_path: Path | None) -> tuple[dict[str, Any], dict[str, list]]:
    if meter_path is None:
        return {}, {}
    from research_dev.scheduler.campaigns.burstgpt.tools.fleet_energy import parse_sampler
    meter = json.loads(Path(meter_path).read_text())
    logs = {}
    for label, anchor in meter.items():
        if isinstance(anchor, dict) and anchor.get("local"):
            local = Path(anchor["local"])
            path = local if local.is_file() else Path(meter_path).parent / local.name
            if path.is_file():
                logs[label] = parse_sampler(path.read_text())
    return meter, logs


def window_energy(host_rows, meter, logs, t0_ns: int, t1_ns: int) -> dict[str, Any]:
    energy: dict[str, Any] = dict(_desktop_energy(host_rows, t0_ns, t1_ns)) if host_rows else {"cpu": None, "gpu": None}
    flags: dict[str, list] = {}
    uncertainty: dict[str, float | None] = {}
    for label, phone in _phone_energy(meter, logs, t0_ns, t1_ns).items():
        energy[label] = phone.get("system_j")
        flags[label] = phone.get("flags", [])
        uncertainty[label] = (phone.get("battery_counter") or {}).get("uncertainty_j")
    return {"j": energy, "flags": flags, "uncertainty_j": uncertainty, "seconds": (t1_ns - t0_ns) / 1e9}


DEVICE_DOMAIN = {"CPU": "cpu", "CUDA0": "gpu", "op15": "op15", "pixel": "pixel"}


def analyze_leg(directory: Path, host_rows, meter, logs) -> dict[str, Any]:
    leg_config = json.loads((directory / "LEG_CONFIG.json").read_text())
    records = read_jsonl(directory / "LEG.jsonl")
    meta = next(r for r in records if r["kind"] == "meta")
    idles = [r for r in records if r["kind"] == "idle"]
    domain = DEVICE_DOMAIN[leg_config["backend"]] if leg_config["kind"] == "desktop" else leg_config["device"]
    idle = [window_energy(host_rows, meter, logs, r["t0_ns"], r["t1_ns"]) for r in idles]
    idle_seconds = sum(w["seconds"] for w in idle)
    domains = sorted({key for w in idle for key, value in w["j"].items() if value is not None})
    idle_w = {d: sum(w["j"][d] for w in idle) / idle_seconds for d in domains
              if idle_seconds > 0 and all(w["j"].get(d) is not None for w in idle)}
    out = {"name": leg_config["name"], "kind": leg_config["kind"], "op": leg_config["op"], "device": domain,
           "backend": leg_config.get("backend", meta.get("backend")), "logical_bytes": meta["logical_bytes"],
           "idle_w": idle_w, "idle_seconds": idle_seconds, "segments": []}
    for segment in (r for r in records if r["kind"] == "segment"):
        calls = [r for r in records if r["kind"] == "call" and r["rows"] == segment["rows"] and not r["warmup"]]
        n = len(calls)
        window = window_energy(host_rows, meter, logs, segment["t0_ns"], segment["t1_ns"])
        incremental = {d: window["j"][d] - idle_w[d] * window["seconds"] for d in idle_w
                       if window["j"].get(d) is not None}
        compute_ms = [c["compute_us"] / 1000 for c in calls]
        rpc_ms = [c["rpc_us"] / 1000 for c in calls if "rpc_us" in c]
        median_compute = statistics.median(compute_ms) if compute_ms else None
        flags = sorted({flag for value in window["flags"].values() for flag in value})
        device_inc = incremental.get(domain)
        unc = window["uncertainty_j"].get(domain)
        if device_inc is not None and unc and abs(device_inc) < 3 * unc:
            flags.append("DEVICE_ENERGY_BELOW_3X_COUNTER_RESOLUTION")
        if domain not in idle_w:
            flags.append("DEVICE_DOMAIN_NOT_MEASURED")
        row = {"rows": segment["rows"], "calls": n, "seconds": window["seconds"],
               "compute_ms_median": median_compute,
               "compute_ms_mean": statistics.fmean(compute_ms) if compute_ms else None,
               "rpc_ms_median": statistics.median(rpc_ms) if rpc_ms else None,
               "bandwidth_gbs": meta["logical_bytes"] / (median_compute / 1000) / 1e9 if median_compute else None,
               "gross_j": window["j"], "incremental_j": incremental,
               "device_gross_j_per_call": window["j"].get(domain) / n if n and window["j"].get(domain) is not None else None,
               "device_j_per_call": device_inc / n if n and device_inc is not None else None,
               "desktop_j_per_call": (sum(incremental.get(d, 0.0) for d in DESKTOP_DOMAINS) / n
                                      if n and domain not in DESKTOP_DOMAINS and all(d in incremental for d in DESKTOP_DOMAINS)
                                      else None),
               "system_j_per_call": sum(incremental.values()) / n if n and incremental else None,
               "flags": sorted(set(flags))}
        row["device_nj_per_byte"] = (row["device_j_per_call"] / meta["logical_bytes"] * 1e9
                                     if row["device_j_per_call"] is not None else None)
        out["segments"].append(row)
    return out


def analyze(output: Path, meter_path: Path | None = None) -> dict[str, Any]:
    host_path = output / "HOST_SAMPLES.jsonl"
    host_rows = read_jsonl(host_path) if host_path.exists() else []
    meter, logs = load_meter(meter_path)
    legs = []
    for directory in sorted(p for p in output.iterdir() if p.is_dir() and (p / "LEG.jsonl").exists()):
        legs.append(analyze_leg(directory, host_rows, meter, logs))
    return {"schema": RESULT_SCHEMA, "output": str(output), "meter": str(meter_path) if meter_path else None,
            "convention": "incremental = (segment energy - leg idle power x duration) / calls; device = executing "
                          "domain; system = sum of all domains; nJ/byte over logical two-byte weights",
            "legs": legs}


def _fmt(value: float | None, digits: int = 2) -> str:
    return "\\TBD" if value is None else f"{value:.{digits}f}"


def latex(result: Mapping[str, Any], cpu_reference: str | None = None) -> str:
    """Rows for the paper: Table 1 (FFN at 1 row), Table 2 (call time vs rows), Table 4 (projections and head)."""
    def seg(leg, rows=1):
        return next((s for s in leg["segments"] if s["rows"] == rows), None)
    legs = result["legs"]
    ffn = [leg for leg in legs if leg["op"] == "ffn"]
    reference = next((leg for leg in ffn if leg["name"] == cpu_reference), None) or \
        next((leg for leg in ffn if leg["device"] == "cpu"), None)
    ref_j = seg(reference)["system_j_per_call"] if reference and seg(reference) else None
    lines = ["% Table 1 (tab:motivation): one FFN layer, 1 row, incremental energy (matched microbenchmark)"]
    for leg in ffn:
        s = seg(leg)
        if s is None:
            continue
        ratio = (ref_j / s["system_j_per_call"]) if ref_j and s["system_j_per_call"] else None
        ratio_cell = "--" if ratio is None else "%.1f$\\times$" % ratio
        lines.append(f"{leg['name']} & {_fmt(s['compute_ms_median'])} & {_fmt(s['bandwidth_gbs'], 1)} & "
                     f"{_fmt(s['device_j_per_call'], 3)} & {_fmt(s['desktop_j_per_call'], 3)} & "
                     f"{_fmt(s['system_j_per_call'], 3)} & " + ratio_cell
                     + " \\\\  % nJ/B " + _fmt(s["device_nj_per_byte"], 3))
    lines.append("% Table 2 (tab:batch): per-call time (ms) at 1 / 2 / 4 rows")
    for leg in ffn:
        cells = [_fmt((seg(leg, r) or {}).get("rpc_ms_median") or (seg(leg, r) or {}).get("compute_ms_median"))
                 for r in (1, 2, 4)]
        lines.append(f"{leg['name']} & " + " & ".join(cells) + " \\\\")
    lines.append("% Table 4 (tab:operators): projections and output head, 1 row, ms and incremental J per call")
    for leg in legs:
        if leg["op"] in ("attn_proj", "output_head") and seg(leg):
            s = seg(leg)
            lines.append(f"{leg['name']} & {_fmt(s['compute_ms_median'])} & {_fmt(s['device_j_per_call'], 3)} "
                         f"& {_fmt(s['system_j_per_call'], 3)} \\\\")
    return "\n".join(lines) + "\n"


def markdown(result: Mapping[str, Any]) -> str:
    lines = ["| leg | op | device | rows | calls | compute ms (p50) | rpc ms (p50) | GB/s | idle W (device) | "
             "device J/call (inc) | device J/call (gross) | desktop J/call (inc) | system J/call (inc) | nJ/B | flags |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for leg in result["legs"]:
        for s in leg["segments"]:
            lines.append("| " + " | ".join([
                leg["name"], leg["op"], leg["device"], str(s["rows"]), str(s["calls"]), _md(s["compute_ms_median"]),
                _md(s["rpc_ms_median"]), _md(s["bandwidth_gbs"], 1), _md(leg["idle_w"].get(leg["device"]), 2),
                _md(s["device_j_per_call"], 4), _md(s["device_gross_j_per_call"], 4), _md(s["desktop_j_per_call"], 4),
                _md(s["system_j_per_call"], 4), _md(s["device_nj_per_byte"], 3), ",".join(s["flags"]) or "-"]) + " |")
    return "\n".join(lines) + "\n"


def _md(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


# ---------------------------------------------------------------- prepare / smoke

def prepare(config: Mapping[str, Any], out_path: Path, run: Callable = subprocess.run) -> dict[str, Any]:
    """Pin the phone files of ``worker`` legs: sha256 of the worker, the shard and every library in the worker's
    library directories, read from the phone (recorded, first use). Evidence legs keep their qualified pins."""
    pinned = json.loads(json.dumps(config))
    for leg in pinned["legs"]:
        if leg["kind"] != "phone" or "worker" not in leg:
            continue
        worker = leg["worker"]
        adb = [worker.get("adb_path", "/usr/bin/adb"), "-P", str(worker.get("adb_port", 5037)), "-s", worker["serial"]]
        listing = run(adb + ["shell", "su -c " + shlex.quote(
            "sha256sum " + " ".join(shlex.quote(p) for p in [worker["worker_path"], worker["shard_path"]])
            + " " + " ".join(shlex.quote(d) + "/*.so" for d in worker["library_directories"]))],
            check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=900).stdout
        pins = {}
        for line in listing.splitlines():
            digest, _, path = line.strip().partition("  ")
            if len(digest) == 64 and path:
                pins[path] = "sha256:" + digest
        if worker["worker_path"] not in pins or worker["shard_path"] not in pins:
            raise SystemExit(leg["name"] + ": worker or shard missing on the phone")
        worker["expected_sha256_by_path"] = pins
    write_json(out_path, pinned)
    return pinned


def smoke_config(config: Mapping[str, Any], leg_name: str) -> dict[str, Any]:
    """A 2-token, 1-row, 2 s-idle copy of one leg (checks launch, HELLO version, EXECUTE and the normal exit)."""
    small = json.loads(json.dumps(config))
    small["legs"] = [leg for leg in small["legs"] if leg["name"] == leg_name]
    if not small["legs"]:
        raise SystemExit("no leg " + leg_name)
    small.update(rows=[1], idle_s=2, segment_seconds=1, cooldown_s=1)
    small["cadence"] = dict(small["cadence"], warmup_tokens=1)
    return small


# ---------------------------------------------------------------- CLI

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan"); p.add_argument("config", type=Path)
    p = sub.add_parser("prepare"); p.add_argument("config", type=Path); p.add_argument("out", type=Path)
    p = sub.add_parser("smoke"); p.add_argument("config", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--leg", required=True)
    p = sub.add_parser("run"); p.add_argument("config", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--only")
    p = sub.add_parser("analyze"); p.add_argument("output", type=Path); p.add_argument("--meter", type=Path)
    p.add_argument("--json", type=Path); p.add_argument("--md", type=Path); p.add_argument("--latex", type=Path)
    p.add_argument("--cpu-reference")
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps(plan(load_config(args.config)), indent=2))
    elif args.command == "prepare":
        prepare(load_config(args.config), args.out)
        print("pinned ->", args.out)
    elif args.command == "smoke":
        results = run(smoke_config(load_config(args.config), args.leg), args.output)
        print(json.dumps(results, indent=2, default=str))
        return 0 if all(r["status"] == "PASS" for r in results) else 1
    elif args.command == "run":
        results = run(load_config(args.config), args.output,
                      only=set(args.only.split(",")) if args.only else None)
        print(json.dumps([{k: r[k] for k in ("name", "status", "error") if k in r} for r in results], indent=2))
        return 0 if all(r["status"] == "PASS" for r in results) else 1
    else:
        result = analyze(args.output, args.meter)
        if args.json:
            write_json(args.json, result)
        if args.md:
            args.md.write_text(markdown(result))
        if args.latex:
            args.latex.write_text(latex(result, args.cpu_reference))
        print(markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
