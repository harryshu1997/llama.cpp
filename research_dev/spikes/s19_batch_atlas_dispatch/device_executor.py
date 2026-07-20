#!/usr/bin/env python3
"""Real-device executor for the S19 CP2 live run.

Wraps the CP1 atlas harness primitives to run ONE real static cohort per call
(one persistent exchange): launch/reuse the resident phone worker, run the host
pipedriver or CUDA monodriver exchange, and return an ExecResult with the real
worker pid/nonce, route wall time, generated token ids, and placement status.

This is one static cohort per persistent exchange (VARIABLE_COHORT_BATCHING),
not token-boundary continuous batching.
"""
from __future__ import annotations

import time
from pathlib import Path

import atlas_measure as AM
import dispatcher as D

HERE = Path(__file__).resolve().parent


def execute_one(route_name: str, batch: int, out_dir: Path, tag: str) -> dict:
    """Run one real exchange. Returns a device-exchange record."""
    route = AM.ROUTES[route_name]
    raw_prefix = out_dir / "raw" / f"live.{tag}"
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    worker = None
    worker_pid = None
    boot_nonce = None
    t0 = time.time_ns() // 1000
    try:
        if route["kind"] == "phone":
            worker, worker_pid = AM.launch_phone_worker(route, batch, raw_prefix)
        run = AM.run_host_process(route_name, route, batch, warmup=0, measured=1,
                                  raw_prefix=raw_prefix)
        results = run["results"]
        ok = bool(results) and results[0]["result"].get("outcome") == "completed"
        tokens = []
        wall = 0
        if ok:
            res = results[0]["result"]
            toks = res.get("token_ids") or []
            tokens = toks[0] if toks and isinstance(toks[0], list) else toks
            wall = int(res["route_wall_us"])
        placement_ok = all(AM.placement_ok(c) for c in run["host_certs"]) if run["host_certs"] else False
        if route["kind"] == "phone" and worker is not None:
            time.sleep(0.3)
            certs = AM.parse_cert(worker.stderr_lines + worker.stdout_lines, "SESSIONCERT")
            for c in certs:
                worker_pid = str(c.get("worker_pid"))
                boot_nonce = str(c.get("worker_boot_nonce"))
                if not AM.placement_ok(c):
                    placement_ok = False
    finally:
        if route["kind"] == "phone":
            AM.kill_worker(route)
    t1 = time.time_ns() // 1000
    return {
        "route": route_name, "batch": batch, "ok": ok,
        "device": route["device"], "backend": route["backend"],
        "worker_pid": worker_pid, "worker_boot_nonce": boot_nonce,
        "route_wall_us": wall, "wall_window_us": [t0, t1],
        "token_ids": tokens, "placement_ok": placement_ok,
    }


class DeviceExecutor(D.Executor):
    """Dispatcher-facing executor that runs one real exchange per cohort."""

    def __init__(self, out_dir: Path = HERE, tag_prefix: str = "dx") -> None:
        self.out_dir = out_dir
        self.tag_prefix = tag_prefix
        self.n = 0

    def execute(self, route, batch, request_ids, epochs, session_end):
        self.n += 1
        rec = execute_one(route, batch, self.out_dir, f"{self.tag_prefix}{self.n}")
        return D.ExecResult(
            ok=rec["ok"], device=rec["device"], worker_pid=rec["worker_pid"],
            worker_boot_nonce=rec["worker_boot_nonce"], route_wall_us=rec["route_wall_us"],
            token_ids=rec["token_ids"], session_end=session_end,
            placement_ok=rec["placement_ok"],
            error=None if rec["ok"] else "worker_failure")
