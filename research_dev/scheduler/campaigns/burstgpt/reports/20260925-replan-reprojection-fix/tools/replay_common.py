"""Shared loader for dispatcher replays of a recorded run (no hardware).

Builds the scheduler with the run's own runner arguments (paths remapped to a local copy),
exposes the recorded snapshots and decision log, and a non-blocking queue probe.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"


def setup(tree: str, run: str = "dev2base", extra_args: tuple[str, ...] = ()):
    tree_root = Path(tree).resolve()
    sys.path.insert(0, str(tree_root))
    run_dir = RUNS / run
    inputs = run_dir / "in"

    from research_dev.scheduler.campaigns.burstgpt import runner
    from research_dev.scheduler import HeterogeneousRuntimeSnapshot, Request
    import research_dev.scheduler.scheduler as schedmod
    from research_dev.scheduler._internal import model_manifest_cache as mmc

    cache_entries = mmc._read_entries(
        inputs / "home__zhihao__.cache__llama.cpp__research-scheduler__gguf-manifests.json"
    )

    def fake_load(model_id, source, cache_path):
        hits = [m for k, m in cache_entries.items() if k[0] == model_id and k[1] == str(source)]
        assert hits, (model_id, source)
        return hits[-1]

    schedmod.load_cached_gguf_manifest = fake_load

    def loc(p):
        q = inputs / p.lstrip("/").replace("/", "__")
        return str(q) if q.exists() else p

    argv = (run_dir / "RUN_COMMAND.txt").read_text().split()[2:]
    mapped = []
    for a in argv:
        if "=" in a and a.split("=", 1)[1].startswith("/"):
            k, v = a.split("=", 1)
            mapped.append((loc(k) if k.startswith("/") else k) + "=" + v)
            continue
        mapped.append(loc(a) if a.startswith("/") else a)
    repo_data = str(tree_root / "research_dev/scheduler/campaigns/burstgpt/data") + "/"
    mapped = [
        repo_data + Path(a).name
        if a.endswith("_MANIFEST.json") and "campaigns/burstgpt/data" in a else a
        for a in mapped
    ]
    mapped = [
        str(run_dir / "UNIFIED_RUNTIME_CATALOG.json")
        if a.endswith("UNIFIED_RUNTIME_CATALOG.json") else a
        for a in mapped
    ]
    mapped.extend(extra_args)
    parser = runner._build_parser()
    args = parser.parse_args(mapped)
    models = runner._load_trace_models(args)
    scheduler, manifests, _ = runner._build_scheduler(args, models)
    aliases, merged, replay_schedule, _ = runner._select_replay(args, models, manifests)
    by_id = {item["row"]["event_id"]: item for item in merged}

    def req(event_id):
        item = by_id[event_id]
        row = item["row"]
        return Request(
            request_id=row["event_id"], workload_id="physical:" + item["model_id"],
            arrival_us=row["arrival_us"], deadline_us=row["arrival_us"] + row["slo_us"],
            input_tokens=row["input_tokens"], output_tokens=row["output_tokens"],
            quality_requirement="semantic",
        ), item["model_id"], item["combined_index"]

    def snap(path):
        return HeterogeneousRuntimeSnapshot.from_json(json.loads(Path(path).read_text()))

    decision_log = json.loads((run_dir / "SCHEDULER_DECISION_LOG.json").read_text())["records"]
    return dict(
        runner=runner, scheduler=scheduler, manifests=manifests, merged=merged,
        by_id=by_id, req=req, snap=snap, run_dir=run_dir, decision_log=decision_log,
        args=args,
    )


class WouldBlock(Exception):
    def __init__(self, timeout):
        super().__init__(timeout)
        self.timeout = timeout


class pinned_clock:
    """Pin time.monotonic_ns to simulated time t (epoch 0) so receipts carry t exactly."""

    def __init__(self, now_us):
        self.now_ns = now_us * 1000

    def __enter__(self):
        self.original = time.monotonic_ns
        time.monotonic_ns = lambda: self.now_ns
        return 0

    def __exit__(self, *exc):
        time.monotonic_ns = self.original
        return False


def probe_ready(queue, request_id, now_us):
    """Evaluate wait_ready once at simulated time now_us without sleeping."""
    condition = queue._condition
    original = condition.wait

    def no_wait(timeout=None):
        raise WouldBlock(timeout)

    condition.wait = no_wait
    try:
        with pinned_clock(now_us) as epoch_ns:
            return queue.wait_ready(request_id, epoch_ns)
    except WouldBlock as exc:
        return exc
    finally:
        condition.wait = original


def mk(cls, d):
    names = {f.name for f in dataclasses.fields(cls)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
    return cls(**kw)
