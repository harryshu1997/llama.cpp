#!/usr/bin/env python3
"""Derive one quality-shard arm input directory from an existing arm input directory.

    python3 -m research_dev.scheduler.campaigns.burstgpt.quality.derive_inputs \\
        SOURCE_INPUTS SHARD_DIR TARGET_INPUTS [--campaign-overrides JSON]

SOURCE_INPUTS is a materialized arm directory (campaign.json plus the rig/models/evidence manifests and the
transport identity it references, e.g. inputs-two-phone-s2a). Every *.json file is copied; any string value
naming a path inside SOURCE_INPUTS is re-pointed into TARGET_INPUTS; campaign.json gets the shard's trace
(REQUESTS_SEMANTIC_SOURCE.jsonl, REQUESTS_OVERLAY.jsonl, TRACE_MANIFEST.json, <trace_name>.json) and a
campaign id suffixed with the trace name. Nothing else changes, so the arm keeps its frozen policy.
--campaign-overrides merges top-level campaign keys (the strict max-offload arm); `trace` cannot be overridden.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class DeriveInputsError(RuntimeError):
    pass


def _repoint(value: Any, source: str, target: str) -> Any:
    if type(value) is str and (value == source or value.startswith(source + "/")):
        return target + value[len(source):]
    if type(value) is dict:
        return {name: _repoint(item, source, target) for name, item in value.items()}
    if type(value) is list:
        return [_repoint(item, source, target) for item in value]
    return value


def shard_trace(shard_dir: Path) -> dict[str, Any]:
    if not (shard_dir / "TRACE_MANIFEST.json").is_file():
        raise DeriveInputsError(f"{shard_dir} has no TRACE_MANIFEST.json")
    manifest = json.loads((shard_dir / "TRACE_MANIFEST.json").read_text(encoding="ascii"))
    trace_name = (manifest.get("derivation") or {}).get("trace_name")
    if type(trace_name) is not str or not (shard_dir / f"{trace_name}.json").is_file():
        raise DeriveInputsError(f"{shard_dir} is not a quality shard (no trace_name / replay schedule)")
    return {"arrival_scale": None,
            "large_requests_path": str(shard_dir / "REQUESTS_SEMANTIC_SOURCE.jsonl"),
            "overlay_requests_path": str(shard_dir / "REQUESTS_OVERLAY.jsonl"),
            "replay_schedule_path": str(shard_dir / f"{trace_name}.json"),
            "request_indices": [],
            "trace_manifest_path": str(shard_dir / "TRACE_MANIFEST.json"),
            "trace_name": trace_name}


def derive(source: Path, shard_dir: Path, target: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    source, shard_dir, target = source.resolve(), shard_dir.resolve(), target.resolve()
    if not (source / "campaign.json").is_file():
        raise DeriveInputsError(f"{source} has no campaign.json")
    if target.exists():
        raise DeriveInputsError(f"{target} exists; derive into a fresh directory")
    if overrides and "trace" in overrides:
        raise DeriveInputsError("the trace comes from the shard, not from overrides")
    trace = shard_trace(shard_dir)
    trace_name = trace.pop("trace_name")
    target.mkdir(parents=True)
    written = {}
    for path in sorted(source.glob("*.json")):
        value = _repoint(json.loads(path.read_text(encoding="utf-8")), str(source), str(target))
        if path.name == "campaign.json":
            value["trace"] = trace
            value["campaign_id"] = f"{value['campaign_id']}-{trace_name}"
            value.update(overrides or {})
        (target / path.name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written[path.name] = str(target / path.name)
    return {"source": str(source), "target": str(target), "trace_name": trace_name, "files": written}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("shard", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--campaign-overrides", type=json.loads, default=None)
    args = parser.parse_args()
    print(json.dumps(derive(args.source, args.shard, args.target, args.campaign_overrides), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
