#!/usr/bin/env python3
"""Add the resident Llama 1B executor to the OP15 F16 slice contract."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
S42_ROOT = HERE.parents[1]
BASE = (
    S42_ROOT
    / "multi_session_phone_v1/results"
    / "OP15_THREE_SESSION_RESIDENCY_PLAN_V1.json"
)
MANIFEST = HERE / "TRACE_MANIFEST.json"
OUTPUT = HERE / "OP15_COMBINED_RESIDENCY_PLAN_V1.json"
MINIMUM_AVAILABLE_BYTES = 768 * 1024**2
LLAMA1 = "llama-3.2-1b-instruct-q4_0"


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def main() -> int:
    plan = json.loads(BASE.read_text(encoding="ascii"))
    manifest = json.loads(MANIFEST.read_text(encoding="ascii"))
    llama = manifest["model_inventory"][LLAMA1]
    if (
        plan.get("plan_id") != "op15-gemma-qwen-three-session-v1"
        or plan.get("minimum_available_bytes") != 2 * 1024**3
        or llama.get("artifact_bytes") != 770_928_288
        or llama.get("artifact_sha256")
        != "4b90b1d7ae7324676194755a6dfce11cb6e457982c4c01a1db2857be1ed064ad"
    ):
        raise ValueError("combined residency input identity mismatch")
    plan["combined_task_residency"] = {
        "artifact_bytes": llama["artifact_bytes"],
        "artifact_sha256": llama["artifact_sha256"],
        "backend": "GPUOpenCL",
        "evidence_ids": [
            "llama1b-op15-whole-task-abba-v1",
            "fp16-plus-llama1b-combined-memory-probe-v1",
        ],
        "model_id": LLAMA1,
        "resource_ids": ["op15-adreno", "op15-ncm", "desktop-usb-root"],
        "resident_before_paid_start": True,
    }
    plan["minimum_available_bytes"] = MINIMUM_AVAILABLE_BYTES
    plan["plan_id"] = "op15-gemma-qwen-three-session-plus-llama1b-v1"
    resident_bytes = sum(
        item["resident_bytes"]
        for session in plan["sessions"]
        for item in session["slices"]
    ) + llama["artifact_bytes"]
    if resident_bytes + MINIMUM_AVAILABLE_BYTES > plan["memory_capacity_bytes"]:
        raise ValueError("combined residency does not fit physical capacity")
    content = canonical(plan)
    if OUTPUT.exists():
        if OUTPUT.read_bytes() != content:
            raise ValueError("existing combined residency plan differs")
    else:
        OUTPUT.write_bytes(content)
    print(json.dumps({
        "minimum_available_bytes": MINIMUM_AVAILABLE_BYTES,
        "output": str(OUTPUT),
        "resident_artifact_bytes": resident_bytes,
        "status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
