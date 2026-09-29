#!/usr/bin/env python3
"""Evaluate state- and shape-sensitive routes for the Llama 1B overlay."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable


HERE = Path(__file__).resolve().parent
S42_ROOT = HERE.parent
REPO_ROOT = HERE.parents[3]
sys.path[:0] = [str(REPO_ROOT), str(S42_ROOT), str(HERE)]

from research_dev.scheduler import (  # noqa: E402
    ProfileBundle,
    Request,
    UnifiedScheduler,
    decision_to_json,
)
from build_small_model_overlay import (  # noqa: E402
    MANIFEST,
    OVERLAY_MODEL,
    OVERLAY_STREAM,
    TRACE,
)
from verify_small_model_overlay import validate as validate_trace  # noqa: E402


PROFILE_ROOT = (
    S42_ROOT / "whole_task_phone_v1/results/4060ti_op15_20260807"
)
ISOLATED_PROFILE = PROFILE_ROOT / "SCHEDULER_PROFILE_ISOLATED.json"
TAIL_REUSED_PROFILE = PROFILE_ROOT / "SCHEDULER_PROFILE_TAIL_REUSED.json"
DYNAMIC_RESIDENCY = (
    S42_ROOT
    / "dynamic_residency_v1/results/DYNAMIC_RESIDENCY_SHADOW_V1.json"
)
OUTPUT = HERE / "PLACEMENT_EXPECTATIONS.json"

WORKLOAD_ID = "llama-1b-resident-task"
GPU_CAPACITY_BYTES = 17_175_674_880
GPU_CAPTURED_USED_BYTES = 816_840_704
GPU_RESERVE_BYTES = 536_870_912
QUALIFIED_LARGE_PLACEMENT_BYTES = 15_168_700_416

EXPECTED_ROUTES = {
    "cuda_epoch_open_all_resident": {
        "phone-adreno": list(range(10)),
    },
    "cuda_epoch_reused_all_resident": {
        "desktop-cuda": [0, 1, 2, 3, 4, 7, 8, 9],
        "phone-adreno": [5, 6],
    },
    "cuda_busy_phone_resident": {
        "phone-adreno": list(range(10)),
    },
    "current_saturated_no_transition": {
        "desktop-cpu": list(range(10)),
    },
}


class EvaluationError(ValueError):
    pass


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


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if type(value) is not dict:
        raise EvaluationError(f"object expected: {path}")
    return value


def overlay_rows() -> list[dict[str, object]]:
    validate_trace(TRACE, MANIFEST)
    rows = [
        json.loads(line)
        for line in TRACE.read_text(encoding="ascii").splitlines()
    ]
    result = [
        row for row in rows if row.get("trace_stream_id") == OVERLAY_STREAM
    ]
    if (
        len(result) != 10
        or [row.get("stream_request_index") for row in result]
        != list(range(10))
    ):
        raise EvaluationError("small-model overlay identity mismatch")
    return result


def request(row: dict[str, object], case_id: str) -> Request:
    return Request(
        request_id=f"{case_id}-{row['stream_request_index']}",
        workload_id=WORKLOAD_ID,
        arrival_us=0,
        deadline_us=int(row["slo_us"]),
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        quality_requirement="bounded_numeric",
    )


def case_decisions(
    case_id: str,
    profile_path: Path,
    configure: Callable[[UnifiedScheduler], None],
) -> dict[str, object]:
    bundle = ProfileBundle.from_json(load_object(profile_path))
    decisions = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in overlay_rows():
        scheduler = UnifiedScheduler((bundle,), "enforce")
        configure(scheduler)
        decision = scheduler.schedule(request(row, case_id))
        index = int(row["stream_request_index"])
        grouped[decision.route_id].append(index)
        decisions.append({
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "reason": decision_to_json(decision)["reason"],
            "route_id": decision.route_id,
            "stream_request_index": index,
        })
    routes = {key: value for key, value in sorted(grouped.items())}
    if routes != EXPECTED_ROUTES[case_id]:
        raise EvaluationError(
            f"placement expectation mismatch for {case_id}: {routes}"
        )
    return {
        "decisions": decisions,
        "profile_path": str(profile_path.relative_to(REPO_ROOT)),
        "profile_sha256": digest_file(profile_path),
        "routes": routes,
    }


def no_change(_: UnifiedScheduler) -> None:
    return


def cuda_busy(scheduler: UnifiedScheduler) -> None:
    scheduler.reserve_external_resource(
        "cuda0", "large-model-gpu-work", 0, 30_000_000
    )


def no_resident_accelerators(scheduler: UnifiedScheduler) -> None:
    scheduler.set_resource_ready("cuda0", False, 0)
    scheduler.set_resource_ready("op15-adreno", False, 0)


def evaluate() -> dict[str, object]:
    manifest = load_object(MANIFEST)
    model = manifest["model_inventory"][OVERLAY_MODEL]
    model_bytes = int(model["artifact_bytes"])
    dynamic = load_object(DYNAMIC_RESIDENCY)
    phone = dynamic["phone_memory"]
    phone_stageable = int(phone["stageable_bytes_beyond_reserve"])
    gpu_stageable = (
        GPU_CAPACITY_BYTES
        - GPU_CAPTURED_USED_BYTES
        - GPU_RESERVE_BYTES
        - QUALIFIED_LARGE_PLACEMENT_BYTES
    )
    if gpu_stageable >= model_bytes or phone_stageable >= model_bytes:
        raise EvaluationError("saturated snapshot unexpectedly fits Llama 1B")

    cases = {
        "cuda_epoch_open_all_resident": case_decisions(
            "cuda_epoch_open_all_resident", ISOLATED_PROFILE, no_change
        ),
        "cuda_epoch_reused_all_resident": case_decisions(
            "cuda_epoch_reused_all_resident",
            TAIL_REUSED_PROFILE,
            no_change,
        ),
        "cuda_busy_phone_resident": case_decisions(
            "cuda_busy_phone_resident", TAIL_REUSED_PROFILE, cuda_busy
        ),
        "current_saturated_no_transition": case_decisions(
            "current_saturated_no_transition",
            ISOLATED_PROFILE,
            no_resident_accelerators,
        ),
    }
    return {
        "cases": cases,
        "evidence": {
            "dynamic_residency_path": str(
                DYNAMIC_RESIDENCY.relative_to(REPO_ROOT)
            ),
            "dynamic_residency_sha256": digest_file(DYNAMIC_RESIDENCY),
        },
        "model": {
            "artifact_bytes": model_bytes,
            "artifact_sha256": model["artifact_sha256"],
            "model_id": OVERLAY_MODEL,
        },
        "saturated_snapshot": {
            "gpu": {
                "capacity_bytes": GPU_CAPACITY_BYTES,
                "captured_used_bytes": GPU_CAPTURED_USED_BYTES,
                "large_placement_bytes": QUALIFIED_LARGE_PLACEMENT_BYTES,
                "reserve_bytes": GPU_RESERVE_BYTES,
                "stageable_bytes_beyond_reserve": gpu_stageable,
                "stageable_shortfall_bytes": model_bytes - gpu_stageable,
            },
            "phone": {
                "capacity_bytes": phone["capacity_bytes"],
                "occupied_bytes": phone["occupied_bytes"],
                "reserve_bytes": phone["mandatory_reserve_bytes"],
                "stageable_bytes_beyond_reserve": phone_stageable,
                "stageable_shortfall_bytes": model_bytes - phone_stageable,
            },
            "policy": (
                "without a qualified residency transition, mark both "
                "accelerator routes unavailable and retain desktop CPU"
            ),
        },
        "schema": "s42-small-model-placement-expectations-v1",
        "status": "PASS",
        "trace_sha256": manifest["trace"]["sha256"],
    }


def write_or_compare(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise EvaluationError(f"existing output differs: {path}")
        return
    path.write_bytes(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = evaluate()
    write_or_compare(args.output, canonical(result))
    print(json.dumps({
        "cases": {
            name: case["routes"] for name, case in result["cases"].items()
        },
        "output": str(args.output),
        "status": result["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
