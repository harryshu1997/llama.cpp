#!/usr/bin/env python3
"""Validate one physical protected/filler phone-arbiter probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


SCHEMA = "research-scheduler-phone-arbiter-probe-v1"


class ProbeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def markers(path: Path, prefix: str) -> list[dict[str, Any]]:
    rows = [
        line.partition(prefix)[2]
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if prefix in line
    ]
    values = [json.loads(row) for row in rows]
    require(
        bool(values) and all(type(value) is dict for value in values),
        f"{prefix.strip()} record is invalid",
    )
    return values


def marker(path: Path, prefix: str) -> dict[str, Any]:
    values = markers(path, prefix)
    require(len(values) == 1, f"{prefix.strip()} record is not unique")
    return values[0]


def token_list(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tokens must be integers") from exc
    if (
        not values
        or len(values) != len(set(values))
        or any(value < 1 or value > 16 for value in values)
    ):
        raise argparse.ArgumentTypeError("tokens must be unique values in 1..16")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def positive(value: object, name: str) -> float:
    require(
        type(value) in {int, float}
        and math.isfinite(value)
        and value > 0,
        f"invalid {name}",
    )
    return float(value)


def resident_receipt(
    session_log: Path,
    workers_log: Path,
    expected_vmem_mib: int,
    minimum_available_kib: int,
) -> dict[str, object]:
    session_rows = [
        line
        for line in session_log.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if line.startswith(
            "[resident-session] HTP0+HTP1+HTP2 WARM mem_available_kib="
        )
    ]
    require(len(session_rows) == 1, "resident memory receipt is not unique")
    try:
        available_kib = int(session_rows[0].rsplit("=", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ProbeError("resident memory receipt is invalid") from exc
    vmem_bytes = [
        int(value)
        for value in re.findall(
            r"op batching:.* vmem ([0-9]+)$",
            workers_log.read_text(
                encoding="utf-8", errors="replace"
            ),
            flags=re.MULTILINE,
        )
    ]
    require(
        available_kib >= minimum_available_kib
        and len(vmem_bytes) == 3
        and set(vmem_bytes) == {expected_vmem_mib * 1024 * 1024},
        "resident memory geometry",
    )
    return {
        "available_kib": available_kib,
        "minimum_available_kib": minimum_available_kib,
        "session_count": len(vmem_bytes),
        "vmem_mib_per_session": expected_vmem_mib,
    }


def analyze(args: argparse.Namespace) -> dict[str, object]:
    probe = marker(args.probe_log, "PHONEARBITERPROBE ")
    bridge = marker(args.bridge_log, "PHONEARBITER ")
    shape_rows = markers(args.bridge_log, "PHONEARBITERSHAPE ")
    router = marker(args.router_log, "RESIDENTROUTER ")
    residency = resident_receipt(
        args.session_log,
        args.workers_log,
        args.expected_vmem_mib,
        args.minimum_available_kib,
    )
    sample_rows = probe.get("samples")
    expected_count = len(args.expected_tokens)

    require(
        probe.get("status") == "PASS"
        and probe.get("protected_groups") == expected_count + 1
        and probe.get("protected_group_calls")
        == (expected_count + 1) * 12
        and probe.get("filler_calls") == expected_count
        and probe.get("configured_gap_us") == args.expected_gap_us
        and type(sample_rows) is list
        and len(sample_rows) == expected_count
        and positive(
            probe.get("validation_group_rpc_us"), "validation group RPC"
        ),
        "probe receipt",
    )
    for expected_tokens, sample in zip(args.expected_tokens, sample_rows):
        require(
            type(sample) is dict
            and sample.get("tokens") == expected_tokens
            and positive(sample.get("observed_gap_us"), "observed gap")
            >= args.expected_gap_us
            and positive(
                sample.get("protected_group_rpc_us"), "protected group RPC"
            )
            and positive(
                sample.get("filler_client_rpc_us"), "filler client RPC"
            ),
            "probe shape receipt",
        )
    require(
        bridge.get("status") == "MECHANICS_ONLY"
        and bridge.get("energy_claim_eligible") is False
        and bridge.get("protected_calls") == (expected_count + 1) * 12
        and bridge.get("filler_calls") == expected_count
        and bridge.get("filler_admitted") == expected_count
        and bridge.get("filler_before_protected_done") == expected_count
        and bridge.get("filler_after_protected_done") == 0
        and bridge.get("protected_group_starts") == expected_count + 1
        and bridge.get("protected_group_ends") == expected_count + 1
        and bridge.get("idle_samples") == expected_count
        and bridge.get("idle_lower_us") == args.expected_idle_lower_us
        and bridge.get("filler_upper_us") == args.expected_filler_upper_us
        and bridge.get("guard_us") == args.expected_guard_us
        and bridge.get("observed_idle_min_us")
        >= args.expected_idle_lower_us
        and bridge.get("filler_sandwich_max_us")
        <= args.expected_filler_upper_us
        and bridge.get("reset_recoveries") == 0
        and bridge.get("filler_upper_violations") == 0
        and bridge.get("guard_violations") == 0
        and bridge.get("idle_lower_violations") == 0
        and bridge.get("protected_pending_after_filler") == 0,
        "bridge mechanics receipt",
    )
    observed_shapes: dict[int, dict[str, Any]] = {}
    for shape in shape_rows:
        tokens = shape.get("tokens")
        require(
            shape.get("status") == "PASS"
            and shape.get("energy_claim_eligible") is False
            and type(tokens) is int
            and tokens not in observed_shapes
            and shape.get("calls") == 1
            and positive(shape.get("sandwich_p50_us"), "shape p50")
            and positive(shape.get("sandwich_max_us"), "shape maximum")
            <= args.expected_filler_upper_us,
            "bridge shape receipt",
        )
        observed_shapes[tokens] = shape
    require(
        tuple(sorted(observed_shapes)) == tuple(sorted(args.expected_tokens)),
        "bridge shape coverage",
    )
    require(
        router.get("status") == "ok"
        and router.get("terminate_requested") is True
        and type(router.get("sessions")) is int
        and router["sessions"] >= 4
        and router.get("requests") == (expected_count + 1) * 12 + expected_count,
        "terminal router receipt",
    )

    rows = [{
        "bridge": observed_shapes[tokens],
        "probe": sample,
        "tokens": tokens,
    } for tokens, sample in zip(args.expected_tokens, sample_rows)]

    result: dict[str, object] = {
        "artifacts": {
            "bridge_log_sha256": sha256(args.bridge_log),
            "probe_log_sha256": sha256(args.probe_log),
            "router_log_sha256": sha256(args.router_log),
            "session_log_sha256": sha256(args.session_log),
            "workers_log_sha256": sha256(args.workers_log),
        },
        "bridge": bridge,
        "energy_claim_eligible": False,
        "probe": probe,
        "residency": residency,
        "rows": rows,
        "router": router,
        "schema": "research-scheduler-phone-arbiter-probe-sweep-v1",
        "status": "PASS",
    }
    result["record_sha256"] = hashlib.sha256(
        (
            json.dumps(
                result,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    ).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-log", type=Path, required=True)
    parser.add_argument("--bridge-log", type=Path, required=True)
    parser.add_argument("--router-log", type=Path, required=True)
    parser.add_argument("--session-log", type=Path, required=True)
    parser.add_argument("--workers-log", type=Path, required=True)
    parser.add_argument("--expected-tokens", type=token_list, required=True)
    parser.add_argument("--expected-gap-us", type=int, required=True)
    parser.add_argument("--expected-idle-lower-us", type=int, required=True)
    parser.add_argument("--expected-filler-upper-us", type=int, required=True)
    parser.add_argument("--expected-guard-us", type=int, required=True)
    parser.add_argument("--expected-vmem-mib", type=int, required=True)
    parser.add_argument("--minimum-available-kib", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(
        args.expected_gap_us,
        args.expected_idle_lower_us,
        args.expected_filler_upper_us,
        args.expected_vmem_mib,
        args.minimum_available_kib,
    ) <= 0 or args.expected_guard_us < 0:
        parser.error("invalid timing bound")
    if (
        args.expected_filler_upper_us + args.expected_guard_us
        > args.expected_idle_lower_us
    ):
        parser.error("filler and guard do not fit the idle bound")
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an unused absolute path")
    try:
        result = analyze(args)
    except (ProbeError, OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"phone arbiter probe analysis failed: {exc}\n")
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "filler_sandwich_max_us": result["bridge"]["filler_sandwich_max_us"],
        "filler_tokens": list(args.expected_tokens),
        "status": result["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
