#!/usr/bin/env python3

import argparse
from pathlib import Path

import cp0_r1_evidence_v2 as v2
import cp0_r1_evidence_v22 as v22
import cp0_r1_phase_preflight_v21 as v21_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture exact CP0-R1 V2.2 readiness before acquisition"
    )
    parser.add_argument("--contract", type=Path, default=v22.DEFAULT_CONTRACT)
    parser.add_argument("--candidate", type=Path, default=v22.DEFAULT_CANDIDATE)
    parser.add_argument("--phase", choices=("A_ONLY", "B_ONLY", "PAIR"), required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument(
        "--model-lock",
        action="append",
        default=[],
        metavar="SLOT=JSONL",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract, _, candidate, _, _, _ = v22.validate_inputs(
            args.contract,
            args.candidate,
        )
        locks = {}
        for value in args.model_lock:
            slot, separator, path = value.partition("=")
            v2.require(separator == "=" and slot in ("A", "B"), "E_LOCK_ARGUMENT")
            v2.require(slot not in locks, f"E_LOCK_REUSE: {slot}")
            locks[slot] = v21_preflight._read_lock(Path(path))
        v2.require(args.timeout_seconds > 0, "E_TIMEOUT")
        raw = v21_preflight.collect(
            contract,
            candidate,
            args.phase,
            args.phase_id,
            locks,
            args.timeout_seconds,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        v21_preflight._write_exclusive(args.output, raw)
        print(f"{v2.sha256_bytes(raw)}  {args.output}")
        return 0
    except (v2.EvidenceError, OSError, KeyError, ValueError) as exc:
        print(f"CP0_R1_V2_2_PREFLIGHT_REFUSED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
