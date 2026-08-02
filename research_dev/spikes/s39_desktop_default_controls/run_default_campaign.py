#!/usr/bin/env python3
"""Run the frozen stock-default control campaign."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "s39_desktop_swap_baseline"
sys.path.insert(0, str(BASE))

import run_desktop_baseline as baseline  # noqa: E402
import validate_contract  # noqa: E402


def write_manifest(path: Path) -> None:
    files = sorted(
        item for item in path.rglob("*")
        if item.is_file() and item != path / "SHA256SUMS.txt"
    )
    (path / "SHA256SUMS.txt").write_text(
        "".join(
            f"{baseline.digest_file(item)}  {item.relative_to(path)}\n"
            for item in files
        ),
        encoding="ascii",
    )


def server_processes() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-ax", "llama-server"],
        capture_output=True, text=True,
    )
    if result.returncode == 1:
        return []
    if result.returncode != 0:
        raise baseline.RunError("could not inspect llama-server processes")
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=18250)
    args = parser.parse_args()

    contract_result = validate_contract.validate()
    if args.output_root.exists():
        raise baseline.RunError(f"output exists: {args.output_root}")
    args.output_root.mkdir(parents=True)
    phases: list[tuple[str, list[str]]] = []
    for repeat in range(3):
        phases.append((
            f"swap-warm-{repeat}",
            [
                sys.executable,
                str(BASE / "run_desktop_baseline.py"),
                "--phase", "replay",
                "--regime", "WARM_CACHE",
                "--repeat-index", str(repeat),
                "--serving-profile", "stock_default",
            ],
        ))
    for repeat in range(3):
        phases.append((
            f"swap-cold-{repeat}",
            [
                sys.executable,
                str(BASE / "run_desktop_baseline.py"),
                "--phase", "replay",
                "--regime", "COLD_NVME",
                "--repeat-index", str(repeat),
                "--serving-profile", "stock_default",
            ],
        ))
    for repeat in range(3):
        phases.append((
            f"dual-warm-{repeat}",
            [
                sys.executable,
                str(HERE / "run_dual_default.py"),
                "--repeat-index", str(repeat),
            ],
        ))

    records: list[dict[str, Any]] = []
    for phase_index, (name, prefix) in enumerate(phases):
        if server_processes():
            raise baseline.RunError("foreign llama-server before phase")
        output = args.output_root / name
        command = [
            *prefix,
            "--server", str(args.server),
            "--cuda-lib-dir", str(args.cuda_lib_dir),
            "--output", str(output),
            "--gpu-index", str(args.gpu_index),
            "--port", str(args.port),
        ]
        if name.startswith("swap-"):
            command.extend(["--request-timeout-s", "600"])
        stdout_path = args.output_root / f"{phase_index:02d}-{name}.stdout"
        stderr_path = args.output_root / f"{phase_index:02d}-{name}.stderr"
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr)
        ended = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if output.exists():
            write_manifest(output)
        lingering = server_processes()
        record = {
            "command": command,
            "ended_utc": ended,
            "lingering_servers": lingering,
            "name": name,
            "output": str(output),
            "returncode": result.returncode,
            "schema": "s39-stock-default-campaign-phase-v1",
            "started_utc": started,
        }
        records.append(record)
        (args.output_root / "campaign_progress.json").write_bytes(
            baseline.canonical({
                "contract_sha256": contract_result["contract_sha256"],
                "phases": records,
                "schema": "s39-stock-default-campaign-v1",
            })
        )
        if lingering:
            raise baseline.RunError(f"lingering server after {name}: {lingering}")
    write_manifest(args.output_root)
    print(json.dumps({
        "phase_count": len(records),
        "returncodes": [row["returncode"] for row in records],
        "status": "STOCK_DEFAULT_CAMPAIGN_ACQUIRED",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (baseline.RunError, OSError, ValueError) as exc:
        print(f"STOCK_DEFAULT_CAMPAIGN_ERROR: {exc}")
        raise SystemExit(2)
