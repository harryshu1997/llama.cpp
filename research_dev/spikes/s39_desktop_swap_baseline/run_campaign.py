#!/usr/bin/env python3
"""Run the frozen CP0-D desktop campaign in separate processes."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent


class CampaignError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) + "\n").encode("ascii")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_directory(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.name != "SHA256SUMS.txt":
            output[str(item.relative_to(path))] = digest_file(item)
    lines = [f"{digest}  {name}\n" for name, digest in output.items()]
    (path / "SHA256SUMS.txt").write_text("".join(lines), encoding="ascii")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=18150)
    args = parser.parse_args()

    if args.output_root.exists():
        raise CampaignError(f"output root exists: {args.output_root}")
    args.output_root.mkdir(parents=True)
    contract = json.loads((HERE / "DESKTOP_BASELINE_CONTRACT.json").read_text())
    run_order = contract["replay"]["run_order"]
    if run_order != [
        "WARM_CACHE", "COLD_NVME", "COLD_NVME",
        "WARM_CACHE", "WARM_CACHE", "COLD_NVME",
    ]:
        raise CampaignError("frozen run order mismatch")

    phases: list[tuple[str, list[str]]] = [
        (
            "qualification-qwen3-14b",
            ["--phase", "qualify", "--model-id", "qwen3-14b-q4_k_m"],
        ),
        (
            "qualification-qwen3-8b",
            ["--phase", "qualify", "--model-id", "qwen3-8b-q8_0"],
        ),
        ("noncoresidency", ["--phase", "noncoresidency"]),
    ]
    for index, regime in enumerate(run_order):
        phases.append((
            f"replay-{index:02d}-{regime.lower()}",
            [
                "--phase", "replay",
                "--regime", regime,
                "--repeat-index", str(index),
            ],
        ))

    records: list[dict[str, Any]] = []
    for phase_index, (name, extra) in enumerate(phases):
        output = args.output_root / name
        command = [
            sys.executable,
            str(HERE / "run_desktop_baseline.py"),
            *extra,
            "--server", str(args.server),
            "--cuda-lib-dir", str(args.cuda_lib_dir),
            "--output", str(output),
            "--gpu-index", str(args.gpu_index),
            "--port", str(args.port),
        ]
        stdout_path = args.output_root / f"{phase_index:02d}-{name}.stdout"
        stderr_path = args.output_root / f"{phase_index:02d}-{name}.stderr"
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr)
        ended = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record = {
            "command": command,
            "ended_utc": ended,
            "name": name,
            "output": str(output),
            "returncode": result.returncode,
            "schema": "s39-cp0d-campaign-phase-v1",
            "started_utc": started,
        }
        records.append(record)
        (args.output_root / "campaign_progress.json").write_bytes(canonical({
            "contract_sha256": digest_file(HERE / "DESKTOP_BASELINE_CONTRACT.json"),
            "phases": records,
            "schema": "s39-cp0d-campaign-progress-v1",
        }))
        if output.is_dir():
            freeze_directory(output)
        if result.returncode != 0:
            freeze_directory(args.output_root)
            raise CampaignError(f"phase {name} failed with rc={result.returncode}")

    manifest = {
        "contract_sha256": digest_file(HERE / "DESKTOP_BASELINE_CONTRACT.json"),
        "input_manifest_sha256": digest_file(HERE / "INPUT_MANIFEST.json"),
        "phases": records,
        "schema": "s39-cp0d-campaign-v1",
        "status": "RAW_CAMPAIGN_PASS_ANALYSIS_PENDING",
    }
    (args.output_root / "CAMPAIGN.json").write_bytes(canonical(manifest))
    freeze_directory(args.output_root)
    print(args.output_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignError as exc:
        print(f"CP0D_CAMPAIGN_ERROR: {exc}")
        raise SystemExit(2)
