#!/usr/bin/env python3
"""Complete frozen CP0-D repetitions after a fail-closed replay result."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from run_campaign import canonical, digest_file, freeze_directory


HERE = Path(__file__).resolve().parent


class RepeatError(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=18150)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if not campaign.is_dir():
        raise RepeatError("campaign directory is missing")
    contract = json.loads((HERE / "DESKTOP_BASELINE_CONTRACT.json").read_text())
    run_order = contract["replay"]["run_order"]
    if run_order != [
        "WARM_CACHE", "COLD_NVME", "COLD_NVME",
        "WARM_CACHE", "WARM_CACHE", "COLD_NVME",
    ]:
        raise RepeatError("frozen run order mismatch")
    first_failure = campaign / "replay-01-cold_nvme"
    if not first_failure.is_dir() or (first_failure / "replay.json").exists():
        raise RepeatError("expected frozen cold failure is absent")

    records: list[dict[str, Any]] = []
    for index in range(2, len(run_order)):
        regime = run_order[index]
        name = f"replay-{index:02d}-{regime.lower()}"
        output = campaign / name
        if output.exists():
            raise RepeatError(f"refusing to overwrite {output}")
        command = [
            sys.executable,
            str(HERE / "run_desktop_baseline.py"),
            "--phase", "replay",
            "--regime", regime,
            "--repeat-index", str(index),
            "--server", str(args.server),
            "--cuda-lib-dir", str(args.cuda_lib_dir),
            "--output", str(output),
            "--gpu-index", str(args.gpu_index),
            "--port", str(args.port),
        ]
        stdout_path = campaign / f"diagnostic-{index:02d}-{name}.stdout"
        stderr_path = campaign / f"diagnostic-{index:02d}-{name}.stderr"
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr)
        ended = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if output.is_dir():
            freeze_directory(output)
        records.append({
            "command": command,
            "ended_utc": ended,
            "name": name,
            "returncode": result.returncode,
            "schema": "s39-cp0d-post-failure-repetition-v1",
            "started_utc": started,
        })
        (campaign / "POST_FAILURE_PROGRESS.json").write_bytes(canonical({
            "records": records,
            "schema": "s39-cp0d-post-failure-progress-v1",
        }))

    expected = {
        2: 2,
        3: 0,
        4: 0,
        5: 2,
    }
    observed = {
        int(record["name"].split("-")[1]): record["returncode"]
        for record in records
    }
    report = {
        "contract_sha256": digest_file(HERE / "DESKTOP_BASELINE_CONTRACT.json"),
        "expected_returncodes": {str(key): value for key, value in expected.items()},
        "observed_returncodes": {str(key): value for key, value in observed.items()},
        "records": records,
        "schema": "s39-cp0d-post-failure-repetitions-v1",
        "status": (
            "REPETITIONS_COMPLETE_WARM_PASS_COLD_FAIL"
            if observed == expected else "REPETITIONS_COMPLETE_UNEXPECTED_RESULT"
        ),
    }
    (campaign / "POST_FAILURE_REPETITIONS.json").write_bytes(canonical(report))
    freeze_directory(campaign)
    print(report["status"])
    return 0 if observed == expected else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RepeatError as exc:
        print(f"CP0D_REPEAT_ERROR: {exc}")
        raise SystemExit(2)
