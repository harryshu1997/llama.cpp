#!/usr/bin/env python3
"""Reproduce the frozen small-workload comparison without running inference."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=REPO,
                        help="Root of extracted reference-source-v7.tar.gz")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    from research_dev.scheduler.campaigns.burstgpt.compare_ab import (
        canonical, load_object, summarize_frozen_reference_set,
    )
    root = Path(__file__).resolve().parent
    specification = root / "provenance/reference-inputs/FROZEN_EXPERIMENT_SPECIFICATION-v6.json"
    spec = load_object(specification)
    results = {name: root / "references-source-v7" / name / "run/RESULT.json" for name in spec["contracts"]}
    summary = summarize_frozen_reference_set(spec, results)
    summary["specification_sha256"] = "sha256:" + hashlib.sha256(specification.read_bytes()).hexdigest()
    analysis = args.source_root / "research_dev/scheduler/campaigns/burstgpt/compare_ab.py"
    summary["postprocessing_source_sha256"] = "sha256:" + hashlib.sha256(analysis.read_bytes()).hexdigest()
    summary["specification_errata"] = "provenance/SPECIFICATION_ERRATA.md"
    with args.output.open("xb") as output:
        output.write(canonical(summary))
    print(json.dumps({"comparison_valid": summary["comparison_valid"],
                      "best_tested_fixed_layout_nominal": summary["best_tested_fixed_layout_nominal"]}))


if __name__ == "__main__":
    main()
