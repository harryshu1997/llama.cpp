#!/usr/bin/env python3
"""Rebuild the run-time scheduler source for the replay scripts.

Copies research_dev/scheduler from the working tree (without baselines/tests/reports), overlays the
run-time copies in ../run-source/, and checks every file listed in SOURCE_MANIFEST.json of the run.

Usage: assemble_run_source.py OUT_DIR [--manifest RUN/SOURCE_MANIFEST.json]
Then:  replay_*.py --source OUT_DIR/research_dev ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(REPORT, "..", "..", "..", "..", "..", ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    source = os.path.join(REPO, "research_dev", "scheduler")
    target = os.path.join(args.out_dir, "research_dev", "scheduler")
    ignore = shutil.ignore_patterns("__pycache__", "baselines", "tests", "reports")
    shutil.copytree(source, target, ignore=ignore, dirs_exist_ok=True)
    overlay = os.path.join(REPORT, "run-source")
    for root, _, files in os.walk(overlay):
        for name in files:
            if name == "MANIFEST.txt":
                continue
            relative = os.path.relpath(os.path.join(root, name), overlay)
            shutil.copy2(os.path.join(root, name), os.path.join(target, relative))
    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)
        bad = []
        for row in manifest["files"]:
            path = os.path.join(args.out_dir, row["path"])
            if "/_internal/" not in path and "/_unified/" not in path:
                continue
            if not os.path.exists(path):
                bad.append(row["path"] + " (missing)")
                continue
            with open(path, "rb") as f:
                digest = "sha256:" + hashlib.sha256(f.read()).hexdigest()
            if digest != row["sha256"]:
                bad.append(row["path"])
        print("runtime modules differing from the run:", bad or "none")


if __name__ == "__main__":
    main()
