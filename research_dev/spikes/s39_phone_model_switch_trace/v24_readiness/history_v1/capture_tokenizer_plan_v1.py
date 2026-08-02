#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import history_common_v1 as common


def build_plan(
    executable_path: Path,
    model_path: Path,
    component_id: str,
    locks: common.Locks = common.PRODUCTION_LOCKS,
) -> dict:
    common.text(component_id, "component_id", 256)
    common.require(
        all(
            character.isalnum() or character in "._-"
            for character in component_id
        ),
        "E_COMPONENT_ID",
    )
    executable = common.snapshot_file(executable_path)
    model = common.snapshot_file(model_path)
    common.require(model["sha256"] == locks.model_sha256, "E_MODEL_SHA256")
    common.require(model["bytes"] == locks.model_bytes, "E_MODEL_BYTES")
    mode = os.stat(executable_path, follow_symlinks=False).st_mode
    common.require(mode & 0o111 != 0, "E_EXECUTABLE_MODE")
    model.update({"model_id": common.MODEL_ID, "vocab_size": locks.vocab_size})
    return {
        "command_template": [
            str(executable_path),
            "-m",
            str(model_path),
            "--ids",
            "-f",
            "{PROMPT_FILE}",
            "--log-disable",
        ],
        "component_id": component_id,
        "cwd": str(executable_path.parent),
        "environment": {
            "LC_ALL": "C",
            "LD_LIBRARY_PATH": str(executable_path.parent),
        },
        "executable": executable,
        "model": model,
        "protocol": {
            "add_bos": "MODEL_DEFAULT",
            "escape": True,
            "output_format": "BRACKETED_DECIMAL_IDS",
            "parse_special": True,
            "prompt_file_placeholder": "{PROMPT_FILE}",
        },
        "schema": common.PLAN_SCHEMA,
        "timeout_seconds": 300,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--component-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        executable = args.executable.resolve(strict=True)
        model = args.model.resolve(strict=True)
        output = args.output.absolute()
        common.write_exclusive(
            output,
            build_plan(executable, model, args.component_id),
        )
    except (OSError, common.HistoryError) as error:
        print(f"B8_TOKENIZER_PLAN_REFUSED: {error}", file=sys.stderr)
        return 2
    print(f"B8_TOKENIZER_PLAN_WRITTEN: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
