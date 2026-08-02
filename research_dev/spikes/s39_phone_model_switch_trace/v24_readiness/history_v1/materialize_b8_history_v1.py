#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import history_common_v1 as common


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-plan", type=Path, required=True)
    args = parser.parse_args()
    try:
        history = common.build_history(
            args.corpus.resolve(strict=True),
            args.candidate.resolve(strict=True),
            args.tokenizer_plan.resolve(strict=True),
        )
        common.write_exclusive(args.output.absolute(), history)
    except (OSError, common.HistoryError) as error:
        print(f"B8_HISTORY_MATERIALIZE_REFUSED: {error}", file=sys.stderr)
        return 2
    print(f"B8_HISTORY_MATERIALIZE_PASS: {args.output.absolute()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
