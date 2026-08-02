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
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--tokenizer-plan", type=Path, required=True)
    args = parser.parse_args()
    try:
        history, _ = common.read_small_canonical(args.history.resolve(strict=True))
        common.validate_history(
            history,
            args.corpus.resolve(strict=True),
            args.candidate.resolve(strict=True),
            args.tokenizer_plan.resolve(strict=True),
        )
    except (OSError, common.HistoryError) as error:
        print(f"B8_HISTORY_VALIDATE_REFUSED: {error}", file=sys.stderr)
        return 2
    print("B8_HISTORY_VALIDATE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
