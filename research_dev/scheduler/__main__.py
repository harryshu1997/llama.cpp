"""Command-line entry point for scheduler support tools."""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {"matmul", "plan"}:
        print(
            "usage: python3 -m research_dev.scheduler {matmul|plan} ...",
            file=sys.stderr,
        )
        return 2
    command = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    if command == "matmul":
        from ._internal.matmul import main as command_main
    else:
        from .plan_cli import main as command_main
    return command_main()


if __name__ == "__main__":
    raise SystemExit(main())
