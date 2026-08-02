#!/usr/bin/env python3

from orchestration_v1 import main


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
