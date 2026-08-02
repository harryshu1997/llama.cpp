#!/usr/bin/python3 -I
"""Entry point for the CP0-R1 V2.3 immutable artifact snapshot."""

import importlib.util
from pathlib import Path
import sys


sys.dont_write_bytecode = True


def load_support():
    try:
        index = sys.argv.index("--support")
        path = Path(sys.argv[index + 1])
    except (ValueError, IndexError) as error:
        raise SystemExit("V23_ARTIFACT_SNAPSHOT_REFUSED: missing --support") from error
    del sys.argv[index:index + 2]
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise SystemExit("V23_ARTIFACT_SNAPSHOT_REFUSED: invalid --support")
    spec = importlib.util.spec_from_file_location("s39_v23_driver_common_v1", path)
    if spec is None or spec.loader is None:
        raise SystemExit("V23_ARTIFACT_SNAPSHOT_REFUSED: cannot load --support")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(load_support().artifact_main())
