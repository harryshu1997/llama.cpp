#!/usr/bin/env python3
"""Build the isolated S41 desktop-smoke bundle."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
S40_EXECUTORS = (
    HERE.parent / "s40_shared_warm_tier_server" / "executors"
)
if str(S40_EXECUTORS) not in sys.path:
    sys.path.insert(0, str(S40_EXECUTORS))

from executor_bundle import (  # noqa: E402
    ISOLATED_PYTHON_FLAGS,
    SOURCES as S40_SOURCES,
    canonical_bytes,
    digest,
    fsync_directory,
    validate_executor_bundle,
    write_new,
)
import desktop_authority  # noqa: E402


SOURCES = {
    **S40_SOURCES,
    "qualification_authority.py": HERE / "desktop_authority.py",
}
GENERATED_CONTRACT = "experiment/EXPERIMENT_CONTRACT.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def build_executor_bundle(output: Path) -> dict[str, Any]:
    require(output.is_absolute() and not output.exists(), "bundle output path")
    output.mkdir(mode=0o700)
    rows = []
    for name, source in sorted(SOURCES.items()):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name == GENERATED_CONTRACT:
            source = HERE / "desktop_authority.py"
            raw = desktop_authority.CONTRACT_BYTES
            with target.open("xb", buffering=0) as out:
                out.write(raw)
                out.flush()
                os.fsync(out.fileno())
        else:
            require(
                source.is_absolute() and source.is_file(),
                f"source {name}",
            )
            with source.open("rb") as inp, target.open(
                "xb",
                buffering=0,
            ) as out:
                shutil.copyfileobj(inp, out, 1024 * 1024)
                out.flush()
                os.fsync(out.fileno())
        require(source.is_absolute() and source.is_file(), f"source {name}")
        rows.append({
            "bytes": target.stat().st_size,
            "name": name,
            "sha256": digest(target),
            "source_path": str(source),
            "source_sha256": digest(source),
        })
    manifest = {
        "files": rows,
        "python_flags": ISOLATED_PYTHON_FLAGS,
        "schema": "s40-executor-bundle-v2",
    }
    manifest_path = output / "MANIFEST.json"
    write_new(manifest_path, canonical_bytes(manifest))
    fsync_directory(output)
    result = validate_executor_bundle(output, manifest_path)
    return {
        **result,
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "S40_EXECUTOR_BUNDLE": "1",
            "S40_EXECUTOR_BUNDLE_MANIFEST": str(manifest_path),
            "S40_EXECUTOR_BUNDLE_SHA256": digest(manifest_path),
        },
    }


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in ("build", "validate"):
        print(
            "usage: build_desktop_smoke_config.py "
            "build|validate /absolute/bundle",
            file=sys.stderr,
        )
        return 2
    bundle = Path(sys.argv[2])
    result = (
        build_executor_bundle(bundle)
        if sys.argv[1] == "build"
        else validate_executor_bundle(bundle, bundle / "MANIFEST.json")
    )
    print(canonical_bytes(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(f"S41 executor bundle failed: {error}", file=sys.stderr)
        raise SystemExit(2)
