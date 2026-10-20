#!/usr/bin/env python3
"""Create or verify the immutable reference artifact checksum inventory."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def files(root, output):
    return [path for path in sorted(root.rglob("*"))
            if path.is_file() and path != output
            and "__pycache__" not in path.parts and path.suffix != ".pyc"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root / "ARTIFACT_SHA256.json"
    paths = files(root, output)
    if any(path.is_symlink() for path in paths):
        raise ValueError("artifact symlinks require explicit resolution")
    rows = [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
             "sha256": digest(path)} for path in paths]
    payload = {"schema": "cuda-reference-artifact-inventory-v1", "files": rows,
               "file_count": len(rows), "total_bytes": sum(row["bytes"] for row in rows),
               "excluded": ["ARTIFACT_SHA256.json", "generated __pycache__ and .pyc files"]}
    if args.verify:
        if json.loads(output.read_text()) != payload:
            raise ValueError("artifact inventory differs: do not overwrite or re-freeze evidence")
    else:
        with output.open("x") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
    print(json.dumps({"verified": args.verify, "files": len(rows),
                      "bytes": payload["total_bytes"], "inventory_sha256": digest(output)}))


if __name__ == "__main__":
    main()
