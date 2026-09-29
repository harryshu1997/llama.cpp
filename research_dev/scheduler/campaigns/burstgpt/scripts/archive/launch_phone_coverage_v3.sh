#!/usr/bin/env bash
set -euo pipefail

root=/home/zhihao/s42-phone-coverage-short-gate-20260831-v3
previous=/home/zhihao/s42-phone-coverage-short-gate-20260831-v2
repo=/home/zhihao/llama.cpp-release

test ! -e "$root"
mkdir -p "$root/inputs"
cp "$previous"/inputs/*.json "$root/inputs/"

python3 - "$repo" "$root/SOURCE_MANIFEST.json" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

source_root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
rows = []
for path in sorted((source_root / "research_dev/scheduler").rglob("*")):
    if path.is_file() and path.suffix in {".py", ".sh", ".zsh"}:
        rows.append({
            "path": str(path.relative_to(source_root)),
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        })

def git(*arguments):
    return subprocess.run(
        ("git", *arguments),
        cwd=source_root,
        check=True,
        capture_output=True,
        encoding="ascii",
        text=True,
    ).stdout.strip()

body = {
    "branch": git("branch", "--show-current"),
    "files": rows,
    "head": git("rev-parse", "HEAD"),
    "schema": "research-scheduler-source-manifest-v1",
}
encoded = json.dumps(
    body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
).encode("ascii")
body["manifest_sha256"] = (
    "sha256:" + hashlib.sha256(encoded).hexdigest()
)
output.write_text(
    json.dumps(
        body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ) + "\n",
    encoding="ascii",
)
PY

old_command=$(<"$previous/RUN_COMMAND.txt")
command=${old_command//$previous/$root}
command=${command//s42-phone-coverage-short-gate-v2/s42-phone-coverage-short-gate-v3}
printf '%s\n' "$command" > "$root/RUN_COMMAND.txt"
sha256sum "$root"/inputs/*.json "$root/SOURCE_MANIFEST.json" \
    > "$root/INPUT_SHA256SUMS.txt"
sha256sum "$previous/RUN_COMMAND.txt" > "$root/PARENT_COMMAND_SHA256.txt"

exec bash -lc "$command"
