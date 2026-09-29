#!/usr/bin/env bash
set -euo pipefail

root=/home/zhihao/s42-session-granular-mixed-gate-20260831-v3
previous=/home/zhihao/s42-session-granular-mixed-gate-20260831-v2
old_repo=/home/zhihao/s42-session-granular-deploy-20260831-v2/llama.cpp-release
repo=/home/zhihao/s42-session-granular-deploy-20260831-v3/llama.cpp-release

test ! -e "$root"
test -d "$previous/inputs"
test -f "$previous/RUN_COMMAND.txt"
mkdir -p "$root"
cp -a "$previous/inputs" "$root/inputs"
cp "$previous/CATALOG_TRANSPORT_REBIND.json" \
    "$root/CATALOG_TRANSPORT_REBIND.json"

python3 - "$repo" "$root/SOURCE_MANIFEST.json" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
rows = []
for path in sorted((root / "research_dev" / "scheduler").rglob("*")):
    if path.is_file() and path.suffix in {".py", ".sh", ".zsh"}:
        rows.append({
            "path": str(path.relative_to(root)),
            "sha256": "sha256:"
                + hashlib.sha256(path.read_bytes()).hexdigest(),
        })

def git(*arguments):
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
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

mapfile -d '' command < <(
    python3 - \
        "$previous/RUN_COMMAND.txt" \
        "$previous" \
        "$root" \
        "$old_repo" \
        "$repo" <<'PY'
import shlex
import sys

path, old_root, new_root, old_repo, new_repo = sys.argv[1:]
arguments = shlex.split(open(path, encoding="ascii").read())
replacements = (
    (old_root, new_root),
    (old_repo, new_repo),
    (
        "/data/local/tmp/s42-session-granular-mixed-gate-v2",
        "/data/local/tmp/s42-session-granular-mixed-gate-v3",
    ),
)
for argument in arguments:
    for old, new in replacements:
        argument = argument.replace(old, new)
    sys.stdout.buffer.write(argument.encode("ascii") + b"\0")
PY
)

printf '%q ' "${command[@]}" > "$root/RUN_COMMAND.txt"
printf '\n' >> "$root/RUN_COMMAND.txt"
sha256sum "$root"/inputs/*.json "$root/SOURCE_MANIFEST.json" \
    > "$root/LAUNCH_INPUT_SHA256SUMS.txt"
exec "${command[@]}"
