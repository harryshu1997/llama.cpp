#!/usr/bin/env bash
set -euo pipefail

root=/home/zhihao/s42-helper-rebind-ggg-ggq-gate-20260901-v11
previous=/home/zhihao/s42-helper-rebind-ggg-ggq-gate-20260901-v9
old_repo=/home/zhihao/s42-helper-rebind-deploy-20260901-v9/llama.cpp-release
repo=/home/zhihao/s42-helper-rebind-deploy-20260901-v10/llama.cpp-release

test ! -e "$root"
test -d "$previous/inputs"
test -f "$previous/RUN_COMMAND.txt"
test -d "$repo/research_dev/scheduler"
mkdir -p "$root"
cp -a "$previous/inputs" "$root/inputs"

python3 - \
    "$repo" \
    "$root/SOURCE_MANIFEST.json" \
    "$previous/SOURCE_MANIFEST.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

source_root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
parent = json.loads(Path(sys.argv[3]).read_text(encoding="ascii"))
rows = []
for path in sorted((source_root / "research_dev/scheduler").rglob("*")):
    if path.is_file() and path.suffix in {".py", ".sh", ".zsh"}:
        rows.append({
            "path": str(path.relative_to(source_root)),
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        })

body = {
    "branch": parent["branch"],
    "files": rows,
    "head": parent["head"],
    "parent_manifest_sha256": parent["manifest_sha256"],
    "schema": "research-scheduler-source-manifest-v1",
}
encoded = json.dumps(
    body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
).encode("ascii")
body["manifest_sha256"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
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
        "/data/local/tmp/s42-helper-rebind-ggg-ggq-v9",
        "/data/local/tmp/s42-helper-rebind-ggg-ggq-v11",
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
sha256sum \
    "$repo/research_dev/scheduler/scheduler.py" \
    "$repo/research_dev/scheduler/adapters/runtime.py" \
    "$repo/research_dev/scheduler/_internal/model_placement_controller.py" \
    "$repo/research_dev/scheduler/_internal/runtime_plan.py" \
    > "$root/KEY_SOURCE_SHA256SUMS.txt"
sha256sum "$previous/run/RESULT.json" "$previous/RUN_COMMAND.txt" \
    > "$root/PARENT_EVIDENCE_SHA256.txt"

"${command[@]}" \
    > "$root/CAMPAIGN_STDOUT.log" \
    2> "$root/CAMPAIGN_STDERR.log"
