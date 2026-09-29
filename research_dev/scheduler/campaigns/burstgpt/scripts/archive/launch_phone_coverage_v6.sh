#!/usr/bin/env bash
set -euo pipefail

root=/home/zhihao/s42-phone-coverage-short-gate-20260831-v6
previous=/home/zhihao/s42-phone-coverage-short-gate-20260831-v5
deploy=/home/zhihao/s42-phone-coverage-scoped-deploy-20260831-v1
source_deploy=/home/zhihao/s42-phone-coverage-scoped-deploy-20260831-v2
repo=/home/zhihao/llama.cpp-release
server_root=$deploy/build-fresh/bin

test -d "$root/inputs"
test -f "$root/CATALOG_TRANSPORT_REBIND.json"
test -d "$source_deploy/source-after"
test ! -e "$root/run"
test ! -e "$root/SOURCE_MANIFEST.json"
test ! -e "$root/RUN_COMMAND.txt"
test -x "$server_root/llama-server"

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
command=${command//s42-phone-coverage-short-gate-v5/s42-phone-coverage-short-gate-v6}
printf '%s\n' "$command" > "$root/RUN_COMMAND.txt"
sha256sum "$root"/inputs/*.json "$root/SOURCE_MANIFEST.json" \
    "$root/CATALOG_TRANSPORT_REBIND.json" \
    > "$root/LAUNCH_INPUT_SHA256SUMS.txt"
sha256sum \
    "$repo/research_dev/scheduler/scheduler.py" \
    "$repo/research_dev/scheduler/_internal/adaptive_decode.py" \
    "$repo/research_dev/scheduler/_internal/model_placement_controller.py" \
    "$repo/research_dev/scheduler/adapters/llama_server.py" \
    "$repo/examples/layersplit/ffn-split-client.cpp" \
    "$repo/examples/layersplit/ffn-split-client.h" \
    "$repo/tools/server/server-context.cpp" \
    "$repo/tools/server/server-context.h" \
    "$repo/tools/server/server.cpp" \
    "$server_root/llama-server" \
    "$server_root/libllama-server-impl.so" \
    "$server_root/libllama-common.so" \
    "$server_root/libllama.so" \
    "$server_root/libggml.so" \
    "$server_root/libggml-base.so" \
    "$server_root/libggml-cpu.so" \
    "$server_root/libggml-cuda.so" \
    > "$root/DEPLOYMENT_SHA256SUMS.txt"
sha256sum "$previous/run/RESULT.json" "$previous/RUN_COMMAND.txt" \
    > "$root/PARENT_EVIDENCE_SHA256.txt"
sha256sum "$source_deploy"/SOURCE_AFTER_SHA256SUMS.txt \
    > "$root/SOURCE_DEPLOYMENT_SHA256.txt"

exec bash -lc "$command"
