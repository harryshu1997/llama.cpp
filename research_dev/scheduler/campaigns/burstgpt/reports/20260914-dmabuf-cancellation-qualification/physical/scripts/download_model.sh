#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="Qwen/Qwen3-30B-A3B"
REVISION="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
MIN_FREE_KIB=$((70 * 1024 * 1024))
AVAILABLE_KIB="$(df --output=avail -k "${HF_HOME:-$HOME/.cache/huggingface}" | tail -1 | tr -d ' ')"

if "${PROJECT_DIR}/.venv/bin/python" - "${MODEL_ID}" "${REVISION}" >/dev/null 2>&1 <<'PY'
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

snapshot = Path(snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True))
index_path = snapshot / "model.safetensors.index.json"
index = json.loads(index_path.read_text(encoding="utf-8"))
required = {"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"}
required.update(index["weight_map"].values())
if not all((snapshot / filename).is_file() and (snapshot / filename).stat().st_size > 0 for filename in required):
    raise SystemExit(1)
PY
then
  echo "Pinned model revision is already complete in the Hugging Face cache."
  exit 0
fi

if (( AVAILABLE_KIB < MIN_FREE_KIB )); then
  echo "Refusing model download: need at least 70 GiB free, have $((AVAILABLE_KIB / 1024 / 1024)) GiB." >&2
  exit 2
fi

"${PROJECT_DIR}/.venv/bin/hf" download "${MODEL_ID}" \
  --revision "${REVISION}" \
  --exclude '*.gguf' '*.bin' '*.h5' '*.msgpack'
