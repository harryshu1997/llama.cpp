#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <qwen-cuda|gemma-cpu> <repeat-index> <output-root>" >&2
    exit 2
fi

route=$1
repeat_index=$2
output=$3

if [[ $route != qwen-cuda && $route != gemma-cpu ]]; then
    echo "server-only acquisition does not accept route: $route" >&2
    exit 2
fi
if [[ ! $repeat_index =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid repeat index: $repeat_index" >&2
    exit 2
fi
if [[ $output != /* || -e $output ]]; then
    echo "invalid output root: $output" >&2
    exit 2
fi
if pgrep -x llama-server >/dev/null 2>&1; then
    echo "foreign llama-server process" >&2
    exit 75
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | grep -Eq '^[[:space:]]*[0-9]+'; then
    echo "foreign CUDA process" >&2
    exit 75
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
command=(
    python3 "$here/run_model_energy.py"
    --route "$route"
    --repeat-index "$repeat_index"
    --cases "$here/CASES_V1.jsonl"
    --catalog "$here/MODELS_V1.json"
    --port 19100
    --output "$output"
    --execute
    --confirm RUN_S42_MODEL_ENERGY_V1
)
if [[ $route == qwen-cuda ]]; then
    command+=(
        --server /home/zhihao/llama.cpp-s40/build-s40-cuda/bin/llama-server
        --model /home/zhihao/models/Qwen3-14B-Q4_K_M.gguf
        --lib-dir /mnt/storage/s21_deps/cuda-13.2.1/lib
    )
else
    command+=(
        --server /home/zhihao/s41-dynamic-ffn-v1/build-server-ffn/bin/llama-server
        --model /home/zhihao/models/gemma-4-12B-it-Q4_0-op15-exact.gguf
        --lib-dir /home/zhihao/s41-dynamic-ffn-v1/build-server-ffn/bin
    )
fi

systemd-run --user --scope -p MemorySwapMax=0 "${command[@]}"
sha256sum "$output/RESULT.json"
