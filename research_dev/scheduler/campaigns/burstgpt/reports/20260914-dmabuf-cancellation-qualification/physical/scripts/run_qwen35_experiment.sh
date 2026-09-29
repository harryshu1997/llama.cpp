#!/usr/bin/env bash
set -euo pipefail

project_root=/home/myid/zs89458/Documents/moe-resident-routing-a6000
cd "$project_root"
mkdir -p results/qwen35_35b_a3b/logs
export HF_HOME=/mnt/data_s4t/zs89458-moe-routing/huggingface
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=89458
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$project_root"

.venv-qwen35/bin/python -m src.run_qwen35_experiment \
  --config configs/qwen35_35b_a3b.json \
  2>&1 | tee results/qwen35_35b_a3b/logs/full_run_console.log
