#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/home/myid/zs89458/Documents/moe-resident-routing-a6000}
cd "$project_root"

export PYTHONPATH="$project_root"
export PYTHONHASHSEED=89458
export HF_HOME=/mnt/data_s4t/zs89458-moe-routing/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets

./scripts/start_op12_llamacpp_service.sh

.venv-qwen35/bin/python -m src.run_llamacpp_a6000_op12 "$@"
