#!/usr/bin/env bash
set -euo pipefail

project_root=/home/myid/zs89458/Documents/moe-resident-routing-a6000
results_dir=results/qwen35_35b_a3b/timing_overlap
cd "$project_root"
mkdir -p "$results_dir"
export PYTHONPATH="$project_root"
export PYTHONHASHSEED=89458
export CUBLAS_WORKSPACE_CONFIG=:4096:8

.venv-qwen35/bin/python -m src.benchmark_qwen35_loading_overlap \
  --config configs/qwen35_35b_a3b.json \
  2>&1 | tee "$results_dir/full_run_console.log"
