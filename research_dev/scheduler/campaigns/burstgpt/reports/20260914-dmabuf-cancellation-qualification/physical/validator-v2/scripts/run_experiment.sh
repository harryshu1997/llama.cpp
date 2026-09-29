#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"${PROJECT_DIR}/.venv/bin/python" -m src.run_experiment \
  --config "${PROJECT_DIR}/configs/experiment.json" \
  2>&1 | tee "${PROJECT_DIR}/results/logs/full_run_console.log"
