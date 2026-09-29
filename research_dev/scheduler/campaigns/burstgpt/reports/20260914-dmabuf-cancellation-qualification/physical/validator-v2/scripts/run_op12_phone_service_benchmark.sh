#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/home/myid/zs89458/Documents/moe-resident-routing-a6000}

cd "$project_root"
exec env OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
  .venv-qwen35/bin/python -m src.benchmark_op12_phone_service "$@"
