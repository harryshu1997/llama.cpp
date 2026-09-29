#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/home/myid/zs89458/Documents/moe-resident-routing-a6000}

cd "$project_root"
exec .venv-qwen35/bin/python -m src.benchmark_llamacpp_op12_score_match "$@"
