#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cd "$project_root"
exec .venv-qwen35/bin/python -m src.benchmark_op12_async_pipeline "$@"
