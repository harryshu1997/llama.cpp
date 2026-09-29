#!/usr/bin/env bash
set -euo pipefail

project_root=/home/myid/zs89458/Documents/moe-resident-routing-a6000
pip_cache=/mnt/data_s4t/zs89458-moe-routing/pip-cache

cd "$project_root"
install -d -m 700 "$pip_cache"
python3 -m venv .venv-qwen35
PIP_CACHE_DIR="$pip_cache" .venv-qwen35/bin/python -m pip install --upgrade pip setuptools wheel
PIP_CACHE_DIR="$pip_cache" .venv-qwen35/bin/python -m pip install -r requirements-qwen35.txt
