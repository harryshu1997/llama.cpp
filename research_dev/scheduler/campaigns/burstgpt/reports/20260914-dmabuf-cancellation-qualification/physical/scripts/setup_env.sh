#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3.12 -m venv "${PROJECT_DIR}/.venv"
"${PROJECT_DIR}/.venv/bin/python" -m pip install --upgrade pip wheel setuptools
"${PROJECT_DIR}/.venv/bin/python" -m pip install -r "${PROJECT_DIR}/requirements.txt"

