#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
out=${1:-"$here/results/w9_profiled_cutover/run_$stamp"}

PYTHONDONTWRITEBYTECODE=1 python3 \
    "$here/run_w9_profiled_cutover_gate.py" \
    --output "$out"
