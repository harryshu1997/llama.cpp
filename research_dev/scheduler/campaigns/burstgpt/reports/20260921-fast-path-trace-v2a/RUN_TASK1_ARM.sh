#!/usr/bin/env bash
set -u -o pipefail

inputs=$1
source_root=/mnt/storage/s42-trace-v2-20260921-prep/source
exec 9>/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
flock -w 900 9 || exit 1
test ! -e "$inputs/run-treatment-1" || exit 1
python3 - "$inputs" <<'PY' || exit 1
import json
from pathlib import Path
import sys
p = Path(sys.argv[1])
assert json.loads((p / 'preflight-1/PHYSICAL_PREFLIGHT.json').read_text())['status'] == 'PASS'
assert json.loads((p / 'TRANSPORT_ADMISSION.json').read_text())['status'] == 'PASS'
PY
cd "$source_root" || exit 1
export LANG=C.UTF-8
export S42_UNIFIED_REPO_ROOT="$source_root"
export LD_LIBRARY_PATH=/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
date -u +%FT%TZ > "$inputs/RUN_STARTED.txt"
python3 research_dev/scheduler/campaigns/burstgpt/launch.py \
    "$inputs/campaign.json" "$inputs/run-treatment-1"
status=$?
printf '%s\n' "$status" > "$inputs/RUN_EXIT.txt"
date -u +%FT%TZ > "$inputs/RUN_FINISHED.txt"
exit "$status"
