#!/usr/bin/env bash
set -u -o pipefail

stage=$1
inputs=$2
source_root=/mnt/storage/s42-trace-v2-20260921-prep/source
exec 9>/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
flock -w 900 9 || exit 1
trap 'status=$?; printf "%s\n" "$status" > "$stage/RETEST_EXIT.txt"; date -u +%FT%TZ > "$stage/RETEST_FINISHED.txt"' EXIT
test ! -e "$inputs" || exit 1
mkdir "$stage/source-before" || exit 1
rsync -rlptc --exclude __pycache__ --exclude .venv --exclude campaigns/burstgpt/reports \
    "$source_root/research_dev/scheduler/" "$stage/source-before/" || exit 1
rsync -rlptc --exclude __pycache__ --exclude .venv --exclude campaigns/burstgpt/reports \
    "$stage/scheduler/" "$source_root/research_dev/scheduler/" || exit 1
cd "$source_root" || exit 1
export LANG=C.UTF-8
export S42_UNIFIED_REPO_ROOT="$source_root"
export LD_LIBRARY_PATH=/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
python3 - "$stage" <<'PY' || exit 1
import hashlib
import json
from pathlib import Path
import sys
stage = Path(sys.argv[1])
expected = json.loads((stage / 'SOURCE_HASHES.json').read_text())
for name, digest in expected.items():
    path = Path('research_dev/scheduler') / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, str(path)
(stage / 'DEPLOYMENT_VERIFIED.json').write_text(json.dumps({
    'status': 'PASS', 'sha256': expected,
}, indent=2, sort_keys=True) + '\n')
PY
PYTHONPATH=.:research_dev/scheduler/tests python3 -m unittest \
    test_catalog_materialization test_llama_server_adapter test_adaptive_coherence \
    > "$stage/DEPLOYMENT_TESTS.log" 2>&1 || exit 1
python3 "$stage/PREPARE_TASK1_INPUTS.py" "$inputs" > "$stage/PREPARE.log" 2>&1 || exit 1
python3 research_dev/scheduler/campaigns/burstgpt/launch.py \
    "$inputs/campaign.json" "$inputs/preflight-1" --preflight-only \
    > "$inputs/PREFLIGHT.log" 2>&1 || exit 1
PYTHONPATH=. python3 "$stage/CHECK_TASK1_ADMISSION.py" "$inputs" \
    > "$inputs/ADMISSION.log" 2>&1 || exit 1
date -u +%FT%TZ > "$inputs/RUN_STARTED.txt"
python3 research_dev/scheduler/campaigns/burstgpt/launch.py \
    "$inputs/campaign.json" "$inputs/run-treatment-1" > "$inputs/RUN.log" 2>&1
status=$?
printf '%s\n' "$status" > "$inputs/RUN_EXIT.txt"
date -u +%FT%TZ > "$inputs/RUN_FINISHED.txt"
exit "$status"
