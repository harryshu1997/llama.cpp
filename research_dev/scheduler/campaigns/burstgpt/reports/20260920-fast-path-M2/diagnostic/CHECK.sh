#!/usr/bin/env bash
set -euo pipefail
deploy=/mnt/storage/s42-fast-path-M2-diagnostic-20260920-a81c09
exec 9>/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
flock -n 9
cd "$deploy/native-source"
export PYTHONDONTWRITEBYTECODE=1
export S42_LLAMA_BUILD_BIN=$deploy/cuda-build/bin
export LD_LIBRARY_PATH=$deploy/cuda-build/bin:/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
export PYTHONPATH=research_dev/scheduler/tests:gguf-py:.
mkdir -p "$deploy/software/tmp"
export TMPDIR=$deploy/software/tmp
python3 -m unittest test_decode_cohort test_kv_decode_relocation_gate test_llama_server_adapter test_remote_resident_native > "$deploy/software/unit-rig.log" 2>&1
PYTHONPATH=$deploy/software/tooling python3 -m pyflakes \
 research_dev/scheduler/adapters/llama_server.py research_dev/scheduler/adapters/llama_server_contracts.py research_dev/scheduler/adapters/http_backend.py \
 research_dev/scheduler/_internal/runtime_decode_cohort.py \
 research_dev/scheduler/campaigns/burstgpt/reports/20260917-decode-relocation-kv-headroom/kv_decode_relocation_gate.py \
 research_dev/scheduler/tests/test_decode_cohort.py research_dev/scheduler/tests/test_kv_decode_relocation_gate.py \
 research_dev/scheduler/tests/test_remote_resident_native.py research_dev/scheduler/tests/test_llama_server_adapter.py "$deploy/ANALYZE.py" > "$deploy/software/pyflakes-rig.log" 2>&1
