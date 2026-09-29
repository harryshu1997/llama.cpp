#!/usr/bin/env bash
set -euo pipefail
deploy=/mnt/storage/s42-fast-path-M0-20260920-LuVndR
exec 9>/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
flock -n 9
cd "$deploy/native-source"
export PYTHONDONTWRITEBYTECODE=1
export S42_LLAMA_BUILD_BIN=$deploy/cuda-build/bin
export LD_LIBRARY_PATH=$deploy/cuda-build/bin:/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
test -x "$S42_LLAMA_BUILD_BIN/llama-ffn-remote-resident-probe"
PYTHONPATH=research_dev/scheduler/tests:gguf-py:. python3 -m unittest \
    test_remote_resident_native.DormantHostShareNativeTests test_kv_lazy_backing \
    test_host_share_release test_kv_decode_relocation_gate test_llama_server_adapter \
    test_decode_split_selection test_dormant_share_coordinator \
    > "$deploy/physical/unit-restore-rig.log" 2>&1
PYTHONPATH=$deploy/software/tooling python3 -m pyflakes \
    research_dev/scheduler/adapters/llama_server_contracts.py \
    research_dev/scheduler/adapters/llama_server.py research_dev/scheduler/adapters/contracts.py \
    research_dev/scheduler/_unified/common.py research_dev/scheduler/_internal/runtime_resources.py \
    research_dev/scheduler/campaigns/burstgpt/reports/20260917-decode-relocation-kv-headroom/kv_decode_relocation_gate.py \
    research_dev/scheduler/tests/test_host_share_release.py \
    research_dev/scheduler/tests/test_remote_resident_native.py \
    research_dev/scheduler/tests/test_kv_decode_relocation_gate.py \
    > "$deploy/software/pyflakes-restore-rig.log" 2>&1
