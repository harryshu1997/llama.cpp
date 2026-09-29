#!/usr/bin/env bash
set -euo pipefail
deploy=/mnt/storage/s42-fast-path-M0-20260920-LuVndR
cd "$deploy/native-source"
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$deploy/cuda-build/bin:/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
from research_dev.scheduler.adapters.transport_profiles import load_transport_qualification_identity
deploy = Path('/mnt/storage/s42-fast-path-M0-20260920-LuVndR')
identity = load_transport_qualification_identity(deploy / 'TRANSPORT_QUALIFICATION_IDENTITY.json')
paths = {'host_binary_sha256': deploy / 'cuda-build/bin/llama-server',
         'transport_client_source_sha256': deploy / 'native-source/examples/layersplit/ffn-split-usb-client.cpp'}
for name in ('ggml', 'ggml-base', 'ggml-cpu', 'ggml-cuda', 'llama', 'llama-common', 'llama-server-impl'):
    paths['host_dependency_sha256:' + name] = deploy / f'cuda-build/bin/lib{name}.so'
for key, path in paths.items():
    with path.open('rb') as stream:
        value = 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()
    if value != identity.software_identity[key]:
        raise RuntimeError('transport identity is stale: ' + key)
(deploy / 'physical/TUNED_PAIR_RESUME_IDENTITY.json').write_text(json.dumps(identity.to_json(), indent=2) + '\n')
PY
systemd-run --user --scope --unit=fast-path-m0-tuned-pair-resume-LuVndR \
    -p MemoryMax=19327352832 -p MemorySwapMax=0 \
    python3 -u research_dev/scheduler/campaigns/burstgpt/reports/20260917-decode-relocation-kv-headroom/kv_decode_relocation_gate.py \
    --config "$deploy/config/tuned-pair.json" --output "$deploy/physical/tuned-pair-resume" \
    --arm pair --host-columns 0 --output-tokens 64 --wait-for-rig-seconds 3600 \
    > "$deploy/physical/tuned-pair-resume.log" 2>&1
