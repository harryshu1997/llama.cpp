#!/usr/bin/env bash
set -euo pipefail
deploy=/mnt/storage/s42-fast-path-M1-20260920-T3ypai
kind=${1:?traced or heldout}
fixture=${2:-pressure-limited}
case "$kind" in traced|heldout|diagnostic) ;; *) exit 2 ;; esac
physical=$deploy/physical
config=$deploy/config
limit=19327352832
case "$fixture" in
    pressure-limited) ;;
    no-pressure)
        physical=$physical/no-pressure
        config=$config/no-pressure
        limit=infinity
        ;;
    *) exit 2 ;;
esac
test -d "$physical"
cd "$deploy/native-source"
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$deploy/cuda-build/bin:/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
if [[ "$kind" == heldout ]]; then
    test -f "$physical/FROZEN_PREDICTIONS.json"
fi
if [[ "$kind" == diagnostic ]]; then
    python3 - "$physical/CHECK_M1.json" <<'CHECK'
import json, sys
check = json.load(open(sys.argv[1]))
assert check['status'] == 'FAIL' and not check['arms']['1024']['within_10_percent']
CHECK
fi
python3 - <<'IDENTITY'
import hashlib
from pathlib import Path
from research_dev.scheduler.adapters.transport_profiles import load_transport_qualification_identity
p=Path('/mnt/storage/s42-fast-path-M1-20260920-T3ypai')
i=load_transport_qualification_identity(p/'TRANSPORT_QUALIFICATION_IDENTITY.json')
paths={'host_binary_sha256':p/'cuda-build/bin/llama-server',
       'transport_client_source_sha256':p/'native-source/examples/layersplit/ffn-split-usb-client.cpp'}
for name in ('ggml','ggml-base','ggml-cpu','ggml-cuda','llama','llama-common','llama-server-impl'):
    paths['host_dependency_sha256:'+name]=p/f'cuda-build/bin/lib{name}.so'
for name,path in paths.items():
    with path.open('rb') as f: digest='sha256:'+hashlib.file_digest(f,'sha256').hexdigest()
    if digest != i.software_identity[name]: raise RuntimeError('stale transport identity: '+name)
IDENTITY
ubatches=(128 1024)
if [[ "$kind" == diagnostic ]]; then ubatches=(1024); fi
for ubatch in "${ubatches[@]}"; do
    test ! -e "$physical/$kind-$ubatch"
    cp "$deploy/software/unit-rig.log" "$physical/$kind-$ubatch.unit.log"
    cp "$deploy/software/pyflakes-rig.log" "$physical/$kind-$ubatch.pyflakes.log"
    systemd-run --user --scope --unit="fast-path-m1-$fixture-$kind-$ubatch-T3ypai" \
        -p MemoryMax="$limit" -p MemorySwapMax=0 \
        python3 -u research_dev/scheduler/campaigns/burstgpt/reports/20260917-decode-relocation-kv-headroom/kv_decode_relocation_gate.py \
        --config "$config/$kind-$ubatch.json" --output "$physical/$kind-$ubatch" \
        --arm control --output-tokens 64 > "$physical/$kind-$ubatch.log" 2>&1
done
