#!/usr/bin/env bash
set -euo pipefail
deploy=/mnt/storage/s42-fast-path-M2-diagnostic-20260920-a81c09
label=${1:-step1-n2-combined}
arm=${2:-combined}
lengths=${3:-256,257}
config=${4:-$deploy/config/n2.json}
case "$arm" in control|combined) ;; *) exit 2 ;; esac
mkdir -p "$deploy/physical/$(dirname "$label")"
test ! -e "$deploy/physical/$label"
bash "$deploy/CHECK.sh"
cd "$deploy/native-source"
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$deploy/cuda-build/bin:/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
python3 - <<'IDENTITY'
import hashlib
from pathlib import Path
from research_dev.scheduler.adapters.transport_profiles import load_transport_qualification_identity
p = Path('/mnt/storage/s42-fast-path-M2-diagnostic-20260920-a81c09')
i = load_transport_qualification_identity(p/'TRANSPORT_QUALIFICATION_IDENTITY.json')
paths = {'host_binary_sha256': p/'cuda-build/bin/llama-server',
         'transport_client_source_sha256': p/'native-source/examples/layersplit/ffn-split-usb-client.cpp'}
for name in ('ggml','ggml-base','ggml-cpu','ggml-cuda','llama','llama-common','llama-server-impl'):
    paths['host_dependency_sha256:'+name] = p/f'cuda-build/bin/lib{name}.so'
for name, path in paths.items():
    with path.open('rb') as stream:
        digest = 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != i.software_identity[name]:
        raise RuntimeError('stale transport identity: ' + name)
IDENTITY
cp "$deploy/software/unit-rig.log" "$deploy/physical/$label.unit.log"
cp "$deploy/software/pyflakes-rig.log" "$deploy/physical/$label.pyflakes.log"
command=(systemd-run --user --scope --unit="fast-path-m2-${label//\//-}-a81c09" -p MemoryMax=infinity -p MemorySwapMax=0
    python3 -u research_dev/scheduler/campaigns/burstgpt/reports/20260917-decode-relocation-kv-headroom/kv_decode_relocation_gate.py
    --config "$config" --output "$deploy/physical/$label" --arm "$arm"
    --output-tokens 64 --prompt-tokens-list "$lengths")
python3 - "$deploy/physical/$label.command.json" "${command[@]}" <<'COMMAND'
import hashlib
import json
from pathlib import Path
import sys
p = Path('/mnt/storage/s42-fast-path-M2-diagnostic-20260920-a81c09')
paths = [p/'RUN_DIAGNOSTIC.sh', p/'CHECK.sh', Path(sys.argv[sys.argv.index('--config')+1]),
         Path('research_dev/scheduler/campaigns/burstgpt/reports/20260917-decode-relocation-kv-headroom/kv_decode_relocation_gate.py')]
record = {'argv': sys.argv[2:], 'cwd': str(Path.cwd()), 'sources': {}}
for path in paths:
    record['sources'][str(path)] = {'text': path.read_text(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
with Path(sys.argv[1]).open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
COMMAND
"${command[@]}" > "$deploy/physical/$label.log" 2>&1
