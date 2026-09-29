#!/usr/bin/env bash
set -euo pipefail
deploy=/mnt/storage/s42-fast-path-M2-20260920-d7BA1a
n=${1:?1, 2, 4, 8, or pair-v1}
arm=${2:?control or combined}
case "$n" in 1|2|4|8|pair-v1) ;; *) exit 2 ;; esac
case "$arm" in control|combined) ;; *) exit 2 ;; esac
label=n$n-$arm
config=$deploy/config/n$n.json
extra=(--output-tokens 576 --prompt-tokens-list)
if [[ "$n" == pair-v1 ]]; then
    config=$deploy/config/pair-v1.json
    extra=(--output-tokens 64)
else
    lengths=$(python3 -c 'import sys; print(",".join(str(256+i) for i in range(int(sys.argv[1]))))' "$n")
    extra+=("$lengths")
fi
test ! -e "$deploy/physical/$label"
bash "$deploy/CHECK.sh"
cd "$deploy/native-source"
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$deploy/cuda-build/bin:/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
python3 - <<'IDENTITY'
import hashlib
from pathlib import Path
from research_dev.scheduler.adapters.transport_profiles import load_transport_qualification_identity
p = Path('/mnt/storage/s42-fast-path-M2-20260920-d7BA1a')
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
command=(systemd-run --user --scope --unit="fast-path-m2-$label-d7BA1a" -p MemoryMax=infinity -p MemorySwapMax=0
    python3 -u research_dev/scheduler/campaigns/burstgpt/reports/20260917-decode-relocation-kv-headroom/kv_decode_relocation_gate.py
    --config "$config" --output "$deploy/physical/$label" --arm "$arm" "${extra[@]}")
python3 - "$deploy/physical/$label.command.json" "${command[@]}" <<'COMMAND'
import hashlib
import json
from pathlib import Path
import sys
p = Path('/mnt/storage/s42-fast-path-M2-20260920-d7BA1a')
paths = [p/'RUN_ARM.sh', p/'ANALYZE.py', p/'CHECK.sh', Path(sys.argv[sys.argv.index('--config') + 1]),
         Path('research_dev/scheduler/campaigns/burstgpt/reports/20260917-decode-relocation-kv-headroom/kv_decode_relocation_gate.py')]
record = {'argv': sys.argv[2:], 'cwd': str(Path.cwd()), 'sources': {}}
for path in paths:
    record['sources'][str(path)] = {'text': path.read_text(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
with Path(sys.argv[1]).open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
COMMAND
"${command[@]}" > "$deploy/physical/$label.log" 2>&1
