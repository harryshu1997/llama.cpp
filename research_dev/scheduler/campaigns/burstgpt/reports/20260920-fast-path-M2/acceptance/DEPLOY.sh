#!/usr/bin/env bash
set -euo pipefail
cd /home/myid/zs89458/Documents/llama.cpp-release
report=research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/acceptance
remote=zhihao@172.20.74.85
deploy=/mnt/storage/s42-fast-path-M2-acceptance-20260921-c09ea6
ssh -o BatchMode=yes "$remote" "mkdir -p $deploy/native-source $deploy/software"
ssh -o BatchMode=yes "$remote" "ln -s native-source/research_dev $deploy/research_dev"
rsync -a --exclude 'build*/' --exclude '__pycache__/' --exclude '*.pyc' --exclude 'node_modules/' \
    CMakeLists.txt LICENSE cmake common src include ggml gguf-py examples tools tests vendor pocs scripts \
    "$remote:$deploy/native-source/"
ssh -o BatchMode=yes "$remote" "mkdir -p $deploy/native-source/research_dev/scheduler"
rsync -a --exclude 'reports/' --exclude 'baselines/' --exclude 'physical/' \
    --exclude '__pycache__/' --exclude '*.pyc' \
    research_dev/scheduler/ "$remote:$deploy/native-source/research_dev/scheduler/"
rsync -aR \
    research_dev/scheduler/campaigns/burstgpt/reports/20260917-decode-relocation-kv-headroom/kv_decode_relocation_gate.py \
    "$remote:$deploy/native-source/"
rsync -aR \
    research_dev/spikes/s42_general_energy_scheduler_v1/small_model_phone_v1/results/4060ti_op15_20260808/SCHEDULER_PROFILE_CUDA_EPOCH_OPEN.json \
    research_dev/spikes/s42_general_energy_scheduler_v1/small_model_phone_v1/results/4060ti_op15_20260808/SCHEDULER_PROFILE_CUDA_EPOCH_REUSED.json \
    "$remote:$deploy/native-source/"
rsync -a cmake/build-info.cmake "$remote:$deploy/native-source/cmake/"
rsync -a common/build-info.cpp.in "$remote:$deploy/native-source/common/"
rsync -a /tmp/fast-path-pyflakes/ "$remote:$deploy/software/tooling/"
rsync -a "$report/BUILD.sh" "$report/MATERIALIZE_TRANSPORT.sh" "$report/CHECK.sh" "$remote:$deploy/"
rsync -aR research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/ANALYZE.py "$remote:$deploy/native-source/"
rsync -a "$report/../ANALYZE.py" "$remote:$deploy/"
rsync -a "$report/../../20260920-fast-path-M0/physical/tuned-pair-resume/control/" "$remote:$deploy/reference-m0/"
rsync -a "$report/config" "$report/RUN_ARM.sh" "$report/RUN_CHECKS.sh" "$remote:$deploy/"
ssh -o BatchMode=yes "$remote" 'bash -s' <<'SH'
mkdir -p /mnt/storage/s42-fast-path-M2-acceptance-20260921-c09ea6/physical
cp /mnt/storage/s42-fast-path-M0-20260920-LuVndR/PROMPT.txt /mnt/storage/s42-fast-path-M2-acceptance-20260921-c09ea6/PROMPT.txt
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
deploy = Path('/mnt/storage/s42-fast-path-M2-acceptance-20260921-c09ea6')
source = deploy / 'native-source'
hashes = {}
for path in sorted(source.rglob('*')):
    if path.is_file():
        with path.open('rb') as stream:
            hashes[str(path.relative_to(source))] = hashlib.file_digest(stream, 'sha256').hexdigest()
(deploy / 'software/SOURCE_SHA256.json').write_text(json.dumps(hashes, indent=2) + '\n')
print(f'Snapshotted {len(hashes)} source files')
PY
SH
