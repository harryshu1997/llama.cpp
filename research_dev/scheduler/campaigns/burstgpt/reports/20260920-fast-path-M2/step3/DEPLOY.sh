#!/usr/bin/env bash
set -euo pipefail
cd /home/myid/zs89458/Documents/llama.cpp-release
remote=zhihao@172.20.74.85
deploy=/mnt/storage/s42-fast-path-M2-table-20260920-ef548a
report=research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/step3
ssh -o BatchMode=yes "$remote" 'bash -s' <<'SH'
set -euo pipefail
test ! -e /mnt/storage/s42-fast-path-M2-table-20260920-ef548a
mkdir -p /mnt/storage/s42-fast-path-M2-table-20260920-ef548a/software /mnt/storage/s42-fast-path-M2-table-20260920-ef548a/physical
cp -a --reflink=auto /mnt/storage/s42-fast-path-M2-diagnostic-20260920-a81c09/native-source /mnt/storage/s42-fast-path-M2-table-20260920-ef548a/native-source
cp -a --reflink=auto /mnt/storage/s42-fast-path-M2-diagnostic-20260920-a81c09/software/tooling /mnt/storage/s42-fast-path-M2-table-20260920-ef548a/software/tooling
cp /mnt/storage/s42-fast-path-M2-diagnostic-20260920-a81c09/PROMPT.txt /mnt/storage/s42-fast-path-M2-table-20260920-ef548a/PROMPT.txt
SH
rsync -aR examples/layersplit/ffn-split-client.cpp examples/layersplit/ffn-split-client.h \
    examples/layersplit/ffn-remote-resident-probe.cpp tools/server/server-context.cpp \
    research_dev/scheduler/tests/test_remote_resident_native.py "$remote:$deploy/native-source/"
rsync -a cmake/build-info.cmake "$remote:$deploy/native-source/cmake/"
rsync -a common/build-info.cpp.in "$remote:$deploy/native-source/common/"
rsync -a "$report/BUILD.sh" "$report/MATERIALIZE_TRANSPORT.sh" "$report/CHECK.sh" "$report/config" "$remote:$deploy/"
rsync -a research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/ANALYZE.py "$remote:$deploy/"
