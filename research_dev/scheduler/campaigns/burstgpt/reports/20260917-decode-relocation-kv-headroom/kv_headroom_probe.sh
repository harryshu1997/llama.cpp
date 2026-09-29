#!/bin/bash
# KV growth into the freed memory, no phone, no long generation:
#   control : load Qwen (16 GPU layers, CPU KV layers 0-31), decode 2 tokens, touch the KV pages of N cells
#   combined: same, but release the phone share (layers 0-17, host columns 0) before the touch
# both inside a fresh MemoryMax scope; the scope's memory.events and the probe's JSON are the record.
# usage: kv_headroom_probe.sh <MemoryMax bytes> <kv touch tokens> <output dir>
set -u
cd /mnt/storage/s42-kv-decode-relocation-20260917-v1-eedc22
CAP=$1; TOUCH=$2; OUT=$3; mkdir -p "$OUT"
PROBE=/mnt/storage/s42-kv-decode-relocation-20260917-v1-eedc22/cuda-build/bin/llama-ffn-remote-resident-probe
export LD_LIBRARY_PATH=/mnt/storage/s42-kv-decode-relocation-20260917-v1-eedc22/cuda-build/bin:/mnt/storage/s21_deps/cuda-13.2.1/lib
KVL=$(python3 -c "print(','.join(str(i) for i in range(32)))")
COMMON="--model /home/zhihao/models/Qwen3-14B-Q4KM-dequant-f16.gguf --ctx-size 32768 --gpu-layers 16 --kv-cpu-layers $KVL --threads 8 --batch-size 512 --ubatch-size 128 --max-tokens 512 --tokens 9707,374,279,6722,315,279,3639 --decode 2 --kv-touch-tokens $TOUCH"
for ARM in control combined; do
  EXTRA=""; [ "$ARM" = combined ] && EXTRA="--dormant-mask 262143 --dormant-host-columns 0 --dormant-no-decode"
  # evict the model file from the page cache so the scope is charged for what it faults
  python3 - <<'PY'
import os
fd = os.open('/home/zhihao/models/Qwen3-14B-Q4KM-dequant-f16.gguf', os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
PY
  echo "START $ARM $(date -Is)"
  systemd-run --user --scope --quiet -p MemoryMax=$CAP -p MemorySwapMax=0 bash -c "
    $PROBE $COMMON $EXTRA --out $OUT/$ARM.json > $OUT/$ARM.stderr 2>&1; echo probe-exit=\$? > $OUT/$ARM.exit
    CG=/sys/fs/cgroup\$(cut -d: -f3 /proc/self/cgroup)
    { echo memory.max=\$(cat \$CG/memory.max); echo memory.peak=\$(cat \$CG/memory.peak 2>/dev/null); cat \$CG/memory.events; } > $OUT/$ARM.cgroup"
  echo "END $ARM $(date -Is) $(cat $OUT/$ARM.exit)"
done
echo KV_HEADROOM_DONE
