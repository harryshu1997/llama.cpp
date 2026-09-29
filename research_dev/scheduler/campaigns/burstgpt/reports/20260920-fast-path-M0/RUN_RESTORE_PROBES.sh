#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
    echo "usage: $0 THREADS CPU_LIST" >&2
    exit 2
fi
threads=$1
affinity=$2
case "$threads" in 8|12|16|24) ;; *) exit 2 ;; esac
deploy=/mnt/storage/s42-fast-path-M0-20260920-LuVndR
probe=$deploy/cuda-build/bin/llama-ffn-remote-resident-probe
model=/home/zhihao/models/Qwen3-14B-Q4KM-dequant-f16.gguf
output=$deploy/physical/restore
export LD_LIBRARY_PATH=$deploy/cuda-build/bin:/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
exec 9>/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)"
mkdir "$output"
sha256sum "$probe" > "$output/PROBE_SHA256.txt"
layers=$(python3 -c 'print(",".join(map(str, range(32))))')
common=(--model "$model" --ctx-size 32768 --gpu-layers 16 --kv-cpu-layers "$layers"
        --threads "$threads" --batch-size 2048 --ubatch-size 1024 --max-tokens 4
        --tokens 9707,374,279,6722,315,279,3639 --decode 2
        --dormant-mask 262143 --dormant-host-columns 0 --dormant-restore-before-decode)
for arm in old-drop keep-populate-warm keep-populate-pressure keep-lazy-warm keep-lazy-pressure; do
    drop=0
    populate=1
    touch=0
    case "$arm" in
        old-drop) drop=1 ;;
        keep-populate-pressure) touch=24576 ;;
        keep-lazy-warm) populate=0 ;;
        keep-lazy-pressure) populate=0; touch=24576 ;;
    esac
    python3 - "$model" <<'PY'
import os
import sys
fd = os.open(sys.argv[1], os.O_RDONLY)
try:
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
finally:
    os.close(fd)
PY
    command=(taskset --cpu-list "$affinity" "$probe" "${common[@]}"
             --dormant-drop-cache "$drop" --dormant-populate "$populate"
             --kv-touch-tokens "$touch" --out "$output/$arm.json")
    printf '%q ' "${command[@]}" > "$output/$arm.command"
    printf '\n' >> "$output/$arm.command"
    date -Is > "$output/$arm.started"
    echo "START $arm"
    systemd-run --user --scope --quiet --unit="fast-path-m0-restore-$arm-LuVndR" \
        -p MemoryMax=19327352832 -p MemorySwapMax=0 bash -c '
            record=$1; shift
            /usr/bin/time -v -o "$record.time" "$@" > "$record.stderr" 2>&1
            result=$?
            echo "$result" > "$record.exit"
            group=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
            for metric in memory.max memory.peak memory.events memory.stat; do
                echo "$metric"
                cat "$group/$metric"
            done > "$record.cgroup"
            date -Is > "$record.finished"
            exit "$result"
        ' _ "$output/$arm" "${command[@]}"
    echo "DONE $arm"
done
