#!/usr/bin/env bash
# Bounded Qwen memory-occupancy gate, not phone inference or a served long prompt.
set -euo pipefail
if [ "$#" -ne 4 ]; then
    echo "usage: $0 PROBE MODEL OUTPUT_DIR RIG_LOCK" >&2
    exit 2
fi
probe=$(realpath "$1")
model=$(realpath "$2")
output=$(realpath -m "$3")
rig_lock=$4
test -x "$probe"
test -f "$model"
exec 9>"$rig_lock"
flock -n 9
mkdir "$output"
sha256sum "$probe" > "$output/PROBE_SHA256.txt"
prefixes=$(python3 -c 'print(",".join(f"{i}:8192" for i in range(25, 40)))')
common=(--model "$model" --ctx-size 32768 --gpu-layers 16 --kv-device-cells "$prefixes"
        --threads 8 --batch-size 512 --ubatch-size 128 --max-tokens 512 --flash-attn on
        --tokens 9707,374,279,6722,315,279,3639 --decode 2 --kv-touch-tokens 24576)
for arm in control released; do
    extra=()
    if [ "$arm" = released ]; then
        extra=(--dormant-mask 262143 --dormant-host-columns 0 --dormant-no-decode)
    fi
    # Only this model's cache is advised away, while holding the rig lock.
    python3 - "$model" <<'PY'
import os
import sys
fd = os.open(sys.argv[1], os.O_RDONLY)
try:
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
finally:
    os.close(fd)
PY
    printf '%q ' "$probe" "${common[@]}" "${extra[@]}" --out "$output/$arm.json" > "$output/$arm.command"
    printf '\n' >> "$output/$arm.command"
    echo "START $arm"
    systemd-run --user --scope --quiet -p MemoryMax=19327352832 -p MemorySwapMax=0 \
        bash -c '
            record=$1; shift
            "$@" > "$record.stderr" 2>&1
            result=$?
            echo "$result" > "$record.exit"
            group=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
            for metric in memory.max memory.peak memory.events; do
                echo "$metric"
                cat "$group/$metric"
            done > "$record.cgroup"
            exit "$result"
        ' _ "$output/$arm" "$probe" "${common[@]}" "${extra[@]}" --out "$output/$arm.json"
    echo "DONE $arm"
done
