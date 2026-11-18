#!/bin/bash
# Host GPU leg: run_gpu.sh OUTDIR BENCH SHARD_ROOT MODEL:N [MODEL:N ...]
#   env: DEVICE (CUDA0) IDLE WARM SECS COOL_GPU_C (GPU temperature to wait for, C) COOL_C (CPU package, milli C)
#        MIN_COOL_S MAX_COOL_S MAX_LOAD (busy cores over 2 s)
#        GPU_INDEX (nvidia-smi index on a host with several GPUs: the bench then sees only that GPU, as CUDA0)
# Per run: wait for the GPU and package temperatures and an idle host, start the GPU board sampler and the CPU
# package sampler, run the bench on the GPU (idle window with the weights resident in GPU memory, warm-up,
# measured back-to-back calls), stop the samplers.
set -u
OUT=$1; BENCH=$2; SH=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
DEVICE=${DEVICE:-CUDA0}; GPU_INDEX=${GPU_INDEX:-}; IDLE=${IDLE:-10}; WARM=${WARM:-3}; SECS=${SECS:-20}
COOL_GPU_C=${COOL_GPU_C:-42}; COOL_C=${COOL_C:-40000}; MIN_COOL_S=${MIN_COOL_S:-30}; MAX_COOL_S=${MAX_COOL_S:-420}; MAX_LOAD=${MAX_LOAD:-1.5}
mkdir -p "$OUT"
log(){ echo "$(date -u +%T) $*" | tee -a "$OUT/campaign.log"; }
SMI=(nvidia-smi); [ -n "$GPU_INDEX" ] && { SMI=(nvidia-smi -i "$GPU_INDEX"); export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_INDEX; }
gtemp(){ "${SMI[@]}" --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d ' '; }
temp(){ python3 - <<'PY'
import glob
best = -1
for h in glob.glob("/sys/class/hwmon/hwmon*"):
    try:
        if open(h + "/name").read().strip() in ("coretemp", "k10temp"):
            for p in glob.glob(h + "/temp*_input"):
                best = max(best, int(open(p).read()))
    except OSError:
        pass
print(best)
PY
}
busy(){ python3 - <<'PY'
import os, time
def snap():
    v = list(map(int, open("/proc/stat").readline().split()[1:]))
    return sum(v), v[3] + v[4]
a = snap(); time.sleep(2); b = snap()
print(round((1 - (b[1] - a[1]) / (b[0] - a[0])) * os.cpu_count(), 2))
PY
}
for item in "$@"; do
  model=${item%%:*}; n=${item##*:}; tag=$model-L$n
  [ -e "$OUT/$tag.jsonl" ] && { log "$tag exists, skipped"; continue; }
  t0=$(date +%s)
  while :; do
    gc=$(gtemp); tc=$(temp); load=$(busy); waited=$(( $(date +%s) - t0 ))
    ok=$(python3 -c "print(int($gc <= $COOL_GPU_C and $tc <= $COOL_C and $load <= $MAX_LOAD))")
    if [ "$ok" = 1 ] && [ $waited -ge $MIN_COOL_S ]; then cool=1; break; fi
    if [ $waited -ge $MAX_COOL_S ]; then cool=0; break; fi
    sleep 5
  done
  log "$tag start cool=$cool waited=${waited}s gpu_c=$gc cpu_mc=$tc load=$load device=$DEVICE gpu_index=${GPU_INDEX:-default}"
  rm -f "$OUT/$tag.stop"
  python3 "$HERE/host_sampler.py" "$OUT/$tag.power" "$OUT/$tag.stop" 0.1 &
  spid=$!
  python3 "$HERE/gpu_sampler.py" "$OUT/$tag.gpu" "$OUT/$tag.stop" 100 "$GPU_INDEX" &
  gpid=$!
  sleep 5
  "$BENCH" --shard "$SH/$model/HTP0.ffn.gguf" --layers $n --threads 1 --device $DEVICE \
      --idle-s $IDLE --warmup-s $WARM --seconds $SECS --out "$OUT/$tag.jsonl" 2> "$OUT/$tag.stderr"
  rc=$?
  sleep 2
  touch "$OUT/$tag.stop"; wait $spid $gpid
  lbytes=$(python3 -c "import json; print(json.load(open('$SH/$model/HTP0.ffn.json'))['layer_bytes_f16'])")
  echo "{\"tag\":\"$tag\",\"model\":\"$model\",\"layers\":$n,\"layer_bytes\":$lbytes,\"device\":\"$DEVICE\",\"cool\":$cool,\"cool_wait_s\":$waited,\"pre_gpu_temp_c\":$gc,\"pre_temp_mc\":$tc,\"pre_load\":$load,\"rc\":$rc}" > "$OUT/$tag.json"
  log "$tag done rc=$rc"
done
log CAMPAIGN_DONE
