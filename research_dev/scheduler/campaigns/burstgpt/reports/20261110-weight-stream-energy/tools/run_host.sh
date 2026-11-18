#!/bin/bash
# Host CPU legs: run_host.sh OUTDIR BENCH SHARD_ROOT THREADS MODEL:N [MODEL:N ...]
#   env: IDLE WARM SECS COOL_C (package temperature to wait for, milli C) MIN_COOL_S MAX_COOL_S MAX_LOAD
# Per run: wait for the package temperature and the 1-min load, start the sampler, run the bench (idle window
# with the weights resident, warm-up, measured back-to-back calls), stop the sampler.
set -u
OUT=$1; BENCH=$2; SH=$3; THREADS=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd)
IDLE=${IDLE:-10}; WARM=${WARM:-3}; SECS=${SECS:-30}
COOL_C=${COOL_C:-45000}; MIN_COOL_S=${MIN_COOL_S:-30}; MAX_COOL_S=${MAX_COOL_S:-300}; MAX_LOAD=${MAX_LOAD:-1.5}
mkdir -p "$OUT"
log(){ echo "$(date -u +%T) $*" | tee -a "$OUT/campaign.log"; }
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
for item in "$@"; do
  model=${item%%:*}; n=${item##*:}; tag=$model-L$n
  [ -e "$OUT/$tag.jsonl" ] && { log "$tag exists, skipped"; continue; }
  t0=$(date +%s)
  while :; do
    tc=$(temp); load=$(cut -d' ' -f1 /proc/loadavg); waited=$(( $(date +%s) - t0 ))
    ok=$(python3 -c "print(int($tc <= $COOL_C and $load <= $MAX_LOAD))")
    if [ "$ok" = 1 ] && [ $waited -ge $MIN_COOL_S ]; then cool=1; break; fi
    if [ $waited -ge $MAX_COOL_S ]; then cool=0; break; fi
    sleep 5
  done
  log "$tag start cool=$cool waited=${waited}s temp_mc=$tc load=$load threads=$THREADS"
  rm -f "$OUT/$tag.stop"
  python3 "$HERE/host_sampler.py" "$OUT/$tag.power" "$OUT/$tag.stop" 0.1 &
  spid=$!
  sleep 5
  CUDA_VISIBLE_DEVICES="" "$BENCH" --shard "$SH/$model/HTP0.ffn.gguf" --layers $n --threads $THREADS \
      --idle-s $IDLE --warmup-s $WARM --seconds $SECS --out "$OUT/$tag.jsonl" 2> "$OUT/$tag.stderr"
  rc=$?
  sleep 2
  touch "$OUT/$tag.stop"; wait $spid
  lbytes=$(python3 -c "import json; print(json.load(open('$SH/$model/HTP0.ffn.json'))['layer_bytes_f16'])")
  echo "{\"tag\":\"$tag\",\"model\":\"$model\",\"layers\":$n,\"layer_bytes\":$lbytes,\"threads\":$THREADS,\"cool\":$cool,\"cool_wait_s\":$waited,\"pre_temp_mc\":$tc,\"pre_load\":$load,\"rc\":$rc}" > "$OUT/$tag.json"
  log "$tag done rc=$rc"
done
log CAMPAIGN_DONE
