#!/bin/bash
# OnePlus 15 NPU legs of the weight-stream energy benchmark.
#   run_op15.sh OUTDIR MODEL:N [MODEL:N ...]      (env: IDLE WARM SECS COOL_MC MIN_COOL_S MAX_COOL_S)
# Per run: wait until the NPU/DDR zones are back under COOL_MC, run phone_run.sh as root (sampler, worker(s),
# device-local client), pull the run directory. Read-only on the phone outside /data/local/tmp/stream-energy-20261110.
set -u
A="adb -s 3C15AU002CL00000"
SH=/mnt/data_h16t/zs89458/xband-data/models/stream-energy-shards
T=/data/local/tmp/stream-energy-20261110
OUT=$1; shift
IDLE=${IDLE:-12}; WARM=${WARM:-2}; SECS=${SECS:-20}
# cool = NPU zone <= COOL_MC, back shell <= SHELL_MC (the vendor policy caps the CPU from ~38.2 C shell), no CPU cap
COOL_MC=${COOL_MC:-43000}; SHELL_MC=${SHELL_MC:-36300}; MIN_COOL_S=${MIN_COOL_S:-45}; MAX_COOL_S=${MAX_COOL_S:-1500}
SESSION_MAX=${SESSION_MAX:-3300000000}
mkdir -p "$OUT"
log(){ echo "$(date -u +%T) $*" | tee -a "$OUT/campaign.log"; }
for item in "$@"; do
  model=${item%%:*}; n=${item##*:}
  tag=$model-L$n
  [ -e "$OUT/$tag" ] && { log "$tag exists, skipped"; continue; }
  read -r art d nff arch lbytes < <(python3 -c "
import json; m=json.load(open('$SH/$model/HTP0.ffn.json'))
print(m['parent_sha256'], m['n_embd'], m['n_ff'], m['shard_architecture'], m['layer_bytes_f16'])")
  geglu=0; [ "$arch" = gemma4 ] && geglu=1
  per=$(( SESSION_MAX / lbytes )); [ $per -gt $n ] && per=$n
  groups=""; start=0; port=27411
  while [ $start -lt $n ]; do
    end=$(( start + per - 1 )); [ $end -ge $n ] && end=$(( n - 1 ))
    list=$(seq -s, $start $end)
    groups="$groups $port:$start-$end:$list"; start=$(( end + 1 )); port=$(( port + 1 ))
  done
  t0=$(date +%s)
  while :; do
    line=$($A shell "su -c 'sh $T/phone_cool.sh'" < /dev/null | tr -d '\r')
    npu=$(echo "$line" | sed -n 's/.*nsphvx-0=\([0-9]*\).*/\1/p'); shell=$(echo "$line" | sed -n 's/.*shell_back=\([0-9]*\).*/\1/p')
    uncapped=$(echo "$line" | sed -n 's/.*cpu_uncapped=\([01]\).*/\1/p')
    waited=$(( $(date +%s) - t0 ))
    if [ -n "$npu" ] && [ -n "$shell" ] && [ "$npu" -le "$COOL_MC" ] && [ "$shell" -le "$SHELL_MC" ] && [ "$uncapped" = 1 ] && [ $waited -ge $MIN_COOL_S ]; then cool=1; break; fi
    if [ $waited -ge $MAX_COOL_S ]; then cool=0; break; fi
    sleep 15
  done
  log "$tag start cool=$cool waited=${waited}s state: $line groups:$groups"
  timeout 300 $A shell "su -c 'sh $T/phone_run.sh $tag $art $d $nff $geglu $IDLE $WARM $SECS $T/shards/$model/HTP0.ffn.gguf$groups'" < /dev/null > "$OUT/$tag.stdout" 2>&1
  rc=$?
  $A pull $T/runs/$tag "$OUT/$tag" > /dev/null 2>&1
  echo "{\"tag\":\"$tag\",\"model\":\"$model\",\"layers\":$n,\"layer_bytes\":$lbytes,\"n_embd\":$d,\"n_ff\":$nff,\"groups\":\"$groups\",\"cool\":$cool,\"cool_wait_s\":$waited,\"pre_state\":\"$line\",\"rc\":$rc}" > "$OUT/$tag.json"
  log "$tag done rc=$rc $(tail -n 2 "$OUT/$tag.stdout" | tr '\n' ' ')"
done
log CAMPAIGN_DONE
