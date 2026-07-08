#!/usr/bin/env bash
# Persistent 3-stage pipeline over adb-forwarded TCP (USB). KV-resident stages, incremental decode.
#   op15 stagenet[0,k2] :5555  |  op12 stagenet[k2,k3] :5556  |  host pipedriver tail[k3,n)
set -uo pipefail
OP15=${OP15:-3C15AU002CL00000}; OP12=${OP12:-5ae7a43d}
K2=${K2:-2}; K3=${K3:-3}; NGEN=${NGEN:-12}
PROMPT=${PROMPT:-"The quick brown fox jumps over the lazy dog and then"}
PDEV=${PDEV:-CPU}; PNGL=${PNGL:-0}
HNGL=${HNGL:-99}   # host tail offload: 99 = A6000/CUDA terminal (system design), 0 = CPU
HOST_BIN=${HOST_BIN:?}; HOST_MODEL=${HOST_MODEL:?}
M15=${M15:-/data/local/tmp/unifer/llamacpp/gemma-4-E2B-it-Q4_0.gguf}
M12=${M12:-/data/local/tmp/llamacpp_models/gemma-4-E2B-it-Q4_0.gguf}
PA=15555; PB=15556; RDIR=/data/local/tmp/ls-npu
L=/tmp/claude-1761612022/-home-myid-zs89458-Documents-llama-cpp-release/dec7565c-0844-4b93-9a46-de8a984e4d9c/scratchpad
export LD_LIBRARY_PATH="$(dirname "$HOST_BIN")"

cleanup(){ adb -s $OP15 forward --remove tcp:$PA 2>/dev/null; adb -s $OP12 forward --remove tcp:$PB 2>/dev/null;
           adb -s $OP15 shell 'pkill -9 -f layersplit' 2>/dev/null; adb -s $OP12 shell 'pkill -9 -f layersplit' 2>/dev/null;
           kill ${J15:-0} ${J12:-0} 2>/dev/null; }
trap cleanup EXIT

adb -s $OP15 shell 'pkill -9 -f layersplit' 2>/dev/null; adb -s $OP12 shell 'pkill -9 -f layersplit' 2>/dev/null
adb -s $OP15 forward --remove tcp:$PA 2>/dev/null; adb -s $OP12 forward --remove tcp:$PB 2>/dev/null
sleep 1
adb -s $OP15 forward tcp:$PA tcp:$PA >/dev/null
adb -s $OP12 forward tcp:$PB tcp:$PB >/dev/null
echo "launching stagenets (engine=$PDEV)..."
adb -s $OP15 shell "cd $RDIR && LD_LIBRARY_PATH=. LLAMA_LAYER_END=$K2 ./llama-layersplit -m $M15 --devices $PDEV -ngl $PNGL --mode stagenet --port $PA" > $L/s15.log 2>&1 & J15=$!
adb -s $OP12 shell "cd $RDIR && LD_LIBRARY_PATH=. LLAMA_LAYER_START=$K2 LLAMA_LAYER_END=$K3 ./llama-layersplit -m $M12 --devices $PDEV -ngl $PNGL --mode stagenet --port $PB" > $L/s12.log 2>&1 & J12=$!

ready=0
for i in $(seq 1 60); do
  if grep -q "stagenet] listening" $L/s15.log 2>/dev/null && grep -q "stagenet] listening" $L/s12.log 2>/dev/null; then ready=1; break; fi
  if grep -q "bind(" $L/s15.log $L/s12.log 2>/dev/null; then echo "bind failed:"; grep "bind(" $L/s15.log $L/s12.log; exit 1; fi
  sleep 1
done
[ $ready -eq 1 ] || { echo "stages not listening:"; tail -n 3 $L/s15.log; echo "---"; tail -n 3 $L/s12.log; exit 1; }

echo "both stages listening. driving decode ($NGEN tokens)..."
t0=$(date +%s%3N)
LLAMA_LAYER_START=$K3 "$HOST_BIN" -m "$HOST_MODEL" -ngl $HNGL --mode pipedriver --host 127.0.0.1 --port $PA --port2 $PB ${CHAT:+--chat} -p "$PROMPT" -n $NGEN 2> >(grep '\[pipedriver\] per-tok\|stageA\|stageB\|tail \|total decode\|chat=' >&2)
t1=$(date +%s%3N)
echo "[pipeline] $NGEN tokens in $((t1-t0)) ms = $(( (t1-t0)/NGEN )) ms/tok (incl prompt prefill + USB RTT/tok)"
