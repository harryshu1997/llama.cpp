#!/usr/bin/env bash
# 3-device Plan-A pipeline over USB (adb): op15[0,k2) -> op12[k2,k3) -> server[k3,n_layer).
# Phones relay through the server (USB/adb); the server is the terminal stage (lm_head+sample).
set -euo pipefail

OP15=${OP15:?set OP15 to the adb serial for the first phone}
OP12=${OP12:?set OP12 to the adb serial for the second phone}
K2=${K2:-2}; K3=${K3:-3}                 # op15 owns [0,K2), op12 owns [K2,K3), server owns [K3,n)
PROMPT=${PROMPT:-"The quick brown fox jumps over the lazy dog and then"}
PDEV=${PDEV:-CPU}; PNGL=${PNGL:-0}       # phone engine (CPU/HTP0/GPUOpenCL) + -ngl

HOST_BIN=${HOST_BIN:?set HOST_BIN to host llama-layersplit}
HOST_MODEL=${HOST_MODEL:?set HOST_MODEL}
OP15_MODEL=${OP15_MODEL:-/data/local/tmp/unifer/llamacpp/gemma-4-E2B-it-Q4_0.gguf}
OP12_MODEL=${OP12_MODEL:-/data/local/tmp/llamacpp_models/gemma-4-E2B-it-Q4_0.gguf}
RDIR=/data/local/tmp/ls-npu
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
ms() { date +%s%3N; }

echo "== 3-device pipeline: op15[0,$K2) -> op12[$K2,$K3) -> server[$K3,n)  engine=$PDEV =="

t0=$(ms)
# --- stage 1: op15 head [0,K2) ---
adb -s "$OP15" shell "cd $RDIR && LD_LIBRARY_PATH=. LLAMA_LAYER_END=$K2 ./llama-layersplit -m $OP15_MODEL --devices $PDEV -ngl $PNGL -p '$PROMPT' --mode head --act-file w1.bin >/dev/null 2>&1"
adb -s "$OP15" pull $RDIR/w1.bin "$W/w1.bin" >/dev/null 2>&1
t1=$(ms); echo "  [op15 head]  $((t1-t0)) ms   ($(stat -c%s "$W/w1.bin") B)"

# --- stage 2: op12 mid [K2,K3] (relayed through server) ---
adb -s "$OP12" push "$W/w1.bin" $RDIR/w1.bin >/dev/null 2>&1
adb -s "$OP12" shell "cd $RDIR && LD_LIBRARY_PATH=. LLAMA_LAYER_START=$K2 LLAMA_LAYER_END=$K3 ./llama-layersplit -m $OP12_MODEL --devices $PDEV -ngl $PNGL --mode mid --act-file w1.bin --act-out w2.bin >/dev/null 2>&1"
adb -s "$OP12" pull $RDIR/w2.bin "$W/w2.bin" >/dev/null 2>&1
t2=$(ms); echo "  [op12 mid ]  $((t2-t1)) ms   ($(stat -c%s "$W/w2.bin") B)"

# --- stage 3: server tail [K3,n) ---
export LD_LIBRARY_PATH="$(dirname "$HOST_BIN")"
PIPE=$(LLAMA_LAYER_START=$K3 "$HOST_BIN" -m "$HOST_MODEL" -ngl 0 --mode tail --act-file "$W/w2.bin" 2>/dev/null | grep ARGMAX)
t3=$(ms); echo "  [server tail] $((t3-t2)) ms"
echo "  PIPELINE next-token : $PIPE"
echo "  total end-to-end    : $((t3-t0)) ms"

# --- reference: whole model on the server ---
REF=$("$HOST_BIN" -m "$HOST_MODEL" -ngl 0 -p "$PROMPT" --mode mono 2>/dev/null | grep ARGMAX)
echo "  server MONO (ref)   : $REF"
[ "$(echo "$PIPE" | grep -oP 'id=\K[0-9]+')" = "$(echo "$REF" | grep -oP 'id=\K[0-9]+')" ] \
  && echo "  => TOP-1 MATCH" || echo "  => TOP-1 MISMATCH"
