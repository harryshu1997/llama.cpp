#!/usr/bin/env bash
# Greedy GENERATION across the 3-device USB pipeline: op15[0,k2) -> op12[k2,k3) -> server[k3,n).
# Stages are stateless (no persistent KV), so each step re-prefills the full sequence and feeds the
# exact token ids back via --tokens-file. Correct greedy decoding; O(N^2) + a model reload per stage.
set -euo pipefail
OP15=${OP15:?set OP15 to the adb serial for the first phone}
OP12=${OP12:?set OP12 to the adb serial for the second phone}
K2=${K2:-2}; K3=${K3:-3}; NGEN=${NGEN:-8}
PROMPT=${PROMPT:-"The quick brown fox jumps over the lazy dog and then"}
PDEV=${PDEV:-CPU}; PNGL=${PNGL:-0}
HOST_BIN=${HOST_BIN:?}; HOST_MODEL=${HOST_MODEL:?}
OP15_MODEL=${OP15_MODEL:-/data/local/tmp/unifer/llamacpp/gemma-4-E2B-it-Q4_0.gguf}
OP12_MODEL=${OP12_MODEL:-/data/local/tmp/llamacpp_models/gemma-4-E2B-it-Q4_0.gguf}
RDIR=/data/local/tmp/ls-npu; W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
export LD_LIBRARY_PATH="$(dirname "$HOST_BIN")"

ptoks() { python3 -c "import struct,sys;d=open(sys.argv[1],'rb').read();ne,N=struct.unpack('<ii',d[:8]);print(' '.join(map(str,struct.unpack('<%di'%N,d[8:8+4*N]))))" "$1"; }

op15_head() { # $1 = extra arg (-p or --tokens-file), $2 = value
  adb -s $OP15 shell "cd $RDIR && LD_LIBRARY_PATH=. LLAMA_LAYER_END=$K2 ./llama-layersplit -m $OP15_MODEL --devices $PDEV -ngl $PNGL $1 \"$2\" --mode head --act-file w1.bin >/dev/null 2>&1"
  adb -s $OP15 pull $RDIR/w1.bin "$W/w1.bin" >/dev/null 2>&1; }
forward() { # w1.bin on host -> op12 mid -> server tail ; echo ARGMAX line
  adb -s $OP12 push "$W/w1.bin" $RDIR/w1.bin >/dev/null 2>&1
  adb -s $OP12 shell "cd $RDIR && LD_LIBRARY_PATH=. LLAMA_LAYER_START=$K2 LLAMA_LAYER_END=$K3 ./llama-layersplit -m $OP12_MODEL --devices $PDEV -ngl $PNGL --mode mid --act-file w1.bin --act-out w2.bin >/dev/null 2>&1"
  adb -s $OP12 pull $RDIR/w2.bin "$W/w2.bin" >/dev/null 2>&1
  LLAMA_LAYER_START=$K3 "$HOST_BIN" -m "$HOST_MODEL" -ngl 0 --mode tail --act-file "$W/w2.bin" 2>/dev/null | grep ARGMAX; }

echo "== GEN over op15[0,$K2)->op12[$K2,$K3)->server[$K3,n) | prompt: \"$PROMPT\" =="
op15_head "-p" "$PROMPT"
SEQ=$(ptoks "$W/w1.bin"); echo "prompt = $(echo $SEQ | wc -w) tokens"
printf '\nOUTPUT: %s' "$PROMPT"
for g in $(seq 1 $NGEN); do
  LINE=$(forward); ID=$(echo "$LINE" | grep -oP 'id=\K[0-9]+'); PIECE=$(echo "$LINE" | sed "s/.*piece='//; s/'\$//")
  printf '%s' "$PIECE"; SEQ="$SEQ $ID"
  echo "$SEQ" > "$W/seq.txt"; adb -s $OP15 push "$W/seq.txt" $RDIR/seq.txt >/dev/null 2>&1
  op15_head "--tokens-file" "seq.txt"
done
printf '\n'
