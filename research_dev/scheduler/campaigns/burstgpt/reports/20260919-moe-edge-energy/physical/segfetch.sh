#!/bin/bash
# Segmented download: 8 ranges in parallel, each in its own retry loop, then sequential append.
cd /home/zhihao/moe-energy-20260919
URL="https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf"
FINAL="Qwen3-30B-A3B-Q4_K_M.gguf"
TOTAL=18556685824
N=8
SEG=$(( (TOTAL + N - 1) / N ))
if [ -f "$FINAL.part" ] && [ ! -f seg0 ]; then
  truncate -s $SEG "$FINAL.part"; mv "$FINAL.part" seg0
fi
fetch_seg() {
  i=$1; lo=$(( i * SEG )); hi=$(( (i + 1) * SEG )); [ $hi -gt $TOTAL ] && hi=$TOTAL; len=$(( hi - lo ))
  for a in $(seq 1 300); do
    cur=$(stat -c %s seg$i 2>/dev/null || echo 0)
    if [ $cur -ge $len ]; then truncate -s $len seg$i; echo "seg$i done $(date +%T)" >> segfetch.log; return 0; fi
    curl -sS -L --http1.1 -r $(( lo + cur ))-$(( hi - 1 )) --speed-limit 100000 --speed-time 20 --connect-timeout 20 "$URL" >> seg$i
    echo "seg$i attempt $a rc=$? size=$(stat -c %s seg$i) $(date +%T)" >> segfetch.log
    sleep 2
  done
  return 1
}
for i in $(seq 0 $((N-1))); do fetch_seg $i & done
wait
for i in $(seq 0 $((N-1))); do
  lo=$(( i * SEG )); hi=$(( (i + 1) * SEG )); [ $hi -gt $TOTAL ] && hi=$TOTAL
  [ "$(stat -c %s seg$i)" = "$(( hi - lo ))" ] || { echo "seg$i incomplete" >> segfetch.log; exit 1; }
done
for i in $(seq 1 $((N-1))); do cat seg$i >> seg0 && rm seg$i; done
if [ "$(stat -c %s seg0)" = "$TOTAL" ]; then mv seg0 "$FINAL"; echo ALL_DONE >> segfetch.log; else echo SIZE_MISMATCH >> segfetch.log; fi
