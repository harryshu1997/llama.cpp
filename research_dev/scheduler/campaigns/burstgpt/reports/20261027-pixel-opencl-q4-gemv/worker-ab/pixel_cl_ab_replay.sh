#!/system/bin/sh
# Replay recorded layer-18..23 activations against the CPU worker (7074) and the OpenCL worker (7075).
B=/data/local/tmp/s43-pixel-cpugpu-20260925-v1
T=/data/local/tmp/s43-pixel-cl-ab
ART=sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718
SEG="--segment burst1:1:40:5:0:0 --segment prod1:1:40:5:2500:150000 --segment burst2:2:30:5:0:0 --segment burst4:4:30:5:0:0 --segment prod4:4:30:5:2500:150000"
for p in 7074:cpu 7075:cl; do
  port=${p%%:*}; name=${p##*:}
  mkdir -p $T/runs/$name; rm -f $T/runs/$name/*
  echo "=== replay vs $name"
  taskset 40 $B/pixel-ffn-replay --port $port --artifact $ART --layers 18,19,20,21,22,23 --columns 17408 --max-tokens 4 --input-dir $B/inputs --out $T/runs/$name/replay $SEG 2>&1 | tail -4
done
ls -la $T/runs/cpu $T/runs/cl
