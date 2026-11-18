#!/system/bin/sh
# quick burst-only replay against the OpenCL worker (port 7075)
B=/data/local/tmp/s43-pixel-cpugpu-20260925-v1; T=/data/local/tmp/s43-pixel-cl-ab; ART=sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718
mkdir -p $T/runs/quick
taskset 40 $B/pixel-ffn-replay --port 7075 --artifact $ART --layers 18,19,20,21,22,23 --columns 17408 --max-tokens 4 --input-dir $B/inputs --out $T/runs/quick/replay --segment burst1:1:120:5:0:0 --segment burst4:4:60:5:0:0 2>&1 | grep done
