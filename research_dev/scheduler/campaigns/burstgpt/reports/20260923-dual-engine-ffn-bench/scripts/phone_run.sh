#!/bin/bash
# usage: phone_run.sh <tag> <bench args...>   (run on desktop, inside flock)
set -u
S=3C15AU002CL00000
A="adb -P 5037 -s $S"
R=/data/local/tmp/s43-dual-ffn-bench-20260923
tag=$1; shift
if $A shell 'ps -A' | grep -E "[f]fn-split" ; then echo "PRODUCTION FFN WORKER RUNNING - abort"; exit 9; fi
$A shell "cd $R && ${EXTRA_ENV:-} LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_NDEV=1 GGML_HEXAGON_MBUF=4192 GGML_HEXAGON_VMEM=3328 GGML_HEXAGON_NHVX=4 ./llama-ffn-dual-bench --model /data/local/tmp/s41-opoffload-dmabuf-v1/Qwen3-14B-Q4KM-dequant-f16.gguf --samples $R/out/$tag.csv --dump $R/out/$tag $*; echo EXIT=\$?"
