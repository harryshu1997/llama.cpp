#!/bin/bash
# Stage the CPU+GPU bench on the Pixel into NEW dirs only. Run on the desktop under the rig lock.
set -euo pipefail
A="adb -P 5037 -s 5A040DLCH004ES"
HOST=/mnt/storage/s43-pixel-cpugpu-20260925-v1
PH=/data/local/tmp/s43-pixel-cpugpu-20260925-v1
SRC=/data/local/tmp/s42-pixel10pro-aoa-20260924-v1
PROD=/data/local/tmp/s42-pixel10pro-packed-server-20260924-v1
if $A shell "test -e $PH" < /dev/null; then echo "phone dir exists"; exit 3; fi
$A shell "mkdir $PH $PH/prod $PH/inputs $PH/runs && cp $SRC/libggml.so $SRC/libggml-base.so $SRC/libggml-cpu.so $SRC/libggml-vulkan.so $SRC/QWEN_PACKED.ffn.gguf $PH/ && cp $PROD/llama-ffn-split-worker $PH/prod/" < /dev/null
$A push $HOST/bin/llama-ffn-split-worker $HOST/bin/pixel-ffn-replay $PH/ > /dev/null
$A push $HOST/inputs/. $PH/inputs/ > /dev/null
$A shell "chmod 755 $PH/llama-ffn-split-worker $PH/pixel-ffn-replay $PH/prod/llama-ffn-split-worker && cd $PH && sha256sum libggml.so libggml-base.so libggml-cpu.so libggml-vulkan.so QWEN_PACKED.ffn.gguf prod/llama-ffn-split-worker llama-ffn-split-worker pixel-ffn-replay inputs/*" < /dev/null > $HOST/PHONE_SHA256.txt
cat $HOST/PHONE_SHA256.txt | head -8
$A shell "df -h /data | tail -1" < /dev/null
