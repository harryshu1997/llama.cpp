#!/bin/bash
# Push rebuilt worker/client into the bench's own phone dir and print hashes.
set -euo pipefail
A="adb -P 5037 -s 5A040DLCH004ES"
HOST=/mnt/storage/s43-pixel-cpugpu-20260925-v1
PH=/data/local/tmp/s43-pixel-cpugpu-20260925-v1
$A shell "ps -A -o PID,ARGS | grep '[l]lama-ffn'" < /dev/null && { echo "worker running, refusing"; exit 3; }
$A push $HOST/bin/llama-ffn-split-worker $HOST/bin/pixel-ffn-replay $PH/ > /dev/null
$A shell "chmod 755 $PH/llama-ffn-split-worker $PH/pixel-ffn-replay && sha256sum $PH/llama-ffn-split-worker $PH/pixel-ffn-replay" < /dev/null
