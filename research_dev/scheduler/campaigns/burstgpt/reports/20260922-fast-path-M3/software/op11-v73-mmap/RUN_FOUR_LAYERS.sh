#!/usr/bin/env bash

qualification_root=/mnt/storage/s42-op11-v73-mmap-20260922-v1
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$qualification_root/WAITING_FOUR_LAYERS.txt"
flock -w 900 -E 75 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock \
    env LANG=C.UTF-8 \
    LD_LIBRARY_PATH=/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64 \
    python3 "$qualification_root/qualify_op11_tcp.py" \
    --output "$qualification_root/run2-layers18-21" \
    --worker /mnt/storage/s42-trace-v2-20260921-prep/cuda-build/bin/llama-ffn-split-worker \
    --model /home/zhihao/s42-op11-qwen-shards-20260921-v1/qwen/HTP0.ffn.gguf \
    --artifact-sha256 sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718 \
    --serial 832358d4 --phone-dir /data/local/tmp/s42-op11-v73-mmap-20260922-v1 \
    --phone-backend HTP0 --phone-nhmx 0 \
    --layers 18 19 20 21 --cpu-port 26957 --candidate-port 26958 \
    > "$qualification_root/RUN_FOUR_LAYERS.log" 2>&1
qualification_status=$?
printf '%s\n' "$qualification_status" > "$qualification_root/FOUR_LAYERS_EXIT_STATUS.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$qualification_root/FOUR_LAYERS_FINISHED.txt"
exit "$qualification_status"
