#!/usr/bin/env bash

qualification_root=/mnt/storage/s42-op11-qualification-20260922-v1
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$qualification_root/WAITING_FOR_LOCK.txt"
flock -w 900 -E 75 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock \
    env LANG=C.UTF-8 \
    LD_LIBRARY_PATH=/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64 \
    python3 "$qualification_root/qualify_op11_tcp.py" \
    --output "$qualification_root/run1" \
    --worker /mnt/storage/s42-trace-v2-20260921-prep/cuda-build/bin/llama-ffn-split-worker \
    --model /home/zhihao/s42-op11-qwen-shards-20260921-v1/qwen/HTP0.ffn.gguf \
    --artifact-sha256 sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718 \
    --serial 832358d4 > "$qualification_root/RUN.log" 2>&1
qualification_status=$?
printf '%s\n' "$qualification_status" > "$qualification_root/EXIT_STATUS.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$qualification_root/FINISHED.txt"
exit "$qualification_status"
