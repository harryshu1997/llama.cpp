#!/usr/bin/env bash

qualification_root=/mnt/storage/s42-op11-v73-sync-20260922-v1
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$qualification_root/WAITING_FOR_LOCK.txt"
flock -w 900 -E 75 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock \
    env LANG=C.UTF-8 \
    LD_LIBRARY_PATH=/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64 \
    bash "$qualification_root/UNDER_LOCK.sh" > "$qualification_root/RUN.log" 2>&1
qualification_status=$?
printf '%s\n' "$qualification_status" > "$qualification_root/EXIT_STATUS.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$qualification_root/FINISHED.txt"
exit "$qualification_status"
