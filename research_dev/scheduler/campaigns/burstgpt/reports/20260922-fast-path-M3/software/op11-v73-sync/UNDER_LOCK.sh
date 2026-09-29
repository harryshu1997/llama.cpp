#!/usr/bin/env bash

qualification_root=/mnt/storage/s42-op11-v73-sync-20260922-v1
phone_root=/data/local/tmp/s42-op11-v73-sync-20260922-v1
old_phone_root=/data/local/tmp/s42-op11-20260921-bin

adb -P 5037 -s 832358d4 shell "test ! -e $phone_root && mkdir $phone_root && cp $old_phone_root/llama-ffn-split-worker $old_phone_root/libggml.so $old_phone_root/libggml-base.so $old_phone_root/libggml-cpu.so $old_phone_root/libggml-opencl.so $old_phone_root/libggml-hexagon.so $old_phone_root/libllama.so $old_phone_root/libllama-common.so $old_phone_root/libomp.so $phone_root/" || exit 1
adb -P 5037 -s 832358d4 push "$qualification_root/libggml-htp-v73.so" "$phone_root/libggml-htp-v73.so" || exit 1
adb -P 5037 -s 832358d4 shell "sha256sum $phone_root/libggml-htp-v73.so" > "$qualification_root/DEPLOYED_SHA256.txt" || exit 1

python3 "$qualification_root/qualify_op11_tcp.py" \
    --output "$qualification_root/run1-layer18" \
    --worker /mnt/storage/s42-trace-v2-20260921-prep/cuda-build/bin/llama-ffn-split-worker \
    --model /home/zhihao/s42-op11-qwen-shards-20260921-v1/qwen/HTP0.ffn.gguf \
    --artifact-sha256 sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718 \
    --serial 832358d4 --phone-dir "$phone_root" --phone-backend HTP0 --phone-nhmx 0 \
    --layers 18 --cpu-port 26953 --candidate-port 26954
qualification_status=$?
printf '%s\n' "$qualification_status" > "$qualification_root/RESULT_STATUS.txt"
exit "$qualification_status"
