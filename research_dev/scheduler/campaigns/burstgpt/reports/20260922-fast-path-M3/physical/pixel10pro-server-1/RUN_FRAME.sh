#!/usr/bin/env bash
cd /mnt/storage/s42-pixel10pro-server-20260922-v1 || exit 2
adb -P 5037 -s 5A040DLCH004ES shell -n 'mkdir /data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1' || exit 3
adb -P 5037 -s 5A040DLCH004ES push software/llama-ffn-split-worker /data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1/llama-ffn-split-worker || exit 4
adb -P 5037 -s 5A040DLCH004ES shell -n 'chmod 755 /data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1/llama-ffn-split-worker' || exit 5
for lib in libggml.so libggml-base.so libggml-cpu.so libggml-vulkan.so; do
 adb -P 5037 -s 5A040DLCH004ES shell -n "ln -s /data/local/tmp/s42-pixel10pro-qualification-20260922-v1/$lib /data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1/$lib" || exit 6
done
export LD_LIBRARY_PATH=/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
python3 -u qualify_op11_tcp.py --output run2-coalesced4352 \
 --worker /mnt/storage/s42-trace-v2-20260921-prep/cuda-build/bin/llama-ffn-split-worker \
 --model /home/zhihao/s42-op11-qwen-shards-20260921-v1/qwen/HTP0.ffn.gguf \
 --artifact-sha256 sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718 \
 --serial 5A040DLCH004ES --phone-label Pixel10Pro --phone-backend Vulkan0 \
 --phone-dir /data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1 \
 --phone-model /data/local/tmp/s42-pixel10pro-qualification-20260922-v1/HTP0.ffn.gguf \
 --phone-library libggml.so --phone-library libggml-base.so --phone-library libggml-cpu.so --phone-library libggml-vulkan.so \
 --layers 18 --quantum 4352 --cpu-port 26977 --candidate-port 26978 </dev/null > FRAME.log 2>&1
run_status=$?
python3 - "$run_status" <<'PY'
import json,sys,datetime
from pathlib import Path
Path('FRAME_DONE.json').write_text(json.dumps({'exit_code':int(sys.argv[1]),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
PY
exit "$run_status"
