#!/usr/bin/env bash
cd /mnt/storage/s42-pixel10pro-server-20260922-v1 || exit 2
export S42_UNIFIED_REPO_ROOT=/mnt/storage/s42-trace-v2-20260921-prep/source
export LD_LIBRARY_PATH=/mnt/storage/s42-trace-v2-20260921-prep/cuda-build/bin:/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
export LANG=C.UTF-8
python3 -u qualify_op11_tcp.py --output run4-six-layer-qualification \
 --worker /mnt/storage/s42-trace-v2-20260921-prep/cuda-build/bin/llama-ffn-split-worker \
 --model /home/zhihao/s42-op11-qwen-shards-20260921-v1/qwen/HTP0.ffn.gguf \
 --artifact-sha256 sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718 \
 --serial 5A040DLCH004ES --phone-label Pixel10Pro --phone-backend Vulkan0 \
 --phone-dir /data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1 \
 --phone-model /data/local/tmp/s42-pixel10pro-qualification-20260922-v1/HTP0.ffn.gguf \
 --phone-library libggml.so --phone-library libggml-base.so --phone-library libggml-cpu.so --phone-library libggml-vulkan.so \
 --layers 18 19 20 21 22 23 --quantum 4352 --cpu-port 26977 --candidate-port 26978 </dev/null > SIX_QUAL.log 2>&1
qual_status=$?
python3 - "$qual_status" <<'PY'
import json,sys,datetime
from pathlib import Path
Path('SIX_QUAL_DONE.json').write_text(json.dumps({'exit_code':int(sys.argv[1]),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
PY
if [ "$qual_status" -ne 0 ]; then exit "$qual_status"; fi
python3 -u qualify_pixel_server.py --config PIXEL_SERVER_SIX_LAYER_CONFIG.json --output run5-six-layer-server </dev/null > SIX_SERVER.log 2>&1
run_status=$?
python3 - "$run_status" <<'PY'
import json,sys,datetime
from pathlib import Path
Path('SIX_SERVER_DONE.json').write_text(json.dumps({'exit_code':int(sys.argv[1]),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
PY
exit "$run_status"
