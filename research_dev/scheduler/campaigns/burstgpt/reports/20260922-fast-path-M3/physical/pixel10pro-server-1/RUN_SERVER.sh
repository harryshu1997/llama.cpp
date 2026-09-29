#!/usr/bin/env bash
cd /mnt/storage/s42-pixel10pro-server-20260922-v1 || exit 2
export S42_UNIFIED_REPO_ROOT=/mnt/storage/s42-trace-v2-20260921-prep/source
export LD_LIBRARY_PATH=/mnt/storage/s42-trace-v2-20260921-prep/cuda-build/bin:/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
export LANG=C.UTF-8
python3 -u qualify_pixel_server.py --config PIXEL_SERVER_CONFIG.json --output run3-server </dev/null > SERVER_RUN.log 2>&1
run_status=$?
python3 - "$run_status" <<'PY'
import json,sys,datetime
from pathlib import Path
Path('SERVER_DONE.json').write_text(json.dumps({'exit_code':int(sys.argv[1]),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
PY
exit "$run_status"
