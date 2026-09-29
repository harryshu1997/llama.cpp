#!/bin/bash
cd /mnt/storage/s42-pixel10pro-gemv-tune-20260923-v1 || exit 2
export LANG=C.UTF-8
export S42_UNIFIED_REPO_ROOT=/mnt/storage/s42-trace-v2-20260921-prep/source
export LD_LIBRARY_PATH=/mnt/storage/s42-trace-v2-20260921-prep/cuda-build/bin:/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
python3 -u tune_pixel_gemv.py --output run1 --software software --config PIXEL_GEMV_SWEEP_CONFIG.json </dev/null > RUN.log 2>&1
run_status=$?
python3 - "$run_status" <<'DONE'
from datetime import datetime, timezone
import json,sys
from pathlib import Path
with Path("DONE.json").open("x") as f:
    json.dump({"exit_code":int(sys.argv[1]),"at":datetime.now(timezone.utc).isoformat()},f,indent=2)
DONE
exit "$run_status"
