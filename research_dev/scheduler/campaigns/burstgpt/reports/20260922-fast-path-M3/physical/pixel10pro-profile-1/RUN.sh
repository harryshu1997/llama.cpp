#!/usr/bin/env bash
cd /mnt/storage/s42-pixel10pro-profile-20260922-v1 || exit 2
export LANG=C.UTF-8
export LD_LIBRARY_PATH=/mnt/storage/s42-trace-v2-20260921-prep/cuda-build/bin:/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
python3 -u profile_pixel_ffn.py --software software --output run1 </dev/null > RUN.log 2>&1
profile_status=$?
python3 - "$profile_status" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

Path('DONE.json').write_text(json.dumps({
    'exit_code': int(sys.argv[1]),
    'finished_utc': datetime.now(timezone.utc).isoformat(),
}, indent=2) + '\n')
PY
exit "$profile_status"
