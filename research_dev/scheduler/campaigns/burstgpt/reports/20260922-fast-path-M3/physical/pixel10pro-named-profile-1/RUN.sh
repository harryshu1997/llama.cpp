#!/usr/bin/env bash
cd /mnt/storage/s42-pixel10pro-named-profile-20260923-v1 || exit 2
export LANG=C.UTF-8
export LD_LIBRARY_PATH=/mnt/storage/s42-trace-v2-20260921-prep/cuda-build/bin:/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
python3 -u profile_pixel_named.py --software software --output run1 </dev/null > RUN.log 2>&1
profile_status=$?
python3 - "$profile_status" <<'END_MARKER'
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
Path('DONE.json').write_text(json.dumps({'exit_code':int(sys.argv[1]),'finished_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
END_MARKER
exit "$profile_status"
