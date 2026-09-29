#!/usr/bin/env bash
set -euo pipefail
deploy=/mnt/storage/s42-fast-path-M2-diagnostic-20260920-a81c09
mkdir -p "$deploy/physical/step2"
printf '%s\n' "$$" > "$deploy/physical/step2/MATRIX_DRIVER.pid"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$deploy/native-source
for lengths in 256,256 257,257 256,257 257,256; do
    for order in normal reversed; do
        name=p${lengths//,/-}-$order
        root=$deploy/physical/step2/$name
        for arm in control combined; do
            bash "$deploy/RUN_DIAGNOSTIC.sh" "step2/$name/$arm" "$arm" "$lengths" "$deploy/config/$name.json"
        done
        set +e
        python3 "$deploy/ANALYZE.py" --matrix-case "$root" --n 2 > "$root/CHECK_PAIR.log" 2>&1
        result=$?
        set -e
        if [[ "$result" -gt 1 ]]; then exit "$result"; fi
        python3 - "$root/CHECK_PAIR.json" <<'RESULT'
import json
from pathlib import Path
import sys
path=Path(sys.argv[1])
check=json.loads(path.read_text())
print(path.parent.name, check['status'], check.get('token_comparisons'), flush=True)
if check['errors'] and check['errors'] != ['slot tokens differ from matched host']:
    raise SystemExit(2)
RESULT
    done
done
