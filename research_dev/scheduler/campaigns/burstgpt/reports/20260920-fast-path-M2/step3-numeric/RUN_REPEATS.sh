#!/usr/bin/env bash
set -euo pipefail
deploy=/mnt/storage/s42-fast-path-M2-numeric-20260920-2d61e7
cd "$deploy/native-source"
export PYTHONDONTWRITEBYTECODE=1
printf '%s\n' "$$" > "$deploy/physical/REPEATS_DRIVER.pid"
for run in run1 run2; do
    bash "$deploy/RUN_DIAGNOSTIC.sh" "$run" combined 256,257 "$deploy/config/n2.json"
    code=0
    python3 "$deploy/ANALYZE.py" --diagnostic "$deploy/physical/$run" --diagnostic-steps 64 \
        --output-tokens 576 --require-metrics > "$deploy/physical/$run.check.log" 2>&1 || code=$?
    python3 - "$deploy" "$run" <<'CHECK'
import importlib.util
import json
from pathlib import Path
import sys
p, run = Path(sys.argv[1]), sys.argv[2]
spec = importlib.util.spec_from_file_location('m2_check', p/'ANALYZE.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rows = []
for reference in (Path('/mnt/storage/s42-fast-path-M2-20260920-d7BA1a/physical/n2-control'),
                  Path('/mnt/storage/s42-fast-path-M2-diagnostic-20260920-a81c09/physical/step2/p256-257-normal/control'),
                  Path('/mnt/storage/s42-fast-path-M2-diagnostic-20260920-a81c09/physical/step2/p256-257-reversed/control')):
    rows.append(module.token_reference(p/'physical'/run, reference))
with (p/'physical'/run/'EXACT_TOKENS.json').open('x') as stream:
    json.dump(rows, stream, indent=2)
    stream.write('\n')
check = module.read(p/'physical'/run/'CHECK_DIAGNOSTIC.json')
print(json.dumps({'run': run, 'status': check['status'], 'max_local_rel_l2': check.get('numeric', {}).get('max_local_rel_l2'),
                  'rows_above_threshold': check.get('numeric', {}).get('rows_above_threshold', []),
                  'exact_tokens': rows}, indent=2), flush=True)
CHECK
    if (( code != 0 )); then exit "$code"; fi
done
python3 "$deploy/ANALYZE.py" --determinism "$deploy/physical" > "$deploy/physical/determinism.log" 2>&1
