#!/usr/bin/env bash
set -euo pipefail
deploy=/mnt/storage/s42-fast-path-M2-acceptance-20260921-c09ea6
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$deploy/native-source
for run in run1 run2; do
    bash "$deploy/RUN_ARM.sh" "determinism/$run" combined 256,257 64
    python3 "$deploy/ANALYZE.py" --diagnostic "$deploy/physical/determinism/$run" --diagnostic-steps 5 --output-tokens 64 --require-metrics > "$deploy/physical/determinism/$run.check.log" 2>&1
done
python3 "$deploy/ANALYZE.py" --determinism "$deploy/physical/determinism" --ordered --diagnostic-steps 5 --output-tokens 64 > "$deploy/physical/determinism/CHECK_DETERMINISM.log" 2>&1
bash "$deploy/RUN_ARM.sh" regression combined pair-v1 64
python3 "$deploy/ANALYZE.py" --regression "$deploy/physical/regression" --reference "$deploy/reference-m0" > "$deploy/physical/regression/CHECK_REGRESSION.log" 2>&1
for n in 1 2 4 8; do
    lengths=$(python3 -c 'import sys; print(",".join(str(256+i) for i in range(int(sys.argv[1]))))' "$n")
    for arm in control combined; do
        bash "$deploy/RUN_ARM.sh" "n$n-$arm" "$arm" "$lengths" 576
    done
    python3 "$deploy/ANALYZE.py" --root "$deploy/physical" --n "$n" > "$deploy/physical/CHECK_N${n}_pair.log" 2>&1
done
