#!/bin/sh
# Drive one bounded V2.4.1 A_ONLY acquisition through the frozen orchestration.
set -u

ROOT=/home/zhihao/s39-v26-a-only
REPO=$ROOT/repo-v1/s39
RUN=$ROOT/a-only-run-5
PHASE_ID=cp0-r1-v24-a-only-20260727-e2e-prototype-5
PY=/usr/bin/python3

mkdir -p "$RUN"

"$PY" -B "$REPO/v24_readiness/production_plan_v1/materialize_config_v1.py" \
    --phase-id "$PHASE_ID" \
    --run-root "$RUN/run-root" \
    --config-output "$RUN/config.json" \
    --plan-output "$RUN/plan.json" || { echo "DRIVER_EXIT:10"; exit 10; }

"$PY" -B "$REPO/v24_readiness/orchestration_v1/orchestration_v1.py" \
    preflight --plan "$RUN/plan.json" > "$RUN/preflight.json" \
    || { echo "DRIVER_EXIT:11"; exit 11; }

"$PY" -B "$REPO/v24_readiness/orchestration_v1/orchestration_v1.py" \
    run --plan "$RUN/plan.json" --execute --confirm RUN_CP0_R1_V24_A_ONLY \
    > "$RUN/run.json" 2> "$RUN/run.err"
echo "DRIVER_EXIT:$?"
tail -c 400 "$RUN/run.err" 2>/dev/null
ls "$RUN/run-root/receipts" 2>/dev/null | tr '\n' ' '
