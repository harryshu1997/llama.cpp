#!/bin/sh
# Drive the A_ONLY acquisition, retrying ONLY pre-acquisition setup aborts.
#
# The frozen V2.4 rule: a setup abort occurs only before the durable paid-start
# marker and the ordinal may restart once the prerequisite is fixed. Any failure
# in a paid stage is terminal and must NOT be retried, so this loop stops dead
# on cuda_monolithic / joint_phone_cuda / fan_in / authority.
#
# The retry exists because the frozen phone-status probe pipes `dumpsys
# thermalservice` into an `awk` that exits at its first match; dumpsys then
# races to SIGPIPE and writes a broken-pipe line to stderr, which the probe
# rejects. Measured ~25% per call, two calls per attempt.
set -u

SCRATCH=$(dirname "$0")
DESKTOP=zhihao@172.20.74.85
ROOT=/home/zhihao/s39-v26-a-only
MIRROR=$ROOT/repo-v1/s39
PAID="cuda_monolithic joint_phone_cuda fan_in authority"
MAX=${MAX_ATTEMPTS:-4}

attempt=1
while [ "$attempt" -le "$MAX" ]; do
    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    RUN=$ROOT/a-only-$STAMP
    PHASE_ID=cp0-r1-v24-a-only-$STAMP
    echo "########## attempt $attempt/$MAX  run=$RUN"

    ssh -o BatchMode=yes "$DESKTOP" \
        "cd $MIRROR/v24_readiness/desktop_deployment_v1 && rm -f \
         phone_runtime_probe_snapshot_v1.py \
         managed_runtime_launcher_snapshot_v1.py \
         originate_from_inventory_v26.py phone_runtime_probe_v1.py; exit 0"

    if ! sh "$SCRATCH/refresh_and_plan_v26.sh" > "/tmp/refresh_$STAMP.log" 2>&1; then
        echo "REFRESH_FAILED (attempt $attempt)"
        tail -5 "/tmp/refresh_$STAMP.log"
        attempt=$((attempt + 1))
        continue
    fi
    echo "refresh ok"

    ssh -o BatchMode=yes "$DESKTOP" "PHASE_ID=$PHASE_ID RUN=$RUN sh -s" <<'REMOTE'
set -u
REPO=/home/zhihao/s39-v26-a-only/repo-v1/s39
PY=/usr/bin/python3
mkdir -p "$RUN"
"$PY" -B "$REPO/v24_readiness/production_plan_v1/materialize_config_v1.py" \
    --phase-id "$PHASE_ID" --run-root "$RUN/run-root" \
    --config-output "$RUN/config.json" --plan-output "$RUN/plan.json" \
    > "$RUN/config.log" 2>&1 || { echo "STAGE_FAIL:materialize_config"; exit 10; }
"$PY" -B "$REPO/v24_readiness/orchestration_v1/orchestration_v1.py" \
    preflight --plan "$RUN/plan.json" > "$RUN/preflight.json" 2> "$RUN/preflight.err" \
    || { echo "STAGE_FAIL:preflight"; exit 11; }
"$PY" -B "$REPO/v24_readiness/orchestration_v1/orchestration_v1.py" \
    run --plan "$RUN/plan.json" --execute --confirm RUN_CP0_R1_V24_A_ONLY \
    > "$RUN/run.json" 2> "$RUN/run.err"
rc=$?
echo "RUN_RC:$rc"
if [ "$rc" -ne 0 ]; then
    sed -n 's/.*E_STAGE_EXIT: \([a-z_]*\).*/STAGE_FAIL:\1/p' "$RUN/run.err" | head -1
    tail -c 300 "$RUN/run.err"
fi
echo "RECEIPTS:$(ls "$RUN/run-root/receipts" 2>/dev/null | tr '\n' ' ')"
REMOTE

    RESULT=$(ssh -o BatchMode=yes "$DESKTOP" \
        "tail -c 4000 $RUN/run.err 2>/dev/null; echo; \
         ls $RUN/run-root/receipts 2>/dev/null | tr '\n' ' '")
    echo "$RESULT" | tail -3

    if echo "$RESULT" | grep -q 'MODEL_A_QUALIFICATION_PASS_V2_4'; then
        echo "ACQUISITION_PASS run=$RUN"
        exit 0
    fi
    STAGE=$(echo "$RESULT" | sed -n 's/.*E_STAGE_EXIT: \([a-z_]*\).*/\1/p' | head -1)
    if [ -z "$STAGE" ]; then
        if ! echo "$RESULT" | grep -q 'REFUSED\|E_'; then
            echo "ACQUISITION_COMPLETED_NO_REFUSAL run=$RUN"
            exit 0
        fi
        STAGE=unknown
    fi
    echo "failed stage: $STAGE"
    for p in $PAID; do
        if [ "$STAGE" = "$p" ]; then
            echo "PAID_STAGE_FAILURE:$STAGE run=$RUN - terminal, not retrying"
            exit 3
        fi
    done
    echo "pre-acquisition setup abort at $STAGE; retrying"
    attempt=$((attempt + 1))
done
echo "EXHAUSTED_ATTEMPTS"
exit 4
