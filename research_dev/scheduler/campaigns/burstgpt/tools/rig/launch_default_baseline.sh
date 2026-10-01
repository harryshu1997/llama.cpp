#!/bin/bash
# launch_default_baseline.sh ATT [TEMPLATE]   (desktop) - the out-of-the-box llama.cpp reference arm under ONE rig lock.
#
# Serves the template's trace (campaign.json "trace") with ONE stock llama-server in router mode (the rig's
# binaries.server, models from the template's models.json, llama.cpp defaults for every placement/performance
# parameter; deviations --models-max 1, client FIFO drain-before-switch, -lv 4 are recorded in RESULT.json) and
# measures host energy like the campaign (RAPL package-0 + NVML board power over the paid window).
# Output: $EVAL_ROOT/inputs-default-ATT/run-eval/run/{RESULT.json,streams/,server.log,MODEL_PLACEMENTS.json,...}
# Wrap it like the other arms (meter_phones.sh takes no lock itself; this script waits for the rig lock):
#   meter_phones.sh --charging controlled $EVAL_ROOT/meter/ATT-default ATT-default -- launch_default_baseline.sh ATT
# then
#   fleet_energy.py --run default=$EVAL_ROOT/inputs-default-ATT --meter default=$EVAL_ROOT/meter/ATT-default/METER.json
#   latency_report.py --run default=$EVAL_ROOT/inputs-default-ATT
# Env: EVAL_ROOT, EVAL_LOCK, EVAL_REPO (repo holding this script, default derived from its path), DEFAULT_BASELINE_PORT
# (router port, default 18600), EVAL_DRY_RUN=1 (print the command instead of running it; tests).
set -u
R=${EVAL_ROOT:-/mnt/storage/s43-two-phone-eval-20260925}
LOCK=${EVAL_LOCK:-/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${EVAL_REPO:-$(cd "$HERE/../../../../../.." && pwd)}
PORT=${DEFAULT_BASELINE_PORT:-18600}
ATT=${1:-}; TEMPLATE=${2:-template-eval2-s2}
[ -n "$ATT" ] || { sed -n '2,15p' "$0"; exit 2; }
T=$R/$TEMPLATE
ARM=$R/inputs-default-$ATT
OUT=$ARM/run-eval/run
for f in campaign.json models.json rig.json; do
  [ -f "$T/$f" ] || { echo "missing $T/$f" >&2; exit 2; }
done
[ -f "$REPO/research_dev/scheduler/campaigns/burstgpt/tools/default_llamacpp_baseline.py" ] || { echo "harness not found under $REPO" >&2; exit 2; }
[ -e "$ARM/run-eval" ] && { echo "arm already ran: $ARM" >&2; exit 13; }
CMD="cd '$REPO' && exec python3 -m research_dev.scheduler.campaigns.burstgpt.tools.default_llamacpp_baseline run --campaign '$T/campaign.json' --models '$T/models.json' --rig '$T/rig.json' --out '$OUT' --port $PORT < /dev/null"
if [ "${EVAL_DRY_RUN:-0}" = 1 ]; then
  echo "flock -w 7200 $LOCK bash -c \"$CMD\""
  exit 0
fi
mkdir -p "$ARM" || exit 9
exec flock -w 7200 "$LOCK" bash -c "$CMD"
