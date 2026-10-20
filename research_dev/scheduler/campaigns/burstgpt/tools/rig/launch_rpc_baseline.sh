#!/bin/bash
# launch_rpc_baseline.sh ATT [TEMPLATE]   (desktop) - WS18 stock llama.cpp RPC whole-layer offload arm under ONE rig lock.
#
# The out-of-the-box arm (launch_default_baseline.sh: stock llama-server router, campaign trace and request body,
# host energy over the paid window) plus llama.cpp's own RPC backend: ggml-rpc-server on both phones (USB TCP),
# whole layers (attention, KV, FFN) placed on them by PLACEMENT.json. Phone rpc servers are started after the lock
# is taken and stopped before it is released (rpc_phone_servers.py: preflight, pinned sha256, SIGTERM only).
# Inputs in $EVAL_ROOT/rpc-baseline/: PHONES.json (phones in --rpc order), PLACEMENT.json (rpc_llamacpp_baseline
# placement). The plan is derived per arm from TEMPLATE + those two files and kept in the arm directory.
# Output: $EVAL_ROOT/inputs-rpc-ATT/run-eval/run/{RESULT.json,streams/,server.log,RPC_PHONES.json,rpc-server-*.log}
# Wrap it like the other arms (meter_phones.sh takes no lock itself; this script waits for the rig lock):
#   meter_phones.sh --charging controlled $EVAL_ROOT/meter/ATT-rpc ATT-rpc -- launch_rpc_baseline.sh ATT
# then fleet_energy.py --run rpc=$EVAL_ROOT/inputs-rpc-ATT --meter rpc=...  and  latency_report.py --run rpc=...
# Env: EVAL_ROOT, EVAL_LOCK, EVAL_REPO, RPC_BASELINE_DIR (default $EVAL_ROOT/rpc-baseline), RPC_SERVER_BIN
# (llama-server built with -DGGML_RPC=ON; default $EVAL_DEPLOY/cuda-build-rpc/bin/llama-server), RPC_BASELINE_PORT
# (router port, default 18610), EVAL_DRY_RUN=1 (print the command instead of running it; tests).
set -u
R=${EVAL_ROOT:-/mnt/storage/s43-two-phone-eval-20260925}
D=${EVAL_DEPLOY:-/mnt/storage/s42-trace-v2-20260921-prep}
LOCK=${EVAL_LOCK:-/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${EVAL_REPO:-$(cd "$HERE/../../../../../.." && pwd)}
B=${RPC_BASELINE_DIR:-$R/rpc-baseline}
SERVER=${RPC_SERVER_BIN:-$D/cuda-build-rpc/bin/llama-server}
PORT=${RPC_BASELINE_PORT:-18610}
ATT=${1:-}; TEMPLATE=${2:-template-eval2-s2}
[ -n "$ATT" ] || { sed -n '2,17p' "$0"; exit 2; }
T=$R/$TEMPLATE
ARM=$R/inputs-rpc-$ATT
OUT=$ARM/run-eval/run
for f in "$T/campaign.json" "$T/models.json" "$T/rig.json" "$B/PHONES.json" "$B/PLACEMENT.json"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 2; }
done
[ -f "$REPO/research_dev/scheduler/campaigns/burstgpt/tools/rpc_llamacpp_baseline.py" ] || { echo "harness not found under $REPO" >&2; exit 2; }
[ -e "$ARM/run-eval" ] && { echo "arm already ran: $ARM" >&2; exit 13; }
LIBS=$(python3 -c 'import json,sys; print(" ".join("--library-dir " + v for v in json.load(open(sys.argv[1])).get("library_directories", {}).values()))' "$T/rig.json")
TOOL="python3 -m research_dev.scheduler.campaigns.burstgpt.tools.rpc_llamacpp_baseline"
CMD="cd '$REPO' && $TOOL plan --campaign '$T/campaign.json' --models '$T/models.json' --rig '$T/rig.json' --placement '$B/PLACEMENT.json' --phones '$B/PHONES.json' --out '$ARM/PLAN.json' < /dev/null && exec $TOOL run --plan '$ARM/PLAN.json' --phones '$B/PHONES.json' --server '$SERVER' $LIBS --out '$OUT' --port $PORT < /dev/null"
if [ "${EVAL_DRY_RUN:-0}" = 1 ]; then
  echo "flock -w 7200 $LOCK bash -c \"$CMD\""
  exit 0
fi
[ -x "$SERVER" ] || { echo "RPC llama-server not found: $SERVER (build with -DGGML_RPC=ON)" >&2; exit 2; }
mkdir -p "$ARM" || exit 9
exec flock -w 7200 "$LOCK" bash -c "$CMD"
