#!/bin/bash
# launch_prima_baseline.sh ATT TEMPLATE   (desktop) - the prima.cpp baseline arm: desktop rank 0 (CUDA) + phones over
# adb port tunnels serving the template's hot model with prima.cpp (ICLR 2026) and its Halda layer scheduler.
#   inputs: $R/$TEMPLATE/{campaign,models,rig}.json -> PLAN.json (default_llamacpp_baseline plan);
#           $PRIMA_PHONES (default $P/PHONES-q8-two.json); server $PRIMA_SERVER (default $P/prima.cpp/llama-server)
#   output: $R/inputs-prima-$ATT/run-eval/run/{RESULT.json,PRIMA_RECEIPT.json,server.log,worker-*.log,streams/}
#   env:    PRIMA_GPU_MEM (optional Halda VRAM budget for rank 0, GiB; unset = all measured free VRAM, prima default), PRIMA_RING_MODEL (default qwen3-32b-q8_0), PRIMA_CTX (16384), PRIMA_PARALLEL (4), PRIMA_LAYER_WINDOW (unset =
#           Halda profiling + HiGHS), PRIMA_EXTRA (extra harness args), PRIMA_PORT (18620), EVAL_DRY_RUN=1 prints the command.
# Run it under meter_phones.sh for the phone energy (as the RPC arm); it takes the rig lock itself.
set -u
R=${EVAL_ROOT:-/mnt/storage/s43-two-phone-eval-20260925}
D=${EVAL_DEPLOY:-/mnt/storage/s42-trace-v2-20260921-prep}
P=${PRIMA_DIR:-/mnt/storage/s43-prima-baseline-20261016}
LOCK=${EVAL_LOCK:-/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock}
REPO=${EVAL_REPO:-$D/source}
SERVER=${PRIMA_SERVER:-$P/prima.cpp/llama-server}
PHONES=${PRIMA_PHONES:-$P/PHONES-q8-two.json}
PORT=${PRIMA_PORT:-18620}
CUDA_LIB=${PRIMA_CUDA_LIB:-/mnt/storage/s21_deps/cuda-13.2.1/lib64}
ATT=${1:-}; TEMPLATE=${2:-template-q8two-ts3-anchor-hfix-v2}
[ -n "$ATT" ] || { sed -n '2,9p' "$0"; exit 2; }
T=$R/$TEMPLATE
ARM=$R/inputs-prima-$ATT
OUT=$ARM/run-eval/run
for f in "$T/campaign.json" "$T/models.json" "$T/rig.json" "$PHONES" "$P/prima_baseline.py"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 2; }
done
[ -e "$ARM/run-eval" ] && { echo "arm already ran: $ARM" >&2; exit 13; }
LW=""; [ -n "${PRIMA_LAYER_WINDOW:-}" ] && LW="--layer-window '$PRIMA_LAYER_WINDOW'"
PLANTOOL="python3 -m research_dev.scheduler.campaigns.burstgpt.tools.default_llamacpp_baseline"
CMD="cd '$REPO' && export PYTHONPATH='$REPO' PATH='$P/deps/bin:$PATH' && $PLANTOOL plan --campaign '$T/campaign.json' --models '$T/models.json' --rig '$T/rig.json' --out '$ARM/PLAN.json' < /dev/null && exec python3 '$P/prima_baseline.py' run --plan '$ARM/PLAN.json' --out '$OUT' --server '$SERVER' --phones '$PHONES' --ring-model '${PRIMA_RING_MODEL:-qwen3-32b-q8_0}' --ctx ${PRIMA_CTX:-16384} --parallel ${PRIMA_PARALLEL:-4} ${PRIMA_GPU_MEM:+--gpu-mem $PRIMA_GPU_MEM }--library-dir '$CUDA_LIB' --workdir '${PRIMA_WORKDIR:-/home/zhihao/prima-profile-scratch}' --port $PORT --overlay-port $((PORT+1)) $LW ${PRIMA_EXTRA:-} < /dev/null"
if [ "${EVAL_DRY_RUN:-0}" = 1 ]; then echo "flock -w 7200 $LOCK bash -c \"$CMD\""; exit 0; fi
[ -x "$SERVER" ] || { echo "prima llama-server not found: $SERVER" >&2; exit 2; }
mkdir -p "$ARM" || exit 9
exec flock -w 7200 "$LOCK" bash -c "$CMD"
