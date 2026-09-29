#!/bin/bash
# One OP15+Pixel per-device arm on longtail_eval_v2 (elastic OFF): prepare ATT -> two-phone arm, ONE rig lock.
# Used as the confirming run for the re-provisioning fixes (F1-F3) after the OP15 cooled down.
set -u
R=/mnt/storage/s43-two-phone-eval-20260925
LOCK=/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
ATT=${1:-tp1}
TEMPLATE=${2:-template-eval2}
# The deploy runs the S2a-rebuilt server: bind the re-pinned transport identity + the r2 server-token-identity run.
export S43_TRANSPORT_IDENTITY=${S43_TRANSPORT_IDENTITY:-/mnt/storage/s42-trace-v2-20260921-prep/TRANSPORT_QUALIFICATION_IDENTITY_S2A_20260926.json}
export S43_SERVER_IDENTITY_DIR=${S43_SERVER_IDENTITY_DIR:-server-identity-r2}
exec flock -w 7200 "$LOCK" bash -c "
set -u
cd $R || exit 9
python3 $R/run_chain_eval.py --status $R/chains/CHAIN-$ATT-prep.jsonl --stage $R/stage --prepare-script $R/prepare_campaign_eval.py --prepare $R $ATT $R/$TEMPLATE ev < /dev/null || exit 10
exec python3 $R/run_chain_eval.py --status $R/chains/CHAIN-$ATT.jsonl --stage $R/stage --arm $R/inputs-two-phone-$ATT < /dev/null
"
