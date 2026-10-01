#!/bin/bash
# launch_one_arm.sh ATT TEMPLATE KIND   (desktop; KIND = legacy | desktop | op15 | two-phone | default)
# KIND=default: the out-of-the-box llama.cpp reference arm (launch_default_baseline.sh, stock llama-server router mode).
# Under ONE rig lock: prepare ATT from TEMPLATE once (creates inputs-{desktop,op15,two-phone}-ATT), derive the legacy
# all-desktop inputs once (--drop-dispatch-policy, as launch_ev2.sh), then run exactly one arm with run_chain_eval.py.
set -u
R=${EVAL_ROOT:-/mnt/storage/s43-two-phone-eval-20260925}
D=${EVAL_DEPLOY:-/mnt/storage/s42-trace-v2-20260921-prep}
LOCK=${EVAL_LOCK:-/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock}
ATT=$1; TEMPLATE=$2; KIND=$3
case "$KIND" in legacy|desktop|op15|two-phone|default) ;; *) echo "KIND must be legacy|desktop|op15|two-phone|default" >&2; exit 2;; esac
if [ "$KIND" = default ]; then EVAL_REPO=$D/source exec bash "$(cd "$(dirname "$0")" && pwd)/launch_default_baseline.sh" "$ATT" "$TEMPLATE"; fi
export S43_TRANSPORT_IDENTITY=${S43_TRANSPORT_IDENTITY:-$D/TRANSPORT_QUALIFICATION_IDENTITY_S2A_20260926.json}
export S43_SERVER_IDENTITY_DIR=${S43_SERVER_IDENTITY_DIR:-server-identity-r2}
if [ "$KIND" = legacy ]; then ARM=$R/inputs-desktop-legacy-$ATT; else ARM=$R/inputs-$KIND-$ATT; fi
exec flock -w 7200 "$LOCK" bash -c "
set -u
cd $R || exit 9
if [ ! -d $R/inputs-desktop-$ATT ]; then
  python3 $R/run_chain_eval.py --status $R/chains/CHAIN-$ATT-prep.jsonl --stage $R/stage --prepare-script $R/prepare_campaign_eval.py --prepare $R $ATT $R/$TEMPLATE ev < /dev/null || exit 10
fi
if [ $KIND = legacy ] && [ ! -d $R/inputs-desktop-legacy-$ATT ]; then
  (cd $D/source && python3 research_dev/scheduler/campaigns/burstgpt/prepare_trace_inputs_v2.py --source $R/inputs-desktop-$ATT --output $R/inputs-desktop-legacy-$ATT --campaign-id s43-two-phone-eval-ev-desktop-legacy-$ATT --old-deploy $D --new-deploy $D --drop-dispatch-policy < /dev/null) || exit 11
  cp $R/inputs-desktop-$ATT/TRANSPORT_QUALIFICATION_IDENTITY.json $R/inputs-desktop-legacy-$ATT/ || exit 12
fi
[ -d $ARM/run-eval ] && { echo 'arm already ran: $ARM' >&2; exit 13; }
exec python3 $R/run_chain_eval.py --status $R/chains/CHAIN-$ATT-$KIND.jsonl --stage $R/stage --arm $ARM < /dev/null
"
