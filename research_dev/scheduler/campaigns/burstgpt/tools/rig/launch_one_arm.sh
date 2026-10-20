#!/bin/bash
# launch_one_arm.sh ATT TEMPLATE KIND   (desktop; KIND = legacy | desktop | op15 | two-phone | default | rpc | derived-NAME)
# KIND=derived-NAME: an arm derived from this attempt's two-phone inputs by $R/rig/derive_NAME.sh BASE_INPUTS OUT_INPUTS
# (e.g. measured placement + Pixel AOA); the derive script runs once, under the same lock, before the arm.
# KIND=default: the out-of-the-box llama.cpp reference arm (launch_default_baseline.sh, stock llama-server router mode).
# KIND=rpc: the stock llama.cpp RPC whole-layer offload to both phones (launch_rpc_baseline.sh; $R/rpc-baseline inputs).
# Fault injection (opt-in, elastic-phones arms only): $R/rig/FAULTS/$ATT holds an inject_helper_loss.py --when value
# on its first line (request=NNN | t=SECONDS | active=SECONDS). The file is the per-run authorization record: create it
# only for runs the user authorized. The injector (SIGTERM, never SIGKILL) waits for the arm's run directory, fires
# once, logs to $R/chains/FAULT-$ATT.log and is ended when the arm ends. Without the file nothing changes.
# Under ONE rig lock: prepare ATT from TEMPLATE once (creates inputs-{desktop,op15,two-phone}-ATT), derive the legacy
# all-desktop inputs once (--drop-dispatch-policy, as launch_ev2.sh), then run exactly one arm with run_chain_eval.py.
set -u
R=${EVAL_ROOT:-/mnt/storage/s43-two-phone-eval-20260925}
D=${EVAL_DEPLOY:-/mnt/storage/s42-trace-v2-20260921-prep}
LOCK=${EVAL_LOCK:-/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock}
ATT=$1; TEMPLATE=$2; KIND=$3
case "$KIND" in legacy|desktop|op15|two-phone|default|rpc|derived-?*) ;; *) echo "KIND must be legacy|desktop|op15|two-phone|default|rpc|derived-NAME" >&2; exit 2;; esac
if [ "$KIND" = default ]; then EVAL_REPO=$D/source exec bash "$(cd "$(dirname "$0")" && pwd)/launch_default_baseline.sh" "$ATT" "$TEMPLATE"; fi
if [ "$KIND" = rpc ]; then EVAL_REPO=$D/source exec bash "$(cd "$(dirname "$0")" && pwd)/launch_rpc_baseline.sh" "$ATT" "$TEMPLATE"; fi
export S43_TRANSPORT_IDENTITY=${S43_TRANSPORT_IDENTITY:-$D/TRANSPORT_QUALIFICATION_IDENTITY_FUSED_20261004.json}   # 2026-10-04: rebuilt server (fused C++), re-pinned identity
export S43_SERVER_IDENTITY_DIR=${S43_SERVER_IDENTITY_DIR:-server-identity-r3}
if [ "$KIND" = legacy ]; then ARM=$R/inputs-desktop-legacy-$ATT; else ARM=$R/inputs-$KIND-$ATT; fi
WHEN=
if [ -f "$R/rig/FAULTS/$ATT" ]; then
  WHEN=$(head -n1 "$R/rig/FAULTS/$ATT" | tr -d '[:space:]')
  [[ $WHEN =~ ^(request=[0-9]+|(t|active)=[0-9]+(\.[0-9]+)?)$ ]] || { echo "bad fault condition in $R/rig/FAULTS/$ATT: $WHEN" >&2; exit 16; }
fi
INJECTOR=${S43_FAULT_INJECTOR:-$D/source/research_dev/scheduler/campaigns/burstgpt/tools/inject_helper_loss.py}
FAULT_DEVICE=${S43_FAULT_DEVICE:-pixel10pro-phone}
FAULT_POLL_S=${S43_FAULT_POLL_S:-5}
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
case $KIND in derived-*)
  if [ ! -d $ARM ]; then
    [ -f $R/rig/derive_${KIND#derived-}.sh ] || { echo 'no derive script: $R/rig/derive_${KIND#derived-}.sh' >&2; exit 14; }
    bash $R/rig/derive_${KIND#derived-}.sh $R/inputs-two-phone-$ATT $ARM < /dev/null || exit 15
  fi;;
esac
[ -d $ARM/run-eval ] && { echo 'arm already ran: $ARM' >&2; exit 13; }
if [ -n '$WHEN' ]; then
  ( for i in \$(seq 1 720); do [ -d $ARM/run-eval/run ] && [ -f $ARM/run-eval/RESOLVED_CONFIGURATION.json ] && break; sleep $FAULT_POLL_S; done
    exec python3 $INJECTOR --device $FAULT_DEVICE --when $WHEN --signal TERM --run-dir $ARM/run-eval/run --authorized --timeout-s 3600 ) > $R/chains/FAULT-$ATT.log 2>&1 < /dev/null &
  INJ=\$!
  python3 $R/run_chain_eval.py --status $R/chains/CHAIN-$ATT-$KIND.jsonl --stage $R/stage --arm $ARM < /dev/null
  rc=\$?
  kill -TERM \$INJ 2>/dev/null
  wait \$INJ 2>/dev/null
  exit \$rc
fi
exec python3 $R/run_chain_eval.py --status $R/chains/CHAIN-$ATT-$KIND.jsonl --stage $R/stage --arm $ARM < /dev/null
"
