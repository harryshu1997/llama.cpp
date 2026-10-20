#!/bin/bash
# setup_ablation_fault.sh FAULT_PREFIX [WHEN_1 WHEN_2 WHEN_3]   (desktop; WS13)
# 1. Creates $R/template-eval2-s2-dp-online-nobf (batch formation off) and $R/template-eval2-s2-dp-online-elastic
#    (g11 elastic phones, mask_out) from $R/template-eval2-s2-dp-online with tools/template_variants.py of the
#    deployed source, and prints the runner-effective flags of each (an existing template is only re-checked).
# 2. Writes the fault files $R/rig/FAULTS/<FAULT_PREFIX><r>flt (r = 1..3) that launch_one_arm.sh turns into one
#    authorized SIGTERM of the Pixel worker per run (default conditions active=500 / active=1150 / active=1700:
#    after that many seconds, the next moment the Pixel serves a call, i.e. mid-call as in g11). Refuses to
#    overwrite a fault file. The fault block then runs as
#    matrix_queue2.sh <FAULT_PREFIX> 3 "flt=derived-mpaoa:template-eval2-s2-dp-online-elastic" <seed>.
set -euo pipefail
R=${EVAL_ROOT:-/mnt/storage/s43-two-phone-eval-20260925}
D=${EVAL_DEPLOY:-/mnt/storage/s42-trace-v2-20260921-prep}
[ $# -ge 1 ] || { echo "usage: $0 FAULT_PREFIX [WHEN_1 WHEN_2 WHEN_3]" >&2; exit 2; }
PREFIX=$1; shift
WHENS=("$@")
[ ${#WHENS[@]} -eq 0 ] && WHENS=(active=500 active=1150 active=1700)
[ ${#WHENS[@]} -eq 3 ] || { echo "give three conditions (one per repeat)" >&2; exit 2; }
for WHEN in "${WHENS[@]}"; do
  [[ $WHEN =~ ^(request=[0-9]+|(t|active)=[0-9]+(\.[0-9]+)?)$ ]] || { echo "bad condition: $WHEN" >&2; exit 2; }
done
BASE=$R/template-eval2-s2-dp-online
TOOL="python3 -m research_dev.scheduler.campaigns.burstgpt.tools.template_variants"
cd "$D/source"
for SPEC in nobf:no-batch-formation elastic:elastic-mask-out; do
  OUT=$BASE-${SPEC%%:*}
  [ -d "$OUT" ] || $TOOL derive --base "$BASE" --variant "${SPEC#*:}" --out "$OUT" > /dev/null
  $TOOL check --template "$OUT" --base "$BASE"
done
mkdir -p "$R/rig/FAULTS"
for r in 1 2 3; do
  FILE=$R/rig/FAULTS/${PREFIX}${r}flt
  [ ! -e "$FILE" ] || { echo "exists: $FILE" >&2; exit 3; }
done
for r in 1 2 3; do
  printf '%s\n# authorized by the user on 2026-10-01 ("yes do it": queue the fault-injection series after the Pixel comparison)\n' \
    "${WHENS[$((r - 1))]}" > "$R/rig/FAULTS/${PREFIX}${r}flt"
  echo "fault file $R/rig/FAULTS/${PREFIX}${r}flt: ${WHENS[$((r - 1))]}"
done
