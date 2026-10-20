#!/bin/bash
# derive_mpaoa.sh BASE_INPUTS OUT_INPUTS   (desktop; called by launch_one_arm.sh for KIND=derived-mpaoa)
# Measured layer placement (WS11/WS12 plan) + the Pixel on direct USB (WS10 AOA bridge), derived from a prepared
# two-phone arm: prepare_measured_placement -> OUT.mp, then prepare_aoa_evidence derive-arm -> OUT.
# The placement work dir W holds the Phase 1 outputs:
#   W/DERIVE_MP_ARGS        prepare_measured_placement arguments, one per line (--plan ... --primary-index ... ...)
#   W/DERIVE_AOA_EVIDENCE   path of the Pixel's direct-USB evidence bundle for the planned layers
# On any failure both outputs are removed, so the launcher derives again next time instead of running a half arm.
set -euo pipefail
BASE=$1; OUT=$2; MP=$OUT.mp
W=${S43_PLACEMENT_DIR:-/mnt/storage/s43-ws12-pixel-placement-20261002}
SRC=${EVAL_DEPLOY:-/mnt/storage/s42-trace-v2-20260921-prep}/source
for f in "$W/DERIVE_MP_ARGS" "$W/DERIVE_AOA_EVIDENCE" "$BASE/campaign.json" "$BASE/TRANSPORT_QUALIFICATION_IDENTITY.json"; do
  [ -f "$f" ] || { echo "derive_mpaoa: missing $f" >&2; exit 2; }
done
[ ! -e "$OUT" ] && [ ! -e "$MP" ] || { echo "derive_mpaoa: $OUT or $MP already exists" >&2; exit 3; }
trap 'rc=$?; [ $rc -eq 0 ] || rm -rf "$OUT" "$MP"; exit $rc' EXIT
mapfile -t ARGS < <(grep -v '^[[:space:]]*$' "$W/DERIVE_MP_ARGS")
cd "$SRC"
python3 -m research_dev.scheduler.campaigns.burstgpt.tools.prepare_measured_placement \
  --inputs "$BASE" --out "$MP" "${ARGS[@]}"
# the materializer writes the four manifests only; derive-arm re-points the transport identity into its copy
cp "$BASE/TRANSPORT_QUALIFICATION_IDENTITY.json" "$MP/"
python3 -m research_dev.scheduler.campaigns.burstgpt.tools.prepare_aoa_evidence derive-arm \
  --inputs "$MP" --evidence "$(head -n1 "$W/DERIVE_AOA_EVIDENCE")" --output "$OUT"
echo "derive_mpaoa: $OUT"
