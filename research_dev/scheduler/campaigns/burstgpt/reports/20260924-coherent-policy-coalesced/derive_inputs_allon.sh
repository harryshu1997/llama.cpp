#!/bin/bash
# dev_v2 A/B under the new dispatcher (changes #2 work-conserving admission + #5 model affinity), CURRENT main tree:
#   dev2baseDP = the 09-23 long-tail baseline inputs (desktop-baseline selection, deploy transport identity copied
#                unchanged, exactly as dev2base) + campaign dispatch_policy {work_conserving_admission, model_affinity}
#                = the new desktop-only reference;
#   dev2allon  = the coherentRP configuration (energy-aware, server_policy_coherence, coalesced-batch qualified for
#                BOTH models, phone_resident_model_reprovisioning {}, identity s43-coalesced-both-20260924, receipts-all,
#                boot f13c7c03) + the same dispatch_policy.
# Usage: derive_inputs_allon.sh dev2baseDP dev2allon   Outside the rig lock: writes only the two input directories.
set -euo pipefail
STAGING=/mnt/storage/s42-allon-20260924-staging          # rsync -rc copy of the local main scheduler tree
ROOT=/home/zhihao
SRC_TREATMENT=$ROOT/s42-trace-longtail-treatment-20260923-inputs
SRC_BASELINE=$ROOT/s42-trace-longtail-baseline-20260923-inputs
DEPLOY=/mnt/storage/s42-trace-v2-20260921-prep
T2=/mnt/storage/burstgpt-source/longtail_dev_v2
RECEIPTS=$ROOT/s43-transport-receipts-61440-20260924/receipts-all
IDENTITY=$DEPLOY/TRANSPORT_QUALIFICATION_IDENTITY_COALESCED_BOTH.json
MATERIALIZE=$ROOT/s42-trace-longtaildev2-coalescedboth-20260924-inputs/MATERIALIZE_COMMAND.json
BOOT=/mnt/storage/s42-dmabuf-cancel-20260914-v1-gBJdFx/candidate-boot.img
BOOT_SHA=sha256:f13c7c033ce74f32ea4aa6349cb531704407708f5fb35756a6c361ffb48322a3
DP='{"work_conserving_admission": true, "model_affinity": true}'
DATE=20260924
export PYTHONDONTWRITEBYTECODE=1 LANG=C.UTF-8
cd "$STAGING"
trace2=(--old-deploy "$DEPLOY" --new-deploy "$DEPLOY"
        --replay-schedule "$T2/burstgpt_longtail_dev_v2.json" --trace-manifest "$T2/TRACE_MANIFEST.json"
        --large-requests "$T2/REQUESTS_SEMANTIC_SOURCE.jsonl" --overlay-requests "$T2/REQUESTS_OVERLAY.jsonl")
coalesced=(--transport-receipts-dir "$RECEIPTS" --phone-boot-image "$BOOT" --phone-boot-image-sha256 "$BOOT_SHA")
for arm in "$@"; do
  out=$ROOT/s42-trace-longtaildev2-${arm#dev2}-$DATE-inputs
  case $arm in
  dev2baseDP)
    python3 research_dev/scheduler/campaigns/burstgpt/prepare_trace_inputs_v2.py \
        --source "$SRC_BASELINE" --output "$out" --campaign-id "s42-burstgpt-longtaildev2-baseDP-$DATE" \
        "${trace2[@]}" --dispatch-policy-json "$DP"
    cp "$DEPLOY/TRANSPORT_QUALIFICATION_IDENTITY.json" "$out/TRANSPORT_QUALIFICATION_IDENTITY.json" ;;
  dev2allon)
    python3 research_dev/scheduler/campaigns/burstgpt/prepare_trace_inputs_v2.py \
        --source "$SRC_TREATMENT" --output "$out" --campaign-id "s42-burstgpt-longtaildev2-allon-coalescedboth-$DATE" \
        "${trace2[@]}" "${coalesced[@]}" \
        --adaptive-decode-overrides-json '{"server_policy_coherence": true}' \
        --qualify-phone-batch-plan hot=coalesced-batch --qualify-phone-batch-plan cold=coalesced-batch \
        --phone-resident-model-reprovisioning-json '{}' --dispatch-policy-json "$DP"
    # The identity is materialized once (MATERIALIZE_TRANSPORT_COALESCED_BOTH.sh); every coalesced arm binds the same file.
    cp "$IDENTITY" "$out/TRANSPORT_QUALIFICATION_IDENTITY.json"
    cp "$MATERIALIZE" "$out/MATERIALIZE_COMMAND.json" ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
  esac
  python3 - "$out" "$arm" <<'PY'
import hashlib, json, sys, pathlib
out, arm = pathlib.Path(sys.argv[1]), sys.argv[2]
ident = json.load(open(out / "TRANSPORT_QUALIFICATION_IDENTITY.json"))
rig = json.load(open(out / "rig.json"))
campaign = json.load(open(out / "campaign.json"))
assert campaign["dispatch_policy"] == {"model_affinity": True, "work_conserving_admission": True}, campaign["dispatch_policy"]
if arm == "dev2allon":
    assert ident["identity_id"] == "s43-coalesced-both-20260924", ident["identity_id"]
    assert campaign["selection_mode"] == "energy-aware", campaign["selection_mode"]
    assert campaign["adaptive_decode_overrides"] == {"server_policy_coherence": True}
    assert campaign["phone_resident_model_reprovisioning"] == {}
else:
    assert campaign["selection_mode"] == "desktop-baseline", campaign["selection_mode"]
    assert "adaptive_decode_overrides" not in campaign and "phone_resident_model_reprovisioning" not in campaign
assert rig["phone"]["boot_image_sha256"] == ident["hardware_identity"]["phone_boot_image_sha256"], out
assert rig["phone"]["serial"] == ident["hardware_identity"]["phone_usb_serial"], out
digest = hashlib.sha256((out / "TRANSPORT_QUALIFICATION_IDENTITY.json").read_bytes()).hexdigest()
print(out.name, campaign["selection_mode"], ident["identity_id"], len(ident["receipt_sha256s"]), "receipts",
      digest[:16], rig["phone"]["boot_image_sha256"][:20], "dispatch_policy", json.dumps(campaign["dispatch_policy"], sort_keys=True))
PY
done
