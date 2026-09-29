#!/bin/bash
# dev_v2 A/B after the evidence fixes, both arms bound to the both-models coalesced identity
# (s43-coalesced-both-20260924, receipts 7,680/10,240/38,400/40,960/61,440 B), so the arms differ only by the flags:
#   dev2plainEF    = the 09-23 long-tail treatment inputs (split-row qualified for both models), no coherence;
#   dev2coherentEF = + server_policy_coherence + coalesced-batch qualified for BOTH models (no mixed plans).
# Usage: derive_inputs_ef.sh dev2plainEF dev2coherentEF   Outside the rig lock: writes only the input directories.
set -euo pipefail
STAGING=/home/zhihao/s42-coherentEF-20260924-staging        # rsync -rc copy of the local scheduler tree
ROOT=/home/zhihao
SRC=$ROOT/s42-trace-longtail-treatment-20260923-inputs
DEPLOY=/mnt/storage/s42-trace-v2-20260921-prep
T2=/mnt/storage/burstgpt-source/longtail_dev_v2
RECEIPTS=$ROOT/s43-transport-receipts-61440-20260924/receipts-all
IDENTITY=$DEPLOY/TRANSPORT_QUALIFICATION_IDENTITY_COALESCED_BOTH.json
MATERIALIZE=$ROOT/s42-trace-longtaildev2-coalescedboth-20260924-inputs/MATERIALIZE_COMMAND.json
BOOT=/mnt/storage/s42-dmabuf-cancel-20260914-v1-gBJdFx/candidate-boot.img
BOOT_SHA=sha256:f13c7c033ce74f32ea4aa6349cb531704407708f5fb35756a6c361ffb48322a3
DATE=20260924
export PYTHONDONTWRITEBYTECODE=1 LANG=C.UTF-8
cd "$STAGING"
common=(--old-deploy "$DEPLOY" --new-deploy "$DEPLOY"
        --replay-schedule "$T2/burstgpt_longtail_dev_v2.json" --trace-manifest "$T2/TRACE_MANIFEST.json"
        --large-requests "$T2/REQUESTS_SEMANTIC_SOURCE.jsonl" --overlay-requests "$T2/REQUESTS_OVERLAY.jsonl"
        --transport-receipts-dir "$RECEIPTS" --phone-boot-image "$BOOT" --phone-boot-image-sha256 "$BOOT_SHA")
for arm in "$@"; do
  out=$ROOT/s42-trace-longtaildev2-${arm#dev2}-$DATE-inputs
  case $arm in
  dev2plainEF)
    python3 research_dev/scheduler/campaigns/burstgpt/prepare_trace_inputs_v2.py --source "$SRC" --output "$out" \
        --campaign-id "s42-burstgpt-longtaildev2-plainEF-$DATE" "${common[@]}" ;;
  dev2coherentEF)
    python3 research_dev/scheduler/campaigns/burstgpt/prepare_trace_inputs_v2.py --source "$SRC" --output "$out" \
        --campaign-id "s42-burstgpt-longtaildev2-coherentEF-coalescedboth-$DATE" "${common[@]}" \
        --adaptive-decode-overrides-json '{"server_policy_coherence": true}' \
        --qualify-phone-batch-plan hot=coalesced-batch --qualify-phone-batch-plan cold=coalesced-batch ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
  esac
  # The identity is materialized once (MATERIALIZE_TRANSPORT_COALESCED_BOTH.sh); every arm binds the same file.
  cp "$IDENTITY" "$out/TRANSPORT_QUALIFICATION_IDENTITY.json"
  cp "$MATERIALIZE" "$out/MATERIALIZE_COMMAND.json"
  python3 - "$out" <<'PY'
import hashlib, json, sys, pathlib
out = pathlib.Path(sys.argv[1])
ident = json.load(open(out / "TRANSPORT_QUALIFICATION_IDENTITY.json"))
rig = json.load(open(out / "rig.json"))
assert ident["identity_id"] == "s43-coalesced-both-20260924", ident["identity_id"]
assert rig["phone"]["boot_image_sha256"] == ident["hardware_identity"]["phone_boot_image_sha256"], out
assert rig["phone"]["serial"] == ident["hardware_identity"]["phone_usb_serial"], out
digest = hashlib.sha256((out / "TRANSPORT_QUALIFICATION_IDENTITY.json").read_bytes()).hexdigest()
print(out.name, ident["identity_id"], len(ident["receipt_sha256s"]), "receipts", digest[:16], rig["phone"]["boot_image_sha256"][:20])
PY
done
