#!/bin/bash
# Derive the dev-trace arms (standard OP15 stack) from the long-tail treatment inputs:
#   plain    = the 2026-09-23 treatment configuration on the dev trace, deploy transport identity copied
#              unchanged (split-row qualified; coalesced declared but not transport-qualified, as on 09-23);
#   coherent = plain + adaptive_decode_overrides.server_policy_coherence + Qwen coalesced-batch qualified.
#              A 4-row coalesced Qwen call carries 4 x 5,120 x 2 = 40,960 bytes, which the deploy identity's
#              7,680/10,240-byte receipts cannot admit (m4a5: TRANSPORT_PROFILE_INCOMPLETE), so this arm binds
#              the 2026-09-22 task1 receipts (7,680/10,240/40,960, depth 4) and the boot image they were taken
#              under (f13c7c03, the running kernel; phone boot_id unchanged since).
# Usage: derive_inputs.sh plain|coherent ...   Runs outside the rig lock: reads the deploy, writes only inputs.
set -euo pipefail
STAGING=/home/zhihao/s42-coherent-20260924-staging          # rsync copy of the local scheduler tree
ROOT=/home/zhihao
SRC=$ROOT/s42-trace-longtail-treatment-20260923-inputs
DEPLOY=/mnt/storage/s42-trace-v2-20260921-prep
TRACE=/mnt/storage/burstgpt-source/longtail_dev_v1
RECEIPTS=/mnt/storage/s42-task1-transport-20260922-v2/receipts
BOOT=/mnt/storage/s42-dmabuf-cancel-20260914-v1-gBJdFx/candidate-boot.img
BOOT_SHA=sha256:f13c7c033ce74f32ea4aa6349cb531704407708f5fb35756a6c361ffb48322a3
DATE=20260924
export S42_UNIFIED_REPO_ROOT=$STAGING PYTHONDONTWRITEBYTECODE=1 LANG=C.UTF-8
cd "$STAGING"
common=(--old-deploy "$DEPLOY" --new-deploy "$DEPLOY"
        --replay-schedule "$TRACE/burstgpt_longtail_dev_v1.json" --trace-manifest "$TRACE/TRACE_MANIFEST.json"
        --large-requests "$TRACE/REQUESTS_SEMANTIC_SOURCE.jsonl" --overlay-requests "$TRACE/REQUESTS_OVERLAY.jsonl")
for arm in "$@"; do
  out=$ROOT/s42-trace-longtaildev-$arm-$DATE-inputs
  case $arm in
    v2a*) out=$ROOT/s42-trace-v2a-${arm#v2a}-$DATE-inputs ;;
    dev2base) out=$ROOT/s42-trace-longtaildev2-baseline-$DATE-inputs ;;
    dev2*) out=$ROOT/s42-trace-longtaildev2-${arm#dev2}-$DATE-inputs ;;
  esac
  case $arm in
  plain)
    python3 research_dev/scheduler/campaigns/burstgpt/prepare_trace_inputs_v2.py --source "$SRC" --output "$out" \
        --campaign-id "s42-burstgpt-longtaildev-v1-plain-$DATE" "${common[@]}"
    cp "$DEPLOY/TRANSPORT_QUALIFICATION_IDENTITY.json" "$out/TRANSPORT_QUALIFICATION_IDENTITY.json"
    ;;
  coherent)
    python3 research_dev/scheduler/campaigns/burstgpt/prepare_trace_inputs_v2.py --source "$SRC" --output "$out" \
        --campaign-id "s42-burstgpt-longtaildev-v1-coherent-coalesced-$DATE" "${common[@]}" \
        --adaptive-decode-overrides-json '{"server_policy_coherence": true}' --qualify-phone-batch-plan hot=coalesced-batch --allow-mixed-phone-batch-plans \
        --transport-receipts-dir "$RECEIPTS" --phone-boot-image "$BOOT" --phone-boot-image-sha256 "$BOOT_SHA"
    # The m4a8b materialize command (same deploy build, same qualifier hashes, same 9 receipts), new id and output.
    python3 - "$ROOT/s42-trace-v2a-m4a8b-20260922-inputs/MATERIALIZE_COMMAND.json" "$out" "s42-trace-longtaildev-coherent-$DATE" <<'PY'
import json, subprocess, sys, pathlib
command, out, identity_id = json.load(open(sys.argv[1])), pathlib.Path(sys.argv[2]), sys.argv[3]
command[command.index("--identity-id") + 1] = identity_id
command[command.index("--output") + 1] = str(out / "TRANSPORT_QUALIFICATION_IDENTITY.json")
json.dump(command, open(out / "MATERIALIZE_COMMAND.json", "w"), indent=1)
subprocess.run(command, check=True, stdin=subprocess.DEVNULL)
PY
    ;;
  v2acoherent)
    # Pair-bearing check (the dev trace never decodes two requests at once): the 24-request v2a trace, derived
    # from its plain treatment inputs (m4a3) with exactly the coherent arm's flags (= the m4a8b configuration).
    python3 research_dev/scheduler/campaigns/burstgpt/prepare_trace_inputs_v2.py \
        --source "$ROOT/s42-trace-v2a-m4a3-20260921-inputs" --output "$out" \
        --campaign-id "s42-trace-v2a-coherent-coalesced-$DATE" --old-deploy "$DEPLOY" --new-deploy "$DEPLOY" \
        --adaptive-decode-overrides-json '{"server_policy_coherence": true}' --qualify-phone-batch-plan hot=coalesced-batch --allow-mixed-phone-batch-plans \
        --transport-receipts-dir "$RECEIPTS" --phone-boot-image "$BOOT" --phone-boot-image-sha256 "$BOOT_SHA"
    python3 - "$ROOT/s42-trace-v2a-m4a8b-20260922-inputs/MATERIALIZE_COMMAND.json" "$out" "s42-trace-v2a-coherent-$DATE" <<'PY'
import json, subprocess, sys, pathlib
command, out, identity_id = json.load(open(sys.argv[1])), pathlib.Path(sys.argv[2]), sys.argv[3]
command[command.index("--identity-id") + 1] = identity_id
command[command.index("--output") + 1] = str(out / "TRANSPORT_QUALIFICATION_IDENTITY.json")
json.dump(command, open(out / "MATERIALIZE_COMMAND.json", "w"), indent=1)
subprocess.run(command, check=True, stdin=subprocess.DEVNULL)
PY
    ;;
  v2aplain)
    # v2acoherent without the two #1 flags; same receipts, boot label and transport identity file.
    python3 research_dev/scheduler/campaigns/burstgpt/prepare_trace_inputs_v2.py \
        --source "$ROOT/s42-trace-v2a-m4a3-20260921-inputs" --output "$out" \
        --campaign-id "s42-trace-v2a-plain-$DATE" --old-deploy "$DEPLOY" --new-deploy "$DEPLOY" \
        --transport-receipts-dir "$RECEIPTS" --phone-boot-image "$BOOT" --phone-boot-image-sha256 "$BOOT_SHA"
    cp "$ROOT/s42-trace-v2a-coherent-$DATE-inputs/TRANSPORT_QUALIFICATION_IDENTITY.json" \
       "$ROOT/s42-trace-v2a-coherent-$DATE-inputs/MATERIALIZE_COMMAND.json" "$out/"
    ;;
  dev2base|dev2plain|dev2coherent|dev2coherentsr)
    # longtail_dev_v2: 9 requests with same-model overlapping pairs (built for this check).
    T2=/mnt/storage/burstgpt-source/longtail_dev_v2
    trace2=(--old-deploy "$DEPLOY" --new-deploy "$DEPLOY"
            --replay-schedule "$T2/burstgpt_longtail_dev_v2.json" --trace-manifest "$T2/TRACE_MANIFEST.json"
            --large-requests "$T2/REQUESTS_SEMANTIC_SOURCE.jsonl" --overlay-requests "$T2/REQUESTS_OVERLAY.jsonl")
    coalesced=(--transport-receipts-dir "$RECEIPTS" --phone-boot-image "$BOOT" --phone-boot-image-sha256 "$BOOT_SHA")
    identity=$ROOT/s42-trace-longtaildev-coherent-$DATE-inputs
    case $arm in
    dev2base)
      # Desktop-only reference: the 09-23 long-tail baseline inputs, deploy identity copied unchanged.
      python3 research_dev/scheduler/campaigns/burstgpt/prepare_trace_inputs_v2.py \
          --source "$ROOT/s42-trace-longtail-baseline-20260923-inputs" --output "$out" \
          --campaign-id "s42-burstgpt-longtaildev2-baseline-$DATE" "${trace2[@]}"
      cp "$DEPLOY/TRANSPORT_QUALIFICATION_IDENTITY.json" "$out/" ;;
    dev2plain)
      python3 research_dev/scheduler/campaigns/burstgpt/prepare_trace_inputs_v2.py --source "$SRC" --output "$out" \
          --campaign-id "s42-burstgpt-longtaildev2-plain-$DATE" "${trace2[@]}" "${coalesced[@]}"
      cp "$identity/TRANSPORT_QUALIFICATION_IDENTITY.json" "$identity/MATERIALIZE_COMMAND.json" "$out/" ;;
    dev2coherent)
      python3 research_dev/scheduler/campaigns/burstgpt/prepare_trace_inputs_v2.py --source "$SRC" --output "$out" \
          --campaign-id "s42-burstgpt-longtaildev2-coherent-coalesced-$DATE" "${trace2[@]}" "${coalesced[@]}" \
          --adaptive-decode-overrides-json '{"server_policy_coherence": true}' --qualify-phone-batch-plan hot=coalesced-batch --allow-mixed-phone-batch-plans
      cp "$identity/TRANSPORT_QUALIFICATION_IDENTITY.json" "$identity/MATERIALIZE_COMMAND.json" "$out/" ;;
    dev2coherentsr)
      # Coherence only (split-row for both models): with Qwen coalesced and Gemma split-row the phone session
      # cannot be partially reconfigured between the models (the transport contract includes the batch plan),
      # so this arm isolates the coherent policy with both models' phone paths intact.
      python3 research_dev/scheduler/campaigns/burstgpt/prepare_trace_inputs_v2.py --source "$SRC" --output "$out" \
          --campaign-id "s42-burstgpt-longtaildev2-coherent-splitrow-$DATE" "${trace2[@]}" "${coalesced[@]}" \
          --adaptive-decode-overrides-json '{"server_policy_coherence": true}'
      cp "$identity/TRANSPORT_QUALIFICATION_IDENTITY.json" "$identity/MATERIALIZE_COMMAND.json" "$out/" ;;
    esac
    ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
  esac
  python3 - "$out" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
ident = json.load(open(out / "TRANSPORT_QUALIFICATION_IDENTITY.json"))
rig = json.load(open(out / "rig.json"))
assert rig["phone"]["boot_image_sha256"] == ident["hardware_identity"]["phone_boot_image_sha256"], out
assert rig["phone"]["serial"] == ident["hardware_identity"]["phone_usb_serial"], out
print(out.name, ident["identity_id"], len(ident["receipt_sha256s"]), "receipts", rig["phone"]["boot_image_sha256"][:20])
PY
done
