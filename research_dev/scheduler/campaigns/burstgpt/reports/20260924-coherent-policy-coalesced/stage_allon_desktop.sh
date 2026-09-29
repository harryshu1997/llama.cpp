#!/bin/bash
# Desktop side of the staging (run AFTER the local `rsync -rc` of the main tree into $STAGING):
#   1. manifest = every file that differs between the staged tree and the deploy source (rsync -n -rc), with the
#      staged file's sha256 -> RIG/SYNC_MANIFEST_ALLON.sha256 (run_arms_allon.py refuses any other change);
#   2. tests in the staged tree on the desktop python (import chain needs gguf-py from the deploy source);
#   3. tooling copied into RIG (check_admission_both.py from the RP rig dir, unchanged).
# Runs outside the rig lock; writes only under $STAGING, $RIG. Nothing in the deploy is touched.
set -euo pipefail
STAGING=/mnt/storage/s42-allon-20260924-staging
SOURCE=/mnt/storage/s42-trace-v2-20260921-prep/source
RIG=/home/zhihao/s42-allon-20260924-rig
export PYTHONDONTWRITEBYTECODE=1 LANG=C.UTF-8
mkdir -p "$RIG"
rsync -n -rc --itemize-changes --exclude __pycache__ --exclude reports --exclude '*.pyc' \
    "$STAGING/research_dev/scheduler/" "$SOURCE/research_dev/scheduler/" > "$RIG/SYNC_DRYRUN_ALLON.txt"
grep -E '^[<>]' "$RIG/SYNC_DRYRUN_ALLON.txt" | awk '{print $2}' | sort > "$RIG/SYNC_FILES_ALLON.txt"
: > "$RIG/SYNC_MANIFEST_ALLON.sha256"
while read -r path; do
  printf '%s  %s\n' "$(sha256sum "$STAGING/research_dev/scheduler/$path" | cut -d' ' -f1)" "$path" >> "$RIG/SYNC_MANIFEST_ALLON.sha256"
done < "$RIG/SYNC_FILES_ALLON.txt"
echo "manifest files: $(wc -l < "$RIG/SYNC_MANIFEST_ALLON.sha256")"
cat "$RIG/SYNC_FILES_ALLON.txt"
cp /home/zhihao/s42-reprovision-20260924-rig/check_admission_both.py "$RIG/"
cd "$STAGING"
export PYTHONPATH=$STAGING:$SOURCE/gguf-py
for t in test_prepare_trace_inputs_v2 test_dispatch_policy test_adaptive_server_probe test_adaptive_coherence \
         test_adaptive_evidence_fixes test_phone_resident_model_reprovision test_phone_reprovision_portfolio \
         test_two_phone_gaps test_campaign_inputs test_campaign_model_configuration; do
  printf '%-45s ' "$t"
  python3 -m unittest "research_dev.scheduler.tests.$t" 2>&1 | grep -E '^(Ran|OK|FAILED)' | tr '\n' ' '
  echo
done
