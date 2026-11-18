#!/bin/bash
# after_identity_opencl.sh: build the Pixel OpenCL evidence bundle from the passed identity run, point the measured-placement
# derivation at it, and launch the 1x fleet pair g16 (desktop + derived-mpaoa).
set -u
R=/mnt/storage/s43-two-phone-eval-20260925; D=/mnt/storage/s42-trace-v2-20260921-prep; A=/mnt/storage/s43-pixel-aoa-20261001-v1
W=/mnt/storage/s43-ws12-pixel-placement-20261002; RUN=${RUN:-$A/server-identity-opencl-1-23-5070ti-r2}; EV=${EV:-$A/evidence-aoa-opencl-1-23-5070ti-r2}
LOG=$A/B-opencl-identity.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
export PYTHONPATH=$D/source PYTHONDONTWRITEBYTECODE=1 LANG=C.UTF-8
python3 -c "import json,sys; r=json.load(open('$RUN/RESULT.json')); sys.exit(0 if r['status']=='PASS' and all(r['token_identity']) else 1)" || { log "identity run not PASS; stop"; exit 1; }
rm -rf $EV
cd $D/source
python3 -m research_dev.scheduler.campaigns.burstgpt.tools.prepare_aoa_evidence rebind-worker --evidence $A/evidence-aoa-1-23-5070ti/PIXEL_EVIDENCE_AOA.json --server-identity $RUN --shard-index $W/pixel-1-23/PIXEL_FFN_SHARDS.json --output-dir $EV >> $LOG 2>&1 || { log "rebind-worker FAILED"; tail -5 $LOG; exit 1; }
log "evidence: $EV/PIXEL_EVIDENCE_AOA.json"
python3 - <<PY | tee -a $LOG
import json; d=json.load(open("$EV/PIXEL_EVIDENCE_AOA.json")); w=d["worker"]; k=[r for r in d["profile_fragment"]["kernels"] if r["kernel_id"].endswith(":ffn")][0]
n=json.load(open(d["receipt_paths"]["numerical-rows-1-2-4"])); s=json.load(open(d["receipt_paths"]["server-token-identity"]))
print("worker", w["worker_path"], "| env +", {k2:v for k2,v in w["worker_environment"].items() if k2.startswith("S43_PIXEL_OPENCL") or k2.startswith("S43_PIXEL_CL")})
print("numerics", n["status"], n["policy"], "worst", n["worst_rel_l2"], "f32", n["worst_rel_l2_f32_reference"], "| identity", s["status"], s["phone_calls"], "calls,", s["tokens_each"], "tokens each")
print("ffn kernel rate", k["effective_bytes_per_s"], "B/s; full-width compute", d["worker_rebound_from"]["kernel_rate"])
PY
bash $R/rig/switch_pixel_evidence.sh $EV/PIXEL_EVIDENCE_AOA.json cpu-path | tee -a $LOG
bash $R/rig/${LAUNCH:-launch_g16r_opencl.sh} 1 | tee -a $LOG
