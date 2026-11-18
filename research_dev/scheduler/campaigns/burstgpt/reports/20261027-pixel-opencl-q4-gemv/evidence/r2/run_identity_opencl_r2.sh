#!/bin/bash
# Pixel OpenCL worker: server token identity + FFN numerical rows over the AOA bridge (same setup as pixel_requal_5070ti.sh).
set -u
R=/mnt/storage/s43-two-phone-eval-20260925; D=/mnt/storage/s42-trace-v2-20260921-prep; A=/mnt/storage/s43-pixel-aoa-20261001-v1
LOG=$A/B-opencl-identity.log; LOCK=/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
OUT=${1:-$A/server-identity-opencl-1-23-5070ti-r2}
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
wait_sockets(){ for i in $(seq 1 90); do ! ss -tan 2>/dev/null | grep -qE ":26991 |:26995 |:26996 |:27170 |:27171 " && ! pgrep -x llama-server >/dev/null && ! pgrep -f "aoa_bridge|qualify_pixel_server" >/dev/null && return 0; sleep 2; done; log "sockets/processes still busy"; return 1; }
export PYTHONPATH=$D/source PYTHONDONTWRITEBYTECODE=1 LANG=C.UTF-8 LD_LIBRARY_PATH=/mnt/storage/s21_deps/cuda-13.2.1/lib:$D/cuda-build/bin
export S43_PIXEL_USB_SYSFS=2-2 S43_SERVER_IDENTITY_DIR=server-identity-r8-5070ti S43_TRANSPORT_IDENTITY=$D/TRANSPORT_QUALIFICATION_IDENTITY_FUSED3_20261005b.json
cd $D/source
wait_sockets || exit 1
rm -rf $OUT
log "identity run start -> $OUT"
flock -w 7200 $LOCK python3 -m research_dev.scheduler.campaigns.burstgpt.tools.qualify_pixel_server_transport --config ${CONFIG:-$A/CONFIG-opencl-1-23-r2.json} --output $OUT --transport aoa-bridge --aoa-bridge $A/AOA_BRIDGE.5070ti.json --forward-port 26991 >> $LOG 2>&1
rc=$?; log "identity run rc=$rc"
python3 - <<PY >> $LOG 2>&1
import json, pathlib
out = pathlib.Path("$OUT")
for name in ("RESULT.json", "FFN_NUMERICS.json"):
    p = out / name
    if p.exists():
        d = json.load(open(p)); print(name, {k: d[k] for k in d if k in ("status", "token_identity", "phone_calls", "call_columns", "output_tokens_each", "worst_rel_l2", "worst_rel_l2_f32_reference", "threshold", "calls")})
    else: print(name, "missing")
if (out / "FAILURE.json").exists(): print("FAILURE", open(out / "FAILURE.json").read()[:600])
PY
exit $rc
