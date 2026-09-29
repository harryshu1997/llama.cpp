#!/bin/bash
DEPLOY=/mnt/storage/s42-phase-concurrency-20260911-v1-nEEi6I
L=$DEPLOY/deploy/research_dev/scheduler/campaigns/burstgpt/launch.py
C=$DEPLOY/inputs/sparse24-v15-control-campaign.json
echo "== resolve $(date +%T)"
python3 $L $C $DEPLOY/resolve-sparse24-v15c --resolve-only || { echo RESOLVE_FAILED; exit 1; }
echo "== preflight $(date +%T)"
python3 $L $C $DEPLOY/preflight-sparse24-v15c --preflight-only || { echo PREFLIGHT_FAILED; exit 1; }
python3 - <<'PY' || { echo PREFLIGHT_NOT_CLEAN; exit 1; }
import json,sys
p=json.load(open('/mnt/storage/s42-phase-concurrency-20260911-v1-nEEi6I/preflight-sparse24-v15c/PHYSICAL_PREFLIGHT.json'))
print("preflight", p['status'], "shapes", p.get('request_shapes'))
sys.exit(0 if p['status']=='PASS' and (p.get('request_shapes') or {}).get('checked')==24 and not p['request_shapes']['unsupported'] else 1)
PY
echo "== prewarm $(date +%T)"
cat /home/zhihao/models/gemma-4-12B-Q40-dequant-f16.gguf > /dev/null; cat /home/zhihao/models/Llama-3.2-1B-Instruct-Q4_0.gguf > /dev/null
echo "== run $(date +%T)"
python3 $L $C $DEPLOY/sparse24-v15c
echo "== done $(date +%T) exit=$?"
