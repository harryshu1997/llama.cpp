#!/bin/bash
# v16: adaptive arm then matched desktop-baseline control, same prematerialized catalog.
DEPLOY=/mnt/storage/s42-phase-concurrency-20260911-v1-nEEi6I
L=$DEPLOY/deploy/research_dev/scheduler/campaigns/burstgpt/launch.py
run_arm() {
  local tag=$1 C=$DEPLOY/inputs/sparse24-$1-campaign.json
  echo "== [$tag] resolve $(date +%T)"; python3 $L $C $DEPLOY/resolve-sparse24-$tag --resolve-only || { echo "RESOLVE_FAILED $tag"; exit 1; }
  echo "== [$tag] preflight $(date +%T)"; python3 $L $C $DEPLOY/preflight-sparse24-$tag --preflight-only || { echo "PREFLIGHT_FAILED $tag"; exit 1; }
  python3 - $tag <<'PY' || { echo "PREFLIGHT_NOT_CLEAN $tag"; exit 1; }
import json, sys
p=json.load(open(f'/mnt/storage/s42-phase-concurrency-20260911-v1-nEEi6I/preflight-sparse24-{sys.argv[1]}/PHYSICAL_PREFLIGHT.json'))
print("preflight", sys.argv[1], p['status'], "shapes", p.get('request_shapes'))
sys.exit(0 if p['status']=='PASS' and (p.get('request_shapes') or {}).get('checked')==24 and not p['request_shapes']['unsupported'] else 1)
PY
  echo "== [$tag] prewarm $(date +%T)"; cat /home/zhihao/models/gemma-4-12B-Q40-dequant-f16.gguf > /dev/null; cat /home/zhihao/models/Llama-3.2-1B-Instruct-Q4_0.gguf > /dev/null
  echo "== [$tag] run $(date +%T)"; python3 $L $C $DEPLOY/sparse24-$tag || { echo "RUN_FAILED $tag"; exit 1; }
  echo "== [$tag] done $(date +%T)"
}
run_arm v16 && run_arm v16c
echo "== ALL DONE $(date +%T)"
