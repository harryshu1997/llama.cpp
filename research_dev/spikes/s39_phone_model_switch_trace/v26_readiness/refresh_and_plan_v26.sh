#!/bin/sh
# Refresh topology + V2.6 lock + V2.4.1 plan set against the current boot IDs.
# Every acquisition attempt reboots the phones, so all three must be renewed
# together before the next attempt.
set -eu

S39=/home/myid/zs89458/Documents/llama.cpp-release/research_dev/spikes/s39_phone_model_switch_trace
V26=$S39/v26_readiness
DESKTOP=zhihao@172.20.74.85
MIRROR=/home/zhihao/s39-v26-a-only/repo-v1/s39
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
TOPO=$V26/results/no_model_topology_$STAMP.json

export PYTHONDONTWRITEBYTECODE=1

echo "=== 1/4 capture fresh topology on the desktop"
ssh -o BatchMode=yes "$DESKTOP" \
    "cd $MIRROR/v24_readiness && /usr/bin/python3 -B \
     desktop_deployment_v1/verify_topology_v1.py capture-topology \
     --contract \$PWD/CP0_R1_EVIDENCE_CONTRACT_V2_4.json \
     --output /home/zhihao/s39-v26-a-only/topology_$STAMP.json"
scp -o BatchMode=yes \
    "$DESKTOP:/home/zhihao/s39-v26-a-only/topology_$STAMP.json" "$TOPO"
echo "topology: $TOPO"

echo "=== 2/4 clear superseded plan artifacts"
rm -f "$S39"/v24_readiness/results/prephase_20260726T0915Z/cuda-route-launch.json \
      "$S39"/v24_readiness/results/prephase_20260726T0915Z/joint-capture-plan.json \
      "$S39"/v24_readiness/results/prephase_20260726T0915Z/phone-route-launch.json \
      "$S39"/v24_readiness/results/prephase_20260726T0915Z/runtime-bundle-plan.json \
      "$S39"/v24_readiness/results/prephase_20260726T0915Z/prospective-runtime-root.json
ssh -o BatchMode=yes "$DESKTOP" \
    "cd $MIRROR/v24_readiness/results/prephase_20260726T0915Z && \
     rm -f cuda-route-launch.json joint-capture-plan.json \
           phone-route-launch.json runtime-bundle-plan.json \
           prospective-runtime-root.json; \
     find /home/zhihao/s39-v26-a-only/v24-plan-inputs-v1 -maxdepth 1 \
          -name '*.json' -delete 2>/dev/null; exit 0"

echo "=== 3/4 materialize V2.6 inventory + phase lock"
cd "$V26"
python3 materialize_production_v26.py materialize --execute \
    --confirm RUN_CP0_R1_V26_PRODUCTION_MATERIALIZATION \
    --topology "$TOPO" > /tmp/v26_materialize.json
V26_ROOT=$(python3 -c "import json,sys; print(json.load(open('/tmp/v26_materialize.json'))['output_root'])")
echo "v26 root: $V26_ROOT"

echo "=== 4/4 materialize the V2.4.1 plan set"
python3 materialize_v24_plan_set_v26.py materialize --execute \
    --confirm RUN_CP0_R1_V26_V24_PLAN_SET \
    --v26-root "$V26_ROOT" --topology "$TOPO" > /tmp/v24_plan_set.json
python3 -c "import json; print('plan set:', json.load(open('/tmp/v24_plan_set.json'))['validation']['status'])"
echo "REFRESH_OK"
