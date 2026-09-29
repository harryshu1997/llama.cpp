#!/bin/bash
B=/mnt/storage/s43-pixel-int2-20260924
ARMS="--arm desktop=$B/inputs-desktop-ii1/run-int2/run --arm op15_r1=$B/inputs-op15-ii1/run-int2/run --arm twophone_r1=$B/inputs-two-phone-ii1/run-int2/run --arm twophone_r2=$B/inputs-two-phone-ii2/run-int2/run --arm op15_r2=$B/inputs-op15-ii2/run-int2/run"
cd $B/analysis/int2 && python3 interval_energy_int2.py $ARMS --out $B/analysis/DEV2_MODEL_WINDOW_ENERGY.json > /dev/null
python3 - <<'PY'
import json
d = json.load(open("/mnt/storage/s43-pixel-int2-20260924/analysis/DEV2_MODEL_WINDOW_ENERGY.json"))
for label, rows in d.items():
    print("QWEN", label, rows["qwen"])
PY
cd $B/analysis/coherent && python3 analyze_allon.py $ARMS --out $B/analysis/DEV2_ALLON_STYLE.json --md $B/analysis/DEV2_ALLON_STYLE.md > /dev/null
for a in inputs-op15-ii1 inputs-two-phone-ii1 inputs-two-phone-ii2 inputs-op15-ii2; do
  for f in $B/$a/run-int2/run/large-model-*-hot-desktop.stderr; do
    echo "== $a $(basename $f)"
    grep -a S41SERVERFFNSHAPE $f | python3 -c '
import sys, json
for line in sys.stdin:
    d = json.loads(line.split(" ", 1)[1])
    print(d.get("helper", "op15"), "B" + str(d["tokens"]), d["calls"], round(d["rpc_mean_ms"], 1), round(d["compute_mean_ms"], 1))
'
  done
done
