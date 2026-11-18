#!/bin/bash
# switch_pixel_evidence.sh EVIDENCE_JSON TAG: point the measured-placement derivation (W/DERIVE_AOA_EVIDENCE and the
# --helper-evidence line of W/DERIVE_MP_ARGS) at another qualified Pixel bundle; backups keep the previous pointers as *.TAG.
set -eu
EV=$1; TAG=$2; W=/mnt/storage/s43-ws12-pixel-placement-20261002
[ -f "$EV" ] || { echo "missing $EV"; exit 2; }
cp -n $W/DERIVE_AOA_EVIDENCE $W/DERIVE_AOA_EVIDENCE.$TAG
cp -n $W/DERIVE_MP_ARGS $W/DERIVE_MP_ARGS.$TAG
echo "$EV" > $W/DERIVE_AOA_EVIDENCE
python3 - "$EV" <<PY
import sys, pathlib
p = pathlib.Path("$W/DERIVE_MP_ARGS"); lines = p.read_text().splitlines(); out = []
for i, line in enumerate(lines):
    if i and lines[i-1].strip() == "--helper-evidence" and line.startswith("qwen3-14b-q4km-dequant-f16:pixel10pro-phone="):
        line = "qwen3-14b-q4km-dequant-f16:pixel10pro-phone=" + sys.argv[1]
    out.append(line)
p.write_text("\n".join(out) + "\n"); print("DERIVE_MP_ARGS helper-evidence ->", sys.argv[1])
PY
echo "DERIVE_AOA_EVIDENCE -> $(cat $W/DERIVE_AOA_EVIDENCE)"
