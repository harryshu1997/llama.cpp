#!/bin/bash
# Re-derive every table from the raw runs: calibration -> per-run analysis -> fits -> markdown fragments.
#   make_results.sh    (writes $B/*/RESULTS.json and $R/tables/*)
set -u
HERE=$(cd "$(dirname "$0")" && pwd); R=$(dirname "$HERE")
B=/mnt/data_h16t/zs89458/xband-runs/stream-energy-20261110
mkdir -p $R/tables
ratio(){ python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['ratio_for_power'])" "$1"; }
# capacity scale = design capacity x SOH / gauge full-charge capacity (read from /sys/class/oplus_chg/battery)
python3 $HERE/calibrate_phone.py $B/op15/session-power.txt --runs $B/op15 --capacity-scale 1.012 --json $B/op15/CALIBRATION.json > /dev/null
python3 $HERE/calibrate_phone.py $B/op12/session-power.txt --runs $B/op12 --capacity-scale 1.438 --json $B/op12/CALIBRATION.json > /dev/null
python3 $HERE/analyze_phone.py $B/op15 --ratio $(ratio $B/op15/CALIBRATION.json) --cap-policy op15 --json $B/op15/RESULTS.json > $R/tables/op15.txt
python3 $HERE/analyze_phone.py $B/op12 --ratio $(ratio $B/op12/CALIBRATION.json) --cap-policy none --json $B/op12/RESULTS.json > $R/tables/op12.txt
PHONES=(--device "OnePlus 15 NPU=$B/op15/RESULTS.json" --device "OnePlus 12 NPU=$B/op12/RESULTS.json")
PTABLES=(--phone "OnePlus 15 NPU (Hexagon v81)=$B/op15/RESULTS.json" --phone "OnePlus 12 NPU (Hexagon v75)=$B/op12/RESULTS.json")
if [ -d $B/op11 ]; then   # no root: bandwidth and temperature only
  python3 $HERE/analyze_phone.py $B/op11 --cap-policy none --hot-zone nspss-0 --json $B/op11/RESULTS.json > $R/tables/op11.txt
  PHONES+=(--device "OnePlus 11 NPU=$B/op11/RESULTS.json")
  PTABLES+=(--phone "OnePlus 11 NPU (Hexagon v73, no root: no power)=$B/op11/RESULTS.json")
fi
if [ -d $B/pixel ]; then  # max77779 gauge: current in uA, charging positive, refresh < 0.5 s
  PX="--battery-ma-factor -0.001"
  [ -f $B/pixel/session-power.txt ] && python3 $HERE/calibrate_phone.py $B/pixel/session-power.txt --runs $B/pixel $PX --json $B/pixel/CALIBRATION.json > /dev/null
  PR=1.0; [ -f $B/pixel/CALIBRATION.json ] && PR=$(ratio $B/pixel/CALIBRATION.json)
  python3 $HERE/analyze_phone.py $B/pixel --ratio $PR --cap-policy none $PX --refresh-s 0.5 --hot-zone BIG --skin-zone quiet_therm --json $B/pixel/RESULTS.json > $R/tables/pixel.txt
  PHONES+=(--device "Pixel 10 Pro CPU=$B/pixel/RESULTS.json")
  PTABLES+=(--phone "Pixel 10 Pro CPU (6 threads, f16)=$B/pixel/RESULTS.json")
  if [ -d $B/pixel-q4k ]; then
    python3 $HERE/analyze_phone.py $B/pixel-q4k --ratio $PR --cap-policy none $PX --refresh-s 0.5 --hot-zone BIG --skin-zone quiet_therm --json $B/pixel-q4k/RESULTS.json > $R/tables/pixel-q4k.txt
  fi
fi
DESK=$B/desktop-cpu; [ -d $DESK ] || DESK=$B/desktop-cpu-norapl
SERV=${SERVER_CPU_DIR:-$B/server-cpu}; [ -d $SERV ] || SERV=$B/server-cpu-norapl
python3 $HERE/analyze_host.py $DESK --json $DESK/RESULTS.json > $R/tables/desktop.txt
python3 $HERE/analyze_host.py $SERV --json $SERV/RESULTS.json > $R/tables/server.txt
HOSTS=(--device "Desktop CPU=$DESK/RESULTS.json")
SHOSTS=(--device "Server CPU=$SERV/RESULTS.json")
GTABLES=()
if [ -d $B/desktop-gpu ]; then   # GPU board power from NVML; weights resident in GPU memory
  python3 $HERE/analyze_gpu.py $B/desktop-gpu --json $B/desktop-gpu/RESULTS.json > $R/tables/desktop-gpu.txt
  HOSTS=(--device "Desktop GPU=$B/desktop-gpu/RESULTS.json" "${HOSTS[@]}")
  GTABLES=(--gpu "Desktop GPU (RTX 4060 Ti 16 GB, CUDA)=$B/desktop-gpu/RESULTS.json")
fi
if [ -d $B/server-gpu ]; then    # one RTX A6000 of FCHLLX01 (nvidia-smi index 1)
  python3 $HERE/analyze_gpu.py $B/server-gpu --json $B/server-gpu/RESULTS.json > $R/tables/server-gpu.txt
  SHOSTS=(--device "Server GPU=$B/server-gpu/RESULTS.json" "${SHOSTS[@]}")
  GTABLES+=(--gpu "Server GPU (RTX A6000 48 GB, CUDA)=$B/server-gpu/RESULTS.json")
fi
HOSTS+=("${SHOSTS[@]}")   # order: desktop GPU, desktop CPU, server GPU, server CPU
python3 $HERE/report.py --bytes $B/MODEL_BYTES.json "${PHONES[@]}" "${HOSTS[@]}" --md $R/tables/FINAL.md --json $R/tables/SUMMARY.json > $R/tables/FIT.txt
python3 $HERE/report.py --bytes $B/MODEL_BYTES.json --key net "${PHONES[@]}" "${HOSTS[@]}" --md $R/tables/FINAL_NET.md --json $R/tables/SUMMARY_NET.json > $R/tables/FIT_NET.txt
python3 $HERE/tables.py "${PTABLES[@]}" "${GTABLES[@]}" --host "Desktop CPU (i9-12900K, 8 threads)=$DESK/RESULTS.json" --host "Server CPU (Threadripper PRO 5995WX, 64 threads)=$SERV/RESULTS.json" > $R/tables/RUNS.md
if [ -d $B/server-cpu-20261112 ]; then   # repeat of the server CPU leg (11-12, after the A6000 GPU leg): a check, not a table source
  python3 $HERE/analyze_host.py $B/server-cpu-20261112 --json $B/server-cpu-20261112/RESULTS.json > $R/tables/server-repeat.txt
  python3 $HERE/report.py --bytes $B/MODEL_BYTES.json --device "Server CPU repeat=$B/server-cpu-20261112/RESULTS.json" --md $R/tables/SERVER_CPU_REPEAT.md --json $R/tables/SERVER_CPU_REPEAT.json > /dev/null
fi
cp $B/MODEL_BYTES.json $R/tables/MODEL_BYTES.json
tail -n +15 $R/tables/FIT.txt
