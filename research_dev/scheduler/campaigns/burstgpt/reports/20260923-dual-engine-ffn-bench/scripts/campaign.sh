#!/bin/bash
# desktop-side campaign driver; one flock per repetition
set -u
cd /home/zhihao/s43-dual-ffn-bench-20260923
LOCK=/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
A="adb -P 5037 -s 3C15AU002CL00000"
mkdir -p logs
FULL="1.0 0.95 0.8 0.7 0.5 0.0"
M1ONLY="0.9 0.85 0.75 0.6"
for rep in 1 2 3; do
  if [ $rep = 1 ]; then REF=""; else REF="--no-cpu-ref"; fi
  # rotate order per rep to spread DVFS/thermal drift
  case $rep in
    1) ORDER="1.0 0.9 0.8 0.7 0.6 0.5 0.0 0.95 0.85 0.75";;
    2) ORDER="0.5 0.75 1.0 0.0 0.85 0.6 0.95 0.7 0.9 0.8";;
    3) ORDER="0.0 0.8 0.6 0.95 0.9 0.5 0.75 1.0 0.7 0.85";;
  esac
  flock -w 1800 $LOCK bash -c "
    for f in $ORDER; do
      case \" $FULL \" in *\" \$f \"*) B=1,4,8;; *) B=1;; esac
      tag=r${rep}_f\${f}
      { echo \"=== \$tag \$(date -Is)\"; $A shell 'dumpsys battery | grep -E \"temperature|level\"; cat /sys/class/kgsl/kgsl-3d0/gpuclk 2>/dev/null; cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null'; } > logs/\$tag.log 2>&1
      ./phone_run.sh \$tag --frac \$f --layers 6 --layer-start 10 --sweeps 50 --warmup 3 --batches \$B $REF >> logs/\$tag.log 2>&1
      rc=\$?; echo \"rc=\$rc\" >> logs/\$tag.log
      [ \$rc = 9 ] && exit 9
      sleep 15
    done
  " || { echo "rep $rep aborted rc=$?" >> logs/campaign.status; exit 1; }
  echo "rep $rep done $(date -Is)" >> logs/campaign.status
done
echo ALLDONE >> logs/campaign.status
