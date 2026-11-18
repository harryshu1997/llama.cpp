#!/bin/bash
# Copy the g161 run records + meter dirs from the new desktop to FCHLLX01 (same layout as the earlier arms).
set -u
R=/mnt/storage/s43-two-phone-eval-20260925
DST=/mnt/data_h16t/zs89458/xband-runs/second-desktop
for att in ${ATTS:-g161nph g16r1new}; do
  for d in $(ssh zhihao@172.21.122.109 "ls -d $R/inputs-*-$att 2>/dev/null"); do
    name=$(basename $d); mkdir -p $DST/$name/run-eval/run
    rsync -a --include='RESULT.json' --include='SCHEDULER_DECISION_LOG.json' --include='resource-samples.jsonl' --include='ADAPTIVE_DECODE_OBSERVATIONS.json' --exclude='*' zhihao@172.21.122.109:$d/run-eval/run/ $DST/$name/run-eval/run/
    rsync -a --include='runner.log' --exclude='*' zhihao@172.21.122.109:$d/run-eval/ $DST/$name/run-eval/ 2>/dev/null
    echo "pulled $name: $(ls $DST/$name/run-eval/run | tr '\n' ' ')"
  done
  for m in $(ssh zhihao@172.21.122.109 "ls -d $R/meter/$att-* 2>/dev/null"); do
    mname=$(basename $m); mkdir -p $DST/$mname; rsync -a zhihao@172.21.122.109:$m/ $DST/$mname/; echo "meter $mname: $(ls $DST/$mname | tr '\n' ' ')"
  done
done
