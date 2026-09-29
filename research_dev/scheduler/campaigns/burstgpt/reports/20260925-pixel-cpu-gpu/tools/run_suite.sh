#!/bin/bash
# Desktop side: run one phone suite under the rig lock (caller wraps in flock). Usage: run_suite.sh NAME
set -uo pipefail
NAME=$1
A="adb -P 5037 -s 5A040DLCH004ES"
HOST=/mnt/storage/s43-pixel-cpugpu-20260925-v1
PH=/data/local/tmp/s43-pixel-cpugpu-20260925-v1
OUT=$HOST/runs/$NAME
mkdir -p $OUT || exit 2
$A shell dumpsys battery < /dev/null > $OUT/HOST_BATTERY_BEFORE.txt
date -u +%FT%TZ > $OUT/HOST_START.txt
$A push $HOST/suites/$NAME/RUN_PHONE.sh $PH/suite-$NAME.sh > /dev/null || exit 3
$A shell "su -c 'sh $PH/suite-$NAME.sh'" < /dev/null > $OUT/SUITE.log 2>&1
echo $? > $OUT/SUITE_EXIT.txt
date -u +%FT%TZ > $OUT/HOST_END.txt
$A shell dumpsys battery < /dev/null > $OUT/HOST_BATTERY_AFTER.txt
SUITE_DIR=$(grep -o 'RUN=$PH/runs/[^ ]*' $HOST/suites/$NAME/RUN_PHONE.sh | head -1 | sed 's#RUN=$PH/runs/##')
$A pull $PH/runs/$SUITE_DIR $OUT/ > /dev/null
$A shell "ps -A -o PID,ARGS | grep '[l]lama-ffn'" < /dev/null > $OUT/REMAINING_WORKERS.txt
cat $OUT/SUITE.log
