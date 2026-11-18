#!/bin/bash
S=5A040DLCH004ES; D=/data/local/tmp/s43-pixel-cl; DF=/sys/class/devfreq
adb -s $S push /tmp/pxcl/clgemv /tmp/pxcl/cllat /tmp/pxcl/clgemv.cl $D/ >/dev/null
echo "=== 1. cache sweep: re-reading a prefix of the weights back-to-back (a cache = bandwidth above 47 GB/s at small sizes)"
adb -s $S shell "cd $D && taskset 80 ./clgemv clgemv.cl cache_sweep 17408 5120 10 32 256 2>&1 | grep cache_sweep" | sed 's/.*"cache_sweep_MB":\([0-9.]*\).*"GBps_med":\([0-9.]*\),"GBps_min":\([0-9.]*\).*/\1 MB: \2 GBps (best \3)/'
echo "=== 2. production cadence: 23 layer calls per token, 2.5 ms host gap between calls, 150 ms between tokens (pinned to core 7)"
echo "governor clocks:"; adb -s $S shell "cd $D && taskset 80 ./cllat clgemv.cl 32 256 0 2.5 150 20"
adb -s $S shell "su -c 'echo 1094000000 > $DF/34f00000.gpu0/min_freq; echo 932000000 > $DF/memss_freq/min_freq'"
echo "GPU + memss clocks pinned high (pre-warmed):"; adb -s $S shell "cd $D && taskset 80 ./cllat clgemv.cl 32 256 0 2.5 150 20"
echo "governor clocks, 8 ms host gap (Gemma-like cadence):"; adb -s $S shell "su -c 'echo 396000000 > $DF/34f00000.gpu0/min_freq; echo 178000000 > $DF/memss_freq/min_freq'"; adb -s $S shell "cd $D && taskset 80 ./cllat clgemv.cl 32 256 0 8 150 15"
adb -s $S shell "su -c 'echo 1094000000 > $DF/34f00000.gpu0/min_freq; echo 932000000 > $DF/memss_freq/min_freq'"
echo "pinned, 8 ms host gap:"; adb -s $S shell "cd $D && taskset 80 ./cllat clgemv.cl 32 256 0 8 150 15"
adb -s $S shell "su -c 'echo 396000000 > $DF/34f00000.gpu0/min_freq; echo 178000000 > $DF/memss_freq/min_freq; echo restored gpu=\$(cat $DF/34f00000.gpu0/min_freq) memss=\$(cat $DF/memss_freq/min_freq)'"
echo "--- GPU clock trace during the governor-clock cadence run (50 Hz)"
adb -s $S shell "(for i in \$(seq 1 60); do cat $DF/34f00000.gpu0/cur_freq; sleep 0.05; done | sort -n | uniq -c | tr '\n' ';') & cd $D && taskset 80 ./cllat clgemv.cl 32 256 0 2.5 150 12 >/dev/null; wait; echo"
