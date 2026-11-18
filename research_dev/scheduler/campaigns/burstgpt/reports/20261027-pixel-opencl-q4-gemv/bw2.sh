#!/bin/bash
S=5A040DLCH004ES; D=/data/local/tmp/s43-pixel-cl; DF=/sys/class/devfreq
adb -s $S push /tmp/pxcl/clgemv /tmp/pxcl/clgemv.cl $D/ >/dev/null
run() { adb -s $S shell "cd $D && taskset 80 ./clgemv clgemv.cl $1 17408 5120 40 ${2:-32} ${3:-256}" 2>&1 | tail -1 | sed 's/.*"variant":"\([a-z0-9_]*\)".*"device_ms_med":\([0-9.]*\).*"GBps_med":\([0-9.]*\).*"wall_ms_batched":\([0-9.]*\).*/\1 device_ms=\2 GBps=\3 wall_batched=\4/'; }
echo "=== devfreq nodes and ranges"
adb -s $S shell "su -c 'for n in gmc_freq memss_freq fabhbw_freq 200c0780.dsufreq 34f00000.gpu0; do echo \$n cur=\$(cat $DF/\$n/cur_freq) min=\$(cat $DF/\$n/min_freq) max=\$(cat $DF/\$n/max_freq) avail=\$(cat $DF/\$n/available_frequencies | tr \" \" \",\"); done'" 2>&1 | cut -c1-230
echo "=== clocks granted DURING a GPU-only loop (sampled every 0.1 s, unpinned clocks)"
adb -s $S shell "su -c '(for i in \$(seq 1 30); do echo \$(cat $DF/gmc_freq/cur_freq) \$(cat $DF/memss_freq/cur_freq 2>/dev/null) \$(cat $DF/fabhbw_freq/cur_freq 2>/dev/null) \$(cat $DF/34f00000.gpu0/cur_freq); sleep 0.1; done | sort | uniq -c | sort -rn | head -4) & cd $D && taskset 80 ./clgemv clgemv.cl stream_read 17408 5120 1500 32 256 >/dev/null; wait'"
echo "=== baseline (governor clocks)"
run stream_read; run q4_global; run q4_global_u2; run q4_global 16 128; run q4_global_u2 16 128
echo "=== memory + fabric clocks pinned to max (root)"
adb -s $S shell "su -c 'for n in gmc_freq memss_freq fabhbw_freq; do m=\$(cat $DF/\$n/available_frequencies | tr \" \" \"\\n\" | sort -n | tail -1); echo \$m > $DF/\$n/min_freq; echo \$n pinned \$(cat $DF/\$n/min_freq); done; echo 1094000000 > $DF/34f00000.gpu0/min_freq'" 2>&1 | head -4
run stream_read; run q4_global; run q4_global_u2; run q4_global 16 128
adb -s $S shell "su -c '(for i in \$(seq 1 20); do echo \$(cat $DF/gmc_freq/cur_freq) \$(cat $DF/memss_freq/cur_freq 2>/dev/null) \$(cat $DF/fabhbw_freq/cur_freq 2>/dev/null); sleep 0.1; done | sort | uniq -c | sort -rn | head -3) & cd $D && taskset 80 ./clgemv clgemv.cl stream_read 17408 5120 1000 32 256 >/dev/null; wait'"
echo "=== restore"
adb -s $S shell "su -c 'for n in gmc_freq memss_freq fabhbw_freq; do m=\$(cat $DF/\$n/available_frequencies | tr \" \" \"\\n\" | sort -n | head -1); echo \$m > $DF/\$n/min_freq; echo \$n min=\$(cat $DF/\$n/min_freq); done; echo 396000000 > $DF/34f00000.gpu0/min_freq; echo gpu min=\$(cat $DF/34f00000.gpu0/min_freq)'" 2>&1 | head -4
