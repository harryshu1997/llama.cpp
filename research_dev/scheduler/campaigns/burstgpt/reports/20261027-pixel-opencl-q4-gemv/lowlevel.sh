#!/bin/bash
S=5A040DLCH004ES; D=/data/local/tmp/s43-pixel-cl; DF=/sys/class/devfreq
adb -s $S push /tmp/pxcl/clgemv /tmp/pxcl/clgemv.cl $D/ >/dev/null
echo "=== 1. governors"
adb -s $S shell "su -c 'for n in gmc_freq memss_freq fabhbw_freq 200c0780.dsufreq 34f00000.gpu0; do echo \$n: gov=\$(cat $DF/\$n/governor) avail_gov=\$(cat $DF/\$n/available_governors | tr \" \" \",\"); done'"
echo "=== 2. trans_stat delta over a 4 s GPU stream (governor clocks)"
adb -s $S shell "su -c 'cat $DF/gmc_freq/trans_stat > /data/local/tmp/ts_a.txt; cat $DF/fabhbw_freq/trans_stat > /data/local/tmp/tsf_a.txt; cat $DF/memss_freq/trans_stat > /data/local/tmp/tsm_a.txt'"
adb -s $S shell "cd $D && taskset 80 ./clgemv clgemv.cl stream_read 17408 5120 4000 32 256 | tail -1 | cut -c1-120"
adb -s $S shell "su -c 'cat $DF/gmc_freq/trans_stat > /data/local/tmp/ts_b.txt; cat $DF/fabhbw_freq/trans_stat > /data/local/tmp/tsf_b.txt; cat $DF/memss_freq/trans_stat > /data/local/tmp/tsm_b.txt'"
adb -s $S shell "su -c 'echo gmc_freq time-in-state delta ms:; paste /data/local/tmp/ts_a.txt /data/local/tmp/ts_b.txt | grep -E \"^ *\\\\*? *[0-9]\" | awk \"{n=NF/2; if (\\\$1==\\\"*\\\"){f=\\\$2;a=\\\$(n);b=\\\$NF} else {f=\\\$1;a=\\\$(n);b=\\\$NF}; if (b-a>0) print f, b-a}\"'" 2>&1 | head -20
echo "--- raw gmc trans_stat tail (after):"; adb -s $S shell "su -c 'tail -16 /data/local/tmp/ts_b.txt'" | cut -c1-200
echo "=== 3. same with gmc/memss/fabhbw min_freq pinned to max"
adb -s $S shell "su -c 'for n in gmc_freq memss_freq fabhbw_freq; do m=\$(cat $DF/\$n/available_frequencies | tr \" \" \"\\n\" | sort -n | tail -1); echo \$m > $DF/\$n/min_freq; echo \$n min=\$(cat $DF/\$n/min_freq) cur=\$(cat $DF/\$n/cur_freq); done; cat $DF/gmc_freq/trans_stat > /data/local/tmp/ts_c.txt'"
adb -s $S shell "cd $D && taskset 80 ./clgemv clgemv.cl stream_read 17408 5120 4000 32 256 | tail -1 | cut -c1-120"
adb -s $S shell "su -c 'cat $DF/gmc_freq/trans_stat > /data/local/tmp/ts_d.txt; tail -16 /data/local/tmp/ts_d.txt; for n in gmc_freq memss_freq fabhbw_freq; do m=\$(cat $DF/\$n/available_frequencies | tr \" \" \"\\n\" | sort -n | head -1); echo \$m > $DF/\$n/min_freq; done; echo restored'" | cut -c1-200
echo "=== 4. ftrace clock_set_rate during a GPU stream (atrace freq)"
adb -s $S shell "su -c 'atrace -t 4 -b 8192 freq -o /data/local/tmp/at.txt >/dev/null 2>&1 & sleep 0.5; cd $D && taskset 80 ./clgemv clgemv.cl stream_read 17408 5120 2500 32 256 >/dev/null; wait; grep -o \"clock_set_rate: [a-zA-Z0-9_]* state=[0-9]*\" /data/local/tmp/at.txt | sort | uniq -c | sort -rn | head -24'" | cut -c1-140
echo "=== 5. device tree / dmesg hints on the DRAM"
adb -s $S shell "su -c 'ls /proc/device-tree | grep -i \"dram\|memory\|gmc\|lpddr\|ddr\" | head; find /proc/device-tree -maxdepth 2 -iname \"*dram*\" -o -maxdepth 2 -iname \"*lpddr*\" -o -maxdepth 2 -iname \"*gmc*\" 2>/dev/null | head -8; dmesg 2>/dev/null | grep -i \"lpddr\|dram freq\|gmc\|ddr.*MHz\|memory.*MT/s\" | head -8'" | cut -c1-160
echo "=== 6. wider loads"
for cfg in "stream_read 32 256" "stream_read16 32 256" "stream_read16 16 128" "stream_read16 64 512"; do set -- $cfg; adb -s $S shell "cd $D && taskset 80 ./clgemv clgemv.cl $1 17408 5120 40 $2 $3 2>&1 | tail -1" | sed 's/.*"variant":"\([a-z0-9_]*\)".*"WG":\([0-9]*\).*"device_ms_med":\([0-9.]*\).*"GBps_med":\([0-9.]*\).*/\1 WG=\2 device_ms=\3 GBps=\4/' | cut -c1-120; done
