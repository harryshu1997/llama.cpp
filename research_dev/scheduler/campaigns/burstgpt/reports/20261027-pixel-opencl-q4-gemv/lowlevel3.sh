#!/bin/bash
S=5A040DLCH004ES; D=/data/local/tmp/s43-pixel-cl; DF=/sys/class/devfreq
gpu() { adb -s $S shell "cd $D && taskset 80 ./clgemv clgemv.cl stream_read16 17408 5120 40 64 512 | tail -1" | sed 's/.*"device_ms_med":\([0-9.]*\).*"GBps_med":\([0-9.]*\).*/device_ms=\1 GBps=\2/'; }
echo "=== PM QoS domains and OPP tables"
adb -s $S shell "su -c 'ls /sys/kernel/debug/google_pm_qos/ | tr \"\n\" \" \"; echo; for d in gmc_freq irm_gmc_freq memss_freq; do echo \"-- opp $d:\"; ls /sys/kernel/debug/opp/platform-$d | tr \"\n\" \" \" | cut -c1-400; echo; done'" | cut -c1-500
echo "=== PM QoS requests at idle (gmc / irm_gmc / memss)"
adb -s $S shell "su -c 'for d in gmc_freq irm_gmc_freq memss_freq; do echo \"-- $d:\"; cat /sys/kernel/debug/google_pm_qos/$d | head -30; done'" | cut -c1-200
echo "=== PM QoS requests DURING a GPU stream"
adb -s $S shell "su -c '(cd $D && taskset 80 ./clgemv clgemv.cl stream_read16 17408 5120 3000 64 512 >/dev/null) & sleep 2; for d in gmc_freq irm_gmc_freq memss_freq; do echo \"-- $d:\"; cat /sys/kernel/debug/google_pm_qos/$d | head -30; done; wait'" | cut -c1-200
echo "=== port hypothesis: GPU stream vs fabhbw / memss clock caps"
echo "baseline: $(gpu)"
adb -s $S shell "su -c 'echo 480000000 > $DF/fabhbw_freq/max_freq; echo 480000000 > $DF/fabhbw_freq/min_freq; cat $DF/fabhbw_freq/cur_freq'"; echo "fabhbw 480 MHz: $(gpu)"
adb -s $S shell "su -c 'echo 343000000 > $DF/fabhbw_freq/max_freq; echo 343000000 > $DF/fabhbw_freq/min_freq; cat $DF/fabhbw_freq/cur_freq'"; echo "fabhbw 343 MHz: $(gpu)"
adb -s $S shell "su -c 'echo 711000000 > $DF/fabhbw_freq/max_freq; echo 172000000 > $DF/fabhbw_freq/min_freq; echo 466000000 > $DF/memss_freq/max_freq; echo 466000000 > $DF/memss_freq/min_freq; cat $DF/memss_freq/cur_freq'"; echo "memss 466 MHz (fabhbw restored): $(gpu)"
adb -s $S shell "su -c 'echo 233000000 > $DF/memss_freq/max_freq; echo 233000000 > $DF/memss_freq/min_freq; cat $DF/memss_freq/cur_freq'"; echo "memss 233 MHz: $(gpu)"
adb -s $S shell "su -c 'echo 932000000 > $DF/memss_freq/max_freq; echo 178000000 > $DF/memss_freq/min_freq; echo 1536000000 > $DF/gmc_freq/max_freq; echo 1536000000 > $DF/gmc_freq/min_freq; cat $DF/gmc_freq/cur_freq'"; echo "gmc devfreq 1536 MHz (memss restored): $(gpu)"
adb -s $S shell "su -c 'echo 4224000000 > $DF/gmc_freq/max_freq; echo 409000000 > $DF/gmc_freq/min_freq; echo restored: fabhbw \$(cat $DF/fabhbw_freq/min_freq)-\$(cat $DF/fabhbw_freq/max_freq) memss \$(cat $DF/memss_freq/min_freq)-\$(cat $DF/memss_freq/max_freq) gmc \$(cat $DF/gmc_freq/min_freq)-\$(cat $DF/gmc_freq/max_freq)'"
echo "final check: $(gpu)"
