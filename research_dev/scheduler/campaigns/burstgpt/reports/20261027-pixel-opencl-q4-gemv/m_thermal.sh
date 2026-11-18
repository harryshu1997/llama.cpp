#!/bin/bash
S=5A040DLCH004ES; D=/data/local/tmp/s43-pixel-cl; G=/sys/class/devfreq/34f00000.gpu0
adb -s $S push /tmp/pxcl/clgemv /tmp/pxcl/clgemv.cl $D/ >/dev/null
echo "=== batch rows M=1,2,4 (q4_global_m, pinned core 7)"
for m in 1 2 4; do for cfg in "32 256" "16 128"; do set -- $cfg; adb -s $S shell "cd $D && taskset 80 ./clgemv clgemv.cl q4_global_m 17408 5120 30 $1 $2 $m" 2>&1 | tail -1 | sed 's/.*"M":\([0-9]*\).*"R":\([0-9]*\),"WG":\([0-9]*\).*"device_ms_med":\([0-9.]*\).*"GBps_med":\([0-9.]*\).*"wall_ms_batched":\([0-9.]*\).*"rel_l2_vs_quantized_ref":\([0-9.e+-]*\).*/M=\1 R=\2 WG=\3 device_ms=\4 GBps(weights)=\5 wall_batched=\6 rel_l2=\7/'; done; done
echo "=== thermal: 60 s continuous GPU layers (gate/up/down loop), sampling clock, temp, battery current every 2 s"
adb -s $S shell "(for i in \$(seq 1 32); do echo t=\$((i*2)) gpu=\$(cat $G/cur_freq) temp=\$(cat /sys/class/power_supply/battery/temp) batt_uA=\$(cat /sys/class/power_supply/battery/current_now) thermal=\$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null); sleep 2; done) & cd $D && taskset 80 ./clgemv clgemv.cl q4_global 17408 5120 40000 32 256 | tail -1 | cut -c1-230; wait" 2>&1 | grep -v "^$"
