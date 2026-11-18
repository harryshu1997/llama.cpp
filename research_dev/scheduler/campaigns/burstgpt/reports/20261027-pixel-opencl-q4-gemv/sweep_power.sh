#!/bin/bash
# Work-group sweep for q4_global, then sustained power of the GPU kernel vs idle (battery current/voltage via sysfs).
S=5A040DLCH004ES; D=/data/local/tmp/s43-pixel-cl
echo "=== sweep q4_global 17408x5120"
for cfg in "4 128" "8 128" "16 128" "8 64" "8 256" "16 256" "32 256"; do
  set -- $cfg
  adb -s $S shell "cd $D && ./clgemv clgemv.cl q4_global 17408 5120 30 $1 $2" 2>&1 | tail -1 | sed 's/.*"R":\([0-9]*\),"WG":\([0-9]*\).*"device_ms_med":\([0-9.]*\).*"GBps_med":\([0-9.]*\).*"wall_ms_batched":\([0-9.]*\).*/R=\1 WG=\2 device_ms=\3 GBps=\4 wall_batched=\5/'
done
echo "=== power: idle 4 s, then q4_global 17408x5120 x 3000 dispatches (~5 s)"
adb -s $S shell "for i in \$(seq 1 20); do echo idle \$(cat /sys/class/power_supply/battery/current_now) \$(cat /sys/class/power_supply/battery/voltage_now) \$(cat /sys/class/devfreq/34f00000.gpu0/cur_freq 2>/dev/null); sleep 0.2; done"
adb -s $S shell "(for i in \$(seq 1 40); do echo busy \$(cat /sys/class/power_supply/battery/current_now) \$(cat /sys/class/power_supply/battery/voltage_now); sleep 0.2; done) & cd $D && ./clgemv clgemv.cl q4_global 17408 5120 3000 8 128 | tail -1 | cut -c1-200; wait"
echo "=== power: f16_rows x 800 dispatches for comparison"
adb -s $S shell "(for i in \$(seq 1 40); do echo busy_f16 \$(cat /sys/class/power_supply/battery/current_now) \$(cat /sys/class/power_supply/battery/voltage_now); sleep 0.2; done) & cd $D && ./clgemv clgemv.cl f16_rows 17408 5120 800 8 128 | tail -1 | cut -c1-120; wait"
