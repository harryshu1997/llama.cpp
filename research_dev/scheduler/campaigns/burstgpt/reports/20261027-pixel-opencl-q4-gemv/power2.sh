#!/bin/bash
# Pixel power during sustained GPU GEMV: total = USB input + battery discharge (both sysfs), sampled at 5 Hz.
S=5A040DLCH004ES; D=/data/local/tmp/s43-pixel-cl
sample() { adb -s $S shell "for i in \$(seq 1 $1); do echo $2 \$(cat /sys/class/power_supply/usb/current_now) \$(cat /sys/class/power_supply/usb/voltage_now) \$(cat /sys/class/power_supply/battery/current_now) \$(cat /sys/class/power_supply/battery/voltage_now); sleep 0.2; done"; }
sample 15 idle
adb -s $S shell "(for i in \$(seq 1 45); do echo q4 \$(cat /sys/class/power_supply/usb/current_now) \$(cat /sys/class/power_supply/usb/voltage_now) \$(cat /sys/class/power_supply/battery/current_now) \$(cat /sys/class/power_supply/battery/voltage_now); sleep 0.2; done) & cd $D && ./clgemv clgemv.cl q4_global 17408 5120 4500 32 256 >/dev/null; wait"
sleep 3
adb -s $S shell "(for i in \$(seq 1 45); do echo f16 \$(cat /sys/class/power_supply/usb/current_now) \$(cat /sys/class/power_supply/usb/voltage_now) \$(cat /sys/class/power_supply/battery/current_now) \$(cat /sys/class/power_supply/battery/voltage_now); sleep 0.2; done) & cd $D && ./clgemv clgemv.cl f16_rows 17408 5120 1000 8 128 >/dev/null; wait"
