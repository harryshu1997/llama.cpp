#!/bin/bash
S=5A040DLCH004ES; D=/data/local/tmp/s43-pixel-cl; DF=/sys/class/devfreq; G=$DF/34f00000.gpu0; M=$DF/memss_freq
adb -s $S push /tmp/pxcl/cllat $D/ >/dev/null
cad() { adb -s $S shell "cd $D && taskset 80 ./cllat clgemv.cl 32 256 0 2.5 150 16 ${1:-0}" | sed 's/.*"layer_wall_ms":{"med":\([0-9.]*\),"p90":\([0-9.]*\).*"first_layer_of_token_ms":{"med":\([0-9.]*\).*/layer med=\1 p90=\2 first-of-token med=\3/'; }
pin() { adb -s $S shell "su -c 'echo $1 > $G/min_freq; echo $2 > $M/min_freq'"; }
echo "=== D. which pin matters (2.5 ms gaps, 150 ms token gap)"
pin 396000000 178000000; echo "none:          $(cad)"
pin 1094000000 178000000; echo "GPU only:      $(cad)"
pin 396000000 932000000; echo "memss only:    $(cad)"
pin 1094000000 932000000; echo "both:          $(cad)"
echo "=== A. keep-alive during the token gap (both pinned): tiny dispatch every 20 / 50 ms"
echo "ka 20 ms:      $(cad 20)"; echo "ka 50 ms:      $(cad 50)"
pin 396000000 178000000
echo "=== C. idle power: governor vs pinned clocks, no work (USB + battery, 4 s each)"
samp() { adb -s $S shell "for i in \$(seq 1 20); do echo \$(cat /sys/class/power_supply/usb/current_now) \$(cat /sys/class/power_supply/usb/voltage_now) \$(cat /sys/class/power_supply/battery/current_now) \$(cat /sys/class/power_supply/battery/voltage_now); sleep 0.2; done" | awk '{u+=$1*$2/1e12; b+=-$3*$4/1e12; n++} END {printf "usb %.2f W + battery %.2f W = %.2f W\n", u/n, b/n, (u+b)/n}'; }
echo "governor idle: $(samp)"; pin 1094000000 932000000; sleep 1; echo "pinned idle:   $(samp)"; pin 396000000 178000000
echo "=== B. PowerVR driver parameters (power management related)"
adb -s $S shell "su -c 'cd /sys/module/pvrsrvkm/parameters 2>/dev/null && for f in *; do v=\$(cat \$f 2>/dev/null | head -c 40); p=\$(stat -c %a \$f); echo \"\$f=\$v (\$p)\"; done'" 2>&1 | grep -i "power\|idle\|apm\|timeout\|gate\|dvfs\|island\|freq" | head -16
echo "restored: gpu min=$(adb -s $S shell cat $G/min_freq) memss min=$(adb -s $S shell cat $M/min_freq)"
