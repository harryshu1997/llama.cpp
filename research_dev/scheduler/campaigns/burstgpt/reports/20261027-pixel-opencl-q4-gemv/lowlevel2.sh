#!/bin/bash
S=5A040DLCH004ES; D=/data/local/tmp/s43-pixel-cl; DF=/sys/class/devfreq
adb -s $S push /tmp/pxcl/chase $D/ >/dev/null; adb -s $S shell "chmod 755 $D/chase"
echo "=== A. debugfs: real clocks?"
adb -s $S shell "su -c 'mount -t debugfs none /sys/kernel/debug 2>/dev/null; ls /sys/kernel/debug | tr \"\n\" \" \" | cut -c1-600; echo; ls /sys/kernel/debug/clk 2>/dev/null | grep -i \"mif\|dram\|memss\|gmc\|ddr\|dmc\|mem\" | head; grep -i \"mif\|dram\|memss\|gmc\|ddr\|dmc\" /sys/kernel/debug/clk/clk_summary 2>/dev/null | head -12; find /sys/kernel/debug -maxdepth 2 \\( -iname \"*mif*\" -o -iname \"*dram*\" -o -iname \"*dvfs*\" -o -iname \"*acpm*\" -o -iname \"*bts*\" -o -iname \"*gmc*\" -o -iname \"*memss*\" \\) 2>/dev/null | head -12'" | cut -c1-600
echo "=== B. platform devices / drivers"
adb -s $S shell "su -c 'ls /sys/devices/platform | grep -i \"gmc\|memss\|dmc\|ddr\|mif\|dvfs\|acpm\|bts\|lpddr\|dram\" | head; ls /sys/bus/platform/drivers | grep -i \"gmc\|memss\|dmc\|ddr\|mif\|dvfs\|acpm\|bts\|dram\" | head; for d in \$(ls -d /sys/devices/platform/*gmc* /sys/devices/platform/*memss* /sys/devices/platform/*dram* 2>/dev/null | head -3); do echo \$d: \$(ls \$d | tr \"\n\" \" \" | cut -c1-300); done'" | cut -c1-400
echo "=== C. dmesg + device tree"
adb -s $S shell "su -c 'dmesg 2>&1 | grep -i \"lpddr\|dram\|ddr\|gmc\|memss\|MT/s\" | head -12; ls /sys/firmware/devicetree/base | grep -i \"mem\|ddr\|gmc\|dram\|dvfs\|acpm\" | head -10'" | cut -c1-200
echo "=== D. GPU stream alone, then with a pointer-chase on core 6 (raises the CPU memory-latency vote, ~0.3 GB/s)"
adb -s $S shell "cd $D && taskset 80 ./clgemv clgemv.cl stream_read16 17408 5120 40 64 512 | tail -1" | sed 's/.*"device_ms_med":\([0-9.]*\).*"GBps_med":\([0-9.]*\).*/alone: device_ms=\1 GBps=\2/'
adb -s $S shell "cd $D && (./chase 64 6 6) & sleep 1.5; cd $D && taskset 80 ./clgemv clgemv.cl stream_read16 17408 5120 40 64 512 | tail -1; wait" | sed 's/.*"device_ms_med":\([0-9.]*\).*"GBps_med":\([0-9.]*\).*/with chase: device_ms=\1 GBps=\2/'
echo "--- CPU votes during the chase (atrace freq, 3 s)"
adb -s $S shell "su -c 'atrace -t 3 -b 4096 freq -o /data/local/tmp/at2.txt >/dev/null 2>&1 & sleep 0.3; cd $D && ./chase 64 2.5 6 >/dev/null; wait; grep -o \"clock_set_rate: cpu6gmc state=[0-9]*\" /data/local/tmp/at2.txt | sort | uniq -c | sort -rn | head -5; grep -o \"clock_set_rate: dsulat state=[0-9]*\" /data/local/tmp/at2.txt | sort | uniq -c | sort -rn | head -3'"
echo "=== E. GPU stream with a CPU stream thread that consumes bandwidth (reference, 1 thread core 6)"
adb -s $S shell "cd $D && (taskset 40 ./cpustream 256 1 4) & sleep 0.8; cd $D && taskset 80 ./clgemv clgemv.cl stream_read16 17408 5120 40 64 512 | tail -1; wait" | sed 's/.*"device_ms_med":\([0-9.]*\).*"GBps_med":\([0-9.]*\).*/gpu with cpu stream: device_ms=\1 GBps=\2/; s/.*"cpu_stream_GBps":\([0-9.]*\).*/cpu stream: \1 GBps/'
