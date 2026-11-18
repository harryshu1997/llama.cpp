#!/bin/bash
S=5A040DLCH004ES; D=/data/local/tmp/s43-pixel-cl; DF=/sys/class/devfreq
adb -s $S push /tmp/pxcl/clgemv /tmp/pxcl/clgemv.cl /tmp/pxcl/cpustream $D/ >/dev/null; adb -s $S shell "chmod 755 $D/cpustream"
run() { adb -s $S shell "cd $D && taskset 80 ./clgemv clgemv.cl $1 17408 5120 40 ${2:-32} ${3:-256}" 2>&1 | tail -2 | sed 's/.*"variant":"\([a-z0-9_]*\)".*"device_ms_med":\([0-9.]*\).*"GBps_med":\([0-9.]*\).*"wall_ms_batched":\([0-9.]*\).*"rel_l2_vs_quantized_ref":\([0-9.e+-]*\).*/\1 device_ms=\2 GBps=\3 wall_batched=\4 rel_l2=\5/'; }
echo "=== kernel v2 (mixed dot + xsum, in-stream aligned scales, pipelined loads)"
run q4_v2 8 128; run q4_v2 16 128; run q4_v2 32 256; run q4_v2 16 256; run q4_global 32 256
echo "=== CPU stream alone (1 / 3 / 6 threads, cores 7..2)"
for t in 1 3 6; do adb -s $S shell "cd $D && ./cpustream 512 $t 3"; done
echo "=== concurrent: GPU q4_global loop (taskset 80 -> core 7) + CPU stream on cores 6..(T) for 4 s, governor clocks"
for t in 1 3 5; do adb -s $S shell "cd $D && (taskset 80 ./clgemv clgemv.cl q4_global 17408 5120 2500 32 256 | tail -1 | sed 's/.*\"GBps_med\":\([0-9.]*\).*/gpu_GBps=\1/') & sleep 0.5; cd $D && taskset 7e ./cpustream 512 $t 3.5; wait"; done
echo "=== same with memory/fabric/GPU clocks pinned to max"
adb -s $S shell "su -c 'for n in gmc_freq memss_freq fabhbw_freq; do m=\$(cat $DF/\$n/available_frequencies | tr \" \" \"\\n\" | sort -n | tail -1); echo \$m > $DF/\$n/min_freq; done; echo 1094000000 > $DF/34f00000.gpu0/min_freq; echo pinned'"
for t in 1 3 5; do adb -s $S shell "cd $D && (taskset 80 ./clgemv clgemv.cl q4_global 17408 5120 2500 32 256 | tail -1 | sed 's/.*\"GBps_med\":\([0-9.]*\).*/gpu_GBps=\1/') & sleep 0.5; cd $D && taskset 7e ./cpustream 512 $t 3.5; wait"; done
adb -s $S shell "cd $D && ./cpustream 512 6 3"
adb -s $S shell "su -c 'for n in gmc_freq memss_freq fabhbw_freq; do m=\$(cat $DF/\$n/available_frequencies | tr \" \" \"\\n\" | sort -n | head -1); echo \$m > $DF/\$n/min_freq; done; echo 396000000 > $DF/34f00000.gpu0/min_freq; echo restored'"
