#!/bin/bash
S=5A440DLCH004ES; S=5A040DLCH004ES; D=/data/local/tmp/s43-pixel-cl; G=/sys/class/devfreq/34f00000.gpu0
echo "=== driver params / governor"
adb -s $S shell "su -c 'cat $G/governor; cat $G/available_governors; cat $G/polling_interval 2>/dev/null; ls /sys/module | grep -i \"pvr\|powervr\|img\" ; ls /sys/module/pvrsrvkm/parameters 2>/dev/null | tr \"\n\" \" \"'" 2>&1 | head -8
echo "=== A: baseline unpinned vs pinned to core 7"
adb -s $S shell "cd $D && ./cllat clgemv.cl 32 256 0" | cut -c1-260
adb -s $S shell "cd $D && taskset 80 ./cllat clgemv.cl 32 256 0" | cut -c1-260
echo "=== D: GPU clock at 50 Hz while a per-layer-finish loop runs (unpinned clocks)"
adb -s $S shell "(for i in \$(seq 1 60); do cat $G/cur_freq; sleep 0.05; done | sort -n | uniq -c | tr '\n' ';') & cd $D && ./cllat clgemv.cl 32 256 0 >/dev/null; wait; echo"
echo "=== B: GPU min_freq pinned to max (root), pinned to core 7"
adb -s $S shell "su -c 'echo 1094000000 > $G/min_freq; cat $G/min_freq'"
adb -s $S shell "cd $D && taskset 80 ./cllat clgemv.cl 32 256 0" | cut -c1-260
adb -s $S shell "cd $D && ./cllat clgemv.cl 32 256 0" | cut -c1-260
adb -s $S shell "su -c 'echo 396000000 > $G/min_freq; cat $G/min_freq'"
