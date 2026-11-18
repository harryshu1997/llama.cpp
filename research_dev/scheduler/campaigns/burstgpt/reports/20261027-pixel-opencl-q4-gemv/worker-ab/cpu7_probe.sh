#!/system/bin/sh
# As root: core-7 clock during a worker replay, then the same replay with core 7 pinned to its max clock.
P=/sys/devices/system/cpu/cpufreq/policy7; T=/data/local/tmp/s43-pixel-cl-ab
echo "cpu7 avail: $(cat $P/scaling_available_frequencies | tr ' ' '\n' | sort -n | tail -1) max, cur=$(cat $P/scaling_cur_freq) min=$(cat $P/scaling_min_freq) gov=$(cat $P/scaling_governor)"
(sh /data/local/tmp/pixel_cl_quick.sh > /dev/null 2>&1) &
sleep 0.4; i=0; while [ $i -lt 40 ]; do cat $P/scaling_cur_freq; usleep 20000; i=$((i+1)); done | sort -n | uniq -c | sed 's/^/ cpu7 during replay: /'
wait
grep -h S43PIXELCLPROF $T/cl.log | tail -3 | head -1 | cut -c1-120
echo "--- core 7 pinned to max"
OLD=$(cat $P/scaling_min_freq); MAX=$(cat $P/scaling_available_frequencies | tr ' ' '\n' | sort -n | tail -1); echo $MAX > $P/scaling_min_freq; echo "min now $(cat $P/scaling_min_freq)"
sh /data/local/tmp/pixel_cl_quick.sh
grep -h S43PIXELCLPROF $T/cl.log | tail -3 | cut -c1-120
echo $OLD > $P/scaling_min_freq; echo "restored min $(cat $P/scaling_min_freq)"
