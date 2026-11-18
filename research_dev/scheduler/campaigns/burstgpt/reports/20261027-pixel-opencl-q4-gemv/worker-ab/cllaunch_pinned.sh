#!/system/bin/sh
# As root: run cllaunch with core 7 pinned to max clock, then restore.
P=/sys/devices/system/cpu/cpufreq/policy7; N=/data/local/tmp/s43-pixel-opencl-20261027-v1
OLD=$(cat $P/scaling_min_freq); echo $(cat $P/scaling_available_frequencies | tr ' ' '\n' | sort -n | tail -1) > $P/scaling_min_freq; echo "cpu7 min now $(cat $P/scaling_min_freq)"
pkill -f llama-ffn-split-[w]orker; sleep 1
cd $N && ./cllaunch ${1:-$N/pixel_opencl_kernels.cl} 100
echo $OLD > $P/scaling_min_freq; echo "restored $(cat $P/scaling_min_freq)"
