#!/system/bin/sh
# As root: per-layer OpenCL time (cllaunch G/F lines) vs core-7 minimum clock.
P=/sys/devices/system/cpu/cpufreq/policy7; N=/data/local/tmp/s43-pixel-opencl-20261027-v1
OLD=$(cat $P/scaling_min_freq); echo "avail: $(cat $P/scaling_available_frequencies)"
pkill -f llama-ffn-split-[w]orker; sleep 1
for f in $OLD 975000 1491000 2010000 2548000 3168000 3782000; do
  echo $f > $P/scaling_min_freq 2>/dev/null || { echo "skip $f"; continue; }
  echo "min_freq=$(cat $P/scaling_min_freq): $(cd $N && ./cllaunch $N/pixel_opencl_kernels.cl 60 2>&1 | grep '^[GK]' | tr -s ' ' | tr '\n' '|')"
done
echo $OLD > $P/scaling_min_freq; echo "restored $(cat $P/scaling_min_freq)"
