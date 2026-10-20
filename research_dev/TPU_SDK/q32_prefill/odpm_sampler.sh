#!/system/bin/sh
# Root ODPM sampler: append the full energy_value dump every ~0.1 s until STOPFILE exists (max N samples).
# usage (as root): odpm_sampler.sh OUTFILE STOPFILE MAX_SAMPLES
OUT=$1; STOP=$2; MAX=${3:-3000}
i=0
while [ ! -f "$STOP" ] && [ $i -lt $MAX ]; do
  cat /sys/bus/iio/devices/iio:device0/energy_value >> "$OUT"
  i=$((i + 1))
  sleep 0.1
done
echo "sampler_end samples=$i" >> "$OUT"
