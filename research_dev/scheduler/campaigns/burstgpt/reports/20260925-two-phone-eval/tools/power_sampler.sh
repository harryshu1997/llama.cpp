# Phone power diagnostic, about 1 Hz, read-only sysfs: sh power_sampler.sh OUT STOPFILE MAX_SECONDS
# Each line: uptime_s node=value ...; exits when STOPFILE exists or after MAX_SECONDS samples.
out=$1; stop=$2; max=$3
nodes="battery/current_now battery/voltage_now battery/charge_counter battery/power_now usb/current_now usb/voltage_now usb/input_current_limit main-charger/current_now gcpm/current_now max77779fg/current_now"
n=0
read up rest < /proc/uptime
echo "# start uptime $up pid $$" >> "$out"
while [ ! -e "$stop" ] && [ "$n" -lt "$max" ]; do
  read up rest < /proc/uptime
  line="$up"
  for node in $nodes; do
    f=/sys/class/power_supply/$node
    if [ -r "$f" ]; then read v < "$f"; line="$line $node=$v"; fi
  done
  echo "$line" >> "$out"
  n=$((n+1))
  sleep 1
done
read up rest < /proc/uptime
echo "# stop uptime $up samples $n" >> "$out"
