# Phone-side power sampler (read-only sysfs), pushed and started by meter_phones.sh:
#   sh phone_power_sampler.sh OUT STOPFILE PERIOD_S MAX_SAMPLES NODE...
# NODE is relative to /sys/class/power_supply (e.g. usb/current_now) or an absolute path. Each line:
#   uptime_s node=value ...      (same format as reports/20260925-two-phone-eval/tools/power_sampler.sh)
# Exits when STOPFILE exists or after MAX_SAMPLES samples; the last line is "# stop uptime U samples N".
out=$1; stop=$2; period=$3; max=$4; shift 4
n=0
read up rest < /proc/uptime
echo "# start uptime $up pid $$ period_s $period nodes $*" >> "$out"
while [ ! -e "$stop" ] && [ "$n" -lt "$max" ]; do
  read up rest < /proc/uptime
  line="$up"
  for node in "$@"; do
    case "$node" in /*) f=$node ;; *) f=/sys/class/power_supply/$node ;; esac
    if [ -r "$f" ]; then read v < "$f"; line="$line $node=$v"; fi
  done
  echo "$line" >> "$out"
  n=$((n+1))
  sleep "$period"
done
read up rest < /proc/uptime
echo "# stop uptime $up samples $n" >> "$out"
