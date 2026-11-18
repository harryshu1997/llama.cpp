# Phone-side sampler (read-only sysfs). sh phone_sampler.sh OUT STOPFILE PERIOD_S
# Line: uptime_s key=value ... ; thermal zones and cooling devices are resolved by type at start.
out=$1; stop=$2; period=$3
P=/sys/class/power_supply
ZONES="nsphvx-0 nsphmx-0 ddr cpu-0-0-0 cpu-1-0-0 gpuss-0 shell_back battery"
CDEVS="cdsp cdsp_hw cdsp_sw_hvx cdsp_sw_hmx ddr-cdev cpufreq-cpu0 cpufreq-cpu6 cpu-cluster0 cpu-cluster1"
conf=$(dirname "$0")/device.conf
[ -f "$conf" ] && . "$conf"   # another phone sets ZONES / CDEVS (thermal zone and cooling device types)
zpaths=""; cpaths=""
for z in /sys/class/thermal/thermal_zone*; do
  read t < $z/type
  for want in $ZONES; do [ "$t" = "$want" ] && zpaths="$zpaths $want=$z/temp"; done
done
for c in /sys/class/thermal/cooling_device*; do
  read t < $c/type
  for want in $CDEVS; do [ "$t" = "$want" ] && cpaths="$cpaths $want=$c/cur_state"; done
done
fpaths=""
for pol in /sys/devices/system/cpu/cpufreq/policy*; do
  n=${pol##*policy}
  fpaths="$fpaths f$n=$pol/scaling_cur_freq m$n=$pol/scaling_max_freq"
done
# memory-bus DCVS (Qualcomm): DDR and LLCC frequency in kHz
for bus in DDR LLCC; do
  [ -r /sys/devices/system/cpu/bus_dcvs/$bus/cur_freq ] && fpaths="$fpaths bus_$bus=/sys/devices/system/cpu/bus_dcvs/$bus/cur_freq"
done
read up rest < /proc/uptime
echo "# start uptime $up period_s $period" >> "$out"
while [ ! -e "$stop" ]; do
  read up rest < /proc/uptime
  if [ -r $P/usb/current_now ]; then
    read ui < $P/usb/current_now; read uv < $P/usb/voltage_now
    read bi < $P/battery/current_now; read bv < $P/battery/voltage_now; read cc < $P/battery/charge_counter; read bt < $P/battery/temp
    line="$up usb_ua=$ui usb_uv=$uv bat_ma=$bi bat_uv=$bv cc_uah=$cc bat_dc=$bt"
  else
    line="$up nopower=1"   # no root: the power nodes are unreadable, temperatures and frequencies only
  fi
  for kv in $zpaths; do read v < ${kv#*=}; line="$line ${kv%%=*}=$v"; done
  for kv in $cpaths; do read v < ${kv#*=}; line="$line cd_${kv%%=*}=$v"; done
  for kv in $fpaths; do read v < ${kv#*=}; line="$line ${kv%%=*}=$v"; done
  echo "$line" >> "$out"
  sleep "$period"
done
read up rest < /proc/uptime
echo "# stop uptime $up" >> "$out"
