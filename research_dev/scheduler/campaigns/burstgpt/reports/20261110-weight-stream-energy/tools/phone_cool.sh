# one line of cool-down state (read-only): uptime zone=temp ... bat_dc cooling_max cpu_uncapped worker status
COOL_ZONES="nsphvx-0 ddr cpu-1-0-0 shell_back"
COOL_CDEVS="cdsp cdsp_hw cdsp_sw_hvx cdsp_sw_hmx ddr-cdev cpufreq-cpu0 cpufreq-cpu6"
conf=$(dirname "$0")/device.conf
[ -f "$conf" ] && . "$conf"
out=""
for z in /sys/class/thermal/thermal_zone*; do
  read t < $z/type
  for want in $COOL_ZONES; do [ "$t" = "$want" ] && { read v < $z/temp; out="$out $t=$v"; }; done
done
mx=0
for c in /sys/class/thermal/cooling_device*; do
  read t < $c/type
  for want in $COOL_CDEVS; do [ "$t" = "$want" ] && { read v < $c/cur_state; [ "$v" -gt "$mx" ] && mx=$v; }; done
done
bt=-1; [ -r /sys/class/power_supply/battery/temp ] && read bt < /sys/class/power_supply/battery/temp
read up rest < /proc/uptime
w=0; pgrep -f "llama-ffn-split-[w]orker" > /dev/null && w=1
cap=1
for pol in /sys/devices/system/cpu/cpufreq/policy*; do
  read a < $pol/scaling_max_freq; read b < $pol/cpuinfo_max_freq
  [ "$a" -lt "$b" ] && cap=0
done
st=$(dumpsys thermalservice 2> /dev/null | grep -m1 "Thermal Status" | tr -dc 0-9)
echo "$up$out bat_dc=$bt cooling_max=$mx cpu_uncapped=$cap worker=$w status=$st"
