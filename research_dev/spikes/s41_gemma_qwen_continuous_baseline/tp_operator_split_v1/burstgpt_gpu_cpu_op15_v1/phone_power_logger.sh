#!/system/bin/sh
set -eu

if [ "$#" -ne 5 ]; then
    echo "usage: $0 <samples> <active> <armed> <done> <start-timeout-seconds>" >&2
    exit 2
fi

samples=$1
active=$2
armed=$3
done=$4
start_timeout=$5

usb_current=/sys/class/power_supply/usb/current_now
usb_voltage=/sys/class/power_supply/usb/voltage_now
battery_current=/sys/class/power_supply/battery/current_now
battery_voltage=/sys/class/power_supply/battery/voltage_now
battery_charge_counter=/sys/class/power_supply/battery/charge_counter

rm -f "$armed" "$done"
for path in "$usb_current" "$usb_voltage" "$battery_current" "$battery_voltage" "$battery_charge_counter"; do
    value=$(cat "$path")
    case "$value" in
        -[0-9]*|[0-9]*) ;;
        *) echo "invalid sensor: $path" >&2; exit 1 ;;
    esac
done

printf 'uptime_s\tusb_current_ua\tusb_voltage_uv\tbattery_current_ma\tbattery_voltage_uv\tbattery_charge_counter_uah\n' > "$samples"
: > "$armed"
chmod 0644 "$samples" "$armed"

attempt=0
while [ ! -f "$active" ]; do
    if [ "$attempt" -ge "$((start_timeout * 10))" ]; then
        echo "active marker timeout" >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    sleep 0.1
done

while [ -f "$active" ]; do
    uptime=$(cut -d ' ' -f 1 /proc/uptime)
    ui=$(cat "$usb_current")
    uv=$(cat "$usb_voltage")
    bi=$(cat "$battery_current")
    bv=$(cat "$battery_voltage")
    bc=$(cat "$battery_charge_counter")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$uptime" "$ui" "$uv" "$bi" "$bv" "$bc" >> "$samples"
    sleep 0.2
done

: > "$done"
chmod 0644 "$samples" "$done"
