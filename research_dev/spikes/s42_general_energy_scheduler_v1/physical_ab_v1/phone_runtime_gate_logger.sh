#!/system/bin/sh
set -eu

if [ "$#" -ne 5 ]; then
    echo "usage: $0 <active> <samples> <armed> <done> <wait-seconds>" >&2
    exit 2
fi

active=$1
samples=$2
armed=$3
done=$4
wait_seconds=$5

case "$wait_seconds" in
    ''|*[!0-9]*)
        echo "invalid wait-seconds: $wait_seconds" >&2
        exit 2
        ;;
esac

max_temperature() {
    gate_max=-1
    for gate_path in "$@"; do
        gate_value=$(cat "$gate_path/temp" 2>/dev/null || true)
        case "$gate_value" in
            ''|*[!0-9-]*) continue ;;
        esac
        if [ "$gate_value" -gt "$gate_max" ]; then
            gate_max=$gate_value
        fi
    done
    printf '%s' "$gate_max"
}

rm -f "$samples" "$armed" "$done"
: > "$armed"

read -r started_uptime _ < /proc/uptime
started_uptime=${started_uptime%%.*}
deadline=$((started_uptime + wait_seconds))
while [ ! -f "$active" ]; do
    read -r current_uptime _ < /proc/uptime
    current_uptime=${current_uptime%%.*}
    if [ "$current_uptime" -ge "$deadline" ]; then
        echo "active marker timeout" >&2
        exit 1
    fi
    sleep 0.1
done

printf 'epoch_s\tuptime_s\tthermal_status\tbattery_millic\tshell_millic\tcpu_millic\tnpu_millic\tgpu_millic\tddr_millic\n' \
    > "$samples"

while [ -f "$active" ]; do
    epoch_s=$(date +%s)
    read -r uptime_s _ < /proc/uptime
    thermal_status=$(dumpsys thermalservice 2>/dev/null | \
        awk '/^Thermal Status:/ { print $3; exit }')
    case "$thermal_status" in
        ''|*[!0-9]*) thermal_status=-1 ;;
    esac
    battery=$(max_temperature /sys/class/thermal/thermal_zone93)
    shell=$(max_temperature \
        /sys/class/thermal/thermal_zone55 \
        /sys/class/thermal/thermal_zone61 \
        /sys/class/thermal/thermal_zone65)
    cpu=$(max_temperature /sys/class/thermal/thermal_zone[0-9] \
        /sys/class/thermal/thermal_zone1[0-8] \
        /sys/class/thermal/thermal_zone2[4-7])
    npu=$(max_temperature /sys/class/thermal/thermal_zone2[89] \
        /sys/class/thermal/thermal_zone3[0-5])
    gpu=$(max_temperature /sys/class/thermal/thermal_zone3[6-9] \
        /sys/class/thermal/thermal_zone4[0-6])
    ddr=$(max_temperature /sys/class/thermal/thermal_zone47)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$epoch_s" "$uptime_s" "$thermal_status" "$battery" "$shell" \
        "$cpu" "$npu" "$gpu" "$ddr" >> "$samples"
    sleep 2
done

: > "$done"
