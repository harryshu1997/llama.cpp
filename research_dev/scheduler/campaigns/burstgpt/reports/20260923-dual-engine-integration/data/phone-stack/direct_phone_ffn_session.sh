#!/system/bin/sh
set -eu

if [ "$#" -ne 10 ] && [ "$#" -ne 13 ]; then
    echo "usage: $0 <worker> <model> <layers> <columns> <backend> <session-root> <restore-script> <timeout-seconds> <max-requests> <artifact-sha256> [<resident-workers> <resident-router> <shard-manifest-rows>]" >&2
    exit 2
fi

worker=$1
model=$2
layers=$3
columns=$4
backend=$5
session_root=$6
restore_script=$7
timeout_seconds=$8
max_requests=$9
artifact_sha256=${10}
multi_session=0
resident_workers=
resident_router=
shard_manifest_rows=
if [ "$#" -eq 13 ]; then
    multi_session=1
    resident_workers=${11}
    resident_router=${12}
    shard_manifest_rows=${13}
fi

g1=${SCHEDULER_ANDROID_GADGET:-}
g2=${SCHEDULER_FUNCTIONFS_GADGET:-}
udc=${SCHEDULER_PHONE_UDC:-}
ffs_root=${SCHEDULER_FUNCTIONFS_ROOT:-}
ready_file=$session_root/descriptors.ready
worker_log=$session_root/worker.log
router_log=$session_root/router.log
terminal_status=$session_root/terminal.status
worker_root=$(CDPATH= cd -- "$(dirname -- "$worker")" && pwd)
ncm_enabled=${S42_USB_NCM:-0}
diagnostic_port=${S42_DIAGNOSTIC_PORT:-}
busybox_binary=${S42_BUSYBOX:-}

case "$g1:$g2:$ffs_root:$udc" in
    /*:/*:/*:[A-Za-z0-9._-]*) ;;
    *) echo "phone USB gadget identity is invalid" >&2; exit 2 ;;
esac

case "$ncm_enabled" in
    0) ;;
    1)
        case "$diagnostic_port" in
            *[!0-9]*|'') echo "invalid S42_DIAGNOSTIC_PORT" >&2; exit 2 ;;
        esac
        if [ -z "$busybox_binary" ] || [ ! -x "$busybox_binary" ]; then
            echo "diagnostic busybox is unavailable" >&2
            exit 2
        fi
        ;;
    *) echo "invalid S42_USB_NCM" >&2; exit 2 ;;
esac

mkdir -p "$session_root"
rm -f "$ready_file" "$terminal_status" "$terminal_status.tmp"
normal_usb_config=$(getprop sys.usb.config)
case "$normal_usb_config" in
    *[!A-Za-z0-9,._-]*|'')
        echo "normal Android USB configuration is invalid" >&2
        exit 2
        ;;
esac
printf '%s\n' "$g1" > "$session_root/android-gadget.path"
printf '%s\n' "$g2" > "$session_root/functionfs-gadget.path"
printf '%s\n' "$ffs_root" > "$session_root/functionfs-root.path"
printf '%s\n' "$udc" > "$session_root/phone-udc.name"
printf '%s\n' "$normal_usb_config" > "$session_root/android-usb-config"

worker_pid=
resident_workers_pid=
router_pid=
watchdog_pid=
diagnostic_http_pid=
diagnostic_power_pid=
diagnostic_snapshot_pid=
ncm_dhcp_pid=
signal_process() {
    process_pid=$1
    if [ -n "$process_pid" ]; then
        kill -TERM "$process_pid" 2>/dev/null || true
    fi
}
reap_process() {
    process_pid=$1
    if [ -z "$process_pid" ]; then
        return
    fi
    attempt=0
    while kill -0 "$process_pid" 2>/dev/null && [ "$attempt" -lt 50 ]; do
        attempt=$((attempt + 1))
        sleep 0.1
    done
    if kill -0 "$process_pid" 2>/dev/null; then
        kill -KILL "$process_pid" 2>/dev/null || true
    fi
    wait "$process_pid" 2>/dev/null || true
}
cleanup() {
    cleanup_status=$?
    printf '%s\n' "$cleanup_status" > "$terminal_status.tmp"
    mv "$terminal_status.tmp" "$terminal_status"
    rm -f "$session_root/active"
    if [ -n "$watchdog_pid" ]; then
        kill "$watchdog_pid" 2>/dev/null || true
    fi
    signal_process "$worker_pid"
    if [ -n "$router_pid" ] && [ "$router_pid" != "$worker_pid" ]; then
        signal_process "$router_pid"
    fi
    signal_process "$resident_workers_pid"
    for pid in "$diagnostic_http_pid" "$diagnostic_power_pid" \
            "$diagnostic_snapshot_pid" "$ncm_dhcp_pid"; do
        signal_process "$pid"
    done
    sh "$restore_script" "$session_root" || true
    reap_process "$worker_pid"
    if [ -n "$router_pid" ] && [ "$router_pid" != "$worker_pid" ]; then
        reap_process "$router_pid"
    fi
    reap_process "$resident_workers_pid"
    for pid in "$diagnostic_http_pid" "$diagnostic_power_pid" \
            "$diagnostic_snapshot_pid" "$ncm_dhcp_pid"; do
        reap_process "$pid"
    done
}
trap cleanup EXIT INT TERM
trap '' HUP

sh "$restore_script" "$session_root" || true
: > "$session_root/active"

probe_android_thermal_state() {
    dumpsys -t 1 thermalservice 2>/dev/null | awk '
        BEGIN { maximum = 0; status = -1 }
        /^Thermal Status: [0-9]+$/ { status = $3 }
        /^Current temperatures from HAL:/ { current = 1; next }
        /^Current cooling devices from HAL:/ { current = 0 }
        current && /Temperature\{/ {
            split($0, fields, /mValue=|, mType=|, mName=|, mStatus=|}/)
            if (fields[2] !~ /^[-+]?[0-9]+([.][0-9]+)?$/ ||
                fields[3] !~ /^(0|1|2|3|4|5|9)$/ ||
                fields[5] !~ /^[0-9]+$/ || fields[2] <= 0) next
            if (fields[2] * 1000 > maximum) maximum = fields[2] * 1000
            if (fields[5] > sensor_status) sensor_status = fields[5]
        }
        END {
            if (status >= 0 && sensor_status > status) status = sensor_status
            printf "%.0f %d\n", maximum, status
        }
    '
}

if [ "$ncm_enabled" = 1 ]; then
    diagnostic_root=$session_root/diagnostics
    mkdir -p "$diagnostic_root"
    : > "$diagnostic_root/power.jsonl"
    (
        while [ -f "$session_root/active" ]; do
            captured_epoch_s=$(date +%s)
            uptime_s=$(cut -d ' ' -f 1 /proc/uptime)
            usb_current_ua=$(cat /sys/class/power_supply/usb/current_now)
            usb_voltage_uv=$(cat /sys/class/power_supply/usb/voltage_now)
            battery_current_ma=$(cat \
                /sys/class/power_supply/battery/current_now)
            battery_voltage_uv=$(cat \
                /sys/class/power_supply/battery/voltage_now)
            battery_charge_counter_uah=$(cat \
                /sys/class/power_supply/battery/charge_counter)
            temporary=$diagnostic_root/power.json.tmp
            power_row="{\"battery_charge_counter_uah\":$battery_charge_counter_uah,\"battery_current_ma\":$battery_current_ma,\"battery_voltage_uv\":$battery_voltage_uv,\"captured_epoch_s\":$captured_epoch_s,\"schema\":\"s42-op15-live-power-v1\",\"uptime_s\":$uptime_s,\"usb_current_ua\":$usb_current_ua,\"usb_voltage_uv\":$usb_voltage_uv}"
            printf '%s\n' "$power_row" > "$temporary"
            mv "$temporary" "$diagnostic_root/power.json"
            printf '%s\n' "$power_row" >> "$diagnostic_root/power.jsonl"
            sleep 0.2
        done
    ) &
    diagnostic_power_pid=$!
    (
        sequence=0
        while [ -f "$session_root/active" ]; do
            sequence=$((sequence + 1))
            captured_epoch_s=$(date +%s)
            mem_total_kib=$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo)
            mem_available_kib=$(awk \
                '/^MemAvailable:/ { print $2 }' /proc/meminfo)
            thermal_summary=$(probe_android_thermal_state)
            temperature_max_millic=${thermal_summary% *}
            android_thermal_status=${thermal_summary##* }
            temperature_source=android-hal
            if [ "$temperature_max_millic" -le 0 ]; then
                temperature_source=sysfs
                temperature_max_millic=$(
                for zone in /sys/class/thermal/thermal_zone*; do
                    IFS= read -r zone_type 2>/dev/null < "$zone/type" || continue
                    case "$zone_type" in
                        *-hw-trip-*) continue ;;
                    esac
                    IFS= read -r zone_temperature 2>/dev/null < "$zone/temp" || continue
                    printf '%s\n' "$zone_temperature"
                done | awk '
                    BEGIN { maximum = 0 }
                    { if ($1 > maximum) maximum = $1 }
                    END { print maximum }
                '
                )
            fi
            battery_level_pct=$(cat \
                /sys/class/power_supply/battery/capacity)
            case "$android_thermal_status" in
                ''|*[!0-9]*) android_thermal_status=-1 ;;
            esac
            task_server_alive=false
            if [ -f "$session_root/worker.pid" ]; then
                observed_worker_pid=$(cat "$session_root/worker.pid" 2>/dev/null)
                case "$observed_worker_pid" in
                    *[!0-9]*|'') ;;
                    *)
                        if kill -0 "$observed_worker_pid" 2>/dev/null; then
                            task_server_alive=true
                        fi
                        ;;
                esac
            fi
            temporary=$diagnostic_root/snapshot.json.tmp
            printf '%s\n' \
                "{\"android_thermal_status\":$android_thermal_status,\"battery_level_pct\":$battery_level_pct,\"captured_epoch_s\":$captured_epoch_s,\"mem_available_kib\":$mem_available_kib,\"mem_total_kib\":$mem_total_kib,\"schema\":\"s42-op15-live-snapshot-v1\",\"sequence\":$sequence,\"task_server_alive\":$task_server_alive,\"temperature_max_millic\":$temperature_max_millic,\"temperature_source\":\"$temperature_source\"}" \
                > "$temporary"
            mv "$temporary" "$diagnostic_root/snapshot.json"
            sleep 0.5
        done
    ) &
    diagnostic_snapshot_pid=$!
    "$busybox_binary" httpd -f -p "$diagnostic_port" \
        -h "$diagnostic_root" \
        > "$session_root/diagnostic-http.log" 2>&1 &
    diagnostic_http_pid=$!
fi
(
    sleep "$timeout_seconds"
    if [ -f "$session_root/active" ]; then
        echo "[ffs-session] watchdog restoring USB" >> "$session_root/session.log"
        sh "$restore_script" "$session_root"
    fi
) &
watchdog_pid=$!

mkdir -p "$ffs_root"
mkdir "$g2/functions/ffs.s41"
if [ "$ncm_enabled" = 1 ]; then
    mkdir "$g2/functions/ncm.usb0"
fi
mount -t functionfs s41 "$ffs_root"

export LD_LIBRARY_PATH=$worker_root
export ADSP_LIBRARY_PATH=$worker_root
export GGML_HEXAGON_MBUF=4192
export GGML_HEXAGON_NHVX=${GGML_HEXAGON_NHVX:-4}
export S41_DISABLE_GRAPH_CACHE=1

# S43 dual-engine FFN split (NPU primary + Adreno secondary), zero-copy: the
# qualified batch plan is split-row, so every phone call is tokens=1 and no NPU
# copy of the GPU columns is kept (MAX_TOKENS=0). Overridable per launch.
export S43_FFN_SECONDARY_BACKEND=${S43_FFN_SECONDARY_BACKEND:-GPUOpenCL}
export S43_FFN_SECONDARY_FRACTION=${S43_FFN_SECONDARY_FRACTION:-0.15}
export S43_FFN_SECONDARY_ALIGN=${S43_FFN_SECONDARY_ALIGN:-64}
export S43_FFN_SECONDARY_MAX_TOKENS=${S43_FFN_SECONDARY_MAX_TOKENS:-0}
export S43_FFN_DUAL_LOG_PERIOD=${S43_FFN_DUAL_LOG_PERIOD:-1}
export S43_FFN_DUAL_WARMUP_ROUNDS=${S43_FFN_DUAL_WARMUP_ROUNDS:-3}

io_flag=
case "${S41_FFN_F16_IO:-0}" in
    0) ;;
    1) io_flag=--f16-io ;;
    *) echo "invalid S41_FFN_F16_IO" >&2; exit 2 ;;
esac
staged_flag=
case "${S41_FFN_STAGED_DMABUF:-0}" in
    0) ;;
    1) staged_flag=--staged-dmabuf ;;
    *) echo "invalid S41_FFN_STAGED_DMABUF" >&2; exit 2 ;;
esac
max_tokens=${S41_FFN_MAX_TOKENS:-1}
column_quantum=${S41_FFN_COLUMN_QUANTUM:-32}
queue_depth=${S41_FFN_QUEUE_DEPTH:-1}
call_log_period=${S42_RESIDENT_CALL_LOG_PERIOD:-16}
case "$queue_depth" in
    1|2|3|4|5|6|7|8) ;;
    *) echo "invalid S41_FFN_QUEUE_DEPTH" >&2; exit 2 ;;
esac
case "$call_log_period" in
    ''|*[!0-9]*|0) echo "invalid S42_RESIDENT_CALL_LOG_PERIOD" >&2; exit 2 ;;
esac

if [ "$multi_session" = 1 ]; then
    if [ ! -x "$resident_workers" ] || [ ! -x "$resident_router" ]; then
        echo "multi-session phone binaries are unavailable" >&2
        exit 2
    fi
    case "$shard_manifest_rows" in
        *[!A-Za-z0-9,\.\;:/_-]*|'')
            echo "multi-session shard manifest is invalid" >&2
            exit 2
            ;;
    esac
    shard_manifest=$session_root/shards.csv
    printf '%s\n' "$shard_manifest_rows" | tr ';' '\n' > "$shard_manifest"
    session_count=$(wc -l < "$shard_manifest" | tr -d ' ')
    case "$session_count" in
        ''|*[!0-9]*|0) echo "multi-session shard count is invalid" >&2; exit 2 ;;
    esac
    session_capacity=${S42_PHONE_SESSION_COUNT:-$session_count}
    case "$session_capacity" in
        ''|*[!0-9]*|0) echo "multi-session capacity is invalid" >&2; exit 2 ;;
    esac
    if [ "$session_capacity" -lt "$session_count" ]; then
        echo "multi-session capacity is below the resident shard count" >&2
        exit 2
    fi
    GGML_HEXAGON_NDEV="$session_capacity" \
        GGML_HEXAGON_VMEM="${S41_FFN_VMEM:-3328}" \
        "$resident_workers" --manifest "$shard_manifest" --worker "$worker" \
        --model "$model" \
        --max-tokens "$max_tokens" --column-quantum "$column_quantum" \
        --max-requests 0 $io_flag > "$worker_log" 2>&1 &
    resident_workers_pid=$!
    printf '%s\n' "$resident_workers_pid" \
        > "$session_root/resident-workers.pid"
    workers_ready=0
    attempt=0
    while [ "$attempt" -lt 9600 ]; do
        if grep -q '^RESIDENTSHARDS ' "$worker_log" 2>/dev/null; then
            workers_ready=1
            break
        fi
        if ! kill -0 "$resident_workers_pid" 2>/dev/null; then
            break
        fi
        attempt=$((attempt + 1))
        sleep 0.05
    done
    if [ "$workers_ready" -ne 1 ]; then
        echo "[ffs-session] resident shards did not become warm" >&2
        tail -n 160 "$worker_log" >&2 || true
        exit 1
    fi
    "$resident_router" --manifest "$shard_manifest" \
        --ffs-root "$ffs_root" --ready-file "$ready_file" \
        --queue-depth "$queue_depth" --max-sessions 0 \
        --call-log-period "$call_log_period" \
        > "$router_log" 2>&1 &
    router_pid=$!
    worker_pid=$router_pid
else
    GGML_HEXAGON_NDEV=1 GGML_HEXAGON_VMEM="${S41_FFN_VMEM:-3328}" \
        "$worker" -m "$model" --layers "$layers" --columns "$columns" \
        --artifact-sha256 "$artifact_sha256" \
        --backend "$backend" --ffs-root "$ffs_root" \
        --ready-file "$ready_file" --max-requests "$max_requests" \
        --max-tokens "$max_tokens" --column-quantum "$column_quantum" \
        --queue-depth "$queue_depth" \
        $io_flag $staged_flag > "$worker_log" 2>&1 &
    worker_pid=$!
fi
if [ "$ncm_enabled" = 1 ]; then
    ln "$worker_log" "$diagnostic_root/residency.log"
    if [ "$multi_session" = 1 ]; then
        ln "$router_log" "$diagnostic_root/router.log"
    fi
fi
printf '%s\n' "$worker_pid" > "$session_root/worker.pid"

ready=0
attempt=0
while [ "$attempt" -lt 4800 ]; do
    if [ -f "$ready_file" ]; then
        ready=1
        break
    fi
    if ! kill -0 "$worker_pid" 2>/dev/null; then
        break
    fi
    attempt=$((attempt + 1))
    sleep 0.05
done
if [ "$ready" -ne 1 ]; then
    echo "[ffs-session] worker did not publish descriptors" >&2
    cat "$worker_log" >&2 || true
    if [ "$multi_session" = 1 ]; then
        cat "$router_log" >&2 || true
    fi
    exit 1
fi

setprop sys.usb.config none
attempt=0
while [ "$attempt" -lt 100 ]; do
    if [ -z "$(cat "$g1/UDC" 2>/dev/null)" ]; then
        break
    fi
    attempt=$((attempt + 1))
    sleep 0.05
done
if [ -n "$(cat "$g1/UDC" 2>/dev/null)" ]; then
    echo "Android USB gadget did not release the controller" >&2
    exit 1
fi
if [ -n "$(cat "$g2/UDC" 2>/dev/null)" ]; then
    printf '\n' > "$g2/UDC"
fi
rm -f "$g2/configs/b.1/f1"
rm -f "$g2/configs/b.1/f2"
printf '0x18d1' > "$g2/idVendor"
printf '0x2d00' > "$g2/idProduct"
printf '0x0320' > "$g2/bcdUSB"
printf '0x0100' > "$g2/bcdDevice"
printf '500' > "$g2/configs/b.1/MaxPower"
printf '0x80' > "$g2/configs/b.1/bmAttributes"
printf 'Heterogeneous inference' > "$g2/strings/0x409/manufacturer"
printf 'Direct FFN DMA-BUF' > "$g2/strings/0x409/product"
printf 'SCHEDFFN0001' > "$g2/strings/0x409/serialnumber"
printf 'ffn_htp_dmabuf' > "$g2/configs/b.1/strings/0x409/configuration"
ln -s "$g2/functions/ffs.s41" "$g2/configs/b.1/f1"
if [ "$ncm_enabled" = 1 ]; then
    ln -s "$g2/functions/ncm.usb0" "$g2/configs/b.1/f2"
fi

printf '%s' "$udc" > "$g2/UDC"
if [ "$ncm_enabled" = 1 ]; then
    attempt=0
    while [ "$attempt" -lt 100 ]; do
        if [ -e /sys/class/net/usb0 ]; then
            sleep 1
            ip link set usb0 up
            ip -4 addr add 192.168.42.1/24 dev usb0 2>/dev/null || true
            /system/bin/dnsmasq \
                --keep-in-foreground \
                --interface=usb0 \
                --bind-interfaces \
                --port=0 \
                --pid-file="$session_root/ncm-dnsmasq.pid" \
                --no-resolv \
                --no-hosts \
                --dhcp-range=192.168.42.2,192.168.42.2,12h \
                --dhcp-option=3 \
                --dhcp-option=6 \
                --dhcp-authoritative \
                --leasefile-ro \
                > "$session_root/ncm-dhcp.log" 2>&1 &
            ncm_dhcp_pid=$!
            break
        fi
        attempt=$((attempt + 1))
        sleep 0.05
    done
    if [ -z "$ncm_dhcp_pid" ] || ! kill -0 "$ncm_dhcp_pid" 2>/dev/null; then
        echo "direct FFN NCM setup failed" >&2
        exit 1
    fi
fi
echo "[ffs-session] custom FFN gadget bound" >> "$session_root/session.log"

set +e
wait "$worker_pid"
worker_status=$?
set -e
worker_pid=
router_pid=
if [ -n "$resident_workers_pid" ]; then
    kill -TERM "$resident_workers_pid" 2>/dev/null || true
fi
echo "[ffs-session] worker_status=$worker_status" >> "$session_root/session.log"
exit "$worker_status"
