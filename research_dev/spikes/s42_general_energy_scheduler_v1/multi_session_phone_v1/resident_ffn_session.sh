#!/system/bin/sh
set -eu

if [ "$#" -ne 8 ]; then
    echo "usage: $0 <workers> <router> <gemma-model> <qwen-model> <session-root> <restore-script> <timeout-seconds> <max-sessions>" >&2
    exit 2
fi

workers=$1
router=$2
gemma_model=$3
qwen_model=$4
session_root=$5
restore_script=$6
timeout_seconds=$7
max_sessions=$8
qwen_only=${S42_QWEN_ONLY:-0}
residency_layout=${S42_PHONE_RESIDENCY_LAYOUT:-gemma23-qwen12-full-v1}
if [ "$qwen_only" != 0 ] && [ "$qwen_only" != 1 ]; then
    echo "invalid S42_QWEN_ONLY: $qwen_only" >&2
    exit 2
fi
case "$residency_layout" in
    gemma23-qwen12-full-v1|gemma46-qwen6-full-v1) ;;
    *)
        echo "invalid S42_PHONE_RESIDENCY_LAYOUT: $residency_layout" >&2
        exit 2
        ;;
esac
if [ "$qwen_only" = 1 ] && \
        [ "$residency_layout" != gemma23-qwen12-full-v1 ]; then
    echo "Qwen-only mode requires gemma23-qwen12-full-v1" >&2
    exit 2
fi

g1=/config/usb_gadget/g1
g2=/config/usb_gadget/g2
udc=a600000.dwc3
ffs_root=/dev/usb-ffs/s41
ready_file=$session_root/router.ready
watchdog_sleep_file=$session_root/watchdog-sleep.pid
gemma_port=${S42_GEMMA_PORT:-26760}
qwen_port=${S42_QWEN_PORT:-26761}
qwen_high_port=${S42_QWEN_HIGH_PORT:-26762}
llama_ffn_model=${S42_LLAMA_FFN_MODEL:-}
llama_ffn_port=${S42_LLAMA_FFN_PORT:-18384}
llama_ffn_layers=${S42_LLAMA_FFN_LAYERS:-0-15}
minimum_available_kib=${S42_MIN_AVAILABLE_KIB:-2097152}
binary_root=$(CDPATH= cd -- "$(dirname -- "$workers")" && pwd)
ncm_function=
case "${S42_USB_NCM:-0}" in
    0) ;;
    1) ncm_function=$g2/functions/ncm.usb0 ;;
    *)
        echo "invalid S42_USB_NCM: ${S42_USB_NCM}" >&2
        exit 2
        ;;
esac
ncm_ipv4=${S42_USB_NCM_IPV4:-0}
case "$ncm_ipv4" in
    0) ;;
    1)
        if [ -z "$ncm_function" ]; then
            echo "S42_USB_NCM_IPV4 requires S42_USB_NCM=1" >&2
            exit 2
        fi
        ;;
    *)
        echo "invalid S42_USB_NCM_IPV4: $ncm_ipv4" >&2
        exit 2
        ;;
esac
task_binary=${S42_TASK_SERVER_BINARY:-}
task_model=${S42_TASK_SERVER_MODEL:-}
task_alias=${S42_TASK_SERVER_ALIAS:-llama-3.2-1b-instruct-q4_0}
task_port=${S42_TASK_SERVER_PORT:-18382}
task_ctx_size=${S42_TASK_SERVER_CTX_SIZE:-4096}
case "$task_ctx_size" in
    ''|*[!0-9]*)
        echo "invalid S42_TASK_SERVER_CTX_SIZE: $task_ctx_size" >&2
        exit 2
        ;;
esac
if [ "$task_ctx_size" -lt 1024 ]; then
    echo "S42_TASK_SERVER_CTX_SIZE is too small" >&2
    exit 2
fi
diagnostic_port=${S42_DIAGNOSTIC_PORT:-}
busybox_binary=${S42_BUSYBOX:-}
combined_ready_file=${S42_COMBINED_READY_FILE:-$session_root/combined.ready}
bind_arm_file=${S42_BIND_ARM_FILE:-}

max_temperature() {
    session_max=-1
    for session_path in "$@"; do
        session_value=$(cat "$session_path/temp" 2>/dev/null || true)
        case "$session_value" in
            ''|*[!0-9-]*) continue ;;
        esac
        if [ "$session_value" -gt "$session_max" ]; then
            session_max=$session_value
        fi
    done
    printf '%s' "$session_max"
}

workers_pid=
router_pid=
task_pid=
watchdog_pid=
ncm_dhcp_pid=
diagnostic_http_pid=
diagnostic_sampler_pid=
diagnostic_power_pid=
cleanup() {
    rm -f "$session_root/active"
    if [ -n "$watchdog_pid" ]; then
        if [ -f "$watchdog_sleep_file" ]; then
            watchdog_sleep_pid=$(cat "$watchdog_sleep_file" 2>/dev/null)
            case "$watchdog_sleep_pid" in
                *[!0-9]*|'') ;;
                *) kill "$watchdog_sleep_pid" 2>/dev/null || true ;;
            esac
        fi
        kill "$watchdog_pid" 2>/dev/null || true
        wait "$watchdog_pid" 2>/dev/null || true
        watchdog_pid=
        rm -f "$watchdog_sleep_file"
    fi
    for pid in "$diagnostic_http_pid" "$diagnostic_sampler_pid" \
            "$diagnostic_power_pid" \
            "$ncm_dhcp_pid" "$router_pid" "$task_pid" "$workers_pid"; do
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    sh "$restore_script" "$session_root" || true
}
trap cleanup EXIT INT TERM
trap '' HUP

mkdir -p "$session_root"
rm -f "$ready_file"
rm -f "$combined_ready_file"
rm -f "$watchdog_sleep_file"
sh "$restore_script" "$session_root" || true
: > "$session_root/active"
parent_pid=$$
(
    sleep "$timeout_seconds" &
    watchdog_sleep_pid=$!
    printf '%s\n' "$watchdog_sleep_pid" > "$watchdog_sleep_file"
    wait "$watchdog_sleep_pid"
    if [ -f "$session_root/active" ]; then
        echo "[resident-session] watchdog expired" >> "$session_root/session.log"
        kill -TERM "$parent_pid"
    fi
) &
watchdog_pid=$!

export LD_LIBRARY_PATH=$binary_root
export ADSP_LIBRARY_PATH=$binary_root
export GGML_HEXAGON_MBUF=4192
export GGML_HEXAGON_NHVX=${GGML_HEXAGON_NHVX:-8}
export S41_DISABLE_GRAPH_CACHE=1

if [ "$qwen_only" = 1 ]; then
    S42_RESIDENT_LAYOUT="$residency_layout" \
        S42_LLAMA_FFN_LAYERS="$llama_ffn_layers" \
        GGML_HEXAGON_NDEV=2 GGML_HEXAGON_VMEM=${S42_FFN_VMEM:-3328} \
        "$workers" "$qwen_model" "$qwen_port" "$qwen_high_port" \
        > "$session_root/resident-workers.log" 2>&1 &
elif [ -n "$llama_ffn_model" ]; then
    if [ ! -f "$llama_ffn_model" ]; then
        echo "[resident-session] missing Llama FFN model" >&2
        exit 1
    fi
    S42_RESIDENT_LAYOUT="$residency_layout" \
        GGML_HEXAGON_NDEV=4 GGML_HEXAGON_VMEM=${S42_FFN_VMEM:-3328} \
        "$workers" "$gemma_model" "$qwen_model" "$llama_ffn_model" \
        "$gemma_port" "$qwen_port" "$qwen_high_port" "$llama_ffn_port" \
        > "$session_root/resident-workers.log" 2>&1 &
else
    S42_RESIDENT_LAYOUT="$residency_layout" \
        GGML_HEXAGON_NDEV=3 GGML_HEXAGON_VMEM=${S42_FFN_VMEM:-3328} \
        "$workers" "$gemma_model" "$qwen_model" \
        "$gemma_port" "$qwen_port" "$qwen_high_port" \
        > "$session_root/resident-workers.log" 2>&1 &
fi
workers_pid=$!
printf '%s\n' "$workers_pid" > "$session_root/resident-workers.pid"

workers_ready=0
attempt=0
while [ "$attempt" -lt 7200 ]; do
    if grep -q '^RESIDENTWORKERS ' \
            "$session_root/resident-workers.log" 2>/dev/null; then
        workers_ready=1
        break
    fi
    if ! kill -0 "$workers_pid" 2>/dev/null; then
        break
    fi
    attempt=$((attempt + 1))
    sleep 0.05
done
if [ "$workers_ready" -ne 1 ]; then
    echo "[resident-session] resident workers did not become warm" >&2
    tail -n 160 "$session_root/resident-workers.log" >&2 || true
    cleanup
    trap - EXIT INT TERM
    exit 1
fi
available_kib=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
if [ "$available_kib" -lt "$minimum_available_kib" ]; then
    echo "[resident-session] memory reserve failed: $available_kib KiB" >&2
    cleanup
    trap - EXIT INT TERM
    exit 1
fi
if [ "$qwen_only" = 1 ]; then
    echo "[resident-session] HTP0+HTP1 QWEN WARM mem_available_kib=$available_kib" \
        >> "$session_root/session.log"
elif [ -n "$llama_ffn_model" ]; then
    echo "[resident-session] HTP0+HTP1+HTP2+HTP3 WARM mem_available_kib=$available_kib" \
        >> "$session_root/session.log"
else
    echo "[resident-session] HTP0+HTP1+HTP2 WARM mem_available_kib=$available_kib" \
        >> "$session_root/session.log"
fi
echo "RESIDENTSESSION {\"status\":\"WARM\",\"layout\":\"$residency_layout\",\"mem_available_kib\":$available_kib}" \
    >> "$session_root/session.log"

if [ -n "$task_binary" ]; then
    if [ -z "$task_model" ] || [ ! -x "$task_binary" ] || \
            [ ! -f "$task_model" ]; then
        echo "[resident-session] invalid task server binding" >&2
        cleanup
        trap - EXIT INT TERM
        exit 1
    fi
    task_binary_root=$(CDPATH= cd -- "$(dirname -- "$task_binary")" && pwd)
    (
        cd "$task_binary_root"
        export LD_LIBRARY_PATH=.
        exec "$task_binary" \
            --model "$task_model" \
            --alias "$task_alias" \
            --ctx-size "$task_ctx_size" \
            --parallel 1 \
            --batch-size 1024 \
            --ubatch-size 256 \
            --cont-batching \
            --cache-type-k f16 \
            --cache-type-v f16 \
            --cache-ram 0 \
            --host :: \
            --port "$task_port" \
            --n-gpu-layers 99 \
            --device GPUOpenCL \
            --flash-attn off \
            --no-webui \
            --log-colors off \
            --log-verbosity 4 \
            --log-timestamps
    ) > "$session_root/task-server.log" 2>&1 &
    task_pid=$!
    printf '%s\n' "$task_pid" > "$session_root/task-server.pid"
    task_ready=0
    attempt=0
    while [ "$attempt" -lt 2400 ]; do
        if grep -q 'listening on http://' \
                "$session_root/task-server.log" 2>/dev/null; then
            task_ready=1
            break
        fi
        if ! kill -0 "$task_pid" 2>/dev/null; then
            break
        fi
        attempt=$((attempt + 1))
        sleep 0.05
    done
    if [ "$task_ready" -ne 1 ]; then
        echo "[resident-session] task server did not become ready" >&2
        tail -n 160 "$session_root/task-server.log" >&2 || true
        cleanup
        trap - EXIT INT TERM
        exit 1
    fi
    combined_available_kib=$(awk \
        '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    combined_minimum_kib=${S42_COMBINED_MIN_AVAILABLE_KIB:-1048576}
    if [ "$combined_available_kib" -lt "$combined_minimum_kib" ]; then
        echo "[resident-session] combined memory reserve failed: $combined_available_kib KiB" >&2
        cleanup
        trap - EXIT INT TERM
        exit 1
    fi
    echo "COMBINEDRESIDENCY {\"status\":\"WARM\",\"task_alias\":\"$task_alias\",\"task_port\":$task_port,\"mem_available_kib\":$combined_available_kib}" \
        >> "$session_root/session.log"
fi

if [ -n "$diagnostic_port" ]; then
    case "$diagnostic_port" in
        *[!0-9]*|'')
            echo "[resident-session] invalid diagnostic port" >&2
            cleanup
            trap - EXIT INT TERM
            exit 1
            ;;
    esac
    if [ -z "$busybox_binary" ] || [ ! -x "$busybox_binary" ]; then
        echo "[resident-session] diagnostic busybox is unavailable" >&2
        cleanup
        trap - EXIT INT TERM
        exit 1
    fi
    diagnostic_root=$session_root/diagnostics
    mkdir -p "$diagnostic_root"
    (
        while [ -f "$session_root/active" ]; do
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
            printf '%s\n' \
                "{\"battery_charge_counter_uah\":$battery_charge_counter_uah,\"battery_current_ma\":$battery_current_ma,\"battery_voltage_uv\":$battery_voltage_uv,\"schema\":\"s42-op15-live-power-v1\",\"uptime_s\":$uptime_s,\"usb_current_ua\":$usb_current_ua,\"usb_voltage_uv\":$usb_voltage_uv}" \
                > "$temporary"
            mv "$temporary" "$diagnostic_root/power.json"
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
            temperature_battery_millic=$(max_temperature \
                /sys/class/thermal/thermal_zone93)
            temperature_shell_millic=$(max_temperature \
                /sys/class/thermal/thermal_zone55 \
                /sys/class/thermal/thermal_zone61 \
                /sys/class/thermal/thermal_zone65)
            temperature_cpu_millic=$(max_temperature \
                /sys/class/thermal/thermal_zone[0-9] \
                /sys/class/thermal/thermal_zone1[0-8] \
                /sys/class/thermal/thermal_zone2[4-7])
            temperature_npu_millic=$(max_temperature \
                /sys/class/thermal/thermal_zone2[89] \
                /sys/class/thermal/thermal_zone3[0-5])
            temperature_gpu_millic=$(max_temperature \
                /sys/class/thermal/thermal_zone3[6-9] \
                /sys/class/thermal/thermal_zone4[0-6])
            temperature_ddr_millic=$(max_temperature \
                /sys/class/thermal/thermal_zone47)
            temperature_max_millic=$(printf '%s\n' \
                "$temperature_battery_millic" \
                "$temperature_shell_millic" \
                "$temperature_cpu_millic" \
                "$temperature_npu_millic" \
                "$temperature_gpu_millic" \
                "$temperature_ddr_millic" | awk '
                    BEGIN { maximum = -1 }
                    { if ($1 + 0 > maximum) maximum = $1 + 0 }
                    END { printf "%d", maximum }
                ')
            android_thermal_status=$(dumpsys thermalservice 2>/dev/null | \
                awk '/^Thermal Status:/ { print $3; exit }')
            case "$android_thermal_status" in
                ''|*[!0-9]*)
                    android_thermal_status=-1
                    thermal_state=unqualified
                    throttling_state=unqualified
                    ;;
                0)
                    thermal_state=nominal
                    throttling_state=not-observed
                    ;;
                *)
                    thermal_state=hot
                    throttling_state=android-thermal-throttling
                    ;;
            esac
            task_server_alive=false
            if [ -n "$task_pid" ] && kill -0 "$task_pid" 2>/dev/null; then
                task_server_alive=true
            fi
            temporary=$diagnostic_root/snapshot.json.tmp
            printf '%s\n' \
                "{\"android_thermal_status\":$android_thermal_status,\"captured_epoch_s\":$captured_epoch_s,\"mem_available_kib\":$mem_available_kib,\"mem_total_kib\":$mem_total_kib,\"schema\":\"s42-op15-live-snapshot-v1\",\"sequence\":$sequence,\"task_server_alive\":$task_server_alive,\"temperature_battery_millic\":$temperature_battery_millic,\"temperature_cpu_millic\":$temperature_cpu_millic,\"temperature_ddr_millic\":$temperature_ddr_millic,\"temperature_gpu_millic\":$temperature_gpu_millic,\"temperature_max_millic\":$temperature_max_millic,\"temperature_npu_millic\":$temperature_npu_millic,\"temperature_shell_millic\":$temperature_shell_millic,\"thermal_state\":\"$thermal_state\",\"throttling_state\":\"$throttling_state\"}" \
                > "$temporary"
            mv "$temporary" "$diagnostic_root/snapshot.json"
            sleep 2
        done
    ) &
    diagnostic_sampler_pid=$!
    "$busybox_binary" httpd -f -p "$diagnostic_port" \
        -h "$diagnostic_root" \
        > "$session_root/diagnostic-http.log" 2>&1 &
    diagnostic_http_pid=$!
fi

printf '%s\n' "${combined_available_kib:-$available_kib}" \
    > "$combined_ready_file"
if [ -n "$bind_arm_file" ]; then
    attempt=0
    while [ "$attempt" -lt 2400 ] && [ ! -f "$bind_arm_file" ]; do
        if ! kill -0 "$workers_pid" 2>/dev/null; then
            break
        fi
        if [ -n "$task_pid" ] && ! kill -0 "$task_pid" 2>/dev/null; then
            break
        fi
        attempt=$((attempt + 1))
        sleep 0.05
    done
    if [ ! -f "$bind_arm_file" ]; then
        echo "[resident-session] USB bind was not armed" >&2
        cleanup
        trap - EXIT INT TERM
        exit 1
    fi
fi

mkdir -p "$ffs_root"
mkdir "$g2/functions/ffs.s41"
if [ -n "$ncm_function" ]; then
    mkdir "$ncm_function"
fi
mount -t functionfs s41 "$ffs_root"
if [ "$qwen_only" = 1 ]; then
    "$router" --ffs-root "$ffs_root" --ready-file "$ready_file" \
        --target "$qwen_port" --target "$qwen_high_port" \
        --max-sessions "$max_sessions" \
        > "$session_root/router.log" 2>&1 &
else
    "$router" --ffs-root "$ffs_root" --ready-file "$ready_file" \
        --target "$gemma_port" --target "$qwen_port" \
        --target "$qwen_high_port" \
        --max-sessions "$max_sessions" \
        > "$session_root/router.log" 2>&1 &
fi
router_pid=$!
printf '%s\n' "$router_pid" > "$session_root/router.pid"

ready=0
attempt=0
while [ "$attempt" -lt 1200 ]; do
    if [ -f "$ready_file" ]; then
        ready=1
        break
    fi
    if ! kill -0 "$router_pid" 2>/dev/null; then
        break
    fi
    attempt=$((attempt + 1))
    sleep 0.05
done
if [ "$ready" -ne 1 ]; then
    echo "[resident-session] router did not publish descriptors" >&2
    cat "$session_root/router.log" >&2 || true
    cleanup
    trap - EXIT INT TERM
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
printf 'Research scheduler' > "$g2/strings/0x409/manufacturer"
printf 'Resident FFN router' > "$g2/strings/0x409/product"
printf 'S42RESIDENT1' > "$g2/strings/0x409/serialnumber"
printf 'resident_ffn' > "$g2/configs/b.1/strings/0x409/configuration"
ln -s "$g2/functions/ffs.s41" "$g2/configs/b.1/f1"
if [ -n "$ncm_function" ]; then
    ln -s "$ncm_function" "$g2/configs/b.1/f2"
fi

printf '\n' > "$g1/UDC"
printf '%s' "$udc" > "$g2/UDC"
if [ -n "$ncm_function" ]; then
    attempt=0
    while [ "$attempt" -lt 100 ]; do
        if [ -e /sys/class/net/usb0 ]; then
            sleep 1
            printf '0' > /proc/sys/net/ipv6/conf/usb0/disable_ipv6
            printf '0' > /proc/sys/net/ipv6/conf/usb0/accept_dad
            ip link set usb0 up
            ip -6 addr add fe80::2/64 dev usb0 2>/dev/null || true
            ip -6 route replace fe80::/64 dev usb0 table 1033
            ip -6 rule add priority 9999 from fe80::2/128 to fe80::/64 \
                lookup 1033 2>/dev/null || true
            if [ "$ncm_ipv4" = 1 ]; then
                ip -4 addr add 192.168.42.1/24 dev usb0 \
                    2>/dev/null || true
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
                sleep 0.1
                if ! kill -0 "$ncm_dhcp_pid" 2>/dev/null; then
                    echo "[resident-session] NCM DHCP failed" >&2
                    cat "$session_root/ncm-dhcp.log" >&2 || true
                    cleanup
                    trap - EXIT INT TERM
                    exit 1
                fi
                ip -4 addr show dev usb0 >> "$session_root/session.log"
            fi
            ip -6 addr show dev usb0 >> "$session_root/session.log"
            break
        fi
        attempt=$((attempt + 1))
        sleep 0.05
    done
fi
echo "[resident-session] router bound" >> "$session_root/session.log"

set +e
wait "$router_pid"
router_status=$?
set -e
router_pid=
echo "[resident-session] router_status=$router_status" \
    >> "$session_root/session.log"
cleanup
trap - EXIT INT TERM
exit "$router_status"
