#!/system/bin/sh
# Temporary OP15-only Gemma session. Run detached with Magisk su -mm.
# Preserve PTP, ADB, USB properties, SELinux and all unrelated gadget functions.
set -eu

session=${GEMMA_FFS_SESSION_DIR:-/data/local/tmp/gemma4_energy_b1_20260910}
bank=/data/local/tmp/gemma4_b1_20260910/bank.g4b
gadget=/config/usb_gadget/g1
function=$gadget/functions/ffs.gemma_b1_20260910
link=$gadget/configs/b.1/gemma_b1_20260910
mount_path=/dev/usb-ffs/gemma_b1_20260910
layers=${GEMMA_FFS_LAYERS:-30}
direct_mode=${GEMMA_FFS_V4_MODE:-}
case "$direct_mode" in ''|bulk|expert) ;; *) exit 2;; esac
if [ -n "$direct_mode" ]; then
    [ "${GEMMA_DIRECT_DEVICE_WINDOW:-}" = coordinated ] || { echo 'A coordinated device window is required.' >&2; exit 2; }
fi
if [ "$session" != /data/local/tmp/gemma4_energy_b1_20260910 ]; then
    case "$session" in /data/local/tmp/gemma_direct_v4_*) ;; *) echo 'Invalid isolated direct session path.' >&2; exit 2;; esac
    session_suffix=${session#/data/local/tmp/gemma_direct_v4_}
    case "$session_suffix" in ''|*[!A-Za-z0-9_]*) echo 'Invalid isolated direct session path.' >&2; exit 2;; esac
    [ -n "$direct_mode" ] || exit 2
fi
watchdog_limit=${GEMMA_FFS_WATCHDOG_SECONDS:-3600}
case "$layers:$watchdog_limit" in *[!0-9:]*) exit 2;; esac
[ "$layers" -ge 1 ] && [ "$layers" -le 30 ]
[ "$watchdog_limit" -ge 30 ] && [ "$watchdog_limit" -le 7200 ]
service_pid=
service_start=
control_token=
created_function=0
created_mount_dir=0
mounted=0
usb_changed=0
initial_udc=
initial_config=
initial_state=

bind_original_controller() {
    # Android's USB HAL may bind the same gadget immediately after our unlink/link.
    # EBUSY is harmless only when the exact recorded controller is already bound.
    [ "$(cat "$gadget/UDC")" = "$initial_udc" ] && return 0
    printf '%s\n' "$initial_udc" > "$gadget/UDC" ||
        [ "$(cat "$gadget/UDC")" = "$initial_udc" ]
}

cleanup() {
    cleanup_status=$?
    trap - EXIT INT TERM HUP
    set +e
    if [ -n "$direct_mode" ] && [ -n "$service_pid" ]; then
        if ! direct_shutdown; then
            printf 'DIRECT_LIFECYCLE_FAILED=1 NO_FORCED_UNBIND_OR_KILL=1\n'
            exit 2
        fi
        if [ "$direct_outcome" = abort ]; then cleanup_status=2; fi
    elif [ -n "$service_pid" ]; then
        kill -TERM "$service_pid" 2>/dev/null
        attempt=0
        while kill -0 "$service_pid" 2>/dev/null && [ "$attempt" -lt 60 ]; do
            sleep 1
            attempt=$((attempt + 1))
        done
        if kill -0 "$service_pid" 2>/dev/null; then
            printf 'DRAIN_FAILED_SERVICE_STILL_RUNNING=1 USB_LEFT_BOUND=1\n'
            exit 2
        fi
        wait "$service_pid" 2>/dev/null
        service_status=$?
        printf 'SERVICE_EXIT_STATUS=%s\n' "$service_status"
        if [ "$service_status" != 0 ] || ! grep -q 'GEMMA_FFS_DRAIN outstanding_dma=0' "$session/service.log"; then
            printf 'DRAIN_FAILED=1 USB_LEFT_BOUND=1\n'
            exit 2
        fi
    fi
    if [ "$usb_changed" = 1 ]; then
        if [ -L "$link" ] && [ "$(readlink -f "$link")" = "$function" ]; then
            "$session/gemma_usb_bind" remove
        fi
    fi
    if [ "$usb_changed" = 1 ]; then
        bind_original_controller
    fi
    if [ "$mounted" = 1 ]; then umount "$mount_path"; fi
    if [ "$created_mount_dir" = 1 ]; then rmdir "$mount_path"; fi
    if [ "$created_function" = 1 ]; then rmdir "$function"; fi
    stable=0
    attempt=0
    while [ "$stable" -lt 2 ] && [ "$attempt" -lt 30 ]; do
        if [ "$(cat "$gadget/UDC")" = "$initial_udc" ] &&
           [ "$(getprop sys.usb.config)" = "$initial_config" ] &&
           [ "$(getprop sys.usb.state)" = "$initial_state" ]; then
            stable=$((stable + 1))
        else
            stable=0
        fi
        attempt=$((attempt + 1))
        sleep 0.2
    done
    printf 'RESTORED_UDC=%s\n' "$(cat "$gadget/UDC")"
    printf 'RESTORED_USB_CONFIG=%s\n' "$(getprop sys.usb.config)"
    printf 'RESTORED_USB_STATE=%s\n' "$(getprop sys.usb.state)"
    if [ "$stable" -lt 2 ] || [ -L "$link" ] || [ -e "$function" ] || [ -e "$mount_path" ] ||
       [ "$(readlink -f "$gadget/configs/b.1/f1")" != "$gadget/functions/ffs.ptp" ] ||
       [ "$(readlink -f "$gadget/configs/b.1/f2")" != "$gadget/functions/ffs.adb" ]; then
        printf 'RESTORATION_FAILED=1\n'
        cleanup_status=2
    fi
    if [ -z "$direct_mode" ]; then rmdir "$session/control.lock"; fi
    printf 'SESSION_EXIT_STATUS=%s\n' "$cleanup_status"
    exit "$cleanup_status"
}

[ "$(id -u)" = 0 ]
[ "$(getprop ro.serialno)" = 3C15AU002CL00000 ]
[ "$(getprop ro.product.model)" = CPH2749 ]
[ "$(getprop ro.soc.model)" = SM8850 ]
# Gate direct invocation too, before any lock, mount, service or USB change.
kernel_notes=$(sha256sum /sys/kernel/notes)
kernel_btf=$(sha256sum /sys/kernel/btf/vmlinux)
kernel_config=$(zcat /proc/config.gz | sha256sum)
if [ "${kernel_notes%% *}" != 40bbacf73c0d35195693c566ae695803f59fbf7ec8ce817a1921fb44b3807b51 ] ||
   [ "${kernel_btf%% *}" != 77a8ce5acc215506e8dff1bf84e4561ef316680bcba803e9dfc16acecc449735 ] ||
   [ "${kernel_config%% *}" != 9f03ed30a44329ebc6337dca7157f3eaa67c3143519883b026c51abd0d7dda43 ]; then
    printf 'KERNEL_GUARD_FAILED=1 NO_USB_OR_SERVICE_CHANGES=1\n' >&2
    exit 2
fi
[ "$(getenforce)" = Enforcing ]
if [ -n "$direct_mode" ] && ps -A | grep -Eq 'direct_phone_service|gemma4_htp_service|op12_htp_service|ffn-split-worker'; then
    echo 'A phone compute worker is already running; refusing the direct session.' >&2
    exit 2
fi
initial_config=$(getprop sys.usb.config)
initial_state=$(getprop sys.usb.state)
[ "$initial_config" = ptp,adb ] && [ "$initial_state" = ptp,adb ]
[ "$(readlink -f "$gadget/configs/b.1/f1")" = "$gadget/functions/ffs.ptp" ]
[ "$(readlink -f "$gadget/configs/b.1/f2")" = "$gadget/functions/ffs.adb" ]
for entry in "$gadget/configs/b.1/"*; do
    if [ -L "$entry" ]; then
        case "$entry" in "$gadget/configs/b.1/f1"|"$gadget/configs/b.1/f2") ;; *) exit 2;; esac
    fi
done
[ ! -e "$function" ] && [ ! -e "$mount_path" ]
[ ! -e "$link" ] && [ ! -L "$link" ]
[ ! -e "$session/stop" ]
if [ -n "$direct_mode" ]; then
    case "$session" in /data/local/tmp/gemma_direct_v4_*) ;; *) echo 'v4 requires a fresh isolated session.' >&2; exit 2;; esac
    [ -f "$session/direct_usb_lifecycle.sh" ]
    . "$session/direct_usb_lifecycle.sh"
    [ -x "$session/direct_phone_service" ]
    env LD_LIBRARY_PATH="$session" "$session/direct_phone_service" --execute --kernel-only
else
    [ -x "$session/gemma4_htp_service" ]
fi
[ -x "$session/gemma_usb_bind" ]
# Android mksh arithmetic is signed 32-bit; the 30-layer FP16 bank exceeds 4 GiB.
expected_bank_bytes=$(awk -v count="$layers" 'BEGIN { printf "%.0f", 160 + count * 16 * 3 * 2816 * 704 * 2 }')
if [ "$direct_mode" != bulk ]; then [ "$(stat -c %s "$bank")" = "$expected_bank_bytes" ]; fi
initial_udc=$(cat "$gadget/UDC")
[ "$initial_udc" = a600000.dwc3 ]
umask 077
mkdir -m 700 "$session/control.lock"
if [ -n "$direct_mode" ]; then
    control_token=$(cat /proc/sys/kernel/random/uuid)
fi
trap cleanup EXIT
trap 'exit 143' INT TERM HUP
printf '%s\n' "$$" > "$session/controller.pid"
printf 'INITIAL_USB_CONFIG=%s INITIAL_UDC=%s\n' "$initial_config" "$initial_udc"

mkdir "$function"
created_function=1
mkdir "$mount_path"
created_mount_dir=1
mount -t functionfs -o uid=0,gid=0,mode=0600,rmode=0700 gemma_b1_20260910 "$mount_path"
mounted=1
if [ -n "$direct_mode" ]; then
    if [ "$direct_mode" = expert ]; then
        set -- --weights "$bank" --layers "$layers" --out-bytes 24576 --in-bytes 24576 --total 0 --depth 4
    else
        direct_out=${GEMMA_DIRECT_OUT_BYTES:-1048576}
        direct_in=${GEMMA_DIRECT_IN_BYTES:-1048576}
        direct_total=${GEMMA_DIRECT_TOTAL:-220}
        direct_depth=${GEMMA_DIRECT_DEPTH:-4}
        case "$direct_out:$direct_in:$direct_total:$direct_depth" in *[!0-9:]*) exit 2;; esac
        set -- --out-bytes "$direct_out" --in-bytes "$direct_in" --total "$direct_total" --depth "$direct_depth"
        if [ "${GEMMA_DIRECT_VERIFY_FULL:-0}" = 1 ]; then set -- "$@" --verify-full; fi
        if [ "${GEMMA_DIRECT_CANCEL_PROBE:-0}" = 1 ]; then set -- "$@" --cancellation-probe; fi
    fi
    env LD_LIBRARY_PATH="$session" ADSP_LIBRARY_PATH="$session" \
        GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPPOLL=0 GGML_HEXAGON_NHVX=0 GGML_HEXAGON_NHMX=1 GGML_HEXAGON_PROFILE=0 \
        "$session/direct_phone_service" --execute --ffs-dir "$mount_path" \
        --control-dir "$session/control.lock" --control-token "$control_token" "$@" > "$session/service.log" 2>&1 &
else
env LD_LIBRARY_PATH="$session" ADSP_LIBRARY_PATH="$session" \
    GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPPOLL=0 \
    GGML_HEXAGON_NHVX=0 GGML_HEXAGON_NHMX=1 GGML_HEXAGON_PROFILE=0 \
    "$session/gemma4_htp_service" --weights "$bank" \
    --layers "$layers" --experts 16 --warmups 0 --port 28165 \
    --q8-weights --group-experts --hmx-pad-rows 8 \
    --transport functionfs --ffs-version 3 --ffs-dir "$mount_path" \
    --dma-heap /dev/dma_heap/system > "$session/service.log" 2>&1 &
fi
service_pid=$!
if [ -n "$direct_mode" ]; then
    service_identity=$(direct_process_identity "$service_pid")
    service_start=${service_identity#* }
fi
printf '%s\n' "$service_pid" > "$session/service.pid"
attempt=0
while ! grep -q FFS_DESCRIPTORS_READY "$session/service.log"; do
    kill -0 "$service_pid"
    attempt=$((attempt + 1))
    [ "$attempt" -lt 180 ]
    sleep 1
done

# Re-enumerate only after the new function has valid descriptors.
# The original PTP/ADB links, USB IDs and properties are never rewritten.
usb_changed=1
if [ -n "$direct_mode" ]; then direct_bind add; else "$session/gemma_usb_bind" add; fi
printf 'TEMPORARY_PTP_ADB_GEMMA_FUNCTIONFS_BOUND=1\n'
elapsed=0
while [ ! -e "$session/stop" ] && [ "$elapsed" -lt "$watchdog_limit" ]; do
    if [ -n "$direct_mode" ] && direct_read_receipt; then
        printf 'DIRECT_SERVICE_QUIESCED=1\n'
        break
    fi
    if ! kill -0 "$service_pid" 2>/dev/null; then
        if [ -n "$direct_mode" ]; then
            printf 'DIRECT_SERVICE_FINISHED=1\n'
            break
        fi
        exit 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done
printf 'STOP_REQUESTED_OR_WATCHDOG_EXPIRED=1\n'
