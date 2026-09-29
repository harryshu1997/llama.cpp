#!/system/bin/sh
# Temporary, OP12-only diagnostic. Run with Magisk su -mm in the master
# mount namespace. Never changes USB properties, SELinux, or boot partitions.
# A stop file or a 600-second watchdog restores the original ADB composition.
set -eu

session=/data/local/tmp/moe_ffs_20260909
gadget=/config/usb_gadget/g1
function=$gadget/functions/ffs.moe_root_20260909
link=$gadget/configs/b.1/moe_root_20260909
mount_path=/dev/usb-ffs/moe_root_20260909
service_pid=
created_function=0
created_mount_dir=0
mounted=0
usb_changed=0
initial_udc=
watchdog_limit=${OP12_FFS_WATCHDOG_SECONDS:-600}
case "$watchdog_limit" in ''|*[!0-9]*) exit 2;; esac
[ "$watchdog_limit" -ge 10 ] && [ "$watchdog_limit" -le 600 ]
completion_poll=${OP12_FFS_COMPLETION_POLL:-0}
cpu_mask=${OP12_FFS_CPU_MASK:-}
nhvx=${OP12_FFS_NHVX:-0}
q8=${OP12_FFS_Q8:-0}
ffs_version=${OP12_FFS_VERSION:-1}
session_tag=${OP12_FFS_SESSION_TAG:-}
case "$completion_poll" in 0|1) ;; *) exit 2;; esac
case "$cpu_mask" in ''|ff|fc|80) ;; *) exit 2;; esac
case "$nhvx" in 0|1|2|4) ;; *) exit 2;; esac
case "$q8" in 0|1) ;; *) exit 2;; esac
case "$ffs_version" in 1|2) ;; *) exit 2;; esac
service_binary=$session/op12_htp_service
if [ "$q8" = 1 ]; then service_binary=$session/op12_htp_service_q8_ffs_v2; fi
case "$session_tag" in *[!a-zA-Z0-9_]*) exit 2;; esac
service_log=$session/service${session_tag:+_$session_tag}.log
stop_file=$session/stop${session_tag:+_$session_tag}

cleanup() {
    cleanup_status=$?
    trap - EXIT INT TERM HUP
    set +e
    if [ "$usb_changed" = 1 ]; then
        printf '\n' > "$gadget/UDC"
        if [ -L "$link" ] && [ "$(readlink -f "$link")" = "$function" ]; then
            rm "$link"
        fi
    fi
    if [ -n "$service_pid" ]; then
        kill -TERM "$service_pid" 2>/dev/null
        wait "$service_pid" 2>/dev/null
    fi
    if [ "$usb_changed" = 1 ]; then
        printf '%s\n' "$initial_udc" > "$gadget/UDC"
    fi
    if [ "$mounted" = 1 ]; then umount "$mount_path"; fi
    if [ "$created_mount_dir" = 1 ]; then rmdir "$mount_path"; fi
    if [ "$created_function" = 1 ]; then rmdir "$function"; fi
    # Android's USB service can briefly unbind/rebind while ADB is restored.
    # Wait for two stable observations before recording the final controller.
    stable=0
    attempt=0
    while [ "$stable" -lt 2 ] && [ "$attempt" -lt 20 ]; do
        if [ "$(cat "$gadget/UDC")" = "$initial_udc" ] &&
           [ "$(getprop sys.usb.state)" = adb ]; then
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
    printf 'RESTORED_SELINUX=%s\n' "$(getenforce)"
    if [ "$stable" -lt 2 ] || [ "$(cat "$gadget/UDC")" != "$initial_udc" ] ||
       [ -L "$link" ] || [ -e "$function" ] || [ -e "$mount_path" ]; then
        printf 'RESTORATION_FAILED=1\n'
        cleanup_status=2
    fi
    printf 'SESSION_EXIT_STATUS=%s\n' "$cleanup_status"
    exit "$cleanup_status"
}

[ "$(id -u)" = 0 ]
[ "$(getprop ro.serialno)" = 5ae7a43d ]
[ "$(getprop ro.product.model)" = CPH2583 ]
[ "$(getprop sys.usb.config)" = adb ]
[ "$(getprop sys.usb.state)" = adb ]
[ "$(getenforce)" = Enforcing ]
[ "$(readlink -f "$gadget/configs/b.1/f1")" = "$gadget/functions/ffs.adb" ]
for entry in "$gadget/configs/b.1/"*; do
    if [ -L "$entry" ]; then [ "$entry" = "$gadget/configs/b.1/f1" ]; fi
done
[ ! -e "$function" ]
[ ! -e "$mount_path" ]
[ ! -e "$link" ] && [ ! -L "$link" ]
[ ! -e "$stop_file" ]
[ -x "$service_binary" ]
[ "$(stat -c %s "$session/bank.fp16")" = 603979776 ]
initial_udc=$(cat "$gadget/UDC")
[ "$initial_udc" = a600000.dwc3 ]
trap cleanup EXIT
trap 'exit 143' INT TERM HUP

printf 'CONTROLLER_PID=%s\n' "$$"
printf '%s\n' "$$" > "$session/controller.pid"
printf 'INITIAL_USB_CONFIG=%s\n' "$(getprop sys.usb.config)"
printf 'INITIAL_UDC=%s\n' "$initial_udc"
printf 'PROFILE completion_poll=%s cpu_mask=%s nhvx=%s tag=%s\n' \
    "$completion_poll" "${cpu_mask:-inherited}" "$nhvx" "$session_tag"
mkdir "$function"
created_function=1
mkdir "$mount_path"
created_mount_dir=1
mount -t functionfs -o uid=0,gid=0,mode=0600,rmode=0700 moe_root_20260909 "$mount_path"
mounted=1

set --
if [ "$q8" = 1 ]; then set -- --q8-weights --hmx-pad-rows 8; fi
set -- "$service_binary" --weights "$session/bank.fp16" \
    --layers 6 --experts 16 --warmups 2 --port 27183 \
    --transport tcp+functionfs --ffs-dir "$mount_path" \
    --ffs-version "$ffs_version" --allow-copy-fallback "$@"
if [ -n "$cpu_mask" ]; then set -- taskset "$cpu_mask" "$@"; fi
env LD_LIBRARY_PATH="$session" ADSP_LIBRARY_PATH="$session" \
    GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPPOLL="$completion_poll" \
    GGML_HEXAGON_NHVX="$nhvx" GGML_HEXAGON_NHMX=1 GGML_HEXAGON_PROFILE=0 \
    "$@" > "$service_log" 2>&1 &
service_pid=$!
printf '%s\n' "$service_pid" > "$session/service.pid"
attempt=0
while ! grep -q FFS_DESCRIPTORS_READY "$service_log"; do
    kill -0 "$service_pid"
    attempt=$((attempt + 1))
    [ "$attempt" -lt 120 ]
    sleep 1
done

# No USB disconnection occurs until the descriptors and service are ready.
# The original ADB symlink, USB IDs, and properties remain untouched.
usb_changed=1
printf '\n' > "$gadget/UDC"
ln -s "$function" "$link"
printf '%s\n' "$initial_udc" > "$gadget/UDC"
printf 'TEMPORARY_ADB_PLUS_FUNCTIONFS_BOUND=1\n'

elapsed=0
while [ ! -e "$stop_file" ] && [ "$elapsed" -lt "$watchdog_limit" ]; do
    kill -0 "$service_pid"
    sleep 1
    elapsed=$((elapsed + 1))
done
printf 'STOP_REQUESTED_OR_WATCHDOG_EXPIRED=1\n'
