# Sourced only by our managed v4 session. No commands run when this file is sourced.
direct_event() {
    read -r direct_uptime direct_unused < /proc/uptime
    printf 'DIRECT_CONTROLLER time=%s stage=%s\n' "$direct_uptime" "$1"
}

direct_process_identity() {
    [ -r "/proc/$1/stat" ] || return 1
    read -r direct_stat < "/proc/$1/stat" || return 1
    direct_fields=${direct_stat##*) }
    set -- $direct_fields
    [ "$#" -ge 20 ] || return 1
    direct_state=$1
    shift 19
    printf '%s %s\n' "$direct_state" "$1"
}

direct_service_alive() {
    direct_identity=$(direct_process_identity "$service_pid") || return 1
    [ "$direct_identity" = "S $service_start" ] ||
    [ "$direct_identity" = "R $service_start" ] ||
    [ "$direct_identity" = "D $service_start" ] ||
    [ "$direct_identity" = "I $service_start" ] ||
    [ "$direct_identity" = "T $service_start" ] ||
    [ "$direct_identity" = "t $service_start" ]
}

direct_read_receipt() {
    [ -f "$session/control.lock/drained" ] && [ ! -L "$session/control.lock/drained" ] || return 1
    direct_receipt=$(cat "$session/control.lock/drained") || return 1
    case "$direct_receipt" in
        "DIRECT_LIFECYCLE_V1 $control_token $service_pid $service_start normal") direct_outcome=normal;;
        "DIRECT_LIFECYCLE_V1 $control_token $service_pid $service_start abort") direct_outcome=abort;;
        *) return 1;;
    esac
}

direct_request_stop() {
    direct_service_alive || return 1
    kill -TERM "$service_pid"
}

direct_bind() {
    direct_event "bind_${1}_begin"
    direct_bind_uncertain=1
    "$session/gemma_usb_bind" "$1" &
    direct_bind_pid=$!
    direct_bind_identity=$(direct_process_identity "$direct_bind_pid") || direct_bind_identity=
    direct_bind_start=${direct_bind_identity#* }
    direct_bind_attempt=0
    while [ "$direct_bind_attempt" -lt 15 ]; do
        direct_bind_current=$(direct_process_identity "$direct_bind_pid") || break
        case "$direct_bind_current" in "Z "*|"X "*) break;; esac
        [ "${direct_bind_current#* }" = "$direct_bind_start" ] || break
        sleep 1
        direct_bind_attempt=$((direct_bind_attempt + 1))
    done
    if [ "$direct_bind_attempt" -ge 15 ]; then
        direct_event "bind_timeout_helper_retained_pid_${direct_bind_pid}"
        return 1
    fi
    wait "$direct_bind_pid"
    direct_bind_status=$?
    direct_bind_uncertain=0
    direct_event "bind_${1}_done_status_${direct_bind_status}"
    return "$direct_bind_status"
}

direct_restore() {
    [ "${direct_bind_uncertain:-0}" = 0 ] || return 1
    direct_service_alive && direct_read_receipt || return 1
    if [ "$usb_changed" = 1 ]; then
        if [ -L "$link" ]; then
            [ "$(readlink -f "$link")" = "$function" ] || return 1
            direct_bind remove || return 1
        else
            [ ! -e "$link" ] || return 1
        fi
    fi
    [ "$(cat "$gadget/UDC")" = "$initial_udc" ] &&
    [ "$(getprop sys.usb.config)" = "$initial_config" ] &&
    [ "$(getprop sys.usb.state)" = "$initial_state" ] &&
    [ ! -e "$link" ] && [ ! -L "$link" ] &&
    [ "$(readlink -f "$gadget/configs/b.1/f1")" = "$gadget/functions/ffs.ptp" ] &&
    [ "$(readlink -f "$gadget/configs/b.1/f2")" = "$gadget/functions/ffs.adb" ] || return 1
    usb_changed=0
}

direct_acknowledge() {
    [ ! -e "$session/control.lock/restored" ] && [ ! -L "$session/control.lock/restored" ] || return 1
    # Same-filesystem rename makes the complete session-bound receipt visible at once.
    (set -C; printf '%s\n' "$direct_receipt" > "$session/control.lock/restored.tmp") || return 1
    mv "$session/control.lock/restored.tmp" "$session/control.lock/restored"
}

direct_wait_exit() {
    direct_attempt=0
    while direct_service_alive && [ "$direct_attempt" -lt 30 ]; do
        sleep 1
        direct_attempt=$((direct_attempt + 1))
    done
    direct_service_alive && return 1
    wait "$service_pid"
    service_status=$?
    printf 'SERVICE_EXIT_STATUS=%s\n' "$service_status"
    if [ "$direct_outcome" = normal ]; then
        [ "$service_status" = 0 ] && grep -q '^GEMMA_FFS_DRAIN outstanding_dma=0 normal_shutdown=1' "$session/service.log"
    else
        [ "$service_status" = 1 ] && grep -q '^DIRECT_USB_ABORT_DRAINED=1$' "$session/service.log"
    fi
}

direct_shutdown() {
    direct_event stop_begin
    direct_read_receipt || direct_request_stop || return 1
    direct_attempt=0
    while ! direct_read_receipt && [ "$direct_attempt" -lt 60 ]; do
        direct_service_alive || { direct_event service_exited_without_receipt; return 1; }
        sleep 1
        direct_attempt=$((direct_attempt + 1))
    done
    direct_read_receipt || { direct_event drain_unconfirmed; return 1; }
    direct_event dma_quiesced_endpoints_open
    direct_restore || { direct_event restore_failed_resources_retained; return 1; }
    direct_event usb_restored
    direct_acknowledge || { direct_event restore_ack_failed; return 1; }
    direct_event restore_acknowledged
    direct_wait_exit || { direct_event service_exit_unconfirmed; return 1; }
    direct_event service_exited
}
