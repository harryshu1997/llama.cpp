#!/system/bin/sh
# Isolated, loopback-only TCP service; does not change USB, SELinux or firmware.
set -eu
session=/data/local/tmp/gemma4_energy_tcp_20260910
bank=/data/local/tmp/gemma4_b1_20260910/bank.g4b
service_pid=
cleanup() {
    controller_status=$?
    trap - EXIT INT TERM HUP
    set +e
    if [ -n "$service_pid" ]; then
        kill -TERM "$service_pid" 2>/dev/null
        attempt=0
        while kill -0 "$service_pid" 2>/dev/null && [ "$attempt" -lt 60 ]; do
            sleep 1
            attempt=$((attempt + 1))
        done
        if kill -0 "$service_pid" 2>/dev/null; then
            printf 'DRAIN_FAILED_SERVICE_STILL_RUNNING=1\n'
            exit 2
        fi
        wait "$service_pid"
        service_status=$?
        printf 'SERVICE_EXIT_STATUS=%s\n' "$service_status"
        if [ "$service_status" != 0 ] || ! grep -q 'GEMMA_TCP_DRAIN active_clients=0' "$session/service.log"; then
            controller_status=2
        fi
    fi
    rmdir "$session/control.lock"
    printf 'SESSION_EXIT_STATUS=%s\n' "$controller_status"
    exit "$controller_status"
}
[ "$(id -u)" = 0 ]
[ "$(getprop ro.serialno)" = 3C15AU002CL00000 ]
[ "$(getprop ro.product.model)" = CPH2749 ]
[ "$(getprop ro.soc.model)" = SM8850 ]
[ "$(getenforce)" = Enforcing ]
[ ! -e "$session/stop" ]
[ -x "$session/gemma4_htp_service" ]
[ "$(stat -c %s "$bank")" = 5709496480 ]
mkdir "$session/control.lock"
trap cleanup EXIT
trap 'exit 143' INT TERM HUP
printf '%s\n' "$$" > "$session/controller.pid"
env LD_LIBRARY_PATH="$session" ADSP_LIBRARY_PATH="$session" \
    GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPPOLL=0 \
    GGML_HEXAGON_NHVX=0 GGML_HEXAGON_NHMX=1 GGML_HEXAGON_PROFILE=0 \
    "$session/gemma4_htp_service" --weights "$bank" \
    --layers 30 --experts 16 --warmups 0 --port 28165 \
    --q8-weights --group-experts --hmx-pad-rows 8 --transport tcp \
    > "$session/service.log" 2>&1 &
service_pid=$!
printf '%s\n' "$service_pid" > "$session/service.pid"
elapsed=0
while [ ! -e "$session/stop" ] && [ "$elapsed" -lt 3600 ]; do
    kill -0 "$service_pid"
    sleep 1
    elapsed=$((elapsed + 1))
done
printf 'STOP_REQUESTED_OR_WATCHDOG_EXPIRED=1\n'
