#!/usr/bin/env bash
# Start/stop the FFN workers on the phones in TCP mode, bound to their WLAN side, plus the echo servers
# for ../wifi_rtt.py. Commands are generated from the JSON config by benchlib.py (one adb command per
# worker) so the server helpers (server_bench.py) and the phone ports can never drift apart.
#
#   ./phone_workers.sh [--config wifi_config.json] [--dry-run] [--local] ACTION
#
# ACTION      start | stop | status   FFN workers (one per worker entry; the OP15 runs one worker per
#                                     HTP session: HTP0 layers 0-5, HTP1 6-11, HTP2 12-17; the Pixel one
#                                     CPU worker with its packed Q4 shard for layers 18-23)
#             echo-start | echo-stop  toybox echo server on phone.echo_port (7070) for wifi_rtt.py
#             ip                      print wlan0's IPv4 (put it into phones[].wlan_ip)
#             wifi-tune               stay-awake + Android low-latency/hi-perf WiFi (+ iw power_save off)
#             hash                    sha256sum of the worker binary and shards on the phone
# --dry-run   print the exact adb commands, run nothing
# --local     run stand-in workers of THIS host's build (build-cuda-s43, CPU backend, 127.0.0.1, same
#             ports, full server model) to exercise the phone arms end-to-end without phones; use with
#             server_bench.py --host-override 127.0.0.1. Mechanics only: the stand-ins share the
#             server's memory bandwidth, their numbers mean nothing.
#
# adb target: phones[].adb is the argv prefix, e.g. ["adb","-P","5037","-s","<usb serial>"] from the
# desktop (phones stay USB-attached there for power), or ["adb","-s","192.168.77.51:5555"] from this
# server after `adb -s <serial> tcpip 5555` on the desktop and `adb connect 192.168.77.51:5555` here.
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
config="$here/wifi_config.json"
dry_run=0
local_mode=()
action=""
while [ $# -gt 0 ]; do
    case "$1" in
        --config) config=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        --local) local_mode=(--local); shift ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        start|stop|status|echo-start|echo-stop|ip|wifi-tune|hash) action=$1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$action" ] || { echo "usage: $0 [--config F] [--dry-run] [--local] ACTION" >&2; exit 2; }
[ -f "$config" ] || { echo "config $config not found (copy wifi_config.example.json)" >&2; exit 2; }

mapfile -t commands < <(python3 "$here/benchlib.py" worker-commands --config "$config" --action "$action" \
    "${local_mode[@]}" --local-dir "$here/local-workers")
status=0
for command in "${commands[@]}"; do
    if [ "$dry_run" -eq 1 ]; then
        printf '%s\n' "$command"
        continue
    fi
    printf '+ %s\n' "$command" | cut -c1-240
    bash -c "$command" || status=$?
done
exit "$status"
