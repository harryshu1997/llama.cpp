#!/usr/bin/env bash

set -euo pipefail

if (( $# != 3 )); then
    echo "usage: $0 TOOL_ROOT BIN_ROOT RESULT_ROOT" >&2
    exit 2
fi

tool_root=$1
bin_root=$2
result_root=$3
phone_root=/data/local/tmp/s42-kernel-energy-v1/ffs

run_case() {
    local label=$1
    local host_mode=$2
    local request_bytes=$3
    local response_bytes=$4
    local iterations=$5
    local depth=$6
    local repetition=$7
    local case_id="usb-dmabuf-${label}-r${repetition}"
    local capture="$result_root/$case_id.capture"
    local transport="$result_root/$case_id.transport"
    local desktop="$result_root/$case_id.desktop.json"
    local phone="$result_root/$case_id.phone.json"

    if [[ -f "$desktop" && -f "$phone" ]]; then
        echo "SKIP $case_id"
        return
    fi
    for path in "$capture" "$transport" "$desktop" "$phone"; do
        if [[ -e "$path" ]]; then
            mv "$path" "$path.invalid-interrupted.$BASHPID"
        fi
    done

    python3 "$tool_root/run_phone_case.py" \
        --case-id "$case_id" \
        --mode external \
        --external-active "$phone_root/$case_id/active" \
        --output "$capture" \
        --timeout-s 180 \
        -- \
        python3 "$tool_root/measure_desktop.py" \
        --case-id "$case_id" \
        --output "$desktop" \
        --window json-monotonic \
        --window-json "$transport/$case_id.json" \
        --env "S41_STAGE_ROOT=$bin_root" \
        --env "S41_PHONE_ROOT=$phone_root" \
        -- \
        "$bin_root/run_desktop_case.sh" \
        dmabuf "$host_mode" devmem \
        "$request_bytes" "$response_bytes" 50 "$iterations" "$depth" \
        "$case_id" "$transport"
    python3 "$tool_root/analyze_phone_case.py" \
        --capture "$capture" \
        --marker-log "$transport/$case_id.phone.log" \
        --output "$phone"
    sleep 3
}

for repetition in 1 2 3; do
    run_case duplex8k-sync sync 8192 8192 60000 1 "$repetition"
    run_case h2p1m async 1048576 64 5000 4 "$repetition"
    run_case p2h1m async 64 1048576 5000 4 "$repetition"
    run_case duplex1m async 1048576 1048576 5000 4 "$repetition"
    run_case h2p4m async 4194304 64 1400 3 "$repetition"
    run_case p2h4m async 64 4194304 1400 3 "$repetition"
done
