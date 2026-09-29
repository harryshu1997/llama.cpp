#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <result-dir>" >&2
    exit 2
fi

result_dir=$1
script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
case_runner=$script_root/run_case.sh

mkdir -p "$result_dir"

run_one() {
    repetition=$1
    workload=$2
    request_bytes=$3
    response_bytes=$4
    warmup=$5
    iterations=$6
    configuration=$7

    case "$configuration" in
        serial_sync)
            phone_mode=serial
            host_mode=sync
            queue_depth=1
            ;;
        buffered_sync)
            phone_mode=buffered
            host_mode=sync
            queue_depth=1
            ;;
        buffered_async_q1)
            phone_mode=buffered
            host_mode=async
            queue_depth=1
            ;;
        buffered_async_q2)
            phone_mode=buffered
            host_mode=async
            queue_depth=2
            ;;
        buffered_async_q4)
            phone_mode=buffered
            host_mode=async
            queue_depth=4
            ;;
        *)
            echo "unknown configuration: $configuration" >&2
            exit 2
            ;;
    esac

    case_name=${workload}_r${repetition}_${configuration}
    if [ -e "$result_dir/$case_name.json" ] ||
            [ -e "$result_dir/$case_name.worker.log" ]; then
        echo "refusing to overwrite case: $case_name" >&2
        exit 3
    fi
    echo "running $case_name"
    S41_CASE_NAME=$case_name S41_WORKLOAD=$workload \
    S41_REPETITION=$repetition S41_FRESH_ACCESSORY=1 \
    S41_KEEP_ACCESSORY=0 \
        "$case_runner" "$phone_mode" "$host_mode" "$queue_depth" \
        "$request_bytes" "$response_bytes" "$warmup" "$iterations" \
        "$result_dir"
}

run_workload() {
    repetition=$1
    workload=$2
    request_bytes=$3
    response_bytes=$4
    warmup=$5
    iterations=$6

    case "$repetition" in
        1)
            order="serial_sync buffered_sync buffered_async_q1 buffered_async_q2 buffered_async_q4"
            ;;
        2)
            order="buffered_async_q4 buffered_async_q2 buffered_async_q1 buffered_sync serial_sync"
            ;;
        3)
            order="buffered_sync buffered_async_q4 serial_sync buffered_async_q2 buffered_async_q1"
            ;;
        *)
            echo "invalid repetition: $repetition" >&2
            exit 2
            ;;
    esac
    for configuration in $order; do
        run_one "$repetition" "$workload" "$request_bytes" \
            "$response_bytes" "$warmup" "$iterations" "$configuration"
    done
}

run_repetition() {
    repetition=$1
    case "$repetition" in
        1)
            workloads="attention hidden_m1 swiglu hidden_m8 host_to_phone_1m phone_to_host_1m"
            ;;
        2)
            workloads="phone_to_host_1m host_to_phone_1m hidden_m8 swiglu hidden_m1 attention"
            ;;
        3)
            workloads="swiglu attention phone_to_host_1m hidden_m1 host_to_phone_1m hidden_m8"
            ;;
        *)
            echo "invalid repetition: $repetition" >&2
            exit 2
            ;;
    esac

    for workload in $workloads; do
        case "$workload" in
            attention)
                run_workload "$repetition" "$workload" 1308 1384 50 300
                ;;
            hidden_m1)
                run_workload "$repetition" "$workload" 10268 10344 50 300
                ;;
            swiglu)
                run_workload "$repetition" "$workload" 69660 34920 50 300
                ;;
            hidden_m8)
                run_workload "$repetition" "$workload" 81948 82024 50 300
                ;;
            host_to_phone_1m)
                run_workload "$repetition" "$workload" 1048604 104 20 100
                ;;
            phone_to_host_1m)
                run_workload "$repetition" "$workload" 64 1048680 20 100
                ;;
            *)
                echo "unknown workload: $workload" >&2
                exit 2
                ;;
        esac
    done
}

run_repetition 1
run_repetition 2
run_repetition 3
