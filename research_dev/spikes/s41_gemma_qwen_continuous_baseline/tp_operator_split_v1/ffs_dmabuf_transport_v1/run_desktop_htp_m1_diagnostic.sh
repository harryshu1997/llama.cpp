#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <output-root>" >&2
    exit 2
fi

output_root=$1
script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
runner=${S41_HTP_CASE_RUNNER:-$script_root/run_desktop_htp_case.sh}

run_case() {
    mode=$1
    variant=$2
    repetition=$3
    S41_HTP_DEVICE_MODE=$mode "$runner" malloc 2560 50 500 \
        "hidden_m1_diag.$variant.r$repetition" "$output_root"
}

mkdir -p "$output_root"
for repetition in 1 2 3 4 5 6; do
    if [ $((repetition % 2)) -eq 1 ]; then
        run_case htp-copy-sqr copy_malloc "$repetition"
        run_case htp-sqr dmabuf_malloc "$repetition"
    else
        run_case htp-sqr dmabuf_malloc "$repetition"
        run_case htp-copy-sqr copy_malloc "$repetition"
    fi
done
