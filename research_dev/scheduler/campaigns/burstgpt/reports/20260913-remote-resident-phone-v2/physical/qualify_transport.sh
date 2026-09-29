#!/bin/bash
set -euo pipefail

root=/mnt/storage/s42-remote-resident-phone-20260913-v2-WTUd4V
probe=$root/source/research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/ffs_dmabuf_transport_v1/run_desktop_case.sh
for payload in 7680 10240 3932160; do
    for direction in h2d d2h duplex; do
        request=$payload
        response=$payload
        case "$direction" in
            h2d) response=64 ;;
            d2h) request=64 ;;
        esac
        name=rrphone-20260913-v2b-$payload-$direction
        adb -P 5037 -s 3C15AU002CL00000 shell \
            "test ! -e /data/local/tmp/s41-ffs-dmabuf-v1/$name"
        env S41_STAGE_ROOT="$root" bash "$probe" \
            dmabuf async devmem "$request" "$response" 5 100 1 \
            "$name" "$root/transport-v2b"
    done
done
