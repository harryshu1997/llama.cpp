#!/usr/bin/env bash
set -euo pipefail

SOURCE_SNAPSHOT=${SOURCE_SNAPSHOT:-/mnt/data_s4t/zs89458-moe-routing/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754}
REMOTE_HOST=${REMOTE_HOST:-zhihao@172.20.74.85}
REMOTE_SNAPSHOT=${REMOTE_SNAPSHOT:-/mnt/storage/moe-resident-routing-4060ti-op15-cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754}
# Both endpoints are rotational storage and the 4060 Ti host is connected over
# an ~87 Mbit/s Wi-Fi link.  A single sequential stream saturates that link;
# concurrent shard reads only add HDD seek contention.  Faster storage/network
# setups can still opt in with PARALLEL_TRANSFERS=N.
PARALLEL_TRANSFERS=${PARALLEL_TRANSFERS:-1}
LOG_ROOT=${LOG_ROOT:-results/qwen35_35b_a3b/op15_4060ti_real/transfer_logs}

mkdir -p "$LOG_ROOT"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" "mkdir -p '$REMOTE_SNAPSHOT'"
export SOURCE_SNAPSHOT REMOTE_HOST REMOTE_SNAPSHOT LOG_ROOT

find -L "$SOURCE_SNAPSHOT" -maxdepth 1 -type f -printf '%f\0' | sort -z | \
  xargs -0 -P "$PARALLEL_TRANSFERS" -I '{}' bash -c '
    set -euo pipefail
    name=$1
    rsync -aL --partial --append-verify \
      -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
      "$SOURCE_SNAPSHOT/$name" "$REMOTE_HOST:$REMOTE_SNAPSHOT/$name" \
      >"$LOG_ROOT/$name.log" 2>&1
    printf "completed %s\n" "$name"
  ' _ '{}'

echo "All snapshot files transferred."
