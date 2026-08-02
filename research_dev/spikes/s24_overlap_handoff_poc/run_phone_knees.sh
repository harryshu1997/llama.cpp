#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
s24_dir="$repo_root/research_dev/spikes/s24_overlap_handoff_poc"
output_dir=${1:-"$s24_dir/results/cp3_phone_knees"}
op12_endpoint=${S24_OP12_ENDPOINT:-192.168.1.193:24280}
op15_endpoint=${S24_OP15_ENDPOINT:-192.168.1.97:24281}
rounds=${S24_KNEE_ROUNDS:-5}
timeout_s=${S24_PHONE_TIMEOUT_S:-600}

if [[ -e "$output_dir" ]]; then
    echo "error: output already exists: $output_dir" >&2
    exit 2
fi
mkdir -p "$output_dir"

python3 "$s24_dir/batch_knee.py" \
    --worker op12-prefix \
    --endpoint "$op12_endpoint" \
    --candidates 1,2,4 \
    --rounds "$rounds" \
    --timeout "$timeout_s" \
    --session-end detach \
    --output "$output_dir/op12-prefix-knee.json" \
    > "$output_dir/op12-prefix-knee.stdout"

python3 "$s24_dir/batch_knee.py" \
    --worker op15-mid \
    --endpoint "$op15_endpoint" \
    --candidates 1,2,4 \
    --rounds "$rounds" \
    --timeout "$timeout_s" \
    --session-end detach \
    --output "$output_dir/op15-mid-knee.json" \
    > "$output_dir/op15-mid-knee.stdout"

sha256sum \
    "$s24_dir/batch_knee.py" \
    "$output_dir/op12-prefix-knee.json" \
    "$output_dir/op15-mid-knee.json" \
    > "$output_dir/artifact-sha256.txt"

printf '%s\n' "$output_dir"
