#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <campaign.json> <new-output-directory>" >&2
    exit 2
fi

campaign=$1
output=$2
if [[ ! -f $campaign ]]; then
    echo "campaign manifest is absent: $campaign" >&2
    exit 2
fi
if [[ $output != /* || -e $output ]]; then
    echo "output must be a new absolute path: $output" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$here/launch.py" "$campaign" "$output"
