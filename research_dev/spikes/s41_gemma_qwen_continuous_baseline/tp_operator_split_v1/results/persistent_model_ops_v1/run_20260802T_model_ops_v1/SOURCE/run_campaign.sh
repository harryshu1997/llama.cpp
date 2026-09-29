#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <result-root>" >&2
    exit 2
fi

result_root=$1
script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
n_kv=${S41_MODEL_OP_N_KV:-8192}
mkdir -p "$result_root"

run_case() {
    repetition=$1
    backend=$2
    op=$3
    output=$result_root/${op}_rep${repetition}
    case_name=$(printf '%s_%s_kv%s' "$backend" "$op" "$n_kv" |
        tr '[:upper:]' '[:lower:]')
    if [ -f "$output/$case_name.json" ]; then
        return
    fi
    "$script_root/run_case.sh" "$backend" "$op" "$n_kv" "$output"
}

for op in rmsnorm swiglu attention; do
    run_case 1 HTPGraph "$op"
    run_case 1 OpenCLGraph "$op"
    run_case 1 OpenCLDispatch "$op"
    run_case 1 OpenCLPersistent "$op"

    run_case 2 OpenCLPersistent "$op"
    run_case 2 OpenCLDispatch "$op"
    run_case 2 OpenCLGraph "$op"
    run_case 2 HTPGraph "$op"

    run_case 3 OpenCLGraph "$op"
    run_case 3 HTPGraph "$op"
    run_case 3 OpenCLPersistent "$op"
    run_case 3 OpenCLDispatch "$op"
done
