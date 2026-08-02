#!/usr/bin/env bash
set -euo pipefail

a6000_repo=/home/myid/zs89458/Documents/llama.cpp-release
desktop_host=zhihao@172.20.74.85
desktop_repo=/home/zhihao/llama.cpp-release
s24_rel=research_dev/spikes/s24_overlap_handoff_poc
s24_dir="$a6000_repo/$s24_rel"
provision_dir="$s24_dir/results/a6000_provision"
mid_shard=/home/myid/zs89458/Documents/models/12b-f16-mid-8-16-s24.gguf
expected_source_sha=2120b6a0e2af73a1f02d0f72879f062518b3dbfaf3da1264c3744da63b958a71
op12_serial=5ae7a43d
op15_serial=3C15AU002CL00000
s23_source_dir="$a6000_repo/scratchpad/s8_gate_a/run-a"
s23_jsonl="$s23_source_dir/burstgpt-burst.jsonl"
s23_manifest="$s23_source_dir/burstgpt-burst.manifest.json"

fail() {
    echo "error: $*" >&2
    exit 2
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"
}

require_usb() {
    local serial=$1
    local name=$2
    adb -s "$serial" get-state | grep -qx device || \
        fail "$name is not in ADB device state"
    local device_path
    device_path=$(adb -s "$serial" get-devpath | tr -d '\r')
    [[ "$device_path" == usb:* ]] || \
        fail "$name is not connected through USB: $device_path"
    printf '%s USB transport: %s\n' "$name" "$device_path"
}

for command_name in adb awk bash grep rsync sha256sum ssh; do
    require_command "$command_name"
done

[[ -d "$a6000_repo/.git" ]] || fail "missing A6000 checkout: $a6000_repo"
[[ -f "$a6000_repo/npu-harness/scripts/build_npu_op12.sh" ]] || \
    fail "missing Android build script"
[[ -f "$s23_jsonl" ]] || fail "missing S23 source: $s23_jsonl"
[[ -f "$s23_manifest" ]] || fail "missing S23 source: $s23_manifest"
[[ ! -e "$provision_dir" ]] || \
    fail "provision evidence already exists; do not overwrite: $provision_dir"
if [[ -e "$mid_shard" ]]; then
    sha256sum "$mid_shard" >&2
    fail "middle shard already exists; do not delete or reuse it without review"
fi

cd "$a6000_repo"

ssh "$desktop_host" true
adb devices -l
require_usb "$op12_serial" OP12
require_usb "$op15_serial" OP15

mkdir -p "$s24_dir"
rsync -a --checksum --itemize-changes \
    --exclude '__pycache__/' \
    "$desktop_host:$desktop_repo/$s24_rel/" \
    "$s24_dir/"

rsync -a --checksum --itemize-changes \
    "$desktop_host:$desktop_repo/examples/layersplit/layersplit.cpp" \
    "$a6000_repo/examples/layersplit/layersplit.cpp"

observed_source_sha=$(sha256sum \
    "$a6000_repo/examples/layersplit/layersplit.cpp" | awk '{print $1}')
[[ "$observed_source_sha" == "$expected_source_sha" ]] || \
    fail "layersplit source hash mismatch: $observed_source_sha"

bash "$a6000_repo/npu-harness/scripts/build_npu_op12.sh" --force

S24_CONFIRM_A6000_USB=YES \
    "$s24_dir/a6000_provision_phones.sh"

ssh "$desktop_host" \
    "mkdir -p '$desktop_repo/$s24_rel/results/a6000_provision'"
rsync -a --checksum --itemize-changes \
    "$provision_dir/" \
    "$desktop_host:$desktop_repo/$s24_rel/results/a6000_provision/"

ssh "$desktop_host" \
    "mkdir -p '$desktop_repo/scratchpad/s8_gate_a/run-a'"
rsync -a --checksum --itemize-changes \
    "$s23_jsonl" \
    "$s23_manifest" \
    "$desktop_host:$desktop_repo/scratchpad/s8_gate_a/run-a/"

sha256sum "$s23_jsonl" "$s23_manifest" |
    ssh "$desktop_host" \
        'umask 077; tee /tmp/s23-trace-source-sha256.txt >/dev/null'

printf '\nS24_A6000_ONE_TIME_SETUP_COMPLETE\n'
printf 'Provision evidence: %s\n' "$provision_dir"
printf 'Phone workers were not started.\n'
