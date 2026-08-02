#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
s24_dir="$repo_root/research_dev/spikes/s24_overlap_handoff_poc"
expected_repo=${S24_A6000_REPO:-/home/myid/zs89458/Documents/llama.cpp-release}
output_dir=${1:-"$s24_dir/results/a6000_provision"}
full_model=${S24_F16_MODEL:-/home/myid/zs89458/Documents/models/gemma-4-12B-it-f16.gguf}
mid_shard=${S24_MID_SHARD:-/home/myid/zs89458/Documents/models/12b-f16-mid-8-16-s24.gguf}
android_build=${S24_ANDROID_BUILD:-"$repo_root/npu-harness/build/llamacpp/android-arm64-hexagon-release-eafdc75e"}
android_dir=${S24_ANDROID_DIR:-"$android_build/bin"}
hexagon_dir=${S24_HEXAGON_DIR:-"$android_build/ggml/src/ggml-hexagon"}
python_bin=${S24_PYTHON:-python3}
op12_serial=${S24_OP12_SERIAL:-5ae7a43d}
op15_serial=${S24_OP15_SERIAL:-3C15AU002CL00000}
remote_dir=${S24_PHONE_DIR:-/data/local/tmp/ls-s24}
op12_head=${S24_OP12_HEAD:-/data/local/tmp/ls-npu/12b-f16-head-0-8.gguf}
op15_mid=${S24_OP15_MID:-/data/local/tmp/ls-s24/12b-f16-mid-8-16.gguf}

full_model_sha=bac4e29337273d1cb41aa36f4209ebcf63baa1162e6ab4dc55f6034b3d69b35a
source_sha=2120b6a0e2af73a1f02d0f72879f062518b3dbfaf3da1264c3744da63b958a71
op12_head_sha=a74845294bc1a3cba6ac26c747615d64d9127c904aae04c43c7ea2ebc1ea25c8

fail() {
    echo "error: $*" >&2
    exit 2
}

if [[ ${S24_CONFIRM_A6000_USB:-NO} != YES ]]; then
    fail "set S24_CONFIRM_A6000_USB=YES on the A6000 provisioning host"
fi
if [[ "$repo_root" != "$expected_repo" ]]; then
    fail "this script must run from the authoritative A6000 checkout: $expected_repo"
fi
if [[ -e "$output_dir" ]]; then
    fail "output already exists: $output_dir"
fi

for path in \
    "$full_model" \
    "$repo_root/examples/layersplit/layersplit.cpp" \
    "$repo_root/research_dev/shard_gguf.py" \
    "$android_dir/llama-layersplit"; do
    [[ -f "$path" ]] || fail "missing required artifact: $path"
done

observed_full_sha=$(sha256sum "$full_model" | awk '{print $1}')
[[ "$observed_full_sha" == "$full_model_sha" ]] || fail "full F16 model hash mismatch"
observed_source_sha=$(sha256sum "$repo_root/examples/layersplit/layersplit.cpp" | awk '{print $1}')
[[ "$observed_source_sha" == "$source_sha" ]] || fail "A6000 layersplit source is not the S24 V3 source"

runtime_libs=()
append_runtime_artifact() {
    local name=$1
    shift
    local candidate
    for candidate in "$@"; do
        if [[ -f "$candidate" ]]; then
            runtime_libs+=("$candidate")
            return
        fi
    done
    fail "missing Android runtime: $name"
}

for required in \
    libggml-base.so \
    libggml-cpu.so \
    libggml-hexagon.so \
    libggml-opencl.so \
    libggml.so \
    libllama-common.so \
    libllama.so; do
    append_runtime_artifact "$required" "$android_dir/$required"
done
append_runtime_artifact libc++_shared.so \
    "$android_dir/libc++_shared.so" \
    "$hexagon_dir/ship/libc++_shared.so"
append_runtime_artifact libggml-htp-v75.so \
    "$android_dir/libggml-htp-v75.so" \
    "$hexagon_dir/libggml-htp-v75.so"
append_runtime_artifact libggml-htp-v81.so \
    "$android_dir/libggml-htp-v81.so" \
    "$hexagon_dir/libggml-htp-v81.so"

declare -A runtime_names=()
for library in "${runtime_libs[@]}"; do
    runtime_name=$(basename "$library")
    [[ -z ${runtime_names[$runtime_name]+x} ]] || \
        fail "duplicate Android runtime basename: $runtime_name"
    runtime_names[$runtime_name]=1
done

mkdir -p "$output_dir"

if command -v readelf >/dev/null 2>&1; then
    readelf -h "$android_dir/llama-layersplit" > "$output_dir/android-elf-header.txt"
    grep -q 'AArch64' "$output_dir/android-elf-header.txt" || fail "phone binary is not AArch64"
fi
if command -v strings >/dev/null 2>&1; then
    strings "$android_dir/llama-layersplit" > "$output_dir/android-binary.strings"
    grep -q 'ls-stagenet-session-v2' "$output_dir/android-binary.strings" || \
        fail "phone binary lacks the required StageNet lifecycle certificate"
fi

mid_creation=created
if [[ -e "$mid_shard" ]]; then
    [[ ${S24_REUSE_MID:-NO} == YES ]] || \
        fail "middle shard exists; set S24_REUSE_MID=YES only after reviewing it"
    [[ -f "$mid_shard" ]] || fail "middle shard path is not a regular file"
    mid_creation=reused_after_hash
else
    mid_parent=$(dirname "$mid_shard")
    mkdir -p "$mid_parent"
    mid_tmp=$(mktemp "$mid_parent/.s24-mid-8-16.partial.XXXXXX")
    cleanup_mid() {
        if [[ -n ${mid_tmp:-} && -e "$mid_tmp" ]]; then
            rm -f "$mid_tmp"
        fi
    }
    trap cleanup_mid EXIT INT TERM
    printf '%q ' \
        "$python_bin" "$repo_root/research_dev/shard_gguf.py" \
        "$full_model" "$mid_shard" --start 8 --end 16 \
        > "$output_dir/shard-command.txt"
    printf '\n' >> "$output_dir/shard-command.txt"
    "$python_bin" "$repo_root/research_dev/shard_gguf.py" \
        "$full_model" "$mid_tmp" --start 8 --end 16 \
        > "$output_dir/shard.stdout" 2> "$output_dir/shard.stderr"
    mv "$mid_tmp" "$mid_shard"
    mid_tmp=
    trap - EXIT INT TERM
fi

mid_sha=$(sha256sum "$mid_shard" | awk '{print $1}')
mid_bytes=$(stat -c '%s' "$mid_shard")

adb devices -l > "$output_dir/adb-devices.txt"
require_usb_device() {
    local serial=$1
    local name=$2
    local row
    row=$(awk -v serial="$serial" '$1 == serial {print}' "$output_dir/adb-devices.txt")
    [[ -n "$row" ]] || fail "$name is absent from adb devices"
    [[ "$row" == *" device "* ]] || fail "$name is not in adb device state"
    adb -s "$serial" get-state | grep -qx device || fail "$name get-state failed"
    local device_path
    device_path=$(adb -s "$serial" get-devpath | tr -d '\r')
    [[ "$device_path" == usb:* ]] || fail "$name ADB transport is not certified USB"
    printf '%s  %s\n' "$name" "$device_path" >> "$output_dir/adb-usb-devpaths.txt"
}
require_usb_device "$op12_serial" OP12
require_usb_device "$op15_serial" OP15

remote_sha256() {
    local serial=$1
    local path=$2
    adb -s "$serial" shell sha256sum "$path" | tr -d '\r' | awk '{print $1}'
}

observed_head_sha=$(remote_sha256 "$op12_serial" "$op12_head")
[[ "$observed_head_sha" == "$op12_head_sha" ]] || \
    fail "OP12 existing [0,8) F16 shard hash mismatch"

deploy_runtime() {
    local serial=$1
    local name=$2
    adb -s "$serial" shell mkdir -p "$remote_dir"
    adb -s "$serial" push \
        "$android_dir/llama-layersplit" "$remote_dir/llama-layersplit" \
        >> "$output_dir/adb-push.log" 2>&1
    for library in "${runtime_libs[@]}"; do
        adb -s "$serial" push "$library" "$remote_dir/$(basename "$library")" \
            >> "$output_dir/adb-push.log" 2>&1
    done
    adb -s "$serial" shell chmod 755 "$remote_dir/llama-layersplit"
    local local_sha
    local remote_sha
    local_sha=$(sha256sum "$android_dir/llama-layersplit" | awk '{print $1}')
    remote_sha=$(remote_sha256 "$serial" "$remote_dir/llama-layersplit")
    [[ "$local_sha" == "$remote_sha" ]] || fail "$name binary USB transfer hash mismatch"
    for library in "${runtime_libs[@]}"; do
        local_sha=$(sha256sum "$library" | awk '{print $1}')
        remote_sha=$(remote_sha256 "$serial" "$remote_dir/$(basename "$library")")
        [[ "$local_sha" == "$remote_sha" ]] || \
            fail "$name runtime USB transfer hash mismatch: $(basename "$library")"
    done
}

deploy_runtime "$op12_serial" OP12
deploy_runtime "$op15_serial" OP15

adb -s "$op15_serial" push "$mid_shard" "$op15_mid" \
    >> "$output_dir/adb-push.log" 2>&1
adb -s "$op15_serial" shell sync
remote_mid_sha=$(remote_sha256 "$op15_serial" "$op15_mid")
[[ "$remote_mid_sha" == "$mid_sha" ]] || fail "OP15 middle-shard USB transfer hash mismatch"

{
    sha256sum \
        "$full_model" \
        "$mid_shard" \
        "$repo_root/examples/layersplit/layersplit.cpp" \
        "$repo_root/research_dev/shard_gguf.py" \
        "$android_dir/llama-layersplit"
    for library in "${runtime_libs[@]}"; do
        sha256sum "$library"
    done
} > "$output_dir/host-artifacts.sha256"

{
    printf 'OP12  %s  %s\n' "$observed_head_sha" "$op12_head"
    printf 'OP15  %s  %s\n' "$remote_mid_sha" "$op15_mid"
    for serial in "$op12_serial" "$op15_serial"; do
        printf '%s  %s  %s\n' \
            "$serial" \
            "$(remote_sha256 "$serial" "$remote_dir/llama-layersplit")" \
            "$remote_dir/llama-layersplit"
        for library in "${runtime_libs[@]}"; do
            printf '%s  %s  %s\n' \
                "$serial" \
                "$(remote_sha256 "$serial" "$remote_dir/$(basename "$library")")" \
                "$remote_dir/$(basename "$library")"
        done
    done
} > "$output_dir/device-artifacts.sha256"

{
    printf 'schema=s24-a6000-usb-provision-v1\n'
    printf 'a6000_repo=%s\n' "$repo_root"
    printf 'full_model=%s\n' "$full_model"
    printf 'full_model_sha256=%s\n' "$observed_full_sha"
    printf 'source_sha256=%s\n' "$observed_source_sha"
    printf 'mid_creation=%s\n' "$mid_creation"
    printf 'mid_shard=%s\n' "$mid_shard"
    printf 'mid_bytes=%s\n' "$mid_bytes"
    printf 'mid_sha256=%s\n' "$mid_sha"
    printf 'op12_head_sha256=%s\n' "$observed_head_sha"
    printf 'op15_mid_remote_sha256=%s\n' "$remote_mid_sha"
    printf 'weight_transport=ADB_USB_SERIAL_TRANSPORT\n'
    printf 'wifi_weight_bytes=0\n'
    printf 'a6000_gpu_use=FORBIDDEN_AND_NOT_INVOKED\n'
} > "$output_dir/provision.env"

(
    cd "$output_dir"
    find . -type f ! -name SHA256SUMS.txt -print0 |
        LC_ALL=C sort -z |
        xargs -0 sha256sum > SHA256SUMS.txt
)

printf '%s\n' "$output_dir"
