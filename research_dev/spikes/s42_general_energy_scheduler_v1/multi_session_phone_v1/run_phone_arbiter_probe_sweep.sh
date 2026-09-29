#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <absolute-new-output> <adb-port> <bridge> <probe>" >&2
    exit 2
fi

output=$1
adb_port=$2
bridge=$3
probe=$4
if [[ ${S42_ENABLE_PHONE_ARBITER_PROBE:-} != YES ]]; then
    echo "physical phone-arbiter probes are disabled" >&2
    exit 2
fi
if [[ $output != /* || -e $output \
        || ! $adb_port =~ ^[0-9]+$ || $adb_port -eq 0 \
        || $bridge != /* || ! -x $bridge \
        || $probe != /* || ! -x $probe ]]; then
    echo "invalid phone-arbiter probe arguments" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
serial=${S42_PHONE_SERIAL:-3C15AU002CL00000}
phone_base=${S42_PHONE_BASE:-/data/local/tmp/s41-opoffload-dmabuf-v1}
phone_workers=${S42_PHONE_WORKERS:-$phone_base/resident_ffn_workers-triple-v3}
phone_router=${S42_PHONE_ROUTER:-$phone_base/resident_ffn_router-terminal-v1}
phone_session=${S42_PHONE_SESSION:-$phone_base/resident_ffn_session-triple-v1.sh}
phone_gemma=${S42_PHONE_GEMMA_MODEL:-$phone_base/gemma-4-12B-Q40-dequant-f16.gguf}
phone_qwen=${S42_PHONE_QWEN_MODEL:-$phone_base/Qwen3-14B-Q4KM-dequant-f16.gguf}
phone_restore=${S42_PHONE_RESTORE:-$phone_base/restore_phone_usb.sh}
phone_policy=${S42_PHONE_POLICY:-$phone_base/phone_power_allow.rules}
phone_ffn_vmem=${S42_PHONE_PROBE_FFN_VMEM:-3264}
phone_minimum_available_kib=${S42_PHONE_PROBE_MIN_AVAILABLE_KIB:-2097152}
token_shapes=${S42_PHONE_PROBE_TOKENS:-"1 2 4 8 16"}
protected_port=${S42_PHONE_PROBE_PROTECTED_PORT:-17980}
filler_port=${S42_PHONE_PROBE_FILLER_PORT:-17981}
gap_us=${S42_PHONE_PROBE_GAP_US:-350000}
idle_lower_us=${S42_PHONE_PROBE_IDLE_LOWER_US:-330000}
filler_upper_us=${S42_PHONE_PROBE_FILLER_UPPER_US:-250000}
guard_us=${S42_PHONE_PROBE_GUARD_US:-20000}

if [[ ! $serial =~ ^[A-Za-z0-9]+$ \
        || ! $protected_port =~ ^[0-9]+$ \
        || ! $filler_port =~ ^[0-9]+$ \
        || $protected_port -eq 0 || $protected_port -gt 65535 \
        || $filler_port -eq 0 || $filler_port -gt 65535 \
        || $protected_port -eq $filler_port \
        || ! $gap_us =~ ^[0-9]+$ || $gap_us -eq 0 \
        || ! $idle_lower_us =~ ^[0-9]+$ || $idle_lower_us -eq 0 \
        || ! $filler_upper_us =~ ^[0-9]+$ || $filler_upper_us -eq 0 \
        || ! $guard_us =~ ^[0-9]+$ \
        || ! $phone_ffn_vmem =~ ^[0-9]+$ \
        || $phone_ffn_vmem -lt 3200 || $phone_ffn_vmem -gt 3328 \
        || ! $phone_minimum_available_kib =~ ^[0-9]+$ \
        || $phone_minimum_available_kib -lt 2097152 \
        || $((filler_upper_us + guard_us)) -gt $idle_lower_us ]]; then
    echo "invalid phone-arbiter probe configuration" >&2
    exit 2
fi
declare -A seen_tokens=()
for tokens in $token_shapes; do
    if [[ ! $tokens =~ ^[0-9]+$ || $tokens -eq 0 || $tokens -gt 16 ]]; then
        echo "invalid phone-arbiter token shape: $tokens" >&2
        exit 2
    fi
    if [[ -n ${seen_tokens[$tokens]:-} ]]; then
        echo "duplicate phone-arbiter token shape: $tokens" >&2
        exit 2
    fi
    seen_tokens[$tokens]=1
done
if [[ $(wc -w <<<"$token_shapes") -eq 0 ]]; then
    echo "phone-arbiter token sweep is empty" >&2
    exit 2
fi
read -r -a token_values <<<"$token_shapes"
tokens_csv=$(IFS=,; echo "${token_values[*]}")
if pgrep -f '^[^ ]*/phone_arbiter_bridge[^ ]*( |$)' >/dev/null \
        || pgrep -f '^[^ ]*/phone_arbiter_probe[^ ]*( |$)' >/dev/null; then
    echo "another phone-arbiter process is active" >&2
    exit 1
fi

mkdir "$output"
adb_cmd=(adb -P "$adb_port" -s "$serial")
bridge_pid=
cleanup() {
    if [[ -n $bridge_pid ]] && kill -0 "$bridge_pid" 2>/dev/null; then
        kill -TERM "$bridge_pid" 2>/dev/null || true
        wait "$bridge_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

phone_root=$phone_base/s42-arbiter-probe-sweep-v1-$$
"${adb_cmd[@]}" get-state >/dev/null
"${adb_cmd[@]}" shell \
    "su -c 'test ! -e $phone_root && mkdir $phone_root'"
"${adb_cmd[@]}" shell \
    "su -c '/product/bin/magiskpolicy --live --apply $phone_policy'"
"${adb_cmd[@]}" shell \
    "su -c 'S42_QWEN_ONLY=0 S42_FFN_VMEM=$phone_ffn_vmem S42_MIN_AVAILABLE_KIB=$phone_minimum_available_kib nohup sh $phone_session $phone_workers $phone_router $phone_gemma $phone_qwen $phone_root $phone_restore 900 0 > $phone_root/session-launch.log 2>&1 < /dev/null &'"

aoa_ready=0
phone_start_failed=0
for _ in $(seq 1 300); do
    if lsusb -d 18d1:2d00 >/dev/null 2>&1; then
        aoa_ready=1
        break
    fi
    if "${adb_cmd[@]}" get-state >/dev/null 2>&1 \
            && "${adb_cmd[@]}" shell \
                "su -c 'grep -Eq \"memory reserve failed|did not become warm|router did not publish|watchdog expired\" $phone_root/session-launch.log'" \
                >/dev/null 2>&1; then
        phone_start_failed=1
        break
    fi
    sleep 1
done
if [[ $aoa_ready -ne 1 ]]; then
    if [[ $phone_start_failed -eq 1 ]]; then
        "${adb_cmd[@]}" exec-out \
            "su -c 'cat $phone_root/session-launch.log'" \
            >"$output/session-failure.log" || true
    fi
    echo "phone session did not bind for the shape sweep" >&2
    exit 1
fi

"$bridge" 127.0.0.1 "$protected_port" "$filler_port" \
    malloc-split 0 11 "$idle_lower_us" "$filler_upper_us" "$guard_us" \
    >"$output/bridge.stdout" 2>"$output/bridge.stderr" &
bridge_pid=$!
bridge_ready=0
for _ in $(seq 1 300); do
    if grep -q 'ready protected=' "$output/bridge.stderr" 2>/dev/null; then
        bridge_ready=1
        break
    fi
    if ! kill -0 "$bridge_pid" 2>/dev/null; then
        break
    fi
    sleep 0.1
done
if [[ $bridge_ready -ne 1 ]]; then
    echo "phone-arbiter bridge did not become ready" >&2
    exit 1
fi

"$probe" 127.0.0.1 "$protected_port" "$filler_port" \
    "$gap_us" "$tokens_csv" \
    >"$output/probe.stdout" 2>"$output/probe.stderr"
wait "$bridge_pid"
bridge_pid=

adb_ready=0
for _ in $(seq 1 1200); do
    if "${adb_cmd[@]}" get-state >/dev/null 2>&1; then
        adb_ready=1
        break
    fi
    sleep 0.1
done
if [[ $adb_ready -ne 1 ]]; then
    echo "phone did not restore ADB after the shape sweep" >&2
    exit 1
fi
mkdir "$output/phone"
for log_name in \
        resident-workers.log router.log session-launch.log session.log; do
    "${adb_cmd[@]}" exec-out \
        "su -c 'cat $phone_root/$log_name'" \
        >"$output/phone/$log_name"
done

python3 "$here/analyze_phone_arbiter_probe.py" \
    --probe-log "$output/probe.stdout" \
    --bridge-log "$output/bridge.stderr" \
    --router-log "$output/phone/router.log" \
    --session-log "$output/phone/session.log" \
    --workers-log "$output/phone/resident-workers.log" \
    --expected-tokens "$tokens_csv" \
    --expected-gap-us "$gap_us" \
    --expected-idle-lower-us "$idle_lower_us" \
    --expected-filler-upper-us "$filler_upper_us" \
    --expected-guard-us "$guard_us" \
    --expected-vmem-mib "$phone_ffn_vmem" \
    --minimum-available-kib "$phone_minimum_available_kib" \
    --output "$output/SWEEP.json"
find "$output" -type f ! -name SHA256SUMS.txt -print0 \
    | sort -z | xargs -0 sha256sum \
    >"$output/SHA256SUMS.txt"
trap - EXIT INT TERM
cat "$output/SWEEP.json"
