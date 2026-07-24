#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
ADB_SERIAL=${ADB_SERIAL:-3C15AU002CL00000}
REMOTE_DIR=${REMOTE_DIR:-/data/local/tmp/wavefront-op15}
REMOTE_MODEL=${REMOTE_MODEL:-/data/local/tmp/hyzheng/elastic/qwen2.5-0.5b-instruct-q8_0.gguf}
LOCAL_BINARY=${LOCAL_BINARY:-${REPO_DIR}/build-wavefront-android/bin/llama-hetero-ubatch}
REPEATS=${REPEATS:-5}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT_DIR=${OUT_DIR:-${SCRIPT_DIR}/raw/${RUN_ID}}

mkdir -p "${OUT_DIR}/runs"

if [[ ! -x "${LOCAL_BINARY}" ]]; then
    echo "missing binary: ${LOCAL_BINARY}" >&2
    exit 2
fi

adb -s "${ADB_SERIAL}" get-state >/dev/null
adb -s "${ADB_SERIAL}" push "${LOCAL_BINARY}" "${REMOTE_DIR}/llama-hetero-ubatch" >/dev/null
adb -s "${ADB_SERIAL}" shell chmod 755 "${REMOTE_DIR}/llama-hetero-ubatch"

git -C "${REPO_DIR}" rev-parse HEAD > "${OUT_DIR}/git-head.txt"
git -C "${REPO_DIR}" status --short > "${OUT_DIR}/git-status.txt"
sha256sum "${LOCAL_BINARY}" > "${OUT_DIR}/local-hashes.sha256"
adb -s "${ADB_SERIAL}" shell sha256sum \
    "${REMOTE_MODEL}" \
    "${REMOTE_DIR}/llama-hetero-ubatch" \
    "${REMOTE_DIR}/libllama.so" \
    "${REMOTE_DIR}/libllama-common.so" \
    "${REMOTE_DIR}/libggml.so" \
    "${REMOTE_DIR}/libggml-base.so" \
    "${REMOTE_DIR}/libggml-cpu.so" \
    "${REMOTE_DIR}/libggml-opencl.so" \
    "${REMOTE_DIR}/libggml-hexagon-wave.so" \
    > "${OUT_DIR}/device-hashes.sha256"

adb -s "${ADB_SERIAL}" shell getprop > "${OUT_DIR}/device-properties.txt"
adb -s "${ADB_SERIAL}" shell dumpsys battery > "${OUT_DIR}/battery-before.txt"
adb -s "${ADB_SERIAL}" shell dumpsys thermalservice > "${OUT_DIR}/thermal-before.txt" 2>&1

COMMON_ARGS="-m ${REMOTE_MODEL} --ubatch 32 --prompt-len 128 --decode-ctx 16 --decode-requests 16 --decode-steps 8 --validation-steps 4"
RUN_INDEX=0

battery_temperature_c() {
    adb -s "${ADB_SERIAL}" shell dumpsys battery |
        awk '/temperature:/ { printf "%.1f", $2 / 10.0; found = 1 } END { if (!found) printf "null" }'
}

run_case() {
    local round=$1
    local mode=$2
    local expected_rc=$3
    shift 3

    RUN_INDEX=$((RUN_INDEX + 1))
    local stem
    stem=$(printf "%02d-round-%02d-%s" "${RUN_INDEX}" "${round}" "${mode}")
    local stdout_file="${OUT_DIR}/runs/${stem}.stdout.log"
    local stderr_file="${OUT_DIR}/runs/${stem}.stderr.log"
    local thermal_before_file="${OUT_DIR}/runs/${stem}.thermal-before.txt"
    local thermal_after_file="${OUT_DIR}/runs/${stem}.thermal-after.txt"
    local temp_before
    local temp_after
    local remote_command

    temp_before=$(battery_temperature_c)
    adb -s "${ADB_SERIAL}" shell dumpsys thermalservice > "${thermal_before_file}" 2>&1
    remote_command="cd ${REMOTE_DIR} && export LD_LIBRARY_PATH=${REMOTE_DIR} && export ADSP_LIBRARY_PATH=${REMOTE_DIR} && export GGML_HEXAGON_MBUF=4192 && ./llama-hetero-ubatch ${COMMON_ARGS} $*"

    set +e
    adb -s "${ADB_SERIAL}" shell "${remote_command}" > "${stdout_file}" 2> "${stderr_file}"
    local rc=$?
    set -e

    temp_after=$(battery_temperature_c)
    adb -s "${ADB_SERIAL}" shell dumpsys thermalservice > "${thermal_after_file}" 2>&1

    local summary
    local full_result
    summary=$(sed -n 's/^HETEROSUMMARY //p' "${stdout_file}" | tail -n 1)
    full_result=$(sed -n 's/^HETEROJSON //p' "${stdout_file}" | tail -n 1)
    [[ -n "${summary}" ]] || summary=null
    [[ -n "${full_result}" ]] || full_result=null

    jq -cn \
        --argjson run_index "${RUN_INDEX}" \
        --argjson round "${round}" \
        --arg mode "${mode}" \
        --arg command "${remote_command}" \
        --arg stdout "runs/${stem}.stdout.log" \
        --arg stderr "runs/${stem}.stderr.log" \
        --arg thermal_before "runs/${stem}.thermal-before.txt" \
        --arg thermal_after "runs/${stem}.thermal-after.txt" \
        --argjson temperature_before_c "${temp_before}" \
        --argjson temperature_after_c "${temp_after}" \
        --argjson exit_code "${rc}" \
        --argjson expected_exit_code "${expected_rc}" \
        --argjson summary "${summary}" \
        --argjson result "${full_result}" \
        '{run_index: $run_index, round: $round, mode: $mode, command: $command,
          stdout: $stdout, stderr: $stderr,
          thermal_before: $thermal_before, thermal_after: $thermal_after,
          battery_temperature_before_c: $temperature_before_c,
          battery_temperature_after_c: $temperature_after_c,
          exit_code: $exit_code, expected_exit_code: $expected_exit_code,
          exit_code_matches: ($exit_code == $expected_exit_code),
          summary: $summary, result: $result}' \
        >> "${OUT_DIR}/runs.jsonl"

    printf 'run=%02d round=%d mode=%s rc=%d expected=%d\n' \
        "${RUN_INDEX}" "${round}" "${mode}" "${rc}" "${expected_rc}"
}

set -e
for round in $(seq 1 "${REPEATS}"); do
    run_case "${round}" gpu-only 0 \
        --dev-gpu GPUOpenCL --single-device-control
    run_case "${round}" npu-only 0 \
        --dev-gpu HTP0 --single-device-control
    run_case "${round}" legacy 0 \
        --dev-gpu GPUOpenCL --dev-npu HTP0 --wavefront-layers 0
    run_case "${round}" wavefront-8 0 \
        --dev-gpu GPUOpenCL --dev-npu HTP0 --wavefront-layers 8
done

run_case 0 strict-placement-audit 5 \
    --dev-gpu GPUOpenCL --dev-npu HTP0 --wavefront-layers 8 \
    --placement-audit --no-warmup

adb -s "${ADB_SERIAL}" shell dumpsys battery > "${OUT_DIR}/battery-after.txt"
adb -s "${ADB_SERIAL}" shell dumpsys thermalservice > "${OUT_DIR}/thermal-after.txt" 2>&1

jq -n \
    --arg schema "hetero-ubatch-op15-capture-v1" \
    --arg run_id "${RUN_ID}" \
    --arg created_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg adb_serial "${ADB_SERIAL}" \
    --arg remote_dir "${REMOTE_DIR}" \
    --arg remote_model "${REMOTE_MODEL}" \
    --arg local_binary "${LOCAL_BINARY}" \
    --argjson repeats "${REPEATS}" \
    --arg run_order "gpu-only,npu-only,legacy,wavefront-8" \
    '{schema: $schema, run_id: $run_id, created_utc: $created_utc,
      adb_serial: $adb_serial, remote_dir: $remote_dir,
      remote_model: $remote_model, local_binary: $local_binary,
      repeats: $repeats, run_order_per_round: ($run_order | split(",")),
      workload: {ubatch: 32, prompt_tokens: 128,
                 decode_context_tokens_per_request: 16,
                 decode_requests: 16, prefill_requests: 1,
                 decode_steps_limit: 8, validation_steps: 4,
                 wavefront_tile_layers: 8},
      files: {runs: "runs.jsonl", git_head: "git-head.txt",
              git_status: "git-status.txt", local_hashes: "local-hashes.sha256",
              device_hashes: "device-hashes.sha256",
              device_properties: "device-properties.txt",
              battery_before: "battery-before.txt", battery_after: "battery-after.txt",
              thermal_before: "thermal-before.txt", thermal_after: "thermal-after.txt"}}' \
    > "${OUT_DIR}/manifest.json"

if jq -e -s 'all(.exit_code_matches)' "${OUT_DIR}/runs.jsonl" >/dev/null; then
    echo "capture complete: ${OUT_DIR}"
else
    echo "capture completed with unexpected exit codes: ${OUT_DIR}" >&2
    exit 1
fi
