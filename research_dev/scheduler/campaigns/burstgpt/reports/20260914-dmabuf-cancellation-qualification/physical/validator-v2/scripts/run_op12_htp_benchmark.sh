#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
adb_serial=${ADB_SERIAL:-5ae7a43d}
remote_dir=${REMOTE_DIR:-/data/local/tmp/moe_htp_benchmark}
llama_build=${LLAMA_BUILD:-$project_root/third_party/llama.cpp/build-android-htp}
binary=${BINARY:-$project_root/build/op12_htp_benchmark}
phone_bank=${PHONE_BANK:-/data/local/tmp/qwen35_six_layer_phone_bank.fp16}

adb_command=(adb -s "$adb_serial")
remote_arguments=
if (( $# > 0 )); then
  printf -v remote_arguments ' %q' "$@"
fi
"${adb_command[@]}" get-state >/dev/null
phone_model=$("${adb_command[@]}" shell getprop ro.product.model | tr -d '\r')
if [[ "$phone_model" != CPH2583 ]]; then
  echo "Expected OnePlus 12 CPH2583, found: $phone_model" >&2
  exit 1
fi

"${adb_command[@]}" shell "mkdir -p '$remote_dir'"
"${adb_command[@]}" push \
  "$binary" \
  "$llama_build/bin/libggml-base.so" \
  "$llama_build/bin/libggml-cpu.so" \
  "$llama_build/bin/libggml-hexagon.so" \
  "$llama_build/bin/libggml.so" \
  "$llama_build/ggml/src/ggml-hexagon/libggml-htp-v75.so" \
  "$remote_dir/" >/dev/null

"${adb_command[@]}" shell "
  cd '$remote_dir' &&
  chmod 755 op12_htp_benchmark *.so &&
  export LD_LIBRARY_PATH=\$PWD &&
  export ADSP_LIBRARY_PATH=\$PWD &&
  export GGML_HEXAGON_DEVICES=HTP0:0 &&
  exec ./op12_htp_benchmark --backend HTP0:0 --weights '$phone_bank'$remote_arguments
"
