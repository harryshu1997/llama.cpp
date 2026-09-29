#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
adb_serial=${ADB_SERIAL:-5ae7a43d}
remote_probe_dir=${PROBE_REMOTE_DIR:-/data/local/tmp/moe_htp_functionfs_probe}
remote_probe=$remote_probe_dir/op12_htp_service_probe
llama_build=${LLAMA_BUILD:-$project_root/third_party/llama.cpp/build-android-htp}
adb_command=(adb -s "$adb_serial")

"${adb_command[@]}" get-state >/dev/null
phone_model=$("${adb_command[@]}" shell getprop ro.product.model | tr -d '\r')
kernel=$("${adb_command[@]}" shell uname -r | tr -d '\r')
usb_config=$("${adb_command[@]}" shell getprop sys.usb.config | tr -d '\r')
root_access=$("${adb_command[@]}" shell 'su -c id 2>/dev/null || true' | tr -d '\r')
configfs_access=$("${adb_command[@]}" shell 'test -r /config/usb_gadget && echo yes || echo no' | tr -d '\r')
functionfs_mount=$("${adb_command[@]}" shell 'test -w /dev/usb-ffs/moe/ep0 && echo yes || echo no' | tr -d '\r')

printf 'PHONE_MODEL=%s\n' "$phone_model"
printf 'KERNEL=%s\n' "$kernel"
printf 'USB_CONFIG=%s\n' "$usb_config"
printf 'ROOT_ACCESS=%s\n' "${root_access:-no}"
printf 'USB_CONFIGFS_ACCESS=%s\n' "$configfs_access"
printf 'MOE_FUNCTIONFS_EP0_WRITABLE=%s\n' "$functionfs_mount"

if [[ ! -x "$project_root/build/op12_htp_service" ]]; then
  "$project_root/scripts/build_op12_htp_service.sh" >/dev/null
fi
"${adb_command[@]}" shell "mkdir -p '$remote_probe_dir'"
"${adb_command[@]}" push "$project_root/build/op12_htp_service" "$remote_probe" >/dev/null
"${adb_command[@]}" push \
  "$llama_build/bin/libggml-base.so" \
  "$llama_build/bin/libggml-cpu.so" \
  "$llama_build/bin/libggml-hexagon.so" \
  "$llama_build/bin/libggml.so" \
  "$llama_build/ggml/src/ggml-hexagon/libggml-htp-v75.so" \
  "$remote_probe_dir/" >/dev/null
"${adb_command[@]}" shell "chmod 755 '$remote_probe' '$remote_probe_dir'/*.so"
"${adb_command[@]}" shell "cd '$remote_probe_dir' && env LD_LIBRARY_PATH='$remote_probe_dir' ./op12_htp_service_probe --probe-dma-heap"

if [[ "$phone_model" != CPH2583 ]]; then
  printf 'BLOCKED=expected OnePlus 12 CPH2583\n' >&2
  exit 2
fi
if [[ "$functionfs_mount" != yes ]]; then
  printf 'BLOCKED=/dev/usb-ffs/moe is not provisioned by the Android USB gadget configuration\n' >&2
  exit 2
fi

printf 'FUNCTIONFS_PREFLIGHT=ready-for-runtime-dmabuf-ioctl-test\n'
