#!/usr/bin/env bash
set -euo pipefail

root=/home/zhihao/s42-dimensional-residency-short-gate-20260830-v6
source=/home/zhihao/s42-work-conserving-mixed-gate-20260830-v1
repo=/home/zhihao/llama.cpp-release
test ! -e "$root"
mkdir -p "$root/inputs"
cp "$source/inputs/UNIFIED_RUNTIME_CATALOG.json" "$root/inputs/"
cp "$source/inputs/OBSERVATION_SOURCE_CATALOG.json" "$root/inputs/"
cp "$source/inputs/AUTOMATED_OBSERVATIONS.json" "$root/inputs/"
cp "$source/inputs/ADAPTIVE_DECODE_OBSERVATIONS.json" "$root/inputs/"
cp "$source/inputs/TRANSPORT_QUALIFICATION_IDENTITY.json" "$root/inputs/"

python3 - "$repo" "$root/SOURCE_MANIFEST.json" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
rows = []
for path in sorted((root / "research_dev" / "scheduler").rglob("*")):
    if path.is_file() and path.suffix in {".py", ".sh", ".zsh"}:
        rows.append({
            "path": str(path.relative_to(root)),
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        })

def git(*arguments):
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        encoding="ascii",
        text=True,
    ).stdout.strip()

body = {
    "branch": git("branch", "--show-current"),
    "files": rows,
    "head": git("rev-parse", "HEAD"),
    "schema": "research-scheduler-source-manifest-v1",
}
encoded = json.dumps(
    body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
).encode("ascii")
body["manifest_sha256"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
output.write_text(
    json.dumps(
        body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ) + "\n",
    encoding="ascii",
)
PY

command=(
    python3 "$repo/research_dev/scheduler/campaigns/burstgpt/runner.py"
    --large-requests "$repo/research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1/REQUESTS_SEMANTIC_SOURCE.jsonl"
    --overlay-requests "$repo/research_dev/spikes/s42_general_energy_scheduler_v1/full_fp16_burstgpt_v1/small_model_overlay_v1/REQUESTS_LLAMA1B_10.jsonl"
    --trace-manifest "$repo/research_dev/spikes/s42_general_energy_scheduler_v1/full_fp16_burstgpt_v1/small_model_overlay_v1/TRACE_MANIFEST.json"
    --source-manifest "$root/SOURCE_MANIFEST.json"
    --capability-catalog "$root/inputs/UNIFIED_RUNTIME_CATALOG.json"
    --qwen-manifest "$repo/research_dev/scheduler/campaigns/burstgpt/data/QWEN_MANIFEST.json"
    --gemma-manifest "$repo/research_dev/scheduler/campaigns/burstgpt/data/GEMMA_MANIFEST.json"
    --qwen-model /home/zhihao/models/Qwen3-14B-Q4KM-dequant-f16.gguf
    --gemma-model /home/zhihao/models/gemma-4-12B-Q40-dequant-f16.gguf
    --llama-model /home/zhihao/models/Llama-3.2-1B-Instruct-Q4_0.gguf
    --gguf-manifest-cache /home/zhihao/.cache/llama.cpp/research-scheduler/gguf-manifests.json
    --server /home/zhihao/s42-helper-attach-build-20260830-v1/bin/llama-server
    --resident-server /home/zhihao/s42-helper-attach-build-20260830-v1/bin/llama-server
    --cuda-lib-dir /mnt/storage/s21_deps/cuda-13.2.1/lib
    --resident-lib-dir /home/zhihao/s42-helper-attach-build-20260830-v1/bin
    --bridge /home/zhihao/s41-dynamic-ffn-v1/ffn_dmabuf_bridge-unified-v9-repair-r3
    --close-helper "$repo/research_dev/scheduler/adapters/close_resident_bridge.py"
    --adb /usr/bin/adb
    --phone-usb-close /home/zhihao/s42-mixed-residency-priority-deploy-20260828-v1/build-server-cuda/bin/llama-ffn-split-usb-close
    --phone-session /data/local/tmp/s42-route-evidence-physical-v1/direct_phone_ffn_session.sh
    --phone-restore /data/local/tmp/s42-unified-direct-v1/restore_android_usb.sh
    --phone-worker /data/local/tmp/s42-route-evidence-physical-v1/llama-ffn-split-worker
    --phone-busybox /data/adb/magisk/busybox
    --qwen-phone-model /data/local/tmp/s41-opoffload-dmabuf-v1/Qwen3-14B-Q4KM-dequant-f16.gguf
    --gemma-phone-model /data/local/tmp/s41-opoffload-dmabuf-v1/gemma-4-12B-Q40-dequant-f16.gguf
    --phone-session-root /data/local/tmp/s42-dimensional-residency-short-gate-v6
    --phone-whole-server /data/local/tmp/llama-ubatch-op15/bin/llama-server
    --phone-whole-library-directory /data/local/tmp/llama-ubatch-op15/bin
    --phone-whole-model /data/local/tmp/unifer/llamacpp/Llama-3.2-1B-Instruct-Q4_0.gguf
    --phone-whole-state-directory /data/local/tmp/s42-unified-whole-model-v1
    --phone-whole-executable-device GPUOpenCL
    --phone-remote-hash-cache /home/zhihao/.cache/llama.cpp/research-scheduler/phone-artifact-hashes-mixed-v1.json
    --phone-diagnostic-endpoint http://192.168.42.1:18383
    --phone-battery-ppm 1000000
    --phone-usb-serial 3C15AU002CL00000
    --phone-android-gadget /config/usb_gadget/g1
    --phone-functionfs-gadget /config/usb_gadget/g2
    --phone-functionfs-root /dev/usb-ffs/s41
    --phone-usb-controller a600000.dwc3
    --adb-port 5037
    --minimum-usb-speed-mbps 5000
    --phone-kernel-release 6.12.23-android16-5-o-g227664cbe007-4k
    --phone-boot-image-sha256 sha256:26e8d41808b10b70264d958582bb6f6c9fba634c34275e0bc28fc820a6d3fb8d
    --usb-qualification-identity "$root/inputs/TRANSPORT_QUALIFICATION_IDENTITY.json"
    --nmcli /usr/bin/nmcli
    --selection-mode energy-aware
    --energy-attribution-kind diagnostic
    --output "$root/run"
    --execute
    --confirm RUN_UNIFIED_FP16_LLAMA_OVERLAY
    --phone-resident-workers /data/local/tmp/s42-route-evidence-physical-v1/llama-ffn-split-resident-workers
    --phone-resident-router /data/local/tmp/s42-route-evidence-physical-v1/llama-ffn-split-resident-router
    --phone-multi-session-port-base 26760
    --transport-host-dependency llama-server-impl=/home/zhihao/s42-helper-attach-build-20260830-v1/bin/libllama-server-impl.so
    --transport-host-dependency llama-common=/home/zhihao/s42-helper-attach-build-20260830-v1/bin/libllama-common.so
    --transport-host-dependency llama=/home/zhihao/s42-helper-attach-build-20260830-v1/bin/libllama.so
    --transport-host-dependency ggml=/home/zhihao/s42-helper-attach-build-20260830-v1/bin/libggml.so
    --transport-host-dependency ggml-base=/home/zhihao/s42-helper-attach-build-20260830-v1/bin/libggml-base.so
    --transport-host-dependency ggml-cpu=/home/zhihao/s42-helper-attach-build-20260830-v1/bin/libggml-cpu.so
    --transport-host-dependency ggml-cuda=/home/zhihao/s42-helper-attach-build-20260830-v1/bin/libggml-cuda.so
    --observation-store-input "$root/inputs/AUTOMATED_OBSERVATIONS.json"
    --observation-source-catalog "$root/inputs/OBSERVATION_SOURCE_CATALOG.json"
    --adaptive-observation-store-input "$root/inputs/ADAPTIVE_DECODE_OBSERVATIONS.json"
    --adaptive-observation-source-catalog "$root/inputs/OBSERVATION_SOURCE_CATALOG.json"
    --request-indices 36,41,42,43,44
)
printf '%q ' "${command[@]}" > "$root/RUN_COMMAND.txt"
printf '\n' >> "$root/RUN_COMMAND.txt"
sha256sum "$root"/inputs/*.json "$root/SOURCE_MANIFEST.json" > "$root/INPUT_SHA256SUMS.txt"
exec "${command[@]}"
