#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zhihao/moe-resident-routing-4060ti-op15}
SNAPSHOT=${SNAPSHOT:-/home/zhihao/models/Qwen1.5-MoE-A2.7B-Chat-ec052fda}
DATA_DIR=${DATA_DIR:-$PROJECT_ROOT/data/qwen15_14b}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/results/qwen15_14b/op15_paged}
DISK_STORE=${DISK_STORE:-/mnt/storage/qwen15_14b_paged/layer4_experts.bf16.raw}
PHONE_PORT=${PHONE_PORT:-27185}

mkdir -p "$OUTPUT_DIR/logs" "$(dirname "$DISK_STORE")"
cd "$PROJECT_ROOT"

PHONE_PORT="$PHONE_PORT" scripts/start_op15_qwen15_service.sh
cleanup() {
  PHONE_PORT="$PHONE_PORT" scripts/stop_op15_qwen15_service.sh || true
}
trap cleanup EXIT

PYTHONPATH=. CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python \
  -m src.run_qwen15_paged_op15_benchmark \
  --snapshot "$SNAPSHOT" \
  --bundle "$DATA_DIR/benchmark_bundle.pt" \
  --artifacts "$DATA_DIR/warm_proxy_artifacts.json" \
  --output "$OUTPUT_DIR/FULL_PAGED_BENCHMARK_RESULT.json" \
  --disk-store "$DISK_STORE" \
  --phone-port "$PHONE_PORT" \
  --gpu-memory 11GiB \
  --cpu-memory 27GiB \
  2>&1 | tee "$OUTPUT_DIR/logs/full_paged_benchmark.log"
