#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zhihao/moe-resident-routing-4060ti-op15}
SNAPSHOT=${SNAPSHOT:-/home/zhihao/models/granite-4.0-h-tiny-791e0d3d}
DATA_DIR=${DATA_DIR:-$PROJECT_ROOT/data/granite4_h_tiny}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/results/granite4_h_tiny/op15_paged}
DISK_STORE=${DISK_STORE:-/mnt/storage/granite4_h_tiny_paged/layer4_experts.bf16.raw}
PHONE_PORT=${PHONE_PORT:-27186}
SERVICE_BINARY=${SERVICE_BINARY:-$PROJECT_ROOT/build/op15_expert_service_granite4}
PHONE_BANK=${PHONE_BANK:-$DATA_DIR/granite4_layer4_phone_bank.fp16}

mkdir -p "$DATA_DIR/logs" "$OUTPUT_DIR/logs" "$(dirname "$DISK_STORE")"
cd "$PROJECT_ROOT"

if [[ ! -f "$DATA_DIR/warm_proxy_artifacts.json" ]]; then
  PYTHONPATH=. CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python \
    -m src.prepare_granite4_op15 \
    --snapshot "$SNAPSHOT" \
    --output-dir "$DATA_DIR" \
    --gpu-memory 11GiB \
    --cpu-memory 27GiB \
    2>&1 | tee "$DATA_DIR/logs/prepare.log"
fi

if [[ ! -x "$SERVICE_BINARY" ]]; then
  OP15_HIDDEN=1536 OP15_INTERMEDIATE=512 \
    OUTPUT_NAME=op15_expert_service_granite4 \
    scripts/build_op15_service.sh
fi

PHONE_PORT="$PHONE_PORT" PHONE_BANK="$PHONE_BANK" \
  SERVICE_BINARY="$SERVICE_BINARY" \
  scripts/start_op15_granite4_service.sh
cleanup() {
  PHONE_PORT="$PHONE_PORT" scripts/stop_op15_granite4_service.sh || true
}
trap cleanup EXIT

PYTHONPATH=. .venv/bin/python \
  -m src.validate_olmoe_phone \
  --phone-bank "$PHONE_BANK" \
  --output "$OUTPUT_DIR/GRANITE4_PHONE_KERNEL_VALIDATION.json" \
  --host 127.0.0.1 \
  --port "$PHONE_PORT" \
  --hidden 1536 \
  --intermediate 512 \
  --bank-experts 16 \
  2>&1 | tee "$OUTPUT_DIR/logs/phone_kernel_validation.log"

PYTHONPATH=. CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python \
  -m src.run_granite4_paged_op15_benchmark \
  --snapshot "$SNAPSHOT" \
  --bundle "$DATA_DIR/benchmark_bundle.pt" \
  --artifacts "$DATA_DIR/warm_proxy_artifacts.json" \
  --output "$OUTPUT_DIR/FULL_PAGED_BENCHMARK_RESULT.json" \
  --disk-store "$DISK_STORE" \
  --phone-port "$PHONE_PORT" \
  --gpu-memory 11GiB \
  --cpu-memory 27GiB \
  --limit-selection 4 \
  --limit-evaluation 8 \
  2>&1 | tee "$OUTPUT_DIR/logs/full_paged_benchmark.log"
