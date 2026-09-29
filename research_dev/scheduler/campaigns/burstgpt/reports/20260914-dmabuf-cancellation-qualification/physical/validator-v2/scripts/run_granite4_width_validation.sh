#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zhihao/moe-resident-routing-4060ti-op15}
SNAPSHOT=${SNAPSHOT:-/home/zhihao/models/granite-4.0-h-tiny-791e0d3d}
DATA_DIR=${DATA_DIR:-$PROJECT_ROOT/data/granite4_h_tiny}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/results/granite4_h_tiny/op15_paged}
PHONE_PORT=${PHONE_PORT:-27186}
PHONE_BANK=${PHONE_BANK:-$DATA_DIR/granite4_layer4_phone_bank.fp16}
SERVICE_BINARY=${SERVICE_BINARY:-$PROJECT_ROOT/build/op15_expert_service_granite4}

mkdir -p "$OUTPUT_DIR/logs"
cd "$PROJECT_ROOT"

PHONE_PORT="$PHONE_PORT" PHONE_BANK="$PHONE_BANK" \
  SERVICE_BINARY="$SERVICE_BINARY" \
  scripts/start_op15_granite4_service.sh
cleanup() {
  PHONE_PORT="$PHONE_PORT" scripts/stop_op15_granite4_service.sh || true
}
trap cleanup EXIT

PYTHONPATH=. CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python \
  -m src.run_granite4_width_validation \
  --snapshot "$SNAPSHOT" \
  --bundle "$DATA_DIR/benchmark_bundle.pt" \
  --artifacts "$DATA_DIR/warm_proxy_artifacts.json" \
  --output "$OUTPUT_DIR/WIDTH_VALIDATION_RESULT.json" \
  --phone-port "$PHONE_PORT" \
  --gpu-memory 11GiB \
  --cpu-memory 27GiB \
  --load-count 1 \
  --warm-widths 1,2,4,8 \
  --limit-evaluation 8 \
  2>&1 | tee "$OUTPUT_DIR/logs/width_validation.log"
