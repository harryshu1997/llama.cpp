#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zhihao/moe-resident-routing-4060ti-op15}
SNAPSHOT=${SNAPSHOT:-/mnt/storage/moe-resident-routing-4060ti-op15-cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754}
BANKS=${BANKS:-$PROJECT_ROOT/data/resident_expert_banks.json}
COEFFICIENTS=${COEFFICIENTS:-$PROJECT_ROOT/data/ensemble_coefficients.json}
REFERENCE=${REFERENCE:-$PROJECT_ROOT/data/wikitext2_test_reference_8x64.pt}
PHONE_BANK_FILES=${PHONE_BANK_FILES:-$PROJECT_ROOT/data/qwen35_all40_phone_banks.low.fp16,$PROJECT_ROOT/data/qwen35_all40_phone_banks.high.fp16}
OUTPUT=${OUTPUT:-$PROJECT_ROOT/results/qwen35_all40_physical/RESULT.json}
POLICY=${POLICY:?Set POLICY to the selected all-layer policy, for example load5_warm4}
EXAMPLES=${EXAMPLES:-8}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-64}
BATCH_SIZE=${BATCH_SIZE:-8}
DECODE_REPETITIONS=${DECODE_REPETITIONS:-3}

cd "$PROJECT_ROOT"
mkdir -p "${OUTPUT%/*}"
export PYTHONPATH="$PROJECT_ROOT"
export TOKENIZERS_PARALLELISM=false
# Torch 2.13's optional Python-native Triton overrides JIT-compile a helper
# against Python.h on first RoPE matmul.  The desktop's Python 3.14 runtime has
# no development headers, so use the ordinary CUDA/ATen kernels instead.  This
# does not alter the custom HDD expert pager or OP15 OpenCL branch.
export TORCH_DISABLE_NATIVE_JIT=${TORCH_DISABLE_NATIVE_JIT:-1}

.venv/bin/python -m src.run_qwen35_all_layer_phone \
  --snapshot "$SNAPSHOT" \
  --banks "$BANKS" \
  --coefficients "$COEFFICIENTS" \
  --reference "$REFERENCE" \
  --phone-bank-files "$PHONE_BANK_FILES" \
  --output "$OUTPUT" \
  --policy "$POLICY" \
  --examples "$EXAMPLES" \
  --sequence-length "$SEQUENCE_LENGTH" \
  --batch-size "$BATCH_SIZE" \
  --decode-repetitions "$DECODE_REPETITIONS"
