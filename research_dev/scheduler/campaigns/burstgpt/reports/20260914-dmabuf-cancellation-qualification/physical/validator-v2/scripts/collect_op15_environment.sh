#!/usr/bin/env bash
set -u

PROJECT_ROOT=${PROJECT_ROOT:-/home/zhihao/moe-resident-routing-4060ti-op15}
ADB_SERIAL=${ADB_SERIAL:-3C15AU002CL00000}
ADB_PORT=${ADB_PORT:-5037}
VENV="$PROJECT_ROOT/.venv"

echo "CAPTURED_AT"
date --iso-8601=seconds
echo "HOSTNAME"
hostname
echo "OS"
cat /etc/os-release
echo "CPU"
lscpu
echo "HOST_MEMORY"
free -h
echo "FILESYSTEMS"
df -hT /home /mnt/storage
echo "BLOCK_DEVICES"
lsblk -d -o NAME,ROTA,TRAN,SIZE,MODEL
echo "GPU"
nvidia-smi
echo "GPU_QUERY"
nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv,noheader
echo "PYTHON_ENVIRONMENT"
"$VENV/bin/python" - <<'PY'
import platform
import numpy
import safetensors
import torch
print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("numpy", numpy.__version__)
print("safetensors", safetensors.__version__)
print("cuda_available", torch.cuda.is_available())
PY
echo "ADB"
adb version
adb -P "$ADB_PORT" devices -l
echo "PHONE_IDENTITY"
for property in ro.product.manufacturer ro.product.model ro.product.device ro.soc.model ro.product.cpu.abilist ro.build.version.release ro.build.version.sdk; do
  printf '%s=' "$property"
  adb -P "$ADB_PORT" -s "$ADB_SERIAL" shell getprop "$property" | tr -d '\r'
done
echo "PHONE_MEMORY"
adb -P "$ADB_PORT" -s "$ADB_SERIAL" shell cat /proc/meminfo | head -n 10
echo "PHONE_STORAGE"
adb -P "$ADB_PORT" -s "$ADB_SERIAL" shell df -h /data
echo "PHONE_BATTERY"
adb -P "$ADB_PORT" -s "$ADB_SERIAL" shell dumpsys battery | grep -Ei 'powered|temperature|PhoneTemp|level|status' | head -n 30
echo "PHONE_SERVICE_LOG"
adb -P "$ADB_PORT" -s "$ADB_SERIAL" shell cat /data/local/tmp/op15_expert_service.log
echo "BANK_HASHES"
sha256sum "$PROJECT_ROOT/data/qwen35_layer20_phone_bank.fp16" "$PROJECT_ROOT/data/qwen35_layer20_server_bank.bf16"
echo "SOURCE_HASHES"
sha256sum "$PROJECT_ROOT/src/op15_opencl_service.cpp" "$PROJECT_ROOT/src/benchmark_4060ti_op15.py" "$PROJECT_ROOT/src/export_qwen35_expert_bank.py"
