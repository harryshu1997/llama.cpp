#!/usr/bin/env bash
set -euo pipefail

project_root=/home/myid/zs89458/Documents/moe-resident-routing-a6000
hf_home=/mnt/data_s4t/zs89458-moe-routing/huggingface
revision=59d61f3ce65a6d9863b86d2e96597125219dc754

cd "$project_root"
install -d -m 700 "$hf_home"
mkdir -p results/qwen35_35b_a3b/logs
available_bytes=$(df --output=avail -B1 "$hf_home" | tail -n 1 | tr -d ' ')
required_bytes=85000000000
if [ "$available_bytes" -lt "$required_bytes" ]; then
  echo "Refusing download: $available_bytes bytes free at $hf_home; need at least $required_bytes." >&2
  exit 1
fi
export HF_HOME="$hf_home"
export HF_HUB_DISABLE_TELEMETRY=1
.venv-qwen35/bin/hf download Qwen/Qwen3.5-35B-A3B --revision "$revision" \
  2>&1 | tee results/qwen35_35b_a3b/logs/download_model.log
