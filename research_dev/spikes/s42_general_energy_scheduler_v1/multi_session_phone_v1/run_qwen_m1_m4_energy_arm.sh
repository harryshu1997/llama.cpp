#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "$0")" && pwd)
export S42_PHONE_RUN_TAG=${S42_PHONE_RUN_TAG:-qwen-m1-m4-energy}
export S42_QWEN_INDICES=7,21,20,12
export S42_QWEN_DISPATCH=concurrent
export S42_QWEN_POLICY=4:17408,512:0
exec "$here/run_qwen_full_energy_arm.sh" "$@"
