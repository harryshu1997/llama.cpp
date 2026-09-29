#!/usr/bin/env bash
set -euo pipefail

adb_bin=${S41_ADB:-adb}
serial=${S41_PHONE_SERIAL:-3C15AU002CL00000}
export ADB_SERVER_PORT=${ADB_SERVER_PORT:-5039}

exec "$adb_bin" -s "$serial" shell \
    /data/local/tmp/continuous-matmul-v1/run_phone_bench.sh "$@"
