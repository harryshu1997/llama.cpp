#!/system/bin/sh
set -eu

runtime=/data/local/tmp/continuous-matmul-v1
export LD_LIBRARY_PATH=$runtime
export ADSP_LIBRARY_PATH=$runtime
export GGML_HEXAGON_NDEV=2
export GGML_HEXAGON_MBUF=4192
export GGML_HEXAGON_NHVX=4

exec "$runtime/backend_matmul_bench" "$@"
