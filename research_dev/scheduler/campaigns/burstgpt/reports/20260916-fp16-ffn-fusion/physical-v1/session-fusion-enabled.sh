#!/system/bin/sh
set -eu
export GGML_HEXAGON_OPFUSION=1
exec sh /data/local/tmp/s42-hal-runtime-probe-20260906-v4/direct_phone_ffn_session.sh "$@"
