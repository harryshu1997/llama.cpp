#!/system/bin/sh
set -eu

script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export LD_LIBRARY_PATH=$script_root
export ADSP_LIBRARY_PATH=$script_root
exec "$script_root/ffs_dmabuf_htp_worker" "$@"
