#!/system/bin/sh
set -eu

if [ "$#" -ne 9 ]; then
    echo "usage: $0 <binary> <model> <layers> <columns> <backend> <session-root> <restore-script> <timeout-seconds> <max-requests>" >&2
    exit 2
fi

binary=$1
model=$2
layers=$3
columns=$4
backend=$5
session_root=$6
restore_script=$7
timeout_seconds=$8
max_requests=$9

g1=/config/usb_gadget/g1
g2=/config/usb_gadget/g2
udc=a600000.dwc3
ffs_root=/dev/usb-ffs/s41
ready_file=$session_root/descriptors.ready
worker_log=$session_root/worker.log
binary_root=$(CDPATH= cd -- "$(dirname -- "$binary")" && pwd)
ncm_function=

case "${S41_USB_NCM:-0}" in
    0) ;;
    1) ncm_function=$g2/functions/ncm.usb0 ;;
    *)
        echo "invalid S41_USB_NCM: ${S41_USB_NCM}" >&2
        exit 2
        ;;
esac

mkdir -p "$session_root"
rm -f "$ready_file"

worker_pid=
head_pid=
head_proxy_pid=
watchdog_pid=
cleanup() {
    rm -f "$session_root/active"
    if [ -n "$watchdog_pid" ]; then
        kill "$watchdog_pid" 2>/dev/null || true
    fi
    if [ -n "$head_proxy_pid" ]; then
        attempt=0
        while [ "$attempt" -lt 20 ] && kill -0 "$head_proxy_pid" 2>/dev/null; do
            attempt=$((attempt + 1))
            sleep 0.05
        done
        kill "$head_proxy_pid" 2>/dev/null || true
        wait "$head_proxy_pid" 2>/dev/null || true
    fi
    if [ -n "$head_pid" ]; then
        kill "$head_pid" 2>/dev/null || true
        wait "$head_pid" 2>/dev/null || true
    fi
    sh "$restore_script" "$session_root" || true
}
trap cleanup EXIT INT TERM
trap '' HUP

sh "$restore_script" "$session_root" || true
: > "$session_root/active"
(
    sleep "$timeout_seconds"
    if [ -f "$session_root/active" ]; then
        echo "[ffs-session] watchdog restoring USB" >> "$session_root/session.log"
        sh "$restore_script" "$session_root"
    fi
) &
watchdog_pid=$!

mkdir -p "$ffs_root"
mkdir "$g2/functions/ffs.s41"
if [ -n "$ncm_function" ]; then
    mkdir "$ncm_function"
fi
mount -t functionfs s41 "$ffs_root"

export LD_LIBRARY_PATH=$binary_root
export ADSP_LIBRARY_PATH=$binary_root
export GGML_HEXAGON_MBUF=4192
export GGML_HEXAGON_NHVX=${GGML_HEXAGON_NHVX:-4}
export S41_DISABLE_GRAPH_CACHE=1
io_flag=
case "${S41_FFN_F16_IO:-0}" in
    0) ;;
    1) io_flag=--f16-io ;;
    *)
        echo "invalid S41_FFN_F16_IO: ${S41_FFN_F16_IO}" >&2
        exit 2
        ;;
esac
staged_dmabuf_flag=
case "${S41_FFN_STAGED_DMABUF:-0}" in
    0) ;;
    1) staged_dmabuf_flag=--staged-dmabuf ;;
    *)
        echo "invalid S41_FFN_STAGED_DMABUF: ${S41_FFN_STAGED_DMABUF}" >&2
        exit 2
        ;;
esac
max_tokens=${S41_FFN_MAX_TOKENS:-1}
column_quantum=${S41_FFN_COLUMN_QUANTUM:-512}
alternate_columns=${S41_FFN_ALTERNATE_COLUMNS:-0}
case "$alternate_columns" in
    ''|*[!0-9]*)
        echo "invalid S41_FFN_ALTERNATE_COLUMNS: $alternate_columns" >&2
        exit 2
        ;;
esac
alternate_flag=
if [ "$alternate_columns" -gt 0 ]; then
    alternate_flag="--alternate-columns $alternate_columns"
fi
GGML_HEXAGON_NDEV=1 GGML_HEXAGON_VMEM="${S41_FFN_VMEM:-3328}" \
    "$binary" -m "$model" --layers "$layers" --columns "$columns" \
    --backend "$backend" --ffs-root "$ffs_root" --ready-file "$ready_file" \
    --max-requests "$max_requests" --max-tokens "$max_tokens" \
    --column-quantum "$column_quantum" $alternate_flag $io_flag \
    $staged_dmabuf_flag \
    > "$worker_log" 2>&1 &
worker_pid=$!
printf '%s\n' "$worker_pid" > "$session_root/worker.pid"

ready=0
attempt=0
while [ "$attempt" -lt 1200 ]; do
    if [ -f "$ready_file" ]; then
        ready=1
        break
    fi
    if ! kill -0 "$worker_pid" 2>/dev/null; then
        break
    fi
    attempt=$((attempt + 1))
    sleep 0.05
done
if [ "$ready" -ne 1 ]; then
    echo "[ffs-session] worker did not publish descriptors" >&2
    cat "$worker_log" >&2 || true
    exit 1
fi

if [ -n "${S41_LM_HEAD_BINARY:-}" ]; then
    head_model=${S41_LM_HEAD_MODEL:-$model}
    head_backend=${S41_LM_HEAD_BACKEND:-GPUOpenCL}
    head_ndev=${S41_LM_HEAD_NDEV:-0}
    head_port=${S41_LM_HEAD_PORT:-25661}
    head_rows=${S41_LM_HEAD_ROWS:-46080}
    head_top_k=${S41_LM_HEAD_TOP_K:-32}
    head_max_requests=${S41_LM_HEAD_MAX_REQUESTS:-0}
    head_log=$session_root/lm-head-worker.log
    head_io_flag=
    case "${S41_LM_HEAD_F16_IO:-1}" in
        0) ;;
        1) head_io_flag=--f16-io ;;
        *)
            echo "invalid S41_LM_HEAD_F16_IO: ${S41_LM_HEAD_F16_IO}" >&2
            exit 2
            ;;
    esac

    GGML_HEXAGON_NDEV="$head_ndev" GGML_HEXAGON_VMEM=256 \
        "$S41_LM_HEAD_BINARY" -m "$head_model" --rows "$head_rows" \
        --top-k "$head_top_k" --backend "$head_backend" \
        --bind 0.0.0.0 --port "$head_port" \
        --max-requests "$head_max_requests" $head_io_flag \
        > "$head_log" 2>&1 &
    head_pid=$!
    printf '%s\n' "$head_pid" > "$session_root/lm-head-worker.pid"

    head_ready=0
    attempt=0
    while [ "$attempt" -lt 1200 ]; do
        if grep -q '^\[lm-head-worker\] ready ' "$head_log" 2>/dev/null; then
            head_ready=1
            break
        fi
        if ! kill -0 "$head_pid" 2>/dev/null; then
            break
        fi
        attempt=$((attempt + 1))
        sleep 0.05
    done
    if [ "$head_ready" -ne 1 ]; then
        echo "[ffs-session] LM-head worker did not become ready" >&2
        cat "$head_log" >&2 || true
        exit 1
    fi
    echo "[ffs-session] LM-head worker ready backend=$head_backend port=$head_port rows=$head_rows" \
        >> "$session_root/session.log"
fi

if [ -n "$(cat "$g2/UDC" 2>/dev/null)" ]; then
    printf '\n' > "$g2/UDC"
fi
rm -f "$g2/configs/b.1/f1"
rm -f "$g2/configs/b.1/f2"
printf '0x18d1' > "$g2/idVendor"
printf '0x2d00' > "$g2/idProduct"
printf '0x0320' > "$g2/bcdUSB"
printf '0x0100' > "$g2/bcdDevice"
printf '500' > "$g2/configs/b.1/MaxPower"
printf '0x80' > "$g2/configs/b.1/bmAttributes"
printf 'Gemma4 operator offload' > "$g2/strings/0x409/manufacturer"
printf 'FFN HTP DMA-BUF' > "$g2/strings/0x409/product"
printf 'S41FFN0001' > "$g2/strings/0x409/serialnumber"
printf 'ffn_htp_dmabuf' > "$g2/configs/b.1/strings/0x409/configuration"
ln -s "$g2/functions/ffs.s41" "$g2/configs/b.1/f1"
if [ -n "$ncm_function" ]; then
    ln -s "$ncm_function" "$g2/configs/b.1/f2"
fi

printf '\n' > "$g1/UDC"
printf '%s' "$udc" > "$g2/UDC"
if [ -n "$ncm_function" ]; then
    attempt=0
    while [ "$attempt" -lt 100 ]; do
        if [ -e /sys/class/net/usb0 ]; then
            sleep 1
            printf '0' > /proc/sys/net/ipv6/conf/usb0/disable_ipv6
            printf '0' > /proc/sys/net/ipv6/conf/usb0/accept_dad
            ip link set usb0 up
            ip -6 addr add fe80::2/64 dev usb0 2>/dev/null || true
            ip -6 route replace fe80::/64 dev usb0 table 1033
            ip -6 rule add priority 9999 from fe80::2/128 to fe80::/64 \
                lookup 1033 2>/dev/null || true
            ip -6 addr show dev usb0 >> "$session_root/session.log"
            ip -6 rule show >> "$session_root/session.log"
            ip -6 route show table 1033 >> "$session_root/session.log"
            break
        fi
        attempt=$((attempt + 1))
        sleep 0.05
    done
    if [ "${S41_LM_HEAD_NCM:-0}" = 1 ] && [ -n "$head_pid" ]; then
        if [ -z "${S41_LM_HEAD_NCM_PROXY:-}" ]; then
            echo "S41_LM_HEAD_NCM_PROXY is required" >&2
            exit 2
        fi
        "$S41_LM_HEAD_NCM_PROXY" "$head_port" usb0 fe80::2 "$head_port" \
            > "$session_root/lm-head-ncm.log" 2>&1 &
        head_proxy_pid=$!
        printf '%s\n' "$head_proxy_pid" > "$session_root/lm-head-ncm.pid"
    fi
fi
echo "[ffs-session] custom FFN gadget bound" >> "$session_root/session.log"

set +e
wait "$worker_pid"
worker_status=$?
set -e
worker_pid=
echo "[ffs-session] worker_status=$worker_status" >> "$session_root/session.log"
exit "$worker_status"
