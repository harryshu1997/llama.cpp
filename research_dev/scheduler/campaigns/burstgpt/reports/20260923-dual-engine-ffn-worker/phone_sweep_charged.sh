#!/system/bin/sh
# Runs on OP15: standalone TCP ffn-split-worker, NPU-only vs NPU+GPU fractions.
# Charged-run variant: logs battery state per config and samples GPU/DDR/LLCC clocks (su).
# usage: sh phone_sweep.sh <dir> <layers> <columns> <quantum> <iters> "<tokens list>" "<fraction list>" [out-dir]
set -u
dir=$1
layers=$2
columns=$3
quantum=$4
iters=$5
token_list=$6
fraction_list=$7
warmup=2
model=/data/local/tmp/s41-opoffload-dmabuf-v1/Qwen3-14B-Q4KM-dequant-f16.gguf
sha=sha256:0000000000000000000000000000000000000000000000000000000000000000
port=39731
out=${8:-$dir/out}
mkdir -p "$out"
cd "$dir" || exit 1

if ps -A | grep -E "[f]fn-split" > /dev/null; then
    echo "another ffn-split process is running; abort" >&2
    ps -A | grep -E "[f]fn-split" >&2
    exit 3
fi

export LD_LIBRARY_PATH=$dir
export ADSP_LIBRARY_PATH=$dir
export GGML_HEXAGON_MBUF=4192
export GGML_HEXAGON_NHVX=4
export GGML_HEXAGON_NDEV=1
export GGML_HEXAGON_VMEM=3328
export S41_DISABLE_GRAPH_CACHE=1

layer_csv=$(echo "$layers" | awk -F- '{ s = $1; for (i = $1 + 1; i <= $2; i++) s = s "," i; print s }')
layer_count=$(echo "$layer_csv" | awk -F, '{ print NF }')
max_tokens=1
for t in $token_list; do
    if [ "$t" -gt "$max_tokens" ]; then max_tokens=$t; fi
done

for t in $token_list; do
    for f in $fraction_list; do
        tag=T${t}_f${f}
        requests=$(( (warmup + iters) * layer_count ))
        if [ "$f" = "0" ]; then
            unset S43_FFN_SECONDARY_BACKEND S43_FFN_SECONDARY_FRACTION
        elif [ "$f" = "none" ]; then
            export S43_FFN_SECONDARY_BACKEND=none
            unset S43_FFN_SECONDARY_FRACTION
        else
            export S43_FFN_SECONDARY_BACKEND=GPUOpenCL
            export S43_FFN_SECONDARY_FRACTION=$f
        fi
        batt() {
            echo "$1 uptime=$(cut -d' ' -f1 /proc/uptime) $(dumpsys battery | grep -E '^  (level|voltage|temperature|status):' | tr -d ' ' | tr '\n' ' ') notify=$(su -c cat /sys/class/oplus_chg/battery/battery_notify_code)" >> "$out/battery_$tag.txt"
        }
        batt before
        rm -f "$out/stop_$tag"
        su -c "while [ ! -f $out/stop_$tag ]; do echo \$(cut -d' ' -f1 /proc/uptime) \$(cat /sys/class/kgsl/kgsl-3d0/gpuclk) \$(cat /sys/devices/system/cpu/bus_dcvs/DDR/cur_freq) \$(cat /sys/devices/system/cpu/bus_dcvs/LLCC/cur_freq) \$(cat /sys/devices/system/cpu/bus_dcvs/DDR/31091000.qcom,bwmon-ddr/cur_freq) \$(cat /sys/devices/system/cpu/bus_dcvs/DDR/soc:qcom,memlat:ddr:gold/cur_freq) \$(cat /sys/devices/system/cpu/bus_dcvs/DDR/soc:qcom,memlat:ddr:prime/cur_freq) \$(cat /sys/devices/system/cpu/bus_dcvs/DDR/soc:qcom,memlat:ddr:gold-compute/cur_freq); sleep \${S43_CLOCK_PERIOD:-0.2}; done > $out/clocks_$tag.txt" &
        ./llama-ffn-split-worker -m "$model" --artifact-sha256 "$sha" \
            --layers "$layers" --columns "$columns" --column-quantum "$quantum" \
            --backend HTP0 --port "$port" --f16-io --max-tokens "$max_tokens" \
            --max-requests "$requests" > "$out/worker_$tag.log" 2>&1 &
        wpid=$!
        i=0
        while [ $i -lt 1800 ]; do
            if grep -q "ready backend" "$out/worker_$tag.log" 2>/dev/null; then break; fi
            if ! kill -0 $wpid 2>/dev/null; then break; fi
            i=$((i + 1))
            sleep 0.1
        done
        if ! grep -q "ready backend" "$out/worker_$tag.log"; then
            echo "[$tag] worker failed to start" >&2
            tail -20 "$out/worker_$tag.log" >&2
            kill $wpid 2>/dev/null
            wait $wpid 2>/dev/null
            touch "$out/stop_$tag"
            continue
        fi
        echo "driver_start $(cut -d' ' -f1 /proc/uptime)" >> "$out/battery_$tag.txt"
        ./ffn-dual-driver --port "$port" --sha "$sha" --layers "$layer_csv" \
            --n-embd 5120 --columns "$columns" --request-columns "${S43_REQ_COLUMNS:-$columns}" --tokens "$t" --max-tokens "$max_tokens" \
            --iters "$iters" --warmup "$warmup" \
            --out "$out/out_$tag.bin" --csv "$out/calls_$tag.csv" > "$out/driver_$tag.txt" 2>&1
        echo "driver_end $(cut -d' ' -f1 /proc/uptime)" >> "$out/battery_$tag.txt"
        touch "$out/stop_$tag"
        echo "[$tag] $(tail -1 "$out/driver_$tag.txt")"
        j=0
        while kill -0 $wpid 2>/dev/null && [ $j -lt 100 ]; do
            j=$((j + 1))
            sleep 0.1
        done
        if kill -0 $wpid 2>/dev/null; then
            kill $wpid 2>/dev/null
        fi
        wait $wpid 2>/dev/null
        batt after
        unset S43_FFN_SECONDARY_BACKEND S43_FFN_SECONDARY_FRACTION
    done
done
