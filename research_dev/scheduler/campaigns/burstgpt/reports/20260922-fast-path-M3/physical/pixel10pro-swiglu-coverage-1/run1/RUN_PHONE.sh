#!/system/bin/sh
cd /data/local/tmp/s42-pixel10pro-gemv-tune-20260923-swiglu-coverage-v1 || exit 2
exec 9>/data/local/tmp/.s42-pixel-ffn-kernels.lock
flock -n 9 9>&9 || exit 3
mkdir raw || exit 4
ps -A -o PID,ARGS > raw/PROCESSES_BEFORE.txt
if grep -q '[l]lama-ffn' raw/PROCESSES_BEFORE.txt; then echo 'FAIL existing worker'; exit 5; fi
cat /proc/sys/kernel/random/boot_id > raw/BOOT_BEFORE.txt
sha256sum /data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1/libggml-base.so /data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1/libggml-cpu.so /data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1/libggml-vulkan.so /data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1/libggml.so /data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1/llama-ffn-split-worker /data/local/tmp/s42-pixel10pro-qualification-20260922-v1/HTP0.ffn.gguf /data/local/tmp/s42-pixel10pro-gemv-tune-20260923-swiglu-coverage-v1/libggml-vulkan.so > raw/HASHES.txt || exit 6
cmp raw/HASHES.txt EXPECTED_HASHES.sha256 || exit 6
run_arm() {
    name="$1"
    shift
    mkdir "raw/$name" || return 7
    dumpsys battery > "raw/$name/BATTERY_BEFORE.txt"
    "$@" > "raw/$name/worker.log" 2>&1 &
    worker_pid=$!
    echo "$worker_pid" > "raw/$name/PID.txt"
    ready=0
    for attempt in $(seq 1 240); do
        if grep -q '\[ffn-worker\] ready backend=' "raw/$name/worker.log"; then ready=1; break; fi
        if ! kill -0 "$worker_pid" 2>/dev/null; then echo "FAIL $name startup"; return 8; fi
        sleep 1
    done
    if [ "$ready" != 1 ]; then echo "FAIL $name readiness"; return 9; fi
    nc -n -w 5 -W 120 127.0.0.1 27141 < REQUESTS.bin > "raw/$name/RESPONSES.bin"
    network_status=$?
    echo "$network_status" > "raw/$name/NETCAT_EXIT.txt"
    if [ "$network_status" != 0 ]; then echo "FAIL $name capture"; return 10; fi
    wait "$worker_pid"
    worker_status=$?
    echo "$worker_status" > "raw/$name/WORKER_EXIT.txt"
    if [ "$worker_status" != 0 ]; then echo "FAIL $name worker"; return 11; fi
    dumpsys battery > "raw/$name/BATTERY_AFTER.txt"
    echo "PASS $name"
}
run_arm 02-on env -u GGML_VK_PERF_LOGGER -u GGML_VK_DISABLE_FUSION -u S42_PIXEL_PROFILE_OPS -u S42_PIXEL_F16_WG -u S42_PIXEL_F16_ROWS -u S42_PIXEL_F16_SUBGROUP -u S42_PIXEL_F16_SHADER -u S42_PIXEL_FUSE_SWIGLU LD_LIBRARY_PATH=/data/local/tmp/s42-pixel10pro-gemv-tune-20260923-swiglu-coverage-v1:/data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1 S42_PIXEL_F16_WG=128 S42_PIXEL_F16_ROWS=8 S42_PIXEL_F16_SUBGROUP=128 S42_PIXEL_F16_SHADER=vec4_u1 GGML_VK_PERF_LOGGER=1 S42_PIXEL_FUSE_SWIGLU=1 /data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1/llama-ffn-split-worker -m /data/local/tmp/s42-pixel10pro-qualification-20260922-v1/HTP0.ffn.gguf --artifact-sha256 sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718 --layers 18,19,20,21,22,23 --columns 17408 --column-quantum 4352 --backend Vulkan0 --port 27141 --bind 127.0.0.1 --f16-io --max-tokens 4 --max-requests 12 || exit $?
cat /proc/sys/kernel/random/boot_id > raw/BOOT_AFTER.txt
ps -A -o PID,ARGS > raw/PROCESSES_AFTER.txt
if grep -q '[l]lama-ffn' raw/PROCESSES_AFTER.txt; then echo 'FAIL remaining worker'; exit 12; fi
cmp raw/BOOT_BEFORE.txt raw/BOOT_AFTER.txt || exit 13
echo PASS > raw/DONE.txt
echo 'PASS all arms'
