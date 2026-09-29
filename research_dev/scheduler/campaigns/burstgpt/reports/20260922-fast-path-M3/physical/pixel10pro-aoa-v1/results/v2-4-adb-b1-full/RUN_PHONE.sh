#!/system/bin/sh
exec 9>/data/local/tmp/.s42-pixel-ffn-kernels.lock
flock -n 9 9>&9 || { echo 73 > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/v2-4-adb-b1-full/EXIT.txt; exit 73; }
cat /proc/sys/kernel/random/boot_id > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/v2-4-adb-b1-full/BOOT.txt
id > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/v2-4-adb-b1-full/UID.txt
dumpsys battery > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/v2-4-adb-b1-full/BATTERY_BEFORE.txt
LD_LIBRARY_PATH=/data/local/tmp/s42-pixel10pro-aoa-20260924-v1 S42_PIXEL_CPU_THREADS=6 S42_PIXEL_CPU_MASK=fc S42_PIXEL_CPU_POOL=1 S42_PIXEL_PACKED_WEIGHTS=1 S42_PIXEL_CPU_QUANT_RESIDUAL=1 S42_PIXEL_FUSED_RESIDUAL=1 S42_PIXEL_CPU_PAIR_DOT=1 S42_PIXEL_CPU_ROW_CHUNK=64 S42_PIXEL_CPU_ROW_PROFILE=0 S42_PIXEL_AOA=0 /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/llama-ffn-split-worker -m /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/QWEN_PACKED.ffn.gguf --artifact-sha256 sha256:940f5f1f2ce0c68d726713e0b1ec86808334c7ca769feac07cd3fa8581c4eae9 --layers 18,19,20,21,22,23 --columns 17408 --column-quantum 4352 --backend CPU --port 27149 --bind 127.0.0.1 --f16-io --max-tokens 4 --max-requests 420 > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/v2-4-adb-b1-full/worker.log 2>&1 &
worker_pid=$!
echo $worker_pid > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/v2-4-adb-b1-full/PID.txt
wait $worker_pid
status=$?
dumpsys battery > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/v2-4-adb-b1-full/BATTERY_AFTER.txt
echo $status > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/v2-4-adb-b1-full/EXIT.txt
exit $status
