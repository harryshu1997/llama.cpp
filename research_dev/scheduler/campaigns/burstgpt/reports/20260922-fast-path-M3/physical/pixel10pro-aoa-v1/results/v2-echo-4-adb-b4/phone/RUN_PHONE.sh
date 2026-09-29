#!/system/bin/sh
exec 9>/data/local/tmp/.s42-pixel-ffn-kernels.lock
flock -n 9 9>&9 || { echo 73 > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/v2-echo-4-adb-b4/EXIT.txt; exit 73; }
cat /proc/sys/kernel/random/boot_id > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/v2-echo-4-adb-b4/BOOT.txt
id > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/v2-echo-4-adb-b4/UID.txt
dumpsys battery > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/v2-echo-4-adb-b4/BATTERY_BEFORE.txt
S42_PIXEL_ECHO_PORT=27149 /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/aoa-echo serial 41000 41008 330 30 1 > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/v2-echo-4-adb-b4/worker.log 2>&1 &
worker_pid=$!
echo $worker_pid > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/v2-echo-4-adb-b4/PID.txt
wait $worker_pid
status=$?
dumpsys battery > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/v2-echo-4-adb-b4/BATTERY_AFTER.txt
echo $status > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/v2-echo-4-adb-b4/EXIT.txt
exit $status
