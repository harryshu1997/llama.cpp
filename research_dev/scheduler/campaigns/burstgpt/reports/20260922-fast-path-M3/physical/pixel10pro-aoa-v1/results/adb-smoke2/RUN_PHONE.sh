#!/system/bin/sh
exec 9>/data/local/tmp/.s42-pixel-ffn-kernels.lock
flock -n 9 || { echo 73 > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/adb-smoke2/EXIT.txt; exit 73; }
cat /proc/sys/kernel/random/boot_id > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/adb-smoke2/BOOT.txt
id > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/adb-smoke2/UID.txt
dumpsys battery > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/adb-smoke2/BATTERY_BEFORE.txt
S42_PIXEL_ECHO_PORT=27149 /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/aoa-echo serial 10280 10288 110 10 1 > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/adb-smoke2/worker.log 2>&1
status=$?
dumpsys battery > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/adb-smoke2/BATTERY_AFTER.txt
echo $status > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/adb-smoke2/EXIT.txt
exit $status
