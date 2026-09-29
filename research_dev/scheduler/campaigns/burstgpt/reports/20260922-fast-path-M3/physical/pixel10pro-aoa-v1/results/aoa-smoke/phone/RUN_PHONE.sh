#!/system/bin/sh
exec 9>/data/local/tmp/.s42-pixel-ffn-kernels.lock
flock -n 9 9>&9 || { echo 73 > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/aoa-smoke/EXIT.txt; exit 73; }
cat /proc/sys/kernel/random/boot_id > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/aoa-smoke/BOOT.txt
id > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/aoa-smoke/UID.txt
dumpsys battery > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/aoa-smoke/BATTERY_BEFORE.txt
/data/local/tmp/s42-pixel10pro-aoa-20260924-v1/aoa-echo serial 10280 10288 110 10 1 > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/aoa-smoke/worker.log 2>&1
status=$?
dumpsys battery > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/aoa-smoke/BATTERY_AFTER.txt
echo $status > /data/local/tmp/s42-pixel10pro-aoa-20260924-v1/aoa-smoke/EXIT.txt
exit $status
