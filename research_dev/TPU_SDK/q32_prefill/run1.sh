#!/system/bin/sh
# Phone-side wrapper: hold the Pixel FFN-kernel lock, run one probe job.
# usage: run1.sh TAG MODEL INPUT CASES WARMUP TIMED METRIC_CALLS HOLD_S   (ODPM=1 env -> energy counters, needs root)
PD=/data/local/tmp/s43-q32-tpu
exec 9>>/data/local/tmp/.s42-pixel-ffn-kernels.lock
# mksh marks exec-redirected fds close-on-exec; "9>&9" re-opens fd 9 inheritable for flock and the probe
flock -n 9 9>&9 || { echo "LOCK_BUSY"; exit 73; }
cd $PD || exit 72
export LD_LIBRARY_PATH=$PD:/vendor/lib64
if [ -n "$PERF" ]; then export Q32_PERF_MODE=$PERF; fi
if [ -n "$NCOPIES" ]; then export Q32_NCOPIES=$NCOPIES; fi
if [ -n "$MINAVAIL" ]; then export Q32_MIN_AVAIL_MB=$MINAVAIL; fi
if [ -n "$ODPM" ]; then export Q32_ODPM=/sys/bus/iio/devices/iio:device0/energy_value; fi
echo "run1 tag=$1 uid=$(id -u) boot=$(cat /proc/sys/kernel/random/boot_id) temp=$(cat /sys/class/power_supply/battery/temp 2>/dev/null)"
exec ./q32_tpu_probe $PD/$2 $PD $PD/$3 $4 $PD/out.$1 $5 $6 $7 $8 9>&9
