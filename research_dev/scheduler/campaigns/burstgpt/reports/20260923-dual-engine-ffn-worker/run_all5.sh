#!/system/bin/sh
# fifth pass (coordinator request): f = 0, 0.15, 0.2, 0.25 at tokens=1 with the
# default tokens==1 gate (NPU copy loaded), two interleaved repeats, then no-copy check
D=/data/local/tmp/s43-dual-ffn-worker-20260923
chmod 755 $D/*
export S43_FFN_DUAL_LOG_PERIOD=1
dumpsys battery | grep -E "level|status"
sh $D/phone_sweep.sh $D 0-3 8704 2176 20 "1" "0 0.15 0.2 0.25" $D/sweep5a
sh $D/phone_sweep.sh $D 0-3 8704 2176 20 "1" "0.25 0.2 0.15 0" $D/sweep5b
export S43_FFN_SECONDARY_MAX_TOKENS=0
sh $D/phone_sweep.sh $D 0-3 8704 2176 20 "1" "0.15 0" $D/sweep5c
dumpsys battery | grep -E "level|status"
