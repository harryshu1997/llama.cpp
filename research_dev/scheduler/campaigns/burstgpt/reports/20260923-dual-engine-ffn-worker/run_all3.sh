#!/system/bin/sh
# third pass: CPU-cast control ("none") vs off vs dual, interleaved repeats
D=/data/local/tmp/s43-dual-ffn-worker-20260923
chmod 755 $D/*
export S43_FFN_DUAL_LOG_PERIOD=1
sh $D/phone_sweep.sh $D 0-3 8704 2176 20 "1" "0 none 0.1 0.15 0.2" $D/sweep3a
sh $D/phone_sweep.sh $D 0-3 8704 2176 20 "1" "0.2 0.15 0.1 none 0" $D/sweep3b
sh $D/phone_sweep.sh $D 0-3 8704 2176 20 "4" "0 none" $D/sweep3c
