#!/system/bin/sh
# second pass: finer T=1 fractions (two interleaved repeats) + T=4 NPU fallback
D=/data/local/tmp/s43-dual-ffn-worker-20260923
chmod 755 $D/*
export S43_FFN_SECONDARY_MAX_TOKENS=1
export S43_FFN_DUAL_LOG_PERIOD=1
sh $D/phone_sweep.sh $D 0-3 8704 2176 20 "1" "0 0.05 0.1 0.15 0.2 0.3" $D/sweep2a
sh $D/phone_sweep.sh $D 0-3 8704 2176 20 "1" "0.3 0.2 0.15 0.1 0.05 0" $D/sweep2b
sh $D/phone_sweep.sh $D 0-3 8704 2176 20 "1 4" "0 0.15" $D/sweep2c
