#!/system/bin/sh
# fourth pass: repeatability of f=0.15 vs off (fresh worker each), and the T=4 NPU fallback
D=/data/local/tmp/s43-dual-ffn-worker-20260923
chmod 755 $D/*
export S43_FFN_DUAL_LOG_PERIOD=1
export S43_FFN_SECONDARY_MAX_TOKENS=1
for r in 1 2 3 4; do
    sh $D/phone_sweep.sh $D 0-3 8704 2176 20 "1" "0 0.15 0.2" $D/sweep4_r$r
done
for r in 1 2; do
    sh $D/phone_sweep.sh $D 0-3 8704 2176 20 "4" "0 0.15" $D/sweep4_t4_r$r
done
