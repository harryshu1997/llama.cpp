#!/system/bin/sh
# CHARGED diagnostic: longer windows (100 iters = 400 calls) and 20 ms clock sampling
D=/data/local/tmp/s43-dual-ffn-worker-charged-20260923
chmod 755 $D/*
export S43_FFN_DUAL_LOG_PERIOD=1
export S43_CLOCK_PERIOD=0.02
S=$D/phone_sweep_charged.sh
sh $S $D 0-3 8704 2176 100 "1" "0.15 0 0.1 0.25" $D/c_diag_r1
sh $S $D 0-3 8704 2176 100 "1" "0 0.25 0.15 0.1" $D/c_diag_r2
sh $S $D 0-3 8704 2176 100 "1" "0.1 0.15 0.25 0" $D/c_diag_r3
