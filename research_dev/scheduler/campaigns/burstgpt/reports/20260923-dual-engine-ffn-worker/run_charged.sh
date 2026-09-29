#!/system/bin/sh
# CHARGED pass: 3 reps of 7 configs, order rotated per rep, then tokens=4 off/0.15 x2
D=/data/local/tmp/s43-dual-ffn-worker-charged-20260923
chmod 755 $D/*
export S43_FFN_DUAL_LOG_PERIOD=1
S=$D/phone_sweep_charged.sh
sh $S $D 0-3 8704 2176 20 "1" "0 none 0.1 0.15 0.2 0.25 0.3" $D/c_r1
sh $S $D 0-3 8704 2176 20 "1" "0.3 0.25 0.2 0.15 0.1 none 0" $D/c_r2
sh $S $D 0-3 8704 2176 20 "1" "0.15 0.2 0.25 0.3 0 none 0.1" $D/c_r3
sh $S $D 0-3 8704 2176 20 "1" "0.2 0.1 0 0.3 0.15 none 0.25" $D/c_r4
sh $S $D 0-3 8704 2176 20 "4" "0 0.15" $D/c_t4_r1
sh $S $D 0-3 8704 2176 20 "4" "0.15 0" $D/c_t4_r2
