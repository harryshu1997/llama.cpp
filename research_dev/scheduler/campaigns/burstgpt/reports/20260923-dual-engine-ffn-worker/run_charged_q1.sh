#!/system/bin/sh
# CHARGED diagnostic 2: one block per layer (quantum 8704) to test the GPU launch-count hypothesis
D=/data/local/tmp/s43-dual-ffn-worker-charged-20260923
chmod 755 $D/*
export S43_FFN_DUAL_LOG_PERIOD=1
S=$D/phone_sweep_charged.sh
sh $S $D 0-3 8704 8704 20 "1" "0 0.1 0.15 0.2 0.25 0.3" $D/c_q1_r1
sh $S $D 0-3 8704 8704 20 "1" "0.3 0.25 0.2 0.15 0.1 0" $D/c_q1_r2
sh $S $D 0-3 8704 8704 20 "1" "0.2 0.3 0 0.15 0.25 0.1" $D/c_q1_r3
