#!/system/bin/sh
# CHARGED pass 3: fused GPU secondary (one GPU matmul chain per layer), quantum 2176 (4 NPU blocks)
D=/data/local/tmp/s43-dual-ffn-worker-charged-20260923
chmod 755 $D/*
export S43_FFN_DUAL_LOG_PERIOD=1
S=$D/phone_sweep_charged.sh
sh $S $D 0-3 8704 2176 20 "1" "0 none 0.1 0.15 0.2 0.25" $D/c_fused_r1
sh $S $D 0-3 8704 2176 20 "1" "0.25 0.2 0.15 0.1 none 0" $D/c_fused_r2
sh $S $D 0-3 8704 2176 20 "1" "0.15 0.2 0 0.25 0.1 none" $D/c_fused_r3
sh $S $D 0-3 8704 2176 20 "1" "0.1 0.25 none 0.15 0 0.2" $D/c_fused_r4
# partial runtime columns (3 of 4 blocks): strided GPU down view correctness
S43_REQ_COLUMNS=6528 sh $S $D 0-3 8704 2176 20 "1" "0 0.15" $D/c_fused_partial
