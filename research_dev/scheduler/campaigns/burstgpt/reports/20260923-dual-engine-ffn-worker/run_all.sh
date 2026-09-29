#!/system/bin/sh
# smoke test, then the full sweep only if both smoke configs produced outputs
D=/data/local/tmp/s43-dual-ffn-worker-20260923
chmod 755 $D/*
sh $D/phone_sweep.sh $D 0-3 8704 2176 3 1 "0 0.2" $D/smoke
if [ -s $D/smoke/out_T1_f0.bin ] && [ -s $D/smoke/out_T1_f0.2.bin ]; then
    sh $D/phone_sweep.sh $D 0-3 8704 2176 20 "1 4" "0 0.1 0.2 0.3 0.4" $D/sweep
else
    tail -30 $D/smoke/worker_T1_f0.2.log
fi
