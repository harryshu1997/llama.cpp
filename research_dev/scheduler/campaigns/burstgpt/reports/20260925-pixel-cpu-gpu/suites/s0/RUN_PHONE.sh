#!/system/bin/sh
PH=/data/local/tmp/s43-pixel-cpugpu-20260925-v1
RUN=$PH/runs/s0-smoke
cd $PH || exit 2
exec 9>/data/local/tmp/.s42-pixel-ffn-kernels.lock
flock -n 9 9>&9 || { echo 'FAIL phone lock busy'; exit 3; }
mkdir $RUN || exit 4
ps -A -o PID,ARGS > $RUN/PROCESSES_BEFORE.txt
if grep -q '[l]lama-ffn' $RUN/PROCESSES_BEFORE.txt; then echo 'FAIL existing worker'; exit 5; fi
cat /proc/sys/kernel/random/boot_id > $RUN/BOOT_BEFORE.txt
snapshot() {
    dumpsys battery > "$1/BATTERY_$2.txt" 2>&1
    for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null)"; done > "$1/THERMAL_$2.txt" 2>&1
    for f in /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq /sys/class/devfreq/34f00000.gpu0/cur_freq; do echo "$f $(cat $f)"; done > "$1/FREQ_$2.txt" 2>&1
}
run_arm() {
    name=$1; calls=$2; worker=$3; client=$4
    mkdir $RUN/$name || return 7
    snapshot $RUN/$name BEFORE
    date +%s.%N > $RUN/$name/START.txt
    sh -c "exec $worker" > $RUN/$name/worker.log 2>&1 &
    pid=$!
    echo $pid > $RUN/$name/PID.txt
    ready=0
    for attempt in $(seq 1 600); do
        if grep -q '\[ffn-worker\] ready backend=' $RUN/$name/worker.log; then ready=1; break; fi
        if ! kill -0 $pid 2>/dev/null; then echo "FAIL $name startup"; wait $pid; echo $? > $RUN/$name/WORKER_EXIT.txt; return 8; fi
        sleep 0.5
    done
    if [ "$ready" != 1 ]; then echo "FAIL $name readiness"; kill -TERM $pid; wait $pid; return 9; fi
    for t in /proc/$pid/task/*; do echo "$(cat $t/comm) $(grep Cpus_allowed_list $t/status) $(grep -E 'uclamp' $t/sched 2>/dev/null | tr -s ' ' | tr '\n' ' ')"; done > $RUN/$name/THREADS.txt 2>&1
    sh -c "exec $client" > $RUN/$name/client.log 2>&1
    cstatus=$?
    echo $cstatus > $RUN/$name/CLIENT_EXIT.txt
    if [ "$cstatus" != 0 ]; then
        sleep 2
        # client gone: the worker is idle in accept(), not in a call
        if kill -0 $pid 2>/dev/null; then kill -TERM $pid; fi
        wait $pid; echo $? > $RUN/$name/WORKER_EXIT.txt
        echo "FAIL $name client $cstatus"; return 10
    fi
    wait $pid
    wstatus=$?
    echo $wstatus > $RUN/$name/WORKER_EXIT.txt
    date +%s.%N > $RUN/$name/END.txt
    snapshot $RUN/$name AFTER
    if [ "$wstatus" != 0 ]; then echo "FAIL $name worker $wstatus"; return 11; fi
    echo "PASS $name"
}
snapshot $RUN SUITE_BEFORE
run_arm cpu 72 'env LD_LIBRARY_PATH=/data/local/tmp/s43-pixel-cpugpu-20260925-v1 S42_PIXEL_CPU_MASK=fc S42_PIXEL_CPU_PAIR_DOT=1 S42_PIXEL_CPU_POOL=1 S42_PIXEL_CPU_QUANT_RESIDUAL=1 S42_PIXEL_CPU_ROW_CHUNK=64 S42_PIXEL_CPU_ROW_PROFILE=0 S42_PIXEL_CPU_THREADS=6 S42_PIXEL_FUSED_RESIDUAL=1 S42_PIXEL_PACKED_WEIGHTS=1 S43_PIXEL_UCLAMP_MIN=1024 /data/local/tmp/s43-pixel-cpugpu-20260925-v1/llama-ffn-split-worker -m /data/local/tmp/s43-pixel-cpugpu-20260925-v1/QWEN_PACKED.ffn.gguf --artifact-sha256 sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718 --layers 18,19,20,21,22,23 --columns 17408 --column-quantum 4352 --backend CPU --port 27190 --bind 127.0.0.1 --f16-io --max-tokens 4 --max-requests 72' '/data/local/tmp/s43-pixel-cpugpu-20260925-v1/pixel-ffn-replay --port 27190 --artifact sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718 --layers 18,19,20,21,22,23 --columns 17408 --max-tokens 4 --input-dir /data/local/tmp/s43-pixel-cpugpu-20260925-v1/inputs --out /data/local/tmp/s43-pixel-cpugpu-20260925-v1/runs/s0-smoke/cpu/replay --freq-files /sys/devices/system/cpu/cpufreq/policy2/scaling_cur_freq,/sys/devices/system/cpu/cpufreq/policy5/scaling_cur_freq,/sys/devices/system/cpu/cpufreq/policy7/scaling_cur_freq,/sys/class/devfreq/34f00000.gpu0/cur_freq --segment burst-m1:1:2:1:0:0 --segment prod-m1:1:1:0:10000:100000 --segment burst-m2:2:2:1:0:0 --segment prod-m2:2:1:0:10000:100000 --segment burst-m4:4:2:1:0:0 --segment prod-m4:4:1:0:10000:100000' || exit $?
sleep 3
run_arm dual25 72 'env LD_LIBRARY_PATH=/data/local/tmp/s43-pixel-cpugpu-20260925-v1 S42_PIXEL_CPU_MASK=fc S42_PIXEL_CPU_PAIR_DOT=1 S42_PIXEL_CPU_POOL=1 S42_PIXEL_CPU_QUANT_RESIDUAL=1 S42_PIXEL_CPU_ROW_CHUNK=64 S42_PIXEL_CPU_ROW_PROFILE=0 S42_PIXEL_CPU_THREADS=6 S42_PIXEL_FUSED_RESIDUAL=1 S42_PIXEL_GPU_HOST_MASK=03 S42_PIXEL_PACKED_WEIGHTS=1 S43_FFN_DUAL_LOG_PERIOD=1 S43_FFN_SECONDARY_BACKEND=Vulkan0 S43_PIXEL_GPU_TRAILING_COLUMNS=4352 /data/local/tmp/s43-pixel-cpugpu-20260925-v1/llama-ffn-split-worker -m /data/local/tmp/s43-pixel-cpugpu-20260925-v1/QWEN_PACKED.ffn.gguf --artifact-sha256 sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718 --layers 18,19,20,21,22,23 --columns 17408 --column-quantum 4352 --backend CPU --port 27190 --bind 127.0.0.1 --f16-io --max-tokens 4 --max-requests 72' '/data/local/tmp/s43-pixel-cpugpu-20260925-v1/pixel-ffn-replay --port 27190 --artifact sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718 --layers 18,19,20,21,22,23 --columns 17408 --max-tokens 4 --input-dir /data/local/tmp/s43-pixel-cpugpu-20260925-v1/inputs --out /data/local/tmp/s43-pixel-cpugpu-20260925-v1/runs/s0-smoke/dual25/replay --freq-files /sys/devices/system/cpu/cpufreq/policy2/scaling_cur_freq,/sys/devices/system/cpu/cpufreq/policy5/scaling_cur_freq,/sys/devices/system/cpu/cpufreq/policy7/scaling_cur_freq,/sys/class/devfreq/34f00000.gpu0/cur_freq --segment burst-m1:1:2:1:0:0 --segment prod-m1:1:1:0:10000:100000 --segment burst-m2:2:2:1:0:0 --segment prod-m2:2:1:0:10000:100000 --segment burst-m4:4:2:1:0:0 --segment prod-m4:4:1:0:10000:100000' || exit $?
sleep 3
run_arm gpu 72 'env LD_LIBRARY_PATH=/data/local/tmp/s43-pixel-cpugpu-20260925-v1 S42_PIXEL_PACKED_WEIGHTS=1 /data/local/tmp/s43-pixel-cpugpu-20260925-v1/llama-ffn-split-worker -m /data/local/tmp/s43-pixel-cpugpu-20260925-v1/QWEN_PACKED.ffn.gguf --artifact-sha256 sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718 --layers 18,19,20,21,22,23 --columns 17408 --column-quantum 4352 --backend Vulkan0 --port 27190 --bind 127.0.0.1 --f16-io --max-tokens 4 --max-requests 72' '/data/local/tmp/s43-pixel-cpugpu-20260925-v1/pixel-ffn-replay --port 27190 --artifact sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718 --layers 18,19,20,21,22,23 --columns 17408 --max-tokens 4 --input-dir /data/local/tmp/s43-pixel-cpugpu-20260925-v1/inputs --out /data/local/tmp/s43-pixel-cpugpu-20260925-v1/runs/s0-smoke/gpu/replay --freq-files /sys/devices/system/cpu/cpufreq/policy2/scaling_cur_freq,/sys/devices/system/cpu/cpufreq/policy5/scaling_cur_freq,/sys/devices/system/cpu/cpufreq/policy7/scaling_cur_freq,/sys/class/devfreq/34f00000.gpu0/cur_freq --segment burst-m1:1:2:1:0:0 --segment prod-m1:1:1:0:10000:100000 --segment burst-m2:2:2:1:0:0 --segment prod-m2:2:1:0:10000:100000 --segment burst-m4:4:2:1:0:0 --segment prod-m4:4:1:0:10000:100000' || exit $?
sleep 3
snapshot $RUN SUITE_AFTER
cat /proc/sys/kernel/random/boot_id > $RUN/BOOT_AFTER.txt
ps -A -o PID,ARGS > $RUN/PROCESSES_AFTER.txt
if grep -q '[l]lama-ffn' $RUN/PROCESSES_AFTER.txt; then echo 'FAIL remaining worker'; exit 12; fi
cmp $RUN/BOOT_BEFORE.txt $RUN/BOOT_AFTER.txt || exit 13
echo PASS > $RUN/DONE.txt
echo 'PASS all arms'
