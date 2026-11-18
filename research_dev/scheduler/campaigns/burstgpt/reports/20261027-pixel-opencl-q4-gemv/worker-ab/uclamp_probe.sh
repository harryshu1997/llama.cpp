#!/system/bin/sh
# As root (core 7 NOT pinned): cllaunch with uclamp; worker with uclamp; cpu7 clock samples in both.
P=/sys/devices/system/cpu/cpufreq/policy7; N=/data/local/tmp/s43-pixel-opencl-20261027-v1; T=/data/local/tmp/s43-pixel-cl-ab
echo "cpu7 min=$(cat $P/scaling_min_freq)"
pkill -f llama-ffn-split-[w]orker; sleep 1
echo "=== cllaunch, uclamp 1024, unpinned"
(cd $N && CLL_UCLAMP=1024 ./cllaunch $N/pixel_opencl_kernels.cl 100) &
sleep 1.5; i=0; while [ $i -lt 40 ]; do cat $P/scaling_cur_freq; usleep 20000; i=$((i+1)); done | sort -n | uniq -c | sed 's/^/ cpu7 during cllaunch: /'
wait
echo "=== cllaunch, uclamp 1024, XQ_SPACE=__global"
(cd $N && CLL_UCLAMP=1024 S43_PIXEL_CL_BUILD_OPTS="-D XQ_SPACE=__global" ./cllaunch $N/pixel_opencl_kernels.cl 100) 2>&1 | grep "^[CFGK]"
echo "=== worker, uclamp 1024 (default), profile"
sh /data/local/tmp/pixel_cl_ab_start.sh 18-23 clonly S43_PIXEL_CL_PROFILE=1 > /dev/null
grep -h "S43PIXELOPENCL ready" $T/cl.log | cut -c1-160
(sh /data/local/tmp/pixel_cl_quick.sh > /dev/null) &
sleep 0.5; i=0; while [ $i -lt 40 ]; do cat $P/scaling_cur_freq; usleep 20000; i=$((i+1)); done | sort -n | uniq -c | sed 's/^/ cpu7 during replay: /'
wait
grep -h S43PIXELCLPROF $T/cl.log | sed -n "3p;6p" | cut -c1-130; grep -h S43PIXELCLPROF $T/cl.log | tail -1 | cut -c1-130
echo "=== worker, uclamp 1024, no profile, full replay"
sh /data/local/tmp/pixel_cl_ab_start.sh 18-23 clonly S43_PIXEL_CL_PROFILE=0 > /dev/null
sh /data/local/tmp/pixel_cl_ab_replay.sh | grep done
