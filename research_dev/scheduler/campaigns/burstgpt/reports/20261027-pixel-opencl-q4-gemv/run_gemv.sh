#!/bin/bash
# run_gemv.sh: push clgemv + kernels to the Pixel and run the variants for the Qwen FFN shapes; samples the GPU clock meanwhile.
S=5A040DLCH004ES; D=/data/local/tmp/s43-pixel-cl
adb -s $S shell "mkdir -p $D" >/dev/null
adb -s $S push /tmp/pxcl/clgemv /tmp/pxcl/clgemv.cl $D/ >/dev/null
adb -s $S shell "chmod 755 $D/clgemv"
adb -s $S shell "su -c 'while true; do cat /sys/class/devfreq/34f00000.gpu0/cur_freq; sleep 0.2; done'" > /tmp/pxcl/gpufreq.txt 2>/dev/null &
SAMPLER=$!
for shape in "17408 5120" "5120 17408"; do
  for v in f16_rows q4_global q4_global_interleaved q4_image; do
    adb -s $S shell "cd $D && ./clgemv clgemv.cl $v $shape 30 ${R:-8} ${WG:-128}" 2>&1 | tail -3
  done
done
kill $SAMPLER 2>/dev/null
echo "gpu clock samples: $(wc -l < /tmp/pxcl/gpufreq.txt), max $(sort -n /tmp/pxcl/gpufreq.txt | tail -1), p50 $(sort -n /tmp/pxcl/gpufreq.txt | awk '{a[NR]=$1} END{print a[int(NR/2)+1]}')"
