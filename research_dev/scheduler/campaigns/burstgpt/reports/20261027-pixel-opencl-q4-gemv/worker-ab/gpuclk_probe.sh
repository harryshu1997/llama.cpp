#!/system/bin/sh
# As root: GPU OPP table, clock during the worker replay, and the microbench at a capped clock.
G=/sys/class/devfreq/34f00000.gpu0
echo "avail: $(cat $G/available_frequencies)"; echo "min=$(cat $G/min_freq) max=$(cat $G/max_freq) cur=$(cat $G/cur_freq)"
echo "memss: min=$(cat /sys/class/devfreq/memss_freq/min_freq) max=$(cat /sys/class/devfreq/memss_freq/max_freq) cur=$(cat /sys/class/devfreq/memss_freq/cur_freq)"
B=/data/local/tmp/s43-pixel-cpugpu-20260925-v1; T=/data/local/tmp/s43-pixel-cl-ab; ART=sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718
(taskset 40 $B/pixel-ffn-replay --port 7075 --artifact $ART --layers 18,19,20,21,22,23 --columns 17408 --max-tokens 4 --input-dir $B/inputs --out $T/runs/clk --segment burst1:1:300:5:0:0 > /dev/null 2>&1) &
sleep 0.5; i=0; while [ $i -lt 30 ]; do cat $G/cur_freq; usleep 50000; i=$((i+1)); done | sort | uniq -c | sed 's/^/ worker-run gpu cur_freq samples: /'
wait
echo "--- microbench q4k_tile2 with max_freq capped to min (1094 MHz)"
OLDMAX=$(cat $G/max_freq); echo 1094000000 > $G/max_freq
cd /data/local/tmp/s43-pixel-cl && taskset 80 ./clgemv clgemv.cl q4k_tile2 17408 5120 30 16 128 2>&1 | tail -1 | cut -c1-160
echo $OLDMAX > $G/max_freq; echo "restored max=$(cat $G/max_freq)"
echo "--- microbench again uncapped, sampling clock"
(taskset 80 ./clgemv clgemv.cl q4k_tile2 17408 5120 200 16 128 2>&1 | tail -1 | cut -c1-160) &
sleep 0.3; i=0; while [ $i -lt 20 ]; do cat $G/cur_freq; usleep 20000; i=$((i+1)); done | sort | uniq -c | sed 's/^/ microbench gpu cur_freq samples: /'
wait
