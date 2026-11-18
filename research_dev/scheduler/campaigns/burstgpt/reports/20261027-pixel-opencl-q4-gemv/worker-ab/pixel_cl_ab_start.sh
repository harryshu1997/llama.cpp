#!/system/bin/sh
# Start the production CPU worker (port 7074) and the OpenCL worker (port 7075) on layers 18-23 of the 1-23 shard.
B=/data/local/tmp/s43-pixel-cpugpu-20260925-v1
N=${5:-/data/local/tmp/s43-pixel-opencl-20261027-v1}
SH=/data/local/tmp/s43-ws12-pixel-v1/QWEN_PACKED_1_23.ffn.gguf
T=/data/local/tmp/s43-pixel-cl-ab
ART=sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718
LAYERS=${1:-18-23}
mkdir -p $T/runs; cd $T
pkill -f llama-ffn-split-worker; sleep 1
rm -f $T/cpu.log $T/cl.log
E="S42_PIXEL_CPU_MASK=fc S42_PIXEL_CPU_PAIR_DOT=1 S42_PIXEL_CPU_POOL=1 S42_PIXEL_CPU_QUANT_RESIDUAL=1 S42_PIXEL_CPU_ROW_CHUNK=64 S42_PIXEL_CPU_ROW_PROFILE=0 S42_PIXEL_CPU_THREADS=6 S42_PIXEL_FUSED_RESIDUAL=1 S42_PIXEL_PACKED_WEIGHTS=1 S43_PIXEL_CPU_BATCH_PAIR=1 S43_PIXEL_CPU_POLL=100 S43_PIXEL_UCLAMP_MIN=1024 $(echo $3 | tr , " ")"
A="-m $SH --artifact-sha256 $ART --layers $LAYERS --columns 17408 --column-quantum 4352 --backend CPU --bind 127.0.0.1 --f16-io --max-tokens ${6:-4} --max-requests 0"
if [ "$2" != "clonly" ]; then
nohup setsid env LD_LIBRARY_PATH=$B $E $B/llama-ffn-split-worker $A --port 7074 > $T/cpu.log 2>&1 < /dev/null &
echo $! > $T/cpu.pid
fi
nohup setsid $4 env LD_LIBRARY_PATH=$N $E S43_PIXEL_OPENCL=1 $N/llama-ffn-split-worker $A --port 7075 > $T/cl.log 2>&1 < /dev/null &
echo $! > $T/cl.pid
i=0; while [ $i -lt 180 ]; do
  n=$(cat $T/cpu.log $T/cl.log 2>/dev/null | grep -c "ready backend=")
  want=2; [ "$2" = "clonly" ] && want=1
  [ "$n" -ge "$want" ] && break
  kill -0 $(cat $T/cl.pid) 2>/dev/null || { echo "OPENCL WORKER DIED"; break; }
  sleep 1; i=$((i+1))
done
echo "=== cpu.log"; grep -h "S43PIXEL\|ready backend\|fail\|FAIL\|error" $T/cpu.log | cut -c1-220
echo "=== cl.log"; grep -h "S43PIXEL\|ready backend\|fail\|FAIL\|error\|Error" $T/cl.log | cut -c1-220
exit 0
