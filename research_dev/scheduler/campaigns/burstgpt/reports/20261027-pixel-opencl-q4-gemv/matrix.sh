#!/bin/bash
S=5A040DLCH004ES; D=/data/local/tmp/s43-pixel-cl
adb -s $S push /tmp/pxcl/clgemv /tmp/pxcl/clgemv.cl /tmp/pxcl/clprobe2 $D/ >/dev/null; adb -s $S shell "chmod 755 $D/clprobe2"
fmt() { sed 's/.*"variant":"\([a-z0-9_]*\)".*"R":\([0-9]*\),"WG":\([0-9]*\).*"device_ms_med":\([0-9.]*\).*"GBps_med":\([0-9.]*\).*"wall_ms_batched":\([0-9.]*\),"wall_ms_single_med":\([0-9.]*\).*"rel_l2_vs_quantized_ref":\([0-9.e+-]*\).*/\1 R=\2 WG=\3 device_ms=\4 GBps=\5 wall_batched=\6 wall_single=\7 rel_l2=\8/'; }
run() { adb -s $S shell "cd $D && $4 taskset 80 ./clgemv clgemv.cl $1 17408 5120 40 $2 $3 2>&1 | tail -1" | fmt | cut -c1-200; }
echo "=== capabilities"; adb -s $S shell "$D/clprobe2"
echo "=== A. dot instruction forms (gate/up 17408x5120)"
for k in q4_global q4_dotpacked q4_nodot q4_half q4_v2; do run $k 32 256; done
for k in q4_global q4_dotpacked q4_nodot; do run $k 16 128; done
echo "=== A2. sub-group reduction (L must equal the sub-group size)"
run q4_subgroup 2 256; run q4_subgroup 1 128; run q4_subgroup 4 256; run q4_subgroup 1 64
echo "=== B. allocation flags (stream_read and q4_v2)"
for a in copy alloc_host use_host; do echo "-- CLG_ALLOC=$a"; run stream_read 32 256 "CLG_ALLOC=$a"; run q4_v2 32 256 "CLG_ALLOC=$a"; done
echo "=== C. queue priority (single-dispatch round trip)"
run q4_global 32 256; run q4_global 32 256 "CLG_PRIO=high"; run q4_global 32 256 "CLG_PRIO=low"
echo "=== D. compiler flags"
run q4_v2 32 256 "CLG_OPTS=-cl-fast-relaxed-math"; run q4_v2 32 256 "CLG_OPTS=-cl-unsafe-math-optimizations"; run q4_global 32 256 "CLG_OPTS=-cl-fast-relaxed-math"; run q4_v2 32 256 "CLG_OPTS=-cl-opt-disable"
