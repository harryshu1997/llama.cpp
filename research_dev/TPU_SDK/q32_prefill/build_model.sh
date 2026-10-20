#!/bin/bash
# Export + compile one full-layer model. usage: build_model.sh NAME FMT M PRECISION [compile flags...]
# Writes $S/models/NAME/{src/ffn.tflite, compiled/ffn_Google_Tensor_G5.tflite} and removes the source tflite
# after a successful compile (sources are reproducible from the Q4_0 bytes).
set -eu
S=/mnt/data_h16t/zs89458/claude-scratch/q32-tpu
Q=/home/myid/zs89458/Documents/llama.cpp-release/research_dev/TPU_SDK/q32_prefill
RP=/home/myid/zs89458/Documents/llama.cpp-release/research_dev/TPU_SDK/installed/v2.0-20260902/run-python
export TMPDIR=$S/tmpdir
NAME=$1; FMT=$2; M=$3; PREC=$4; shift 4
D=$S/models/$NAME
mkdir -p $D
cd $Q
if [ ! -f $D/src/ffn.tflite ] && [ ! -f $D/compiled/RESULT.json ]; then
  $RP export_q32_ffn.py $S/blk11_shard_gate_up_down.q4_0 $D/src --weights-format $FMT --batch $M > $D/export.log 2>&1
fi
rm -rf $D/compiled
/usr/bin/time -v $RP compile_q32.py $D/src/ffn.tflite $D/compiled --precision $PREC "$@" > $D/compile.log 2>&1 || { echo "$NAME FAIL"; exit 1; }
cp $D/src/MODEL.json $D/MODEL.json
rm -f $D/src/ffn.tflite
rm -rf $D/compiled/intermediate
echo "$NAME PASS $(stat -c %s $D/compiled/ffn_Google_Tensor_G5.tflite) $(grep -E 'Maximum resident|Elapsed' $D/compile.log | tr -s ' ' | tr '\n' ' ')"
