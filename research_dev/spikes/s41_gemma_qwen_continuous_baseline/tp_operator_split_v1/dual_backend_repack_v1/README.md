# OP15 one-time Q4 repack screen

This benchmark gives disjoint Gemma4 FFN columns to HTP and Adreno. Each
backend converts its Q4 weights once during preparation and reuses them for
all warmup and measured iterations.

The F32 input and both partial outputs live in one Hexagon rpcmem allocation.
OpenCL imports the input and its output region by DMA-BUF, so the timed dual
path does not upload two inputs or download the GPU result. The phone CPU adds
the two partial outputs after both device fences.

The control runs the same combined columns on HTP. This is a one-layer screen,
not full-model integration.

See [RESULTS.md](RESULTS.md) for the native-Q4 result and
[RESULTS_XMEM.md](RESULTS_XMEM.md) for the Q4-to-F16 xmem result.

```sh
./build.sh
./run_op15_sweep.sh
./run_op15_xmem_sweep.sh
```

The default sweep keeps 9,664 phone columns and varies the GPU share over
batch sizes 1, 8, 32, and 128. `DUAL_REPACK_RESULT` reports one-time packing,
resident weight bytes, solo legs, concurrent wall time, merge time,
correctness, and speedup against the intact HTP control.

The xmem sweep reconstructs F16 GPU weights from the same Q4 values once,
primes the opt-in Adreno weight and image caches once, and tests token batches
16 through 128. Xmem does not support this FFN's batch-1 decode shape.
