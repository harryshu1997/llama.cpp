# Pixel Tensor SDK and FFN TPU investigation

Installation and real-weight FFN execution PASS: 528 FFN calls / 1056 rows, plus 24 exact ADD calls. Initial speed goal FAIL relative to earlier tuned CPU results; a fresh CPU comparison has not been run. BF16 speed improvement FAIL. Qualification is complete; production integration is unverified.

## Installation and proof

The supplied SDK v2.0 archive is installed in `research_dev/TPU_SDK/installed/v2.0-20260902`, using isolated Python 3.12 and LiteRT 2.2.0. No system packages changed. The compiler plugin loads through the LiteRT wheel adapter. `pip check` PASS; the adapted Android probe builds with `-Wall -Wextra -Werror`. See [installation instructions](../../../../../TPU_SDK/README.md).

The ADD and FFN graphs compile all 1/1 and 6/6 operations respectively. Every measured model has one dispatch node, which the phone delegates to Google Tensor NPU with CPU/GPU execution disabled. The vendor runtime is `libedgetpu_litert.so`. Model buffers are AHardwareBuffer (type 2). Runtime registration messages mentioning CPU do not mean CPU execution.

## Initial physical arms

All latency entries below are medians in milliseconds per call/batch. FFN arms have 88 calls (80 warm), ADD has 24 (20 warm). Invocation is synchronous LiteRT plus vendor execution, not a pure hardware-kernel timestamp.

| Arm | Status | TPU invoke | Phone worker | USB/ADB round trip | Max relative L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| [run-add-1](physical/pixel10pro-tpu-sdk-1/run-add-1/RESULT.json) | PASS | 1.371 | 1.712 | 4.616 | 0.000000000 |
| [run-ffn512-2](physical/pixel10pro-tpu-sdk-1/run-ffn512-2/RESULT.json) | PASS | 3.351 | 4.021 | 7.523 | 0.000571125 |
| [run-ffn17408b1-1](physical/pixel10pro-tpu-sdk-1/run-ffn17408b1-1/RESULT.json) | PASS | 33.794 | 35.007 | 40.832 | 0.000547193 |
| [run-ffn17408b4-1](physical/pixel10pro-tpu-sdk-1/run-ffn17408b4-1/RESULT.json) | PASS | 32.906 | 36.686 | 47.593 | 0.000547222 |

All repeated inputs produce identical repeated outputs. FFN outputs pass the 1 percent per-row relative-L2 bound, but are not bit-identical to the FP32 reference. First run-ffn512-1 rejected an occupied/TIME_WAIT port before launching a worker; run-ffn512-2 used a fresh port and passed.

## Confirmed measurements and precision comparison

Warm medians pooled across both FP16 full-width processes (160 warm calls per batch shape); tail 512 and BF16 each have one process (80 warm calls). Each process excludes its first 8 calls. All saved outputs were independently rechecked per row.

| FFN mode | Rows/call | TPU invoke ms | Phone worker ms | USB/ADB ms | Worker ms/row | Max relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tail 512 FP16 | 1 | 3.351 | 4.021 | 7.523 | 4.021 | 0.0571% |
| Full 17408 FP16 | 1 | 33.500 | 34.671 | 40.646 | 34.671 | 0.0547% |
| Full 17408 FP16 | 4 | 31.806 | 35.998 | 47.213 | 9.000 | 0.0547% |
| Full 17408 BF16 | 1 | 33.854 | 34.952 | 41.537 | 34.952 | 0.4687% |

FP16 full-width B1 worker repeats 35.007/34.349 ms; B4 repeats 36.686/35.366 ms.
Repeated output bytes are identical across the separate FP16 processes.
BF16 passes the 1 percent bound but error is about 8.56x FP16 and its worker
34.952 ms provides no benefit over pooled FP16 34.671 ms. Retain FP16 for any
next experiment. Four-row batching amortizes the same weights, yielding
9.000 ms/row worker time, but host/phone buffer handling and transfer remain.

Full FP16 median buffer-handling intervals are 1.172 ms (B1) and 3.864 ms (B4),
with 5.839/10.758 ms outside the worker. These are medians of individual
intervals, so their sums need not equal the median round trip. The latter
includes USB/ADB, socket handling and host scheduling. The probe uses FP32
I/O (20,480 bytes each direction per row), whereas production can use F16.
B4 has a 71.739 ms worker outlier and 87.599 ms RPC outlier; this is not a
sustained tail-latency qualification.

Useful matrix throughput from invocation wall time is 15.963 GFLOP/s at B1
and 67.255 GFLOP/s at B4. Dividing one F16 weight pass by that interval gives
15.963/16.814 GB/s; these are nominal ratios, not physical DRAM counters.
The nearly unchanged invocation time for 4x arithmetic is consistent with
weight movement or fixed scheduling dominating these shapes, but does not
prove the hardware bottleneck.

Earlier CPU F16 worker means were about 18.291 ms at B1 and 21.697 ms at B4;
the corrected packed CPU B1 result was 10.792 ms. Those used different runs
and six-layer samples, so they are historical context, not a matched
CPU-vs-TPU comparison or a new energy claim. This investigation does not
support replacing the tuned CPU path with the current TPU implementation.

SDK installation and actual FFN TPU feasibility PASS. Numerical repeat and
all-NPU dispatch checks PASS. Demonstrated advantage over existing CPU
measurements FAIL. BF16 speed improvement FAIL. Full-model token equality,
server overlap, energy, and production integration remain NOT VERIFIED.

Cleanup PASS: same boot, Pixel lock free, no qualification worker or ADB
forward, no pending job. Battery endpoint 100 percent / 28.9 C; no temperature
trace or power measurement. No phone reboot, root change, or forced kill.
The initial full B1 run overlapped a host-side model-file transfer; its
separate repeat ran without that transfer. Neither run used host inference.

Source/SDK/dependency hashes, exact invocation commands, compiler coverage,
raw outputs and retained failed startup are in
[physical/pixel10pro-tpu-sdk-1](physical/pixel10pro-tpu-sdk-1).
[Aggregate JSON](PIXEL_TPU_SDK_RESULTS.json) is reproduced by
`summarize_pixel_tpu_sdk.py` using the saved vectors and outputs.

## Shapes, weights and reference

Layer 18 actual deployed Qwen F16 weights; embedding 5120, full intermediate width 17408. Tail 512 begins at column 16896. Operations: gate/up fully connected, sigmoid(gate), gate*sigmoid(gate), multiply by up, down fully connected. No bias/residual. The exporter expands F16 constants exactly into an FP32 source graph, then the compiler explicitly uses `half`. Source reference and LiteRT CPU agree within 3.17e-6 relative L2.

Inputs are eight deterministic synthetic cases from the existing Gaussian qualification vector plus scaled/sign/random variants. They are not captured production activations. Each batch 4 case rotates four of these vectors. CPU reference arrays and all inputs/outputs are retained. Source shardsha `dd705a75a3047ead41dbc01984afebc23440e2ea575b25af5c5b22252c69f4c3`; distinct parent/protocol artifactsha `d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718`.

## Offload implications and gaps

The SDK removes the custom-FFN compilation blocker. For the current probe, weights are static in an AOT model per layer/slice/batch shape. Full-width compiled model size is 535 MB per shape. Six full layers would contain roughly 3.21 GB of F16 constants before other runtime memory; multi-layer residency and shape-sharing have not been measured.

The production FFN v6 protocol includes artifact identity, layer, column count, tokens, payload size and hash; the probe uses a smaller diagnostic protocol and FP32 I/O. Integration still requires selecting the right persistent compiled model, retaining these v6 checks, handling F16 I/O, and qualifying all layers/column/batch combinations. The server can issue phone work while its own FFN share runs, but this new TPU probe has not been connected to that path.

Full-model token equality, server overlap/latency, phone/host energy, thermal endurance, and multi-phone scheduler integration are NOT VERIFIED. No performance conclusion should be inferred from the old tiny ADD timing. The reported intervals do not establish peak compute or DRAM bandwidth.

All runs are finite and take the Pixel-specific lock, use ADB port 5037, remove their own forward, and exit normally. No other phone worker is killed. Raw evidence and commands: [physical/pixel10pro-tpu-sdk-1](physical/pixel10pro-tpu-sdk-1).

The follow-up [runtime audit](PIXEL_TPU_RUNTIME_AUDIT.md) measures vendor-reported hardware time separately: B1/B4 means 27.337/27.419 ms, plus 5.680/5.625 ms elsewhere inside invocation. It uses 48 additional diagnostic calls and does not replace the uninstrumented latency results above.
