# Pixel FFN partition concurrency review

Source/shape review PASS. Concurrent-execution speedup NOT VERIFIED; no hardware
run or kernel modification in this review.

For x of length5120 and FFN hidden width17408, split hidden channels into
four parts of4352. Part i uses Wgate_i/Wup_i of shape[4352,5120] and
Wdown_i of shape[5120,4352]. It computes

y_i = Wdown_i * (silu(Wgate_i * x) elementwise_mul (Wup_i * x))
y = y_0 + y_1 + y_2 + y_3

This is algebraically the same FFN. All parts can share one read-only input
buffer; no physical input duplication is necessary. Separate FP32 partial
outputs and a fixed-order sum preserve the current reduction structure.
Changing the reduction order can change floating-point bits. The nonlinear
activation must operate on complete gate/up dot products for each channel.

The worker already implements this four-way partition at quantum4352 in
examples/layersplit/ffn-split-worker.cpp:1048. Its graph contains independent
per-part gate/up/activation/down paths plus a sum. The Vulkan backend uses one
compute queue and inserts barriers for read/write overlap, including buffer
reuse. One queue alone does not prove serial execution: independent Vulkan
commands are allowed to overlap. Conversely multiple queues do not guarantee
additional hardware execution capacity. Actual overlap remains unmeasured.

At the current eight-output-row setting, each gate/up dispatch already has
544 workgroups and each partial-down dispatch640. Extra partitions do not
by themselves add GPU execution units or memory channels. The one-token full
FFN contains534,773,760 matrix FLOPs and534,773,760 bytes (510MiB) of F16 weights,
about1 matrix FLOP per weight byte before activation/reduction traffic. Splitting
preserves that weight traffic. This makes bandwidth an important hypothesis,
but the Pixel bandwidth ceiling and occupancy have not been measured.

Existing graph-reordering experiment:33.321ms vs33.055ms matched controls,
0.804% slower. That result does not test or disprove explicit partition overlap.
The current local replay candidate averages20.455ms vs21.244ms controls, with
control drift; it is a separate measurement protocol from the earlier USB runs.

A bounded next experiment should keep the same four weight partitions, input,
precision and total work, and vary1/2/4 concurrently eligible parts. Group same
stage work across partitions, use separate intermediate outputs, and merge on
GPU. Compare total FFN worker latency with interleaved controls and archived
CPU/original outputs. A GPU timeline diagnostic is needed to establish actual
overlap; leave profiling disabled in headline timing arms. Independent queues
are an option only after checking exposed queue families and driver behavior.
This can run locally on Pixel without desktop model execution or a whole-rig
campaign lock. No implementation or speedup claim is made by this proposal.

Sources:
- [Current graph](../../../../../../examples/layersplit/ffn-split-worker.cpp)
- [Vulkan queue semantics](https://docs.vulkan.org/guide/latest/queues.html)
- [PowerVR compute guidance](https://docs.imgtec.com/performance-guides/compute-recommendations/html/topics/performance-guidelines.html)
- [Measured local results](PIXEL_DENSE_LOCAL_RESULTS.md)
- [Earlier graph-order result](PIXEL_KERNEL_TUNING.md)
