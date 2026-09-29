# Pixel 10 Pro up-matvec/SwiGLU fusion

2026-09-23. Numerical correctness PASS, complete fusion coverage PASS,
performance FAIL. Retain the previous kernel selection. All changes and
binaries in this experiment are private; production defaults are unchanged.

## Matched confirmation

Mean worker latency for one token in one Qwen layer, pooled equally across
layers18-23. Each arm has240 calls; the first two of20 repetitions are excluded
from timing, leaving108 measurements per width per arm. Two fused arms are
bracketed by three unfused controls. Both use the same new library, the same
input-retaining worker and vec4_u1 with128 threads, subgroup128 and8 rows.
Other existing fusions remain enabled. Profiling is off.

| FFN hidden width | Fusion off | Fusion on | Latency change | Result |
| --- | ---: | ---: | ---: | --- |
| Half,8704 | 11.652174 ms | 11.818894 ms | +1.4308% | FAIL |
| Full,17408 | 20.352023 ms | 21.001449 ms | +3.1910% | FAIL |

For each fused arm, its control is the mean of the immediately preceding and
following off arms. The table averages the two fused arms and their matched
controls. Full-width latency is worse in both repeats:21.430000 vs20.517745ms
(+4.4462%), and20.572898 vs20.186301ms (+1.9151%). Each fused full-width arm is
also slower than both its individual surrounding controls.

The previous unmodified vec4_u1 library/worker was measured before and after
these arms. Its full-width mean is19.782759ms, and half-width11.255083ms; the
complete candidate is6.1604%/5.0094% slower respectively. Those more widely
spaced controls vary19.270241-20.295278ms full-width, so use the closer same-build
on/off comparisons above to isolate the effect of fusion. Battery temperature
was28.8-29.7C; this is not a GPU clock or throttling measurement.

Correctness PASS: all2160 confirmation outputs are byte-identical to the
original Vulkan worker output. CPU-reference maximum relative L2 is
0.0003254826924654395. Both fused arms record744/744 actual fused dispatches:
24 startup dispatches plus720 across240 half/full requests. Cleanup PASS,
boot unchanged, all finite workers exited0, and no ADB forwards were created.

## Change and tuning process

1. In each4352-channel block, retain the gate GEMV and fuse the up GEMV with
   the following SwiGLU: `activation[i] = gate[i] / (1 + exp(-gate[i])) * up[i]`.
   The up GEMV uses F16 weights with F32 input, accumulation and epilogue.
   The completed gate is bound through the existing fusion-input binding.
   The guard requires contiguous, aligned one-row tensors and exact supported
   shapes. Existing graph consumer, dependency and alias checks remain.
2. The first840-call run passes numerical/exact checks, but full coverage
   FAILS:264/384 possible fused dispatches per fused arm. The partial-fusion
   full-width gains of0.616% and3.010% are exploratory, not the final result.
3. A separate12-call diagnostic has exact outputs and confirms one standalone
   GLU remains per request after fusion selection. The existing alias check
   rejects that fusion. The profiler's fusion label remains set even after
   rejection, so the label alone is not proof of execution. The dispatch
   counter and remaining GLU counts are the coverage evidence.
4. Mark the FP32 input as a graph output in an isolated worker so its buffer
   remains alive until graph completion. This preserves the20KiB one-token
   input instead of allowing its storage to be reused. The alias safety check
   remains active. Both control and treatment use this identical worker.
5. The2160-call confirmation achieves complete fusion and exact output, but
   fails the speed test. The candidate is retained as an experiment, not
   selected for deployment. Total3012 outputs including the diagnostic and
   initial sweep pass CPU checks and match the original Vulkan outputs.

The full FFN still computes12 GEMVs over510MiB of logical F16 weights.
Fusion removes four activation dispatches and136KiB of logical intermediate
write/read traffic, approximately0.026% of the weight size. It does not reduce
the matrix work. This explains why a large speedup should not be assumed;
the cause of the measured slowdown has not been isolated. Register allocation,
occupancy, actual DRAM traffic and GPU clocks were not measured.

## Validation and scope

Builds PASS with empty warning logs; worker compilation uses-Werror. All27
SPIR-V modules pass Vulkan1.2 validation. Pyflakes passes on all three changed
Python tools. Prior dense sweep/confirmation and the first fusion audit
reproduce exactly. Archived CPU inputs/outputs, models and phone libraries
are SHA256 checked; no new desktop model computation is part of these runs.

These are phone-local FFN worker measurements, not isolated shader timings,
USB round-trip measurements or full server runs. No phone/host energy,
full-model token equality or multi-phone integration was measured.

Library SHA256:
`513d8eb1ea5931b26190a97dc5bf09c6942c0ee6e957613bb67e905dd1c8e40c`.
Input-retaining worker SHA256:
`0450721e4c42bb80b7e5943f531de947a48334accb70bab370f60d038ef9e4ac`.

- [Machine-readable results](PIXEL_SWIGLU_RESULTS.json)
- [Fusion patch generator](pixel_swiglu_fusion.py)
- [Backend and shader build provenance](software/pixel10pro-swiglu-fusion-v1/BUILD_PROVENANCE.json)
- [Input lifetime patch](software/pixel10pro-swiglu-input-v1/PRESERVE_INPUT.patch)
- [Input lifetime worker build](software/pixel10pro-swiglu-input-v1/BUILD_PROVENANCE.json)
- [First sweep audit](physical/pixel10pro-swiglu-local-1/run1/SWEEP_AUDIT.json)
- [First coverage failure](physical/pixel10pro-swiglu-local-1/run1/FUSION_COVERAGE.json)
- [Coverage diagnostic](physical/pixel10pro-swiglu-coverage-1/run1/DIAGNOSTIC_AUDIT.json)
- [Confirmation configuration](PIXEL_SWIGLU_CONFIRM_CONFIG.json)
- [Confirmation raw audit](physical/pixel10pro-swiglu-local-confirm-1/run1/SWEEP_AUDIT.json)
- [Confirmation complete fusion proof](physical/pixel10pro-swiglu-local-confirm-1/run1/FUSION_COVERAGE.json)
- [Confirmation cleanup](physical/pixel10pro-swiglu-local-confirm-1/run1/CLEANUP.json)
