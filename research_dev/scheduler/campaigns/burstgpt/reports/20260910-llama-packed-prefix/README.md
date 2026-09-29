# Llama packed CPU-prefix calibration, 2026-09-10

This is a bounded native calibration, not scheduler qualification or a mixed
trace. It continues the prior `20260910-llama-htp-split` diagnosis. Prior physical
artifacts and the deployed phone worker/shard are unchanged.

## Result

The packed-prefix fix removes the observed extra prefill cost. The final
`packed-v2` sweep completes 10/10 requests and logs CPU_REPACK selection.
The 50% split improves request time by 6.88% and saves 2.42% request energy
at assumed phone power 4.5 W. This is a modest measured joint win, not a 25%
saving or production qualification. Two repetitions per fraction do not
establish a calibrated confidence interval or the globally optimal split.

| FFN columns on phone | Decode tokens/s | Request s | Fleet J, 4.5 W | Saving vs CPU |
| --- | ---: | ---: | ---: | ---: |
| 0%, CPU control | 32.37 | 10.68 | 1,005.03 | 0% |
| 25% | 34.37 | 10.10 | 1,023.86 | -1.87% |
| 50% | 35.17 | 9.95 | 980.68 | 2.42% |
| 75% | 31.97 | 10.79 | 931.38 | 7.33% |
| 100%, NPU FFNs | 29.00 | 11.75 | 835.72 | 16.85% |

50% has the lowest request time of the tested splits. 75% is about 1% slower
than CPU; 100% is about 10% slower. 25% is faster but energy-negative, so more
CPU/NPU overlap alone does not guarantee a useful energy policy.

The preceding `packed-v1` also completed 8/8 requests: at 50%, 10.03 versus
10.56 s and 969.85 versus 1,009.03 J (3.88% saving). Its default verbosity
suppressed the loader message. Its execution/terminal evidence is preserved,
but its packing-log status remains UNAVAILABLE in `AUDIT_V1.json` and the
summary. The final sweep enables verbosity 4 in both arms and adds 25%.
No physical result was overwritten or relabelled to supply the missing log.

## Narrow implementation

The original packed CPU matrix kernel treated a shortened input-column view
as contiguous. FFN down-projection views retain the full tensor's row stride,
so crossing an interleaved row-group boundary read the wrong packed weights.
The focused pre-fix regression records 56 numerical failures plus failure to
reject unsupported views. `CPU_VIEWS_BEFORE.log` is preserved.

- `ggml/src/ggml-cpu/repack.cpp`: traverse packed K-prefixes using the original
  row-group stride; reject offset, incompatible geometry and packing views.
- `src/llama-model-loader.cpp`: the existing view-safe dense FFN path may use
  CPU_REPACK for Q4_0 only after full and aligned-prefix support checks pass.
  Other types retain the existing ordinary-CPU fallback. Existing fenced
  tensor-loader changes are untouched.
- `tests/test-cpu-repack-views.cpp` and `tests/CMakeLists.txt`: direct numerical
  and fail-closed view regressions, without extra prefix-weight allocations.

There is no new prefix cache, model-specific rule, wire format, shard format,
phone-worker change, or production scheduler policy. Original overlapping
files are preserved under `source-before/`.

## Validation

- 296 packed-prefix numerical cases pass locally and on the physical desktop:
  Q4_0 and Q4_K, one/four threads, GEMV/GEMM, odd token counts, shortened output
  rows and input columns, and real Llama gate/up/down dimensions.
- The same 296 cases and two unsupported-view checks pass with both GGML and
  the test instrumented by AddressSanitizer and UndefinedBehaviorSanitizer.
- Standard CPU Q4_0 MUL_MAT checks: 46/46 supported cases pass.
- Focused CTest registration, loader compilation and diff whitespace pass.
- No broad scheduler suite or trace was run.

## Physical method

Fresh source/build: `/mnt/storage/s42-llama-packed-prefix-20260911-v1-JcmCWJ`.
The existing native source deployment was copied into this fresh directory;
only the four native/test files above were overlaid. CPU-only Release build,
CPU_REPACK and the existing server FFN client enabled. The copied deployment
omits `app/`, so `LLAMA_BUILD_APP=OFF` was required. The previous binary was
CUDA-capable, although its tested execution was CPU-only. Both new arms use
the exact same new binary and libraries; this is not a clean upstream control.

Model and prompt are unchanged: Llama 3.2 1B Q4_0, request 37, 915 prompt token
IDs, 292 output tokens, seed 42, temperature 0, no prompt cache. Four decode
threads, eight prefill threads, strict P-core mask `0x5555`, polling zero,
context 4096, batch 1024, ubatch 256, parallel one, zero GPU layers. CUDA is
hidden only from the owned CPU server subprocesses. GDM is left running.

The existing 452,988,576-byte phone shard holds FFNs for layers 0-15 and 8192
columns at HTP0 generation 1. Fractions 0/100/75/50/25 are measured twice each
in the final sweep (the initial sweep omitted 25%);
positive controls use the existing request ACK path. The assisted endpoint
and shard are reused across fractions, without reloading.

`100%` means all FFN columns, not whole-model NPU inference. Attention and
other non-FFN operations still execute on CPU. No whole-model NPU-only
reference or scheduler lease/ticket qualification is claimed.

Energy boundaries match the prior diagnostic: request service includes
prefill and decode; staged phone preparation and desktop initialization are
measured separately. CPU package energy is RAPL, GPU board energy is NVML,
phone active power is assumed at 3/4.5/6 W, idle power 0.875 W in both arms.
This is not desktop wall-plug energy. No overlapping windows are subtracted.
`SERVER_MEMORY.jsonl` records process-identity-bound RSS/high-water samples.

## Energy sensitivity and startup accounting

| Phone FFN fraction | Fleet J, 3 W | Fleet J, 4.5 W | Fleet J, 6 W |
| --- | ---: | ---: | ---: |
| 0% | 1,005.03 | 1,005.03 | 1,005.03 |
| 25% | 1,011.12 | 1,023.86 | 1,036.61 |
| 50% | 968.22 | 980.68 | 993.13 |
| 75% | 917.68 | 931.38 | 945.09 |
| 100% | 820.62 | 835.72 | 850.83 |

At 50%, the CPU-package component falls from 907.44 to 852.42 J, while the
GPU-board component is 88.25 versus 89.45 J. The phone term reduces the net
saving to 24.35 J at 4.5 W. The corresponding savings at 3/4.5/6 W are
3.66/2.42/1.18%. These are point measurements, not uncertainty bounds.

Final worker/transport launch to host-observed readiness is 9.713489 s.
Native phone-clock durations: weight reads 1.869618 s, HTP initialization
0.049751 s, weight upload 0.257491 s. Remaining time includes launch,
FunctionFS/NCM setup and readiness detection, not merely weight transfer.
Raw phone and host timestamps remain separate in READY/PHONE_LOG records.
The prior same-binary sweep reached readiness in 8.825053 s.

Measured server energy during final phone startup is 125.037245 J. With
phone power at 3/4.5/6 W, startup totals are 154.18/168.75/183.32 J.
CPU desktop initialization is 0.608195 s / 29.04 J; assisted desktop
initialization is 0.505244 s / 23.30 J, including the same phone idle power.
These are separate measured intervals, not a reconstructed cold experiment.
Earlier shard staging, identity/hash checks preceding the timed launch,
and cleanup are not inside those startup/request totals. Therefore **no
complete cold/end-to-end energy saving is claimed**.

For 50%, `ceil(measured_phone_startup_energy / mean_request_energy_saving)`
is 5/7/16 requests at 3/4.5/6 W. This accounts only for the measured startup
component and assumes the measured per-request saving persists; it is not
an observed complete-lifecycle break-even. At 4.5 W, one 50% request plus
phone startup costs 1,149.43 J, already exceeding CPU request energy before
adding desktop initialization. Reuse is necessary.

## Physical correctness and remaining bottleneck

- Both arms report CPU_REPACK weights 522.00 MiB and a 718.75 MiB model file
  mapping. These are not two independent model identities or prefix caches.
  Measured maximum RSS/HWM is 1,553,428,480 bytes for CPU and 1,561,964,544
  bytes for the assisted process. No additional per-fraction weight buffer
  appears; small graph/runtime allocations still exist.
- Mean prefill: CPU 1.659481 s, 50% 1.643554 s. The previous raw view-safe
  implementation was 1.6234 versus 2.6037 s. That older build remains a
  historical diagnostic, not an identical-build causal comparison.
- One shard-set load per sweep, one assisted endpoint per sweep, no reload
  or endpoint restart when fractions/requests change. HTP0 generation 1
  remains unchanged. Total final phone calls: 37,120 (9,280 per positive
  fraction); 64,960 calls across both new sweeps.
- Every positive request has control ACK at token index 2 and 4,640 calls
  for the remaining 290 positions x 16 CPU-resident FFN layers. Token-position
  coverage is 99.315%; fraction-weighted coverage is 24.829/49.658/74.486/
  99.315% for 25/50/75/100% splits, respectively. Phone call counts do not
  convert partial-column assistance into full-FFN coverage.
- All final 10/10 requests pass existing semantic sanity and exact token
  count checks. Native terminal status, control/policy hashes, shard parent,
  worker weight hash and generation checks pass. All 54 health samples are
  valid; 547 host-power samples report no sampler error. These are not
  scheduler ticket/lease or task-accuracy qualification.
- FunctionFS terminal reports zero recoveries and status zero. No execution
  fallback or stale-generation failure was observed. Expected initial USB
  setup and final normal restoration are not claimed as zero USB mode changes.
  Final USB is `ptp,adb`, kernel unchanged, no owned inference workers remain;
  GDM still occupies 3,178 MiB. Cleanup reports PASS for both sweeps.

At 50%, native mean CPU-prefix work is 0.524 ms per FFN call, NPU compute
0.413 ms and RPC 0.853 ms, leaving 0.328 ms exposed wait. At 100%, CPU-prefix
work is effectively zero, but RPC is 1.179 ms with 0.702 ms NPU compute,
leaving almost the full 1.176 ms exposed. Packing repairs CPU efficiency;
per-layer transport/synchronization now limits additional offload. Neither
raising the fraction nor forcing the route removes that cost.

## Artifacts and identities

- Final result: `physical/packed-v2/RESULT.json`, SHA256
  `c3cfbe22da45db9ff712a4f7ab29c5c910c835a39dbc8ddc954c11f4c8833c02`.
- `SUMMARY.json` SHA256:
  `5f3c2babdf7c0e863a779c227a194841ad25b872491f3bf847b589c01637ad3f`.
- New desktop server SHA256:
  `4ac94a721ccd681486b0fa6540d7f63836737913a6a0bed2252febaa6403dc3e`.
- Parent model SHA256:
  `4b90b1d7ae7324676194755a6dfce11cb6e457982c4c01a1db2857be1ed064ad`.
- FFN shard SHA256:
  `698facd8ef7e54f8f3057407d6f6a14741734112408e75c4f43a545d5b4ee7df`.
- Shard index SHA256:
  `66729a297f9d0565891a92e67ae4041d49cc0f7d395fb37bb09197d49fb2dd09`.
- Unchanged phone worker SHA256:
  `43adcb755f8ff10ba30073ae55a31c7af95ee4f14071cfd2a5ba961b8317da19`.

`BUILD_PROVENANCE.json` records exact changed/source file hashes and the
dirty-tree origin. SPEC/LAUNCH/WORKER_LAUNCH/PHONE_HASHES records preserve
resolved commands, parent/shard/index identities and all desktop/phone library
hashes. `EVIDENCE_SHA256.json` inventories report evidence. Nothing was
committed, pushed, or deployed over an existing reference. The phone binaries,
shards, production scheduler and prior results were not edited.
