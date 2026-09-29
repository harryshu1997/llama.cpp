# Pixel dense shader experiment, 2026-09-23

Latest: phone-local sweep and confirmation complete,3,960 numerical checks PASS.
All2,160 confirmation outputs exact. Best repeated vec4_u1 full20.455ms vs
21.244ms matched controls (-3.714%); control drift limits confidence. No new
server/energy/USB claim or production promotion. Old desktop queue cancelled.
[Results and limitations](PIXEL_DENSE_LOCAL_RESULTS.md).

Build PASS: nine shader-body variants, 27 SPIR-V modules (three existing
reduction modes per variant), glslc -O -Werror and SPIR-V validation for
Vulkan 1.2. No compiler warnings. Arithmetic result types remain float32;
the modules require StorageBuffer16BitAccess and do not require Float16
arithmetic capability. Python pyflakes PASS. The extended auditor reproduces
all four archived sweep reports exactly.

Physical launch FAIL at 2026-09-23 03:46 UTC: the 900-second shared-lock wait
exited75. The longtail treatment retest continued to own the rig. No run
output directory, RUN.log or DONE.json was created, no phone deployment or
call occurred, and no tuning job remains queued. This is a blocked launch,
not a measured kernel failure. Correctness, performance and full-model
output identity remain NOT VERIFIED. [Lock evidence](physical/pixel10pro-dense-gemv-1/LOCK_ATTEMPT.json).

Current state at04:34 UTC: user chose to keep tests queued. A second900s
wait also exited75 with zero phone calls. Detached queue PID2282252 now
retries the lock wait and will execute the finite sweep once, then audit it.
Do not launch another copy. Watch rig QUEUE_EVENTS.jsonl, DONE.json and
SWEEP_AUDIT.json. The longtail campaign remains untouched.

Current state at05:13 UTC: supersedes the queue above. User clarified Pixel-only
testing, so the unused desktop queue was cancelled and guarded against re-entry.
The same15 arms now run entirely on Pixel using local-loopback replay and archived
CPU references. No new desktop model work, USB timing or energy measurement.
The phone-local launch retries the fd-inheritance preflight fix; first7 arms have
completed. Numerical/performance audit pending. See
[pixel_local_sweep.py](pixel_local_sweep.py) and
[local input/reference provenance](physical/pixel10pro-dense-local-1/run1/REFERENCE_PROVENANCE.json).

## Scope

The previous sweep changed launch specialization constants while reusing
all existing shader bodies. This experiment adds a small dense aligned loop
to the existing GEMV shader and reuses its bindings, offsets, reduction,
fused-bias handling and output path. The new loop is selected only for
one-column F16-weight/F32-input GEMV with K=5120 or 4352 and aligned offsets.
Other shapes use the original body. Backend selection requires explicit
`S42_PIXEL_F16_SHADER` plus the existing workgroup/row settings. The original
runtime and production source remain unchanged.

Weights remain in their existing row-major representation. The striped
variant changes the mapping between lanes and matrix columns, without
prepacking weights. FP32 accumulation is retained. Eight elements per lane
uses two 4-element vector loads; it is not an asserted 128-bit machine load.

| Variant | Elements per lane/iteration | Unroll | Accumulator chains | Weight access |
| --- | ---: | ---: | ---: | --- |
| pair4_u4 | 4 | 4 | 1 | Original two f16vec2 loads, streamlined loop |
| vec4_u1 | 4 | 1 | 1 | One f16vec4 load |
| vec4_u2 | 4 | 2 | 1 | One f16vec4 load |
| vec4_u4 | 4 | 4 | 1 | One f16vec4 load |
| vec4_u8 | 4 | 8 | 1 | One f16vec4 load |
| vec4_u4_a2 | 4 | 4 | 2 | Independent accumulation chains |
| vec4_u4_a4 | 4 | 4 | 4 | Independent accumulation chains |
| vec8_u2 | 8 | 2 | 1 | Two adjacent f16vec4 loads |
| striped4_u4 | 4 | 4 | 1 | Consecutive lanes read consecutive scalar elements |

All first-pass variants use 128 threads, 128-lane subgroup and 8 output rows;
quantum 4352. These settings match the qualified winner from the earlier
launch sweep. A later subgroup/row sweep is warranted only after measuring
which new body is promising.

## Physical protocol

15 arms x 120 calls = 1800 phone calls. Two original-runtime reference arms
verify numerical consistency; four interleaved current-best row8 controls
bracket three groups of three candidate shaders. Each arm uses the same
six layer inputs at widths 8704/17408, 10 repeats, first 2 warmup. CPU comparison
limit relative L2 0.01. Every raw output and timing is retained. The original
references and row8 controls must be bit-identical to the original. A changed
accumulation order may change low bits in candidate outputs.

No profiling in timing arms. CPU reference and phone runs are all under
`flock -w900` on the shared rig. Each worker has a finite 120-call budget and
normal shutdown. The harness removes only its own forward and records the
boot before/after. A winner needs repeated comparison and a real-server
output-token check before it can replace the earlier qualified selection.

## Evidence

- [Shader loop](pixel_dense_gemv.glsl), [builder](build_pixel_dense_gemv.py).
- [Shader patch](software/pixel10pro-dense-gemv-v1/DENSE_SHADER.patch),
  [backend patch](software/pixel10pro-dense-gemv-v1/DENSE_BACKEND.patch).
- [Build provenance](software/pixel10pro-dense-gemv-v1/BUILD_PROVENANCE.json),
  [SPIR-V validation](software/pixel10pro-dense-gemv-v1/SPIRV_VALIDATION.json).
- [Physical sweep configuration](PIXEL_DENSE_GEMV_CONFIG.json).

Library SHA256:
`4d2b7058688ecc253dc2d95b11e1e954314191f823538741c7479db6357ce37b`.
Rig envelope: `/mnt/storage/s42-pixel10pro-dense-gemv-20260923-v1`.

Static SPIR-V inspection confirms native f16vec4 loads in the vector variants
and preserved unroll hints. This does not prove the driver emits a specific
machine instruction or a faster kernel. [Inspection](software/pixel10pro-dense-gemv-v1/SPIRV_STATIC_ANALYSIS.json).

Historical desktop launch command, now guarded as cancelled. Use the completed
phone-local evidence below; do not queue this old run:

```sh
ssh zhihao@172.20.74.85 'flock -w 900 -E 75 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock bash /mnt/storage/s42-pixel10pro-dense-gemv-20260923-v1/RUN.sh'
```

After completion, use `analyze_pixel_gemv.py` on the run1 directory. Its explicit
comparison_controls select the interleaved current-best row8 arms; the two
original-runtime reference arms are excluded from performance ranking. Select
a candidate only after CPU/output audits pass, repeat it against those controls,
then use the existing real-server qualification with its shader environment.
