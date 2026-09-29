# Llama-only OpenCL speed retest

The slow result was reproducible, but it was not the speed of full GPU offload.
With the same deployed binaries, changing GPU layers from 16 to 17 increased
decode from 4.64 to 48.93 tokens/s on this request. Flash Attention enabled
executed successfully but was slightly slower, at 45.82-45.84 tokens/s.

This is a standalone runtime diagnostic, not scheduler route qualification,
a mixed trace, an energy comparison, or a general performance ceiling.
Production scheduling, model placement and frozen baseline files were not edited.

## Results

The unchanged Llama 37 input has 915 prompt tokens and 292 output tokens.
All five executions used the same Q4_0 artifact, prompt token IDs, seed 42,
temperature 0, ignore_eos=true, cache_prompt=false, context 4096, batch 1024,
ubatch 256, parallel 1, and F16 KV. Native startup reports eight CPU threads.
Every execution completed and passed the existing semantic-sanity-v1 check.

| Configuration | Repeats | Decode tokens/s | Request wall time (s) | Startup to READY (s) |
| --- | ---: | ---: | ---: | ---: |
| GPU layers 16, Flash Attention off | 2 | 4.630 / 4.648 | 64.592 / 64.311 | 14.250 |
| GPU layers 17, Flash Attention off | 1 | 48.933 | 6.964 | 13.935 |
| GPU layers 17, Flash Attention on | 2 | 45.821 / 45.841 | 7.418 / 7.333 | 15.558 |

Startup includes runtime initialization/warmup and is reported separately from
request service, not silently removed from an energy total. We did not flush OS
caches: a new endpoint/model load is not proof of cold storage. Each configuration
loaded once; its repeats reused the endpoint with no weight reload. There were
three endpoint loads total across the three configurations.

The full-offload/FA-off decode improvement is 10.548x versus the mean of the two
partial-offload repeats. This is a small sequential diagnostic, not a replicated
controlled experiment across thermal states or workloads. Full-offload FA-off
had one sample; its superiority over FA-on is not a universal conclusion.

## Concrete launch defect

The current Android launcher requires `contract.gpu_layers == manifest.block_count`
at `adapters/android_llama_server.py:506`, then passes that value directly to
`--n-gpu-layers`. For this model, block_count is 16, while this runtime's complete
offload includes the output layer and needs 17.

- The 16 setting logs `offloaded 16/17 layers to GPU`, 8 MiB CPU KV and 120 MiB
  OpenCL KV. One transformer block remains on CPU.
- The 17 setting logs `offloaded 17/17 layers to GPU` and 128 MiB OpenCL KV,
  with no CPU KV allocation line. Input embedding/host bookkeeping still exists.
- Both log `using kernels optimized for Adreno (GGML_OPENCL_USE_ADRENO_KERNELS)`.
- FA-on logs both `flash_attn = enabled` and the OpenCL DK=64/DV=64 prepass
  compilation. This was not a configuration flag silently ignored by the backend.

The older runtime's secondary "repeating layers" log says 15 in both cases;
the total-offload count and CPU/OpenCL KV allocation are the consistent evidence.
This test does not provide a per-operator kernel trace. It isolates a major
placement sensitivity, not the exact CPU, synchronization and governor shares
of the slowdown. KGSL clocks varied automatically, reaching 1.2 GHz in full
offload; no clock, governor, thermal or power-limit setting was changed.

The next production change should represent and validate full GPU offload using
the model/runtime contract, including output-layer accounting, rather than
hardcoding 17 for every model. Re-measure memory, latency and energy qualification
for that exact launch identity. Do not copy qualification from the partial route
or force selection in the campaign runner. That implementation was not performed
as part of this speed-only request; normal scheduling is unchanged.

## Isolation and health

Unlike the previous NCM measurement, these runs used normal USB ADB, no HTP
inference or residency, and no scheduler or peak-memory sampler. Log verbosity
was increased from 3 to 4 for backend evidence. Health plus lightweight KGSL
clock sampling ran every approximately two seconds using the existing probe.
No separate one-prompt/two-output-token warmup request was submitted; native
startup warmup remained enabled. These differences are frozen in each LAUNCH.json.

All 90 health samples were valid and thermally qualified. Maximum physical HAL
sensor temperatures were 51.9 C, 66.7 C and 71.7 C for the three configurations,
below the existing 90 C limit. The 3 GB declared whole-model peak and 768 MiB
reserve fit fresh live memory before every launch. No memory threshold was relaxed.
Actual peak-allocation qualification was not repeated in this speed diagnostic.

All three cleanups passed. Only each test's PID/boot/start-time-verified endpoint
and its own ADB forwarding were stopped. Existing idle launch wrappers and GDM
were left untouched. USB remained `ptp,adb`; no USB mode change or reset command
was issued. No native llama-server remained after cleanup.

The KGSL raw `reset_count` was 1116 during the first configuration and 1157 in
the final audit. It is not treated here as a zero-reset proof. A read-only
post-audit reported zero GPU fault and page-fault counters and no matching
kernel fault messages. Historical [KGSL source](https://android.googlesource.com/kernel/msm.git/+/4c9d42b9ad46c6ca85ab345279715aaecc24648a/drivers/gpu/msm/adreno.c)
increments that counter on GPU startup as well as reset paths; the exact
semantics of the deployed kernel were not established by this experiment.

Generated output was deterministic within each repeated configuration, but
text differs between placement configurations. Passing semantic sanity is not
a logit-equivalence or comprehensive model-quality proof.

## Immutable evidence

Remote roots:

- `/mnt/storage/s42-llama-speed-20260911-v1`: pre-launch recorder failure.
- `/mnt/storage/s42-llama-speed-20260911-v2`: successful original-setting repeats;
  also a preserved pre-launch all-GPU attempt blocked by an overbroad port check.
- `/mnt/storage/s42-llama-speed-20260911-v3`: successful full-GPU FA-off and FA-on.

The recorder failure was an `asdict()` call on an immutable mapping. The port
check incorrectly matched TCP TIME_WAIT entries; it now checks actual LISTEN
entries only. Neither failed attempt launched inference. Both failures and their
cleanup records remain under `physical/`; no failed artifact was overwritten.

Each successful directory includes REQUEST.json, LAUNCH.json, binary hashes,
process identity/maps, native stdout/stderr, raw completion streams, health and
clock samples, executions and cleanup. `SUMMARY.json` verifies identical model,
shared-library and request identities across configurations.

| Result | SHA-256 |
| --- | --- |
| original-16g-faoff | `0c0e7e96131f199e479359e22d64f84b0b6d5d6f10d678b02124f67cd7190602` |
| all-17g-faoff | `48cfcb7a8146f84d3f7d3dc52edfddc568216353a0a2a3d9bf454f15910e2a31` |
| all-17g-faon | `fef2269fe0e05c8f4d2220e220ef23e445c9a0a8374039d17cc0f22d7c929659` |

Shared identities:

- Model: `4b90b1d7ae7324676194755a6dfce11cb6e457982c4c01a1db2857be1ed064ad`.
- Server: `3fad5e2f2730d1e240f176010994c49dc4a6ea59fb970708df8d2a85bb0abe1c`.
- OpenCL: `3dd6b590d39ed7897af367f79661ffa881d57037d24ddc4aa75d218aec2bcd59`.
- Request JSON: `f1f2fe2573cdfc2cf059e0ec80fc66337030ae8483f91caeda9dd4fdf20bd4d8`.

Only the report-local diagnostic script, new artifacts/report, and talks.md were
added or edited. The script reuses the existing process manager, identity-fenced
Android cleanup, HTTP completion/quality client and phone-health probe. No
production tests, replay goldens, scheduler logic, native binaries, baselines or
traces were changed; no full test harness or mixed trace was run. No commit/push.
