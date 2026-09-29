# FFN microbatch attribution repair

Status: repaired native attribution and the bounded long-document relocation
gate PASS on the RTX 4060 Ti plus OP15. Same-boot idle postflight PASS.

## Physical result

Both arms completed 5261 input plus 64 output tokens, without truncation, at
context 8192. Batch 2048 and ubatch 512 are unchanged. The old row mismatch is
absent. The independent audit checks all 1800 document calls against the exact
request, tensor row count, full FFN width, ticket, owner, artifact, geometry,
operator plan and session generation. It excludes 96 startup/validation calls
using execution-proof boundaries, not just a request-ID substring.

| Measurement | Full desktop | Remote FFN parent |
| --- | ---: | ---: |
| Completed prompt / output tokens | 5261 / 64 | 5261 / 64 |
| Process RSS | 15.190 GB | 6.899 GB |
| Process peak RSS | 24.277 GB | 15.785 GB |
| Process GPU memory | 12.809 GB | 12.805 GB |
| Native prefill time | 22.710 s | 46.435 s |
| Native decode time | 29.505 s | 26.113 s |
| Request-attributed phone calls | 0 | 1800 |

GB is decimal. The native omission proof covers 72 FFN tensors and
8,493,465,600 tensor bytes; 8,493,170,688 complete-page bytes were unmapped,
with zero remaining VMA overlap. Process RSS fell by 8,291,872,768 bytes.
GPU memory is essentially unchanged and receives no relocation credit.
Both launches allocate 328 MiB CPU KV and 280 MiB CUDA KV.
The three resident shards contain 2,831,155,200 tensor bytes each. Their sum
plus the unchanged 805,306,368-byte workspace allowance is 9,298,771,968 bytes
under the 10 GB phone pool. The workspace allowance is not a measured phone
allocation high-water mark.

The phone performs 288 prefill and 1512 decode calls. Actual prefill shapes
are 512 rows (240 layer calls), 137 rows (24), and 4 rows (24). These are the
real microbatches, not a clipped logical count. Per-session totals are 600
calls, 42,592 input rows and 327,106,560 payload bytes. The request's aggregate
127,776 rows equal `(5261 + 63) * 24`; the final output token needs no further
decode evaluation. All three sessions remain generation 1, one load each.
The server's no-control policy generation 0 is distinct from the physical
session fencing generation; no physical session uses generation 0.

| Session | Native load authorization to READY | Scheduler stage to verified | Verified from first stage |
| --- | ---: | ---: | ---: |
| HTP0 | 10.615 s | 19.778 s | 19.778 s |
| HTP1 | 9.243 s | 9.656 s | 29.439 s |
| HTP2 | 9.393 s | 9.793 s | 39.236 s |

Outer phone preparation takes 39.242 s. Native read/init/upload times and
host receipt/verification timestamps are separate in `physical/AUDIT-v2.json`;
phone and host monotonic clocks are never subtracted. The complete remote-owner
parent starts after all its required owners are READY; this gate does not
claim concurrent serving during preload or replacement. Native terminal status
0, zero reset recoveries, no fallback, unchanged residency, Android USB restored,
and the original boot ID preserved.

## Energy and remaining performance limit

The remote path is not an overall speedup: decode is 11.50% faster, but prefill
is 2.045x as long. Full request wall time is 52.218 s; the reduced request
interval is 78.009 s and includes its desktop launch, unlike the full request
interval. Full desktop loading separately takes 5.149 s.

CPU-package and GPU-board energy are measured; phone energy is assumed.
Raw `HOST_SAMPLES.json` and native execution logs are preserved.

| Phase | CPU package | GPU board | Server total |
| --- | ---: | ---: | ---: |
| Full desktop loading | 175.126 J | 124.327 J | 299.453 J |
| Full request, weights already loaded | 3256.786 J | 2179.385 J | 5436.172 J |
| Phone preparation | 238.397 J | 342.094 J | 580.491 J |
| Reduced request including desktop launch | 1182.919 J | 2709.297 J | 3892.216 J |

| Assumed phone active power | Reduced request interval | Phone preparation interval |
| --- | ---: | ---: |
| 3 W | 4.126 kJ | 0.698 kJ |
| 4.5 W | 4.243 kJ | 0.757 kJ |
| 6 W | 4.360 kJ | 0.816 kJ |

For reference, charging 0.875 W phone idle in the full request window gives
5.482 kJ. These are scoped phase measurements, not an end-to-end savings
percentage: preparation is not free, cleanup/gaps are not incorporated into an
arm total, and launch/cache boundaries are not identical. Nothing is subtracted
from overlapping intervals. No trace or break-even claim is made.

Both raw outputs retrieve ORCHID-731, but their 64-token explanations contain
thought markers and are incomplete. The existing semantic-sanity gate passes;
this is not task-accuracy qualification. The first 40 generated tokens agree,
with divergence at index 40. Earlier exact-output FAILs remain unchanged.

This establishes usable long-document inference with the FFN weights actually
absent from desktop mappings. Since both parents fit at 8192, it does not yet
establish a larger maximum-context frontier. The remaining performance issue
is long-prefill cost, not request attribution.

## Cause and repair

The preserved V3 attempt failed before document prefill: server request context
counted the logical 2048-token batch, while the tensor callback saw a physical
512-token microbatch. `FFN split runtime context differs from tensor rows` was
the correct fail-closed response to that inconsistent attribution.

The existing native FFN interface now has a synchronous microbatch-context
callback. `llama_context::process_ubatch` supplies the actual row/sequence view
immediately before execution, including graph reuse. The server resolves those
sequences to its active request IDs, slots and policy generations, requiring
the same policy and unambiguous single-sequence rows as before. It then uses
the existing FFN client context callback. Failure aborts before that microbatch
executes. No context is retained from an entire logical batch.

The client's exact tensor-row equality and per-request accounting checks are
unchanged. Startup/validation calls are excluded by physical execution-proof
boundaries; actual request calls carry their exact context. No worker/shard format, scheduler policy, session lifecycle,
mask, fraction, batch size, context size, kernel or graph-mode change.

Production changes:

- `src/llama-ext.h`: internal experimental callback declaration.
- `src/llama-context.h` and `.cpp`: context-owned callback and actual microbatch view.
- `tools/server/server-context.cpp`: request attribution at the physical boundary.

Test changes:

- `examples/layersplit/ffn-remote-resident-probe.cpp`: explicit microbatch/context
  modes and per-request accounting in the existing tiny-model native probe.
- `research_dev/scheduler/tests/test_remote_resident_native.py`: multi-microbatch
  and two-sequence numerical/accounting regression, original failure reproduction,
  and rejection before any phone call.

Before-images, local native records, build scripts and frozen experiment inputs
are here. All previous physical artifacts remain intact. No commit, push or PR.

## Software validation

10 native tests PASS, 67 focused scheduler tests PASS, two replay tests PASS.
Both replay goldens unchanged. The native suite includes the existing exact
tiny-model logits/argmax and independent VMA-unmapping checks. The rejected
logical-batch reproduction still fails with the original row mismatch and zero
phone calls. The corrected 37-token case uses 8-token microbatches, including
interleaved requests with distinct generation identities, with exact row totals.

Local CPU server build PASS. See `TESTS.json` and `LOCAL_SERVER_BUILD.log`.

## Physical provenance and limits

Fresh attempt: `/mnt/storage/s42-ffn-microbatch-20260916-v4-arwuw3/`.
The new CUDA binary is built from the previously deployed native source plus
only the narrow patch above. Phone binaries and shards are not rebuilt.
Nine fresh measured USB checks PASS and bind the new host executable and library
hashes. Canonical cold/hot desktop calibration PASS revalidates the exact
8192-context, 23-GPU-layer parent before one long-document request per arm.
Calibration peak process VRAM: 12,723,421,184 bytes; unchanged parent placement
`a87d0996e54e49c0280e35e138a375464404efa62c3d236cedcba4f4ed8c3920`.

The prompt is unchanged: 5261 input and 64 requested output tokens, batch 2048,
ubatch 512, parallel 1, seed 42, default CUDA graph mode. CPU FFNs 0-23 reside
in three existing F16 phone shards. Strict generation, shard, geometry, memory,
transport and terminal-proof checks remain enabled.

This is a bounded memory/execution validation, not a trace. Both parents already
fit at this context: success alone would not prove an increased maximum context
frontier. Freed CPU RAM is not GPU VRAM. Report preparation separately, and do
not turn asymmetric cold/warm intervals into an end-to-end savings claim.

Preserved setup failures: `BUILD.log` records a missing CUDA library search
path at link time, fixed in the build subprocess environment; `MATERIALIZE.log`
records tuple/list JSON normalization in the evidence wrapper; `PREFLIGHT.log`
records the missing deployment GGUF reader, added before preflight v2 PASS.
None of these ran a failed document inference. Old V3's native failure remains
in its original directory. No worker/shard rebuild, reboot, flash, GDM change,
unrelated process stop, long trace, commit or push.

SHA-256 records:

- Gate result: `7cce57c00acc88ee61b3d855400c31a9cbb383501e446c38a0db0e84471f2eb0`
- Independent audit v2: `2f0a4b861c0e96729ea5555e5ca97e4cec52c23f62663d919f40f92333b0bed5`
- READY: `cd143c073516984857b96ae07c3c297816f5cbff9333507facb793159e69b9a1`
- Terminal: `6b53992c95018d7d6c170ef9697a9989bcbb43052dc1af8d30b6e3219f1779a8`
- Calibration: `dda241207d70917b31ad7756064bd48c1dca1996351686cff1e388bd4b42436d`
- Server: `6cfe48d01a5c0a2b1b5cda73fc9f723bb51e339885dcc107291cd9e61dca27b9`
- Server implementation: `489b1156a3f2e4655305c862fa142d47752892c71fa73f33bfd91d00ffc5502e`
- libllama: `959f48a2061003325bc7f9021b19e3559cfad20ba71efdae57fab856222de745`
- Transport identity (canonical): `032fb83c3f0ef3be6378bce492e71c33cb5789ce4f066754d2dea4883f1ea041`
- Source binding file: `11c3309101a361afd0a053d94c16e61e1912fd7a257f0d2b0aba6351f13f5c78`

Other native-library, parent, per-shard, operator-plan and execution-proof
hashes are in the immutable physical records. Reproduce the independent audit:

```sh
python3 audit_result.py physical/gate-run-v1 --output AUDIT_FRESH.json
```
