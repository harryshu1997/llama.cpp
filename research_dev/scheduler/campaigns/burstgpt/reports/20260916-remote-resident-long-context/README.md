# Long-context relocation: native prefill blocker localized

Status: desktop long-document completion PASS; phone long-document completion
FAIL before output. Do not report this as a successful long-context relocation
or a new energy-saving result. All failed attempts remain preserved.

## What ran

RTX 4060 Ti plus OP15, unchanged Gemma F16 artifact, server and native libraries,
existing full-width FFN shard files for layers 0-23, 23 GPU layers, CUDA graph
mode default, batch 2048, ubatch 512, parallel 1. Context increased from 2560
to 8192 after fresh canonical cold/hot desktop calibration. The short calibration
used existing request 50, 271 input plus 41 output tokens, twice. No worker,
kernel, driver or shard rebuild; no long trace or unrelated process intervention.

The document was tokenized once by the full parent: 5261 input tokens, followed
by 64 output tokens. Exact prompt tokens were reused for the attempted phone arm.
No truncation, input slicing, context shifting or lowered memory check was used.

| Measurement | Result |
| --- | ---: |
| Desktop document completion | 5261 input + 64 output, untruncated |
| Desktop native prefill / decode | 22.654 s / 29.475 s |
| Desktop request wall time | 52.132 s |
| First phone session verified from first stage start | 19.402 s |
| All phone sessions verified from first stage start | 39.195 s |
| Outer phone preparation interval | 39.204 s |
| Phone document output tokens | 0 |
| Reset recoveries | 0 |

| Session | Tensor bytes | Stage start to verified | Loads | Generation | Warmup calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| HTP0 | 2,831,155,200 | 19.402 s | 1 | 1 | 32 |
| HTP1 | 2,831,155,200 | 9.657 s | 1 | 1 | 32 |
| HTP2 | 2,831,155,200 | 10.126 s | 1 | 1 | 32 |

Receipt-ready timestamps and later scheduler verification timestamps are separate
fields in `ATTEMPT_SUMMARY.json`. The 96 native calls are startup/warmup calls,
not successful long-document calls. There is no completed request execution
proof for the phone arm. The router's clean terminal status 0 does not change
the failed request verdict.

Phone preparation measured 556.983 J of server CPU-package plus GPU-board energy.
Charging assumed phone power throughout that interval gives 674.594 / 733.399 /
792.204 J at 3 / 4.5 / 6 W. Desktop document execution measured 5405.308 J of
server energy. These are different phases, not comparison arms. No savings or
break-even count is calculated from them. Raw host samples are retained.

Both native launches allocated 328 MiB of CPU KV and 280 MiB of CUDA KV. The
full parent's measured post-request RSS was 15,190,958,080 bytes, high-water RSS
24,276,729,856 bytes, and process VRAM 12,809,404,416 bytes. The reduced launch
again logged page-exact unmapping of 8,493,170,688 bytes from 72 FFN tensors,
but the failed execution did not reach its post-request RSS/VMA measurement.
The prior successful 8.49 GB relocation evidence is unchanged. No increased
maximum-context capacity is established, and freed CPU RAM is not GPU VRAM.

The desktop retrieved ORCHID-731 but its 64-token raw completion contains thought
markers and an incomplete explanation. Existing semantic-sanity checks passed;
this is not task-accuracy qualification. Exact cross-hardware wording equality
remains diagnostic under the user's prior direction.

## Failures and narrow repair

1. `v1-5ARAnu`: calibration wrapper lacked `--execute --confirm`. Rejected before
   inference; clean postflight. Fixed only the wrapper invocation.
2. `v2-QHjpyd`: new context calibrated; desktop document completed. Offline
   planning rejected `offline phone residency demand is unavailable` before any
   phone load. The wrapper had marked the helper coordinator CALIBRATION_PENDING;
   adaptive helper discovery correctly refuses an unqualified coordinator.
3. `v3-HUdlD0`: separated existing qualified phone loading from new-context
   execution. Three shards became READY, generations stayed 1, warmup succeeded,
   then native prefill aborted. Clean USB restoration and same-boot idle postflight.

The gate now accepts an explicit prior preload catalog. It requires identical
desktop layer placement, launch settings apart from context and its qualification
metadata, helper geometry/configuration, devices, memory pools and transport.
Only phone preparation commands run under that catalog. The same scheduler and
resource ledger then register the new-context execution catalog without changing
READY identities. New-context assisted-copy coordinators remain SHADOW; the
remote-resident mechanism being tested is CALIBRATION_PENDING. Normal calibration
selection chooses it. No route qualification or energy evidence is fabricated.

Files changed this turn:

- `campaigns/burstgpt/remote_resident_gate.py`: preload-catalog compatibility and
  phase separation, plus early validation and phone-only preparation guard.
- `tests/test_remote_resident_reuse.py`: new-context preload/publication/admission
  regression, fresh real leases, unchanged READY state and incompatibility checks.
- This report's configuration, wrapper, reproducible audit, results and hashes.
- `research_dev/talks.md`.

The native binaries, scheduler policy, session lifecycle and wire formats are
unchanged. No commit, push or PR.

## Remaining native defect

The phone server's SSE stream contains `Compute aborted.`; its native terminal
record explains `FFN split runtime context differs from tensor rows`.

`tools/server/server-context.cpp:3896` counts request rows across `batch_view`
before calling `llama_decode`. The physical log records batch 2048, ubatch 512.
`src/llama-context.cpp:1807` splits that batch for graph execution. The FFN eval
callback in `examples/layersplit/ffn-split-client.cpp:1456` compares the unchanged
request-context total against the individual tensor's row count and fails closed.
The previous 271-309 token prompts did not exceed one microbatch, hiding this defect.

Next repair must bind attribution to the actual microbatch while preserving
request IDs, generations, row totals and the exact equality check. Merely
allowing unequal totals, disabling the check, or changing one arm's batch size
is not an acceptable repair. It requires native server work, new binary hashes
and transport/software requalification before another bounded test. This exceeds
this attempt's frozen unchanged-native-binary scope; no native change was made.
No in-flight owner-loss cancellation was attempted.

## Validation and artifacts

67 focused tests PASS in 26.045 s; both replay tests PASS in 80.354 s, unchanged
goldens. Wrapper syntax and `git diff --check` pass. No broad harness repeated.
See `TESTS.json` and `RETRY_V3.json`.

Remote attempts:

- `/mnt/storage/s42-remote-resident-long-context-20260916-v1-5ARAnu/`
- `/mnt/storage/s42-remote-resident-long-context-20260916-v2-QHjpyd/`
- `/mnt/storage/s42-remote-resident-long-context-20260916-v3-HUdlD0/`

Local copies, excluding duplicated deployment source: `physical/v1`, `v2`, `v3`.
Original remote source trees remain intact. V3 reuses V2's successful physical
calibration, same boot, model, binaries and placement; calibration was not repeated.
The phone was already idle on boot `a3fea180-44e8-4564-9c82-522876ad07e6` when work
resumed. Its kernel notes/BTF hashes match the qualified candidate. This task did
not reboot it or stop another campaign.

Reproduce the audit without physical work:

```sh
python3 summarize_attempt.py physical/v3 --output SUMMARY_FRESH.json
```

Hashes (SHA-256):

- Calibration: `be862453b4a2feec4d3ef97184b5fe9cd0f1ebf3a71f898ed347018071e3d31c`
- Failed gate: `5e49096424bd6ede6820763e6fa8bbf700aa6732fe835d51c8bbb0aa1a392b6c`
- READY map: `531762165e62c890085f1fdb9c44515addba392331761c032d27c758d5987bfa`
- Terminal: `6c13a4c21ac803833613af20e5e6e4a509ae40420f02637a140e680faed888ab`
- Audit summary: `9ab3c7057d2011bc09ad53b3c83043e07ceecc9a8ffaa96a9f40e18aeefbc4e3`
- Server: `00220ffd27aa167752de240d82cdf1062104ec087ce62c388e2e193b8b1d974a`
- Parent model: `ed76f2183d2d1d65091986033023e6c78d27f6276c1b0c5826cc92acf73538cf`
- Shard index: `5d0fc13e8be1875269a6c8ef2eeaaded8c968ecf262703de529bdbc71ba13681`

Full native-library, worker, per-shard, transport, generation and operator-plan
identities are preserved in the preflight, READY, commands and terminal records.
