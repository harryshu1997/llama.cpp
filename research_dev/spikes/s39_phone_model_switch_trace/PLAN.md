# S39 Active warm-tier model-switch proof of concept

Status: `TRACE_REPLAY_READY; DIRECT_MIXED_BATCH_MECHANICS_PASS; QWEN_ROUTE_PROVISIONAL; GEMMA_Q4_CORRECTNESS_FAIL; W0_BLOCKED`

Authority: [../../ACTIVE_WARM_TIER_DESIGN.md](../../ACTIVE_WARM_TIER_DESIGN.md).

S39 is the first bounded implementation of the active warm-tier design. It does
not implement a general multi-model cache, direct KV migration, or an energy
claim.

## Question

Can OP15 and OP12 collectively serve a GPU-nonresident model while a 16 GiB
desktop GPU changes models, then move all live requests into one CUDA
continuous batch by replaying committed token histories without stopping
service?

After cutover, can the phones release the promoted model and prepare the
displaced model so the same transition can run in reverse?

## Provisional model pair

- model A: Gemma 4 12B IT;
- model B: Qwen3 14B.

The exact source and shard GGUF artifacts are provisionally frozen in
`SHARD_MANIFEST.json`. W0 must still bind one complete executable route per
model, including tokenizer, chat template, KV types, RoPE parameters, context
parameters, and sampling configuration.

The current Qwen assignment remains a candidate, not an eligible route. Real
B1, B8, and B32 collective-phone cohorts match their same-artifact CUDA
control, but corpus, repeated-process, memory, and measured-network-path gates
remain. The real Gemma Q4 collective-phone run executes with clean placement
but produces different tokens from its same-artifact CUDA control. S39 must
not route around either fact.

Every checkpoint is stored on desktop NVMe and every assigned phone shard is
stored on phone UFS before a run. The treatment does not send a 9-12 GB model
from phones to the desktop during promotion.

Provisioning checkpoint, 2026-07-23:

- both frozen source GGUF hashes match `model_assignment.json`;
- OP15 stores Gemma `[0,36)` and Qwen `[0,32)`;
- OP12 stores Gemma `[24,48)` and Qwen `[24,40)`;
- the overlaps preserve Gemma candidate cuts 24-36 and Qwen cuts 24-32;
- all four final on-device SHA-256 values match
  `SHARD_MANIFEST.json`;
- readiness is `STAGED_ONLY`, not executable or scheduler-ready.

The partial-load graph and layer-filtered KV path now support dense Qwen3 as an
experimental LayerSplit route. Storage readiness remains distinct from
execution readiness. `CURRENT_ROUTE_READINESS.json` is derived from hashed raw
phone and CUDA records by `build_route_readiness.py`; it cannot issue `PASS`
from the current evidence.

## Frozen trace

`build_trace.py` verifies the pinned BurstGPT v2 source and scans aligned
20-minute windows without changing time. A successful request has
`output_tokens > 0`. The frozen source bin is 21,424:

- 20 source rows, including 19 successful requests and one recorded failure;
- 15 ChatGPT and 4 GPT-4 successful requests;
- 4,919 input tokens and 1,038 output tokens;
- arrivals from 12 to 1,143 seconds in a 1,200-second window;
- three same-session GPT-4 requests at 180, 198, and 234 seconds;
- one later GPT-4 request at 543 seconds.

The raw trace remains `real`. Mapping ChatGPT to model A and GPT-4 to model B is
`semi_synthetic`. Prompt payloads and SLO classes are also separate, hashed
sidecars. The recorded failed request remains in the trace and is not executed.

The second model-B request within 60 seconds, at trace time 198 seconds, is the
first provisional promotion trigger. The isolated model-B request at 543
seconds is the hysteresis control and should remain on phones because a model-A
request arrives two seconds later.

Two trace profiles are retained:

- `bundle/` is the low-load correctness trace from source bin 21,424.
- `bundle_frequent/` is the active frequent-switch trace from source bin
  16,522. It has 74 successful requests, five useful model-B promotion cycles,
  and nine target changes during arrivals.

`ACTIVE_TRACE.json` selects `bundle_frequent/` for current testing. The
correctness trace remains the fallback for debugging ownership, replay, and
token-conservation failures.

`build_replay_intents.py` deterministically converts the selected trace into
five promotion windows and nine ordered policy intents at 60, 129, 359, 421,
498, 546, 911, 989, and 1,154 seconds. The replay manifest binds the active
trace, assignment, shard manifest, builder, and output bytes. These are policy
intents only, not proof that a route is ready or that a transition executed.

## W0 route screen, 2026-07-23

Both phones ran one complete model route at cut 30 with GPUOpenCL and greedy
decoding. The same prompt and token budget ran on CUDA from the exact source
GGUF.

| Model | Phone route | CUDA control | Derived status |
|---|---:|---:|---|
| Qwen3 14B Q4_K_M | 8 tokens, exact match | 8 tokens | `PROVISIONAL_B1` |
| Gemma 4 12B Q4_0 | 8 tokens, mismatch | 8 tokens | `FAIL_CORRECTNESS` |

Qwen's phone TTFT was 1.574 seconds and service wall time was 3.668 seconds for
the five-token prompt plus eight generated tokens. Its CUDA control completed
in 0.163 seconds. Gemma's phone TTFT was 1.317 seconds and service wall time was
2.724 seconds; its CUDA control completed in 0.149 seconds, but token zero
already differed (`496` versus `9079`).

The Qwen placement certificates report 9,000 OP15 OpenCL compute nodes and
3,060 OP12 OpenCL compute nodes, with only declared OP15 `GET_ROWS` work on
CPU. Gemma placement is similarly clean, so placement alone does not repair
its numerical failure. A same-host CUDA split at cut 30 reproduces Gemma's
monolithic top-5 exactly, which narrows the failure to the heterogeneous phone
route or backend numerics rather than the layer boundary.

`warm_tier_controller.py` consumes the hash-bound intents and derived route
readiness. With current evidence it exits nonzero with
`E_ROUTE_NOT_READY`; with test-only PASS routes it exercises all nine
transitions through drain, load, catch-up, edge drain, reprepare, and atomic
ready publication. No physical promotion is authorized.

## W0 batch and transport screen, 2026-07-23

The Qwen workers remained resident for three sessions: B1 and B8 ended with
`DETACH`, and B32 ended with `STOP`. OP15 ran `[0,30)` and OP12 ran `[30,40)`.
Every cohort generated the same eight greedy tokens as the frozen CUDA
control.

| Requests | P50 service | Request rate | Gain over B1 | Activation payload |
|---:|---:|---:|---:|---:|
| 1 | 3.825 s | 0.261 req/s | 1.00x | 0.47 MiB |
| 8 | 24.859 s | 0.322 req/s | 1.23x | 3.75 MiB |
| 32 | 87.548 s | 0.365 req/s | 1.40x | 15.00 MiB |

The B32 prefill call contains 160 token rows. Each of its seven decode calls
contains 32 rows. This proves static cohort batching and persistent session
reset. It does not yet prove continuous admission across unequal arrivals,
inter-stage overlap, or a useful serving throughput.

Weights were provisioned by USB ADB before the paid route and loaded from phone
UFS. Runtime hidden states traveled by WiFi TCP through the host coordinator:
OP15 to host, then host to OP12. The route reports did not capture interface
counters or socket peers, so the exact path remains a post-run operator record.
See `TRANSPORT_CONTRACT.md`.

The host relay is the W0 control, not the target path. Direct OP15-to-OP12 WiFi
reachability is confirmed in both directions with zero packet loss. The next
runtime gate keeps host-side scheduling and ownership but sends reserved
activation batches directly to OP12.

`summarize_qwen_batch_wifi.py` re-derives the batch certificate from all three
reports, both worker logs, the CUDA control, the shard manifest, and the run
context. The derived readiness is `PROVISIONAL_BATCH`, and the controller still
refuses it.

## W1 direct phone chain, 2026-07-23

An additive relay now runs on OP15. The host sends token and lineage rows to
the relay, OP15 computes `[0,30)`, the relay sends each F32 cut activation
directly to OP12 over WiFi, OP12 computes `[30,40)`, and only terminal token
rows return to the host.

The corrected relay binary ran persistent B1 and B32 sessions on both phones.
All 264 generated-token checks matched the same-artifact CUDA sequence. B32
again formed one 160-row prefill batch and seven 32-row decode batches. The
direct relay moved 7,864,320 activation bytes to OP12 and reported zero host
activation payload bytes.

This is a mechanics result, not a latency result. The corrected direct B1
point was 5.653 seconds versus 5.696 seconds for one separately loaded
host-relay control, only 0.76 percent lower. Direct B32 was 90.796 seconds.
Process-to-process variation is larger than the apparent B1 difference, and
phone compute remains dominant. The current status is
`DIRECT_CHAIN_MECHANICS_PASS_REPEATS_PENDING`.

The relay enforces model identity, a contiguous cut, route capacities, lineage,
sequence cleanup, and persistent reset. It still relies on a single-client
implicit reservation. Interface counters and a host-issued cryptographic
batch descriptor were not captured, so W1 does not promote Qwen to an eligible
route. See `RESULTS_W1.md` and `results/w1_direct_phone_chain/`.

## W2 mixed-phase direct chain, 2026-07-24

The direct route now has a bounded phase-aware admission queue for one fixed
model and cut. Decode rows are physically ordered first. Prefill rows can fill
the remaining row capacity until the batch knee or earliest dispatch deadline.
Accepted groups are atomic, queue capacity is finite, and expired prefill
cannot starve behind a continuing decode stream.

One real-device run first admitted 16 Qwen requests, then submitted 16 live
decode rows with 80 new prefill rows. The batcher emitted one 96-row
`llama_decode` with the 16 decode rows first and all 80 prefill rows after
them. It continued all 32 sequences with six B32 decode calls and one final
B16 call. All 256 generated-token checks matched the CUDA sequence.

The physical batch sequence was:

```text
80P, 16D+80P, 32D, 32D, 32D, 32D, 32D, 32D, 16D
```

The route moved 7,864,320 activation bytes directly from OP15 to OP12 and zero
activation bytes through the host. Worker placement, session-step
conservation, sequence cleanup, and relay row counts passed the independent
reducer. Status is `DIRECT_MIXED_BATCH_MECHANICS_PASS`.

W2 does not yet prove a throughput or latency gain. Its 23.219-second maximum
is much lower than W1's 90.796-second unordered B32 point, but ordering,
prefill shape, and device state changed together. The next acquisition is a
matched sorted-versus-shuffled row-order control. Inter-stage overlap remains
unimplemented because the relay permits only one in-flight batch.

## Controls

- `C0 server_queue`: drain A, load B, then begin B work.
- `C1 phone_finish`: phones serve B during the switch; phone-started requests
  finish on phones.
- `C2 catchup_handoff`: phones serve B while CUDA loads; CUDA batch-prefills
  committed histories, consumes the token delta, and takes ownership.
- `C3 rotating_warm_tier`: C2, followed by phone preparation of A and one
  reverse A promotion.

## CP0 - Freeze eligible models

- [x] Hash exact desktop sources and derived phone shard artifacts.
- [ ] Verify identical tokenizer, template, KV, RoPE, context, and sampler
      configuration.
- [ ] Prove full collective-phone execution for model A.
- [ ] Prove full collective-phone execution for model B.
- [ ] Record zero-swap memory and prepared-image bytes on both phones.
- [x] Record the first placement and correctness screen against same-artifact
      CUDA controls.
- [x] Record token-exact Qwen B1/B8/B32 cohort mechanics with persistent
      workers.
- [x] Record token-exact direct OP15-to-OP12 B1/B32 mechanics with zero host
      activation payload.
- [x] Record one token-exact direct mixed decode/prefill batch with live KV,
      bounded wait, and deterministic decode-first row order.
- [x] Reject storage-only, provisional, incorrect, or partial-server routes.

Exit: both models have complete CUDA and phone route certificates. If Qwen3
fails, select a smaller already-supported second decoder before changing any
runtime.

## CP1 - Measure the transition budget

- [ ] Measure desktop model unload and load from warm host cache.
- [ ] Measure desktop model load from cold NVMe.
- [ ] Measure phone ready-to-first-token at `N={1,8,32}`.
- [ ] Measure phone continuous decode rate and useful batch candidates.
- [ ] Isolate sorted versus shuffled sequence order with matched workers,
      batch shapes, and thermal brackets.
- [ ] Add bounded downstream credits and overlap OP15 batch `k+1` with OP12
      batch `k`.
- [ ] Measure CUDA batched prefill of the exact S39 histories.
- [ ] Measure phone rewarm from local UFS after releasing the other model.

Primary gate:

```text
phone_warm_TTFT < desktop_model_ready_time
```

If it fails, retain only the low-rate phone-residency hypothesis: phones must
avoid a GPU switch rather than bridge it.

The W0 cohort screen is not this gate. It records end-to-end service time but
does not separate first-token readiness, repeated-process variance, or
continuous-arrival throughput.

## CP2 - One-request catch-up

- [ ] Add a versioned ownership record with request, model, route, owner, epoch,
      committed position, and token-history digest.
- [ ] Keep the phone route decoding while CUDA loads.
- [ ] Snapshot one committed frontier.
- [ ] Reconstruct native CUDA KV from prompt plus committed token IDs.
- [ ] Consume any phone-generated token delta without publishing it twice.
- [ ] Commit one atomic phone-to-CUDA ownership transition.
- [ ] Continue at least 32 CUDA tokens.

Use greedy decoding. Direct KV import is out of scope. Any duplicate, missing,
stale, or conflicting token fails the gate.

## CP3 - Batched catch-up

- [ ] Run `N={1,8,32}` live phone requests.
- [ ] Batch CUDA history reconstruction.
- [ ] Keep phones authoritative during reconstruction.
- [ ] Catch up the delta and switch the group at a token boundary.
- [ ] Admit unequal prompt and output lengths.
- [ ] Prove exact conservation and sequence-slot cleanup.

Exit: one CUDA batch absorbs independently progressing phone requests, and a
useful `N` beats sequential reconstruction.

## CP4 - Rewarm the displaced model

- [ ] Drain and release all phone state for promoted model B.
- [ ] Load and prepare model A shards from phone-local UFS.
- [ ] Publish a new boot/residency/prepared-image readiness generation.
- [ ] Reject stale B readiness and incomplete A readiness.
- [ ] Execute a reverse A promotion and catch-up.

Exit:

```text
A hot / B warm
-> B hot / A unavailable
-> B hot / A warm
-> A hot / B warm
```

No step may claim both models executable in phone RAM unless physical
measurement proves dual residency.

## CP5 - Trace replay

- [ ] Run C0-C3 over identical request identities and token budgets.
- [ ] Predeclare a finite promotion and hysteresis sweep.
- [ ] Preserve model-A steady-state SLO while model B is warm.
- [ ] Record every queue, load, phone batch, replay batch, delta, ownership
      transition, and rewarm event.
- [ ] Run failure injections for phone disconnect, CUDA load failure, stale
      epoch, and rewarm failure.

Report:

- switch-period TTFT and SLO goodput;
- maximum inter-token handoff gap;
- useful phone tokens during loading;
- model unload/load, replay, catch-up, and rewarm times;
- phone and CUDA batch distributions;
- duplicate/missing/stale token counts;
- bytes by payload class;
- number of avoided GPU switches;
- selected-GPU energy only after mechanics pass.

## Decision

S39 passes the mechanism gate only if C2 or C3:

- conserves every request and token exactly once;
- has zero stale ownership or false readiness;
- improves switch-period P95 TTFT or SLO goodput by at least 20 percent over C0;
- does not regress the steady-state hot-model SLO;
- completes phone rewarming before the measured reverse-switch opportunity.

Phone, host, network, and total-system energy remain unknown. A latency pass
does not imply an energy pass.
