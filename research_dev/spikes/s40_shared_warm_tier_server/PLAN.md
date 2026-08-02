# S40 shared warm-tier server experiments

Status: `OFFLINE_MECHANICS_AND_DESKTOP_SMOKE_PASS_A_ONLY_ACQUISITION_BLOCKED`

## Implementation checkpoint

The common server controller, GPU/CPU/phone gateway contracts, trace driver,
event reducer, and physical-run manifest are implemented and under
fail-closed testing. Initial routes start `ABSENT`; a separate activation
request loads the prospectively selected routes before admission. Controller
event v3 binds every issued command to the raw executor result, its disposition
(`RECEIVED` or `QUARANTINED`), and the subset of output actually committed.
Quarantined output cannot count as useful work.

The fresh-process Python bridge failed the transport diagnostic and is no
longer an allowed physical executor. The final direct persistent C++
Unix-socket executor was rebuilt and measured on the exact RTX 4060 Ti host.
Its P95 was 574,042 ns at fanout 1 and 1,215,926 ns at fanout 8. Both values
were below the corresponding direct-control P95, so incremental P95 was zero.
A strong performance result requires:

```text
incremental_bridge_p95 <= min(1 ms, 5% of fastest desktop EXECUTE quantum)
```

The frozen transport-v2 record binds the final native benchmark binary, peer
PID and start time, socket path, exact argv, and executor bundle. The two
desktop models also pass real B1 and B8 smoke through the native router on the
exact RTX 4060 Ti. Physical A_ONLY acquisition remains blocked while the V2.4
production plan, post-reboot identity/network binding, complete orchestration
source closure, and no-model preflight are finalized.

The internal controller capability is an accidental-use and unauthenticated
HTTP guard, not an isolation boundary between mutually untrusted same-UID
processes. The router and desktop gateways receive its private token file;
phone gateways, model children, cache helpers, observers, samplers, and trace
drivers do not. Stronger isolation would require separate UIDs, namespaces, or
a peer-credential control socket and is outside this checkpoint.

## Purpose

S40 evaluates one experimental llama.cpp server controller with interchangeable
GPU, CPU, and phone executors. The controller, request bytes, scheduling policy,
sampling, and evidence format stay identical across new comparisons. Only
executor placement and cache regime may differ.

The implementation is private-fork research code, disabled by default. S40 does
not replace or modify the frozen S39 CP0-D evidence.

## Frozen workload

All new modes use the exact S39 desktop workload:

- 74 requests in the original arrival order;
- Qwen3-8B Q8_0 and Qwen3-14B Q4_K_M;
- exact prompt token arrays;
- greedy sampling;
- exactly 8 output tokens per request;
- a synthetic 30-second per-request SLO.

`EXPERIMENT_CONTRACT.json` binds the request, switch, and historical C0
artifacts by SHA-256. `validate_inputs.py` reopens and checks the request bytes.
It also binds the exact Qwen3-8B and Qwen3-14B bytes, the RTX 4060 Ti UUID, and
the common llama.cpp serving envelope from the frozen desktop contract.

## Common policy

The runtime policy is work-conserving and queue-aware:

1. Dispatch an arrived request immediately when a compatible READY executor has
   credit.
2. Preserve FIFO order within each model queue.
3. Choose the model of the oldest nonterminal demand not matching the current
   GPU model as the next GPU target. CPU or phone execution remains
   work-conserving but does not suppress promotion.
4. A proposal is advisory and does not stop compatible work. Starting a switch
   atomically closes new GPU admission, lets already active GPU work drain, and
   makes the target immutable. A proposed but unstarted switch may be coalesced
   when its target no longer has pending demand.
5. After every completion, readiness change, or transition, dispatch all newly
   feasible work before proposing another switch.
6. The synthetic 30-second SLO is a metric, never a cancellation deadline.
   Every mode uses the same 900-second campaign horizon from an immutable
   host-monotonic trace-start record. At that horizon, finalization atomically
   closes admission and switching, explicitly strands queued requests, and
   gives already active work at most 120 seconds to terminalize and finish
   cleanup. A run is evidence-complete only after finalization reports
   `FINALIZED` and the event ledger contains `run_end`.

The new policy is demand-derived. The first 14B demand at 1.95 seconds may
propose and start a switch immediately. The nine frozen S39 switch rows are
bound only as historical C0 provenance and never drive C1-C4, T1, or T2. T2
alone disables promotion.

`policy_oracle.py` is a deterministic reference for this policy. It is not a
second serving runtime. The server controller must produce decisions consistent
with the same invariants.

## Experimental matrix

`C0_EXISTING_GPU_SWITCH`

- Historical CP0-D warm-cache and cold-NVMe results.
- Uses the frozen non-coalescing target-change policy.
- Preserved byte-for-byte and never presented as the new controller.

`C1_GPU_ONLY_OPTIMIZED`

- One RTX 4060 Ti and one GPU-resident model at a time.
- No warm executor exists. Alternate-model requests stay queued until the
  common controller switches the GPU; no fake route represents this absence.
- Common queue-aware policy.
- Separate warm host-cache and cold-NVMe regimes.

`C2_GPU_PLUS_CPU_WARM_EXECUTOR`

- Hot model remains fully on the RTX 4060 Ti.
- Alternate model is executable from desktop CPU/RAM.
- The warm executor is explicitly `CPU_RAM`.
- CPU may serve alternate-model requests while GPU replacement proceeds.

`C3_DUAL_PARTIAL_OFFLOAD`

- One `GPU_CPU_PARTIAL` placement domain with two native child processes, each
  using a prospectively fixed GPU-layer allocation plus CPU RAM.
- Both logical model routes are READY before the trace. Promotion is disabled;
  this mode does not fabricate model-replacement semantics.
- The exact allocation must be locked before trace results.
- Requires at least 512 MiB measured GPU headroom and clean service. Failure to
  fit or serve is a valid result.

`C4_TWO_GPU_ORACLE`

- Run only if a second suitable GPU is already available without changing
  scope.
- Non-resource-matched upper bound, never the primary baseline.

`T1_PHONE_WARM_TIER`

- One model is hot on the RTX 4060 Ti.
- The alternate model is executable across OP15 and OP12.
- The warm executor is explicitly `OP15_OP12`.
- Phones admit and publish work while the GPU drains and loads.
- Path-matched token-history replay, `k_extra=0`, atomic ownership transfer, and
  local-UFS preparation of the displaced model are mandatory.

`T2_PHONE_NO_PROMOTION`

- Alternate model remains on the phones for one bounded run.
- Isolates phone capacity from promotion and replay benefit.

## Execution order

1. Pass deterministic controller and evidence tests with fake executors.
   Complete: S40 passes 253/253 tests; release and ASan/UBSan CTests pass 2/2.
2. Pass real desktop B1 and B8 smoke tests for both models.
   Complete: Qwen3-8B Q8_0 and Qwen3-14B Q4_K_M both pass through the native
   RTX 4060 Ti router and unload cleanly.
3. Qualify Qwen3-14B A_ONLY on RTX 4060 Ti, OP15, and OP12.
4. Only after A_ONLY passes, provision versioned Qwen3-8B shards and run B_ONLY.
5. Only after B_ONLY passes, run PAIR and both local-UFS reprepare directions.
6. Run one real A -> B -> A cycle through the shared server controller.
7. Run one shortened development trace for C1, C2, feasible C3, and T1.
8. Freeze commands, placements, policy, and evidence roots.
9. Run three alternating repetitions of primary C1, C2, and T1. Run T2 once,
   or three times when feasible.

Physical acquisitions are sequential and isolated. Any correctness, ownership,
cleanup, placement, memory, or identity failure stops the affected mode.
The primary campaign is a prospectively ordered v3 campaign. Its software lock
binds the controller, native transport benchmark, executor and evidence
bundles, Python, `ldd`, `nvidia-smi`, and the exact campaign-plan,
orchestration, run-manifest, and campaign-reduction sources. All repetitions
must use one host boot and one stable ID plus boot ID per device role. Selected
GPU exclusion uses one canonical lock derived from the GPU UUID plus a bounded-
cadence process observer; it does not claim exclusion of an overlap shorter
than the observer cadence.
Every new physical server command uses the prospectively frozen
`--threads-http 128`. Admissions are serialized only through their immediate
asynchronous acknowledgements, preserving the exact frozen order and arrival
schedule. Status polling is concurrent and cannot delay later admissions.
The value 128 is a frozen capacity setting, not a claim that 128 threads are
required for correctness.

Both physical preflight and independent manifest validation derive the serving
command from the frozen contract. The canonical command must contain exactly
one each of `--ctx-size 4096`, `--parallel 8`, `--batch-size 2048`,
`--ubatch-size 512`, `--flash-attn on`, `--cont-batching`,
`--cache-type-k f16`, `--cache-type-v f16`, and `--split-mode none`.
Aliases, duplicate flags, equals-form overrides, and negative continuous-
batching flags are rejected. Desktop executor evidence must apply the same
check to each realized child process command line.

## Metrics

The reducer derives:

- completed and stranded requests;
- SLO count and SLO goodput;
- queue delay, TTFT, and completion latency P50/P95/P99;
- per-model and total throughput;
- maximum model-publication and token-publication gap;
- drain, unload, load, replay, and commit timing;
- peak VRAM, controller-process RSS, host available RAM, host swap growth,
  controller CPU utilization, and host aggregate CPU utilization;
- phone memory, network bytes, peer identity, and thermal state;
- selected-GPU board energy from bracketed samples.

The global token-publication gap is measured only while demand is outstanding.
For every merged interval from a frozen scheduled arrival through that
request's completed or stranded terminal, the interval start and end are
included as boundaries around the ordered committed-token times. This includes
initial TTFT and final completion/stranding delay, but excludes idle time when
no request is outstanding. The reducer also reports the maximum gap between
tokens within one request.

The S40 reducer's initial selected-GPU integration is development-only. A final
selected-GPU comparison additionally requires the existing E2 instrument
quality, update-count, uncertainty, status, and paired-window gates.
Selected-GPU board energy is not server-wall or total-system energy. Phone and
total-system energy remain unknown until independently measured.

## Evidence

Every physical run must retain:

- exact command arrays and binary hashes;
- model and shard hashes;
- GPU UUID, phone serials, boot IDs, and socket peers;
- canonical controller event ledger;
- raw executor records and resource samples;
- ownership, cleanup, and reprepare records;
- a complete SHA-256 manifest.

The reducer derives conclusions from raw records. Stored verdict strings are not
trusted.

## Current stop

Do not run the comparison matrix or provision Qwen3-8B phone shards yet.
First freeze and test the production V2.4 A_ONLY plan plus its artifact,
fresh-readiness, and acquisition drivers. The plan must bind their captured
bytes and produce the complete raw A_ONLY role set; fabricated summaries are
not accepted. Then rerun exact device readiness immediately before one
Qwen3-14B A_ONLY acquisition and independently validate the resulting bundle.
C3 remains a clean feasibility failure until role-tagged raw dual-residency
evidence exists. C4 is skipped because no second suitable GPU is in scope.
