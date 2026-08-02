# S24 Fixed-Diamond Overlap and Handoff Design

Status: frozen before CP1 physical execution.

## Research claim

This spike screens one bounded mechanism:

1. An SLO decision selects one route from a finite route set.
2. The selected route remains pinned across prefill and decode.
3. OP15 continuously batches compatible layer-8 activations arriving from both
   the desktop CUDA prefix and OP12.
4. Every route reforms batches at the shared layer-16 CUDA tail.
5. No worker waits for a specific upstream cohort or a global barrier.

Ordinary layer splitting and continuous batching are not standalone novelty.
The mechanism under test is cross-source batch convergence with stateful,
SLO-selected routes.

## Physical ownership

The RTX 4060 Ti desktop owns:

- the SLO scheduler and request state;
- the CUDA prefix, CUDA middle, and shared CUDA tail workers;
- the OP12-to-OP15 relay when R2 is selected;
- runtime result collection and GPU-board measurement.

The A6000 host owns only:

- USB/ADB provisioning;
- phone worker process start, stop, and log collection;
- creation and hashing of phone shards;
- USB transfer of phone weights.

Neither A6000 GPU may be used. The A6000 host is not a runtime activation relay.

OP12 and OP15 remain USB-connected to the A6000 host for weights and control.
Their StageNet V3 endpoints are reached directly by the desktop over WiFi.

## Frozen graph

The only resident stages are:

| Worker | Layer range | Role |
| --- | --- | --- |
| cuda-prefix | [0,8) | nonterminal CUDA prefix |
| cuda-mid | [8,16) | nonterminal CUDA middle |
| op12-prefix | [0,8) | nonterminal phone prefix |
| op15-mid | [8,16) | nonterminal phone middle |
| cuda-tail | [16,48) | terminal shared CUDA tail |

The only routes are:

| Route | SLO class | Ordered stages |
| --- | --- | --- |
| R0 | tight | cuda-prefix -> cuda-mid -> cuda-tail |
| R1 | medium | cuda-prefix -> op15-mid -> cuda-tail |
| R2 | loose | op12-prefix -> op15-mid -> cuda-tail |

There are no arbitrary cuts, layer-8/layer-10 overlap, direct phone-to-phone
transport, dynamic weight streaming, or general DAG scheduling in S24.

## Route and row state

A request is identified by request_id and route_epoch. Admission creates one
immutable route binding. The binding contains the ordered resident stages and
remains live until every participating worker has removed its KV state.

Every physical row carries:

- request_id;
- route_epoch;
- the sequence ID leased on the receiving worker;
- token position;
- token ID;
- hidden activation for a non-prefix receiver.

Each participating worker leases a separate sequence ID. Sequence IDs are local
to a worker and are never treated as cross-worker identities. Position zero is
the first row on every stage, and later positions must be contiguous.

For prefill, each chunk is passed through every route stage before that request
advances to the next chunk. Decode feeds the terminal token back into the same
pinned route at the next position. A nonterminal stage must return a finite
hidden vector and no token. The terminal stage must return a token and no hidden
vector.

## Typed route runner

S24 adds a small ordered route runner rather than extending S22 run_request().
A resident-stage descriptor contains:

- worker name;
- exact layer range;
- endpoint;
- SequenceSlotPool;
- one DeviceBatcher-compatible queue;
- the serialized StageNet V3 client.

The runner validates adjacent layer boundaries and requires the final stage to
be terminal. It allocates all per-worker leases before the first row. On normal
completion or failure it removes KV from every leased worker before releasing
any corresponding software slot. Batch and remove RPCs share the same
per-socket serialization lock.

Cleanup is reverse stage order. Cleanup errors are retained as evidence and do
not silently release a slot whose KV removal was not acknowledged.

## Continuous convergence

R1 and R2 submit to one shared op15-mid batcher. Route-private OP15 batchers are
forbidden in the treatment. R0, R1, and R2 submit to one shared cuda-tail
batcher.

A row may join a physical batch only when its worker, boundary width, result
kind, and protocol shape are compatible. A batch is dispatched at the first of:

1. the measured batch knee;
2. the earliest row latest-safe-start deadline;
3. the bounded gather timer.

The batch knee is measured per worker and is not assumed to be B32. For each
bounded candidate batch size, use the median physical RPC time after warmup,
convert it to rows per second, and select the smallest batch reaching at least
95 percent of the peak observed rows per second. Every physical batch records
worker, start/end times, queue time, request IDs, route IDs, upstream worker
names, positions, and dispatch reason.

No queue waits for a named route, a full upstream cohort, or a global phase
barrier.

The synthetic CP4 convergence cases align route arrivals using the measured B1
enqueue time at the boundary under test. When the measured OP15 knee is above
B1, each upstream contributes fewer rows than that knee, and the shared queue
uses a fixed 250 ms gather bound. A mixed batch must therefore form from
compatible work already visible to the generic queue; no worker is told to wait
for a route or cohort. An OP15 B1 knee fails this physical mixed-batch screen
closed. These alignment delays apply only to the deterministic mechanism
screen. They do not alter either real-arrival trace.

## Physical correctness gates

The required executions are:

1. R0, R1, and R2 separately at B1.
2. R2 at B4.
3. Concurrent R1 and R2 through the shared OP15 queue.
4. Concurrent R0, R1, and R2 through the shared CUDA tail.

Each execution must have:

- finite outputs;
- stable same-route repeats;
- no missing activation buffers;
- an accepted placement certificate on every stage;
- exact request, epoch, sequence, and position lineage;
- zero live worker KV and zero software leases after completion;
- no CPU compute fallback except explicitly certified metadata operations.

Tokens and layer-8/layer-16 activations are compared with a CUDA control. The
known F16-phone/Q8-server boundary remains numerically uncertified unless it
passes the frozen numerical gate. Mechanics success must not be relabeled as
quality certification.

## Equal-work controls

All controls use the same arrivals, prompt token counts, decode step counts,
seeds, route-capacity limits, and completed-request denominator.

| Control | Definition |
| --- | --- |
| C0 | Every request uses R0. |
| C1 | R1 and R2 use route-isolated OP15 queues. |
| C2 | R1 and R2 share one OP15 convergence queue. |
| C3 | The SLO scheduler selects among R0, R1, and R2. |

The primary gates are frozen as follows:

- C2 must strictly increase pooled mean OP15 batch size or strictly reduce
  median makespan versus C1.
- For priority-0 requests, C2 p95 TTFT and p95 latency must each be no more than
  1.05 times C1, and C2 must introduce no additional priority-0 SLO miss.
- C3 must strictly reduce summed measured RTX 4060 Ti island compute time
  versus C0.
- RTX 4060 Ti GPU_BOARD energy is reported with the existing NVML method even
  when it does not improve.

For makespan and latency comparisons, the small deterministic control is run at
least three times and medians are used. Batch means pool all physical batches
from the same repetitions.

The equal-work control cohort is frozen to the first two deterministic requests
from each route class. The B1-measured OP15 input alignment delays and the same
250 ms OP15 gather bound are applied to C0, C1, C2, and C3. Thus C1 and C2
differ in OP15 queue ownership, not request identities, arrivals, work, or wait
bounds. The evaluator reopens each report and requires exact scheduled-arrival
equality before computing a benefit gate.

Phone energy, WiFi/network energy, A6000 host energy, and total-system energy
remain UNKNOWN.

## Workloads

The first workload is a deterministic three-class trace that exercises R0, R1,
and R2.

The dense mechanics trace is the pinned BurstGPT-derived S23 trace:

- 60 requests over 2 seconds;
- 21 arrivals at 0 seconds;
- 17 arrivals at 1 second;
- 22 arrivals at 2 seconds;
- source-derived arrivals and observed token counts;
- synthetic token values, priorities, and SLOs.

Its one-input-token/four-step execution remains labeled mechanics-only. A
separate 28-request context-compatible cohort uses observed input and output
lengths and is reported independently.

The 60-request mechanics trace uses C3 so the finite SLO policy selects and
pins each route under the real arrival pressure. The 28-request observed-length
cohort uses C2 route hints and 64-token prefill chunks so every context-valid
request is physically executed rather than rejected by a B1 proxy estimate.
Neither real-arrival workload uses a route-delay override.

## Stop condition

S24 ends with exactly one verdict:

- FIXED_DIAMOND_POC_PASS
- MECHANICS_PASS_NUMERICALLY_UNCERTIFIED
- BENEFIT_GATE_FAIL
- PHYSICAL_EXECUTION_BLOCKED

No layer-10 overlap or protocol redesign starts before this result is reviewed.
