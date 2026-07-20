# S11-B Results

Verdict:

`BATCH_MECHANICS_PASS; KV_CAPACITY_PASS; SERVER_THROUGHPUT_RELIEF_FAIL;
PHONE_LEG_STABILITY_NOT_ESTABLISHED; ENERGY_NOT_RUN`

## Exactness And KV

The existing two-phone route now runs on both connected phones and remains
exact:

```text
SERVER_ONLY tokens:    100,45518,107,236829
A0_OP15_OP12 tokens:   100,45518,107,236829
selected A6000 relief: 1288 MiB
```

The layer-window KV repair removes the dominant phone allocation defect:

| Stage | Old KV | Repaired KV at B=1 | Owned KV layers |
|---|---:|---:|---:|
| OP15 `[0,2)` | 1280 MiB | 4 MiB | 2 |
| OP12 `[2,3)` | 1280 MiB | 2 MiB | 1 |

The context is padded to 256 cells, so these are real backend allocations, not
an analytical estimate. Exact output still passes after the repair.

The selected-A6000 relief is:

```text
856 MiB resident weight relief + 4 MiB x B of KV relocated to OP15
```

Total KV is not eliminated; the owned layers' KV moves from A6000 HBM to phone
memory.

## Static Batch Sweep

Each row is one process pair with one measured group after one warmup group.
The server-only control and treatment use the same batch size.

| B | Server group ms | Phone route group ms | Server req/s | Phone route req/s | OP15 stage ms | A6000 tail ms | A6000 relief MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 159.619 | 314.395 | 6.265 | 3.181 | 154.962 | 159.396 | 860 |
| 2 | 166.957 | 368.980 | 11.979 | 5.420 | 205.152 | 163.820 | 864 |
| 4 | 175.763 | 392.632 | 22.758 | 10.188 | 217.513 | 175.115 | 872 |
| 8 | 204.341 | 497.402 | 39.150 | 16.084 | 296.433 | 200.961 | 888 |
| 16 | 277.966 | 627.816 | 57.561 | 25.485 | 354.614 | 273.191 | 920 |

The phone route's useful throughput improves 8.01x from `B=1` to `B=16`,
while group latency increases 2.00x. Batched execution therefore uses the phone
substantially better than batch-1 execution.

It is not additive A6000 throughput. At every measured `B`, the complete phone
route reaches only 41-51 percent of the equally batched A6000 control. The
server batches the same workload very efficiently, and the phone stage remains
serially upstream of the A6000 tail.

## Repeated B=8 Gate

Three ABBA process pairs were run with two measured groups per route and two
warmup groups:

| Metric | Server only | OP15 `[0,2)` + server |
|---|---:|---:|
| Exact requests | 48/48 | 48/48 |
| Median group wall | 202.068 ms | 452.356 ms |
| Group-wall CoV | 0.65% | 4.74% |
| Median useful throughput | 39.669 req/s | 17.118 req/s |
| Treatment/control throughput | - | 0.430x median |

OP15 reported thermal status 0. The maximum current HAL NSP temperature at
the ready and done snapshots was 33.7-34.1 C, so this short repeated run did
not encounter thermal throttling.

The complete route passes the frozen 5 percent wall-time stability gate. The
OP15 stage alone does not: its six group times have 8.52 percent CoV. This
short run therefore does not establish a stable sustained phone service rate.

## Correctness Bug Found During The Sweep

The first `B=8` attempt crossed the old 128-row physical microbatch boundary.
The stage requested dense unmasked cut activations while selecting only the
last prompt row of each sequence as an output. Output-row reordering then did
not describe the complete dense activation buffer, and exact output failed.

The final path keeps the bounded 224-row and 448-row prefills in one physical
microbatch and fails closed when the real prefill would exceed 512 rows. The
repaired `B=8` and `B=16` routes are exact.

## Decision

Keep static batched phone execution as a scheduler primitive, but do not route
ordinary work to it for latency or aggregate throughput. The slow loop may
select it only when its measured memory release, future server power result, or
an otherwise infeasible admission justifies the latency cost.

The next system experiment should use the virtual queue with varied prompts and
real arrival times. It must compare:

1. optimized server-only batching;
2. memory-pressure admission with no phone;
3. fixed phone placement; and
4. bounded queue-triggered phone batches.

Energy remains `NOT_RUN`. NVML can support an exploratory A6000-board result,
but phone and total-wall energy require external instrumentation.

All successful final-build runs use host binary
`786defa77b157aad6745aa2b37a0e1d11a2cb89a31ac938e5ea81d0694717c4d`
and Android binary
`9472537438612242b5ce9ce25aa739e531b0f1013af7245bfd5a0c3703728979`.
Across the final B sweep, repeated B=8 gate, and repaired two-phone checkpoint,
80 paired requests and 320 generated tokens per route are exact. This remains
one duplicated prompt at short context, not broad model-equivalence evidence.

## Post-Sweep Fail-Closed Repair

An adversarial code review found that the original mechanics path could:

- abort at `B=1` when the prompt exceeded `--driver-max-prefill`;
- accept token-only input at a middle stage and return a corrupted activation;
- report a premature stage EOF as success;
- terminate through `SIGPIPE`; and
- label an offloaded route even when a full-model host ignored the phone
  activation.

The current source closes those paths. A versioned stage hello now validates the
complete layer chain before any request, and the v2 runner derives evidence
eligibility from strict records instead of accepting handed labels.

A fresh hash-bound `B=8` run on the repaired source remains exact:

```text
server group:       204.764 ms, 39.069 req/s
OP15 route group:   449.656 ms, 17.791 req/s
treatment/control:  0.455x
A6000 relief:       888 MiB
energy:             NOT_RUN
```

This one-pair rerun validates repaired mechanics, not the whole historical batch
curve. The original sweep remains bound to the older binary hashes below. A
per-node backend-placement certificate is still absent, so the result does not
claim that every substantial stage op executed on HTP without fallback.
The wire hello binds topology and execution capabilities, not a cryptographic
model digest; model and binary identity are bound by the runner's pre-run hashes.

The repaired hello also passes the full two-phone chain:

```text
SERVER_ONLY:       159.671 ms, 6.263 req/s
OP15+OP12 route:   409.197 ms, 2.444 req/s
treatment/control: 0.390x
A6000 relief:      1288 MiB
exact work:        PASS
```

## Artifacts

Raw runs are under:

```text
scratchpad/s11_kv_filtered_b1_20260716/
scratchpad/s11_batch_b1_smoke_20260716/
scratchpad/s11_batch_b2_final_20260716/
scratchpad/s11_batch_b4_final_20260716/
scratchpad/s11_batch_b8_oneubatch_20260716/
scratchpad/s11_batch_b16_smoke_20260716/
scratchpad/s11_batch_b8_repeat3_20260716/
scratchpad/s11_runner_v2_final_r2_b8_20260716/
scratchpad/s11_runner_v2_final_r2_two_phone_b1_20260716/
```
