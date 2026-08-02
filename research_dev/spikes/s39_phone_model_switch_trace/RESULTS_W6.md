# S39 W6 concurrent phone-delta catch-up

Status:
`CONCURRENT_DELTA_MECHANICS_PASS; MODEL_LOAD_AND_TASK_QUALITY_OPEN;
SCHEDULER_INELIGIBLE`.

## Question

Can the phones continue decoding after a snapshot while CUDA reconstructs the
snapshot state, then can CUDA ingest only the newly generated token delta and
take ownership without losing or duplicating a token?

This checkpoint tests that bounded mechanism at B8. It does not replace the
failed W4 Qwen2.5 Q8_0 task-quality gate.

## Frozen gates

`W6_DELTA_CONTRACT.json` fixes:

- batch 8;
- eight-token prompts;
- four phone snapshot tokens;
- two additional phone-delta tokens;
- eight CUDA continuation tokens;
- CUDA snapshot, delta, and control chunks of 8, 2, and 2 tokens;
- at least 80 percent overlap of the shorter concurrent leg;
- durable CUDA ownership before phone state release;
- a run-unique, hash-chained six-record ownership journal;
- `scope=MECHANICS_ONLY` and `scheduler_eligible_on_pass=false`.

Contract SHA-256:
`f9487a10d6ad79f7c6c7a7af7516b44c79caa533b78d4e239306aa855df01a30`.

The independent `W6_PHYSICAL_GATE.json` fixes the exact worker, relay, model,
and shard artifacts plus the permitted backend placements. Its SHA-256 is
`3303d1270a59399a5c5d90b01f7398b862b3a099ffa8a8b446825abde65aa39e`.

## Physical route

The phone route was OP15 GPUOpenCL `[0,30)` followed directly over WiFi by
OP12 GPUOpenCL `[30,48)`. The CUDA route used two local CUDA0 workers at the
same cut. All four workers used Qwen2.5 14B Q8_0 with the same model identity.

The execution order was:

1. phones generated four snapshot tokens for eight requests;
2. phones generated two more tokens while CUDA replayed the prompt and
   snapshot histories;
3. CUDA ingested the two exact phone-delta tokens;
4. the coordinator durably recorded CUDA ownership;
5. phone sequence state was removed;
6. CUDA generated eight continuation tokens;
7. a fresh full-history CUDA control generated the same continuation.

## Result

| Metric | Measured value |
|---|---:|
| Phone snapshot, 88 rows | 11,666,495 us |
| Phone delta, 16 rows | 2,698,645 us |
| Concurrent CUDA snapshot replay, 96 rows | 174,719 us |
| Overlap of shorter concurrent leg | 99.81 percent |
| CUDA delta ingestion, 16 rows | 40,304 us |
| CUDA continuation, 56 rows | 292,753 us |
| Full-history control plus continuation, 168 rows | 522,081 us |
| Post-frontier delta plus continuation | 333,057 us |
| Measured transition critical-path reduction | 36.21 percent |
| Continuation equality | 8/8 exact |

The CUDA snapshot replay completed while the phone route was still advancing.
After the phone frontier became available, CUDA needed only one delta batch
instead of seven full-history control batches. The 36.21 percent difference is
one matched mechanics run, not a repeated latency claim.

Every request published exactly:

```text
4 phone snapshot tokens
+ 2 phone delta tokens
+ 8 CUDA continuation tokens
```

The six durable journal phases are `PHONE_FRONTIER`, `CUDA_PREPARED`,
`CUDA_COMMITTED`, `PHONE_RELEASED`, `CUDA_CONTINUATION`, and `COMPLETE`.
The terminal owner is `NONE`, and all phone and CUDA sequence counters are
zero.

## Placement and provenance

The physical validator parsed the realized session certificates:

- OP15: 5,940 OpenCL nodes and 9 declared CPU `GET_ROWS` nodes;
- OP12: 3,609 OpenCL nodes;
- CUDA head: 15,840 CUDA0 nodes and 24 declared CUDA-host `GET_ROWS` nodes;
- CUDA tail: 9,624 CUDA0 nodes;
- all routes: zero missing compute buffers and exact executed row counts.

OP12 printed the known compile diagnostic for an optional split
flash-attention OpenCL variant. Its realized graph still completed entirely on
OpenCL, which the physical validator checks from the session certificate.

The first physical acquisition passed token mechanics but was rejected as the
reportable result because its original certificate did not consume placement
logs. The selected rerun added a prospective physical gate and passed it.

Selected evidence:
`results/w6_phone_cuda_delta/run_20260725T001551Z/`.

The report SHA-256 is
`954c085324140b78de5b2fb078b7e714d2c7e15dbab6ea2b048aeecf1c52d83f`.
The certificate SHA-256 is
`34f55c40435d033d8fb5cc9955ff45580950ec30aa394d69801b67e4aea029ab`.
The run-context SHA-256 is
`67b580b86365c2c78e7754ecf1cd7d5527619b93b67fc131c91c96b6328d90a5`.
`SHA256SUMS.txt` verifies all logs, records, reports, and journal entries.

Offline verification after the placement hardening:

```text
W6 focused tests: 18 PASS
S39 full suite:   175 PASS
```

## Limits and next gate

Both CUDA model workers were resident before the token probe began. W6 proves
that state reconstruction can overlap phone progress; it does not prove that
model loading can be hidden behind phone service.

The route remains scheduler-ineligible because:

- the independent W4 task-quality gate failed;
- only fixed B8, equal histories, greedy decoding, and eight CUDA tokens ran;
- crash recovery from each partial journal phase is not implemented;
- no model unload/load, reverse rewarm, trace, SLO, or energy gate ran.

The next bounded checkpoint is a real switch control: start without the Qwen
CUDA workers, measure CUDA model readiness while the phones keep serving, then
apply the certified W6 replay/delta seam. Compare switch-period TTFT against a
server-queue control. No energy claim is authorized.
