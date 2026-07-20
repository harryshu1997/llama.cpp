# S19 batch atlas + variable-cohort dispatcher: results

Verdict: `S19_CP1_BATCH_ATLAS_MEASURED` + `S19_CP2_VARIABLE_COHORT_BATCHING_MECHANICS_PASS`.
No energy of any kind was acquired. One selected A6000
(GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f), OP15 (Hexagon v81), and OP12
(Hexagon v75). This is VARIABLE_COHORT_BATCHING (one static cohort per persistent
exchange), NOT token-boundary continuous batching.

## Directory note

Assigned path `s19_dynamic_batch_runtime/` was already held at CP0 by a live
concurrent agent building a different, later S19 effort (distributed continuous
batching). To preserve that concurrent work, this task's deliverables live in the
sibling directory `s19_batch_atlas_dispatch/`; the concurrent directory was left
untouched. See PLAN.md section 0.

## What was built

- CP0: PLAN.md freezes devices, candidate batches B={4,8,16,24,32,48,64}, the
  atlas row schema (s19-batch-atlas-row-v1), the dispatcher contract, the
  decision-log schema (s19-dispatch-decision-v1), tests, and stop rules, all
  before acquisition.
- CP1: `atlas_measure.py` drives the existing persistent LayerSplit drivers (no
  C++ change) to measure device-specific batch service curves for CUDA_R0, OP15
  R1 ([0,8)->CUDA[8,48)), and OP12 R1 ([0,6)->CUDA[6,48)). `build_atlas.py`
  applies the same-batch correctness gate and emits `batch_atlas.json`.
- CP2: `dispatcher.py` (fail-closed variable-cohort dispatcher), `run_scenario.py`
  (canonical arrival-varying scenario), `validate_dispatch.py` (independent
  fail-closed validator), `tests/test_dispatcher.py` (9 unit/adversarial tests),
  `device_executor.py` + `live_devices.py` (real-device proof).

## CP1: measured batch atlas

Envelope frozen (all rows): prompt "Explain batching." (sha256 c544378e...),
n_gen 8, per-request context 16, max_prefill 8. Model gemma-4-12B-it-f16
(bac4e293...). Host binary da12f925..., phone binary d26075bc... (the frozen S14
persistence build under /data/local/tmp/ls-s14-persistent). Placement certified
on every exchange (LAYERSPLIT_PLACEMENT_CERT=1); GPU1 idle before/after.

Eligible operating points per route (a batch the dispatcher may select):

~~~text
CUDA_R0 : {4, 8, 16, 24, 32, 48, 64}   knee B64   peak 869 tok/s
OP15_R1 : {4, 8, 24, 32, 48}           knee B48   peak 123 tok/s
OP12_R1 : {4, 8, 24, 48, 64}           knee B48   peak  30 tok/s
~~~

CUDA_R0 (full model, selected A6000) - all exact by self-control:

| B | p50 ms | p95 ms | tok/s | peak GPU MiB | level |
|---:|---:|---:|---:|---:|---|
| 4 | 317.3 | 331 | 100.8 | 23525 | screen |
| 8 | 329.4 | 341 | 194.3 | 23897 | screen |
| 16 | 365.2 | 380 | 350.5 | 24635 | screen |
| 24 | 415.4 | 430 | 462.3 | 25397 | screen |
| 32 | 458.2 | 470 | 558.7 | 26131 | eligible (7 proc) |
| 48 | 506.7 | 520 | 757.8 | 27616 | eligible (7 proc) |
| 64 | 589.1 | 600 | 869.1 | 29101 | eligible (7 proc) |

OP15 R1 ([0,8) HTP0 -> CUDA [8,48)):

| B | p50 ms | tok/s | peak GPU MiB | exact vs CUDA B | verdict |
|---:|---:|---:|---:|---|---|
| 4 | 1388.1 | 23.1 | 20013 | yes | ELIGIBLE_SCREEN |
| 8 | 1988.3 | 32.2 | 20325 | yes | ELIGIBLE_SCREEN |
| 16 | 2242.9 | 57.1 | 20947 | NO | INELIGIBLE (inexact) |
| 24 | 2451.9 | 78.3 | 21595 | yes | ELIGIBLE (9 samp) |
| 32 | 2729.6 | 93.8 | 22219 | yes | ELIGIBLE (9 samp) |
| 48 | 3121.2 | 123.0 | 23471 | yes | ELIGIBLE (9 samp) |
| 64 | 3585.7 | 142.8 | 24720 | NO | INELIGIBLE (inexact) |

OP12 R1 ([0,6) HTP0 -> CUDA [6,48)):

| B | p50 ms | tok/s | peak GPU MiB | exact vs CUDA B | verdict |
|---:|---:|---:|---:|---|---|
| 4 | 2220.1 | 14.4 | 20884 | yes | ELIGIBLE_SCREEN |
| 8 | 3297.3 | 19.4 | 21215 | yes | ELIGIBLE_SCREEN |
| 16 | - | - | 21869 | - | INELIGIBLE (flaky hang) |
| 24 | 7176.8 | 26.8 | 22543 | yes | ELIGIBLE (9 samp) |
| 32 | - | - | 23113 | - | INELIGIBLE (flaky hang) |
| 48 | 12950.2 | 29.7 | 24517 | yes | ELIGIBLE_SCREEN |
| 64 | 16832.7 | 30.4 | 25833 | yes | ELIGIBLE_SCREEN |

Sweet region: CUDA_R0 throughput keeps climbing to B64 (869 tok/s, the largest
useful measured batch); OP15 saturates near B48 (123 tok/s); OP12 is nearly flat
from B24 (27 tok/s) to B64 (30 tok/s) and dominated by per-exchange latency.
Selected-GPU peak: CUDA B64 29,101 MiB; each phone tail 20-26 GiB. Two phone
tails run concurrently in the live proof, reproducing the S18 two-image tail
layout (~45 GiB) - not reduced here (see limitations). Phone NSP HVX/HMX thermal
spans OP15 27.9-34.9 C and OP12 28.3-34.5 C; no throttling observed.

Unsupported / OOM-prevented / inexact points:

- OP12 B16, B32: nondeterministic v75 HTP exchange hangs (the documented OP12
  reload/exchange fragility). Both hung on 2/2 multi-exchange attempts (screen +
  retry) despite a single B32 exchange succeeding in calibration; larger B48/B64
  certified in the same sweep, so this is transient flakiness, not a batch-size
  ceiling. Recorded INELIGIBLE, not rescued.
- OP15 B16, B64: correct-but-inexact - the HTP greedy path diverges from the
  same-batch CUDA control (near-tie argmax; batched-GEMM accumulation order).
  OP12 B64 by contrast MATCHES CUDA B64 (both land on the second path), showing
  the divergence is per-(device,batch), which is exactly what the gate certifies.
- No true OOM occurred at the frozen 16-token envelope; every eligible row's
  selected-GPU peak stayed within budget and GPU1 was idle before/after.

### Correctness gate (the load-bearing CP1 finding)

Each phone route row at batch B is exact only if its generated tokens equal the
SAME-batch CUDA_R0 control at B. This gate is real: at several batches the HTP
greedy path diverges from the same-batch CUDA path because the argmax is a
near-tie and the batched GEMM accumulation order differs between HTP and CUDA.
This is consistent with the S18 observation that CUDA itself produces a different
token sequence at B64 than at B32. Batches that diverge are INELIGIBLE
(inexact_tokens), never interpolated or rescued. S18 only exercised B32 (which
agrees on both phones); the atlas exposes the batches that do not.

## CP2: fail-closed variable-cohort dispatcher

The dispatcher consumes real READY requests plus the measured atlas and selects a
batch online. It never selects an unmeasured batch, never launches phone work
without reserved downstream CUDA-tail + phone-lane + USB credit and free KV,
splits oversized queues into measured microbatches, waits only within SLO slack,
forces partial release at the earliest latest-start, and never drops or
duplicates a request. Every decision is one s19-dispatch-decision-v1 record.

### Scheduling mechanics (deterministic, mock executor over the real atlas)

The canonical scenario (`run_scenario.py`) exercises, and its decision_log.jsonl
contains, every required behavior:

- different selected batch sizes over time;
- wait_for_batch then a larger cohort;
- split_microbatch (queue above the selected point split into measured batches);
- deadline_release below the preferred point;
- an OP12 refusal from an exhausted phone-lane credit (no_phone_credit);
- an R0 fallback for that refused cohort;
- a concurrent OP15 + OP12 launch epoch;
- request conservation verified at shutdown (every request reaches exactly one
  terminal outcome, none lost/duplicated/double-completed).

The decision log is byte-identical across PYTHONHASHSEED 0, 1, and 777 (all
ordering is explicit sort; serialization uses sort_keys).

### Independent validation

`validate_dispatch.py`, which shares no scheduling code with the dispatcher,
re-derives from the decision log + atlas + initial credits alone: schema and
reason-code validity, request conservation (no loss/duplication/double), that no
dispatched decision selected an unmeasured batch, that no phone cohort launched
without positive downstream credit, that B1/B2 appear only under the urgent
branch, and that no credit balance goes negative. It prints VALIDATION_PASS.

### Required tests (9/9 pass)

`python3 tests/test_dispatcher.py`:

1. unmeasured/unknown batch rejected;
2. insufficient memory rejected;
3. absent downstream credit -> phone launch rejected;
4. B1/B2 rejected unless urgent and explicitly certified (and the validator
   rejects a forged B2-outside-urgent log);
5. earliest SLO forces release (deadline_release);
6. no request loss, duplication, or double completion;
7. stale route/residency/session epoch rejected;
8. deterministic decision log across PYTHONHASHSEED values;
9. worker failure produces fallback or terminal failure, never fabricated
   success.

### Real-device proof (VARIABLE_COHORT_BATCHING)

`live_devices.py` drove the dispatcher-selected cohorts on real hardware. Six
exchanges, one static cohort each, all with exact tokens matching the atlas
same-batch control, HTP0/CUDA0 placement OK, and real worker identities:

| scenario | route | B | route wall | tokens match control | worker pid |
|---|---|---:|---:|---|---|
| varying batch | CUDA_R0 | 8 | 394 ms | yes | - |
| varying batch | CUDA_R0 | 16 | 426 ms | yes | - |
| varying batch | CUDA_R0 | 32 | 528 ms | yes | - |
| concurrent | OP15_R1 | 8 | 2067 ms | yes | 25833 |
| concurrent | OP12_R1 | 8 | 3375 ms | yes | 16639 |
| R0 fallback | CUDA_R0 | 8 | 394 ms | yes | - |

- Different selected batch sizes executed on real hardware (B8, B16, B32).
- The OP15 and OP12 B8 cohorts launched concurrently (distinct worker PIDs, their
  execution windows overlapped by 12.4 s of wall time) - a real simultaneous
  two-phone launch into two selected-A6000 tails.
- The R0 fallback for a credit-refused OP12 cohort executed on CUDA_R0 (real).
- `all_ok`, `all_token_match`, `all_placement_ok` = true;
  verdict `S19_CP2_VARIABLE_COHORT_BATCHING_MECHANICS_PASS`
  (`results/live_report.json`).

## Boundaries and limitations

- No energy (phone, USB, host-wall, GPU-board, total) was acquired.
- Two CUDA tail images still exist for the two phone cuts (S18 carryover); peak
  selected-GPU memory is reported, not reduced. No HBM-relief claim.
- No continuous batching: one static cohort per persistent exchange.
- "Independent process" is realized as resident-driver DETACH-reset exchanges
  plus fresh-process reloads at the knee+adjacent rows, because every driver
  loads the 23GB full model. This is labeled on every row and is weaker than 7
  fresh OS processes per batch.
- OP12 (v75) shows reload/exchange fragility (transient exchange hangs at some
  batches); the atlas records what actually certified, and the dispatcher treats
  phone-lane credit as a hard fail-closed input.
- validate_stage_chain requires the phone worker and host driver to share
  n_seq_max/n_batch/n_ubatch, so a resident worker is bound to one batch size;
  the phone worker is relaunched per batch (intra-batch persistence holds).

## Reproduce

~~~text
# CP1 atlas (per route)
python3 atlas_measure.py --route CUDA_R0 --batches 4,8,16,24,32,48,64 --measured 3 --tag screen
python3 atlas_measure.py --route OP15_R1 --batches 4,8,16,24,32,48,64 --measured 3 --tag screen
python3 atlas_measure.py --route OP12_R1 --batches 4,8,16,24,32,48,64 --measured 3 --tag screen
python3 build_atlas.py

# CP2 tests + validation + scenario
python3 tests/test_dispatcher.py
python3 run_scenario.py --atlas batch_atlas.json --executor mock --out results/decision_log.jsonl \
    --credits-out results/initial_credits.json --requests-out results/all_requests.json
python3 validate_dispatch.py --decisions results/decision_log.jsonl --atlas batch_atlas.json \
    --credits results/initial_credits.json --requests results/all_requests.json

# CP2 real-device proof
python3 live_devices.py --atlas batch_atlas.json
~~~
