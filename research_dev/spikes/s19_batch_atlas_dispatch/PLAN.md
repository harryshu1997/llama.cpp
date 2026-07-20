# S19 batch atlas + variable-cohort dispatcher: CP0 contract

Status: contract frozen before acquisition on 2026-07-19. This file is CP0. It is
written before any CP1 atlas row or CP2 decision is recorded.

## 0. Directory note (concurrent effort)

This task ("real device batch atlas and online variable-cohort dispatcher for
R0/R1") was assigned the path `research_dev/spikes/s19_dynamic_batch_runtime/`.
At CP0 that directory was already occupied by a live concurrent agent building a
DIFFERENT and later S19 effort: a distributed token-boundary CONTINUOUS batching
runtime (its PLAN.md and SERVER_REUSE_AUDIT.md; its REJECT list even names this
task's boundary, "treating a fixed B32 cohort as continuous batching"). To avoid
destroying that concurrent work through colliding PLAN.md / RESULTS.md /
SHA256SUMS.txt filenames, and to honor "preserve all pre-existing dirty changes,"
this task's deliverables live in the sibling directory
`research_dev/spikes/s19_batch_atlas_dispatch/`. The concurrent
`s19_dynamic_batch_runtime/` directory is left exactly as its author had it.

This spike is the measured-atlas + variable-cohort precursor. It explicitly does
NOT implement continuous batching; that is the concurrent effort's job and the
next checkpoint after this one.

## 1. What this spike is

Turn the S18 hard-coded B32 geometry into a measured batch atlas plus a first
live online mechanism that selects a batch size from that atlas and drives real
OP15, OP12, and one selected A6000 with changing cohort sizes. It is not a
simulator. The CP2 result is labeled VARIABLE_COHORT_BATCHING and executes one
static cohort per persistent exchange.

Authority: MIXED_WORKLOAD_DESIGN.md (system/claim boundary) ->
TWO_LEVEL_SCHEDULER.md (online policy, lazy claims, atlas semantics) ->
NEXT_PLAN.md (this is the "measured per-device batch candidates and online
selection" step that precedes the shared multi-ingress tail).

Goals:

1. measure device-specific batch service curves (CP1 atlas);
2. select a batch size online from certified measurements (CP2 policy);
3. drive real OP15, OP12, and the selected A6000 with changing cohort sizes,
   priorities, synthetic SLOs, memory limits, and downstream credits (CP2 live);
4. preserve request conservation, exact tokens, placement, and route ownership.

Explicitly NOT in scope (stop after CP2 for review):

- token-boundary continuous batching (admission/retirement mid-batch);
- KV-slot manifests across activation boundaries;
- shared multi-ingress CUDA tail (still two tail images per S18);
- S17 R2 hierarchical route or phone-to-phone fan-in;
- weight streaming / dynamic residency generations;
- any energy acquisition (phone, USB, host-wall, GPU-board, total).

## 2. Devices and boundaries (frozen)

~~~text
selected A6000 : GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f  (nvidia-smi index 0)
idle A6000     : GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf  (must stay idle;
                 checked before and after every acquisition step)
OP15 (v81)     : adb serial 3C15AU002CL00000   Hexagon v81 / HTP0
OP12 (v75)     : adb serial 5ae7a43d           Hexagon v75 / HTP0
~~~

- CUDA processes are pinned with CUDA_VISIBLE_DEVICES to the selected UUID.
- GPU1 utilization and memory are sampled before and after each acquisition step;
  any non-zero experiment residency on GPU1 invalidates that step.
- Independent R1 routes only (no serial phone chain, no R2 funnel):

~~~text
CUDA full model R0 : CUDA [0,48)            (fallback / control / high-priority)
OP15 R1            : OP15 [0,8)  -> CUDA [8,48)
OP12 R1            : OP12 [0,6)  -> CUDA [6,48)
~~~

- Do not modify gemma4.cpp, llama-graph.cpp, ggml_backend_sched, kernels, or any
  model internals. layersplit.cpp is used AS IS for CP1 (existing persistent
  drivers; no C++ change needed). If CP2 needed a layersplit.cpp change it would
  have to be additive and avoid the graph/scheduler/kernel path; the plan needs
  none.
- Preserve all pre-existing dirty changes. No commit or push. ASCII only.

## 3. Pinned identities (measured at CP0, 2026-07-19)

~~~text
full model  gemma-4-12B-it-f16.gguf   sha256 bac4e29337273d1cb41aa36f4209ebcf63baa1162e6ab4dc55f6034b3d69b35a
OP15 head   12b-f16-head-0-8.gguf     sha256 a74845294bc1a3cba6ac26c747615d64d9127c904aae04c43c7ea2ebc1ea25c8
OP12 head   12b-f16-head-0-6.gguf     sha256 d507b7bb453242dff12ba1ce0add53189755b8a9a2b960743ead1578d8f1a6b5
host binary llama-layersplit (CUDA)   sha256 da12f9255d2e7acf276c3803cbfab2e7e2aaed1ed50230a43bd8bd1f79f2c8f7
phone binary llama-layersplit (arm64) sha256 d26075bcf64e90ee04e51c2f86188d709e4c2906add252d209014ad1e046646c
~~~

The host binary is frozen into artifacts/ before hashing. Phone binaries are the
frozen S14 persistence build already deployed to /data/local/tmp/ls-s14-persistent
(NOT ls-s14-cpe, which is not touched). Every batch_atlas row and every CP2
decision binds these hashes. Evidence never points at a mutable build-*/bin path.

## 4. Frozen candidate batches and the exclusion of B1/B2

~~~text
B_CANDIDATES = {4, 8, 16, 24, 32, 48, 64}
~~~

- Only configurations that pass the CP1 support/memory preflight are executed.
- B1 and B2 are NOT target operating points. They are executed only as an
  explicitly labeled urgent correctness control (route "urgent") and never
  populate an eligible atlas operating-point row. A CP2 decision may pick B1/B2
  only under the urgent policy branch and only when a B1/B2 urgent-control row
  is present and certified.
- No interpolation. The dispatcher may select only a batch value that has an
  eligible atlas row for the exact (device, route, context envelope). An
  unmeasured batch is rejected fail-closed.

## 5. Frozen input envelope

~~~text
prompt_text     : "Explain batching."   (utf-8)
prompt_sha256   : c544378e66bf7a3d6640824b9e4cc28deaef2e35a424772b56b95d27c17fe6c8
n_gen           : 8 generated tokens per request
driver_context  : 16 tokens per request (KV envelope per sequence)
driver_max_pref : 8   (max prefill bound; invariant max_prefill + n_gen <= context
                       holds: 8 + 8 <= 16)
~~~

This is the S18 frozen envelope (input_manifest.json digest
ea1f2d47b8c6dec6096c8b95b6073181ecba6838d97741ca92ec2ceca6937858), reused for
continuity so CP1 batch curves are comparable to the S18 B32 point. The small
16-token envelope avoids reserving unused 512-token KV and was the S18 fix that
let two CUDA tails fit; it still covers the complete 8+8 request lifetime.

## 6. CP1 atlas row schema (frozen): s19-batch-atlas-row-v1

Every atlas row is a JSON object with exactly these keys:

~~~text
schema              "s19-batch-atlas-row-v1"
route               "CUDA_R0" | "OP15_R1" | "OP12_R1" | "urgent"
device              selected A6000 UUID or phone adb serial
backend             "CUDA0" | "HTP0"
layer_range         [start, end)  phone head cut, or [0,48] for CUDA R0
host_layer_start    tail entry layer (0 for R0, 6 for OP12, 8 for OP15)
model_hashes        { full, head }   sha256 identities from section 3
binary_hashes       { host, phone }  sha256 identities from section 3
context_envelope    { context, max_prefill, n_gen, prompt_sha256 }
batch               integer in B_CANDIDATES (or 1/2 for urgent control rows)
support             true | false      (decode ran without kernel/shape error)
memory_ok           true | false      (no OOM; peak within budget)
oom                 true | false
selected_gpu_peak_mib   integer, selected A6000 peak memory during the row
correctness         { control_token_ids, route_token_ids, exact_match }
placement           { scheduled_placement_ok, missing_buffer, cpu_get_rows,
                      compute_nodes }
latency_us          { p50, p95, p99, route_wall_samples: [ ... ] }
throughput_tok_s    float = batch * n_gen / (p50_route_wall_us / 1e6)
thermal             { phone_start_c, phone_end_c }   (nulls for CUDA R0)
process_evidence    { screen_processes, measured_exchanges,
                      worker_pid, worker_boot_nonce, reset_applied,
                      eligible_level }   eligible_level in {"screen","eligible"}
verdict             "ELIGIBLE" | "ELIGIBLE_SCREEN" | "INELIGIBLE"
ineligible_reason   null or one of: unsupported, oom, inexact_tokens,
                    placement_fail, gpu1_busy
raw                 { worker_log, host_log, result_jsonl, nvml_before,
                      nvml_after }  relative artifact paths
~~~

Row verdicts:

- ELIGIBLE: support and !oom and memory_ok; exact_match true (route tokens equal
  the SAME-batch CUDA control tokens; CUDA R0 is its own control); placement ok
  (scheduled_placement_ok, missing_buffer==0, only declared GET_ROWS on CPU);
  eligible_level=="eligible" (7 measured exchanges + 3 fresh-process reloads);
  GPU1 idle before and after. The dispatcher may select these.
- ELIGIBLE_SCREEN: all of ELIGIBLE except only screen level (>=3 measured
  exchanges, no fresh-process replication). Selectable with a screen caveat.
- INELIGIBLE: fails support, memory, correctness, or placement. Never selectable.
  Same-batch numeric divergence from the CUDA control is inexact_tokens
  (INELIGIBLE), not approximate equivalence.

## 7. CP1 measurement protocol (frozen)

Reuse the existing persistent LayerSplit paths (no new C++):

- CUDA R0 control at batch B: `--mode monodriver --persistent-jsonl
  --driver-batch B --driver-context 16 --driver-max-prefill 8 -n 8`, model
  resident, one JSON command per exchange, DETACH between exchanges, STOP last.
  Its token ids ARE the same-batch control oracle for every phone route at B.
- OP15 R1 / OP12 R1 at batch B: a resident `stagenet` worker sized at B
  (`--driver-batch B --driver-context 16 --driver-max-prefill 8 -n 8`) plus a
  host `--mode pipedriver --host 127.0.0.1 --port P --persistent-jsonl
  --driver-batch B ...`. adb reverse maps host 127.0.0.1:P to the phone worker.

Constraint discovered and honored: validate_stage_chain requires the phone
worker HELLO params (n_seq_max, n_ctx_seq, n_batch, n_ubatch) to EQUAL the host
driver params. A resident worker is therefore bound to one batch size; the phone
worker is relaunched once per batch B (intra-B persistence still holds: one load
serves all measured exchanges and DETACH resets for that B).

Per candidate (route, B):

1. preflight: one isolated exchange to confirm support (decode rc==0), memory
   (no OOM, capture selected-GPU peak), and same-batch token match. A failing
   preflight records an INELIGIBLE row with the reason and the batch is not
   measured further.
2. same-batch CUDA control: the CUDA R0 row at B provides control_token_ids for
   the phone routes at B.
3. screen: >=3 measured exchanges (DETACH-reset between). Because each host and
   CUDA driver loads the 23GB full model, a "process" is realized as a resident
   driver performing >=3 independent DETACH-reset exchanges; the resident phone
   worker identity (pid, boot_nonce) is recorded and held constant across them,
   and every exchange records reset_applied via the SESSIONCERT. This resident
   multi-exchange realization of "independent process" is a declared limitation
   (section 9), forced by the 23GB model-load cost; it is labeled honestly and
   not presented as fresh-OS-process independence.
4. knee + adjacent: the throughput knee per route and its two adjacent supported
   candidates are raised to 7 measured exchanges AND replicated across 3 fresh
   host OS processes (3 model reloads) to expose cross-process/thermal variance;
   only then is eligible_level=="eligible".
5. placement: LAYERSPLIT_PLACEMENT_CERT=1 on every process; require
   SCHEDULED_PLACEMENT_OK, missing_buffer==0, HTP0-only + declared GET_ROWS on
   the phone and CUDA0 + declared CUDA_Host GET_ROWS on the tail/control.
6. thermal: phone HTP/skin temperature captured at start and end of each phone
   row. Selected-GPU peak memory captured for every row.
7. no energy acquisition.

## 8. CP2 dispatcher contract (frozen): fail-closed variable-cohort

Inputs consumed at each decision epoch:

~~~text
compatibility_key   (model, route family, context envelope, KV owner class)
ready_request_ids   concrete READY request ids
priority            "high" | "low"   (synthetic, labeled)
earliest_slo_us     earliest latest-finish deadline across the ready set
atlas               eligible s19-batch-atlas-row-v1 rows (CP1 output)
free_kv_slots       per device free KV/memory slot count
phone_lane_credits  per phone remaining exchange credit
usb_credits         activation/result USB credits
cuda_tail_credits   reserved downstream CUDA-tail credits per route
~~~

Policy (fail-closed):

1. never select a batch without an eligible atlas row for that exact
   (device, route, context envelope);  -> reject "unmeasured_batch".
2. never launch a phone batch without reserved downstream CUDA-tail credit and
   sufficient free KV/memory and phone/USB credit;  -> reject and try next route
   or fall back to CUDA R0 / terminal.
3. urgent work (tight earliest_slo, or explicitly urgent) uses the smallest
   certified feasible batch, or B1/B2 urgent control, or R0 fallback.
4. memory-bound decode chooses the LARGEST useful measured batch whose
   conservative finish is before the earliest latest-start; a queue larger than
   the selected point is split into measured microbatches (each a certified
   batch value); a queue smaller than the preferred point waits only within SLO
   slack, else releases at the smaller certified batch.
5. deadline-forced partial release: when the earliest latest-start is reached the
   lane releases whatever certified batch is feasible now, even below preferred.
6. request conservation: every ready request reaches exactly one terminal
   outcome (completed, fell_back, rejected_terminal); none dropped or duplicated;
   verified at shutdown.
7. stale epochs: a decision whose route/residency/session epoch does not match
   the live worker generation is rejected fail-closed.
8. worker failure: a failed exchange yields an explicit fallback (R0) or a
   terminal failure record; never a fabricated success.

Every decision emits one s19-dispatch-decision-v1 record (section 9). The live
run executes one static cohort per persistent exchange (VARIABLE_COHORT_BATCHING)
and drives real devices.

## 9. CP2 decision log schema (frozen): s19-dispatch-decision-v1

~~~text
schema            "s19-dispatch-decision-v1"
epoch             monotonic integer decision index
now_us            synthetic monotonic clock (integer; deterministic)
event             "arrival" | "form_cohort" | "release" | "complete" | "shutdown"
ready_before      sorted ready request ids at decision start
compatibility_key string
priority          "high" | "low"
earliest_slo_us   integer
selected_route    "CUDA_R0" | "OP15_R1" | "OP12_R1" | "urgent" | null
selected_batch    integer | null
cohort_request_ids sorted request ids committed to this cohort (or [])
microbatch_plan   list of certified batch values when a queue is split
reason_code       one of the frozen reason codes below
credits_after     { free_kv, phone_lane, usb, cuda_tail } snapshot
epochs            { route, residency, session } committed for the cohort
outcome           "dispatched" | "waited" | "rejected" | "fell_back" |
                  "completed" | "terminal_failure"
executed          null OR { device, worker_pid, worker_boot_nonce,
                            route_wall_us, token_ids_sha256, session_end,
                            placement_ok } for a real exchange
~~~

Frozen reason codes: unmeasured_batch, insufficient_memory,
no_downstream_credit, no_phone_credit, urgent_small_batch, r0_fallback,
form_largest_useful, split_microbatch, wait_for_batch, deadline_release,
stale_epoch, worker_failure, conserve_shutdown.

The decision log is deterministic: identical inputs produce byte-identical
decision_log.jsonl across PYTHONHASHSEED values (all ordering is explicit sort;
no set/dict-iteration order reaches serialized output).

## 10. Declared limitations (frozen before results)

1. "Independent process" is realized as resident-driver DETACH-reset exchanges
   plus 3 fresh-OS-process reloads only at the knee+adjacent rows, because every
   driver loads the 23GB full model. This is weaker than 7 fresh OS processes per
   batch and is labeled as such on every row.
2. Two CUDA tail images still exist for the two phone cuts (S18 carryover). Peak
   selected-GPU memory is reported, not reduced. No HBM-relief claim.
3. No continuous batching: one static cohort per persistent exchange.
4. No energy of any kind.
5. Phone same-batch exactness is empirical; batches that diverge from the CUDA
   control are INELIGIBLE and reported, not interpolated or rescued.
6. OP12 has historically shown a two-exchange credit and reload fragility; the
   dispatcher treats phone_lane_credits as a hard fail-closed input and the CP1
   atlas records whatever OP12 batches actually certify.

## 11. Required CP2 tests (frozen)

Unit / adversarial, deterministic, no device:

- unknown/unmeasured batch rejected (unmeasured_batch);
- insufficient memory rejected (insufficient_memory);
- absent downstream credit -> phone launch rejected (no_downstream_credit);
- B1/B2 rejected unless urgent and explicitly certified;
- earliest SLO forces release (deadline_release);
- no request loss, duplication, or double completion (conservation);
- stale route/residency/session epoch rejected (stale_epoch);
- deterministic decision log across PYTHONHASHSEED values;
- worker failure produces fallback or terminal failure, never fabricated success.

A fail-closed validator independently re-derives conservation, checks every
decision against the atlas and the frozen policy, recomputes credit accounting,
and rejects any decision that selected an unmeasured batch or launched a phone
cohort without downstream credit.

## 12. Live CP2 arrival-varying scenario (frozen)

One deterministic arrival schedule (synthetic, integer microsecond clock) whose
decision_log must contain at least:

- different selected batch sizes over time;
- at least one wait_for_batch then form a larger cohort;
- at least one deadline_release below the preferred point;
- at least one OP12 refusal from memory/SLO/credit;
- at least one r0_fallback;
- at least one epoch where OP15 and OP12 cohorts launch concurrently on real
  devices;
- queue conservation verified at shutdown.

Real executions use the resident persistent workers (one static cohort per
exchange). Pure-refusal scenarios (unmeasured, no-credit, stale) need no device
execution; dispatching scenarios (R0 fallback, concurrent OP15+OP12, varying
batch) execute real exchanges and record executed{} evidence.

## 13. Verdict labels and stop rules

- CP1 verdict: `S19_CP1_BATCH_ATLAS_MEASURED` with a per-route eligible set, or
  `S19_CP1_ATLAS_PARTIAL` if coverage is narrowed by device limits (honestly
  scoped, not silently truncated).
- CP2 verdict: `S19_CP2_VARIABLE_COHORT_BATCHING_MECHANICS_PASS` (fail-closed
  dispatcher drives real devices through the frozen scenario with full request
  conservation) or a FAIL with the failing gate named. Never CONTINUOUS_BATCHING.
- Stop after CP2 for review. Do not start shared-tail integration,
  token-boundary continuous batching, R2, weight streaming, or energy.
- Stop and report if the atlas cannot certify at least one eligible operating
  point per phone route, or if the dispatcher cannot preserve request
  conservation.
