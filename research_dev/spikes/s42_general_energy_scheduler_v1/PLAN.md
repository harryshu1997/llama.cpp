# S42 General Energy Scheduler V1

## Goal

Build a bounded, device-agnostic scheduler prototype for heterogeneous model
inference. The primary objective is total fleet energy. Server capacity and
latency are secondary objectives. The first physical binding is the RTX 4060
Ti plus OP15 and the 74-request two-model BurstGPT-derived trace.

This spike is outside `llama-server`. Another workstream owns the server and
phone execution integration. S42 only defines the scheduling contract,
calibrates route profiles, replays traces, and exposes a small decision API.

## Design boundary

The runtime selects among certified execution alternatives. An alternative
may represent:

- a complete request on one device;
- a contiguous layer island; or
- an operator island whose split, merge, transport, and residency behavior
  was measured before admission.

The runtime does not invent tensor cuts or extrapolate an unqualified kernel.
New devices and workloads enter through data, not scheduler source changes.

The unified successor extends this boundary with adaptive model registration.
An unseen model may enter through predicted routes generated from its graph and
calibrated device models. Predicted routes are distinguished from measured and
stable routes, retain conservative bounds, and are promoted only after shadow
or physical validation. Expensive cut enumeration belongs to model
registration, not the per-token scheduling loop.

Each route binds:

- workload identity and granularity;
- required resources and concurrency slots;
- a measured latency model with a one-sided error bound;
- total energy and uncertainty, when a valid synchronized boundary exists;
- quality class, placement, residency, and measurement gates;
- server busy-time and resident-memory cost for capacity decisions; and
- evidence paths and hashes.

## Policy modes

- `control`: use the baseline route for every request.
- `enforce`: select offload only when its energy upper bound is at least 5%
  below the baseline lower bound, its latency upper bound is within 5% of the
  baseline, its quality is sufficient, and its SLO remains feasible.
- `shadow`: make no energy claim; choose the fastest qualified alternative to
  identify physical measurements worth running.
- `capacity`: minimize server busy time and then server resident memory while
  preserving the latency and SLO gates. This mode makes no energy claim.
- `adaptive`: keep an immediately available baseline; when it is queued, use
  only measured alternatives for verified energy savings, deadline recovery,
  or a configured minimum tardiness reduction. Otherwise fail closed to the
  baseline.

An unknown or estimated energy boundary can never authorize an `enforce`
offload decision. The baseline is always the fail-closed fallback.

## First build-test-refine loop

1. Import the existing physical CPU-only and CPU-plus-OP15 BurstGPT results.
2. Fit nonnegative token-shape latency models and retain the maximum observed
   positive residual as the scheduling upper-bound additive.
3. Replay the same 74 arrivals under all five policy modes.
4. Compare predicted route counts, makespan, SLOs, and model error with the
   physical campaigns.
5. Probe the live RTX 4060 Ti and OP15 identities without changing device
   state or interfering with the server integration work.
6. Refine only model terms whose errors are exposed by physical evidence.

## V1 pass conditions

- deterministic decisions and replay output;
- exactly one baseline per workload;
- no unmeasured, unplaced, nonresident, or quality-incompatible route is used;
- `enforce` cannot use estimated or missing energy evidence;
- `enforce` cannot use a multi-resource split without measured overlap whose
  conservative exposed join wait is at most 5%;
- composite CPU, phone, and USB resource queues are accounted once;
- lower power with higher joules is rejected;
- the BurstGPT adapter preserves all 74 arrivals and token shapes; and
- focused tests cover fail-closed gates, queueing, uncertainty, and objectives.

## Explicit non-claims

S42 V1 does not claim production integration, semantic equivalence,
multi-phone scaling, or a universally accurate latency model. The fleet-energy
claim is limited to the exact synchronized and integrated I3 cohort; any wider
route requires its own physical energy and correctness evidence.

External download ending in a durable model file is treated as an offline
one-time cost. Runtime loading, device transfer, repacking, switching, and
residency changes remain part of scheduling latency, energy, and capacity.

## Completion status

The certified-route gate V1 is complete. The implementation, physical replay,
live readiness overlay, limitations, and successor acquisition are recorded in
`RESULTS.md` and `INTEGRATION.md`. The hierarchical placement compiler is now
implemented in `placement_planner.py`, but production integration and the full
four-engine physical profile are not complete. I3 supplies a synchronized
three-pair accounted-device energy profile for one fixed BurstGPT workload
mix. General per-shape energy enforcement remains fail-closed outside that
measured profile epoch.

The unified package now owns the canonical scheduling contracts,
compatibility adapters, general policy engine, runtime gates, placement
compiler, operator split, offload and phone transport contracts, hash-bound
execution plans, matmul virtual queue, and residency planner. The old S42
module paths are compatibility imports. Public-symbol identity tests cover
every migrated engine, and mixed-trace consumers import the unified package.
On 2026-08-08, a fresh physical control/treatment pair configured both
backends from unified scheduler plans and passed the strict paired validator.

All successor optimization loops follow `PHYSICAL_LOOP_CONTRACT.md`. Frozen
I0 iterations use `physical_iteration.py`; a separately identified successor
trace uses an equivalently strict, trace-bound aggregate validator. Every
headline uses only real-device latency, energy, completed work, SLO, and
phone-work values.
Energy is the sum of synchronized CPU-package, GPU-board, and whole-phone
average watts times the common paid trace duration. Split routes additionally
require arithmetic-mean overlap counters and at most 5% exposed join wait.

## Current physical integration gate

The llama-server integration now has a bounded balanced FFN policy, normal
desktop CPU settings, arithmetic-mean branch counters, runtime-library
bindings, fail-closed phone loss, synchronized server and phone energy, and a
pinned MMLU64 quality gate. The final policy and campaign are recorded in
`../s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/RESULTS_LLAMA_SERVER_I3_ENERGY_V1.md`.

Three alternating source-length pairs are complete. Every real CPU control
and OP15 treatment completed 74 requests and 11,605 output tokens. On average,
the treatment reduces makespan by 14.38%, server CPU-package plus GPU-board
energy by 17.30%, and accounted server-plus-phone energy by 16.76%. Each
treatment executes 52,320 phone calls with zero worker, transport, swap, or
cleanup failure.

This acquisition is a successor workload to physical loop I0: it uses
`REQUESTS_SEMANTIC_SOURCE.jsonl`, SHA-256
`b20a9ba66ee3558d835a0e19ed3cfa4c31a4a9e8b4f9c085b29a14f80250a0ff`,
with 33,843 input and 11,605 output tokens. It does not alter the frozen I0
trace or its 2,175-token results.

The exact measured workload/profile epoch has the evidence needed for bounded
approximate `enforce`: mean exposed join wait is 2.67%, the fleet-energy
saving exceeds 10%, latency and SLO gates pass, and MMLU64 is 27 / 64 in both
arms. It is installed as an exact-workload cohort route rather than assigning
shared continuous-batch energy additively to requests. The unified execution
plan now configures the server, bridge, worker, split, DMA-BUF, and USB path.
It remains ineligible for exact-token service or universal per-shape use. Rare
M=1 through M=3 buckets individually exceed the 5% wait target. The next
declared integration change is an atomic pre-dispatch runtime snapshot that
includes phone thermal state, followed by shape-bucket qualification and a
held-out arrival-mix replay.
