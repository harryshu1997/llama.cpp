# S10-V0 Results: Q-PIM power-frontier falsification screen

## VERDICT: FAIL

Date: 2026-07-15 (ASCII only). No commit, no push. CP5 physical reproduction was
NOT run (the contract forbids it unless all analytic gates pass; they do not).

Q-PIM's dependency-aware power-frontier mechanism does not survive the S10-V0
falsification screen on the measured A6000 + OP12 + OP15 hardware. The perfect-future
oracle (C4) beats the optimized server-only baseline (C1) by >= 15% only under a
mechanism-FAVORABLE, physically-UNMEASURED (blocked) phone/USB power model AND a
degenerate all-distinct-weight slack-rich cohort; under the conservative
measured-plausible power model C4 = +0.0% in EVERY load bin (the oracle never
offloads). Where a favorable-power gain appears it is NOT explained by any of the
three certified causal levers - it is plain per-island offload substitution
("skipped GPU-us"), which the design explicitly excludes and grants no energy
credit. The required causal levers are physically unavailable here (measured): the
A6000 has no power state below its auto-entered P8 idle (25 W), power caps are not
settable without root, and offloading only ever SHRINKS server batches, never
enlarges them. The physical energy boundary is ENERGY_BLOCKED.

Programmatic verdict: `artifacts/cp_verdict.json` (`scripts/verdict.py`).

Only PASS authorizes PF1; FAIL means stop Q-PIM runtime work and preserve S9 as
transport/residency substrate.

---

## Evidence classes

This screen mixes evidence types. They are labeled everywhere:

- MEASURED  : taken from real A6000 / OP12 / OP15 hardware this session.
- DERIVED   : computed from MEASURED values (e.g. the L_ffn(M) latency table).
- INFERRED  : an unmeasured quantity given an explicit, mechanism-FAVORABLE value
              (chosen to MAXIMIZE offload benefit, so a FAIL is conservative).
- SIMULATED : the oracle/policy schedules and their energies (analytic model).
- BLOCKED   : cannot be measured or resolved to 10% in this setup.

---

## CP0 - integrity and physical boundary: ENERGY_BLOCKED

Full record: `artifacts/cp0_integrity.txt`, `artifacts/cp0_*`.

- HEAD 933c722f6 (unchanged). Pre-existing dirty tree preserved byte-for-byte
  (`artifacts/cp0_preexisting_tracked.diff`; re-checked identical after all edits).
- Pre-edit tests: phone-pim CTest release 3/3, ASan/UBSan 3/3. Post-edit: 3/3, 3/3.
- Smallest certified resident dense-FFN reproduced with the CERTIFIED worker
  (0a50ca72..e749, intact both phones): OP15 PRESTAGED_FFN_PASS rel-L2 2.924e-4;
  OP12 PRESTAGED_FFN_PASS rel-L2 2.947e-4. (`artifacts/cp0_ffn_op{15,12}.json`.)
- Devices: 2x RTX A6000 (driver 580.159.03), OP12 5ae7a43d/HTP v75/USB 6-2,
  OP15 3C15AU002CL00000/HTP v81/USB 8-3; adb 127.0.0.1:5037.

PHYSICAL BOUNDARY = ENERGY_BLOCKED (MEASURED reasons):
1. No synchronized external meter for the complete boundary (host CPU/DRAM/PSU +
   A6000 + USB + phone/charger). Only A6000 board power (nvidia-smi) is available.
2. That board sensor samples at ~1.5 Hz; the relevant GPU islands run in single-digit
   ms, so it cannot resolve a 10% effect on them.
3. A6000 power caps / application clocks are NOT settable (root required; no
   passwordless sudo) -> a POWER_CAP or clock/P-state trigger cannot be actuated.
4. The A6000's only low-power state is P8 idle (~25 W), entered AUTOMATICALLY; there
   is no scheduler-triggerable deeper sleep. PLAN CP2: do not proceed to a physical
   energy gate when no measured lower state exists.
5. Phone charger/VBUS draw is not measurable (USB rail pinned; no root OP12; no
   WiFi-adb). Phones draw from host USB, inside the (unmetered) host wall power.

Consequence: the best achievable verdict is MECHANISM_PASS_ENERGY_BLOCKED, and only
if the analytic mechanism screen passes. It does not (below), so the verdict is FAIL.

---

## CP2 - minimum measured atlas

Full record: `artifacts/cp2_summary.txt`, `artifacts/cp2_atlas.json`, raw
`artifacts/cp2_a6000_ffn_latency.jsonl`, `cp2_a6000_power_trace.csv`,
`cp2_a6000_sustain.jsonl`. Harness: `scripts/s10_gpu_ffn_atlas.cpp` (reuses the exact
certified island `examples/phone-pim/phone_pim_ffn.cpp` on CUDA0).

MEASURED - A6000 FFN island (gemma4 dense FFN blk.2, n_embd 3840, n_ff 15360, f16
weights 354 MB), on-device compute p50 vs batch M (reps=300, p05==p50==p95):

| M | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|----|----|----|-----|-----|-----|------|
| us | 517 | 525 | 529 | 532 | 545 | 599 | 615 | 700 | 998 | 1842 | 3625 |

Bandwidth-bound to ~M64 (fixed 354 MB weight read), then compute-bound. Per-token:
M16 34.1 us, M256 3.9 us, M1024 3.5 us; marginal (M16->M256) 1.887 us/token.

MEASURED - A6000 board power: idle P8 ~25.0 W (floor); sustained back-to-back FFN
active mean 281.2 W, ceiling 299.8 W (near the 300 W cap).

MEASURED - phone FFN island (identical graph, M=16, warm resident): OP15 e2e 27.14 ms
(compute 12.17, transport 14.89); OP12 e2e 45.00 ms (compute 16.40, transport 28.59);
rel-L2 2.9e-4 both. The A6000 does the same island in 631 us e2e -> phone is 43x
(OP15) / 71x (OP12) slower end to end.

DERIVED - integer-us L_ffn(M) table stored as DATA in every instance (oracle and the
independent checker both look it up; neither re-derives interpolation).

INFERRED (mechanism-FAVORABLE): phone SoC 2.0 W (real 2-6 W), USB/host relay 0 W
(real > 0), A6000 active-for-energy 300 W (ceiling), transitions free/instant.
Conservative counterpart (also evaluated): phone 6 W, USB/host 8 W, A6000 281 W,
server idle counted.

BLOCKED: complete wall energy; A6000 power-vs-cap curve; phone charger/VBUS.

BREAK-EVEN GAP (PLAN CP2 t_break_even = wake + transition_energy/(idle - lower)):
UNDEFINED - there is no measured lower state below auto-P8, so (idle - lower) has no
value. The SLEEP/POWER-cap mechanism has no physically available lower state here.

The decisive per-token asymmetry (all measured/favorable-inferred): even at the most
favorable settings the phone costs 3.4 (OP15) / 5.6 (OP12) mJ/token vs the A6000's
batched 0.57 mJ/token. A LONE (unbatchable) server FFN, however, pays the full
354 MB weight read (~0.16 J at 300 W); whether offloading such a lone island ever
wins therefore hinges ENTIRELY on the BLOCKED phone/USB power (favorable: phone
0.054 J < server 0.164 J; conservative: phone 0.38 J > server 0.15 J).

---

## CP1 - frozen controlled instance

`scripts/freeze_instance.py` -> `fixtures/frozen/`. Three DAG templates (T_gen
server-pre -> phone-FFN -> server-suffix; T_service stateless server-pre -> phone-FFN;
T_rag on a second weight set), >= 2 model_id and >= 2 weight_set values, mid_tokens
fixed at the only phone-certified point (16), exact activation bytes
(M*n_embd*4 = 245,760 B in/out at M16), an explicit horizon, and explicit
release/deadline/wave records. Primary representative cohort = 12 requests over 2
waves, m0-heavy with a rare m1 (batchable repeats + a lone rare weight). Sweep grid
(`fixtures/frozen/sweep/`, 44 instances): load bins x weight-mix
{batchable, twomodel, lone} x slack {tight, slack} x power model {favorable,
conservative}. All instances embed the DERIVED L_ffn table and MEASURED phone e2e.

---

## CP3 - exact tiny oracle + independent checker

- `oracle/oracle.py`: deterministic exact enumeration over MID route placements
  (symmetry-reduced to per-class device-count compositions - exact) and per-group
  server batch/split choices, scored by the fixed lexicographic objective
  (L1 SLO misses, L2 timeouts, L3 -met, L4 total wall energy). Emits a canonical
  certificate with a sha256. `oracle/model_data.py` is the shared simulator/energy
  model (used by oracle + policies only).
- `checker/checker.py`: STANDALONE. Imports no solver / candidate-generator /
  simulator / objective module. Re-derives every constraint and the energy from the
  instance + certificate alone and fails closed (nonzero). Validates: instance and
  certificate sha256; one terminal outcome per request; precedence and READY-before-run
  (pre>=release, mid>=pre, suf>=mid = D2H-before-credit); fixed op latencies and
  L_ffn(sum tokens) batch latency; server single-machine and phone single-lane
  non-overlap; batch legality (one weight_set, one wave, tokens==sum); energy
  recomputation (exact integer nJ); outcome vs deadline/horizon; activation-memory
  peak <= bound; and the HBM rule (mirrored earns no HBM credit, exclusive only).
- Adversarial mutation self-test (`fixtures/mutation_tests.py`,
  `artifacts/cp3_mutation_selftest.txt`): 13/13 CAUGHT, 0 slipped. Covers every
  PLAN mutation: removed dependency, execution-before-READY, omitted D2H/transfer,
  double-counted phone/USB energy, HBM credit in mirrored mode, free/instant server
  state transition, server claim after latest_start, activation-memory overflow,
  duplicate/stale completion, unfinished-work-omitted-at-horizon, plus two integrity
  (body-tamper, instance-binding) mutations.
- >= 1000 generated fixtures (`fixtures/gen_fixtures.py`,
  `artifacts/cp3_fixtures_summary.jsonl`): 1200 deterministic tiny instances; the
  oracle solved all 1200 and the INDEPENDENT checker ACCEPTED all 1200 valid certs
  (0 wrongly rejected) and REJECTED all 1200 deterministically corrupted copies
  (0 wrongly accepted). Fail-closed self-test exit 0.

---

## CP4 - policies C0..C5 and the opportunity sweep

`policies/policies.py` (C0 eager server-only FIFO; C1 optimized server-only DAG order
+ lazy batch + power control [BASELINE]; C2 fixed phone placement, no shaping;
C3 frontier shaping without power-trigger credit; C4 perfect-future Q-PIM oracle;
C5 bounded causal Q-PIM per-wave beam over arrived nodes only). Runner
`scripts/run_cp4.py` re-VALIDATES every one of the ~300 policy certificates with the
independent checker (all pass). Full rows: `artifacts/cp4_results.jsonl`; gates
`artifacts/cp4_gate_summary.json`.

C4/C5 relief vs C1 (SIMULATED energy; % = (E_C1 - E)/E_C1):

CONSERVATIVE power model (measured-plausible): C4 = C5 = +0.0% in ALL 22 bins
(batchable, twomodel, lone; every load; every slack). The perfect oracle offloads
NOTHING - the phone (6 W + 8 W USB) is more expensive than the A6000 for every island,
lone or batched.

FAVORABLE power model (blocked, mechanism-favorable): gains appear ONLY by offloading
lone/rare islands. Representative points:

| instance (favorable) | C4 | C5 | C5/C4 | larger server batch? |
|---|---:|---:|---:|:--:|
| primary (12 req, 1 rare weight) | +20.4% | +20.4% | 1.00 | NO |
| batchable (1 weight), any load | +0.0..+14.8% | same | 1.00 | NO |
| twomodel load6 slack | +20.4% | +20.4% | 1.00 | NO |
| lone load2 slack | +47.2% | +42.3% | 0.90 | NO |
| lone load3 slack | +34.6% | +28.2% | 0.82 | NO |
| lone load4 slack | +25.9% | +25.9% | 1.00 | NO |

`larger server batch?` is NO in EVERY instance in the entire sweep
(`any_C4_creates_larger_server_batch = false`). Offloading only removes work from the
server; it never densifies a batch. Batchable-heavy cohorts show ~0% because lazy
batching (available to C1) already amortizes the weight read - phones cannot help.

### Gate evaluation (`artifacts/cp4_gate_summary.json`, `artifacts/cp_verdict.json`)

- OPPORTUNITY gate (C4 >= 15% in two adjacent load bins, SLO no worse): met ONLY in
  favorable + lone + slack (load pairs [2,3] and [3,4]). NOT robust: 0.0% under the
  conservative model everywhere. Requires the degenerate all-distinct-weight cohort,
  slack SLOs, AND the blocked-favorable power simultaneously.
- CAUSAL gate (C5 >= 10% and retains >= 2/3 of C4): met in those same favorable-lone
  bins (moot, see mechanism).
- MECHANISM gate (a MEASURED larger server batch / lower cap / break-even low-power
  interval must explain the benefit, and C5 timely phone islands > C2): FAILS. None of
  the three levers is available or triggered (measured): no A6000 state below auto-P8;
  caps unsettable; offload never enlarges a batch. The favorable-power gain is
  skipped-GPU-us per-island offload, which the design excludes and credits at zero.
  C5-timely > C2 also fails in the first hit pair.
- PHYSICAL gate: ENERGY_BLOCKED.

---

## CP5 - physical reproduction: NOT RUN

The PLAN/HANDOFF run CP5 only after all analytic gates pass. The opportunity gate is
not robust and the mechanism gate fails, so CP5 was deliberately not run. The
smallest resident-FFN correctness case WAS reproduced on both phones at CP0.

---

## Why the mechanism fails here (measured, not asserted)

1. The A6000 batches almost for free: M16->M256 (16x tokens) raises compute only
   545->998 us (1.83x); marginal ~1.887 us/token. C1's lazy batching already captures
   this, so there is no batch-densification headroom for phones to unlock.
2. The A6000 has no scheduler-triggerable low-power state below its automatic P8 idle
   (25 W), and its caps are not settable, so SLEEP_BUNDLE / POWER_CAP_BUNDLE have no
   physical actuator. The break-even gap term is undefined.
3. Offloading a lone island to a phone is ~43-71x slower and, per token, 6-10x more
   energy than the A6000's batched marginal even at the most favorable phone power. It
   can only "win" against a lone unbatchable server FFN, and only if the BLOCKED
   phone/USB power is assumed at its favorable extreme (it reverses sign under the
   conservative model).
4. Phones add ~1-2 lanes of ~27-45 ms islands; at any non-trivial load they are a tiny
   fraction of A6000 throughput (consistent with prior S5), so the offloadable share
   -> 0 as load rises and C4 -> C1.

The apparent favorable-power opportunity is exactly the design's own excluded case:
"skipped GPU-us receives no energy credit"; a phone bundle earns credit only when a
MEASURED batch or A6000 power-state change lowers total wall energy. Here no such
change exists or is measurable.

---

## Deviations / blocked cells / remaining blockers

- ENERGY is BLOCKED end to end: every energy number is SIMULATED under an explicit
  INFERRED power model. No total-wall joule was measured. This is the primary blocker.
- A6000 power caps and P-states could not be actuated (no root), so power-cap / sleep
  bundles are untested physically and were excluded from the oracle (modeling an
  unmeasured lower state is forbidden by the PLAN).
- The `lone` mix load was capped at 4 so C4 stays exactly enumerable; higher loads add
  no new mechanism (offload share only falls with load).
- CP5 not run by design (gates failed). No sustained thermal run performed.

## Integrity

- HEAD 933c722f6 unchanged; no commit/push/stage/reset/revert.
- Pre-existing dirty files preserved byte-for-byte (verified identical post-edit).
- Protected scope untouched: no edits to tools/server, llama-server, model graphs, KV,
  ggml_backend_sched, HTP/OpenCL/CUDA kernels, or protocol-v3 wire. All new files live
  under `research_dev/spikes/s10_power_frontier/`. The CP2 harness is a standalone
  measurement target that only reuses the existing island source and public ggml/CUDA
  APIs; it added no build target to the tree.
- Certified worker 0a50ca72..e749 intact on both phones; pushed model_s10.gguf removed.
- File hashes: `artifacts/cp_file_hashes.txt`. Tests/exit codes: `MANIFEST.md`.

STOP for human review. Verdict FAIL: do not build the Q-PIM runtime / PF1.
