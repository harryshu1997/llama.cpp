# S9-V0 Plan: Dynamic Weight Residency for a PIM-Style Phone Accelerator

Follow-on update (2026-07-14): the original design-only scope below is preserved
as executed history. A later user-authorized slice added the bounded pre-staged
runtime under `examples/phone-pim/` and documented it in
`../s9_phone_pim_runtime/`. The dynamic downloader, scheduler, and energy claims
remain unimplemented. The static bundle contract is now append-only v5; see
`V0R3_REPAIR.md`.

Status: S9-V0 (contracts + schemas + simulator + tests) executed; S9-L0 link
evidence captured; S9-V1 link-model correction planned below. This is a DESIGN +
VALIDATION spike. No phone daemon, downloader service, server library, kernel, or
production scheduler is built. No energy, novelty, or PIM-hardware claim is made.
Nothing committed or pushed.

Version note: this file preserves the V0/V1 plan. The later R1/v3 fail-closed
closure is in `V0R1_REPAIR.md`. Active two-level scheduler policy and gate order
are in `../../TWO_LEVEL_SCHEDULER.md` and `../../NEXT_PLAN.md`.

## Goal

Design and validate a HOST-MANAGED phone accelerator where model weights are
prefetched into phone UFS, promoted into LPDDR, prepared for HTP/OpenCL, leased,
and executed under server commands. PIM-STYLE, not literal cache-coherent PIM: the
phone cannot read arbitrary server memory; the host transfers weights and boundary
tensors. OP12 and OP15 now both negotiate USB 3.2 Gen 1 (5 Gbps raw endpoint
ceiling) on separate SuperSpeed root buses. Practical ADB staging goodput is also
measured: host-to-phone-file medians are 215.9 MiB/s on OP12 and 261.9 MiB/s on
OP15; the concurrent fleet aggregate is 409.4 MiB/s. These are provisioning-path
rates that include filesystem ingestion effects, not raw transport rates
(`CURRENT_SLOW_LINK.md`).

## Scope + guardrails (honored)

- Work only under `research_dev/spikes/s9_dynamic_weight_residency/`; doc updates
  allowed to `research_dev/MIXED_WORKLOAD_DESIGN.md`, `research_dev/NEXT_PLAN.md`,
  and `research_dev/talks.md`.
- Do NOT modify model graphs, llama-server, ggml backend scheduling, Hexagon/OpenCL
  kernels, or production transport code. Preserve the dirty worktree. ASCII only.
- Synthetic fixtures only until S8 Gate A is independently confirmed passing; no
  real-trace residency result is claimed.

## Deliverables (this spike)

Checkpoint 1 -- Substrate audit:
- `SUBSTRATE_AUDIT.md`: file/line-cited REUSE/ADAPT/REFERENCE_ONLY/REJECT over the
  loader offsets + load_data_for, GGUF split/shard + shard_gguf.py, downloader
  resume/verify/fsync/atomic, ggml-rpc hash/cache, per-tensor HTP/OpenCL sharing,
  OpenCL xmem prepared cache, and partial-load/arbitrary-model limits. Produced by a
  7-agent read-only audit fan-out; twelve load-bearing citations re-verified by hand.

Checkpoint 2 -- Residency contract + schemas:
- 15 self-contained JSON Schemas (`schemas/`) for ModelManifest, WeightSegment +
  transfer chunks, atomic WeightSet, backend PreparedImage, IslandExecutable,
  TransferTicket, ReadyCertificate, ResidencyLease, StateLease, DeviceInventory +
  physical-byte accounting, DispatchDecision, TransportFrame, SimConfig,
  SimRunManifest (+ informative `_defs`). SHA-256 identities; the derived-image
  identity binds all nine required fields.
- `WEIGHT_RESIDENCY_CONTRACT.md`: two linked state machines, content-identity
  preimages, physical-byte ledger, and the dispatch eligibility rule.
- `PREFETCH_POLICY.md`, `SIMULATOR_SPEC.md`.

Checkpoint 3 -- Transport contract:
- `TRANSPORT_CONTRACT.md` + `schemas/transport_frame.schema.json`: two bounded
  versioned channels, frame limits + checksums, chunk hashes + resumable verified
  ranges, staging + file/dir fsync + atomic publish, credits + bounded buffers,
  activation/result priority over background weights, deadlines/cancellation/
  reconnect/idempotency/mutation-seq, boot/residency/route/state epochs,
  drain-before-eviction + backend fence, fail-closed for stale/duplicate. Does NOT
  reuse LayerSplit or ggml-rpc framing unchanged.

Checkpoint 4 -- Deterministic offline simulator:
- `sim/residency_sim.py`: integer-microsecond discrete-event sim with a stable total
  event order; models the server GPU queue + HBM, a V0 shared USB controller +
  symmetric per-phone links, UFS/verify/materialize/prepare, bounded HTP + GPU
  lanes, canonical + backend-derived RAM, activation buffers + state leases +
  thermal + deterministic
  failures, interference only when measured. Eight baselines, the causal score, the
  never-wait candidate, a goodput sweep, byte-identical replay, and a FrameSequencer.
- `sim/make_scenarios.py`, `sim/scenarios/baseline_sweep.config.json`,
  `sim/test_residency_sim.py` (28 checks covering all ten required behaviors).

Tests + tooling:
- `run_schema_tests.py` (both validators), `validate_manifests.py` (semantic),
  `make_fixtures.py`, `fixtures/` (53 schema + 23 semantic).

Reporting: this file, `RESULTS.md`, `CURRENT_SLOW_LINK.md`; talks.md + the design
doc updated.

## Approach (why this shape)

The one error this spike guards against is treating a STRUCTURAL check (name/count/
byte-range) as a CONTENT-IDENTITY check. The audit shows the current tree has NO
SHA-256 of any weight payload; the only content hash anywhere is FNV-1a 64-bit in
ggml-rpc, served on a cache hit WITHOUT re-comparing bytes. So the residency
VERIFYING state, the derived-image identity, the durable atomic publish, and the
generation-qualified teardown are all NET-NEW, wrapped around reusable byte
plumbing. The schemas make the fail-closed rules machine-checkable; the simulator
proves the contract MECHANICS (state machine, ledger, eligibility, preemption,
recovery, determinism) are internally consistent -- it is NOT a capacity number,
because the V0 device pipeline rates are symbolic. The new ADB measurement narrows
one end-to-end provisioning leg, but cannot be inserted into V0 as raw USB goodput
without double-counting its separately modeled UFS stage.

## Measured-link update and S9-V1 plan

The architecture is unchanged: predict residency, transfer weights in the
background, and dispatch only to a READY certificate. The faster measured path
changes the prefetch horizon, not this invariant. A 900 MiB set still needs about
4.2 s on OP12 or 3.4 s on OP15 before verify/prepare/warmup.

Freeze the existing S9-V0 simulator and its 40/100/250/400/550 MiB/s sensitivity
table as mechanics evidence. Do not relabel it as measured. Implement S9-V1 as the
smallest follow-on in this order:

1. Version the link/config contract. Add per-device H2D and D2H profiles, an
   evidence ID, a path kind (`staged_file` or `native_buffer`), a contention-domain
   ID, and per-domain directional aggregate caps. Do not silently change schema v1.
2. Replace the static global-controller division with dynamic contention among only
   active streams in the same domain. OP12 Bus 006 and OP15 Bus 008 are separate
   domains in the measured inventory.
3. Add the missing D2H result-transfer event. A phone request is not complete and
   its resources are not released until `output_bytes` returns or the route fails.
4. Prevent double accounting. A `staged_file` H2D profile lumps transport and file
   ingestion/page-cache effects; do not blindly add a full UFS write. Measure and
   charge durable fsync/publish separately. A `native_buffer` profile may model
   transport and UFS/materialization as separate measured stages.
5. Add a measured two-phone staged scenario beside the existing broad sensitivity
   sweep. Bind it to the payload hash, byte count, elapsed samples, paths, and date
   in `CURRENT_SLOW_LINK.md`. Keep old V0 results for regression only.
6. Measure native memory-to-memory H2D/D2H, UFS read/write, durable fsync/publish,
   hash, materialization, preparation, warmup, and transfer-vs-HTP/GPU interference
   separately. Only then run the decomposed capacity scenario.
7. Add optional 800/1000 MiB/s USB 10 Gbps sensitivity points as an unmeasured
   future-device class. They are not OP12/OP15 evidence.

S9-V1 exit gates:

- schema v1 fixtures and replay remain byte-identical;
- v2 schemas reject missing direction, evidence, or contention-domain data;
- one active stream receives its full per-link rate; unrelated domains do not share;
- same-domain streams obey the directional cap deterministically;
- staged profiles account for file ingestion and durable storage exactly once;
- output transfer affects completion and D2H contention;
- measured-profile manifests bind raw evidence and reproduce deterministically; and
- capacity remains `UNPROVEN` until the remaining physical stages and S8 Gate A are
  measured/passed.

## Explicit stop conditions (honored)

Stopped after contracts, schemas, simulator, tests, and the optional link evidence.
Did not build a phone daemon, downloader service, server library, kernel, or
production scheduler; claimed no energy savings, novelty priority, or PIM-hardware
equivalence; did not commit or push.
