# S24 Fixed-Diamond Overlap and Handoff Plan

Status: frozen before physical execution.

## Scope

Prove or reject the fixed R0/R1/R2 diamond in DESIGN.md on the RTX 4060 Ti
desktop, OP12, and OP15. This is a private research prototype. Do not generalize
the scheduler or transport.

## CP0 - Freeze the claim

- [x] State the five-part finite-route and convergence claim.
- [x] Separate the mechanism from ordinary layer splitting and continuous
  batching.
- [x] Freeze the desktop/A6000/phone ownership boundary.
- [x] Freeze R0, R1, and R2 and the excluded follow-on work.
- [x] Freeze the high-priority regression limit at 5 percent for p95 TTFT and
  p95 latency, with no additional priority-0 SLO miss.

Exit: DESIGN.md is frozen before CP1 measurements.

## CP1 - Capacity and model artifacts

- [x] Hash the desktop model, V3 binary, CUDA runtime, and relevant source.
- [x] Start cuda-prefix [0,8), cuda-mid [8,16), and cuda-tail [16,48)
  simultaneously.
- [x] Record process RSS/PSS and per-process GPU memory after all workers are
  resident.
- [x] Record total RTX 4060 Ti memory and idle baseline.
- [x] Fail closed if a worker loads full-model GPU weights, the disjoint set
  exceeds device memory, or placement is not CUDA.
- [x] On the A6000 host, verify or create OP12 F16 [0,8) with
  research_dev/shard_gguf.py.
- [x] On the A6000 host, create OP15 F16 [8,16) with the same tool.
- [x] Hash shards before and after USB transfer.
- [x] Confirm no large weight transfer used WiFi.

Exit: all five resident stages have capacity and artifact evidence.

## CP2 - Generic ordered route runtime

- [x] Add typed resident-stage and finite-route descriptors under S24.
- [x] Add an ordered route runner; do not extend S22 run_request().
- [x] Preserve request_id, route_epoch, per-worker seq_id, token, and position.
- [x] Lease one sequence on every participating worker.
- [x] Serialize batch and remove RPCs on every socket.
- [x] Remove worker KV before releasing each software lease.
- [x] Pin a route across prefill and decode.
- [x] Add failure-injection and cleanup tests.

Exit: unit tests prove ordered propagation and exact lifecycle conservation.

## CP3 - Continuous batch convergence

- [x] Use one OP15 batcher for R1 and R2 in the treatment.
- [x] Use one CUDA-tail batcher for R0, R1, and R2.
- [x] Measure the batch knee per resident worker as the smallest candidate
  reaching at least 95 percent of peak median rows per second.
- [x] Dispatch on the knee, earliest latest-safe start, or gather timeout.
- [x] Record route and upstream contribution for every physical batch.
- [x] Add tests for mixed-source batches, all three dispatch reasons, and the
  absence of cohort barriers.

Exit: synthetic tests demonstrate cross-source and shared-tail convergence.

## CP4 - Physical correctness

- [x] Run R0 B1 twice.
- [x] Run R1 B1 twice.
- [x] Run R2 B1 twice.
- [x] Run R2 B4.
- [x] Run concurrent R1+R2 with a shared OP15 batcher.
- [x] Run concurrent R0+R1+R2 with one CUDA-tail batcher.
- [x] For the deterministic convergence cases only, align measured B1 boundary
  arrivals, keep each source below the OP15 knee, and use the frozen 250 ms
  gather bound without a route/cohort wait condition.
- [x] Capture placement certificates and worker logs.
- [x] Verify finite outputs, lineage, positions, no missing buffers, zero KV,
  and zero software leases.
- [x] Compare output tokens and boundary activations against CUDA controls.
- [x] Report phone-F16/server-Q8 numerical differences without upgrading an
  uncertified result.

Exit: physical mechanics pass or a precise blocker is recorded.

## CP5 - Equal-work benefit controls

- [x] Run C0 all-R0.
- [x] Run C1 with route-isolated OP15 queues.
- [x] Run C2 with one shared OP15 queue.
- [x] Run C3 with SLO route selection.
- [x] Use the same first-two-per-route identities, aligned arrivals, and 250 ms
  OP15 gather bound in every control.
- [x] Record completions, misses, TTFT, latency, queue time, route counts,
  OP15/tail batches, makespan, and CUDA island compute time.
- [x] Measure selected RTX 4060 Ti GPU_BOARD energy with the existing NVML
  method.
- [x] Evaluate the shared-batch, high-priority, and CUDA-work gates exactly as
  frozen in DESIGN.md.
- [x] Keep phone, network, A6000, and total-system energy UNKNOWN.

Exit: benefit passes or BENEFIT_GATE_FAIL is justified by raw evidence.

## CP6 - Workloads

Not run. CP5 returned `BENEFIT_GATE_FAIL`, so expanding the same fixed policy
to broader traces would not rescue the frozen claim.

- [ ] Run the deterministic three-class trace first.
- [ ] Run the pinned 60-request dense trace with 21/17/22 arrivals at 0/1/2
  seconds.
- [ ] Preserve real arrivals and observed token counts.
- [ ] Label synthetic token values, priorities, and SLOs.
- [ ] Keep the one-token/four-step run mechanics-only.
- [ ] Build and run the separate 28-request context-compatible cohort.
- [ ] Run the 60-request mechanics trace with C3 and the 28-request
  observed-length cohort with pinned C2 route hints and 64-token prefill
  chunks; apply no arrival alignment to either trace.

Exit: both proxy mechanics and observed-length behavior are explicitly scoped.

## CP7 - Freeze evidence and verdict

- [x] Save exact commands, logs, placement certificates, raw JSON, artifact
  hashes, and SHA256SUMS.txt.
- [x] Record S23 and S22 synchronization source/destination hashes.
- [x] Update this plan, RESULTS.md, research_dev/talks.md, and
  research_dev/NEXT_PLAN.md only after physical evidence exists.
- [x] Select exactly one allowed final verdict.
- [x] Stop before layer-10 overlap, direct phone transfer, or protocol redesign.

## Allowed final verdicts

- FIXED_DIAMOND_POC_PASS
- MECHANICS_PASS_NUMERICALLY_UNCERTIFIED
- BENEFIT_GATE_FAIL
- PHYSICAL_EXECUTION_BLOCKED
