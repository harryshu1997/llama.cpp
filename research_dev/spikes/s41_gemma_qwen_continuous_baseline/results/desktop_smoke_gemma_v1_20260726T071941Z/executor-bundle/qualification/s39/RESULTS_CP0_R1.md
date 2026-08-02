# S39 CP0-R1 two-route eligibility

Verdict:

`CONTRACT_VALID; TWO_ROUTE_ELIGIBILITY_NOT_RUN; CYCLE_BLOCKED`

No route was promoted, no forward/reverse cycle was run, and no trace,
controller, performance, capacity, or energy claim is made.

## Frozen gate

`CP0_R1_TWO_ROUTE_ELIGIBILITY_CONTRACT.json` is model-independent and
fail-closed. It requires, for each of two models:

- independent B8 service on the bound RTX 4060 Ti with 512 MiB serving
  headroom;
- complete OP15 -> OP12 execution, direct activation transfer, realized
  placement, 512 MiB phone memory headroom, zero process swap, and zero system
  swap growth;
- exact history, position, ownership, and cleanup mechanics;
- exact agreement with an independent, path-matched monolithic CUDA oracle;
- prospective 64-item MMLU task-quality noninferiority;
- useful B8 phone publication before the same model becomes CUDA-ready.

At pair level it requires a measured non-shareable model-plus-KV lower bound
that exceeds target VRAM with serving headroom, plus local-UFS reprepare in
both directions within 30 seconds. Cross-backend and cross-geometry greedy
token agreement are recorded but do not decide eligibility.

The evaluator issues `TWO_ROUTE_ELIGIBILITY_PASS` only when every required
record is present and consistent. That status authorizes one reduced
A -> B -> A cycle only. It does not authorize trace, controller, or energy
work.

## Candidate binding

The incumbent is Qwen3 14B Q4_K_M, SHA-256
`500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0`.
`CURRENT_ROUTE_READINESS.json` labels it `PROVISIONAL_BATCH`; its B1/B8/B32
single-prompt evidence is not complete route qualification.

The only new candidate is Qwen3 8B Q8_0:

- upstream model: `Qwen/Qwen3-8B`, 8.2B parameters and 36 layers;
- GGUF repository: `Qwen/Qwen3-8B-GGUF`;
- revision: `6cfbfc7d8ab95bf485c79fcc40be60930d5b4c8c`;
- file: `Qwen3-8B-Q8_0.gguf`, 8,709,518,112 bytes;
- SHA-256:
  `408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6`.

This is one same-architecture capacity candidate, not an architecture-diversity
result. If it fails, candidate search stops.

## Verification

- contract SHA-256:
  `ffb2abeb33e818477e8a7181e177a6d296ed3b021767bc83a8df0d2450e5d095`;
- candidate SHA-256:
  `ee3196ca660fa7eb7ea293260dc98dd6fdbf14571a4d4c5aece5343fb29b28d8`;
- focused CP0-R1 tests: 25/25 pass;
- pre-existing S39 tests before the CP0-R1 edit: 229/229 pass;
- complete S39 suite after the edit: 254/254 pass.

The focused suite includes a complete synthetic record only to prove evaluator
mechanics. It mutates B8 capacity, GPU and phone headroom, direct transfer,
placement, swap, state exactness, oracle independence, task quality, bridge
ordering, pair capacity, and reprepare. It is not physical evidence.

## Physical blockers

At freeze time:

- neither phone is visible on ADB port 5037 or 5038;
- the RTX 4060 Ti host is reachable over SSH, but `nvidia-smi` fails with a
  driver/library version mismatch;
- Qwen3 8B Q8_0 is not present locally or on the target desktop;
- its phone route cut and shard digests therefore are not frozen.

The next physical action is not a model-switch cycle. Restore target-GPU NVML
and both phones, materialize and hash the single candidate, freeze its route,
then finish Qwen3 14B qualification before screening Qwen3 8B.

## Historical boundary

Gemma 4 Q4/Q8 and Qwen2.5 Q4/Q8 results are unchanged negative evidence. Do
not rerun them or reinterpret their greedy-token failures under the new
task-quality gate. No further Qwen2.5 phone, trace, controller, or energy
acquisition is justified by CP0-R1.
