# Related Work and Claim Boundary

This note prevents the S4 experiment from claiming novelty already established
by prior systems. It is a positioning contract, not a completed systematic
literature review.

## Cautious Research Statement

Use this wording until the experiment and a broader literature review finish:

> We study whether NanoFlow-style inter-operator pipelining can be realized
> across Hexagon HTP and Adreno GPU for continuous multi-sequence decode,
> without sacrificing HMX batching efficiency or increasing gross
> server-plus-phone energy, when phones act as layer stages in a distributed
> LLM serving pipeline.

Do not use "first" or claim that batching, concurrent requests, operator
affinity, operation pipelining, mobile heterogeneous inference, distributed
phone execution, or J/token is individually novel.

## Closest Systems

| Work | What it already demonstrates | Remaining S4 question |
|---|---|---|
| HeteroInfer / HeteroLLM | Real Adreno-Hexagon execution, operator-affinity placement, GPU-NPU output-row splitting, mobile decode, and device energy | Its experiments use batch size one; it does not evaluate a reentrant attention/projection pipeline across independent KV-owning decode groups. |
| llm.npu | Linears on NPU, floating-point work on CPU/GPU, and input-ready out-of-order subgraph scheduling | Its scheduling target is prefill; decode remains on CPU in the evaluated system and real GPU-NPU decode coordination is not implemented. |
| PowerInfer-2 | Mobile batch-adaptive decode, parallel candidate sequences, CPU-NPU FFN and attention partitioning, and phone J/token | It uses sparse/quantized models and CPU-NPU within-operator work division, not dense FP16 HTP-Adreno inter-operator pipelining or gross server-plus-phone energy. |
| NanoFlow | Multi-request decode, operation nano-batches, attention/projection overlap, interference-aware schedule search, and end-to-end serving throughput | It targets NVIDIA datacenter GPUs and does not evaluate mobile GPU-NPU execution or energy. Its nano-batch pipeline is direct scheduling prior art for S4. |
| PowerBench | Modern Snapdragon CPU/GPU/NPU throughput, DVFS, host-control effects, and accelerator/SoC energy | It is a measurement study, not a multi-stream operator scheduler. It makes CPU polling, sleep state, and whole-SoC power mandatory S4 controls. |
| LinguaLinked and EdgeShard | Distributed layer-partitioned inference across mobile or edge devices | The outer server-phone pipeline is not by itself a new contribution. |

## Primary Sources

- HeteroInfer: https://arxiv.org/abs/2501.14794
- HeteroInfer DOI: https://doi.org/10.1145/3731569.3764808
- llm.npu: https://arxiv.org/abs/2407.05858
- llm.npu artifact: https://zenodo.org/records/14392760
- llm.npu implementation: https://github.com/UbiquitousLearning/mllm
- PowerInfer-2: https://arxiv.org/abs/2406.06282
- PowerInfer repository and announcement: https://github.com/SJTU-IPADS/PowerInfer
- NanoFlow: https://arxiv.org/abs/2408.12757
- NanoFlow implementation: https://github.com/efeslab/Nanoflow
- PowerBench: https://arxiv.org/abs/2607.05475
- Snapdragon operator/pipeline study: https://arxiv.org/abs/2605.27435
- LinguaLinked: https://arxiv.org/abs/2312.00388
- EdgeShard: https://arxiv.org/abs/2405.14371

The public PowerInfer repository announces PowerInfer-2 but is not treated here
as a separately verified reproduction artifact. No HeteroInfer artifact is
assumed.

## Required Comparisons

S4 must isolate its mechanism with equal-work controls:

```text
HTP-only complete B-way decode
GPU-only complete B-way decode
HTP-only decode split into the same microstream sizes
existing HTP-decode plus GPU-prefill on different requests
whole-request sequence-affine HTP/GPU lanes
serial HTP-A -> GPU-B -> HTP-C
pipelined HTP-A and HTP-C with GPU-B across independent groups
```

It must report both phone-only and gross fleet energy boundaries. A phone-only
J/token result cannot support a distributed energy claim.

## Evidence Needed for a Defensible Result

1. Real multi-sequence decode with private KV and dynamic occupancy.
2. A precise request-group DAG and no duplicate KV mutation.
3. Solo and co-run operator profiles, including interference.
4. HMX/HVX path logging and the batching loss caused by microstreaming.
5. Correct GPU fused attention on the tested phone; no fallback substitution.
6. Measured activation handoff, cache visibility, and synchronization.
7. Equal-work throughput, TTFT, TPOT, and queue behavior.
8. Whole-phone and gross fleet J/completed-token with physical instrumentation.
9. Sustained thermal results and paired confidence intervals.

If successful, the result is evidence for one measured combination, not a claim
that its component techniques are new. A negative result is also useful: it
would show whether batching loss, shared-memory contention, handoff, or energy
prevents a datacenter nano-batch schedule from transferring to mobile GPU-NPU
hardware.
