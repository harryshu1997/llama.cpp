# S8 Related-Work Audit Contract

Status: literature audit required before novelty language.

The possible contribution is the intersection of:

- reverse server-to-phone offload;
- mixed AI service queues;
- resident operator islands and state leases;
- phone NPU/GPU backend selection and interference;
- cross-workload slack pooling; and
- measured server power-state shaping.

None of these terms alone is a novelty claim.

## Required comparison groups

Before publication framing, audit primary papers and available code for:

1. heterogeneous mobile inference and multi-DNN schedulers;
2. edge/fog/cloud task offload and reverse offload;
3. LLM prefill/decode disaggregation and multi-model serving;
4. model placement, caching, and cold-start-aware scheduling;
5. HEFT, earliest-finish, DRF, CONWIP, and SLO-aware scheduling;
6. cross-accelerator NPU/GPU co-execution and shared-memory interference;
7. operator partitioning, grouped GEMM, and persistent command streams;
8. server power-state-aware scheduling and energy proportionality; and
9. trace-driven RAG, multimodal, and agent serving.

## Existing project comparison boundary

- HeteroLLM/llm.npu-style partitioning informs phone kernels but does not by
  itself establish mixed-workload scheduling novelty.
- NanoFlow and PowerInfer-2 constrain any claim around pipelined or batched LLM
  decode.
- The historical Unifer VQ/HEFT implementation is prior project substrate and
  must be disclosed as such, not presented as a new algorithm.
- S7 ragged attention is a local kernel mechanism; variable-length attention
  already exists and is not the system contribution alone.

## Evidence required for differentiation

The final related-work table must compare:

```text
direction of offload
workload diversity and service DAGs
state/weight residency and cache churn
device x backend decision space
cross-workload queue externalities
total-system energy boundary
server idle/power-state objective
real trace and failure handling
```

Do not use "first" or "novel" until this audit is complete and reviewed.
