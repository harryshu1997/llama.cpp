# Related-Work Boundary

This direction has close prior work. The contribution cannot be described as
the first distributed or edge RAG system.

## Closest systems

- [DRAG](https://arxiv.org/abs/2505.00443) decentralizes private knowledge and
  uses topic-aware peer discovery. It focuses on locating distributed knowledge,
  not heterogeneous NPU/GPU execution of the downstream large model.
- [EACO-RAG](https://arxiv.org/abs/2410.20299) chooses among local,
  edge-assisted, and cloud RAG strategies and updates edge knowledge. It is the
  closest route-selection precedent.
- [EdgeRAG](https://arxiv.org/abs/2412.21023) reduces on-device index memory by
  generating and caching embeddings on demand. It is relevant to CP2.
- [PerLLM](https://arxiv.org/abs/2405.14636) performs SLO- and energy-aware
  edge/cloud LLM service scheduling, but does not schedule a distributed RAG DAG
  whose phones also execute layers of the same reasoning model.
- [HyGen](https://openreview.net/forum?id=cQxLCVa9u7) co-batches heterogeneous
  online/offline prefill and decode under SLOs on a server. It is relevant to
  local continuous-batch policy, not distributed data ownership.

## Defensible system contribution

The potentially new unit is the joint plan across two forms of locality:

1. `data locality`: each phone owns a different searchable library and returns
   only ranked evidence;
2. `model-state locality`: each phone keeps embedding/reranking weights plus
   selected reasoning-model layers and sticky KV state;
3. `phase locality`: stateless RAG stages can be batched or moved independently,
   while a request's decode route remains state-affine;
4. `SLO coupling`: the scheduler chooses retrieval fanout, evidence depth,
   reasoning layer cut, and batch delay from one end-to-end budget.

This remains a hypothesis until C3 beats C0-C2 on a measured quality/SLO/server-
energy frontier. A combination of distributed retrieval and ordinary layer
splitting without that joint decision would not be a sufficient contribution.

