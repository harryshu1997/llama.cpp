# Resident BGE batch-32 phone profile V1

This profile makes the measured RTX 4060 Ti plus OP15 BGE-small result
selectable by `UnifiedScheduler`. It covers resident batches of 32 using the
measured 34-token query shape. `batch32_groups` is the number of consecutive
batch-32 units assigned in one scheduling decision.

## Policy

The route manager selects a profile at the atomic precommit snapshot:

1. Select `cuda_epoch_reused` only when an active or queued CUDA epoch already
   owns a tail-charge receipt.
2. Select `cuda_epoch_open` when no receipt exists or the state is unknown.
3. If an epoch-open decision selects CUDA, create one tail-charge receipt
   before admitting another request to that epoch.
4. Retire the receipt after CUDA returns to P8 and the epoch closes.

The epoch-open profile charges 313,014,997 uJ once to CUDA. The reused profile
charges only response energy. With the standard 5 percent saving margin and
measured error bounds, enforce mode selects OP15 through 106 batch groups, or
3,392 embeddings. The raw mean-energy crossover is between 119 and 120 groups,
but it is not used as the enforcement threshold.

Phone admission also requires a current runtime snapshot proving:

- the exact BGE Q8_0 model residency;
- the embedding RPC session residency;
- heartbeat age at most 1.1 s;
- nominal thermal and qualified contention buckets;
- phone temperature at most 45 C;
- no failures, reset mismatch, or open circuit; and
- an embedding request whose semantics do not require KV, logits, grammar, or
  a sampler.

Unknown epoch state, missing runtime evidence, residency loss, excessive
temperature, or an infeasible deadline fails closed to desktop CUDA.

## Usage

```python
from research_dev.scheduler import UnifiedScheduler
from small_model_phone_v1.profile_selector import (
    CudaEpochReceipt,
    CUDA_EPOCH_REUSED,
    load_selected_profile,
    make_request,
)

# Use None until the route manager can prove an existing tail charge.
receipt = None
selected = load_selected_profile(receipt)
scheduler = UnifiedScheduler(
    (selected.profile,),
    "enforce",
    runtime_snapshot=runtime_snapshot,
)
request = make_request(
    "embedding-cohort-1",
    batch32_groups=1,
    arrival_us=now_us,
    deadline_us=now_us + 500_000,
)
decision = scheduler.schedule(request, runtime_now_us=now_us)

# A later request may use the reused profile only after the route manager has
# persisted a charge receipt for the still-open CUDA epoch.
receipt = CudaEpochReceipt(CUDA_EPOCH_REUSED, "tail-charge-epoch-7")
```

The route manager, not `UnifiedScheduler`, owns the receipt transaction. A
scheduler decision does not itself prove that CUDA stayed active or that its
tail was charged.

## Files

- `profile_selector.py`: hash-validating, fail-closed lifecycle selection and
  exact request construction.
- `results/4060ti_op15_20260808/BGE_BATCH32_RESULT.json`: compact measured
  result and scheduler coefficients.
- `results/4060ti_op15_20260808/RAW_EVIDENCE_SHA256SUMS`: hash manifest for the
  synchronized raw captures under `models/s42-small-models-v1/results-v4`.
- `results/4060ti_op15_20260808/SCHEDULER_PROFILE_CUDA_EPOCH_OPEN.json`: CUDA
  response plus one dynamic tail charge.
- `results/4060ti_op15_20260808/SCHEDULER_PROFILE_CUDA_EPOCH_REUSED.json`:
  response-only CUDA energy for an already-accounted epoch.

## Scope

This is a measured cohort profile, not a generic embedding model. Do not map
arbitrary batch sizes, token lengths, models, or phone backends to its workload
ID. The OP15 route is Adreno OpenCL. EmbeddingGemma HTP and CNN routes need
their own resident energy campaigns before receiving measured profiles.

Run the focused tests from the repository root:

```sh
python3 \
  research_dev/spikes/s42_general_energy_scheduler_v1/tests/test_small_model_phone.py \
  -v
```
