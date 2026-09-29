# Whole-task phone route V1

This experiment adds a resident whole-model OP15 Adreno route to the S42 task
scheduler and compares it with resident desktop CPU and RTX 4060 Ti CUDA
routes. All three servers use the same byte-identical
`Llama-3.2-1B-Instruct-Q4_0.gguf` model. Model loading and warmup are outside
the paid interval.

The V2 scheduler cost separates direct request energy from shared background
power. Direct route energy is measured above the resident idle fleet baseline;
causal tail and critical-path extension are accounted separately:

```text
E_cuda_epoch_open = E_cuda_incremental + E_cuda_post_response_dynamic_tail
E_cuda_epoch_reused = E_cuda_incremental
E_phone = E_phone_incremental
```

The tail is charged once when opening a CUDA activity epoch. Later requests
that join an epoch whose tail is already accounted use the tail-reused
profile. If the runtime cannot prove reuse, it uses the isolated profile.
This avoids charging every queued CUDA request for the same tail and avoids
pretending that a P8 GPU request has response-window energy only.

## Physical result

The checked-in result used an RTX 4060 Ti 16 GiB and an OP15 with Adreno 840.
It measured three BurstGPT-derived long request shapes, three repetitions per
route, with every paid request starting from verified CUDA P8 idle.

| Resident route | Mean latency | Mean accounted fleet energy | Mean energy above resident idle |
| --- | ---: | ---: | ---: |
| Desktop CPU | 10.033 s | 1280.461 J | 1132.674 J |
| Desktop CUDA response | 1.364 s | 102.155 J | 82.064 J |
| OP15 Adreno whole task | 6.855 s | 116.775 J | 15.803 J |

The RTX 4060 Ti took another 20.281 s on average to return to P8 and consumed
399.168 J of dynamic GPU energy above P8 during that interval. Therefore:

- OP15 reduced response-window fleet energy by 90.88 percent and latency by
  31.68 percent relative to desktop CPU.
- OP15 was 5.03x slower than CUDA and used 14.31 percent more fleet energy in
  the response window.
- When CUDA opened a new activity epoch, its response plus causal dynamic
  tail cost was 501.323 J. OP15 used 76.71 percent less energy on this
  lifecycle boundary.

For the representative 915-input, 292-output-token request, the isolated
profile selects `phone-adreno` when CUDA is idle, queued, or unavailable. The
tail-reused profile selects `desktop-cuda` when CUDA is available and the
30-second SLO permits the queue wait; it selects `phone-adreno` when CUDA is
unavailable.

## Files

- `run_resident_route_campaign.py`: starts the three resident servers, warms
  them, normalizes each paid request to CUDA P8, and records synchronized CPU,
  GPU, and whole-phone energy.
- `measure_cuda_tail.py`: measures response completion through the return to
  resident P8.
- `analyze_resident_route_campaign.py`: verifies non-overlapping request
  windows, subtracts measured resident idle power, fits direct affine latency
  and energy costs, emits isolated, tail-reused, and composite-USB profiles,
  and exercises idle, queued, and unavailable CUDA states.
- `results/4060ti_op15_20260807/`: compact physical evidence and generated
  profiles.

The original V1 profiles are retained as historical response-window evidence.
`ANALYSIS_INCREMENTAL_V2.json` and the `INCREMENTAL_*_V2` profiles are the
current route-selection inputs.

## Runtime policy boundary

The profile selector runs at the atomic precommit snapshot:

1. Use `cuda_epoch_reused` only when an active or queued CUDA epoch has
   already been charged one tail.
2. Otherwise use `cuda_epoch_open`.
3. Let the normal S42 deadline, readiness, queue, quality, and energy gates
   select CPU, CUDA, or OP15 from that profile.

This V1 validates resident single-request routes, not an executor-integrated
continuous-batch dispatcher. The phone route uses token-array HTTP over an ADB
USB forward; the payload is small because the model remains resident. It does
not exercise activation DMA, AOA, phone HTP, model switching, KV migration, or
mixed-model continuous batching. Cross-backend output hashes differ, so the
generated routes are classified as `bounded_numeric`, despite the identical
model file and completed requested token counts.

## Validation

From the repository root:

```sh
python3 -m unittest \
  research_dev.spikes.s42_general_energy_scheduler_v1.tests.test_whole_task_phone -v
python3 research_dev/scheduler/tests/run_all.py
```
