# S33 Full Quantized Routes

## Objective

Determine whether a full Gemma-4 12B Q8_0 or Q4_0 GGUF can support useful
approximate routes across OP12, OP15, and one A6000. The full GGUF may be
stored on every device while each phone prepares only a resident layer window.
"Full model" does not imply that all 48 layers fit in one phone's HTP memory.

S33 does not alter S32's exact-route verdict. Quantized phone execution enters
the scheduler only through the separate approximate-quality contract below.

## Frozen candidate order

1. Q8_0 is primary because it has lower quantization risk and already exists on
   the host and both phones.
2. Q4_0 is a capacity fallback. Test it only when it creates a larger useful
   resident window than Q8_0.
3. First test full-graph executable residency on OP15. Test OP12 only if OP15
   completes without process swap, DSP failure, or CPU fallback for a supported
   compute operation.
4. If the full graph does not fit, retain the full file on storage and profile
   partial resident windows. Do not describe this as full-model RAM residency.

## Physical viability gate

A full graph or partial window is viable only when all conditions hold:

1. Host and phone GGUF SHA-256 values are identical.
2. The worker uses the current-source binary and expected HTP architecture.
3. Model preparation and the measured execution both complete with exit zero.
4. Process `VmSwap` is zero before and after the paid execution.
5. Outputs are finite, request and sequence state drains to zero, and the
   placement certificate reports no missing buffer.
6. B1 fallback is not used to certify a B32 route. Every scheduled shape needs
   its own placement evidence.

## Approximate quality gate

Freeze the prompt corpus and tokenization before running the phone route. Use
the same quantized GGUF, greedy decoding, prompt rows, context, and CUDA tail
for treatment and reference.

For at least 128 nonempty natural-language prompts:

1. every request produces eight finite token decisions on both routes;
2. first-token agreement is at least 0.95;
3. aggregate token-decision agreement is at least 0.95;
4. exact eight-token sequence agreement is at least 0.80;
5. no service-class accuracy metric drops by more than 0.02 absolute on any
   labeled subset included in the corpus; and
6. all mismatches and denominators are retained, not filtered after execution.

These are prototype route gates, not a general language-quality claim. If they
fail, the quantized route remains measurement-only. The thresholds must not be
changed after looking at the phone outputs.

## Scheduling gate

Only quality-eligible resident windows may enter the route catalog. The runtime
still chooses among a finite set of profiled boundaries using batch, memory,
queue, priority, SLO, and downstream-capacity constraints. It must preserve a
CUDA-only fallback and never wait for a phone route that cannot finish before
the request's latest-safe boundary time.

## Checkpoints

- [x] CP1: guarded full Q4_0 and Q8_0 executable-residency screen. Both fail.
- [x] CP2: freeze and tokenize 128 natural prompts for both model files.
- [x] CP3: same-GGUF CUDA and phone-route quality comparison. Both fail.
- [x] CP4: profile the viable overlapping resident windows at B32.
- [x] CP5: apply the eligibility gate. Zero rows are installed.
- [ ] CP6: stopped because CP3 produced no eligible quantized route.

No phone-energy or total-system-energy claim is authorized by this plan.
