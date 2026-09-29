# Gemma F16 shape-balance qualification

Status: `RETAIN_FIXED_POLICY`.

This experiment tests whether changing the OP15 FFN suffix width by live
continuous-batch shape improves the existing two-model F16 BurstGPT policy.
It is a physical A-B-B-A comparison on the RTX 4060 Ti 16 GiB desktop plus
OP15 phone. Each arm executes the same 17 Gemma requests from the preserved
74-request trace: 11,476 input tokens and 6,919 output tokens.

The screen keeps the model, GPU/CPU layer placement, resident phone weights,
arrival trace, output lengths, and fleet-energy boundary fixed. Only the
phone/host division of each qualified Gemma FFN operator changes.

## Policies

| Variant | Split table |
|---|---|
| Fixed qualified fallback | `1:6144,16:6144,512:0` |
| Shape-balanced shadow | `4:6144,6:5632,8:5120,10:6144,11:5120,13:6144,14:5120,15:6144,16:5120,512:0` |

The shadow table is materialized by the public unified scheduler API from the
two prior fixed-policy per-shape receipts. Unmeasured physical shapes retain
the qualified 6,144-column width. The phone still holds the complete resident
6,144-column weight slice, so this test adds no model load or phone-memory
allocation.

## Physical A-B-B-A result

| Metric | Fixed mean | Shape-balanced mean | Saving |
|---|---:|---:|---:|
| Makespan | 694.686 s | 680.995 s | +1.971% |
| CPU package energy | 51.862 kJ | 52.946 kJ | -2.091% |
| GPU board energy | 24.899 kJ | 24.478 kJ | +1.693% |
| Whole-phone energy | 1.768 kJ | 1.726 kJ | +2.386% |
| Fleet energy | 78.529 kJ | 79.150 kJ | -0.790% |

Positive saving means the shadow candidate used less time or energy. Its two
paired fleet-energy savings were +1.501% and -3.074%; its paired makespan
savings were +4.538% and -0.609%. The candidate therefore fails the
predeclared requirements that both pairs save fleet energy, neither pair
regress makespan, and mean fleet saving reach 0.5%.

## Why it failed

The operator-level balance change behaved as intended. Call-weighted mean
join wait fell from about 0.860 ms under the fixed split to about 0.035 ms
under the shadow split. Call-weighted island service fell from about 8.466 ms
to about 8.248 ms.

The narrower phone widths moved more columns to the host branch. Mean CPU
package power increased from about 74.66 W to 77.75 W. The resulting
1.084 kJ increase in mean CPU energy was larger than the combined GPU and
phone reduction, so the fleet used 0.621 kJ more despite finishing sooner on
average. Continuous-batch geometry also changed enough between repeats to
reverse both paired outcomes.

Latency balance alone is therefore not an energy objective. A successor may
consider an intermediate width only after measuring its CPU, phone, and total
island energy under matched batch shapes. The 74-request promotion run is not
authorized by this result.

## Validity and evidence

All four runs passed these gates:

- 17 requests, 11,476 input tokens, and 6,919 output tokens completed;
- identical F16 Gemma model hash and source-trace hash;
- identical GPU/CPU layer placement;
- unified scheduler execution plans bound to each result;
- synchronized CPU-package, GPU-board, and whole-phone paid intervals;
- nonzero conserved phone work with zero reset recoveries; and
- zero sampled process swap.

The aggregate record is
[`results/physical_abba_v1/SHAPE_BALANCE_ABBA.json`](results/physical_abba_v1/SHAPE_BALANCE_ABBA.json).
Its internal canonical record SHA-256 is
`008175d5930fe4afb16e9c95ff1f05ba4b18f22a3a20a25521e64b2329ee1384`;
the stored file SHA-256 is
`fe6026281216a3cfbf267b72617bb1a8d347d17a56bf8299cfdf2ee92c58d8ac`.
Selected plans, results, and phone-energy receipts for all four arms are in
the same directory. Immutable raw logs and samples remain on the measurement
host under `/home/zhihao/s42-shape-balance-fp16-v4`.

