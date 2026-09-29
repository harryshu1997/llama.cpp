# FP16 BurstGPT phone-arbiter screen V1

Status: `RETAIN_CONTROL_PHONE_ROUTE`

This screen isolates Gemma decode FFN offload through the protected-first
phone arbiter. It is not the 74-request full-trace result and it does not
replace the qualified fixed scheduler.

## Fixed work and placement

All four runs used the same source trace rows and output lengths:

- Qwen requests 52, 53, and 31: 59 input tokens and 57 output tokens;
- Gemma request 50: 271 input tokens and 41 output tokens;
- Qwen3-14B with 15 CUDA layers and OP15 layers 0 through 11;
- Gemma4-12B on eight pinned CPU cores with its 49,152-row LM-head suffix
  resident on the RTX 4060 Ti; and
- three resident OP15 sessions with the same weight hashes and phone-memory
  accounting boundary.

The control kept Gemma FFN local. The treatment routed Gemma decode layers 0
through 22, columns `[9216,15360)`, through HTP0. Gemma prefill phone columns
were zero in both arms. GPU backfilling was disabled so this screen changed
only the Gemma phone route.

The physical order was treatment, control, control, treatment (B-A-A-B).
Model load and warmup were outside the paid interval. CPU-package, GPU-board,
and synchronized whole-phone energy covered the same paid work interval.

## Per-run results

| Arm | Duration s | CPU J | GPU J | Phone J | Fleet J | Qwen s | Gemma prefill s | Gemma decode s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Treatment 1 | 87.214 | 6215.984 | 2379.356 | 196.872 | 8792.212 | 43.511 | 42.387 | 38.474 |
| Control 1 | 104.114 | 6357.225 | 2646.424 | 207.154 | 9210.802 | 38.799 | 56.414 | 38.309 |
| Control 2 | 95.086 | 6463.639 | 2527.839 | 189.902 | 9181.380 | 35.447 | 52.066 | 36.374 |
| Treatment 2 | 100.357 | 6063.607 | 2729.035 | 217.687 | 9010.329 | 44.007 | 51.664 | 39.534 |

The two treatment arms used less fleet energy than their paired controls by
4.545% and 1.863%. Their duration savings were +16.232% and -5.544%, so the
second treatment was slower.

| Mean metric | Control | Treatment | Treatment change |
|---|---:|---:|---:|
| Duration | 99.600 s | 93.786 s | -5.837% |
| CPU-package energy | 6410.432 J | 6139.795 J | -4.222% |
| GPU-board energy | 2587.132 J | 2554.195 J | -1.273% |
| Whole-phone energy | 198.528 J | 207.280 J | +4.408% |
| Fleet energy | 9196.091 J | 8901.270 J | -3.206% |
| Qwen interval | 37.123 s | 43.759 s | +17.876% |
| Gemma prefill | 54.240 s | 47.025 s | -13.301% |
| Gemma decode | 37.342 s | 39.004 s | +4.451% |

The fleet-energy direction is positive, but it is below the scheduler's 5%
minimum margin. More importantly, the routed Gemma decode is slower and whole
phone energy is higher. The apparent duration improvement comes from CPU-only
Gemma prefill, which this treatment does not change. It therefore cannot be
attributed to phone offload.

## Mechanics and quality

Each treatment completed:

- 1,296 protected Qwen calls in 108 balanced 12-layer groups;
- 920 Gemma calls, exactly 40 decode steps times 23 layers;
- a timestamped Qwen completion transition;
- four serialized router sessions and 2,216 accounted phone requests; and
- zero filler-upper, guard, idle-lower, pending-protected, reset, and terminal
  router violations.

The measured Qwen phone-idle minimums were 353.134 and 353.252 ms. Maximum
Gemma switch-execute-switch time was 12.166 and 12.861 ms, below the 50 ms
upper bound with a 20 ms guard.

Qwen output was byte-identical across all four runs. Gemma was stable within
each arm. The approximate HTP treatment had 37/41 positional agreement with
the CPU control and a 36-token common prefix.

## Decision and next bounded change

The checked reducer is
[`PHONE_ARBITER_ABBA_V1.json`](../results/PHONE_ARBITER_ABBA_V1.json). It fails
three promotion gates: one duration pair regresses, mean fleet saving is below
5%, and mean Gemma decode time regresses. Its decision is
`RETAIN_CONTROL_PHONE_ROUTE`; `energy_claim_eligible` remains false.

The trace shape explains why the arbiter did not fill Qwen phone gaps. Gemma's
271-token CPU prefill did not expose a decode-ready HTP call until Qwen was
finishing. The next candidate must first measure bounded Gemma prefill chunks
at M=1, 2, 4, 8, and 16. Only shapes whose complete switch-execute-switch
upper bound and energy interval fit the measured Qwen idle window may be
scheduled. A decode-only causal screen should also move identical prefill work
outside the comparison boundary. No full-trace run is authorized until that
candidate passes repeated alternating fleet-energy and duration gates.
