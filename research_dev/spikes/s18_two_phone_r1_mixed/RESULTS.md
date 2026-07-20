# S18 two-phone independent R1 result

Verdict: `S18_R1_FLEET_MECHANICS_PASS_RELIEF_INSUFFICIENT`

The corrected independent-lane R1 design ran end to end on one selected A6000,
OP15, and OP12. The six-row matched acquisition and an independent replay both
completed. No phone, USB, host-wall, or total-system energy claim is made.

## Frozen experiment

- High-priority service: BGE-small-en-v1.5 on CUDA0 at its measured B16 knee.
- Low-priority service: Gemma-4-12B, six 64-request rounds per row, eight decode
  tokens per request.
- P0 control: two sequential full-model CUDA B32 groups per round.
- P4 treatment: OP15 `[0,8)` and OP12 `[0,6)` own independent B32 request groups
  with their own CUDA tails. OP12 has two exchanges of certified credit. Later
  loose-SLO groups run on OP15 after the tight-SLO group.
- Three rotated matched pairs: P0,P4 / P4,P0 / P0,P4.
- GPU1 remained idle. Model preparation and loading were outside the paid BGE
  interval.

## Result

| Metric | P0 control | P4 two-phone R1 | Gate |
|---|---:|---:|---:|
| BGE p95, median | 3,875 us | 3,877 us | PASS, 1.0005x <= 1.05x |
| BGE work per row | 339,440 encodes | 339,440 encodes | PASS |
| Gemma work per row | 384 requests / 3,072 tokens | same | PASS |
| Gemma token identity | exact B32 oracle | exact B32 oracle | PASS |
| Tight-class completion | n/a | 2.956 s median, 3.140 s max | PASS, <= 5 s |
| Loose-class completion | n/a | 5.908 s median, 9.703 s max | PASS, <= 12 s |
| OP15/OP12 overlap | n/a | 3.049 s median | PASS |
| Selected-GPU peak memory | 26,555 MiB | 45,674 MiB | diagnostic |
| Selected-GPU energy saving | n/a | 0.342% median | FAIL, < 10% |
| Uncertainty-adjusted saving | n/a | -764.1 J median | FAIL |

All three raw pair differences favor P4, but only by 0.075%, 0.342%, and
0.610%. Each is smaller than the Ampere board-power uncertainty allowance.
The selected-GPU energy gate therefore fails even though the execution,
correctness, SLO, overlap, and high-priority isolation gates pass.

The independent validator reopened the six power traces, BGE output and
placement records, all CUDA control results, all host placements, 30 OP15
sessions, six OP12 sessions, and every content digest. It reproduced report
digest `sha256:df76f20561fcb065744d8047efff2fbc37e3d72ecc8d6415948a74490e63fbf5`.

## Interpretation

R1 is a valid real three-device mechanism, but it is not yet an energy-saving
configuration. A full-model CUDA B32 group takes about 0.616 s, compared with
2.875 s through OP15 and 9.412 s through OP12. Concurrent high-priority BGE
keeps CUDA0 useful and hides the phone latency from that service, but two
different layer cuts require two resident CUDA tail images. Peak selected-GPU
memory consequently rises by about 19.1 GiB instead of falling.

The next gate is one multi-ingress CUDA tail that accepts both phone boundaries
without duplicate tail weights. It must preserve independent request ownership,
bounded device credits, exact token replay, and the current SLO gates. This is
the smallest implementation that can turn the demonstrated concurrency into
an HBM-capacity benefit. Repeating the current duplicated-tail energy run is not
useful.

## Negative evidence retained

- CUDA B64 produced a stable token sequence different from the certified B32
  sequence at both 512- and 1,024-token KV capacity. It is not an exact-output
  control for this gate.
- Two 512-token CUDA tail contexts exceeded the A6000 memory budget. The frozen
  16-token request envelope fits and still covers the complete 8+8 bound.
- Extrapolating OP12 from two certified exchanges to six caused exchange 4 to
  time out. R1 now enforces an explicit credit of two.
- S17 R2 middle-island coalescing remains numerically ineligible and is not used
  by this result.
