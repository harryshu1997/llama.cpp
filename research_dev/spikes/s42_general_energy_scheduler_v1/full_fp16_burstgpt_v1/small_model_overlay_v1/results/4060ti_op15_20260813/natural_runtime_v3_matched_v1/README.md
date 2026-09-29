# Natural runtime route V3 matched result

This directory preserves the 2026-08-13 physical matched comparison on the
RTX 4060 Ti desktop and OP15. Both arms execute the same 74-request FP16
BurstGPT trace plus the same ten Llama 3.2 1B requests. Both use the
`cpu-overflow` large-model policy, the same model hashes, and runtime profile
hash `b2a9b719397bbf294e3d3429c8434360f56386144fe3a844ccc04b83dbfe4c69`.
Only the small-model policy changes.

| Metric | Static CPU | Runtime scheduler | Runtime change |
| --- | ---: | ---: | ---: |
| Fleet energy | 320.429 kJ | 311.752 kJ | -2.708% |
| Server compute energy | 312.790 kJ | 304.002 kJ | -8.789 kJ |
| Whole-phone energy | 7.639 kJ | 7.750 kJ | +0.111 kJ |
| Makespan | 2914.275 s | 2875.218 s | -1.340% |
| Mean GPU utilization | 39.663% | 40.376% | +0.714 points |
| All-trace SLOs met | 2/84 | 10/84 | +8 |
| Mean 1B completion | 60.724 s | 5.760 s | -90.515% |
| P95 1B completion | 109.034 s | 11.942 s | -89.047% |

The unified scheduler selected `phone-adreno` for all ten Llama requests.
The phone server log contains all ten endpoint task IDs. Every active lease
remained occupied until physical completion and all ten requests finished
inside the calibrated service upper bound. The static arm physically ran all
ten requests on four desktop CPU lanes. Both physical qualification receipts
are `PASS`.

`RUNTIME_PROFILE_AUDIT.json` records the disjoint calibration. The CPU route
has 21 training and 13 holdout observations; the phone route has nine
training and seven holdout observations. Both routes have zero holdout
upper-bound violations. The subsequent runtime trace is not part of either
calibration split.

The mean unified scheduler core cost was 0.284 ms per scheduling attempt.
The complete controller path averaged 35.824 ms because it includes live host
and GPU capacity capture; endpoint health and phone state came from bounded
background snapshots. Nineteen attempts covered ten arrivals plus
completion-driven virtual-queue replans.

The saving comes from avoiding concurrent 1B CPU inference while Qwen uses
CPU-resident layers. Server CPU package energy fell by 5.843 kJ and GPU board
energy fell by 2.945 kJ. The phone was resident in both arms, so executing the
ten requests increased its full-trace energy by only 0.111 kJ. The net fleet
saving is 8.677 kJ.

This is one matched physical pair. It proves runtime route availability,
competitive physical selection, bounded virtual queuing, and a positive
observed result. It is not a repeated confidence-interval energy claim. The
earlier 25.54% result changes the large-model OP15 policy across all 74
BurstGPT requests; it is not an achievable comparison target for rerouting
only ten 1B requests at fixed `cpu-overflow` policy.

After this matched pair, the physical wrapper gained a `runtime-auto` mode.
`AUTO_DRY_SNAPSHOT.json` is a fresh live snapshot from the same desktop and
phone. `AUTO_DRY_PLAN.json` records that the unified planner selected
`fp16-server-gpu-cpu-op15-switch-v1`, derived the physical `op15` arm, and
reported a conservative 24.8767% energy margin. It rejected the full-GPU and
staged alternatives for live VRAM and host-RAM capacity, respectively. The
runner independently parsed the plan hash and resolved `auto` to `op15`.
This is a planning and physical-binding receipt, not another inference result;
the long replay was not started because the OP15 reported 1% battery. On the
final readiness check it remained at 1%, had fallen to 3.096 V, and reported
USB power without charging. Starting a 45-50 minute run in that state would
not provide a trustworthy complete-trace energy comparison.

The autonomous implementation and its deployed desktop mirror pass the full
unified scheduler and S42 regression suites. The runtime-auto profile rebuilds
byte-for-byte from its four measured inputs, and every immutable result in
`SHA256SUMS.txt` still verifies. A new end-to-end autonomous physical result
must remain a separate artifact after the phone is charged; it must not be
inferred from the dry plan or from the fixed-policy matched pair above.
