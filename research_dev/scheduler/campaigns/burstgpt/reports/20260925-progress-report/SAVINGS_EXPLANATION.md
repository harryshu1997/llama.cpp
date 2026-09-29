# Why the headline changed from about 25% to 58.7%

Read-only analysis of saved results on 2026-09-25. Geometry validation PASS:
current policy-derived layer counts exactly match request-scoped physical call proofs.
The current OP15-only result is 54.289%; 58.687% is the scheduler + OP15 + Pixel configuration.
Both use the new dispatcher and compare against the legacy desktop baseline.

## A larger FFN layer set is served by the phones

The historical 25.827% host result is v16/v16c from September 12: 145.041 -> 107.582 kJ.
Its 25.118% figure included assumed phone energy. The old and current pairs use different
traces and deployments, so the difference between their percentages is not a controlled ablation.

| Geometry when assistance is active | Historical v16 | Current OP15-only | Current OP15 + Pixel |
| --- | --- | --- | --- |
| Qwen FFN layers on OP15 | Mostly 12; some 18, one 6-layer window | 18, layers 0-17 | 18, layers 0-17 |
| Qwen FFN layers on Pixel | 0 | 0 | 6, layers 18-23, when Pixel is active |
| Gemma FFN layers on OP15 | 8, layers 8-15 | 24, layers 0-23 | 24, layers 0-23 |

In the old observation store, Qwen has 548 assisted window tokens with 12 layers, 79 with
18, and one with 6. Gemma has 1,014 assisted window tokens with 8 layers: 981 at full width,
and 11 each at 25%, 50%, 75% width. These historical counts are observation-window counts;
current full-output coverage below includes the separately verified tails.

"100% split" describes all columns of the selected layer set. The old eight-layer Gemma
policy and current 24-layer Gemma policy therefore remove substantially different amounts
of CPU FFN work, even at the same column fraction. Geometry expansion is a measured change;
its isolated contribution to total energy has not been measured by an ablation.

## The measured reduction is mostly CPU energy

| Arm | Average CPU W | Average GPU W | Duration s | Host kJ |
| --- | ---: | ---: | ---: | ---: |
| Current legacy desktop | 71.233 | 30.594 | 2244.348 | 228.535 |
| Current dispatcher + OP15 | 27.663 | 29.677 | 1821.864 | 104.465 |
| Current dispatcher + OP15 + Pixel | 20.887 | 30.622 | 1832.951 | 94.414 |

Against legacy, the two-phone CPU saving is 121.587 kJ and GPU saving 12.534 kJ.
CPU accounts for 90.655% of the total 134.121 kJ difference. This is an accounting
decomposition, not a separate causal allocation to each optimization.

Total host power falls from 101.827 to 51.509 W, and duration falls 18.330%:

```text
saving = 1 - (51.509 / 101.827) * (1832.951 / 2244.348) = 58.687% (rounded inputs)
```

## Scheduling and workload also changed

Both current phone arms enable work-conserving admission and model affinity. Queued work
can use free capacity, and requests for an already loaded model can avoid unnecessary model
switches. Current model reloads are 9 for legacy and 7 for each phone arm. The dispatcher-only
control on this same trace was still running at 17:42 UTC; it must finish before assigning an
independent scheduler percentage on eval_v2. The separate 31-request trace measured 26.716%
dispatcher-only saving; that percentage cannot be added to this trace's phone saving.

| Workload | Historical v16 | Current eval_v2 |
| --- | ---: | ---: |
| Requests | 24 | 14 |
| Output tokens | 2,274 | 3,604 |
| Mean output tokens per request | 94.75 | 257.43 |
| Unassisted Llama share of output tokens | 14.51% | 1.28% |

The longer decodes amortize preparation costs and provide more opportunities for assistance.
The model mix also puts more token work on Qwen/Gemma, the assisted models. These are reasons
the current workload can favor offload; their separate energy effects remain unmeasured.

## Interpret the Pixel difference as a configuration result

The two-phone arm uses 10.051 kJ / 9.622% less host energy than OP15-only, equivalent to
4.398 additional percentage points against legacy. Duration is 0.609% longer. Coverage differs:

| Model | OP15-only assisted tokens | Two-phone assisted tokens |
| --- | ---: | ---: |
| Qwen | 1459/1524 (95.73%) | 1298/1524 (85.17%) |
| Gemma | 1275/2034 (62.68%) | 1687/2034 (82.94%) |
| All models | 2734/3604 (75.86%) | 2985/3604 (82.82%) |

Pixel serves Qwen, yet the two-phone arm also has more Gemma assistance through OP15.
Thus the 9.622% difference includes changed scheduling/assistance behavior; it is not an
isolated measurement of Pixel's six-layer execution efficiency. Single runs cannot establish
repeatability. The shorter dev_v2 repeat means favor OP15 instead.

Historical v16 matched 24/24 output sequences exactly. Current strict acceptance FAILS:
OP15 12/14 and OP15+Pixel 13/14 vs legacy, 12/14 between phone arms. No quality-equivalence
exception is established. Host energy excludes phones; no measured whole-system saving is claimed.

Sources: [extracted geometry and source hashes](SAVINGS_EXPLANATION.json),
[current energy/output audit](sources/rig_audit_20260925T171405.json),
[historical host-energy audit](../20260921-fast-path-trace-v2a/physical/HISTORICAL_V16_ENERGY_AUDIT.json),
[historical v16 report](../20260912-admission-context-watcher/README.md).
