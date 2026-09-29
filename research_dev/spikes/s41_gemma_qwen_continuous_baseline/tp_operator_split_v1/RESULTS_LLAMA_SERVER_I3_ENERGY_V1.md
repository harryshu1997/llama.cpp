# llama-server I3 real-device fleet-energy campaign

Date: 2026-08-06 EDT.

Verdict: `THREE_PAIR_PHYSICAL_PASS; BURSTGPT_MIX_ENERGY_PASS; SERVER_ENERGY_PASS; MMLU64_NONINFERIOR; MEAN_OVERLAP_PASS; GENERAL_PER_SHAPE_ENFORCE_BLOCKED`.

Three alternating control/treatment pairs completed the same source-length
BurstGPT-derived trace on the real RTX 4060 Ti desktop and OP15. The final I3
FFN policy reduces average trace makespan by 14.38%, server CPU-package plus
GPU-board energy by 17.30%, and accounted compute-device fleet energy by
16.76%. Every individual pair improves both makespan and fleet energy.

This is a physical result for the measured workload mix. It is not an AC wall
energy result and it is not a universal per-batch route certificate.

## Headline

Both arms complete 74 requests, 33,843 input tokens, and 11,605 output tokens.
They use the same arrivals, artifacts, continuous-batch settings, default
desktop CPU selection, normal repacking, hot Qwen3-14B CUDA route, and
`memory.swap.max=0` scope.

| average over three pairs | CPU control | CPU plus OP15 | change |
| --- | ---: | ---: | ---: |
| trace makespan | 736.468 s | 630.594 s | -14.38% |
| output throughput | 15.758 token/s | 18.404 token/s | +16.79% |
| cold mean prefill/request | 61.636 s | 46.875 s | -23.95% |
| cold mean decode/request | 241.689 s | 212.841 s | -11.94% |
| cold mean service/request | 452.455 s | 376.117 s | -16.87% |
| cold mean TTFT/request | 210.766 s | 163.276 s | -22.53% |
| CPU package energy | 116.059 kJ | 93.623 kJ | -19.33% |
| GPU board energy | 20.085 kJ | 18.963 kJ | -5.59% |
| server compute-device energy | 136.144 kJ | 112.586 kJ | -17.30% |
| whole connected phone energy | 1.073 kJ | 1.631 kJ | +51.99% |
| accounted fleet energy | 137.217 kJ | 114.217 kJ | -16.76% |
| accounted fleet J/output token | 11.824 J | 9.842 J | -16.76% |
| completed work | 74 req / 11,605 tok | 74 req / 11,605 tok | equal |
| SLO requests met | 55 | 55 | equal |

Individual repetitions are stable:

| pair | control time | treatment time | time change | control fleet | treatment fleet | fleet change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 737.290 s | 626.189 s | -15.07% | 137.564 kJ | 114.516 kJ | -16.75% |
| 2 | 737.310 s | 632.916 s | -14.16% | 137.419 kJ | 113.866 kJ | -17.14% |
| 3 | 734.804 s | 632.677 s | -13.90% | 136.670 kJ | 114.269 kJ | -16.39% |

The connected phone is charged to the control for its complete paid interval.
Its treatment energy rises, but only by 0.558 kJ on average. The 23.558 kJ
server reduction is much larger, so the phone does not erase the system win.

## Final I3 policy

The one declared execution change from I1 is the batch-shape cut table:

```text
policy id: i3-hidden-wait
table: 1:9664,3:8192,8:4096,128:8192,512:11136
```

This assigns 9,664 phone columns at M=1, 8,192 at M=2 through M=3, 4,096 at
M=4 through M=8, 8,192 through M=128, and all 11,136 columns above M=128.
Models, weights, F16 activation exchange, kernels, worker residency, arrivals,
and the 11,136-column safety ceiling are unchanged.

The treatment performs 52,320 paid per-layer phone calls. It sends and returns
6,876,610,560 bytes per trace, executes 88.245 trillion phone MACs, and covers
76.82% of eligible dense-FFN MACs after weighting each shape by tokens. This is
not 76.82% of the full model: attention, embeddings, norms, the vocabulary
head, and the unsliced FFN work remain on the desktop.

Arithmetic-mean exposed join wait falls from I1's 22.55% to 2.67%, passing the
5% trace-mix gate. There are zero bridge reset recoveries in all three paid
treatments.

The route is still shape-conditioned. M=1, M=2, and M=3 expose about 21%, 16%,
and 14% wait in pair 1, while the high-frequency M=4 through M=8 shapes expose
approximately 0% to 1%. A general online scheduler must use the qualified
shape bucket and observed batch mix. It must not extrapolate the 2.67% trace
average to a low-load M=1 workload.

## Why the whole-trace saving is smaller

The phone does not accelerate the complete inference graph. It overlaps only
the sliced dense FFN branch of the 17 cold Gemma requests. The remaining Gemma
operators, the host FFN suffix, merge and transport, all 57 hot Qwen requests,
and the 55.75-second arrival window remain. Prefill improves 23.95%, but the
much longer decode phase improves 11.94%; decode therefore limits the final
14.38% makespan reduction. This is the measured Amdahl limit for this route,
not evidence that the phone branch is idle.

## Quality gate

Cross-run greedy equality is diagnostic only. Continuous-batch geometry also
changes untouched CUDA output between independent controls. The I3 path-matched
four-prompt probe is exact for 2 / 4 request sequences and 21 / 32 token
positions, so no bit-exact or greedy-exact claim is made.

The pinned 64-item MMLU gate uses the same Gemma artifact and byte-identical
server runtime manifest in both arms:

| route | correct | parseable | duration |
| --- | ---: | ---: | ---: |
| CPU control | 27 / 64 | 64 / 64 | 146.975 s |
| I3 CPU plus OP15 | 27 / 64 | 64 / 64 | 95.760 s |

Only item 7 and item 61 change answer, with one loss and one gain. The final
score is equal, above the frozen 25 / 64 floor, and has no accuracy regression.
This authorizes bounded approximate/task-quality use, not exact-token use or a
broad model-quality claim.

Quality evidence:

```text
/home/zhihao/s41-dynamic-ffn-v1/server-quality/i2-pathmatched-cpu-20260806T1513Z/RESULT.json
sha256:8ba63d24b93522fd48fb3088c88bd72805840a2555ba997621bffb7d1685b9ef
/home/zhihao/s41-dynamic-ffn-v1/server-quality/i3-pathmatched-op15-20260806T1646Z/RESULT.json
sha256:72550555b68e043eceb17556e7086d8e8bb2aa07bcb172eaed4e9951f5361fbd
/home/zhihao/s41-dynamic-ffn-v1/server-quality/i2-r2-cohort-cpu-r2-20260806T1556Z/RESULT.json
sha256:43e98334b94edb0da5409b73ed266d86a2a6690c79f54f68f01510698fdc566b
/home/zhihao/s41-dynamic-ffn-v1/server-quality/i3-cohort-op15-20260806T1651Z/RESULT.json
sha256:f4197e5daa925339711275f045de73eed40fd70f94cff97cd1e599be7f037e07
/home/zhihao/s41-dynamic-ffn-v1/server-quality/i2-r2-mmlu64-cpu-20260806T1618Z/RESULT.json
sha256:9201e33e9ea889835cacfc00c969bbc3c0b6c43132de330982040841e0fbf498
/home/zhihao/s41-dynamic-ffn-v1/server-quality/i3-mmlu64-op15-20260806T1701Z/RESULT.json
sha256:3ac6a371cbad5367fe15c28135e3f13f13aa370ddb62b159056aa693c359967e
```

## Energy boundary and sensor correction

All components use the same monotonic paid interval:

```text
server = Intel package RAPL + RTX 4060 Ti NVML board power
phone  = OP15 USB input + simultaneous battery discharge
fleet  = server + phone
```

GPU and phone power are trapezoid-integrated. CPU energy uses the unwrapped
RAPL counter with boundary interpolation. OP15 uses USB current and voltage
plus the OPLUS battery current node and battery voltage. On this phone, the
battery node reports mA and positive values mean discharge. Pair 3 records a
-10,000 uAh charge-counter change while the sampled current is positive,
independently confirming the vendor sign.

The first phone logger mislabeled the OPLUS value as uA. A V2 reducer corrected
the magnitude but initially retained the generic Linux current sign. Both are
invalid and preserved. V3 is the sole energy authority: it binds the raw sample
hash, clock anchors, result hash, mA unit, positive-discharge convention, and
the USB-plus-discharge method. No raw sample or server result was overwritten.

This boundary excludes AC conversion, motherboard, fans, storage, and DRAM
outside package RAPL. The desktop exposes no whole-platform power sensor. A
true desktop or total-system wall-energy claim still requires an external AC
meter and must be reported as a separate boundary.

## Physical evidence

```text
pair 1 control RESULT  5a2ee5dfb451c6fa22fdc8775d21d3210b962e377d77160443712a323449479f
pair 1 control phone   a549c54504df4817cfff8e599208491267d3601f64cadc8e3688aec55af4a6a9
pair 1 treatment RESULT fcba79e95866a21751661e77caff2c7b4a5fab1e5e5989feac755010ed3f9f6d
pair 1 treatment phone  c1f6c15c3e7aed95654798d7a3dab5f9d73a0800d30080aed08c72c1c16eddea

pair 2 control RESULT  40a09b20843e7f6ce8e906892e3c725de5c6bb9abce484fd9d21c141619a6252
pair 2 control phone   6b0de6ace0616cc78f59527c7431e0aac9bb81bd703abe7160b9732b56afc37b
pair 2 treatment RESULT 5bc6b5b13c2aceff34f80c723ffdcdc040ac4f2e3ccd881113a06efb5a2f1ffd
pair 2 treatment phone  b42cd4d5e547758f997b0fad275067d42940b70def26c6bca23f93d6678412a8

pair 3 control RESULT  bc6494fbc678bf5f52c25f946cc36e3ebf43cb087dffd5801884579f6c51f416
pair 3 control phone   a8cb15ccf1103b317c3a2be88ac7d1f7aef71d0fdb425529c248f08b9df15583
pair 3 treatment RESULT a3669f9c4e0eba1a0187d5fc3603f7609453106e1fc591cbeb3189d64ba657ce
pair 3 treatment phone  d268d28938c504e4d5aa26654ca6103a168295499c1834e3bd2cbb58ac8d29a4

aggregate JSON: /home/zhihao/s41-dynamic-ffn-v1/server-traces/I3_CAMPAIGN_V3.json
aggregate file sha256:d4e4586125b27eecea0c2e02af903f6ef52922a2a29457eaff715d4312094da0
aggregate record sha256:5d144dd797511b4f3aa15f356ef1a8d6bbdacae81abbec82807622f02dc4df42
rendered Markdown sha256:af63ddbf173ede7b92b64dbe4c0b06d968b79ed327e6c8718e9273027dc30265
trace sha256:b20a9ba66ee3558d835a0e19ed3cfa4c31a4a9e8b4f9c085b29a14f80250a0ff
```

V3 re-reduces the immutable paid artifacts and adds explicit gates for actual
control/treatment interval alternation, nondecreasing SLO count, default CPU
affinity, exact GPU identity, identical runtime manifests, role work, the I3
shape table, and exact phone call and byte reconciliation. It changes no raw
sample or physical result. Every added gate passes. V2 remains preserved as
the earlier aggregate.

Two setup attempts are excluded. One control reached CUDA allocation while a
foreign dual-CUDA experiment held VRAM and failed before the paid trace. One
treatment entered accessory setup just as another foreign run launched; it
was stopped before `run_server_trace.py` and before any paid marker. A
zero-call protocol shutdown restored OP15. Neither attempt is a replacement
or a measured repetition.

## Scheduler conclusion

The four user-requested implementation points are now covered:

1. The control uses the default desktop CPU without affinity or frequency
   throttling.
2. The I3 split balances the dominant M=4 through M=8 branches and passes the
   trace-wide 5% overlap gate.
3. MMLU64 is noninferior, while exact-token quality remains correctly
   unclaimed.
4. Server CPU-package plus GPU-board energy and whole connected-phone energy
   are measured on synchronized intervals. AC wall energy remains unavailable.

The fixed BurstGPT-mix policy passes the stricter 10% fleet-energy gate. It may
be admitted for the exact measured workload/profile epoch. The general S42
per-request energy model must remain fail-closed outside that profile because
continuous batching shares power and work across requests, and several rare
shape buckets individually exceed the wait limit. The next narrow scheduler
change is shape-bucket eligibility plus a cohort/epoch energy scope, not more
operator mechanisms.

## Verification

- Three uncontended paid control/treatment pairs: pass.
- Completed work: 74 / 74 requests and 11,605 / 11,605 output tokens in every run.
- CPU and hot-process swap: zero in every run.
- Phone placement, work counters, byte reconciliation, and cleanup: pass.
- Bridge reset recoveries: zero.
- Corrected V3 phone sensor unit/sign and charge-counter consistency: pass.
- Pinned MMLU64 score and runtime binding: pass.
- Focused policy, MMLU, phone-energy, campaign, and server-energy tests:
  24 / 24 pass.
- S42 scheduler tests: 48 / 48 pass.
- Python and shell syntax: pass.
- `git diff --check`: pass.
- OP15 rebooted after acquisition; normal ADB and USB mode restored; no worker remains.
- No commit or push performed.
