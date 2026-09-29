# M2 Step 2: prompt and submission-order matrix

Step 2 PASS at 2026-09-20T23:08:37.490765+00:00. All 8 matched cases and 16 slot comparisons pass before any mapping change.

All arms are single diagnostic runs with 64 outputs per request. The phone arm
includes five host-shadow FFN steps. These timings are not M2 performance acceptance.

| Prompt lengths, requests 0 / 1 | Submission | Slots, requests 0 / 1 | Own ack, requests 0 / 1 | Slot 0 exact | Slot 1 exact | Check |
| --- | --- | --- | --- | --- | --- | --- |
| 256 / 256 | normal | 1 / 0 | 4 / 3 | 64/64 | 64/64 | PASS |
| 256 / 256 | reversed | 0 / 1 | 3 / 4 | 64/64 | 64/64 | PASS |
| 257 / 257 | normal | 1 / 0 | 4 / 3 | 64/64 | 64/64 | PASS |
| 257 / 257 | reversed | 0 / 1 | 3 / 4 | 64/64 | 64/64 | PASS |
| 256 / 257 | normal | 1 / 0 | 4 / 3 | 64/64 | 64/64 | PASS |
| 256 / 257 | reversed | 0 / 1 | 3 / 4 | 64/64 | 64/64 | PASS |
| 257 / 256 | normal | 1 / 0 | 4 / 3 | 64/64 | 64/64 | PASS |
| 257 / 256 | reversed | 0 / 1 | 3 / 4 | 64/64 | 64/64 | PASS |

Host energy is measured RAPL package plus NVML board. Phone power is an
assumption in its own column. Memory scopes are uncapped, with swap disabled.

| Case | Arm | Request host J | Decode host W | ms/token, slot 0 / 1 | Phone W, assumed | memory.peak bytes | events.max ready / finish |
| --- | --- | ---: | ---: | --- | --- | ---: | --- |
| 256/256 normal | control | 5298.780924908763 | 120.26560579331343 | 616.981828125 / 651.707703125 | 0.875 idle | 29481676800 | 0 / 0 |
| 256/256 normal | combined | 3040.7133714876427 | 70.07993165561022 | 537.330515625 / 576.1848125 | 4.5 active; 0.875 idle | 29849464832 | 0 / 0 |
| 256/256 reversed | control | 5293.97336778805 | 119.79118270124881 | 617.53953125 / 652.674609375 | 0.875 idle | 29458706432 | 0 / 0 |
| 256/256 reversed | combined | 3049.9364672241045 | 70.71800538500376 | 536.028453125 / 572.512703125 | 4.5 active; 0.875 idle | 29817171968 | 0 / 0 |
| 257/257 normal | control | 5301.360009392191 | 120.61831100677226 | 611.52496875 / 646.577203125 | 0.875 idle | 29429293056 | 0 / 0 |
| 257/257 normal | combined | 3044.0088789769616 | 70.16180596740556 | 535.325140625 / 570.397609375 | 4.5 active; 0.875 idle | 29838364672 | 0 / 0 |
| 257/257 reversed | control | 5329.425969632785 | 120.52425976713174 | 617.35065625 / 651.535703125 | 0.875 idle | 29394903040 | 0 / 0 |
| 257/257 reversed | combined | 3075.002837177294 | 70.49786567390296 | 538.765515625 / 573.952265625 | 4.5 active; 0.875 idle | 29833818112 | 0 / 0 |
| 256/257 normal | control | 5316.964061593702 | 120.66040498157932 | 617.149015625 / 652.524015625 | 0.875 idle | 29406851072 | 0 / 0 |
| 256/257 normal | combined | 3068.9648335215993 | 70.53791062928875 | 540.09959375 / 576.983015625 | 4.5 active; 0.875 idle | 29831446528 | 0 / 0 |
| 256/257 reversed | control | 5271.929055403554 | 119.3133118731653 | 615.533921875 / 650.1015 | 0.875 idle | 29490085888 | 0 / 0 |
| 256/257 reversed | combined | 3081.1558332108843 | 70.50704542111028 | 539.50421875 / 576.231078125 | 4.5 active; 0.875 idle | 29861371904 | 0 / 0 |
| 257/256 normal | control | 5297.869298107007 | 120.7273493225216 | 611.265765625 / 646.36790625 | 0.875 idle | 29479571456 | 0 / 0 |
| 257/256 normal | combined | 3092.0692225370867 | 71.2133517502612 | 536.94384375 / 573.66084375 | 4.5 active; 0.875 idle | 29882372096 | 0 / 0 |
| 257/256 reversed | control | 5269.510210500916 | 120.92406315850538 | 609.69734375 / 643.665 | 0.875 idle | 29447135232 | 0 / 0 |
| 257/256 reversed | combined | 3070.4155446203013 | 70.97402518297318 | 537.503390625 / 574.56103125 | 4.5 active; 0.875 idle | 29803532288 | 0 / 0 |

The frozen diagnostic runtime identity is
`sha256:bd0dd3ec3b6429ff03e4ea5bd4c2458c9731f47594f93329f2deaa61bb8dea13`.
The mapping is unchanged. Each predecessor must reach its native prefill-start
log before the next request is submitted. Every pair checks native runtime hashes,
fixture hashes, expected slot assignments, per-member proof rows and exact outputs.

Exact driver command from the controller:

```bash
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M2-diagnostic-20260920-a81c09/RUN_MATRIX.sh'
```

Per-arm commands, configuration and runner source are preserved in
`physical/diagnostic/step2/<case>/<arm>.command.json`; results and raw records are
beside them. All 94 rig tests and pyflakes run before each physical arm.
The [fixture notes](README.md) preserve both earlier pilots separately.

The matched (256,257) normal case is exact before any mapping change. Its new
host-only reference, with request slots [1,0], matches both original phone
prefixes for all 64 outputs. For request 1 (257 tokens), its output 4 is token
271; the original host reference in slot 1 produced 198. The new host reference
matches 42/64 positions of that original host prefix, with differences at output
4 and then from output 43. The native build and submission fixture also differ
between the historical and new runs, so this comparison does not isolate the
cause. It shows the original mismatch can also occur between host references.
[Historical comparison record](../physical/diagnostic/step2/p256-257-normal/HISTORICAL_REFERENCE_COMPARISON.json).

The two new host-only (256,257) references also differ under identical native
runtime hashes: request 1 produces token 271 at output 4 in normal order
(slot 0), and token 198 in reversed order (slot 1). They match 42/64 positions
with the same later difference pattern as the original failed pair. Request 0
matches 64/64 across orders. This reproduces the discrepancy without phone FFN
calls; submission order, slot assignment and prefill history change together.
[Host-only order comparison](../physical/diagnostic/step2/HOST_ORDER_COMPARISON_256_257.json).

[Final matrix check](../physical/diagnostic/step2/MATRIX_CHECK.json) verifies all
16 arms used identical native runtime hashes, passed all 94 rig tests and
pyflakes, and recorded zero memory.events.max events at ready and finish.
All eight phone arms have 180 row triples for the first five assisted steps.
Normal submission acknowledged indices 4/3 and reversed submission 3/4.
Cleanup at 2026-09-20 23:07:49 UTC verified all 16 owned scopes inactive, no
owned server, GPU idle, rig lock free and OP15 restored on ADB 5037 with the
expected kernel. [Cleanup record](../physical/diagnostic/step2/CLEANUP.json).
