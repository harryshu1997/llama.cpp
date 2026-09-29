# Three-request test after whole-phone control recovery

The one adaptive run completed 3/3 requests with semantic output validation and
exact terminal proofs. No baseline, 24-request trace, or 84-request trace was run.
No production scheduler code changed during this retest.

The unchanged `burstgpt_dev3_long_v1.json` supplied Gemma 36 at 1 s, Llama 37 at
61 s, and Qwen 50 at 91 s, with 292/292/71 output tokens. Selection was normal
`energy-aware`, not calibration or forced routing. CUDA graphs were enabled.
Desktop placements, native binaries, artifacts, FFN shard indexes, initial
evidence and accounting boundaries match the frozen modified-runtime references.

The new configuration registers the existing whole-phone endpoint with explicit
NCM control and a conservative 3,000,000,000-byte peak reservation. That number
is configuration, not measured OpenCL peak qualification. Shared-resource
exclusions, memory admission and route qualification were not bypassed.

## Results

Paid duration was **316.882832 s**, including runtime preparation and cleanup.
The preflight artifact checks are outside that boundary, as in the references.
No cache flush, prefetch, advance residency or request truncation was used.

| Request | Output | Service time (s) | Arrival-to-completion (s) | FFN calls | Assisted tokens | Fraction-weighted all-token coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 36 | 292/292 | 133.936 | 179.264 | 6,720 | 280/292, 95.89% | 90.24% |
| Llama 37 | 292/292 | 1.596 | 129.683 | 0 | 0/292 | 0% |
| Qwen 50 | 71/71 | 62.542 | 219.628 | 66 | 11/71, 15.49% | 15.49% |

Gemma used 100% for 247 tokens, and 75%, 50%, and 25% for 11 tokens each;
12 tokens were unassisted. All assisted calls covered CPU-resident FFN layers
0-23, with 3,840/7,680/11,520/15,360 columns. Each HTP session recorded 2,240
Gemma calls. The previous sustained-assistance tail failure did not recur.

Qwen used 100% for 11 tokens on HTP2, covering CPU FFN layers 12-17 at 17,408
columns. Sixty tokens were unassisted. These percentages describe the selected
FFN slices, not offloading the whole model or all its FFN layers. Counts reconcile
native per-layer calls, acknowledged policies and terminal proofs.

### Remaining control and routing limitations

- Qwen reached `VERIFICATION_INCONCLUSIVE` at token 20, with 51 requested output
  tokens remaining. Reason: `BOUND_RESOLUTION_UNAFFORDABLE`. It returned to 0%
  after its probe rather than manufacturing qualification. This is not a
  measured energy-negative rejection; sustained Qwen assistance remains unsolved.
- Llama's phone candidate was recorded in the decision journal but remained
  `MODEL_EPOCH_AUDIT_ONLY`. The selected model epoch and actual execution stayed
  on the qualified desktop parent. This run did not execute whole-phone Llama,
  did not demonstrate joint Adreno/HTP compute, and did not physically exercise
  whole-service memory reuse. Those must not be inferred from successful FFN use.
- Registering the endpoint is not sufficient to authorize it in the model epoch.
  Resolving that path must preserve memory/transport qualification and shared
  resources, not just remove the audit-only marker.

## Energy and historical references

CPU package energy: **12.599853 kJ**, measured through RAPL. GPU board energy:
**9.134781 kJ**, measured through NVML. Phone energy: **0.667623 kJ**, assumed
at 4.5 W active and 0.875 W idle. Recorded phone activity was 107.682941 s active
and 209.199891 s idle, exactly covering the paid boundary.

| Run | Fleet energy at 4.5 W (kJ) | Duration (s) | New run saving vs reference |
| --- | ---: | ---: | ---: |
| This adaptive run | 22.402 | 316.883 | - |
| Frozen clean upstream desktop | 31.866 | 371.198 | 29.70% |
| Frozen matched modified desktop | 31.699 | 355.145 | 29.33% |
| Frozen fixed GGG | 22.492 | 312.197 | 0.40% |
| Frozen fixed GGQ | 25.337 | 315.719 | 11.58% |
| Frozen fixed GQQ | 31.963 | 331.500 | 29.91% |
| Frozen fixed QQQ | 33.325 | 384.840 | 32.78% |

Only final `references-source-v7` results were read. Their saved hashes match.
These are historical-reference differences, **not a fresh matched A/B**: scheduler
source and capability catalog changed. The strict existing A/B validator rejects
the comparison with `A/B identity differs: catalog_sha256`; it was not weakened.
`SUMMARY.json` preserves decoded identity differences and checks the unchanged
workload, native binaries, desktop parents, phone artifacts, evidence and boundary.
Upstream-versus-modified binary and qualification identities remain explicit.

The 0.40% advantage over GGG is too small to establish superiority from one run;
GGG was also 4.69 s faster. Do not attribute these differences solely to the new
NCM control, which was armed but did not carry a whole-model request here.

| Assumed active phone power | Fleet energy (kJ) | Saving vs frozen matched desktop | Saving vs frozen GGG |
| --- | ---: | ---: | ---: |
| 3 W | 22.241 | 29.84% | 0.50% |
| 4.5 W | 22.402 | 29.33% | 0.40% |
| 6 W | 22.564 | 28.82% | 0.30% |

The previous telemetry/Qwen adaptive run used 23.327 kJ in 290.021 s. This run
used less energy but took 26.862 s longer. Desktop transition receipts changed
as follows: Gemma 13.562 -> 40.876 s; Llama 2.363 -> 7.965 s; Qwen 58.545 ->
55.019 s. These are measured preparation intervals, not an attribution of every
second of run-time difference. No overlapping windows were subtracted to invent
steady-state energy. The 390 host observations recorded no compiler processes;
sampled peak total VRAM was 15,521 MiB, including GDM.

## Session lifecycle and telemetry

| Load | Session generation | Scheduler PREPARING to verified observation (s) | Physical load authorization to READY (s) | READY publication (s) |
| --- | ---: | --- | ---: | ---: |
| Gemma HTP0 | 1 | 4.282 - 24.142 | 9.387 | 24.408 |
| Gemma HTP1 | 1 | 24.460 - 34.351 | 9.591 | 34.784 |
| Gemma HTP2 | 1 | 35.348 - 46.457 | 10.390 | 46.725 |
| Qwen replaces HTP2 | 2 | 182.616 - 199.897 | 16.645 | 200.408 |

First session published READY at 24.408 s; all three at 46.725 s. Initial
proposal-to-preparation was 0.813 s using scheduler observation timestamps.
Physical durations above use same-phone monotonic timestamps; host and phone
wall clocks are not subtracted. Each load has independent LOADING, VERIFIED
and READY records and an actual `weight_source=ffn_shard` worker proof.

Four loads succeeded: three initial shards and one replacement. Only HTP2's
generation changed, 1 -> 2. HTP0/HTP1 retained their Gemma artifacts, geometry,
operator plans and generation 1. Final resident weights were 8,870,952,960 bytes.
There were no load failures, abandoned proposals, execution fallbacks, stale
execution failures, request restarts or USB reset recoveries.

Gemma's envelope first attached to HTP0/HTP1 at 45.870 s and expanded to all
three before positive execution. Its desktop execution started at 46.328 s,
before the last session's READY publication. Gemma's first positive control was
acknowledged at 69.199 s. Qwen attached at 246.867 s and its positive control was
acknowledged at 267.819 s. No weights reloaded for fraction changes.

Replacement began after Gemma completed at 180.264 s. Therefore this workload
does not prove retained active calls during replacement, reverse replacement,
or injected rollback. Those previously working mechanisms were not rebuilt.

All 404 saved phone health observations were VALID, maximum age 2.569265 s under
the unchanged 5 s limit. No telemetry recovery/reset was needed. The NCM control
bootstrap was armed with the verified boot ID; normal HTTP telemetry remained
available. RESULT.json contains 502 direct helper lifecycle events and physical
control/session evidence.

The existing wait analyzer reports 4/4 layouts READY, zero failed or unprepared,
23.1 s with no initial layout, and 1.6 s executing without model residency (Llama).
Qwen's first covering shard was 108.9 s after its arrival: useful Gemma residency
was retained while Gemma ran. This is not evidence that the old 24-request
coverage targets are all met. `PHONE_WAIT.json` preserves the analyzer output;
its rounded observed timestamps differ from explicit publication timestamps.

Physical CUDA evidence records 76 captures, 48 instantiations, 2,388 graph
launches and 28 recaptures by the existing update-minus-instantiation metric.
All recorded graph API return values were successful.

## Validation, changes and artifacts

- Full physical preflight PASS; 3/3 execution and terminal-proof validation PASS.
- Post-run audit PASS: USB restored to `ptp,adb`, no owned phone workers,
  qualified kernel unchanged, VRAM back at 3,178 MiB, GDM PID 6871 unchanged.
- 383 executed source files still match their manifest. All 857 physical artifact
  files copied locally and verified, zero hash mismatches.
- Reused the preceding 229 distinct focused/related test results. No production
  edit or additional broad suite was needed; replay inputs/goldens were untouched.
- Changed only this report directory (`experiment.py`, `analyze.py`, `audit.py`,
  reports and archived evidence) and `research_dev/talks.md`. No commit or push.

Remote run: `/mnt/storage/s42-ncm-dev3-20260910-v1/`.
Deployment: `/mnt/storage/s42-ncm-dev3-20260910-v1-deploy/`.
Local immutable run copy: `physical/`. Machine-readable comparison: `SUMMARY.json`.

| Artifact | SHA-256 |
| --- | --- |
| RESULT.json | `ef89cf3412e2bf87e28f6e6f817f6fd862b85651567d9f4ce4490135f71624f7` |
| Execution source manifest | `66c19572a7a9ed73ccd13f95ebdb8817fd0dcb928a25416afefd3176a12e1250` |
| Capability catalog | `6382cd07d5c7079e085443aca251c9186b0a5133f960b10270158fdb97daccc5` |
| Physical artifact inventory | `4cdac208a40e6a5d0132089f45f7f285bce33abe0341a3d692b6697421ca6661` |

Next work remains Qwen verification affordability and safe model-epoch admission
for whole-phone Llama, plus the unproven concurrent-compute/peak-memory conditions.
This test does not authorize or trigger a longer trace.
