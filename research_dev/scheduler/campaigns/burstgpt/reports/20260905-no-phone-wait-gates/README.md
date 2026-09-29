# Short no-phone-wait gates, 2026-09-05

The v6 Qwen reuse gate, matched Gemma positive screen, and final v8
publication-clock validation pass. All 940 scheduler and 316 historical
tests pass, and both replay goldens are unchanged.
Earlier raw results and the earlier short-FFN report remain unchanged.
No long trace was run.

Full identities, timestamps, coverage numerators, and measurements:
[COMPARISON.json](COMPARISON.json). The chronological failures and repairs
are recorded in [talks.md](../../../../../talks.md).

## Useful assistance

| Request | Phone calls | Any-phone coverage | Fraction-weighted coverage |
| --- | ---: | ---: | ---: |
| Qwen 43, 341 output tokens | 5,886 | 96.18% | 91.62% |
| Qwen 67, 133 output tokens | 2,160 | 90.91% | 90.91% |
| Gemma, 837 output tokens | 19,352 | 98.09% | 95.84% |

All complete with accepted semantic output. Coverage starts at eligible
decode boundary 1, with denominators 340/132/836. Actual request-scoped
native calls and executed column widths give full-fraction-equivalent
tokens 311.5/120/801.25. Fractions apply to the resident CPU FFN subset,
not the whole model or GPU-resident operators. The consistently present
first layer anchors per-token counting without counting each layer again.

Qwen 43 finishes before 67 arrives. Both use the same artifact, exact
desktop parent, shard geometry, operator plans, and session generations
1/1/1. Request 67 gets a new ticket and disjoint helper leases: 31-44
versus 11-24. Its first positive acknowledgement is at token 3, not 51.
Qwen session calls are 1,962 each for 43 and 720 each for 67. One shard set
loads, once per session, with zero intervening reloads or generation changes.
These diagnostic learning runs demonstrate high-first control and early
reuse, not qualification promotion or an exploration-free winner cache.

Gemma's exact live-VRAM parent has 22 GPU layers, first GPU layer 26.
Stored/assisted layers 0-23 are all CPU-resident; layers 24-25 stay desktop.
The scheduler does not hardcode those layers. Qwen's stored/assisted subset
is 0-17. Actual worker evidence names each indexed FFN shard path, parent
and shard hashes, stored/executed masks, widths, bytes, and timestamps.
The existing worker and shard generator were reused without modification.

## Matched Gemma energy

The pair uses the same v16 source manifest, native binaries, artifacts,
catalog, exact CPU/GPU parent/operators, live-VRAM qualification, prompt,
seed, context, batch, parallelism, and 837-token requirement. The explicit
dormant FFN runtime is the only desktop parameter difference. The control
makes zero phone calls and performs no phone load. These are one-pair
diagnostic results, not a broad workload qualification.

| Measurement boundary, at 4.5 W assumed phone active power | Desktop | Assisted | Energy reduction |
| --- | ---: | ---: | ---: |
| Paid interval, including runtime preparation and termination | 49.115 kJ | 22.235 kJ | 54.73% |
| Matched post-READY decode suffix, 789 tokens | 45.003 kJ | 18.743 kJ | 58.35% |

Complete request execution takes 397.079 vs 316.743 s, 20.23% less time.
The assisted full execution window overlaps remaining preparation and is
therefore not labelled steady state. The post-READY comparison uses the
same native token-position interval, 46 through 835, anchored to each
HTTP first-token timestamp. It excludes prefill and terminal tail.
Native/controller alignment offsets range from -52 to 11,132 us.
It is a matched decode interval, not a separate full warm-request run.

| Assumed phone active power | Paid-interval reduction | Post-READY decode reduction |
| --- | ---: | ---: |
| 3 W | 55.23% | 59.31% |
| 4.5 W | 54.73% | 58.35% |
| 6 W | 54.23% | 57.39% |

CPU energy is physical RAPL package measurement and GPU energy is physical
NVML board-power integration. Phone power is assumed; idle power is
0.875 W in both arms. The paid metric uses the recorded activity union.
The decode comparison conservatively charges active phone power for the
entire assisted interval and idle power for the desktop interval.
This is not a wall-plug energy measurement.

Paid-interval measured CPU energy is 35.889 vs 9.785 kJ, GPU energy
12.865 vs 11.547 kJ, and nominal assumed phone energy 0.361 vs 0.903 kJ.
Peak total VRAM is 16,205,742,080 vs 16,220,422,144 bytes. The greeter's
3,178 MiB allocation was not stopped or altered. No memory limit,
safety factor, workspace, reserve, or qualification check was weakened.

The valid-window fraction screen is positive at all tested nonzero
fractions for 3/4.5/6 W. At 4.5 W it records 57.43 J/token at 0%,
49.08 at 25%, 40.17 at 50%, 35.84 at 75%, and 22.14 at 100%.
The 25/50% points each have only four tokens; 75% has eight, the baseline
six, and 100% has 753. These sparse points remain diagnostic, not qualified.

## Preparation and progressive use

| Model | HTP0 load-to-READY | HTP1 load-to-READY | HTP2 load-to-READY | First authorization to all physically READY |
| --- | ---: | ---: | ---: | ---: |
| Qwen | 10.885 s | 10.629 s | 10.780 s | 44.081 s |
| Gemma | 9.421 s | 12.608 s | 12.094 s | 43.602 s |

This table uses the phone monotonic clock. Each interval begins at that
session's LOAD_AUTHORIZED and ends at its READY event. Staged file reads,
HTP initialization, upload, endpoint setup, verification, and publication
are separately retained in COMPARISON.json and raw receipts.
These are storage-to-residency measurements, not USB file-transfer timings.

Qwen scheduler logical READY times are 21.192/33.697/47.330 s.
Its desktop loading interval is 2.084-43.548 s; the phone preparation
interval spans 2.070-47.330 s, proving concurrent preparation.
Qwen attaches a two-session subset before all three are READY, but its
first positive call comes afterward; this run alone does not prove serving
during loading.

Gemma scheduler logical READY times are 21.874/35.206/48.422 s.
Desktop loading is 3.607-13.964 s, overlapping the first phone transition
3.610-21.874 s. Desktop execution starts at 21.418 s, before first READY.
HTP0 records calls 1-16 while HTP1 loads. During HTP2 loading, HTP0 records
48-240 and HTP1 records 1-192. These direct call markers share the phone
clock with the load events. Each session stays generation 1 and loads once.
Terminal totals are 6,560/6,512/6,280 calls.

The three shard files total 9,625,933,376 bytes for Qwen and 8,493,472,480
bytes for Gemma (including GGUF metadata). Their FFN tensor totals are
9,625,927,680 and 8,493,465,600 bytes. Index, parent, physical path, stored
coverage and executed coverage are checked exactly; shard mode does not
fall back to the parent GGUF.

The first-through-last Gemma preparation interval is 44.812 s and brackets
3.476 kJ fleet energy at 4.5 W, including concurrent desktop work.
It is not an isolated incremental phone-load measurement and must not be
added again to the paid total. Conservatively charging that entire interval
against the matched post-READY decode saving gives
ceil(3475.90 / 26260.69) = 1 comparable long decode unit, also 1 at 3/6 W.
This is an estimate, not a measured reuse sweep or a cold-install claim.
Generation/transfer of staged files, preflight, and common warm CPU-service
startup are excluded and unmeasured in this screen. One measured long
request already has a positive paid-interval sign within the stated boundary.

## Final timing validation

The v6 Gemma initial PROPOSED-to-PREPARING logical timestamp difference
is 1.378 s. A decision-only profile found those events reused the snapshot
timestamp: submission took 1.348 s, and actual proposal emission occurred
1.044 s into it. Only 0.304 s remained after emission. Logical timestamps
are not being rewritten or used to assert the physical <=1 s target.

The first clock validation, v7 on deployment v17, completed but missed the
timing gate: initial clock registration came after submission, and the next
stage took 1.170923 s. Registration now happens in the canonical coordinator
before submission. Immutable tensor indexing is reused instead of rebuilding
the index on every lookup; the read-only decision replay drops from 1.348 to
0.912 s without changing its route. Logical timestamps are preserved.

Final v8 on deployment v18 records all publication timestamps:

| Session added | Actual proposal-to-preparation | Actual READY publication after paid start | Physical load-to-READY |
| --- | ---: | ---: | ---: |
| HTP0 | 0.201 s | 20.513 s | 9.338 s |
| HTP1 | 0.211 s | 31.496 s | 10.325 s |
| HTP2 | 0.464 s | 44.045 s | 11.749 s |

All three meet the <=1 s preparation-start target. Desktop loading is
2.492-12.859 s and phone preparation begins at 2.489 s. Desktop execution
begins at 19.121 s, before first READY. The first READY subset is attached
before all sessions are ready; HTP0 and HTP1 each record calls 1-208 during
HTP2 loading. No serving is claimed during HTP1 loading in this final run;
the earlier v6 run supplies that evidence. First positive control is
acknowledged at decode boundary 11 (32.240 s), before HTP2 is READY.
The request completes all 837 tokens with 19,456 calls, 98.09% any-phone
coverage and 95.87% fraction-weighted coverage. One load per shard and
generations 1/1/1 are verified against terminal session proofs.

The final physical request takes 348.944 s, versus 316.743 s in the v6
assisted screen. It is a timing/identity validation on a different frozen
source; no energy saving or performance improvement is inferred from that
cross-version difference. The matched energy comparison remains v16 only.
Request-helper event observed_at_us values are logical snapshot times;
they must not be confused with layout published_at_us or physical policy
acknowledgement timestamps. Decoded audits retain those distinctions.

## Files changed in this bounded repair

Production:

- tools/server/server.cpp
- examples/layersplit/ffn-split-resident-router.cpp
- research_dev/scheduler/_internal/model_placement_controller.py
- research_dev/scheduler/_unified/phone_residency.py
- research_dev/scheduler/_unified/automated_selection.py
- research_dev/scheduler/_unified/helper_envelopes.py
- research_dev/scheduler/_unified/adaptive_decode_control.py
- research_dev/scheduler/adapters/runtime.py
- research_dev/scheduler/adapters/http_backend.py
- research_dev/scheduler/adapters/native/direct_phone_ffn_session.sh
- research_dev/scheduler/adapters/coordinator.py
- research_dev/scheduler/_internal/model_manifest.py

Tests:

- research_dev/scheduler/tests/test_offline_phone_residency.py
- research_dev/scheduler/tests/test_session_cow_transaction.py
- research_dev/scheduler/tests/test_llama_server_adapter.py
- research_dev/scheduler/tests/test_resident_router_subset.py
- research_dev/scheduler/tests/native/resident_router_subset.cpp
- research_dev/scheduler/tests/test_adaptive_runtime.py
- research_dev/scheduler/tests/test_phone_power_probe.py
- research_dev/scheduler/tests/test_model_placement_controller.py
- research_dev/scheduler/tests/test_physical_adapter.py
- research_dev/scheduler/tests/test_arrival_coordinator.py
- research_dev/scheduler/tests/test_gguf_cost.py
- research_dev/scheduler/tests/test_multi_session_phone.py
- research_dev/spikes/s42_general_energy_scheduler_v1/tests/test_full_fp16_burstgpt.py

Documentation: research_dev/talks.md and this new report directory.
No files were deleted in this repair. No parallel helper subsystem or
new compatibility layer was added. Unrelated dirty-worktree changes and
old results are preserved. Worker/shard-format/generator sources were
not changed. All generated configurations, commands, native build sources,
failed attempts, and raw artifacts remain under their versioned desktop paths.

## Validation and limitations

Focused validation: 137 tests pass after the watcher/fixture repair, then
15 coordinator/model/replay tests pass after final clock/index fixes.
These overlap and are not summed as unique tests.
Complete harness: 1,256 tests pass across all 82 existing modules (940
scheduler tests in 50 modules, 316 historical tests in 32 modules).
The first traversal stopped at an incomplete controller test double;
resumption stopped at an incomplete historical rig test double. Their
missing fields were supplied without changing assertions or runtime code,
then the existing harness manifest resumed at each failed module. Already
passed modules were not rerun. See [TESTS.json](TESTS.json) and the three
FULL_TESTS logs for exact commands and output.

Only those two fixture repairs followed the final physical gate. Comparison
against all 208 files in the frozen v18 source manifest finds one local
change: test_multi_session_phone.py. The historical fixture is outside that
manifest. All tested production files still match v18 exactly. The frozen
desktop deployment and physical source manifest are preserved, not rewritten
to incorporate later test edits.
Replay identities remain:

- v3: sha256:f78d2b2c37a3880a523eba4f5315ada0207678c841d633229782bfa3a05c1829
- v8: sha256:965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d

Passing v6 artifacts have zero rejected helpers, control failures, request
recoveries, stale proofs, and USB reset recoveries. They prove progressive
addition and cross-request reuse, not a new reverse-replacement/rollback
gate or the stricter 2x retained-session inter-call-gap bound. Logged phone
calls are sampled every 16 calls, so they do not establish a per-call maximum
interruption bound. Existing lifecycle/rollback paths remain covered by tests.
The thermal maximum's pre-existing all-sensors-unreadable case remains
documented in talks.md and was not changed in this bounded work.

No 24-/84-request trace, GDM stop, unrelated process kill, commit, or push.

## Physical artifacts

All paths are on the desktop under /home/zhihao:

- s42-no-phone-wait-20260905-v6-qwen-gate/run/RESULT.json:
  sha256:83045b71124c53fa55b16ae208d79642ce669a499498d7dcb8ce874842e2dd77
- s42-no-phone-wait-20260905-v6-gemma-desktop-r1/run/RESULT.json:
  sha256:f7e7025ea10f310eb04822952e6b69e6e0a10fd608ab8e081fc620eb2dc88de1
- s42-no-phone-wait-20260905-v6-gemma-assisted-r1/run/RESULT.json:
  sha256:791d5a76d4d7c6f7cb61ab2b5146464a5339fd2ac5613855b7f7f09546b94d65
- s42-no-phone-wait-20260905-v6-gemma-calibration/run/DESKTOP_PARENT_CALIBRATION.json:
  sha256:a504b2314c2bb2c2b4edddc5bffa286817b27fadeb9740b06ada25cf5480c318

v6 SOURCE_MANIFEST.json file SHA:
f748100f42a3b25df3cd896b2a5937a362d5c745549ab72eb1f7f4ac5b4e543e.
Its canonical identity is
sha256:f85f73d4b211a73d7114b0a352f7541a97ff62f287855fedbedc933f3e8abb45.
Inputs and raw commands: s42-no-phone-wait-20260905-v6-inputs.
Intermediate timing inputs: s42-no-phone-wait-20260905-v7-inputs.
Final timing inputs and decoded audits: s42-no-phone-wait-20260905-v8-inputs.
Final result: s42-no-phone-wait-20260905-v8-gemma-timing-gate/run/RESULT.json,
SHA 8557329dea53b4ba582bca2e3f23397099708a46391b70825ff62fbe863ffb27.
Final source manifest canonical identity:
sha256:96468cd5754a2cda562b54d9fd1a577d044b87ee47ea17bdcddced823deb05bd;
file SHA 29b0df0abf289870f159f64553cd11b58c3000bc4f0553b94e5f3f71c15f6513.
The uncommitted source HEAD remains 5f89a2d9d33be547a1bdef5fd0f504a279c50800;
the manifests, not HEAD alone, identify the tested dirty trees.
