# Short FFN-shard gates, 2026-09-05

Usage, semantic output, and the matched Gemma energy screen pass. The complete
zero-wait/progressive-publication acceptance gate does not pass yet. No long
trace, commit, push, worker rebuild, GDM change, or unrelated process shutdown
was performed. Raw artifacts remain unchanged on the desktop.

Machine-readable measurements, identities, per-session weight sources, and
remaining gaps are in [COMPARISON.json](COMPARISON.json).

## Useful phone assistance

| Request | Output tokens | Phone calls | Any-phone coverage | Fraction-weighted coverage |
| --- | ---: | ---: | ---: | ---: |
| Qwen 43 | 341 | 5,886 | 96.18% | 91.62% |
| Qwen 67 | 133 | 1,836 | 77.27% | 72.35% |
| Gemma 3 | 837 | 19,752 | 98.44% | 96.59% |

All three requests completed with accepted semantic output. Coverage starts
at eligible decode boundary 1: denominators are 340, 132, and 836 tokens.
The numerator comes from native, request-scoped call records and actual column
widths, including warmup and terminal work. Full-fraction-equivalent tokens
are 311.5, 95.5, and 807.5. Fractions refer to the resident FFN subset, not the
whole model. Qwen uses layers 0-17; Gemma uses layers 0-23. Every assisted FFN
is CPU-resident under the exact corresponding desktop parent.

Qwen 43 completed before 67 arrived. The second request reused generation
1/1/1 with no load or restart. Active helper leases are disjoint: 11-24 for
43 and 31-44 for 67. Its first 100% acknowledgement was at token 3. Each
Qwen session made 1,962 calls for 43 and 612 for 67. Each Gemma session made
6,584 calls. Both model runs loaded one shard set, exactly once per session.
There were no fallback recoveries, rejected helpers, stale execution proofs,
or USB resets in the passing artifacts. Final USB restoration was 5,000 Mbps.

The existing scheduler-owned high-first order exercised 100/75/50/25 on Qwen
43 and Gemma, then used 100% from token 62 onward. Qwen 67 used 100/75 and
temporarily returned to desktop when a bid was negative. These are diagnostic
learning outcomes, not qualification promotion or proof that a qualified
winner cache was reused without exploration.

## Matched Gemma energy screen

One identical 837-token request per arm, on the same frozen v10 deployment.
The model artifact, prompt, seed, output requirement, binary hashes, catalog,
CPU/GPU operator placements, desktop parameters, live-VRAM parent
qualification, and energy boundary match. The only desktop adapter parameter
difference is the explicit dormant FFN runtime. The parent uses 22 GPU layers,
with first GPU layer 26, not 24. The desktop control made zero phone calls,
loaded zero phone sessions, and emitted zero helper events.

| Arm | Warm execution | Warm fleet energy | Preparation-inclusive fleet energy |
| --- | ---: | ---: | ---: |
| Desktop control | 395.47 s | 49.015 kJ | 49.940 kJ |
| FFN-assisted | 315.56 s | 21.354 kJ | 23.576 kJ |

At nominal 4.5 W phone active power, warm-execution energy is 56.43% lower
and execution latency is 20.21% lower. CPU package energy is 36.038 vs
9.068 kJ; GPU board energy is 12.630 vs 11.549 kJ. These are physical
RAPL/NVML measurements. Phone energy is assumed, with identical 0.875 W idle
treatment in both arms; this is not a whole-machine wall-plug measurement.

| Assumed phone active power | Warm energy saving | Preparation-inclusive saving |
| --- | ---: | ---: |
| 3 W | 56.82% | 53.29% |
| 4.5 W | 56.43% | 52.79% |
| 6 W | 56.04% | 52.30% |

The valid-window diagnostic screen has positive nominal energy differences
at all four nonzero fractions. Baseline is 59.54 J/token; 25/50/75/100% are
51.44/41.81/32.13/23.59 J/token. The 25/50/75% points each contain one valid
four-token window; the baseline has five valid tokens, and 100% has 781.
These sparse probe points must not be promoted to qualified routes. The
separate matched whole-request comparison above is the stronger energy result.

The paid boundary includes request replay, runtime load/verification, and
executor/phone termination. It excludes staged artifact generation/transfer,
preflight, and common CPU warm-service startup. Those excluded costs were
not measured as part of this screen; this is not a complete cold-install
energy claim. The warm boundary begins after both desktop weights and phone
shards are READY and includes the request's prefill and decoding.

Gemma's phone-preparation interval contains 1.344 kJ of fleet energy at
4.5 W, including 0.168 kJ of assumed phone energy. That interval overlaps
desktop preparation, so its fleet energy is not separately additive to the
desktop load receipt. Using the full interval as a conservative preparation
bracket gives `ceil(1344 / 27660) = 1` request for this exact long workload;
the same calculation yields 1 at 3 and 6 W. The observed incremental
non-execution cost of the matched pair is 1.296 kJ. This is a single-pair
diagnostic amortization estimate, not a reuse sweep or a claim for shorter
requests. No Qwen energy saving is claimed against an older baseline.

## Physical preparation

Times below use phone monotonic phase timestamps, starting at the first
LOAD_AUTHORIZED event. The files were staged before the run; storage reads
are not host-to-phone USB transfers.

| Model | HTP0 load-to-READY | HTP1 load-to-READY | HTP2 load-to-READY | First / all physically READY |
| --- | ---: | ---: | ---: | ---: |
| Qwen | 11.221 s | 11.147 s | 12.366 s | 11.221 / 34.734 s |
| Gemma | 9.391 s | 9.806 s | 10.034 s | 9.391 / 29.231 s |

Qwen file bytes are 3,208,644,448 / 3,208,644,448 / 3,208,644,480; tensor
residency is 3,208,642,560 bytes per session. Gemma file bytes are
2,831,157,472 / 2,831,157,504 / 2,831,157,504, storing disjoint masks
0-7 / 8-15 / 16-23 at width 15,360. Each physical receipt records
`weight_source=ffn_shard`, path, file/index/parent hashes, stored/executed
masks, widths, byte counts, generation, load, verification, and READY times.

Qwen PROPOSED-to-PREPARING is 0.862034 s; desktop preparation spans
3.027522-74.168895 s and phone preparation 3.032684-46.544661 s.
Gemma PROPOSED-to-PREPARING is 1.380600 s, which misses the 1 s target.
Its desktop preparation spans 3.278802-43.486898 s and phone preparation
3.266046-40.620464 s. Attachment is recorded at 43.674375 s and request
execution begins at 44.043895 s. These intervals overlap, but overlapping
transition receipts alone do not establish zero blocking inside the server.

Peak total GPU use was 16,205,742,080 bytes for control and
16,212,033,536 bytes for assisted. Initial use was 3,332,374,528 bytes in
both arms; peak increments were 12,873,367,552 and 12,879,659,008 bytes.
The existing 3,178 MiB GNOME allocation was not altered.

## Remaining blockers: do not start a trace

1. `tools/server/server.cpp` calls `client_->connect(error)` synchronously
   during FFN initialization, before model loading, even for runtime control.
   `examples/layersplit/ffn-split-usb-client.cpp` polls device discovery up to
   600 times at 50 ms. The Gemma server logs show approximately 30 s before
   model loading and `connection=deferred`. Deferred mode is reached only
   after the blocking attempt. The fix needs a scoped native desktop-server
   startup change and a rebuilt/requalified binary, not a smaller inference
   timeout, a runner route override, or weakened proof validation. No native
   source or binary was modified in this pass.
2. The normal initial three-session path publishes scheduler VERIFIED/READY
   events together: 46.544661 s for Qwen and 40.620464 s for Gemma. Physical
   per-session readiness is earlier and distinct. These runs do not establish
   first-session-only attachment while the other sessions load.
3. Gemma missed the 1 s proposal-to-preparation target. Independent replacement
   and retained-session serving were not rerun in these two short screens.

The physical RESULT files say PASS for their execution/terminal checks. That
does not mean the larger zero-wait/progressive-publication acceptance gate is
complete. The complete harness remains deferred under the requested gate
ordering; no additional physical workload is running.

## Code and validation

Latest focused validation: 19 tests passed, comprising 16 physical-adapter
tests plus desktop-baseline exclusion, initial learning preparation, and the
determinism test containing both replays. Before the baseline-only repair,
the v9 focused helper/adaptive/session/replay set passed 166 tests and the
physical-residency set passed 24 tests. These are overlapping runs, not an
additive unique-test count. Both golden payloads are unchanged:

- v3: `sha256:f78d2b2c37a3880a523eba4f5315ada0207678c841d633229782bfa3a05c1829`
- v8: `sha256:965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

Compared with the paused v4 deployment, exact changed source files are:

- `_internal/adaptive_decode.py`
- `_internal/phone_shards.py`
- `_internal/route_generation/envelopes.py`
- `_unified/adaptive_decode_control.py`
- `_unified/helper_envelopes.py`
- `_unified/helper_preparation.py`
- `_unified/phone_residency.py`
- `adapters/heterogeneous_rig.py`
- `adapters/http_backend.py`
- `campaigns/burstgpt/offline_residency_gate.py`
- `campaigns/burstgpt/runner.py`
- `tests/test_adaptive_decode.py`
- `tests/test_llama_server_adapter.py`
- `tests/test_offline_phone_residency.py`
- `tests/test_physical_residency.py`
- `tests/test_session_cow_transaction.py`

Also updated `research_dev/talks.md` and added this report and its JSON data.
The v9-to-v10 change is limited to helper-envelope authorization and the
desktop-baseline regression. It prevents desktop-baseline tickets from
starting phone preparation; terminal proof validation remains strict.

The bounded inventory classified 752 dirty/untracked paths: 536 outside the
active path, 174 scheduler production/configuration/documentation paths,
40 test paths, and two pre-existing native deletions. The test category also
contains one pre-existing deletion. No generated-cache candidates appeared
in that inventory. No user files, physical artifacts, or prior deletions were
removed or reverted. Duplicate storage-coverage projection was consolidated
into the existing phone-shard helper, and repeated tail-boundary arithmetic
uses one adapter method; no parallel scheduler subsystem was added.

## Artifact identities

Qwen result:
`/home/zhihao/s42-ffn-ready-reuse-20260905-v9-run/run/RESULT.json`

SHA-256: `43c6977d207a53884f45ce6a7dff6ec35cd5f48215b2790af14c005bc9f06a5a`

Gemma desktop result:
`/home/zhihao/s42-ffn-gemma-positive-screen-20260905-v2-desktop/run/RESULT.json`

SHA-256: `d1feccf333b336d024ab3567c5113a5a5d8bed63a8be500be8915e70e4704813`

Gemma assisted result:
`/home/zhihao/s42-ffn-gemma-positive-screen-20260905-v2-assisted/run/RESULT.json`

SHA-256: `45b6f57fcb3b920447f579f21a563c3f8226f274dd2b3ab7b1197b22eedda872`

Gemma shared inputs are under
`/home/zhihao/s42-ffn-gemma-positive-screen-20260905-v2-inputs/`.
Source-manifest file SHA-256:
`3497a759db268e781f48269efdc5e00913daee61fc4cd69a21c082c443eb5edb`.
The frozen v10 deployment contains all 207 verified source-manifest entries.
The two Gemma DIRECT_PHONE_PREFLIGHT files have identical SHA-256
`27a9e518443756fe56a07c0930faeae6fde583a8e52cc3d591a5f89da2c0647d`,
including matching worker/router/session binary hashes and FFN indexes.
Per-session shard and operator-plan hashes are in COMPARISON.json and the
original execution proofs. Each arm preserves its exact command manifest,
RUN_COMMAND.json, resolved configuration, journals, streams, and host samples.
