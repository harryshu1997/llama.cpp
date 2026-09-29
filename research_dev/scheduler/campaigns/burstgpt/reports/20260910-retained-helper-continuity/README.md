# Retained-helper continuity: implemented; physical preflight blocked

Latest follow-up: [atomic refresh and queued-control invalidation](REFRESH_RACE_FIXES.md)
pass 203 focused tests, including both unchanged replay goldens. These corrections
are local only and are not part of the preserved deployment or physical attempt.

Follow-up: [deferred-control and zero-fraction review corrections](REVIEW_FIXES.md)
pass 190 focused tests. Those later corrections are local only; the deployment
and physical artifacts below preserve the original attempt.

The scoped fix is implemented and deployed. **No adaptive experiment has run**:
another project's Gemma-26B benchmark acquired the GPU during preflight. No
baseline was rerun, no long trace started, and no existing process was stopped.

## Fix and evidence boundaries

READY publication used to replace candidate/component IDs, restart LEARNING,
and make the incumbent and its old policy-hash measurements inaccessible. This
could leave a still-compatible retained helper at zero after its exploration
budget was mostly spent.

The refreshed helper is now checked against the existing per-request envelope
history. Evidence reuse requires exact artifact, desktop parent, dtype,
executor/endpoint/protocol, participant identity, batch contract, shard geometry,
resident bytes, shard operator plan and per-session generation. Only policies
whose actual layers, columns and fraction match are aliased. Their original
measurement receipts and hashes are not rewritten. A new envelope still needs
physical acknowledgement; execution authorization and lease validation are
unchanged. Request-wide probe counters, attempts and historical energy remain.

Temporary drain-mask windows remain excluded from qualification. The regression
for the previous Gemma tail reconstructs the smaller-set -> expanded-set ->
retained-set sequence, then refreshes at token 251 with 41 tokens left. It reuses
earlier valid measurements of that exact retained set, not measurements of the
larger set or of the load-interference window. The synthetic request stays
assisted through completion without another probe. This is software evidence,
not a new physical coverage or savings measurement.

Real execution-shape changes retain the bounded revalidation path. Missing
compatibility proof does not manufacture qualification. Previously measured
rejection is retained for an equivalent candidate, and physical unavailability
still disables use. No fraction, placement selection, energy threshold, memory
limit, session lifecycle or native wire format was changed.

## Validation

180 focused tests passed in 11.540 s: sustained assistance, adaptive decode,
adaptive runtime, late-helper energy policy and session COW. Both replay tests
passed in 72.878 s, including byte-identical repeated replay output.

| Oracle | Unchanged SHA-256 |
| --- | --- |
| v3 | ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d |
| v8 | 965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d |

The complete harness was not rerun. `git diff --check` passes. All eight
production files and the focused test file match the staged source-manifest
hashes. Four new regressions cover bookkeeping-only rebind, retained-set tail
continuity, incompatible shape/unverified generation, and exact physical
identity/parent validation.

Changed production files, relative to `research_dev/scheduler/`:

- `_internal/adaptive_decode.py`: forwards the optional compatibility proof;
  checkpoint copies own their evidence aliases.
- `_internal/adaptive_decode_state.py`: request-local evidence alias/context state.
- `_internal/adaptive_decode_ops/helpers.py`: checked continuity and fresh ack.
- `_internal/adaptive_decode_ops/budgeting.py`: stable compatible context/attempt keys.
- `_internal/adaptive_decode_ops/bounds.py`: reads applicable immutable receipts.
- `_internal/adaptive_decode_ops/promotion.py`: replaces incompatible incumbents safely.
- `_unified/helper_envelopes_ops/refresh.py`: exact retained-session comparison;
  preserves the previous context until refresh completes.
- `_unified/helper_envelopes_ops/materialization.py`: uses existing envelope
  history to pass the checked per-plan retained masks and exports that decision.

Also changed: `tests/test_sustained_assistance.py`, `research_dev/talks.md`, and
this report's command-persistence/analysis files. Other dirty-worktree changes
were preserved. Nothing committed or pushed.

## Physical blocker and preserved attempt

At the first check, VRAM was 3,178 MiB used / 12,770 MiB free. During preflight,
the independent process below started a native Gemma-26B server, using another
11,636 MiB. The GPU then had 14,822 MiB used / **1,126 MiB free**.

- Benchmark owner PID 2936721: `moe-resident-routing-4060ti-op15`,
  `src.prepare_recent_moe_tests baseline --models gemma4_26b_q8 --phase performance`.
- Server PID 2938491: native Gemma-4-26B-A4B Q8_0, port 18186, parallel 4.
- Preflight blocker: qualified Qwen desktop placement
  `sha256:071a9a0b8112e5701e9055a4a254104e1dd31f04a850795500c098fadc542aad`
  was not admitted; its candidate reports `MEMORY_CAPACITY` and
  `ROUTE_NOT_QUALIFIED`. The hardware/executable and phone telemetry checks pass.
- `physical_inference_executed=false`. No phone shards were loaded by this attempt.

The separate benchmark and GDM remain untouched. The frozen placements cannot
be changed to fit around it without changing this experiment. A fresh preflight
in a new artifact directory is needed once the GPU is available.

Desktop attempt: `/mnt/storage/s42-retained-helper-20260910-v1/`.
Deployment: `/mnt/storage/s42-retained-helper-20260910-v1-deploy/`.
The failed preflight, commands, inputs and source manifest are copied under
[physical/](physical/). In particular:

- [Failed preflight](physical/preflight/PHYSICAL_PREFLIGHT.json)
- [Resolved run command](physical/RUN_COMMAND.json)
- [Source manifest](physical/inputs/SOURCE_MANIFEST.json)
- [Comparison intent](physical/REFERENCE_COMPATIBILITY_INTENT.json)

## Frozen references, not new adaptive results

Only final `references-source-v7` results were read. The frozen COMPARISON.json
still hashes to `83054fdc4179b4c64939a1ed07ce0891e833a3f642218464c47e62eaf27ba022`.

| Arm | Duration (s) | Fleet energy, 4.5 W phone (kJ) |
| --- | ---: | ---: |
| Clean upstream desktop | 371.20 | 31.866 |
| Matched modified desktop | 355.14 | 31.699 |
| Fixed GGG | 312.20 | 22.492 |
| Fixed GGQ | 315.72 | 25.337 |
| Fixed GQQ | 331.50 | 31.963 |
| Fixed QQQ | 384.84 | 33.325 |
| New adaptive | Not run | Not measured |

GGG remains the best measured fixed reference. There is no new measured saving,
latency, fraction-weighted coverage or retained-session continuity result to
compare with it yet. The 25% adaptive savings target is untested.

The staged adaptive arm uses the unchanged dev3 requests/arrivals, frozen
catalog and initial evidence, graph-enabled binaries, qualified desktop
placements and preparation-inclusive boundaries. It removes only the fixed
assignment and uses ordinary energy-aware scheduling. The source manifest
differs through the completed cleanup and this shared adaptive-controller fix.
Local working-tree HEAD is `5f89a2d9d33be547a1bdef5fd0f504a279c50800`; the fresh
desktop deployment starts from the existing research checkout
`99449bafade0b2c15de4410feda832035c2f2d83` with the tested scheduler files overlaid.

Any eventual comparison must be **historical-reference**, not fresh matched A/B
or isolated proof of dynamic-placement superiority. The matched comparison
validator is unchanged; `analyze.py` records decoded identity differences and
checks unchanged workload, binaries, placements, phone identity and accounting
separately. CPU/GPU energy remains measured, with phone sensitivity at 3/4.5/6 W
and the same 0.875 W idle assumption. No completed reference is overwritten.
