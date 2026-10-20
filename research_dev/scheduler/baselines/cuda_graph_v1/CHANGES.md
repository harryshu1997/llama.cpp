# Changes and validation for this experiment

No commit, push, PR, native phone rebuild, long trace or scheduler redesign was
performed. Pre-existing dirty changes and completed physical artifacts remain
untouched. These are the task-owned changes, not a list of every dirty file.

Paths below are relative to `research_dev/scheduler/`.

| File | Scoped change |
| --- | --- |
| `config.py` | Explicit immutable fixed-residency assignment and startup-accounting configuration |
| `scheduler.py` | Fixed configuration state; existing live-VRAM parent selection accepts explicit runtime-default launch settings |
| `_unified/phone_residency.py` | Reuse existing progressive preload with exact artifact/session assignments; disable only policy replacement in fixed mode; expose READY subsets normally |
| `adapters/llama_server.py` | Explicit `runtime-defaults` launch contract, preserving graph-mode identity and separate FFN environment |
| `adapters/heterogeneous_rig.py` | Optional unused-phone completion for fixed references; exact partial session-add excludes the retained aggregate router from whole-endpoint eviction |
| `adapters/offline_phone_residency.py` | Use actual snapshot capture time for the next progressive stage, without extending snapshot expiry |
| `campaigns/burstgpt/desktop_parent_calibration.py` | Qualify live-VRAM parents with explicit context, batching and native-default launch mode |
| `campaigns/burstgpt/launch.py` | Forward fixed assignment/startup configuration and named small replay through canonical launch/preflight |
| `campaigns/burstgpt/preflight.py` | Validate actual named replay and FFN-only transport capabilities; retain identity, safety and capacity checks |
| `campaigns/burstgpt/runner.py` | Invoke scheduler-owned preload; paid preparation boundary; capture phone evidence before cleanup; persist fixed state, events, power assumptions and initial evidence hashes |
| `campaigns/burstgpt/compare_ab.py` | Frozen-arm identity checking; actual CUDA capture/replay; native acknowledged layer/column/token coverage; power sensitivity and six-arm comparison |
| `tests/test_fixed_phone_residency.py` | Exact assignment, CPU-only coverage, no silent shrink, progressive publication and no queue-policy replacement |
| `tests/test_cuda_reference.py` | CUDA trace, frozen identity, default launch/control metadata and canonical evidence ordering |
| `tests/test_cuda_graph_mode.py` | Default launch and per-binary qualification identity regressions |
| `tests/test_offline_phone_residency.py` | Late snapshot capture and genuinely expired observations |
| `tests/test_physical_residency.py` | Unmocked mixed-artifact empty-session add through the real stop-set checks |
| `tests/test_matched_comparison.py` | Executed calibrated desktop parent, including mismatch rejection |

`research_dev/talks.md` records progress. This baseline directory contains only
reporting scripts, documentation and immutable evidence. Campaign callbacks
provide transport and persistence; they do not choose sessions or fractions.

## Focused validation

The final session-add correction passed 74 focused tests across physical
residency, session COW and fixed residency in 16.617 seconds. The two full
decision replay goldens then passed in 71.670 seconds, unchanged:

| Replay | Canonical SHA-256 |
| --- | --- |
| v3, 16 requests | `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d` |
| v8, 24 requests | `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d` |

Earlier scoped checks passed: 45 offline/fixed/reference/comparison tests,
33 reference/replay-input tests, 19 executed-parent comparison tests, 44
reference/replay-input/matched tests, and 21 canonical-evidence comparison
tests. These overlap and must not be summed into a claimed full-suite count.
One attempted module name (`test_burstgpt_runner`) did not exist; the corrected
`test_burstgpt_replay` invocation passed. No complete harness was rerun.

## Preserved failed attempts and causes

1. Initial upstream configure/build could not find CMake on noninteractive SSH
   PATH. The next link attempt could not resolve CUDA 13 runtime/cuBLAS shared
   libraries; the final link uses their installed paths in its subprocess
   library search path. Build logs are retained.
2. Initial preflight assumed a whole-phone endpoint in an FFN-only portfolio
   and did not propagate the named three-request replay. The corrected
   preflight passes without inventing a whole-model phone service.
3. Bounded graph check attempt 1 published the first session, then rejected a
   stale time argument preceding the new snapshot's capture. The adapter fix
   carries the actual capture time; expiry checks remain strict.
4. Bounded attempt 2 completed inference but exported live phone events after
   cleanup. Evidence is now captured before cleanup, then persisted afterward.
5. First upstream reference claimed unused FFN transport qualification for a
   binary that has no FFN path. The upstream desktop control now omits that
   claim; phone execution is forbidden and artifact preflight remains enabled.
6. Source-v6 `fixed-ggq` added Gemma to an empty session behind a retained Qwen
   router. Whole-endpoint exclusive eviction incorrectly ran before filtering
   that authorized partial router. The focused fix affects only exact partial
   session adds; mismatched sessions and invalid evictions still fail closed.

The last defect required a new common source freeze. All six final references
use source v7; four completed source-v6 arms remain preserved separately and
are not used in the final comparison. The final mixed retry proves the real
Gemma/Qwen shard combination loads and executes without whole-router eviction.

The reference checker also corrected two reporting-only errors: comparing the
executed calibrated parent rather than the earlier candidate parent, and
canonicalizing the order of energy evidence IDs without dropping IDs. The
ordering erratum is decoded in `provenance/SPECIFICATION_ERRATA.md`.

## Work deliberately not changed

Adaptive fractions, incumbent/challenger logic, measured rejection, session
replacement/rollback, graph-mode behavior and helper lease/proof protection
were not redesigned. A poor fixed-layout result is retained as measured, not
repaired by forcing assistance. This task does not establish an adaptive
residency advantage over the best fixed arm.

Concurrent local runner-extraction edits appeared after source v7 was frozen.
They were not overwritten or deployed. Tests and physical results in this
report describe the archived source v7, not those subsequent local edits.
