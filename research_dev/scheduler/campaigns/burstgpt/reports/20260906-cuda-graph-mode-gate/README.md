# Declared CUDA graph mode and bounded mixed-session gate

Status: graph-mode implementation, fresh controls, forward replacement,
and Gemma execution are validated. The complete bounded gate is still
FAIL. V5 records physical reverse loading and rollback but lacks some
interruption references; V6 stops before reverse loading on unavailable
phone telemetry. No gate or trace is currently running.

The existing CUDA graph-disable switch is now a declared desktop process
and qualification identity. No native code, binary, model, shard, GPU layer
placement, interruption threshold, or memory limit changed. Default mode
leaves `GGML_CUDA_DISABLE_GRAPHS` absent, including when the parent process
inherited value `0`. Disabled mode sets `1` only in its server subprocess.
Mode is never toggled during inference and is not part of `ffn_environment`.
Historical profiles with an omitted mode mean default; disabled profiles
have a distinct parent hash. Fresh calibration and comparison records
name the mode explicitly, and cross-mode comparisons fail identity checks.

## Fresh cold and hot controls

Both modes passed physical cold and hot semantic validation with the same
server, artifacts, requests, decoding, and placement operators. Qwen uses
GPU16, context 4096, parallel 4, batch 2048, ubatch 512, 341 output tokens.
Gemma uses GPU22, context 32768, parallel 2, batch 4096, ubatch 512, 837
output tokens. GPU headroom was not obtained by stopping GDM or another
process. CPU+GPU energy is physical RAPL package plus NVML board power.

| Hot control | Default | Graphs disabled |
| --- | ---: | ---: |
| Qwen median token latency | 613.581 ms | 613.206 ms |
| Qwen CPU+GPU energy | 26.284 kJ | 27.015 kJ |
| Gemma median token latency | 471.022 ms | 470.616 ms |
| Gemma CPU+GPU energy | 50.424 kJ | 49.833 kJ |
| Qwen capture/update log markers | 27 | 0 |
| Gemma capture/update log markers | 4 | 0 |

These are single-pair mode screens, not a fleet-saving or significance
claim. Markers include initial captures and recaptures, not exclusive
capture duration. Full cold/hot energy, peak VRAM, identities, and separate
model-load measurements are in [CALIBRATION_SUMMARY.json](CALIBRATION_SUMMARY.json).
Disabled Gemma's 263.091 s load included directly observed host storage
waiting; this is not attributed to graph-disabled token execution.

## Preserved attempts

All roots below are on the desktop under
`/home/zhihao/s42-cuda-graphs-disabled-20260906-vN-` with `inputs`, `deploy`,
and `gate/run` suffixes. Qualification roots use v2-default-hot,
v2-default-cold, v2-disabled-hot, and v2-disabled-cold.

- V1: bootstrap failed before inference on a missing `replace` import.
  Fixed with a public-entry regression; the failed artifact is preserved.
- V2: progressive preload, retained-service forward replacement, fraction
  sweep, and Gemma READY reuse passed. Online Qwen/Gemma made 3,312/6,416
  calls. The gate failed before reverse loading; an incomplete-phase
  assertion hid the underlying exception. It is not a rollback pass.
- V3: forward and Gemma again passed. New diagnostic persistence proved
  reverse was rejected before loading because the ticket contained an
  aggregate eviction for the retained Qwen pair as well as the exact
  changed Gemma session. The fault injection was never consumed.
- V4: an initial compiler-wide eviction check regressed READY-helper
  refresh. Qwen completed, but assistance returned to zero and the fraction
  sweep failed. This was an implementation regression, not a negative
  energy result. The failed run and its rejection events remain preserved.
- V5: the check now runs only at the authorized phone-preparation boundary.
  Compiler behavior and READY-template derivation are restored unchanged.
  Forward and Gemma passed. Reverse physically loaded Qwen generation 3;
  the injected verification failure restored Gemma generation 4. Retained
  Qwen sessions stayed generation 1 with one load each. The gate remains
  FAIL: adaptive 75%/50% classes had insufficient equivalent reference
  intervals outside the reverse/fault interval. The 100% classes passed.
  The final reverse retry was not reached. See [V5_GATE_AUDIT.json](V5_GATE_AUDIT.json).
- V6: the same bounded gate ran in a fresh directory. A measurement
  collection change lets only missing-reference results defer their final
  FAIL until after the reverse retry and terminal proof are persisted.
  Observed over-bound gaps and all identity/safety errors still stop
  immediately. The 30-reference and 2x requirements are unchanged; no
  missing-reference result can become PASS. Forward and Gemma passed, but
  reverse planning encountered unavailable phone telemetry. Its snapshot
  reports the conservative configured 10 GB fully occupied, zero available
  bytes, battery 0, and temperature 100 C; these are unknown-state safety
  defaults, not measured exhaustion or overheating. The scheduler retained
  the existing layout and raised `offline phone target is already resident`.
  No reverse load or injected fault occurred in V6. See
  [V6_GATE_AUDIT.json](V6_GATE_AUDIT.json).

An authorized preparation envelope now removes an unused phone aggregate
only if its artifact and bytes exactly match target sessions retained by
the assignment. Unknown artifacts or byte mismatches still fail closed.
The selected source session keeps its exact artifact, bytes, geometry,
operator plan, and epoch in the eviction. READY execution templates do not
consume historical transition metadata as a new load command. Physical
validation, rollback fencing, and assignment identity are unchanged. No
native or physical-adapter rewrite was made.

## Latest bounded measurements (v6)

Qwen and Gemma complete on the freshly qualified GPU16/GPU22 parents.
Both have no request recovery/fallback recorded. Gemma's load counts are
1/1/2 both before attachment and after execution: it reused the READY
FFN shard. Fractions cover 0/25/50/75/100%, with no fraction-triggered load.

| Online request | Phone calls | Weighted eligible coverage | Median token latency | CPU energy | GPU energy | Assumed phone energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen, 341 tokens | 3,726 | 75.0% | 558.998 ms | 12.850 kJ | 6.572 kJ | 0.336 kJ |
| Gemma, 837 tokens | 6,416 | 89.683% | 457.986 ms | 27.065 kJ | 12.974 kJ | 0.532 kJ |

Token medians are native per-server decoded intervals, mixing the tested
fractions. Physical CPU/GPU energy covers request execution; phone energy
uses 4.5 W active and 0.875 W idle. These gate/control rows have different
Python manifests, request seeds, and interval definitions. They are
absolute diagnostics, not a matched fleet-saving comparison. Coverage is
postprocessed by the unchanged canonical coverage function, not inferred
from call counts; see [COVERAGE.json](COVERAGE.json).

All gate server logs contain zero CUDA capture/update markers. Online
Qwen's full-width retained layer-11-to-layer-0 host submission gap is
354.629 ms median, 366.604 ms maximum (193 intervals). Its pooled RPC
median/p95/max are 10.137/14.127/22.107 ms; Gemma's are
7.940/11.775/30.861 ms. These RPC distributions mix fractions and are not
used as the equivalent-class gate reference.

The strict forward checker passes all 22 matched classes across retained
HTP0/HTP1. Worst class ratios are 1.433462x/1.369400x; largest next-token
intervals are 563.649/566.815 ms against same-layer medians around 561 ms.
Each retained session makes 114 calls during loading. The earlier
graph-enabled comparison had next-token reference medians around 555 ms
and maxima around 1,130 ms. Thus the lower maximum is not explained by
deliberately slowing the reference cadence. This remains diagnostic,
since the historical gate is not a same-revision A/B.

| Forward milestone | Seconds from online measurement epoch |
| --- | ---: |
| Drain requested | 59.766830 |
| Next safe decode boundary | 60.049404 |
| Reduced-mask control issued | 60.578649 |
| Applied acknowledgement and quiescence | 61.097252 |
| Replacement loading started | 61.588066 |
| Physical READY acknowledgement | 71.666914 |
| Scheduler READY publication | 71.698581 |

Drain-to-quiescence is 1.330422 s, distinct from 10.078848 s between host
load start and physical acknowledgement. The shortened seven-token
old-policy window retains energy/work accounting with
`measurement_eligible=false`; the configured minimum stays 32.

Cold session authorization-to-READY times are 12.417571, 15.247351, and
15.171066 s. First-session usability is 12.417571 s; all-session usability
is 51.768621 s from first authorization. HTP0 makes 150 calls while HTP1
loads; HTP0/HTP1 make 168/162 calls while HTP2 loads. All three initially
have generation 1 and one load each. Total resident weights are
9,625,927,680 bytes. Forward Gemma authorization-to-READY is 10.028748 s.

V5 separately records HTP2 reverse Qwen/gen3 loading in 11.400117 s and
Gemma/gen4 physical restoration in 9.515955 s. Retained HTP0/HTP1 make
138/140 calls during reverse loading and 108/108 during restoration.
Logical/physical equality and load-count assertions passed before the
missing-reference failure. A clean reverse retry and final terminal proof
are not yet certified by either new run.

## Remaining blockers

1. V6's reverse-plan phone-runtime observation is unavailable. The HTTP
   diagnostic endpoint timed out in a read-only follow-up; the existing
   ADB fallback subsequently produced three valid samples after cleanup.
   See [POST_GATE_TELEMETRY.json](POST_GATE_TELEMETRY.json). Those later
   samples do not retroactively authorize the rejected reverse command.
   The snapshot does not preserve whether its runtime observation was
   missing, stale, or errored. This probe-diagnostics gap is not repaired
   by changing memory limits or substituting a cached capacity.
2. V5's reverse/fault interval has too few outside-load samples for its
   50% and 75% call classes. All 100% classes pass; the missing classes
   remain insufficient. A future measurement must obtain 30 exact
   references without forcing fractions or changing the 2x bound.

No further physical retry was launched after V6. Raw artifacts and
command/source manifests are indexed in
[ARTIFACT_HASHES.json](ARTIFACT_HASHES.json): 320 individually hashed
top-level/stream/selected-snapshot files, plus recursive directory index
hashes covering all preserved run files. The actual paths and hashes,
including every prior FAIL, are retained there.

## Absolute forward measurements from v2

The unchanged `s42-retained-session-call-gap-v2` checker compares equivalent
call classes, fractions, and retained masks with 30 reference intervals.
Its canonical reconstruction is in [V2_FORWARD_GAP_SUMMARY.json](V2_FORWARD_GAP_SUMMARY.json).
The prior v6 FAIL is preserved, not relabelled.

| Retained session | Matched median | Largest next-token interval | Ratio |
| --- | ---: | ---: | ---: |
| HTP0 generation 1 | 536.833 ms | 779.177 ms | 1.451433x |
| HTP1 generation 1 | 536.896 ms | 783.106 ms | 1.460393x |

Each retained session had 108 calls during loading. Dynamic S was HTP2;
only S changed to Gemma generation 2. Loads remained 1/1/2 after Gemma
execution. Drain-to-quiescence was 1.187219 s; physical load authorization
to READY was 9.668457 s. These are different intervals.

Qwen's native online log contains zero capture/update markers. At full
width, retained layer 11 to layer 0 host submission gaps have median
354.977143 ms and maximum 586.412084 ms. The preserved graph-enabled v6
log has median 355.513130 ms and maximum 907.483216 ms for that class.
Full-width layer-11 RPC median/max are 9.131238/10.328798 ms (64 calls),
versus 10.758130/17.327835 ms (204 calls) in v6. This is diagnostic, not a
same-revision fleet A/B. Pooled fractions are not compared as equivalent.

Raw host-gap classes may span intentionally zero-assistance windows; such
gaps are not service-interruption measurements. The strict gate uses the
policy-aware matched-call classifier, not those pooled raw maxima. The
native USB ring slot rotates per call and is not a request slot. The v2
absolute parser's slot-equality mistake was corrected for postprocessing
and v3 onward; original v2 artifacts were not overwritten.

## Tests and remaining findings

- Initial graph-mode focused set: 182 PASS, including both replay goldens.
- Diagnostic persistence follow-up: 25 PASS.
- Reverse repair plus physical/session/replay tests: 90 PASS.
- Final preparation-boundary repair plus helper/adaptive/session/replay
  tests: 115 PASS, with both goldens unchanged.
- Incomplete-reference evidence collection plus graph mode and both
  replays: 30 PASS. Regression tests retain the final FAIL and immediate
  rejection of an observed over-bound gap.
- Entire multi-session plus graph-mode files: 42 PASS, one pre-existing
  FAIL in `test_energy_aware_request_executes_qualified_discovered_shards`.
  Its final cold-projection helper is absent. The identical failure was
  reproduced in `/tmp/s42-graph-before-check-20260906-6nflTt`, an isolated
  copy overlaid with the pre-change source backups. It was not hidden by
  weakening an assertion or changing unrelated helper policy.

Replay goldens remain unchanged:

- session_cow_gate_v3:
  `sha256:f78d2b2c37a3880a523eba4f5315ada0207678c841d633229782bfa3a05c1829`
- sparse_locality_v8:
  `sha256:965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

No full harness, trace, native build, commit, push, or PR was run.

## Files changed in this task

Paths relative to `research_dev/scheduler/`:

- `adapters/llama_server.py`
- `adapters/catalog_materialization.py`
- `_internal/runtime_capabilities.py`
- `_internal/desktop_parent.py`
- `_internal/route_generation/candidates.py`
- `_internal/runtime_plan.py`
- `scheduler.py`
- `campaigns/burstgpt/catalog.py`
- `campaigns/burstgpt/desktop_parent_calibration.py`
- `campaigns/burstgpt/runner.py`
- `campaigns/burstgpt/compare_ab.py`
- `campaigns/burstgpt/offline_residency_gate.py`
- `tests/test_cuda_graph_mode.py`
- `tests/test_multi_session_phone.py`
- `tests/test_burstgpt_replay.py`

Also this report directory and `research_dev/talks.md`. Existing unrelated
dirty-worktree changes and previous physical artifacts are preserved.
