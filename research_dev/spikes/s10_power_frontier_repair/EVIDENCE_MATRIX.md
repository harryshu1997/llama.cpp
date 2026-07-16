# S10-V0-R-E1 evidence matrix

Status: ATLAS BLOCKED. **Zero eligible measured rows.**

This is an audit of what is ON DISK, against the frozen gates in
EVIDENCE_CONTRACT.md section 8. A row is imported only when the artifact exists,
is hashed, and meets the gate. Prose, summaries, and stale claims are not
evidence, and nothing here was promoted to fill a gap.

Method: two independent read-only sweeps of `research_dev/spikes/s6_*`, `s8_*`,
`s9_*`, the historical `s10_power_frontier/`, `research_dev/energy/`, and the
untracked `scratchpad/` trees, each instructed to report file paths rather than
conclusions, then reconciled against the frozen gates.

## 1. Verdict per required oracle input

The oracle needs six evidence-derived kinds. None is currently satisfiable.

| oracle input | needs | best candidate on disk | status | reason code |
|---|---|---|---|---|
| `routes.*.duration_us` | RouteProfile PASS | A6000 FFN atlas (`s10_power_frontier/artifacts/cp2_a6000_ffn_latency.jsonl`) | INELIGIBLE | `SINGLE_PROCESS_NO_THERMAL` |
| `batch_profiles.*` | RouteProfile PASS at batch size | same atlas, M=1..1024 | INELIGIBLE | `SINGLE_PROCESS_NO_THERMAL` |
| `server_power.*`, `devices.SERVER.active_mw` | PowerProfile SERVER_WALL | `cp2_a6000_power_trace.csv` (NVML) | INELIGIBLE | `GPU_BOARD_NOT_WALL` |
| `devices.<phone>.active_mw` | PowerProfile PASS | none | UNKNOWN | `PHONE_ENERGY_PHYSICALLY_UNMEASURABLE` |
| `output_bytes` | BoundaryProfile PASS | `cp2_atlas.json` boundary bytes | INELIGIBLE | `ARITHMETIC_NOT_MEASURED` |
| `routes.*.extra_energy_nj` | BoundaryProfile PASS | none | UNKNOWN | `NO_TRANSPORT_ENERGY_MEASUREMENT` |
| (supporting) correctness | CorrectnessCertificate PASS | `s9_pipelined_transport/artifacts_r/*_gate.jsonl` rel_L2 2.9e-4 | INELIGIBLE as island | `COMPONENT_NOT_ISLAND` |
| (supporting) thermal | ThermalInterferenceProfile PASS | none anywhere | UNKNOWN | `NO_THERMAL_RECORD` |

**No PowerProfile of any device passes the gate.** The zero-row result comes from
this documented disk audit; an automated atlas importer has not been implemented.
Independently of the empty atlas, E1 refuses every `MEASURED` instance because
aggregate wall timelines cannot be added as per-device solver terms. A later
typed matched comparison is required before any physical claim is possible.

## 2. Rows that are real measurements but do not meet the gate

These are genuine and worth keeping. They are not eligible, and the reason is
mechanical, not editorial.

| id | claim | artifact | why INELIGIBLE |
|---|---|---|---|
| S9-T1 | staged-window speedup OP12 2.61x / OP15 2.28x median | `s9_pipelined_transport/artifacts_r/matrix/{OP12,OP15}_gate.jsonl`, 20 rows each, full per-row provenance (`worker_sha`, `host_sha`, `harness_version`, device serial) | 1 process per phone (gate needs 3); transport rate, not a route latency the oracle consumes |
| S9-T2 | byte accounting exact: 0 wasted / 0 retried / 0 duplicate, 111/111 chunks, 464,114,176 B on all 40 runs | same gate JSONL | strongest integrity evidence in the tree, but it is a transport invariant, not a RouteProfile or BoundaryProfile field |
| S9-T3 | rel_L2 2.9e-4 vs the production Gemma4 oracle | same gate JSONL | a component correctness point for the transported FFN, not a complete operator island compared end to end; `COMPONENT_NOT_ISLAND` |
| S9-T4 | simultaneous 2-phone: OP12 37.45 / OP15 22.86 MiB/s, no cross-interference | `artifacts_r/simultaneous/simultaneous.jsonl` | 2 concurrent processes, still below the 3-process gate; transport, not route |
| S10-A1 | A6000 FFN latency M=1..1024 (M16 545 us, M256 998 us), reps=300/point | `s10_power_frontier/artifacts/cp2_a6000_ffn_latency.jsonl` | single process; no thermal record; harness source `scripts/s10_gpu_ffn_atlas.cpp` is untracked, so no source revision pins it |
| S10-A2 | A6000 board power: idle 25.3 W, sustained mean 281.2 W, ceiling 299.8 W | `cp2_a6000_power_trace.csv` (323 rows) | `GPU_BOARD` scope (NVML `power.draw`). Also: the sensor changes value only ~57 times in 32.3 s (~1.77 Hz effective) against a 10 Hz poll, so n is ~57 independent samples, below `MIN_POWER_SAMPLES=100`. Two independent reasons to refuse. |
| S6-E4 | D-solo B16/C512 = 25.2 ms, B32 = 33.5 ms, rel_L2 5.1e-4, rounds 40/30 | `scratchpad/s6_energy_scheduler/cert/cert_dsolo.jsonl` | 1 process; `"dev":"HTP0"` is a backend, not a device identity; no build/model/graph hash; no boundary bytes |

S10-A2 is the single most instructive row in the audit. It is a real, reproducible
measurement, and the contract still refuses it twice over: wrong boundary, and too
few independent samples. That is the gate working, not a gap to be argued away.

## 3. Quantities that are UNKNOWN, and are not zero

| quantity | why | reason code |
|---|---|---|
| phone SoC / package energy | USB rail pinned at ~497/500 mA (`s6_energy_scheduler/cert/usb_pinned_proof.txt`, 56 bytes) so the rail clips against a ~5-7 W draw; battery reads Charging at 99% so the coulomb fallback is dead; OP12 has no root; no WiFi-adb; no external meter | `PHONE_ENERGY_PHYSICALLY_UNMEASURABLE` |
| USB / charger / host-relay energy | no instrument exists at that boundary | `NO_INSTRUMENT_AT_BOUNDARY` |
| complete `TOTAL_WALL` energy | requires a synchronized meter covering server AND phones AND supplies; none exists | `NO_TOTAL_WALL_BOUNDARY` |
| A6000 break-even gap (idle minus a lower state) | the only low-power state is auto-P8 (~25 W); nothing below it; power caps unsettable (sudo needs a password) | `NO_STATE_BELOW_P8` |
| phone thermal envelope during a route | OP15 throttling is visible (95 C / 883 MHz) but no ThermalInterferenceProfile was ever recorded | `NO_THERMAL_RECORD` |

`research_dev/energy/pwr_sampler.sh` exists but **has never been run**: there is no
output CSV anywhere on disk. An unrun sampler is not evidence of anything.

These stay UNKNOWN with a reason. The contract rejects encoding any of them as 0
(`E_UNKNOWN_AS_ZERO`). TOTAL_WALL timelines are also rejected as additive
per-device solver inputs; a system claim needs a later matched comparison record.

## 4. Negative and contradicted evidence (retained, never bindable)

Retained because negative results are useful. Recorded because the audit found
claims that their own cited artifacts do not support. **None of this is in this
spike's scope to fix** - it is recorded here so no later gate imports it.

| finding | detail |
|---|---|
| S6-E "xmem GEMM confirmed" is contradicted by its own citation | `s6_energy_scheduler/RESULTS.md` cites `cert/cl_profiling.csv` for a `kernel_gemm_xmem_f16_f32_os8` at 126x. In that CSV, `grep -ci xmem`, `prepack`, `os8`, and `gemm` all return **0**. The 126x kernel is `kernel_mul_mm_f16_f32_l4_lm` - the stock OpenCL kernel. The only "xmem" on disk is a self-declared `"tag"` string, not a kernel-name proof. |
| S6-E "every eligible M>=5 uses HMX" is not backed at 6 of 7 asserted M values | `cert/hmx_profile_units.txt` contains only M=1 (45 ops, all `hvx-tiled`) and M=8 (7 ops, all `hmx-tiled`). M=5,16,32,128,512,1024 unit selection does not exist on disk. The "7 hmx" are the 7 MUL_MATs of one M=8 graph, not 7 M values. |
| S6-E "native HMX FA" is contradicted | all 8 `FLASH_ATTN_EXT` lines in `hmx_profile_units.txt` carry unit `----`, i.e. no HMX attribution. |
| S6-L section 2 headline numbers are prose-only | 1.93x saturated speedup, +19% service, 1.336 fixedpair, 2.95e-3 / 3.61e-3 / 5.02e-3 rel_L2 - **none appears in any file on disk**, under a heading naming a directory that does not contain them. `1.9327` occurs once, as a different field (`sat_eff`) at a different shape (T=256) in a v1 log. |
| S6-L ffnmerge 16+48 row is a splice | the 16+48 run on disk (`ffn_op15.out`) has `relL2 0.00e+00, finite:0` - no correctness at all. The quoted correctness (2.18e-3) belongs to different shapes (32+96, 32+480 in `ffn4.out`). The published row joins a speedup from one run to a correctness value from another. |
| adb-push figures are wrong and device-swapped | `s9_phone_pim_runtime/DYNAMIC_RESULTS.md:88-89` and `s9_pipelined_transport/PLAN.md:19` say "~262 MiB/s OP15 / 216 MiB/s OP12". The only artifact, `s9_pipelined_transport/artifacts/adb_push_control.json`, says **OP12 median 248.8, OP15 median 86.5**. Labels swapped and OP15 ~3x too high. `s9_pipelined_transport/RESULTS.md:77` states it correctly (OP12 249, OP15 86 - UFS-write bound). |
| S9 dynamic route provenance does not match the runs it certifies | `s9_phone_pim_runtime/artifacts/worker_route_provenance.txt` carries `session_epoch` values (8398055730943420310 / 3391291238735785491) that match neither `op15_dynamic_v3.json` (3469643097045755382) nor `op12_dynamic_v3.json` (131250901754610570). The `artifacts_r/` gate rows do not have this problem: their provenance is per-row. |
| no source revision pins ANY phone measurement | `git ls-files examples/phone-pim/` is empty - the entire runtime under test is untracked - and the HTP backend is dirty at HEAD (`ggml-hexagon.cpp`, `hmx-flash-attn-ops.c`, `htp-ops.h`). Build identity exists only as binary SHA-256. Reproducible in principle; not reconstructable from any commit. This alone blocks `ArtifactDescriptor.source_revision` for every phone row. |
| S8 has no measurements at all | `s8_operator_island_affinity/` is contract and schema only. Its `fixtures/valid/profile_row.pass.json` is a **synthetic schema-test vector** (every hash `sha256:000...0`, `artifact_paths:["run0.json"]` missing on disk). It must never be counted as a measured row; under this contract it would be rejected as `E_ARTIFACT_MISSING` anyway. |
| historical S10 energy numbers are simulator output | `cp4_results.jsonl`, `cert_*.json` are oracle outputs over a synthetic instance grid. The historical S10 verdict is FAIL and remains INVALID/INCONCLUSIVE. |
| S9 residency has no measured artifact | simulator plus schema fixtures only; self-labeled `STATIC_SNAPSHOT_COHERENT`, "CAPACITY UNPROVEN". |

Two prior conclusions are **confirmed correct against disk** and are worth
recording as such: the S9-V1A repair honestly replaced an inflated 6.714x OP15
figure with 2.28x after finding a pathological w1 baseline, and the phone-PIM
no-fallback proof is structural, not a JSON field (`examples/phone-pim/phone_pim_ffn.cpp`
resolves a single backend and hard-errors at line 404 if it cannot take the op;
there is no `ggml_backend_sched` in that path, so there is no fallback to fall into).

## 5. The one valid bundle

`fixtures/evidence/mechanics_bundle.json` is a fully synthetic `MECHANICS_ONLY`
bundle: 27 records, 7 artifacts, every artifact `provenance: SYNTHETIC`. The seven
canonical artifact payloads are materialized and byte-hash verified. The bundle exists
to prove the mechanics end to end and it reproduces both frozen temporal optima
through the evidence-bound path (147250000 nJ and 2974000 nJ).

It can never support a physical claim. `E_PROVENANCE` rejects a `MEASURED`
instance built on it, and `classify_energy_claim` returns `NONE_MECHANICS_ONLY`
regardless of how well formed it is. Both are tested.

## 6. What would unblock the atlas

Not authorized here, and listed so the blockers are concrete rather than vague:

1. **Phone energy**: an external meter, or physical unplug plus WiFi-adb plus a
   real discharge window. The USB rail is pinned and the coulomb counter is dead
   while charging; no amount of care with the current setup produces a number.
2. **Server wall power**: a wall meter. NVML is the GPU board and can never be
   promoted to `SERVER_WALL`, let alone `TOTAL_WALL`.
3. **Route latency**: re-run the atlas at >=3 processes with a thermal record and
   a committed harness revision. The measurement is cheap; the identity fields are
   what is missing.
4. **Island correctness**: compare a complete operator island end to end against a
   reference route with an op-trace no-fallback proof, at pinned shapes.
5. **Source identity**: commit `examples/phone-pim/` and the HTP backend changes,
   or accept that no phone row can ever carry a `source_revision`.

Even after 1 and 2 exist, every physical relief label remains unreachable until a
typed matched comparison record is implemented. E1 emits mechanics certificates
only; it cannot issue GPU-board relief, server relief, or a total-system saving.
