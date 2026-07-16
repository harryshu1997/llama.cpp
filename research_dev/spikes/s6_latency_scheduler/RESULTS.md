# S6-L Latency Scheduler Screen -- RESULTS (canonical)

This file is the single canonical verdict. It supersedes all earlier "both hypotheses PASS" and the
first repair pass. Energy is DEFERRED (no physical J claim). Nothing is committed. The measurement
harnesses were repaired to be fail-closed (Phase 1-4 below) and re-run on real devices; the honest
verdicts follow. Do NOT start production scheduler / graph integration from these results.

Devices: OP15 = OnePlus 15 / Hexagon v81 + Adreno 840 (serial 3C15AU002CL00000, opt_arch=81).
OP12 = OnePlus 12 / Hexagon v75 + Adreno 750 (serial 5ae7a43d, opt_arch=75). Shard
`12b-f16-mid-2-3.gguf` (blk.2, headless, md5 identical on both). Deploy dir `/data/local/tmp/s6v2`.

## 0. Verdict table

| hypothesis / check | verdict | evidence |
|---|---|---|
| profiler + correctness oracle (Phase 1) | REPAIRED, device-validated | OP15 HTP vs CPU rel_L2 **2.95e-3** PASS; placement proven `[HTP0]` no CPU fallback; reset {ok,method}; attn_out captured |
| dualengine harness (Phase 2) | REPAIRED, runs with NO hang | epoch handshake + timed waits + watchdog; fail-closed emit; CPU cross-backend + service microtrace added |
| overlap -- SATURATED throughput | **PROVISIONAL** | OP15 FA-off B16/C512/T64: speedup **1.93x**, interference ~0 (conc/maxsolo 1.02), but decode CoV **0.060 > 0.05** at 1 rep; needs the 7-process sweep |
| overlap -- SERVICE (request) latency | **FAIL** | real advancing-KV microtrace: decode per-request p50 6.9 -> 8.2 ms under concurrent prefill = **+19%** (svc_slowdown_D=1.19 > 1.10) |
| complete-FFN merge (Phase 3) | **LOWER_BOUND** | OP15 real residual 16+48: all 3 controls correct (2.18e-3), merged 1.91x vs serial-HTP, xfer 2.8%, HMX x3 no-fallback; but 2 weight copies (single_copy=0) -> not PASS |
| Hexagon FA -- v75 (OP12) | **FIXED**, revalidated | AUTO builds explicit HTP attention, placement `[HTP0]` no CPU fallback; v75 HTP vs OP12 CPU rel_L2 **3.61e-3** PASS |
| Hexagon FA -- v81 (OP15) strict gate | **REPORT FOR REVIEW** | FA-ON decode B16/C512 vs CPU **5.02e-3** (marginally FAILS 5e-3); FA-OFF explicit **2.98e-3** PASS. v81 support policy UNCHANGED pending human review |
| energy (Phase 4) | **DEFERRED** | run_energy.sh refuses (exit 3); energy_align hardened; 16/16 synthetic cases pass; NO physical J |

## 1. Infrastructure repairs (all in the harnesses, none in model/graph/sched)

### Phase 1 -- fail-closed profiler + correctness oracle (`examples/layersplit/oplayerprof.cpp`, `research_dev/spikes/s6_latency_scheduler/resdiff.py`)
- reset returns `{ok, method, elapsed}`; `seq_pos_max` checked for EVERY sequence after `seq_rm`;
  reprefill fallback checks every `llama_decode` and stops the case on failure.
- `--fa` defaults to `auto`, rejects anything but off/on/auto; B/C/T/warmups/iters/secs/samples are
  validated positive and bounded BEFORE any allocation.
- correctness decode + residual dump are fatal on failure; file I/O checked (open/write/flush/close)
  against the exact expected byte size; finite test is an IEEE bit test (fast-math safe).
- requested backend AND proven graph placement recorded via `cb_eval`; a CPU compute node when an
  accelerator was requested fails the case (`cpu_fallback`).
- canonical `resdiff.py` rejects empty / truncated / trailing-byte / dimension-mismatched / non-finite
  files and exits nonzero on any failed comparison (7/7 self-tests pass on host).
- `--dump-attn-out` captures the REAL post-attention residual (gemma4 `cb(attn_out,...)`) via the
  eval callback -- no protected-file edit -- feeding ffnmerge `--residual-file`.

### Phase 2 -- dualengine (`examples/layersplit/layersplit.cpp` `run_dualengine`)
- start state is an OWNED generation counter (no raw pointers into caller stack); barrier arrival and
  completion use TIMED waits; a blown deadline triggers a process watchdog that dumps phase/generation
  and aborts (so a hung DSP `llama_decode` is diagnosable, not a silent hang). The prior intermittent
  hang did NOT reproduce across the device runs here.
- errors latch and name the first failing phase; every subsequent phase is skipped; undersample is
  rejected; a failed/timed-out/partial run emits ONLY identity + `valid:0` + status (NO gate metrics).
- compute completion is stamped right after `llama_decode`; reset runs outside compute timing; the
  compute-only makespan (`sat_conc_compute`) is reported separately from the reset-inclusive cycle.
- the SAME persistent workers drive solo, serial, and concurrent (the fixed-pair serial control no
  longer runs on the main thread); control order rotates across processes via `--rep`.
- the fixed-state metric is renamed `fixedpair_*`; a REAL SERVICE microtrace advances decode KV and
  issues fresh prefill requests, timing to output-ready, solo and concurrent.
- a separate CPU reference context cross-checks the decode output; solo and concurrent outputs are
  captured and compared (`xcorr`); `--no-cpu-ref` for RAM-tight devices.
- host-side proof: `scratchpad/s6_latency_repair_v2/worker_stress.cpp` (10,000-epoch handshake,
  delayed arrival, forced step/reset failure, hang-timeout, and a 50 ms-reset invariant showing
  compute-only latency unchanged while cycle time grows) = 9/9 PASS. `parse_dual.py --selftest` =
  11/11 PASS (rejects missing/duplicate/malformed/failed/timed-out/insufficient records).

### Phase 3 -- FFN merge (`examples/layersplit/ffnmerge.cpp`)
- resident weights use their real `*.weight` names (name-keyed sharing can match); `--share` toggles
  the rpcmem publish/import; physical memory is reported SEPARATELY from latency; `single_copy` is
  only asserted when a share actually collapsed the GPU allocation.
- every graph-compute status is checked; support is checked for all graphs (hA,hB,hM,cA,cB,gB); GPU
  absence is `UNSUPPORTED`, never a host-copy substitute that can PASS.
- one common boundary (A HTP-resident, B GPU-resident, in and out); correctness downloads are OUTSIDE
  timing; every control (serial, affine, merged) is validated vs the CPU reference.
- certification requires REAL residuals (`--residual-file`); synthetic input yields
  `SYNTHETIC_UNCERTIFIED`. The verdict ladder is
  FAIL_STATUS -> FAIL_CORRECTNESS -> UNSUPPORTED -> FAIL_GATE -> SYNTHETIC_UNCERTIFIED ->
  LOWER_BOUND -> PASS, so a false PASS is structurally impossible.

### Phase 4 -- energy (DEFERRED) (`research_dev/energy/`)
- `run_energy.sh` refuses immediately (exit 3) with a clear DEFERRED message (the `--idle-secs`
  marker mode does not exist in oplayerprof); `ENERGY_FORCE=1` gates the doomed path.
- `energy_align.py` requires samples bracketing both marker endpoints, enforces a max sample gap,
  counts REAL (non-interpolated) samples, rejects absent/zero USB rails, prefers coulomb while
  unplugged, interpolates charge-counter AND voltage from the full bracketing series, and applies the
  positive-window / rounds / toks / coverage / sample-count gates to BOTH the USB and battery paths.
- `synth_validate.py` = 16/16 PASS (baseline math + truncated head/tail, dropout, non-aligned coulomb
  edges, zero USB, unknown status, zero/negative denominator, duplicate markers, empty output).

## 2. Device measurements (raw JSON under `scratchpad/s6_latency_repair_v2/`)

OP15 dualengine B16 / C512 / T64, rep 0, GGML_DECODE_NO_FA=1 (FA-off, valid record):
```
sat_speedup=1.9327  sat_conc_vs_maxsolo=1.0204  sat_conc_compute=2644.5ms (cycle 2646.7ms)
sat_slowdown_D=1.017 sat_slowdown_P=1.013  sat_D_conc_cov=0.060 sat_P_conc_cov=0.0115
svc_D_solo_p50=6.91  svc_D_conc_p50=8.22  svc_slowdown_D=1.190   (SERVICE decode +19% under contention)
svc_P_solo_p50=80.4  svc_P_conc_p50=80.6  svc_slowdown_P=1.002
fixedpair_speedup=1.336  fixedpair_slowdown_D=1.129
xcorr=pass xcorr_solo=2.98e-3 xcorr_conc=2.98e-3 self_sc=0.0
```
OP15 FA-on (same shape): xcorr=FAIL xcorr_solo=5.02e-3 -> harness fail-closed, valid=0, no metrics.

OP15 ffnmerge 16+48 real residual: corr 2.18e-3, serial-HTP 25.32ms, affine 67.38ms, merged 13.26ms,
speedup_vs_best 1.91x, p95_ratio 0.534, xfer 2.8%, mem htp/gpu 354/354 MB, single_copy=0 => LOWER_BOUND.

OP12 v75 AUTO: placement [HTP0] no fallback; HTP fa=auto vs OP12 CPU rel_L2 3.61e-3 PASS.

## 3. Remaining work (set up, not yet run to completion)

- 7-process OP15 sweep across B={16,32} C=512 T={64,256,512} to upgrade SATURATED from PROVISIONAL
  (scripts `scratchpad/s6_latency_repair_v2/dual_sweep.sh` + `parse_dual.py`, validated on synthetic
  data). Command: `dual_sweep.sh 3C15AU002CL00000 /data/local/tmp/s6v2 12b-f16-mid-2-3.gguf <out> 7`.
- v81 FA on/off vs CPU at B={1,32} C={32,1024} to fully characterise the strict-gate margin.
- OP12 dualengine (RAM-tight ~5.2 GB; the v75 correctness is already validated via oplayerprof).
- FFN merge 32+96 and 32+480 with a 512-token real residual, 7 processes.
- S7-V1 trace test (separate spike `s7_ragged_attention`).

## 4. Do not

Do not enable a production scheduler or edit gemma4.cpp / llama-graph.cpp / ggml_backend_sched from
these results. Do not waive the v81 FA 5e-3 gate without human review. Do not claim any physical
energy number. Nothing here is committed or pushed.
