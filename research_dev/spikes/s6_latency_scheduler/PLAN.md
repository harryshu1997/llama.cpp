# S6-L Latency Scheduler Screen -- PLAN

Latency-only screen for phone-local operator overlap. Energy is DEFERRED (no valid power
boundary; see [[m5-crossover-remote-energy-blocked]] and the energy note below). This is a
bounded measurement screen: it certifies mechanisms and reports PASS/FAIL/BLOCKED per
hypothesis. It does NOT build a production scheduler or edit the Gemma graph / KV / backend
scheduler. Do not touch gemma4.cpp, llama-graph.cpp, or ggml_backend_sched.

Pre-existing uncommitted user edits (layersplit.cpp decode-FA toggle, ggml-hexagon.cpp fused-FA
/ per-tensor sharing, talks.md) are preserved; the baseline diff is saved at
`scratchpad/s6_latency_scheduler/baseline_git_diff.txt`.

## Measurement infrastructure repairs (must pass before the screen)

- **Phase A -- oplayerprof**: explicit `--mode decode|prefill`. Prefill times only the T-token
  `llama_decode` (KV clear / input build / graph compile / xmem prepack excluded via warmups).
  Decode uses a fixed output/logits shape (no `pos%2` alternation) and holds a FIXED context C
  by trimming the appended positions with `seq_rm` OUTSIDE the timed window (reset time reported
  separately; reprefill fallback if a backend lacks partial rm). KV sized for the transient peak
  (independent of round count -> `min_secs` cannot overrun). `--min-samples` gates p95/`ok`
  status and the exit code. `LAYER_END==LAYER_START+1` validated. `--dev CPU` forces CPU
  (`n_gpu_layers=0`). Output-file creation checked; `ggml_backend_load_all()` init; every
  allocation / decode / sample-count / fallback failure propagates through JSON status + exit
  code. Truncate by default (`--append` opt-in).
- **Phase B -- correctness**: cross-backend residual-tensor diff at identical inputs and KV state
  (CPU / stock-OpenCL / xmem-OpenCL / HTP), at real context C (not position zero). Reports finite,
  max-abs, relative L2, and residual-component argmax agreement (NOT token argmax). Production gate
  rel_L2 <= 5e-3; a miss labels the backend PERF_ONLY and excludes it from scheduler integration.
  Reference + candidate residuals are dumped (`.f32`) with an FNV-1a hash for reproducibility;
  `resdiff.py` computes the diff.
- **Phase C -- dual-engine benchmark**: four distinct cases -- D solo, P solo, directly-measured
  D-then-P serial, and D||P concurrent -- via persistent worker threads with a start barrier,
  preallocated batches, matched warmups, and fixed graph shapes (thread spawn / batch alloc
  excluded from timing). Tracks completed rounds, propagates worker failure to a nonzero return,
  and reports per-leg solo/concurrent p50/p95/CoV, outer wall, overlap speedup, overlap efficiency,
  and per-leg slowdown. Concurrent worker durations are labelled CONCURRENT, never "alone".
- **Phase D -- xmem cache containment**: the global `static std::map<(cl_mem,offset)>` prepack
  cache is not lifetime-safe across contexts / recycled allocations. For S6-L the cache is DISABLED
  and stock OpenCL is used (xmem is PERF_ONLY per Phase B anyway). A minimal lifetime-safe design
  is proposed for human review; no invasive cache subsystem is added.
- **Phase E -- Hexagon FA gate**: the fused-graph decode is tested on OP15 and OP12 at
  B={1,16,32}, C={32,512,1024} vs a CPU / serial reference (finite + rel_L2 <= 5e-3). Where only a
  subset meets the strict cross-backend bar, the accuracy envelope is documented in the support
  predicate; the inaccurate "bit-identical" wording and unicode are replaced with ASCII.

## S6-L screen (latency)

- HTP decode: B={16,32}, C=512. GPU prefill: T={64,256,512}, stock OpenCL (xmem only if it passed
  correctness -- it did not). 10 warmups; >=100 measured ops or 5 s; 7 independent processes,
  discard the first; alternate experiment order; retain OpenCL profiling CSVs proving the selected
  kernel and no CPU fallback.
- **Static-overlap pass gate**: correctness passes; CoV <= 5%; overlap speedup >= 1.20x; concurrent
  wall <= 1.10 x max(true solo legs); each concurrent leg p95 slowdown <= 10%; at least two T values
  pass for both B=16 and B=32.
- **Complete-FFN merge (independent hypothesis)**: 16+48, 32+96, 32+480 rows merged into one
  gated-GELU FFN island vs two separate islands, counting pack, synchronization, the complete FFN
  compute, scatter, and real backend-resident transfers. Gate: end-to-end speedup >= 1.20x, merged
  p95 <= 0.85x separate p95, transfer/sync <= 15% of merged wall, correctness <= 5e-3, HMX for all
  three GEMMs.

## Energy (DEFERRED -- do not infer from latency)

No physical J/token is claimed. The offline energy scripts (`research_dev/energy/`) are repaired
and validated against SYNTHETIC traces only: trapezoidal integration of the USB-input rail, a
battery charge-counter (coulomb) path, gross vs incremental per-layer-token fields, strict validity
gates (usb-pin, idle-window, sample-count), `set -euo pipefail`, stale-file cleanup, and sampler
EXIT cleanup. `synth_validate.py` asserts the math recovers known ground-truth energy. A physical
run additionally needs a valid power boundary (a high-wattage charger un-pinning the USB rail, or an
unplugged discharge) and re-adding the oplayerprof marker emitter.

## Stop rules

Stop a hypothesis when its gate fails; continue the independent hypothesis. Test OP12 only after
OP15 passes. Do not begin production scheduler or model-graph integration even if a hypothesis
passes; report the passing result and propose the smallest integration design for human review.
Nothing is committed.
