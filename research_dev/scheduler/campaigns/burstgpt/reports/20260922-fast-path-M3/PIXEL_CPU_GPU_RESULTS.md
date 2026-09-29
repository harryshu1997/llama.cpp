# Pixel 10 Pro CPU versus GPU FFN

2026-09-23. Numerical qualification and comparison PASS for600 calls.
The GPU is2.09x faster at half width and2.30x faster at full width with
the current backends. Cleanup PASS. This comparison runs entirely on Pixel;
the saved desktop CPU reference timings are not Pixel CPU measurements.

## Matched measurements

Mean worker time for one token through the FFN of one Qwen layer, averaged
equally across layers18-23. Same F16 weights, F16 input/output wire format,
5120 input/output channels,4352-channel partition size and saved inputs.

| FFN hidden width | Pixel CPU,4 configured threads | Pixel GPU,vec4_u1 | GPU speedup | GPU latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Half,8704 | 23.974813 ms | 11.486870 ms | 2.0871x | 52.0878% |
| Full,17408 | 45.848635 ms | 19.929104 ms | 2.3006x | 56.5328% |

Full-width matrix work divided by worker time is11.664 GFLOP/s on CPU and
26.834 GFLOP/s on GPU. These rates include graph and worker overhead; they
are not isolated-kernel throughput or hardware peaks.

Order: GPU,CPU,GPU,CPU,GPU. Each arm has120 calls,10 repetitions of six layers
at each width. The first two repetitions are excluded from timings, leaving
48 measurements per width per arm. Each CPU arm is compared against the
mean of its immediately preceding/following GPU controls, then the two
comparisons are averaged. No profiling is enabled.

| Repeat | CPU half | Matched GPU half | CPU full | Matched GPU full |
| --- | ---: | ---: | ---: | ---: |
| 1 | 23.180083 ms | 11.557510 ms | 43.646271 ms | 20.086719 ms |
| 2 | 24.769542 ms | 11.416229 ms | 48.051000 ms | 19.771490 ms |

Both CPU repeats are slower than each of their individual surrounding GPU
controls. Battery temperature28.2-28.9C; GPU/CPU clocks during execution were
not sampled, so the reason for the CPU timing variation is unverified.

## Configuration and correctness

CPU uses the existing generic AArch64-O3 library. The worker does not override
the backend's default of four CPU threads. No explicit-march/-mcpu, KleidiAI,
OpenMP or affinity tuning is enabled. The readiness snapshot has one idle
thread: the backend creates its temporary compute threadpool for graph
execution, so that idle snapshot is not a count of active compute threads.
No CPU binary was rebuilt or CPU fusion implemented for this comparison.

GPU uses the previous vec4_u1 shader:128 threads,128-lane subgroup,8 rows,
4352-channel blocks. The rejected up-matvec/SwiGLU fusion is disabled.

All600 outputs pass saved desktop CPU reference checks, maximum relative
L2=0.0003254826924654395, below the0.01 qualification threshold. CPU-only
maximum relative L2 is0.00009888407822192234 against that reference.
Each layer/width output is deterministic within each arm. CPU and GPU outputs
are not byte-identical; full-model token equality was not tested. The three
GPU arms are byte-identical to each other.

The harness checks model/runtime hashes, backend readiness, response IDs,
shapes, payload hashes, finite outputs, repeat counts and cleanup. Both CPU
and GPU run locally under the Pixel lock using the same replay procedure.
All finite workers exit0; boot unchanged and no ADB forwards created.
Pyflakes and raw audit PASS. No host model computation, USB timing, energy,
server latency or multi-phone measurement is part of the result. This is a
comparison of current builds; CPU-specific kernel/thread tuning is untested.

- [Measured summary](PIXEL_CPU_GPU_RESULTS.json)
- [Arm configuration](PIXEL_CPU_GPU_CONFIG.json)
- [CPU build and thread provenance](physical/pixel10pro-cpu-local-1/run1/BACKEND_PROVENANCE.json)
- [Raw numerical/timing audit](physical/pixel10pro-cpu-local-1/run1/SWEEP_AUDIT.json)
- [Cleanup](physical/pixel10pro-cpu-local-1/run1/CLEANUP.json)
- [Phone script](physical/pixel10pro-cpu-local-1/run1/RUN_PHONE.sh)
