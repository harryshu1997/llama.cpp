# Pixel CPU FFN tuning after bandwidth measurements

2026-09-23. Numerical and repeated performance **PASS**. Selected private
configuration: six persistent CPU threads, polling disabled, no forced
core affinity, using the original generic AArch64 CPU arithmetic library.

| One-token FFN width | Original CPU, matched controls | Tuned CPU | GPU references | CPU improvement |
| --- | ---: | ---: | ---: | ---: |
|8704|23.576ms|9.162ms|11.848ms|61.14% less latency|
|17408|44.291ms|18.291ms|20.610ms|58.70% less latency|

Full CPU throughput improves **2.421x**, and its latency is
11.25% below the mean GPU references.
Both selected CPU repeats beat both GPU references. Logical full-width weight
rate rises from12.074 to
29.237GB/s. These divide510MiB by complete
worker time; they are not physical DRAM counters or pure-kernel bandwidth.

## Process and measured arms

The first1200-call sweep found a large threadpool benefit but failed affinity
coverage: every requested mask remained0-7. The qualified CPU source enables
its Linux affinity path only under`__gnu_linux__`; Android silently reaches a
stub returning success. The private one-line fix also accepts`__ANDROID__`,
using the existing`sched_setaffinity` implementation. All other arithmetic
objects and compile flags are retained. The1320-call confirmation verifies
actual per-thread masks4-7 or2-7 via`/proc`, as well as output correctness.

| Confirmation arm | CPU threads / affinity | Half mean ms | Full mean ms |
| --- | --- | ---: | ---: |
|00-gpu|GPU vec4_u1|12.134|21.013|
|01-cpu-original|original4/disposable|23.016|44.704|
|02-cpu4-pool|4/persistent/mask 0|13.562|26.057|
|03-cpu6-pool|6/persistent/mask 0|9.169|18.265|
|04-cpu6-pin|6/persistent/mask fc|9.548|18.951|
|05-cpu-original|original4/disposable|23.770|44.133|
|06-cpu4-pin|4/persistent/mask f0|10.574|20.913|
|07-cpu6-pin|6/persistent/mask fc|9.508|18.920|
|08-cpu6-pool|6/persistent/mask 0|9.156|18.317|
|09-cpu-original|original4/disposable|23.749|44.195|
|10-gpu|GPU vec4_u1|11.562|20.207|

CPU6 without pinning is slightly faster than strict CPU6 pinning in both
repeats. Four persistent unpinned threads also improve latency; pinning those
four helps further. Persistent threads are the selected change; the Android
affinity fix is separately verified and is not required by that selection.
The experiment does not attribute the whole gain to thread-creation time:
thread lifetime, CPU placement and DVFS can all affect the result.

## Qualification and limits

- 2520/2520 saved desktop-reference checks PASS; max relative L2=0.000325483.
- All2040 CPU outputs are byte-identical to corresponding original Pixel CPU
 outputs. CPU/GPU outputs still differ in low bits as previously measured.
- Each arm repeats six layers18-23 and two widths ten times. First two repeats
 are warmups; the48 warm samples/width determine means. CPU candidates use
 the mean of nearest original-CPU controls before/after. GPU endpoints are
 references, not part of CPU-candidate baseline interpolation.
- The first generic comparison audit rejected missing reference-only metadata
 on the outer GPU arms. The original CONFIG remains intact; an explicit
 ANALYSIS_CONFIG corrects only that analysis metadata. Numerical results and
 raw timings were retained. The confirmation config includes this up front.
- Worker builds, private affinity-library build, pyflakes, output checks and
 cleanup PASS. Earlier build errors are archived and were never run on phone.
- No server, full-model token equality, energy, USB latency or simultaneous
 CPU/GPU FFN test. No production backend/worker/default deployment changed.

[Full measured summary](PIXEL_CPU_TUNE_RESULTS.json),
[confirmation audit](physical/pixel10pro-cpu-tune-confirm-1/run1/SWEEP_AUDIT.json),
[actual thread masks](physical/pixel10pro-cpu-tune-confirm-1/run1/CPU_AFFINITY.json),
[worker patch](software/pixel10pro-cpu-tune-v3/CPU_TUNE.patch),
[Android guard fix](software/pixel10pro-cpu-tune-v3/ANDROID_AFFINITY.patch).
