# S33 Full Quantized Route Results

## Verdict

`CAPACITY_PASS_QUALITY_FAIL`

Full Q4_0 and Q8_0 files can be stored on both phones, and partial Q4_0
windows are substantially deeper than the earlier F16 route. Neither
quantized HTP route passes the quality contract, so no S33 route is eligible
for the scheduler.

## Runtime identity repair

The failed routes already used the same quantized GGUF on CUDA and both
phones, so their quality failure was not caused by mixed quantization. The
physical runtime nevertheless lacked an admission-time identity field, which
allowed older F16-phone/Q8-CUDA mechanics routes.

StageNet now optionally publishes two values after HELLO: the loaded GGUF
`general.file_type` and a launcher-verified SHA-256. The active S31 topology
requires file type 7 and one identical digest across all CUDA and phone
workers before creating batchers. The phone and desktop launch controls hash
their files before setting the advertised digest, and measurement validation
rechecks the three launch records.

The exchange was tested with the rebuilt current-source binary on both phones,
a CUDA test worker, and the three resident StageNet slices on the RTX 4060 Ti.
Every worker reported Q8_0 file type 7 and SHA-256
`7b56cbd0e0d96d5c8d7df9c21b39d67264e58cbabf7a96d4681eff8b3d492848`.
The desktop's older Q8_0 file was rejected because its digest differed.
Quantization and digest mutations are rejected. This is an admission-contract
pass only; it does not change the quality verdict below.

## Full-graph screen

Neither full 48-layer graph is executable on OP15 under the zero-swap gate.

| Format | Observation | Decision |
| --- | --- | --- |
| Q4_0 | 106,240 KiB process swap followed by DSP abort, exit 134 | reject |
| Q8_0 | 1,152,256 KiB process swap before paid execution | reject |

The full GGUF may remain on phone storage, but only a bounded layer window can
be prepared for HTP execution.

## B32 resident capacity

All complete rows used seven measured cohorts after two warmups. All compute
ran on HTP0 except the declared stage-zero `GET_ROWS` CPU operation.

| Device | Q4_0 window | Median/step | p95 | VmHWM | Swap |
| --- | --- | ---: | ---: | ---: | ---: |
| OP12 | `[0,5)` | 935.925 ms | 943.324 ms | 4.75 GiB | 0 |
| OP12 | `[0,6)` | 995.933 ms | 1003.018 ms | 5.00 GiB | 0 |
| OP12 | `[0,8)` | 1361.535 ms | 1368.285 ms | 5.46 GiB | 0 |
| OP12 | `[0,12)` | 1970.863 ms | 1979.869 ms | 5.49 GiB | 0 |
| OP15 | `[4,17)` | 235.657 ms | 240.815 ms | 6.64 GiB | 0 |
| OP15 | `[4,20)` | 280.771 ms | 286.326 ms | 6.63 GiB | 0 |
| OP15 | `[4,24)` | 341.885 ms | 349.361 ms | 6.63 GiB | 0 |

OP15 `[4,32)` was rejected at load because it used 122,112 KiB swap. The
largest clean tested overlapping Q4_0 residency therefore gives OP12
`[0,12)`, OP15 `[4,24)`, and duplicated choice over `[4,12)`. Capacity alone
does not make those choices valid inference routes.

## Natural-prompt quality gate

The corpus is the first 128 eligible WikiText-2 raw test rows from pinned
revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Each request uses eight
frozen prompt tokens and eight greedy output decisions. Regeneration through
the Q4_0 and Q8_0 model tokenizers produced byte-identical corpus JSONL with
SHA-256 `1f51470235191e246e77f90236e0fa59907866f662eab55bac3030be9c207444`.

The frozen gates were first-token agreement at least 0.95, all-token agreement
at least 0.95, and exact eight-token sequence agreement at least 0.80.

| Format and physical route | First token | Token decisions | Exact sequence | Physical/reference B32 cohort | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Q4_0: OP12 `[0,4)` -> OP15 `[4,24)` -> CUDA `[24,48)` | 103/128 (80.5%) | 631/1024 (61.6%) | 55/128 (43.0%) | 16.839/0.828 s (20.34x) | FAIL |
| Q8_0: OP12 `[0,2)` -> OP15 `[2,16)` -> CUDA `[16,48)` | 105/128 (82.0%) | 721/1024 (70.4%) | 75/128 (58.6%) | 11.276/0.850 s (13.27x) | FAIL |

Both comparisons used the same quantized GGUF and shared CUDA tail for the
phone treatment and CUDA reference. Every stage emitted a passing placement
certificate. The final Q4_0 case also binds zero-swap snapshots to the exact
OP12 and OP15 worker PIDs before and after execution. The earlier Q8_0 token
run omitted PID identity in its memory records; the binder therefore marks
its resource evidence unbound. This cannot affect the Q8_0 decision because
the independently recomputed quality gate already fails.

## Fail-closed validation

`validate_quality.py` recomputes token metrics, thresholds, timing medians,
corpus and model hashes, topology, worker step counts, backend placement, and
resource identity. A raw probe never authorizes scheduling. The binder emits:

| Case | Binder result | Scheduler eligible |
| --- | --- | --- |
| Q4_0 | `QUALITY_FAIL`, resource identity bound | no |
| Q8_0 | `QUALITY_FAIL`, resource identity unbound | no |

Offline tests pass: corpus 5/5, probe 11/11, binder 10/10. Mutations cover
tokens, labels, thresholds, model hashes, memory PIDs, compute fallback, and a
would-be quality pass with unbound memory identity.

## Decision

CP5 installs zero quantized routes, and CP6 is stopped by contract. The
prototype should keep using its measured F16 phone windows for valid physical
execution. Quantized files can still serve as storage or capacity research
artifacts, but direct Q4_0/Q8_0 HTP kernels need a separate numerical-kernel
repair before they are reconsidered for scheduling.

No phone-energy or total-system-energy claim is made.
