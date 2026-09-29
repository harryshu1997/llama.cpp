# llama-server default CPU and OP15 balance probe

Date: 2026-08-06 EDT.

This I1 report is preserved as the predecessor. The current three-pair I3
latency, quality, overlap, server-energy, and whole-phone-energy result is in
`RESULTS_LLAMA_SERVER_I3_ENERGY_V1.md`.

Verdict: `MATCHED_TRACE_SERVER_ENERGY_PASS; DEFAULT_CPU_PASS; BALANCED_POLICY_PASS; FAIL_CLOSED_PASS; QUALITY_PROVISIONAL; FLEET_ENERGY_UNMEASURED; SINGLE_PAIR`.

One matched real-device pair completed all 74 source-length BurstGPT requests
with the default desktop CPU configuration. Adding the OP15 FFN split reduced
trace makespan by 11.1% and measured server compute-device energy by 25.5%.
This is a one-pair server-energy result. It is not yet a repeated fleet-energy
or semantic-quality claim.

## Changes under test

- The CPU control uses llama-server's default thread selection and normal CPU
  repacking. No CPU affinity or frequency limit is applied.
- The split route keeps normal repacking for non-FFN tensors and uses plain,
  view-safe storage only for the dense FFN gate, up, and down weights that are
  sliced by the graph.
- F16 activation exchange is retained because it halves USB bytes relative to
  F32.
- The split policy is a validated threshold table:

```text
1:9664,2:8192,4:6144,128:8192,512:11136
```

  OP15 advertises a 2,048-column quantum plus the 9,664-column alternate. This
  represents the four policy widths using seven resident blocks. A
  512-column quantum produced 23 blocks and is rejected because HTP aborted
  during warmup.

## Default CPU layout check

One 857-token prompt plus two generated tokens was run on the real i9-12900K
with no explicit thread or affinity setting.

| route | prefill | decode | tokens |
| --- | ---: | ---: | --- |
| stock CPU, default repack | 17,385.292 ms | 253.267 ms | exact reference |
| CPU with global `--no-repack` | 28,518.261 ms | 255.599 ms | exact reference |
| selective FFN layout plus OP15 | 12,516.130 ms | 178.694 ms | exact reference |

Normal repacking improves the CPU-only prefill by 39.0% relative to global
`--no-repack`. The selective split is 38.9% faster than stock CPU for prefill
and 41.7% faster for this two-token decode. These are bounded probes, not a
trace-level claim.

Raw roots:

```text
/home/zhihao/s41-dynamic-ffn-v1/server-smoke/default-cpu-20260806/
/home/zhihao/s41-dynamic-ffn-v1/server-smoke/no-repack-auto-20260806/
/home/zhihao/s41-dynamic-ffn-v1/server-smoke/selective-repack-op15-v1/
```

## Matched source-length BurstGPT trace

Both arms used the same 74 requests, 33,843 input tokens, 11,605 requested
output tokens, model artifacts, arrivals, default CPU thread selection, normal
CPU frequency policy, continuous batching, and hot Qwen3-14B CUDA route. A
per-run user cgroup set `memory.swap.max=0`; both model processes remained at
zero swap. The only treatment change was the balanced OP15 FFN split.

| metric | CPU control | CPU plus OP15 | change |
| --- | ---: | ---: | ---: |
| trace makespan | 733.730 s | 651.964 s | -11.14% |
| output throughput | 15.816 token/s | 17.800 token/s | +12.54% |
| completed work | 74 req / 11,605 tok | 74 req / 11,605 tok | equal |
| cold mean prefill | 61.598 s/request | 46.897 s/request | -23.87% |
| cold mean decode | 240.659 s/request | 221.586 s/request | -7.93% |
| cold mean service | 451.010 s/request | 388.759 s/request | -13.80% |
| cold mean TTFT | 210.351 s/request | 167.173 s/request | -20.53% |
| hot mean service | 11.385 s/request | 11.444 s/request | +0.52% |

The hot GPU route is effectively unchanged. The cold CPU-plus-phone route
sets the full-trace makespan and accounts for the observed improvement.

Raw roots and result identities:

```text
control: /home/zhihao/s41-dynamic-ffn-v1/server-traces/default-balanced-energy-cpu-noswap-r2-20260806T165709Z/
         RESULT.json sha256:04c35a16bcce2ada5a794ece46ce3cd51248ef057779d97eed1ee1e57f587ab8
treat:   /home/zhihao/s41-dynamic-ffn-v1/server-traces/default-balanced-energy-op15-noswap-r2-20260806T171049Z/
         RESULT.json sha256:c0587ed4ff71917bab4e534294cd343bf991b3fef5a8d3d760b007744c80597c
requests sha256:b20a9ba66ee3558d835a0e19ed3cfa4c31a4a9e8b4f9c085b29a14f80250a0ff
```

An earlier CPU attempt completed the request work but was rejected before any
comparison because the hot and cold processes accumulated 185.46 and
281.70 MiB of swap. It remains an invalid diagnostic and was not replaced
silently. The successor condition declared `memory.swap.max=0` and was applied
identically to the valid control and treatment.

```text
/home/zhihao/s41-dynamic-ffn-v1/server-traces/default-balanced-energy-cpu-r1b-20260806T164152Z/
FAILURE.json sha256:491f66953c0b42fe2e77c7898758c068277ab7fb6ca7aaf57a828ce16d1b7d47
```

## Quality diagnosis

Four real trace prompts were decoded greedily for eight tokens each. The CPU
control, coarse F32 split, coarse F16 split, and balanced F16 split used the
same Q4_0 model bytes.

| route | probe duration | exact requests | equal token positions |
| --- | ---: | ---: | ---: |
| CPU control | 84.916 s | 4 / 4 | 32 / 32 |
| coarse F32 split | 79.098 s | 2 / 4 | 21 / 32 |
| coarse F16 split | 60.184 s | 2 / 4 | 21 / 32 |
| balanced F16 split | 61.227 s | 4 / 4 | 32 / 32 |

F32 does not restore exact agreement, so F16 wire rounding is not the primary
cause of the earlier drift. The successful split changes FFN reduction
geometry, and greedy choices near a logit tie can change. The coarse F16 and
F32 routes diverged on different prompts while remaining coherent English.
Changing the M=4 geometry to the balanced width restored all 32 tested token
positions. This is evidence of split-geometry numerical sensitivity, not a
general quality proof. A broader task-quality gate is still required.

The full trace reinforces that limitation. Against the matched control,
2 / 17 cold request token sequences are exact and the median common prefix is
30 tokens. However, the untouched hot CUDA route is also exact for only
42 / 57 requests across the two runs. Continuous-batching timing changes the
reduction geometry even without offload. BurstGPT supplies workload shapes
and output lengths, not task answers, so cross-run greedy equality is a
reproducibility diagnostic rather than a semantic-quality score. The phone
route therefore remains quality-provisional pending a path-matched numeric
gate and a task-quality corpus.

Raw result identities:

```text
CPU:      100ded49752fc135ede71d00737e3f2b9914903bdbb79d5c0b243772ed55bec5
F32:      6eaac589f06339db81d7482ca53de21b0143f0e76f2d1643d0b026ddf2fb50da
F16:      24237b4c5c0cfc707a7920cb1157e66a44f382d647a139b3b358ed241440cedc
balanced: 42601a354ca6cbde70474ec30c172d40d8f5a0fc067e0cdacfa11e75574c6a19
```

## Branch balance

The same 857-token probe compared the coarse and balanced F16 policies.

| M | coarse width | coarse overlap | balanced width | balanced overlap | balanced wait |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 8,192 | 2.048 ms | 8,192 | 2.047 ms | 0.084 ms |
| 4 | 8,192 | 3.916 ms | 6,144 | 3.046 ms | 0.272 ms |

At M=4, the balanced table reduces overlapped island latency by 22.2% and
exposed join wait by 82.1%. The four-prompt balanced probe was 1.7% slower than
the coarse F16 probe overall because larger-shape timing varied, so only the
full trace can decide whether the table improves the workload objective.

## Failure behavior

A larger 11,776-column resident suffix reproducibly aborted HTP with
`dspqueue_read failed: 0x0000002e`. Before correction, llama-server ignored the
callback cancellation during model warmup and later emitted tokens with a
missing phone partial.

The graph scheduler now propagates callback cancellation as
`GGML_STATUS_ABORTED`; llama-server handles decode status 2 as an error; and
the split runtime rejects a server whose built-in warmup poisoned the phone
client. Repeating the same 11,776-column fault produced a server-load failure,
zero request streams, and no tokens.

Fail-closed evidence:

```text
/home/zhihao/s41-dynamic-ffn-v1/server-quality/failclosed-large11776-v3/
FAILURE.json sha256:
a6717e171e6ef29b2d12bc083d71d5aacb3ac55997e2c80119852ab38946921e
```

The safe resident ceiling remains 11,136 columns.

## Server energy

The runner now samples the Intel package RAPL counter and RTX 4060 Ti NVML
board power over one common monotonic paid interval. It reports:

```text
server_compute_device_energy_j = cpu_package_energy_j + gpu_board_energy_j
```

GPU energy uses trapezoidal integration of sampled board power. CPU energy
uses the unwrapped RAPL package counter with boundary interpolation.

| energy metric | CPU control | CPU plus OP15 | change |
| --- | ---: | ---: | ---: |
| CPU package average power | 158.439 W | 126.055 W | -20.44% |
| CPU package energy | 116.252 kJ | 82.183 kJ | -29.31% |
| GPU board average power | 27.169 W | 29.508 W | +8.61% |
| GPU board energy | 19.935 kJ | 19.238 kJ | -3.49% |
| server compute-device energy | 136.186 kJ | 101.421 kJ | -25.53% |
| server compute-device energy/output token | 11.735 J | 8.739 J | -25.53% |

The higher treatment GPU average power is a denominator effect: the same hot
work occupies the beginning of a shorter trace. Its total board energy still
falls by 3.5%. The server scope excludes AC conversion, motherboard, fans,
storage, and DRAM energy not included by package RAPL.

Whole-phone power was not sampled on the same paid interval. A sensitivity
calculation at 4.5 W for the full treatment adds 2.934 kJ, producing
104.355 kJ and a 23.37% reduction from control. This is not a measured fleet
result. The measured server saving has a 53.32 W phone-power break-even point.

## Phone work and overlap

All 17 cold requests invoked OP15. The phone executed 52,320 paid per-layer
calls and a call-weighted 64.61% of eligible dense-FFN hidden-column work. It
held 3,303.31 MiB of Q4_0 FFN slices and transferred 6,876,610,560 bytes in
each direction. This percentage is not a share of full-model work because
attention, embeddings, norms, and the vocabulary head remain on the server.

| mean component | time |
| --- | ---: |
| phone compute | 2.745 ms |
| complete phone RPC | 5.246 ms |
| host branch | 4.748 ms |
| exposed join wait | 1.382 ms |
| overlapped FFN island | 6.130 ms |

The arithmetic-mean exposed wait is 22.55% of island time, above S42's 5%
promotion target. The phone RPC is 9.50% longer than the host branch on
average, and 73.65% of phone RPC time is hidden. The current table is a real
energy and latency win, but it is not yet a well-balanced general scheduler
route. The dominant M=6 and M=8 decode shapes should use smaller cuts, while
large prefill shapes already hide most phone work behind the host branch.

## Verification

- Remote llama-server build: pass.
- Balanced four-prompt execution on RTX 4060 Ti desktop plus OP15 HTP: pass.
- HTP failure injection and zero-stream fail-closed check: pass.
- Matched 74-request control and treatment: pass.
- Control and treatment process swap: 0 bytes.
- Phone requests/status/restore: 52,320 / 0 / pass.
- Energy/statistics tests: 6 / 6 pass locally and on the desktop.
- Python syntax checks: pass.
- `git diff --check`: pass.
- No commit or push performed.

Current runtime library identities:

```text
libllama-server-impl.so 2954345b36c4fc69cca2379ec6a01b175d9b9827c2ed3f335e0ee1f553c6226a
libllama.so.0.0.0       580b81eba9c87efd0ca1aa0bc89fc10e08a1c38eb320c5cbe1c87a012b4a5946
libggml-base.so.0.15.3  fa78bcd5457b67bb1c987ac64dfae1b1dd65af746838a729ddf1530409c97d6b
```

## Next operation

Start a successor iteration with one declared scheduler change: rebalance the
high-frequency decode shapes using the measured arithmetic means, without
changing models, kernels, arrivals, CPU policy, or the safe 11,136-column
ceiling. Screen the new shape table with a path-matched numeric gate before a
paid trace. Add synchronized whole-phone power, then run at least three
alternating pairs before promoting the route to S42 energy enforcement.
