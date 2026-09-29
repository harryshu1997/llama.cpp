# Full BurstGPT two-model RTX 4060 Ti plus OP15 test V1

Date: 2026-08-07 EDT.

## Result

Both physical arms completed all 74 source-length BurstGPT requests and all
11,605 requested output tokens. The CPU plus OP15 overlap route improved three
cold request deadlines, but it did not improve whole-trace makespan or energy.

| Metric | GPU switch control | GPU switch plus CPU/OP15 overlap | Change |
| --- | ---: | ---: | ---: |
| Trace makespan | 183.150 s | 184.250 s | +0.600% |
| Total output throughput | 63.363 tok/s | 62.985 tok/s | -0.597% |
| Trace SLOs met | 55/74 | 58/74 | +3 |
| Protected Qwen end | 100.297 s | 100.545 s | +0.247% |
| Cold completion mean | 120.229 s | 106.681 s | -11.268% |
| Cold completion p50 | 116.551 s | 115.412 s | -0.977% |
| Cold completion p95 | 162.986 s | 180.650 s | +10.837% |
| CPU package energy | 5.391 kJ | 7.867 kJ | +45.927% |
| GPU board energy | 23.270 kJ | 23.331 kJ | +0.259% |
| Server compute-device energy | 28.661 kJ | 31.197 kJ | +8.848% |

Server energy is RAPL CPU package plus NVML GPU board energy over the paid
trace. It excludes the phone, motherboard, DRAM outside package RAPL, fans,
storage, and AC conversion. No process swap was observed in either arm.

## Workload and capacity

The source-length trace contains 74 requests, 33,843 input tokens, and 11,605
output tokens. Its SHA-256 is:

```text
b20a9ba66ee3558d835a0e19ed3cfa4c31a4a9e8b4f9c085b29a14f80250a0ff
```

The two physical model artifacts were:

| Model | File bytes | SHA-256 |
| --- | ---: | --- |
| Qwen3-14B Q4_K_M | 9,001,752,960 | `500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0` |
| Gemma4-12B Q4_0 | 6,975,878,176 | `494518c2262a26e2a607af0e40bca11c4de5a0b108e7c21308906dbbfb1c6f8c` |

The RTX 4060 Ti reports 17,175,674,880 bytes of VRAM. Measured full-route
residency was 13,737,394,176 bytes for Qwen and 11,512,315,904 bytes for
Gemma. Their combined 25,249,710,080-byte runtime requirement cannot fit.
Only one full model was resident on CUDA at a time.

The control held every Gemma request until Qwen drained, unloaded Qwen, loaded
full Gemma on CUDA, and released the 17-request Gemma backlog. The treatment
sent requests 42, 46, and 50 to a CPU plus OP15 server as they arrived, held
the other 14 Gemma requests, then performed the same one-way GPU switch.

## Early CPU plus OP15 requests

| Request | Control completion | CPU plus OP15 completion | Reduction |
| ---: | ---: | ---: | ---: |
| 42 | 104.897 s | 18.095 s | 82.75% |
| 46 | 88.374 s | 14.949 s | 83.08% |
| 50 | 87.791 s | 17.226 s | 80.38% |

All three treatment requests met their 30-second SLO. Their requested token
counts completed, and sampled output was coherent. Their token sequences were
not identical to the full-CUDA control.

## OP15 operator path

The phone held 3,303.31 MiB of repacked Q4_0 FFN weights on HTP0. It executed
dynamic output-column slices in all 48 Gemma layers with F16 activation I/O.

| Transport or compute metric | Value |
| --- | ---: |
| FFN calls | 2,352 |
| Decode calls | 1,392 |
| Prefill calls | 960 |
| Upload bytes | 450,478,080 |
| Download bytes | 450,478,080 |
| Maximum wire payload | 3,932,288 bytes |
| USB p50 | 1.733 ms |
| USB p90 | 3.087 ms |
| Decode USB p50 | 1.626 ms |
| Prefill USB p50 | 2.981 ms |
| Final rolling phone-compute p50 | 1.385 ms |
| USB reset recoveries | 0 |

The worker exited with status zero, FunctionFS restored normal 5 Gb/s ADB,
and the bridge reported no reset recovery.

## Critical path and policy conclusion

All three CPU plus OP15 requests finished by 41.776 seconds. Qwen remained the
protected resource until 100.545 seconds, and full Gemma CUDA became ready at
106.381 seconds. The 14-request CUDA backlog then remained critical until
184.250 seconds. Phone execution therefore had no direct makespan overlap to
remove after the GPU switch.

The control CUDA backlog phase took 76.504 seconds. The treatment CUDA backlog
phase took 77.868 seconds despite containing three fewer requests. Continuous
batch shape and run-to-run service variation increased its tail by 1.364
seconds, which determined the final result.

The scheduling rule for this measured state is:

```text
deadline-first objective: dispatch requests 42, 46, and 50 to CPU plus OP15
makespan-first objective: hold all 17 requests for the GPU switch
energy-first objective:   hold all 17 requests for the GPU switch
```

The OP15 route is valuable for reducing selected request latency and deadline
misses, not for this trace's throughput or energy objective. A runtime policy
must make the objective explicit instead of treating every phone speedup as a
whole-trace speedup.

## Qualification boundary

This test exercises the existing physical hierarchical executor and the
integrated operator-level CPU/OP15 FFN split. The new
`matmul_vq_scheduler.schedule_next()` implementation is still a shadow planner
and is not called by this physical runner. Connecting its live resource
calendar directly to llama-server dispatch remains required before claiming
that the new VQ API itself is physically certified.

## Follow-up optimization probes

Four follow-up ideas were tested against the same RTX 4060 Ti plus OP15
system. Whole-trace times are shown for completeness, but the causal decision
uses the resource or request affected by each change. CUDA backlog time varied
substantially between single runs, so a lower overall makespan does not by
itself prove that a CPU or USB change helped.

| Probe | Physical observation | Scheduler decision |
| --- | --- | --- |
| Offload the predicted GPU tail | Request 51 finished 0.162 s before its control completion, but request 51 took 158.087 s on CPU plus OP15 and another CUDA request became the tail. Makespan rose to 185.810 s and server energy to 40.200 kJ. | Reject. A candidate must reduce the complete cohort critical path, not only the completion time of the request that was previously last. |
| Reduce cold CPU threads from 8 to 4 | Requests 42, 46, and 50 slowed from 18.094/14.948/17.226 s to 23.949/20.871/23.066 s. Protected-phase package power changed from 51.559 W to 52.422 W, so the run saved only 1.2% CPU energy. | Keep 8 threads for this host. Four threads are deadline-safe while hidden, but they provide no measured power advantage. |
| Stage large-prefill USB input before HTP attachment | The staged path completed all 2,352 FFN RPCs with zero resets, but large-shape RPC means increased by 0.268-1.068 ms. Aggregate USB p50 increased from 1.733 to 1.837 ms. | Keep direct HTP DMA-BUF input. The extra attach/copy work costs more than the attempted overlap saves. |
| Stage cold weights in spare VRAM | Qwen leaves at most 2,993,684,480 usable bytes, 26.0% of Gemma's 11,512,315,904-byte CUDA footprint. The current separate-process loader cannot adopt those allocations after Qwen exits. | Do not credit a staging gain. Require same-process allocation adoption plus a measured reduced-load row before enabling this route. |

The four-thread and staged runs reported 178.008 s and 171.120 s whole-trace
makespans, respectively, but their selected CPU plus OP15 requests were slower
than the direct eight-thread route. Their lower tails came from a faster CUDA
backlog in those individual runs and are not attributed to the tested change.

### Staged-VRAM upper bound

The measured host-to-CUDA model is 1,676,850,544 bytes/s. Even perfect reuse
of all 2,993,684,480 spare bytes could avoid at most:

```text
2,993,684,480 / 1,676,850,544 = 1.785 s
```

This is less than 1% of the 183.150-second control trace and does not remove
CUDA allocation, graph setup, or initialization. The earlier dual-residency
probe slowed protected Qwen by 2.80-2.97%, or about 2.8 seconds on this trace,
already exceeding the ideal upload saving. Cross-process staging has exactly
zero reusable CUDA bytes, so rerunning it cannot qualify the route.

The placement planner now fails closed on a staged-allocation claim unless
the allocation is explicitly adoptable by the execution process. This keeps
the measured full load cost in the route until an in-process partial-to-full
promotion mechanism exists and is profiled.

Follow-up raw artifacts:

```text
/home/zhihao/s41-dynamic-ffn-v1/campaign/raw/vq-tail51-op15-direct-final-20260807T1530Z
/home/zhihao/s41-dynamic-ffn-v1/campaign/raw/vq-short-op15-t4-20260807T1540Z
/home/zhihao/s41-dynamic-ffn-v1/campaign/raw/vq-short-op15-staged-t8-20260807T1800Z
```

Follow-up `RESULT.json` hashes:

```text
ca5d345689679fe899c147d5644673d6338b1ed7cf7eaaa8e4d2426ce0c7e9f7  tail51 direct
2764ed9a3d2449e0ef1888f1158c6093a7f67fa54a4b117fe6f5ee41e43e0177  four threads direct
7cced86aad3ff8bf43930b0cbd27e78c7f722f31ad63e6d152adbbb719e9c87b  eight threads staged
```

## Raw evidence

The raw artifacts remain on the physical 4060 Ti host:

```text
/home/zhihao/s41-dynamic-ffn-v1/campaign/raw/vq-full-gpu-switch-control-20260807T135404Z
/home/zhihao/s41-dynamic-ffn-v1/campaign/raw/vq-full-op15-treatment-20260807T135850Z
```

Result hashes:

```text
02f3782b22fe680a044a72956926ef9f9e564a2a52ec51d2aa19f84d0542e6e0  control RESULT.json
9ff0b2dc4ba7be6538f7e3791e8ebc52cb47a35cd1583761dab0c3ab2bba7ed1  treatment RESULT.json
```
