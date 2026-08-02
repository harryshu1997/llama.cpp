# TP-slice probe v1 - measured per-matmul speeds for 8-way tensor parallelism

Date: 2026-07-27. Status: `PROBE_GRADE_MEASURED; NOT_A_QUALIFICATION`.

Question: if Gemma-4-12B Q8_0 matmuls are split 8 ways across phones with the
partial results aggregated on the host over USB, what is the speed - and what
does the server CPU deliver while the GPU is busy with another model?

## Method

- New env-driven `TP_SLICE` perf case in `tests/test-backend-ops.cpp`
  (`TP_SLICE="q8_0:NxK;..."`, M=1 decode GEMV, self-contained struct).
  Same patch and tool on every backend.
- Phone builds from `build-op15-server-android` (NDK r27c, OpenCL), pushed to
  `/data/local/tmp/tp_slice/` on OP15 (Adreno 840) and OP12 (Adreno 750).
- Desktop build `build-s21-cuda` (i9-12900K + RTX 4060 Ti, CUDA 13.2.1 libs
  at `/mnt/storage/s21_deps`; link needed
  `-Wl,-rpath-link,/mnt/storage/s21_deps/cuda-13.2.1/lib` in the cache).
- Shapes from the GGUF itself (`gemma4`, 48 layers: 40 standard + 8
  wide-attention; standard layer used): q 4096x3840, k=v 2048x3840,
  o 3840x4096, gate=up 15360x3840, down 3840x15360; slices are N/8
  (col-split q/k/v/gate/up) or K/8 (row-split o/down).
- Engine level: `llama-bench` Gemma Q8 `-ngl 0 -t 8` pure CPU
  (`CUDA_VISIBLE_DEVICES=`), idle vs while `llama-bench` Qwen3-14B `-ngl 99`
  decoded continuously (GPU 99-100% for the whole window).
- USB collective: nc echo on each phone, payload striped 4 x <=4 KiB per
  phone (the adb forward chain stalls ~42 ms for any single write >=8 KiB -
  delayed-ACK signature, measured separately), both phones concurrent,
  scatter 15 KiB f32 + gather 15 KiB f32. 8-phone fan-in on 2 buses emulated
  by 4 sequential exchanges per link (pessimistic bound; 2-phone number is
  the ideal-8-bus bound). Python harness; a tuned C aggregator would sit
  somewhat lower.

## Exp 1 - server CPU while the GPU is busy (measured, llama-bench)

| condition | Gemma CPU prefill pp64 | Gemma CPU decode tg24 |
| --- | ---: | ---: |
| GPU idle | 31.6 +- 2.1 tok/s | 2.35 +- 0.00 tok/s |
| GPU 100% (Qwen decode loop) | 18.4 +- 0.5 (-42%) | 2.34 +- 0.00 (-0.4%) |

GPU side: idle pp512 1944 +- 10 / tg128 30.46 +- 0.01; under full CPU bench
1893 +- 29 (-2.6%) / 30.32 +- 0.03 (-0.5%).

CPU decode effective bandwidth 2.35 x 12.65 GB = 29.7 GB/s. Streaming
per-matmul rows (62.65 MB, larger than the 30 MB L3): idle 1.22-1.26 ms
(~50 GB/s raw GEMV), GPU-busy 1.29-1.36 ms (+2-11%). Matrices <=16.7 MB fit
L3 (CPU) / 32 MB L2 (4060 Ti) in the microbench and report cache speeds; real
decode streams all weights, so small-matrix rows must be scaled by streaming
bandwidth (CPU ~50, CUDA ~269 GB/s measured on the FFN rows).

Verdict: GPU occupancy costs CPU **decode nothing** and CPU **prefill 42%**;
the GPU is near-immune in both directions. The earlier C2-derived ~21 GB/s
CPU estimate was pessimistic; engine-level CPU decode is 2.35 tok/s.

## Exp 2 - 8-way TP slices on real phones + USB aggregation (measured)

Per-matmul microbench, us/run (Q8_0, M=1):

| matmul | bytes | OP15 full | OP12 full | OP15 1/8 | OP12 1/8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| q | 16.7 MB | 241 | 321 | 49 | 65 |
| k = v | 8.35 MB | 108 | 147 | 18 | 24 |
| o | 16.7 MB | 249 | 319 | 33 | 40 |
| gate = up | 62.7 MB | 945 | 1106 | 107 | 135 |
| down | 62.7 MB | 932 | 1202 | 105 | 149 |

Effective GEMV bandwidth: OP15 66-77 GB/s, OP12 52-57 GB/s - about 3x the
~22 GB/s whole-chain effective rate of the T2 decode step, i.e. most of the
current 379 ms chain step is engine/relay/attention overhead, not matmul.

Slice scaling efficiency is excellent: OP15 slice sum 436 us vs 3528/8 =
441 us (~100%); OP12 570 us vs 4348/8 = 544 us (95%). Compute is NOT the
problem.

USB collective (scatter+gather 15 KiB f32 per phone, host sum 3.8 us):

| variant | median | p95 |
| --- | ---: | ---: |
| 2 phones concurrent (= ideal 8 buses) | 2.85 ms | 3.75 ms |
| 8 phones on 2 buses (4x serialized) | 10.15 ms | 11.96 ms |

Standard-layer roll-up (7 matmuls, 238 MB):

| design | per layer | per token (x48) | tok/s |
| --- | ---: | ---: | ---: |
| 1x OP15 full (hypothetical 12.6 GB phone) | 3.53 ms | ~169 ms | 5.9 |
| 1x OP12 full | 4.35 ms | ~209 ms | 4.8 |
| TP8 compute only (straggler=OP12) | 0.57 ms | ~27 ms | (36.5) |
| TP8 + aggregate EVERY matmul (7 collectives) | 20.5-71.6 ms | 0.98-3.44 s | 0.29-1.0 |
| TP8 Megatron pairing (2 collectives) | 6.3-20.9 ms | 0.30-1.00 s | 1.0-3.3 |
| server CPU (measured engine) | - | 426 ms | 2.35 |
| 4060 Ti (scaled from 269 GB/s + measured Qwen tg) | - | ~47 ms | ~21 est |

Break-even: one collective costs 2.85-10.15 ms of wire; the LARGEST matmul in
the model only saves ~1.05 ms when split 8 ways (down: 1202->149 us). No
matmul in a 12B model pays for its own aggregation round; at the as-built
wire cost the matrices would need to be ~200 MB+ (70B-class FFN) to break
even.

Verdict: per-matmul TP over USB is 5.8-20x SLOWER than leaving the matmul on
one phone, despite ~perfect slice-compute scaling, because every exchange
costs more than the compute it saves. Consistent with S5
("USB transport RTT > compute") now quantified at matmul granularity on real
hardware.

## Optimized transport (second pass, same day)

The 42 ms >=8 KiB stall was root-caused: the phone-side echo (`nc`+`cat`)
never sets TCP_NODELAY, so Nagle + delayed-ACK ping-pong on the adbd->app
localhost hop. A 50-line static C echo daemon (`echo_nodelay.c`: TCP_NODELAY
+ 256 KiB buffers on the accepted socket) kills the cliff entirely:

| single socket, single write, OP15 | median RTT |
| --- | ---: |
| 4 KiB | 0.50 ms |
| 15 KiB (was 42 ms via nc) | 0.58 ms |
| 128 KiB | 1.86 ms |

Collective with persistent sockets + single-thread epoll client (no
per-exchange thread spawn):

| variant | old (nc+threads) | new | per-token wire at B=8 |
| --- | ---: | ---: | ---: |
| f32 15 KiB, 2 phones concurrent | 2.85 ms | 1.02 ms | - |
| 8 phones emulated on 2 buses (4x) | 10.15 ms | 4.52 ms | - |
| B=8 batched f32 120 KiB, 2 phones | - | 2.45 ms | 0.31 ms |

Re-rolled standard-layer table with the optimized wire:

| design | per layer | per token (x48) | tok/s |
| --- | ---: | ---: | ---: |
| 2-phone layer pipeline (realistic baseline) | 3.83 ms avg | ~184 ms | ~5.2-5.4 |
| TP8 + collective per matmul | 7.7-32.2 ms | 0.37-1.55 s | 0.6-2.7 |
| TP8 Megatron 2-sync, ideal buses | 2.61 ms | ~125 ms | ~8.0 |
| TP8 Megatron 2-sync, 2 buses | 9.61 ms | ~461 ms | ~2.2 |
| TP2 Megatron (the 2 real phones) | 4.21 ms | ~202 ms | ~4.9 |

First crossing: with the fixed transport and true per-phone buses,
Megatron-style TP8 would beat the 2-phone pipeline ~1.5x on single-stream
decode. Per-matmul aggregation still loses everywhere; TP2 on the real
hardware still loses; both bounds stay projection-grade until 8 phones and
a C aggregator exist.

Transfer to the real pipeline (checked in-tree):
- The C++ relay already sets NODELAY on every connect AND accepted socket
  (`layersplit.cpp:749`, `set_nodelay(cli)` in stagenet/tailstream,
  `stage-direct-relay.cpp:166`).
- `stage_v3_client.py:113` (host driver) does NOT set NODELAY, but its
  writes are one small sendall per batch (28 B/row; B=64 => 1.8 KiB) in
  strict request-response, so Nagle cannot stall it. Hygiene fix only.
- The T2 B>=2 decode penalty is FLAT (~+750 ms: 1123/1211/1278/1158 ms at
  B=2/3/4/5 vs 379 ms at B=1) - a fixed-stall signature, not linear
  serialization; specific bug, still unlocated.
- The OP15->OP12 WiFi hop (RTT 11-73 ms, spikes 474 ms) could route through
  USB via the host at ~1.6 ms/hop with this transport - a 10-45x per-hop
  latency cut plus jitter elimination, without waiting for any TP design.

## Slice-count sweep (third pass): the T(N) optimum

Measured slice compute at N in {1,2,4,8,16,32} on both phones (per standard
layer, sum of 7 matmuls, us) and wire vs phone count (fork-per-connection
NODELAY daemons, C=N/2 concurrent 15 KiB round-trip streams per real link):

| N | OP15 slices | OP12 slices (straggler) | OP12 scaling eff | wire/collective 2-bus | wire ideal buses |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3528 | 4348 | - | - | - |
| 2 | 1807 | 2148 | 101% | 0.95 ms | 0.95 ms |
| 4 | 917 | 1179 | 92% | 1.52 ms | ~0.95 ms |
| 8 | 436 | 570 | 95% | 2.32 ms | ~0.95 ms |
| 16 | 245 | 347 | 78% | 3.86 ms | ~1.0 ms |
| 32 | 168 | 243 | 56% | 3.90 ms | ~1.0 ms |

Kernel-launch floor ~13 us (OP15) / ~16-21 us (OP12) per matmul caps compute
scaling; DVFS repeat variance ~12% (same shape 20.9 vs 26.2 us on OP12).

Megatron 2-sync totals (compute straggler + 2x wire), single stream:

| N | 2-bus T/token | 2-bus tok/s | per-phone-bus T/token | tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 194 ms | 5.1 | 194 ms | 5.1 |
| 4 | 203 ms | 4.9 | 148 ms | 6.8 |
| 8 | 250 ms | 4.0 | 119 ms | 8.4 |
| 16 | 387 ms | 2.6 | 108 ms | 9.3 |
| 32 | 386 ms | 2.6 | 103 ms | 9.7 |

Baselines: 2-phone layer pipeline ~184 ms (5.3-5.4 tok/s); one hypothetical
12.6 GB OP15 ~169 ms (5.9).

B=8 batched wire (120 KiB/phone/collective): 2.59 ms (1/link) to 10.97 ms
(16/link); per-token wire at ideal buses 0.65 ms/layer -> ~34 ms/token ->
~28-30 tok/s aggregate at N=8-16 (GEMV+wire only; attention/KV excluded).

Structural ceiling: Megatron all-reduce moves 2 x 30 KiB per layer per token
through EVERY phone's USB regardless of N => 2.88 MB/token/phone; at the
measured ~220 MiB/s ADB one-way rate that is ~13 ms/token of unavoidable
wire => ~76 tok/s hard transmit ceiling for this model over USB TP. The
latency->bandwidth crossover is at B~7; beyond that the links run at
maximum transmit speed and the ceiling binds.

Optima: with the real 2 buses, N=2 is optimal and ties the pipeline (no
win). With one bus per phone, T(N) flattens at N=16 (9.3 tok/s, 1.75x the
pipeline) and wire is 89% of the token at N=32 - more slices buy nothing.

## Prefill pass (fourth): TP_SLICE @M, M=64

`TP_SLICE` extended with `@M` (a=[K,M]); rebuilt android + desktop. Per
standard layer, M=64 prompt, us (sum of 7 matmuls):

| device | full | N8 slices | N16 slices | slice eff |
| --- | ---: | ---: | ---: | ---: |
| OP15 | 15,684 | 5,285 | 4,981 | 37% / 20% |
| OP12 | 21,873 | 7,261 | 7,019 | 33% / 19% |
| 4060 Ti | 949 | - | - | - |
| i9 CPU | 46,074 | - | - | - |

Prefill slices collapse: compute-bound GEMM tiles lose 3x at narrow N
(N=256 slice runs 119 GFLOPS vs 1.1-2.2 TFLOPS full-width on Adreno 840);
slicing beyond N=8 buys nothing (launch+tile floor ~5 ms/layer OP15).

Prefill collective (per phone, P=64): f32 960 KiB = 13.94 ms median,
f16 480 KiB = 7.59 ms (bandwidth regime, ~140 MB/s round-trip effective).

Engine anchors (llama-bench, measured): GPU pp64 1010 +- 54, pp512 2199
+- 14; CPU pp64 31.6 idle / 18.4 GPU-busy.

Prefill totals for a 64-token prompt (component level = matmuls+wire only):

| route | prompt time | prefill tok/s |
| --- | ---: | ---: |
| 4060 Ti engine pp64 | 63 ms | 1010 |
| 4060 Ti matmuls only | 46 ms | 1406 |
| one OP15, no split | 753 ms | 85 |
| 2-phone pipeline 30/18 | 864 ms | 74 |
| TP8, f32 wire | 1,687 ms | 38 |
| TP8, f16 wire | 1,077 ms | 59 |
| USB TP bandwidth ceiling | ~840 ms | ~76 |
| CPU engine, GPU idle | 2,025 ms | 31.6 |
| CPU engine, GPU busy | 3,478 ms | 18.4 |

Prefill verdicts: (1) TP makes prefill WORSE than one unsplit phone at
every N - slice compute loses 3x efficiency AND wire scales with P; (2)
the same ~76 tok/s USB ceiling binds prefill and decode (2x30 KiB per
layer per token per phone, independent of M); (3) phones beat the CPU
route (85 vs 31.6/18.4) but sit 12-26x under the GPU; (4) engine-level
phone numbers will land below these component ceilings (T2's real chain
achieved ~30% of component on a different model/relay).

## NPU pass (fifth): engine-level HTP vs OpenCL, real Gemma-4-12B

`test-backend-ops` CANNOT drive HTP for this study: every TP_SLICE perf case
aborts in `flush_pending` -> `dspqueue_read failed: 0x2e`, at any graph size
(TP_RUNS=200 and TP_RUNS=8 both abort; added an opt-in `TP_RUNS` cap to
eval_perf while diagnosing). Known Hexagon instability, same family as the
S9 `dspqueue_read` abort. So NPU numbers here are ENGINE level
(`llama-bench`, real model, whole graph) - not per-matmul components, and
therefore not directly composable into the TP tables above.

OP15 (Adreno 840 / Hexagon v81), `gemma-4-12B-it-Q4_0.gguf` 6.48 GiB,
identical model+quant on both backends, r=2:

| test | NPU (HTP0) | GPU (OpenCL) | NPU/GPU |
| --- | ---: | ---: | ---: |
| pp64 | 111.89 +- 1.05 | 87.96 +- 0.68 | 1.27x |
| pp256 | 258.12 +- 8.68 | 101.73 +- 1.76 | 2.54x |
| pp512 | 299.10 +- 0.79 | 94.25 +- 0.18 | 3.17x |
| tg24 | 6.20 +- 0.02 | 5.94 +- 0.05 | 1.04x |

The NPU advantage is entirely a prefill/HMX effect and grows with prompt
length (1.27x -> 3.17x); decode is a tie (1.04x) because both backends are
bandwidth-bound on the same LPDDR. Matches the earlier op12 roofline
(NPU knee M=512; GPU>NPU only at M=1) on a different SoC and real model.

Server anchors on the same model family: 4060 Ti pp64 1010, pp256 2134,
pp512 2199, tg128 20.48 (Q8_0 - no Q4_0 copy on the desktop, so the phone
Q4_0 vs server Q8_0 comparison favours the phone on bytes moved).

Engine-level phone-vs-server, prefill (pp512): 299 vs 2199 = 7.4x gap
(was 12-26x from GPU components). Decode: 6.20 vs 20.48 = 3.3x gap.

Consequence for the TP tables: the NPU changes the PREFILL verdict's
magnitude but not its direction - TP still slices a compute-bound GEMM
whose tiles underfill, and the >=270 us/token/layer all-reduce tax still
exceeds the ~245 us/token/layer unsplit prefill compute. Decode TP
conclusions are unchanged (NPU decode = GPU decode within 4%).

Not measured: HTP per-matmul slice efficiency (tool aborts), OP12 v75 HTP
on this model (v75 FLASH_ATTN is known-broken; see the op12-v75 memo),
NPU+TP combined.

## Q8 vs Q4_0 residency on one phone (sixth pass, NPU)

OP15 MemTotal 15.5 GB, MemAvailable 9.80 GiB; `gemma-4-12B-it-Q8_0.gguf` is
11.80 GiB on disk (11.78 GiB reported). It RUNS (mmap streams it) but
thrashes:

| model on OP15 HTP0 | pp64 | decode |
| --- | ---: | ---: |
| gemma-4-12B Q4_0 (6.48 GiB, fits) | 111.89 | 6.20 tok/s |
| gemma-4-12B Q8_0 (11.78 GiB, does NOT fit) | 40.60 | 1.53 tok/s |
| ratio | 2.76x | 4.05x |

Q8 costs 4.05x decode on one phone purely from residency - not compute.
Correct Q8 deployment therefore needs a 2-phone layer split, which consumes
both phones for ONE model instance. Component estimate for a 32/16 Q8 split
(measured Q8 slice data: OP15 3528 us/layer x32 = 113 ms, OP12 4348 x16 =
70 ms, plus relay) is ~190 ms/token = 5.3 tok/s component, ~3.0 tok/s after
the measured 1.77x engine/component derate.

Capacity relief vs the 4060 Ti (20.48 tok/s decode, 2199 pp512):

| phone deployment | phones | decode tok/s | % of server decode | prefill tok/s | % of server prefill |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_0, 2 independent replicas | 2 | 12.40 | 61% | 598 | 27% |
| Q8_0, 2-phone layer split, 1 instance | 2 | ~3.0 | 15% | ~90 est | 4% |
| Q8_0, 1 phone thrashing | 1 | 1.53 | 7% | 40.6 | 2% |

Quantization choice, not phone count, decides relief: Q4_0 replicas deliver
4.1x the decode relief of a Q8 split on the SAME two phones.

Coupling: the phones hang off the controller host over USB, not off the
4060 Ti desktop, so phone compute contends for zero server resource. The
only way phone work can extend server time is if the server WAITS on a
phone - i.e. intra-request layer splitting with the server in the chain.
Whole-request ownership keeps server time untouched by construction.

## Caveats

- Microbench weights are resident and re-run in a loop: cache-fast rows on
  CPU/CUDA corrected as noted; phone SLC is small relative to these sizes.
- Python collective harness overhead included (~thread spawn per collective);
  ping-level floor is 0.5/1.1 ms per phone, so a tuned aggregator could
  reach ~1.3-1.7 ms per collective - which still loses everywhere.
- DVFS uncontrolled on phones; single-session numbers.
- The 8 wide-attention layers (q/o 2x wider, k 512-out) are not separately
  tabulated; per-token roll-ups scale the standard layer by 48.

## Cleanup

GPU loop killed (0% util, 299 MiB), no llama processes left on desktop, nc
listeners and adb forwards removed, phones idle. Tool left at
`/data/local/tmp/tp_slice/` on both phones for reuse.
