# Gemma 4 12B system offload policy for RTX 4060 Ti plus OP15

Date: 2026-08-06 EDT

Status: performance and capacity policy. Fleet energy promotion remains
fail-closed because synchronized desktop CPU, desktop GPU, and whole-phone
energy has not been measured.

## Outcome

Use two placement levels:

1. If the Gemma model is resident on an available RTX 4060 Ti, run the whole
   graph on CUDA. Do not export individual CUDA operators to OP15.
2. If the RTX 4060 Ti is leased to the hot model, keep cold Gemma activations
   in desktop RAM and apply the CPU plus OP15 operator policy below.

For the second case, the target operator policy is:

- split the dense FFN in every layer for every physical graph size;
- split local-layer Q/K/V when physical `M >= 12`;
- split local-layer attention output projection when `M >= 48`;
- split the Q6_K vocabulary head when requested logit rows `Mh = 1..8` and
  the sampler accepts a compact top-k contract;
- keep embeddings, standalone normalization, residuals, local attention, KV
  updates, and sampling on the activation owner;
- keep wide global projections and long-context attention in shadow mode
  until their exact shapes pass an end-to-end p90 gate.

Only the FFN row has completed the full llama-server continuous-batch path.
The projection and vocabulary rows are qualified operator candidates, not yet
production server routes.

## Model and scheduling variables

This policy is for the measured dense Gemma 4 12B Q4_0 GGUF:

| item | value |
| --- | ---: |
| transformer layers | 48 |
| local sliding-attention layers | 40 |
| wide global-attention layers | 8 |
| hidden width `H` | 3,840 |
| dense FFN width `I` | 15,360 |
| local Q / K / V outputs | 4,096 / 2,048 / 2,048 |
| global Q / K / V outputs | 8,192 / 512 / 512 |
| local / global attention input to O projection | 4,096 / 8,192 |
| vocabulary | 262,144 |
| FFN and projection weights | Q4_0 |
| tied embedding and output weights | Q6_K |
| measured model file | 6.48 GiB |

The size variables are:

- `M`: physical rows in the current llama.cpp graph, not HTTP request count.
  With continuous batching, `M = decode rows + prefill rows`.
- `Mh`: rows for which the graph actually requests logits. This is often much
  smaller than `M` because prompt rows normally do not all request logits.
- `C`: resident KV context tokens used by one attention query.
- `Np`, `Kp`, and `Ip`: phone output rows, reduction columns, and FFN columns.

The measured server used `n_ubatch = 512`, so `M = 1..512` is the qualified
range. Reprofile larger physical graphs instead of extrapolating this table.

## Whole-system envelope

These rates are deliberately labeled by precision and measurement scope.
Peak compute numbers alone must not select an operator.

| resource | capacity | measured or usable rate | power evidence | policy consequence |
| --- | ---: | --- | --- | --- |
| i9-12900K and DDR | 30 GiB visible RAM | 50 GB/s isolated weight streaming; 29.7 GB/s whole-engine decode; 1.41 TFLOP/s saturated FP32 GEMM | configured PL1/PL2 180/241 W; synchronized package power missing | baseline owner for the cold model |
| RTX 4060 Ti | 16,380 MiB VRAM | 247.1 GB/s measured D2D traffic, 288 GB/s rated; 22.1 TFLOP/s rated FP32 | 7.4 W idle, 165 W limit; busy-hot-run board p50 about 107.6 W | use as a whole-model route when free |
| desktop RAM to CUDA | pinned host staging | 1.680 GB/s H2D, 1.705 GB/s D2H in the live probe | included in GPU board and host power only partially | do not create per-operator CPU/GPU ping-pong |
| direct FunctionFS DMA-BUF USB | one 5 Gbit/s link | 465.81 MB/s H2P, 465.15 MB/s P2H; 434.61 MB/s each way simultaneously; about 0.235 ms fixed RPC floor | USB-controller power not isolated | only compact, overlapped boundaries qualify |
| OP15 HTP v81 | HTP0 mapping about 3,328 MiB; separate HTP1 mapping about 3,200 MiB | 11.66 TFLOP/s saturated FP16 HMX; exact Q4/Q6 kernels and shape timings override this peak | whole-phone active estimate 3.99 W, 1.74 W marginal over idle; use 4.5 W as a conservative screen | primary phone backend for resident Q4/Q6 matmuls |
| OP15 Adreno 840 | shares 14.8 GiB phone RAM and LPDDR with HTP | 1.1-2.2 TFLOP/s dense GEMM; 66-77 GB/s Q8 GEMV; exact FFN xmem path loses to HTP | no isolated GPU rail; persistent polling power unmeasured | LM-head fallback only; not an additional FFN lane |
| OP15 RAM and UFS | 14.8 GiB visible RAM | 3.444 GB/s UFS direct read | DRAM and storage power not isolated | prepare weights before serving, never per request |

The CUDA-to-phone route is blocked. There is no qualified direct VRAM to
FunctionFS path, and the current solver has no measured CUDA D2H plus H2D
staging fit. Assuming zero staging would create a false phone win.

## Top-level placement policy

| condition | route | reason |
| --- | --- | --- |
| Gemma CUDA weights resident and CUDA lease available | all CUDA | VRAM bandwidth is over 500x the USB operator link, and no boundary is paid |
| CUDA free but Gemma not resident | load and use CUDA only if `Tload / Nexpected < Tcpu_phone - Tcuda` | amortize model load and avoid repeated eviction |
| CUDA leased to the hot model | CPU plus OP15 table below | preserves hot-model capacity and overlaps cold CPU work with OP15 |
| complete Q4 model resident on OP15 | whole-phone HTP is a capacity shadow route | engine probes reached 299.1 pp512 and 6.20 tg24, but the complete llama-server contract is not qualified |
| worker, residency, USB epoch, thermal state, or lease invalid | desktop baseline | readiness is an executor fact, not a scheduler prediction |

For a continuously busy hot CUDA service, the default CPU placement is the
measured keep-hot-speed policy: cold Gemma on P-core primaries
`0,2,4,6,8,10,12,14`, bridge/control work on `16-23`, and the hot server
allowed on `0-23`. With FFN offload this preserved hot throughput within
0.015% and improved cold throughput by 11.61% against the busy baseline.

## Matmul and fused-island policy

The CUDA column means that, once the whole model is CUDA resident, the
per-operator CPU plus phone ranges are not evaluated.

| operator | exact shape per row | legal split | CUDA-resident route | CPU plus OP15 size policy | phone backend | qualification |
| --- | --- | --- | --- | --- | --- | --- |
| token embedding lookup | tied Q6_K `262144 x 3840` table | token rows | CUDA lookup | CPU lookup for every `M` | none | reject standalone offload |
| local Q/K/V, 40 layers | Q `4096 x 3840`; K,V `2048 x 3840` | concatenate output rows | CUDA | `M=1..11` CPU; `12..23` phone half-Q; `24..127` phone Q; `128..512` phone K+V | HTP1 | physical CPU/HTP kernels plus direct-DMA model; server integration pending |
| wide global Q/K/V, 8 layers | Q `8192 x 3840`; K,V `512 x 3840` | Q output rows; K/V independent | CUDA | deploy CPU for now; shadow quarter-Q at `M=12..23` and half-Q at `M>=24` | HTP1 | conservative model from measured component shapes; exact wide sweep missing |
| local attention output | `3840 x 4096` | reduction width K, then sum | CUDA | `M=1..47` CPU; `M=48..512` split K 50/50 | HTP1 | physical CPU/HTP kernels plus 0.150 ms join reserve; server integration pending |
| global attention output | `3840 x 8192` | reduction width K, then sum | CUDA | deploy CPU for now; shadow 50/50 K split at `M>=48` | HTP1 | conservative gate; exact wide shape and join missing |
| dense GELU FFN, all 48 layers | gate/up `15360 x 3840`, down `3840 x 15360` | common intermediate width `Ip`, one input and one residual return | CUDA | `M=1`: phone 9,664 cols; `M=2..128`: 8,192; `M=129..512`: 11,136 | HTP0 | full llama-server continuous-batch and busy-GPU path measured |
| final vocabulary head | Q6_K `262144 x 3840` | vocabulary output rows, compact local top-k | CUDA | compatible sampler only: `Mh=1..2` phone 131,072 rows; `3..4` 98,304; `5..8` 65,536; `Mh>8` CPU | HTP1 | Q6_K kernel and worker pass; coarse split sweep passes; server sampler integration pending |
| vocabulary-head fallback | same | vocabulary rows | CUDA | if HTP1 is unavailable, `Mh=1` may send 46,080 suffix rows | Adreno | exact one-request path measured, but only 1.12% beyond FFN-only |
| MoE router and experts | absent in this dense 12B model | whole active experts for an MoE model | not applicable | not applicable | none | do not reuse a Gemma 26B-A4B policy here |

The local projection thresholds use a minimum 10% predicted p50 and p90 gain
and at least 0.100 ms absolute saving. The first raw Q/K/V median win was
`M=10`, but the safe gate is `M=12`. The first raw output-projection win was
`M=36`; reserving 0.150 ms for the join moves the safe gate to `M=48`.

The wide global projections are intentionally shadow-only. Their Q width is
twice the local width and their K/V width is smaller, so the natural cut is a
Q-row cut, not the local layer's Q-versus-K+V cut. The listed gates avoid
claiming an exact-shape result that has not run.

## Boundary bytes and phone arithmetic

All projection and FFN rows below use F16 wire activations. Byte counts are
request plus response, excluding the small protocol header. FLOPs count one
multiply and one add as two operations.

| route | phone shard | boundary bytes per graph | phone work per graph |
| --- | --- | ---: | ---: |
| local Q/K/V, half-Q | `Np=2048` | `11776 * M` | `15.73 * M` MFLOP |
| local Q/K/V, Q or K+V | `Np=4096` | `15872 * M` | `31.46 * M` MFLOP |
| wide Q/K/V, quarter-Q | `Np=2048` | `11776 * M` | `15.73 * M` MFLOP |
| wide Q/K/V, half-Q | `Np=4096` | `15872 * M` | `31.46 * M` MFLOP |
| local output projection | `Kp=2048`, output 3,840 | `11776 * M` | `15.73 * M` MFLOP |
| global output projection | `Kp=4096`, output 3,840 | `15872 * M` | `31.46 * M` MFLOP |
| dense FFN, 8,192 columns | `Ip=8192` | `15360 * M` | `188.74 * M` MFLOP |
| dense FFN, 9,664 columns | `Ip=9664` | `15360 * M` | `222.66 * M` MFLOP |
| dense FFN, 11,136 columns | `Ip=11136` | `15360 * M` | `256.57 * M` MFLOP |
| LM head, 4 Q6_K shards | 131,072 vocabulary rows | about `7936 * Mh` | `1.007 * Mh` GFLOP |
| LM head, 3 Q6_K shards | 98,304 vocabulary rows | about `7936 * Mh` | `0.755 * Mh` GFLOP |
| LM head, 2 Q6_K shards | 65,536 vocabulary rows | about `7936 * Mh` | `0.503 * Mh` GFLOP |

For the head, about 7,680 input bytes and 256 bytes of top-32 `(id, score)`
pairs cross per requested logit row. Full vocabulary logits do not cross USB.

The bandwidth-only screen is:

```text
Tphone_p = Qphone_p
         + Lrpc_p
         + (Bin + Bout) / Busb_p
         + Cphone_p(shape, split)

Tsplit_p = max(Thost_remaining_p, Tphone_p) + Tmerge_p
```

Use p50 for optimization and p90 for admission. Direct-DMA screening uses
`Busb = 465 MB/s` and `Lrpc = 0.235 ms`, but deployed routes must replace the
linear wire estimate with the exact-shape RPC lookup. For example, the real
long-prefill FFN path has protocol and transfer costs not predicted by ideal
bulk bandwidth alone.

Admit a new performance route only when:

```text
Tsplit_p50 <= 0.90 * Tbaseline_p50
Tbaseline_p50 - Tsplit_p50 >= 0.100 ms
Tsplit_p90 <= 0.90 * Tbaseline_p90
phone_queue_p90 is included
```

The already integrated FFN widths use their direct physical calibration
instead of this generic 10% gate. A fork should normally place the phone p90
completion just before the host remainder. If queueing makes the phone branch
critical, choose a smaller shard or fall back before dispatch.

## Non-matmul and attention policy

| operator | legal decomposition | size policy | location and fusion rule | reason or evidence |
| --- | --- | --- | --- | --- |
| attention RMSNorm | M rows or hidden channels with a reduction | no standalone offload for `M=1..512` | activation owner | CPU also needs the normalized vector and USB is slower than desktop RAM |
| Q/K RMSNorm, V RMSNorm, RoPE | heads or token rows | no independent RPC | activation owner; later fuse into an already selected projection island without another boundary | elementwise work is too small by itself |
| KV-cache write | token rows or heads | no standalone offload | KV owner | moving KV state every layer defeats residency |
| local sliding attention core | heads or KV context | CPU for `C<=1024` in cold mode | CPU | compact HTP attention exists, but no Gemma local CPU split is qualified |
| wide global attention core | KV context with online-softmax merge | no deployable phone range; shadow only at `M=1`, `C=262144`, phone suffix 8,192 | CUDA plus HTP only in shadow | continuous schedule gained 2.24%, but gapped p90 regressed 10.47%; `C<=131072` CUDA is faster |
| attention post-norm and residual | M rows or hidden channels | no standalone offload | CPU after partial-output join | post-norm depends on the complete merged result |
| GELU between gate/up and down | intermediate columns | offload only inside the fused FFN width island | same backend as its FFN shard | standalone activation traffic is larger than the fused hidden-state boundary |
| FFN post-norm and residual | M rows or hidden channels | no standalone offload | CPU after FFN sum | depends on the complete FFN residual |
| final RMSNorm | M rows or hidden channels | no standalone offload | CPU, or CUDA in all-CUDA route | insufficient work per transferred byte |
| layer scale and control vector | M rows or hidden channels | no standalone offload | activation owner | elementwise and dependency-bound |
| final logit softcap and suppress-token bias | vocabulary rows | keep with complete logits | CPU unless the head contract proves equivalent candidate handling | a compact top-k list may not contain a bias-promoted token |
| sampling | vocabulary candidates | top-k candidate merge only | CPU exact rescore and sampling | grammar, logit bias, repetition penalties, and full-vocabulary samplers disable head offload |

The global-attention shadow route has a 16,428-byte request and 16,564-byte
response at `M=1`, independent of `C`. Its 8,192-token F16 KV suffix occupies
16 MiB per global layer, or 128 MiB for all eight. The boundary is attractive,
but the measured gain is too small and too sensitive to USB wake gaps for
deployment.

## Phone backend and residency policy

| resident set | placement | size |
| --- | --- | ---: |
| maximum FFN suffix for all 48 layers | HTP0 | 3,303.29 MiB |
| all Q/K/V projections | HTP1 | about 810 MiB |
| all output projections | HTP1 | about 405 MiB |
| four repacked Q6_K head shards | HTP1 | 600 MiB |
| HTP1 projection plus head upper bound | HTP1 | about 1,815 MiB |

HTP0 and HTP1 have separate virtual mappings but share the physical DSP and
phone memory system. The operators in one model graph are dependent, so run
them sequentially on HTP. Do not treat HTP0 and HTP1 throughput as additive.

Do not split one FFN suffix across HTP and Adreno. The physical xmem sweep at
`M=16,32,64,128` found every HTP plus GPU split slower than intact HTP. At
`M=16`, even a 64-column Adreno shard changed 2.789 ms HTP control into a
4.331 ms dual result. HTP and Adreno also contend for shared LPDDR.

All weights must be converted, packed, allocated, hashed, and warmed before
the worker advertises readiness. A policy decision never triggers weight
repacking or storage reads on the request path.

## Power and energy overlay

The latency policy can run in performance or capacity mode now. Energy mode
must remain on the desktop baseline until synchronized measurements exist.

For one paid interval, compare the same completed work with:

```text
Ebaseline = integral(Pcpu + Pgpu + Pphone_idle + Pplatform) dt
Esplit    = integral(Pcpu + Pgpu + Pphone_active + Pplatform) dt
```

An implementation may use per-resource active and idle intervals, but the
component set and time boundary must match. Promote a route for energy only
when all of these pass in at least three alternating pairs:

```text
UCB(Esplit) <= 0.95 * LCB(Ebaseline)
Tsplit does not regress
mean exposed join wait / mean island time <= 0.05
completed work and requested quality are equal
```

The 3.99 W active-phone estimate is useful for screening, not promotion. An
older analytical bound placed CPU active-power break-even near 11.5-11.9 W
for the FFN and 3.3-3.4 W for the vocabulary head, but it omitted fixed
system power, CPU DVFS, phone idle power, USB-controller power, and shared
memory effects. It is not a fleet-energy result.

## Runtime guardrails

1. Select using physical `M`, `Mh`, and `C` after llama.cpp builds the graph.
2. Require exact worker identity, model hash, resident allocation receipt,
   transport epoch, DMA-capable FunctionFS kernel identity, thermal
   eligibility, and resource leases.
3. Include phone queue delay. Do not launch a phone branch that has already
   missed its host-overlap window.
4. Reserve HTP, the phone memory contention domain, and the USB root for the
   actual intervals in which each is used.
5. On a pre-dispatch readiness failure, use CPU. An in-flight failure needs a
   graph restart or a separately implemented exact fallback; it cannot silently
   reuse a missing partial result.
6. The measured FFN server path compares matched `--no-repack` CPU layouts.
   Reprofile all thresholds if the host-prefix implementation regains the
   normal shape-specific CPU repack path.
7. The current split path is approximate because F16 activation and partial
   sum boundaries can change token choices. Exact-quality requests stay on
   CUDA or CPU until an exact route is qualified.

## Evidence status

| route | strongest current evidence |
| --- | --- |
| dense FFN | full llama-server, continuous batching, mixed prefill/decode, original BurstGPT lengths, and three-repeat busy-GPU policies |
| local Q/K/V and output projection | physical i9 and OP15 exact-shape kernel sweeps plus measured direct-DMA fit |
| wide global projections | model-derived policy only; exact shape sweep still required |
| Q6_K head | real OP15 Q6_K backend and worker, plus `Mh=1..8` operator split sweep; sampler integration pending |
| global attention | correct context-shard mechanism, but only a narrow continuous median win and a gapped p90 failure |
| HTP plus Adreno FFN | physical rejection across `M=16..128` |
| energy | incomplete: no synchronized CPU, GPU, and whole-phone receipt |

Primary local reports:

- `RESULTS_CONT_BATCH_MATMUL_FINE_OP15_V2.md`
- `RESULTS_CONT_BATCH_MATMUL_SWEEP_OP15_V1.md`
- `RESULTS_BURSTGPT_GPU_CONTENTION_OPTIMIZED_OP15_V2.md`
- `RESULTS_LLAMA_SERVER_CONT_BATCH_OP15_V1.md`
- `RESULTS_Q6K_HTP_V1.md`
- `RESULTS_GEMMA_GLOBAL_ATTENTION_V1.md`
- `dual_backend_repack_v1/RESULTS_XMEM.md`
- `../arch_pyramid_v1/README.md`
- `../../s42_general_energy_scheduler_v1/RESULTS.md`
