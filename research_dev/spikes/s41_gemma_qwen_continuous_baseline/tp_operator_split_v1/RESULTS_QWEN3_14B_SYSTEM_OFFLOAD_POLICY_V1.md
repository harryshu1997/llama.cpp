# Qwen3 14B Q4_K_M offload-policy trial

Date: 2026-08-06 EDT

Status: exact GGUF structure plus a model-derived CPU/OP15 shortlist. This is
not a deployed llama-server policy. Only one new Q6_K down-projection shape
has passed a physical OP15 correctness check.

## Outcome

The Gemma policy does not transfer unchanged to Qwen3 14B Q4_K_M.

- If the model is resident on an available RTX 4060 Ti, keep the entire model
  on CUDA. The measured 8.4 GB checkpoint fits the 16 GB card and has already
  run fully GPU-resident in the hot-model experiments.
- If CUDA is leased to another model, keep all Q4_K projections on the desktop
  CPU. The current OP15 HTP backend has no qualified Q4_K kernel.
- Shortlist a reduction-width split of the Q6_K FFN down projection. Gate and
  up remain on the CPU; after SwiGLU, OP15 consumes a disjoint intermediate
  slice and returns one 5,120-element partial residual.
- Shortlist the Q6_K vocabulary head only with a compact local top-k response.
  Never return the phone's full 151,936-row logit shard over USB.
- Keep V projection, attention, normalization, RoPE, residual operations,
  embedding lookup, and sampling on the CPU for the first implementation.

This is a useful fail-closed result. Gemma Q4_0 can place a complete fused FFN
slice on HTP. Qwen Q4_K_M cannot, because gate/up are Q4_K while down is Q6_K.
The best exact-source Qwen candidate is therefore a later split inside the
FFN, not the same fused island used by Gemma.

## Exact model structure

The metadata and tensor types below came from
`/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf` on the RTX 4060 Ti host.

| item | value |
| --- | ---: |
| transformer layers | 40 |
| context length | 40,960 |
| hidden width `H` | 5,120 |
| FFN width `I` | 17,408 |
| query heads / KV heads | 40 / 8 |
| key and value head width | 128 |
| vocabulary | 151,936 |
| model file | 8.4 GB |
| model SHA-256 | `500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0` |

Representative tensor shapes use `W[N,K] * X[K,M] = Y[N,M]`:

| tensor | shape | GGUF type |
| --- | ---: | --- |
| token embedding | `151936 x 5120` | Q4_K |
| Q projection | `5120 x 5120` | Q4_K |
| K projection | `1024 x 5120` | Q4_K |
| V projection | `1024 x 5120` | Q6_K |
| attention output | `5120 x 5120` | Q4_K |
| FFN gate | `17408 x 5120` | Q4_K |
| FFN up | `17408 x 5120` | Q4_K |
| FFN down | `5120 x 17408` | Q6_K |
| output head | `151936 x 5120` | Q6_K |

All 40 layers have the same dense attention and FFN dimensions. This model is
not MoE, so there is no expert-placement row.

## Trial inputs and admission rule

The sweep used the measured system envelope:

| resource | screening value |
| --- | ---: |
| full CPU engine weight bandwidth | 29.7 GB/s |
| CPU compute ceiling | 300 GFLOP/s |
| OP15 Q6_K HTP compute | 111.2 GOP/s at M=1 |
| OP15 HTP internal weight bandwidth | 54 GB/s |
| direct DMA USB bandwidth | 465 MB/s |
| direct DMA fixed RPC floor | 0.235 ms |
| Q4_K GGUF storage | 144 bytes per 256 weights |
| Q6_K GGUF storage | 210 bytes per 256 weights |
| Q6_K HTP resident layout | 320 bytes per 256 weights |

The HTP Q6_K layout is a one-time, lossless backend repack of the source Q6_K
blocks. It occupies 10 bits per weight instead of the GGUF representation's
6.5625 bits per weight. It is the same logical model weight, but not the same
in-memory byte layout.

The screen requires at least 10% modeled median improvement. Every new row
still requires exact-shape correctness, p50, p90, USB-queue, and full-graph
validation before runtime promotion.

## Operator table

`M` is the physical row count in the current llama.cpp graph. `Mh` is the
number of rows that actually request logits.

| operator | legal split | CPU plus OP15 trial policy | backend | status |
| --- | --- | --- | --- | --- |
| token embedding | token rows | CPU for all `M` | CPU | lookup is too small; HTP Q4_K unavailable |
| Q projection | output rows | CPU for all `M` | CPU | Q4_K blocks HTP; Adreno is shadow-only |
| K projection | output rows | CPU for all `M` | CPU | Q4_K blocks HTP |
| V projection | output rows | CPU for all `M` | CPU | Q6_K is supported, but this narrow matrix did not pass the 10% screen alone |
| attention output | output or reduction width | CPU for all `M` | CPU | Q4_K blocks HTP |
| attention core | GQA groups or KV context | CPU at `C=8192`; no deployed phone range | CPU | prior CPU/phone component study kept CPU faster |
| FFN gate and up | common intermediate width | CPU for all `M` | CPU | both weights are Q4_K |
| FFN down | reduction width `Kp` | shortlist the table below for `M=1..512` | HTP | M=1 shape correctness passed; latency is still modeled |
| vocabulary head | vocabulary rows plus compact local top-k | shortlist the table below for `Mh=1..8`; otherwise CPU | HTP | Q6_K kernel qualified at K=3840; K=5120 end-to-end head is unmeasured |
| normalization, RoPE, residuals | rows or channels | CPU; fuse only when no extra boundary is added | CPU | insufficient work per USB byte |
| sampling | candidate rows | CPU exact candidate merge and sampler | CPU | phone route must obey sampler semantics |

The Adreno Q4_K rows are not promoted. GGML has Q4_K OpenCL machinery, but the
exact Qwen shapes, recurring queue cost, p90, and fused mixed-quant graph are
not qualified. Earlier complete phone-GPU FFN tests were slower than HTP and
desktop CPU controls.

## Q6_K down-projection shortlist

The phone receives `M x Kp` F16 intermediate values and returns `M x 5120`
F16 partial residual values. The desktop computes the remaining reduction
columns concurrently.

```text
wire bytes   = 2 * M * (Kp + 5120)
phone FLOPs  = 2 * M * Kp * 5120
resident     = 1.25 * Kp * 5120 bytes per layer
Tsplit       = max(Tcpu(K-Kp), Tusb + Thtp(Kp)) + Tsum
```

| physical M | screening Kp | phone fraction | wire | phone weights per layer | modeled down-op gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7,936 | 45.6% | 25.5 KiB | 48.4 MiB | 45.4% |
| 2 | 6,144 | 35.3% | 44.0 KiB | 37.5 MiB | 35.1% |
| 4 | 3,840 | 22.1% | 70.0 KiB | 23.4 MiB | 21.8% |
| 8 | 4,096 | 23.5% | 144 KiB | 25.0 MiB | 22.8% |
| 16 | 4,096 | 23.5% | 288 KiB | 25.0 MiB | 23.4% |
| 32 | 4,352 | 25.0% | 592 KiB | 26.6 MiB | 23.9% |
| 64 | 4,352 | 25.0% | 1,184 KiB | 26.6 MiB | 24.7% |
| 128 | 4,352 | 25.0% | 2,368 KiB | 26.6 MiB | 24.9% |
| 256 | 4,352 | 25.0% | 4,736 KiB | 26.6 MiB | 24.9% |
| 512 | 4,352 | 25.0% | 9,472 KiB | 26.6 MiB | 24.9% |

Keeping the largest M=1 shard for all 40 layers needs about 1,937.5 MiB of
HTP virtual mapping. It fits one approximately 3,200 MiB HTP mapping with
substantial room for buffers. Runtime may select any smaller prefix of the
prepared 7,936-column shard.

The physical OP15 check executed:

```text
Q6_K [7936,5120] x [7936,1]
HTP status 0, CPU status 0
source-byte mismatches 0
NMSE 1.61403341e-5
finite output true
```

This validates the M=1 matrix shape and the lossless HTP repack. It does not
measure recurring latency or the complete `gate -> up -> SwiGLU -> split down`
island.

## Q6_K vocabulary-head shortlist

The useful boundary is independent of phone vocabulary rows when the worker
returns local top-32 `(token, score)` pairs:

```text
wire bytes   = Mh * (2 * 5120 + 32 * 8) = 10496 * Mh
phone FLOPs  = 2 * Mh * Np * 5120
resident     = 1.25 * Np * 5120 bytes
```

| logits rows Mh | screening phone rows Np | phone fraction | wire | resident HTP weights | modeled head gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 79,104 | 52.1% | 10.2 KiB | 482.8 MiB | 51.9% |
| 2 | 62,464 | 41.1% | 20.5 KiB | 381.2 MiB | 41.1% |
| 3 | 48,384 | 31.8% | 30.8 KiB | 295.3 MiB | 31.8% |
| 4 | 39,424 | 25.9% | 41.0 KiB | 240.6 MiB | 25.9% |
| 5 | 38,656 | 25.4% | 51.2 KiB | 235.9 MiB | 25.4% |
| 6 | 38,656 | 25.4% | 61.5 KiB | 235.9 MiB | 25.4% |
| 7 | 38,656 | 25.4% | 71.8 KiB | 235.9 MiB | 25.4% |
| 8 | 38,912 | 25.6% | 82.0 KiB | 237.5 MiB | 25.5% |

These values come from a smooth roofline balance, so benchmark the proposed
point and its neighboring HTP shard boundaries. Do not encode the exact row
count as a runtime constant before that sweep.

Head offload is disabled when the request needs full logits or when grammar,
logit bias, repetition penalties, suppress-token rules, or another sampler
can promote a token omitted from the phone's compact candidate set. The CPU
must merge candidates, exactly rescore when required, and perform sampling.

## What the general solver learned

The existing v1 solver correctly found that unsupported Q4_K HTP paths must
stay blocked and that V alone is too small. This trial also exposed three
schema gaps that must be fixed before Qwen can be generated by one command:

1. a fused FFN needs independent gate/up and down quantization fields;
2. device-specific resident layouts must distinguish Q6_K GGUF storage from
   the larger HTP repack;
3. vocabulary matmul needs a compact top-k output contract instead of assuming
   that every phone logit crosses USB.

For this report, Q4_K/Q6_K storage and direct-DMA transport were supplied as
an in-memory screening overlay. The Q6_K down and head rows were then solved
with the explicit equations above. The checked-in `current_profile.json` was
not changed, so the existing Gemma examples retain their previous behavior.

## Next physical test

The narrowest useful implementation order is:

1. run CPU and OP15 exact-shape sweeps for Q6_K down at the listed Kp values;
2. integrate one FFN's down split with Q4_K gate/up remaining on CPU and check
   p50, p90, numerical error, and exposed join wait;
3. load the exact Qwen Q6_K output tensor into the HTP head worker, sweep the
   proposed vocabulary rows, and validate compact top-k semantics;
4. only after those pass, integrate the selected route into continuous-batch
   llama-server and run the original BurstGPT lengths.

The phone's ADB interface disappeared after the standalone Q6_K correctness
check while the phone remained enumerated on USB. No USB reset or gadget
reconfiguration was attempted because another remote workspace was active.
