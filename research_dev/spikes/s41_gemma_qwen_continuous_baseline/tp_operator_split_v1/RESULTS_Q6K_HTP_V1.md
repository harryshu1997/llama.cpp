# Q6_K HTP kernel result

Date: 2026-08-05

## Verdict

The Q6_K x F32 `MUL_MAT` path passes on the OnePlus 12 HTP v75 and the
OnePlus 15 HTP v81. The prior Gemma LM-head blocker is removed at the
operator and real-worker levels on both phones.

## Implementation

The host expands each Q6_K block once while loading it into the HTP buffer.
A 32 x 32 weight tile contains:

- 1024 signed int8 quants;
- 32 original fp16 block scales and 64 original int8 subscales, so a backend
  get restores the source Q6_K bytes exactly;
- 64 fp16 precombined weight scales used by the DSP dot kernel.

The resulting tile is 1280 bytes. Q6_K uses the existing dynamic F32 to Q8_0
activation quantization and an HVX dot path. It is intentionally excluded
from HMX selection. Tiled and flat 32 x 1 and 32 x 2 kernels are present, as
are the existing normal, fused QKV, and fused FFN dispatch forms.
F16 graph activations are rejected because this HVX quantizer consumes F32;
the LM-head worker expands f16 RPC payloads to F32 before graph execution.

The real worker now marks each mutable activation tensor as a graph input and
marks its graph buffer for compute use. This is required for HTP cache
maintenance when the host replaces an activation between requests. The CPU
reference route also retains the default CPU buffer type instead of querying
accelerator-specific quantized buffer types.

The repack uses 10 bits per weight instead of the GGUF Q6_K representation's
6.5625 bits per weight. The 32768 x 3840 shard grows from 98.44 MiB to
150.00 MiB. The complete 46080-row Gemma suffix grows from 138.43 MiB to
210.94 MiB.

## Real HTP correctness and latency

The independent oracle initializes CPU and HTP tensors separately from the
same logical weight bytes. This avoids the existing backend graph-copy path,
which directly copies the host-visible internal tiled buffer and is not a
valid reference for any repacked HTP weight.

All comparisons use the `test-backend-ops` NMSE limit of 5e-4.

| case | shape | route | result |
| --- | --- | --- | --- |
| minimal decode | `[256,16] x [256,1]` | tiled | NMSE 4.41e-6, zero byte mismatches |
| minimal decode | `[256,16] x [256,1]` | flat | NMSE 1.32e-5, zero byte mismatches |
| row padding and batch | `[512,33] x [512,2]` | tiled | NMSE 1.94e-5, zero byte mismatches |
| mixed 32 x 2 and 32 x 1 | `[3840,4096] x [3840,3]` | tiled | NMSE 1.56e-5, zero byte mismatches |
| mixed 32 x 2 and 32 x 1 | `[512,33] x [512,3]` | flat | NMSE 1.97e-5, zero byte mismatches |
| model width and batch | `[3840,64] x [3840,8]` | tiled | NMSE 1.65e-5, zero byte mismatches |
| output-row edge | `[256,1] x [256,64]` | tiled | NMSE 1.44e-5, zero byte mismatches |
| full HTP shard | `[3840,32768] x [3840,1]` | tiled | NMSE 1.44e-5 to 1.63e-5, zero byte mismatches |
| OP15 model-width batch | `[3840,4096] x [3840,3]` | v81 tiled | NMSE 1.50e-5, zero byte mismatches |
| OP15 full HTP shard | `[3840,32768] x [3840,1]` | v81 tiled | NMSE 1.58e-5, zero byte mismatches |

Four OP12 full-shard DSP profiles were 2.422, 2.480, 2.510, and 2.414 ms. The
mean is 2.457 ms, equivalent to 102.4 GOP/s when one multiply and one add are
counted as two operations. A final OP12 rerun was 2.407 ms with NMSE 1.55e-5.

Four OP15 v81 profiles were 2.223, 2.234, 2.234, and 2.364 ms. The mean is
2.264 ms, equivalent to 111.2 GOP/s.

## Real Gemma LM-head worker

The worker loaded the exact Q6_K output suffix from the existing Gemma 4 12B
Q4_0 GGUF:

```text
K=3840
vocabulary rows=[216064,262144)
offloaded rows=46080
HTP shards=32768+13312
resident HTP weight buffer=210.95 MiB
model-derived hash=99e12d32c01ddcb9
```

The first large-worker test exposed stale activation reads: the worker had not
called `ggml_set_input` and had not marked the graph buffer for compute use.
Repeated outputs could settle over the first requests and differ across fresh
processes even though the standalone operator oracle was stable. Adding those
two existing HTP buffer-contract operations removed the drift. An explicit
post-graph synchronization was tested, found insufficient by itself, and
removed because the Hexagon graph call already drains its queue.

With graph caching disabled, the corrected worker produced bit-identical top
IDs and scores for 10/10 repeated A inputs. A fresh process reproduced the
same output. The changing-input sequence A/B/A produced distinct B output and
the final A reproduced the initial A exactly. All request and response hashes
passed.

The corrected CPU worker consumed the same GGUF suffix and f16 payloads as a
real-model reference. On OP12, A shared 32/32 top candidates with score NMSE
7.24e-6 and maximum absolute difference 0.00330; B shared 31/32 with score
NMSE 1.09e-6 and maximum absolute difference 0.000660. On OP15, the same
figures were 32/32, 6.30e-6, and 0.00280 for A, and 31/32, 8.08e-7, and
0.000477 for B. Visible ordering differences were limited to near-tied
candidates; the top-8 sets matched.

The final 10-request OP12 HTP run averaged 5.049 ms for two graph executions
and output copies, 0.643 ms for CPU top-k reduction, and 8.066 ms over the
local ADB-forward socket. The corresponding OP15 averages were 3.363, 0.243,
and 4.644 ms. All 10 OP15 responses were also bit-identical. These are phone
worker measurements, not an end-to-end phone plus desktop speedup claim; the
desktop GPU did not participate. The final Android worker SHA-256 is
`f565552147409959be4e2916dee5be64e0aa2d5ddc86bd9b39ccf3917a50f7e0`.

## Regression and build checks

- Existing Q4_0 `[3840,4096] x [3840,2]`: NMSE 2.19e-5 and zero byte
  mismatches.
- Existing Q8_0 `[3840,4096] x [3840,2]`: NMSE 2.19e-5 and zero byte
  mismatches.
- `test-backend-ops support` reports all 11 generated Q6_K `MUL_MAT` cases
  supported on HTP0.
- Real-model A/B/A HTP-versus-CPU comparison passes on OP12 and OP15.
- Android host code, HTP v75, and HTP v81 skeletons build successfully.
- `git diff --check` passes.

The normal full-shard scheduler selects the tiled route. Forcing the flat
route at 4096 output rows exceeds the 8 MiB VTCM budget; the supported small
flat cases pass.
