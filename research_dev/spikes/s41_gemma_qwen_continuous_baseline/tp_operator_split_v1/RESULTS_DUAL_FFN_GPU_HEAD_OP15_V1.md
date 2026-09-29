# OP15 dual-resident FFN and LM-head result

Date: 2026-08-04

## Result

The 3328 MiB HTP mapping ceiling was not increased. Instead, the maximum
FFN split stays resident in the HTP0 FastRPC address space and the exact
LM-head suffix stays resident in the Adreno address space.

| resident operator | backend | model type | shape or range | resident bytes |
| --- | --- | --- | --- | ---: |
| FFN suffix, all 48 layers | HTP0 | Q4_0 | columns `[4224, 15360)` | 3303.29 MiB |
| LM-head suffix | Adreno 840 | Q6_K | rows `[216064, 262144)` | 138.43 MiB |
| combined | separate address spaces | exact GGUF types | 11136 FFN columns + 46080 vocabulary rows | 3441.72 MiB |

Both workers allocated their weights, warmed their graphs, and remained
ready concurrently. Their model-derived hashes were `4ada42f7ae1d721e` for
the FFN split and `99e12d32c01ddcb9` for the LM-head split.

This preserves the quantization in
`gemma-4-12B-it-Q4_0.gguf`. The FFN tensors are Q4_0 and the tied token
embedding/output tensor is the GGUF's single Q6_K tensor. There is no runtime
requantization.

## Why the second HTP session is not used for this head

An HTP1 FastRPC session can be opened while HTP0 holds the 3303.29 MiB FFN
buffer, so the remaining problem is not aggregate phone RAM or the HTP0
mapping budget. The current Hexagon `MUL_MAT` path supports Q4_0, Q4_1,
Q8_0, IQ4_NL, MXFP4, F16, and F32 weights, but not Q6_K.

The direct negative test opened both HTP0 and HTP1 sessions successfully,
then rejected the first 32768-row Q6_K shard with
`backend does not support MUL_MAT for 32768 rows`. This distinguishes the
type limitation from a session-creation or mapping failure.

The worker now shards HTP-compatible heads at 32768 output rows, working
around the backend's per-matmul LM-head row guard. That does not make the
native Gemma 4 Q6_K head executable on HTP. Requantizing it to Q4_0 would
create a different weight representation, so the exact path uses Adreno
instead.

## Target full-model result

The repeated test ran on the target `zhihao-Z690-C-ac` desktop with an
Intel Core i9-12900K, an RTX 4060 Ti 16 GiB, and the USB-connected OP15
(`CPH2749`). The cold Gemma model intentionally stayed on the desktop CPU
with `-ngl 0`; the GPU was present but was not in this model's execution
path. No CPU or GPU frequency override was applied.

Each mode used the same exact GGUF (SHA-256
`494518c2262a26e2a607af0e40bca11c4de5a0b108e7c21308906dbbfb1c6f8c`),
eight pinned P-core hardware threads, `--no-repack`, a six-token prompt,
16 generated tokens, one warmup, and five paid requests. Table entries are
paid-sample medians.

| mode | prefill | decode | request wall | speedup | latency reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| desktop CPU only | 273.383 ms | 3576.144 ms | 3849.100 ms | 1.000x | 0.00% |
| CPU + HTP0 FFN | 363.613 ms | 2425.762 ms | 2797.248 ms | 1.376x | 27.33% |
| CPU + HTP0 FFN + Adreno head | 392.670 ms | 2373.045 ms | 2765.999 ms | 1.392x | 28.14% |

The dual route is 1.011x faster than FFN-only, or 1.12% lower latency. The
larger result comes from FFN decode overlap. Prefill is slower than CPU-only,
but the decode reduction is large enough to lower the complete request wall
time.

## Operator breakdown

The FFN worker completed all 4608 expected calls: 4320 one-token decode calls
and 288 six-token prefill calls. It returned the same model-derived hash as
the resident allocation check.

| FFN component | median |
| --- | ---: |
| complete decode RPC | 1.675 ms |
| phone HTP decode compute | 1.407 ms |
| desktop FFN prefix branch | 1.161 ms |
| desktop wait after its branch | 0.542 ms |
| complete prefill RPC | 4.645 ms |
| phone HTP prefill compute | 3.333 ms |

The LM-head split ran on 90 autoregressive decode evaluations. The first
token's prefill head stayed on the desktop. The phone RPC remained below the
desktop prefix branch, so its work was completely hidden at the median:

| LM-head decode component | median |
| --- | ---: |
| desktop prefix branch | 23.085 ms |
| phone Adreno compute | 16.140 ms |
| phone top-k reduction | 0.280 ms |
| complete phone RPC | 19.331 ms |
| desktop wait after its branch | 0.001 ms |
| exact desktop candidate rescore | 0.275 ms |

The measured condition is therefore satisfied:

```text
max(T_cpu_prefix, T_transport + T_phone_suffix + T_phone_topk)
    + T_exact_rescore < T_cpu_full_head
```

## Transport

The phone temporarily booted the fixed FunctionFS kernel image with SHA-256
`26e8d41808b10b70264d958582bb6f6c9fba634c34275e0bc28fc820a6d3fb8d`.
No partition was flashed. The FFN path used direct FunctionFS DMA-BUF buffers:

```text
desktop libusb -> FunctionFS DMA-BUF -> HTP input
HTP output -> FunctionFS DMA-BUF -> desktop libusb
```

The paid dual run moved 46,448,640 bytes in each direction over 4608 FFN
calls. USB transfer p50 was 1.655 ms and p90 was 1.956 ms; host-to-phone p50
was 0.046 ms and phone-to-host p50 was 1.608 ms.

The temporary GKI cannot load this phone build's current Wi-Fi modules, so a
composite USB gadget carried the small LM-head RPC over NCM while FunctionFS
remained interface 0. The desktop bound interfaces 1 and 2 to `cdc_ncm` at
USB 5 Gb/s. A temporary IPv6 source rule routed `fe80::2` through Android
table 1033. The restore script removed the rule and both custom functions.

After acquisition the phone rebooted to the installed stock kernel
`6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k`. Normal SuperSpeed USB was
restored and no host or phone worker process remained.

## Output and placement checks

All five paid requests were internally stable in each mode. FFN-only and the
dual route produced the same 16-token sequence. CPU-only produced a different
sequence because the HTP split changes numerical accumulation, but both
sequences detokenize to a response identifying Paris. This is a performance
result, not a claim of token-exact CPU equivalence.

The dual placement certificate reported run status 0, zero compute nodes with
missing buffers, 4608 FFN calls, and 90 LM-head calls. The maximum exact
candidate rescore error was 0.00527.

## Implementation points

- `examples/layersplit/lm-head-split-worker.cpp` keeps the native Q6_K head
  suffix in an Adreno-native buffer and returns a reduced top-k candidate set.
- `ffs_dmabuf_transport_v1/phone_ffn_session.sh` can bind FunctionFS and NCM
  in one composite gadget, start both resident workers, install the scoped
  IPv6 rule, and restore the normal gadget on exit.
- `ffs_dmabuf_transport_v1/ncm_ipv6_proxy.c` relays the phone's scoped IPv6
  NCM socket to the IPv4 LM-head worker.
- `ffs_dmabuf_transport_v1/ncm_ipv6_relay.py` presents a local IPv4 endpoint
  to a benchmark binary that uses one host address for both operator clients.
