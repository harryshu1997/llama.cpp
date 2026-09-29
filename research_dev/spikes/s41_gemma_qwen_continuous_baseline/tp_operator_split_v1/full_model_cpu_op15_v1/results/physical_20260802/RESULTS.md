# Gemma 4 12B full-model CPU and OP15 result

Physical run date: 2026-08-02.

The best measured split puts transformer layers `[0,28)` on the OP15 Adreno
840 and layers `[28,48)` plus the output head on the i9-12900K CPU. It lowers
median request latency by 21.1% and increases aggregate decode throughput by
28.4% relative to running all 48 layers on the desktop CPU.

## Paid result

Both routes used the same Gemma 4 12B IT Q8_0 file, 8 desktop CPU threads, a
28-token prompt, greedy decoding, 32 generated tokens, one warmup, and five
paid requests.

| Route | Placement | Request p50 | Request p90 | Prefill p50 | Decode p50 | Decode rate | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU only | CPU layers 0-47 + head | 14,008.981 ms | 14,083.442 ms | 751.454 ms | 13,262.382 ms | 2.338 tok/s | 1.000x |
| CPU + OP15 K28 | Adreno layers 0-27; CPU layers 28-47 + head | 11,050.963 ms | 11,094.533 ms | 719.713 ms | 10,330.435 ms | 3.003 tok/s | 1.268x |

The split request consists of 4,295.592 ms in the phone stage and 6,754.652
ms in the desktop tail at the median. This is a sequential layer split, not
parallel execution of the same layers. It wins because the phone replaces an
estimated 7,254 ms CPU prefix with a 4,296 ms phone stage, including transport.

Model loading, phone kernel setup, and connection setup are outside paid
request latency. This is the prepared, resident-weight mode required by the
planner. The RTX 4060 Ti identified the desktop, but stayed idle because both
requested routes use the desktop CPU rather than the desktop GPU.

## Prefix calibration

The calibration used the same 28-token prompt, 8 generated tokens, one
warmup, and two paid requests. The K28 improvement over K27 is within normal
run-to-run noise, but K28 is the deepest prefix that fits the reported OpenCL
budget. K29 is estimated to exceed it.

| Phone backend | Prefix K | Request p50 | Speedup vs CPU |
| --- | ---: | ---: | ---: |
| CPU control | 0 | 3,762.511 ms | 1.000x |
| HTP F16 | 2 | 3,881.205 ms | 0.969x |
| HTP F16 | 6 | 3,951.910 ms | 0.952x |
| HTP F16 | 8 | 3,970.757 ms | 0.948x |
| Adreno Q8 | 2 | 3,809.528 ms | 0.988x |
| Adreno Q8 | 6 | 3,786.412 ms | 0.994x |
| Adreno Q8 | 8 | 3,777.994 ms | 0.996x |
| Adreno Q8 | 12 | 3,590.372 ms | 1.048x |
| Adreno Q8 | 16 | 3,444.203 ms | 1.092x |
| Adreno Q8 | 20 | 3,324.992 ms | 1.132x |
| Adreno Q8 | 24 | 3,184.363 ms | 1.182x |
| Adreno Q8 | 27 | 3,096.646 ms | 1.215x |
| Adreno Q8 | 28 | 3,091.504 ms | 1.217x |

HTP K10 and K12 were excluded using prior physical evidence: the F16 prefix
loads, but the DSP aborts when a single HTP weight allocation crosses its
approximately 4 GiB limit. HTP also changes prefix precision, so it is not a
matched-quantization route.

## Placement and memory evidence

- Host CPU placement certificate: 112,128 compute nodes on CPU, zero missing
  buffers, status `SCHEDULED_PLACEMENT_OK`.
- Phone placement certificate: 166,080 compute nodes on OpenCL and 192
  `GET_ROWS` embedding nodes on CPU, zero missing buffers, status
  `SCHEDULED_PLACEMENT_OK`.
- Phone allocations: 6,432.49 MiB OpenCL model weights, 1,020.00 MiB mapped
  embedding, 50 MiB OpenCL KV, and 28.50 MiB OpenCL compute storage. The
  backend reported 6,532 MiB free before allocation, so K28 has little safety
  margin. K27 is the safer deployment cut.
- Desktop peak RSS was 11.995 GiB for CPU only and 11.959 GiB for the split.
  The current host still maps the complete GGUF in both routes. Desktop swap
  counters did not change and both commands had zero major faults.
- During the paid phone window, Android reported 19,060 swap-in pages and
  1,485 swap-out pages at a 4 KiB page size. Net `SwapFree` increased by
  64,780 KiB. Across phone load, execution, and teardown, `SwapFree` decreased
  by 36,540 KiB. There is no sign of progressive paid-window thrashing, but
  this route does not satisfy a literal zero phone zram-I/O criterion.

## Transport and boundary volume

This run uses the existing persistent LayerSplit TCP connection through
`adb forward` over USB. It is not the tuned NCM or direct AOA path.

The cut tensor has 3,840 FP32 elements, or 15,360 bytes per evaluated token.
One request evaluates 28 prompt tokens and 31 decode steps, so the phone sends
906,240 bytes (0.864 MiB) of cut activations to the host. Token IDs in the
other direction are negligible. Weights and KV state remain resident on their
owning device.

## Output agreement

The desktop and phone model files have the identical SHA-256:

```text
7b56cbd0e0d96d5c8d7df9c21b39d67264e58cbabf7a96d4681eff8b3d492848
```

Each route is deterministic across its five paid requests. The two routes
share the first 9 generated token IDs and diverge at token 10; 45 of 160 token
positions match in total. This is consistent with different CPU and OpenCL
floating-point execution changing a close greedy decision, not with a model
file mismatch. The speed result therefore compares identical shapes and
token counts, but it is not a bit-identical correctness result.

## Deferred route

The all-phone route remains deferred. One OP15 cannot safely hold the complete
11.80 GiB Q8 model plus KV, compute buffers, and Android within its currently
available memory. A multi-phone full-model run needs another phone and a
separate placement plan.

Machine-readable results are in `ANALYSIS.json` and `CALIBRATION.json`.
