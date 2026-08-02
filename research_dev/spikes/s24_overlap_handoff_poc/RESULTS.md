# S24 Fixed-Diamond Overlap and Handoff Results

Status: `BENEFIT_GATE_FAIL`.

The selected final verdict is:

- BENEFIT_GATE_FAIL

The allowed verdict set was:

- FIXED_DIAMOND_POC_PASS
- MECHANICS_PASS_NUMERICALLY_UNCERTIFIED
- BENEFIT_GATE_FAIL
- PHYSICAL_EXECUTION_BLOCKED

## Frozen inputs

- Desktop model:
  /mnt/storage/s21_models/gemma-4-12b-it-Q8_0.gguf
- Model SHA256:
  f20e7ff1be28c283eeeb18fc895733791c56a5851d5cd3fe9691b7f7d12afa72
- Desktop binary:
  build-s21-cuda/bin/llama-layersplit
- Binary SHA256:
  aefcd595b2f51ca4c8d74113cead66d28a0cc2ac56c1a5fbdf4b98202fdf3523
- V3 source SHA256:
  2120b6a0e2af73a1f02d0f72879f062518b3dbfaf3da1264c3744da63b958a71
- CUDA runtime:
  /mnt/storage/s21_deps/cuda-13.2.1/lib64

## Synchronization record

The S23 tree was copied from:

  /home/myid/zs89458/Documents/llama.cpp-release/research_dev/spikes/s23_dense_trace_runtime/

to:

  /home/zhihao/llama.cpp-release/research_dev/spikes/s23_dense_trace_runtime/

The normalized source and destination per-file manifest digest is:

  7532d59a0e61a2a3db78b4219fdd008b9b4735ac45b40701e05208b1841da81a

The complete authoritative S22 Python artifact set was staged from the A6000
host. Its normalized source, staging, and local destination manifest digest is:

  379e67ce74cf567585b1e0afba8d352c95634f5f66b082a136c44213dd1a7ae1

S22 RESULTS.md was synchronized separately with SHA256:

  dc09b961d850d7f2c512a3236efa4343625d016650b3e5cd3f3e94aaca635076

No historical S22 or S23 result was rewritten.

## Baseline tests

- S22 authoritative Python suite: 42 tests passed.
- S23 runtime-support tests: 4 tests passed.
- S23 dense-trace tests: pending the exact external normalized BurstGPT source
  pair referenced by the synchronized generator.
- S24 fixed-diamond suite: 49 tests passed, including route cleanup failure
  injection, three dispatch reasons, mixed-source and shared-tail convergence,
  physical-certificate validation, numerical comparison, and benefit gates.

## Physical evidence

### Desktop CUDA capacity

The three fixed CUDA workers were resident simultaneously in
`results/cp1_cuda_capacity_resident/`:

| Worker | Range | Process GPU memory | PSS | CUDA model buffer |
| --- | --- | ---: | ---: | ---: |
| cuda-prefix | [0,8) | 2134 MiB | 4469333 KiB | 1835.31 MiB |
| cuda-mid | [8,16) | 2134 MiB | 4465621 KiB | 1835.31 MiB |
| cuda-tail | [16,48) | 9928 MiB | 4478605 KiB | 8397.10 MiB |

The board used 298 MiB before loading, 14512 MiB with all three workers
resident, and 298 MiB after their clean stop. The resident free-memory margin
was 1436 MiB. The three CUDA model buffers sum to 12067.72 MiB, matching the
disjoint full-model CUDA allocation rather than three duplicate full-model
allocations. The HELLO records certify exact ranges [0,8), [8,16), and [16,48)
with stream capacities 4, 4, and 8.

The dynamically linked CUDA 13.2 libraries are recorded in
`results/cp1_cuda_capacity_resident/cuda-runtime-sha256.txt`:

- libcudart.so.13.2.75:
  c7eaa64c99c20484b7401352aff99a8c5ab947abaab3a9bbb70142312da1fe18
- libcublas.so.13.4.0.1:
  5589b43e2aae4790ccb2abd22733eae0b6d4fe9503dcc4dbfe219fd857ddac41
- libcublasLt.so.13.4.0.1:
  ac76bbb85d74313bba0d71029c71389616205f4018c8ff31d57c38f6e61100bd

### Desktop CUDA batch knees

The frozen 95-percent-of-peak rule produced these physical knees in
`results/cp3_cuda_knees/`:

| Worker | Median B1/B2/B4/B8 RPC time | Selected knee |
| --- | --- | ---: |
| cuda-prefix | 7737/7949/8243 us | B4 |
| cuda-mid | 7860/8180/8598 us | B4 |
| cuda-tail | 34972/36097/37594/41183 us | B8 |

The knee sessions stopped cleanly with zero missing buffers. Their active
placement certificates assign compute to CUDA0, with only the declared
CUDA_Host GET_ROWS metadata seam.

### A6000 phone provisioning

Both phones were provisioned over independently identified USB transports:
OP12 on `usb:6-2` and OP15 on `usb:8-3`. The existing OP12 F16 [0,8) shard was
verified as SHA256 `a74845294bc1a3cba6ac26c747615d64d9127c904aae04c43c7ea2ebc1ea25c8`.
The OP15 F16 [8,16) shard was created from the frozen full model, is 3637819776
bytes, and matched SHA256
`37959e304f5f251171f2e74a7c3bdebdd6279eb99a9136806147da15f15e9394`
before and after transfer.

The current AArch64 `llama-layersplit` binary matched SHA256
`01aab37abaf3af658edb1856f333636d7b2f2ac14661b7b6cc953f4b6ac8600f`
on both phones. The explicit runtime manifest includes the linked ggml
libraries, OpenCL and Hexagon bridges, both v75/v81 HTP skels, and the Android
C++ runtime. All host and device hashes are frozen under
`results/a6000_provision/`; its `SHA256SUMS.txt` validates locally and in the
desktop checkout. The provisioning record states `wifi_weight_bytes=0` and
`a6000_gpu_use=FORBIDDEN_AND_NOT_INVOKED`.

No phone worker was started during provisioning.

### CP4 physical fixed diamond

The measured phone batch knees were B4 for both OP12 [0,8) and OP15 [8,16).
All 11 CP4 sessions passed placement, finite-output, lineage, position, missing
buffer, KV cleanup, and software-lease gates. R2 completed four requests with
physical B4 on OP12, OP15, and the CUDA tail. The shared OP15 queue formed
three mixed-source batches from R1 and R2. The shared CUDA tail formed four
mixed-route batches from R0, R1, and R2. There were no cohort or global
barriers and no SLO misses.

The validator selected `CP4_PHYSICAL_PASS`. Exact reports, worker logs,
activations, and validation records are under `results/cp4_fixed_diamond/`.

## Benefit evidence

CP5 executed C0, C1, C2, and C3 three times each in a rotated order with six
equal-work requests per run. Every run completed all six requests with no
rejections. The frozen gates produced:

| Gate | Control comparison | Result |
| --- | --- | --- |
| Shared convergence | OP15 mean batch 2.0 -> 3.0; median makespan 3.728 -> 2.450 s | PASS |
| Priority-0 regression | TTFT +30.0%; latency +32.5%; limit +5% | FAIL |
| CUDA work relief | 2.456 -> 3.649 s summed island compute, +48.6% | FAIL |
| GPU-board reporting | C0 35.18 J; C3 163.26 J median | PASS (reporting only) |

C3 also introduced two SLO misses. The shared queue demonstrates the intended
batch-convergence mechanism, but the fixed route policy is too slow and causes
more CUDA work rather than relief. The selected verdict is therefore
`BENEFIT_GATE_FAIL`. CP6 was not run.

## Numerical evidence

Same-route R0, R1, and R2 repeats were exact, and the phone routes produced the
same greedy tokens as their CUDA controls for this synthetic screen. However,
individual R1 boundary rows exceeded the frozen relative-L2 limit despite an
aggregate relative L2 of 0.00403. R2 aggregate relative L2 was 0.02202. Both
phone routes therefore fail the boundary gate and remain
`NUMERICALLY_UNCERTIFIED`. This result is not an output-quality certificate.

## Energy scope

RTX 4060 Ti GPU_BOARD energy used the existing NVML method. C3 median board
energy was 4.64x C0 for the equal-work screen. This is selected-GPU evidence,
not total-system energy. Phone energy, network energy, A6000 host energy, and
total-system energy remain UNKNOWN.
