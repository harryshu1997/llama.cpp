# RTX 4060 Ti plus OP15 kernel-energy acquisition

This directory contains the repeatable acquisition and materialization tools
for the first S42 physical placement profile. The paid energy boundary is the
sum of the desktop CPU package, RTX board, and whole connected phone when all
three domains are available. Isolated kernel rows retain only the domains that
were synchronously observed; a later full-model validation must qualify any
composition before enforcement.

## Measurements

The campaign collects three repetitions of:

- connected idle power for the CPU package, RTX board, and whole phone;
- Gemma4 dense Q4_0 FFN buckets on the desktop CPU and CUDA;
- the 9,664-column HTP suffix at M=1, 8, 32, and 128;
- a correctness-qualified 512-column Adreno shard using native Q4_0 at M=1
  and persistent F16 xmem at M=16, 32, and 128;
- pinned CPU RAM to CUDA VRAM transfers in both directions;
- FunctionFS DMA-BUF transfers in both directions and duplex mode;
- HTP packing, Adreno upload, Q4-to-F16 reconstruction, and complete cold
  Adreno F16+xmem preparation;
- CUDA model load for Gemma4-12B Q8_0 and Qwen3-14B Q4_K_M; and
- directed Gemma-to-Qwen and Qwen-to-Gemma CUDA residency switches.

Every measurement retains its raw power samples and paid-window markers. The
materialized profile records SHA-256 evidence IDs, median and maximum costs,
and leave-one-repetition-out error. Runtime certification uses the maximum,
not the median. Invalid, interrupted, wrong-binary, and warm-idle captures have
`.invalid-` in their names and are excluded by the materializer.

## Tools

- `measure_desktop.py` samples package RAPL and NVML against one paid window.
- `run_phone_case.py` keeps the whole-phone power logger alive around an ADB
  or USB-gadget workload.
- `analyze_phone_case.py` integrates USB input plus battery discharge.
- `pcie_energy_bench.cu` measures pinned H2D, D2H, and duplex transfers.
- `model_load_probe.py` proves every model layer was placed on CUDA.
- `model_switch_probe.py` includes source teardown and target-ready time.
- `run_phone_kernel_campaign.sh`, `run_phone_prepare_campaign.sh`,
  `run_usb_energy_campaign.sh`, and `run_model_switch_campaign.sh` are
  restartable and skip completed result files.
- `materialize_profile.py` validates the three-repeat evidence set and emits
  `s42-kernel-energy-profile-v1`.

The raw physical files stay outside Git under the ignored `results/` tree.
Only the compact materialized profile and result report are checked in.

## Qualification boundary

The emitted kernel and transfer rows are measured only for their exact shape
and payload buckets. Directional link equations are derived over the measured
payload interval and remain non-enforceable outside it. Per-operator rows do
not become a certified runtime route until a held-out full-model run has at
most 10% latency and energy error for the same model, quantization, device,
residency, transport, thermal, and concurrency epoch.
