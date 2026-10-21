# OnePlus 15 FunctionFS / DMA-BUF kernel: build recipe and provenance

The phone-side transport used by the paper's OnePlus 15 arms runs on a
RAM-booted candidate kernel (never flashed). This directory keeps everything
needed to rebuild it; the 30 GB vendor source trees were removed from
~/Documents on 2026-10-20 because they are re-fetchable.

| Item | Where |
|---|---|
| Vendor source | https://github.com/OnePlusOSS/android_kernel_modules_and_devicetree_oneplus_sm8850 (tree checkout 7ec148b0, kernel common 227664cbe007bbad49aa74259179ac99608a2113) |
| Patches applied | research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/ffs_dmabuf_transport_v1/functionfs_dmabuf_fixes.patch, .../aoa_kernel_rx_v1/android_f_accessory_rx64k.patch, research_dev/scheduler/campaigns/burstgpt/reports/20260913-dmabuf-cancellation-repair/fence_lifetime.patch |
| Build plan, scripts, logs | cancel-20260913/ (BUILD_PLAN.md, build.sh, verify_build*.sh, BUILD.log, VERIFY*.log, candidate and reference boot notes) |
| Kernel config of the earlier rx-v1 build | s41-rx-v1/kernel.config, gki-info.txt, logs/ |
| Booted image (desktop) | /mnt/storage/s42-dmabuf-cancel-20260914-v1-gBJdFx/candidate-boot.img, SHA-256 f13c7c033ce74f32ea4aa6349cb531704407708f5fb35756a6c361ffb48322a3 |
| RAM-boot script (desktop) | /mnt/storage/s43-two-phone-eval-20260925/rig/op15_ramboot_20261014.sh (user runs it) |
| Built images and stock backups (this host) | /mnt/data_h16t/zs89458/xband-runs/kernel-op15/ (s41 dist: control, ffs_dmabuf_fixed_v1, rx64k boot images; stock images; cancel-20260913 verification ELF/BTF and transport candidates) |

The candidate kernel has no WiFi (cfg80211 missing); WiFi tests need the stock
kernel (adb reboot restores it because the candidate is RAM-booted).
