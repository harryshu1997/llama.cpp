# OP15 restored after the recorded kernel crash

Result: KERNEL_RESTORED, verified at 2026-09-13 18:59 EDT. The user authorized
booting OP15 into the correct research kernel after the DMA-BUF cancellation
test crash. No cancellation test, inference, trace, or production edit ran.

## Crash evidence correction

The user's evidence was located at:
`/home/myid/zs89458/Documents/moe-resident-routing-a6000/results/direct_usb_fast_retest_20260913/KERNEL_CRASH_EVIDENCE.txt`.
It contains `kernel crash Minidump [2026-09-13 18:46:40]` and persistent reset
records. A byte-identical copy is preserved in this report's physical artifacts.
SHA-256: `2f0310138578e182872e2d27b8d8e7249e90ff815d803174b111b618bd355f9a`.

This supersedes the earlier diagnosis that the reboot trigger was unknown from
generic bootreason and empty pstore checks. Empty pstore did not exclude a
vendor minidump. The reported cancellation sequence makes that path the likely
trigger; restoration does not repair or qualify it. The original crash records
and failed test were not changed or rerun.

## Restoration and verification

- Exact device: `3C15AU002CL00000`, CPH2749, desktop ADB port 5037.
- Image: `/home/zhihao/s41-ffs-dmabuf-fixed-v1/boot.img`, 67,108,864 bytes.
- Image SHA-256:
  `26e8d41808b10b70264d958582bb6f6c9fba634c34275e0bc28fc820a6d3fb8d`.
- Fastboot SHA-256:
  `76dde33fee8b1fd00bcaf2e7f94ddef6407f0beb5bc3a98a3d4127307af23f3a`.
- Method: existing recorder and temporary `fastboot boot`, no partition flash.
- Invocation through verification: 60.183 s; fastboot command: 1.941 s wall time.
- Verified kernel: `6.12.23-android16-5-o-g227664cbe007-4k`.
- Verified kernel BTF SHA-256:
  `f3afcf985b24de3eb5a1d5453bf4ffd95963b806ce99a59430c5a8bf7d201d17`.
- New boot ID: `d03a2ef4-6392-4de3-a8a7-1f2376eb9b00`.
- Android boot complete; slot `_b` unchanged; USB `ptp,adb` at 5,000 Mbps.
- Battery 100%, temperature 26.0 C; no research phone workers observed afterward.

No conflicting phone workers or host inference jobs were observed beforehand.
The image and fastboot hashes were checked; the recorder checked the image
again immediately before boot and verified the fastboot serial and active slot.
No wipe, unlock, slot change, desktop reboot, GDM stop, or unrelated process
termination occurred. Original GDM PID 6350 and ADB PID 9886 remain running.

The boot remains temporary: a reboot returns to the installed stock image.
The cancellation path remains unvalidated and was not exercised. This result
is kernel restoration only, not proof that the crash defect has been fixed.
Maintenance energy was not measured.

## Evidence

Remote: `/mnt/storage/s42-phone-kernel-restore-20260913-v2-p8tJSH/`.
The [physical](physical) copy preserves the original recorder, invocation,
57 command receipts, crash evidence, terminal result, and post-boot checks.
`ARTIFACTS.sha256` covers all evidence files except itself.

The unmodified archived recorder retains its original dev3 purpose string in
`BOOT_PLAN.json`; `RESTORE_INVOCATION.json` records this restoration-only scope
and the fresh output-root override. No dev3 workload was launched.

[RESTORE_RESULT.json](physical/run/RESTORE_RESULT.json) SHA-256:
`99b272779b91a904e942599ef79cf7b06726558c8088431e69bc50cf9248924f`.

Only this report, its evidence copy, and the project log changed. No software
test suite, commit, push, scheduler deployment, or crash reproduction.
