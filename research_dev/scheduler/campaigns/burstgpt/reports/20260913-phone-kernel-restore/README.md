# OP15 qualified kernel restored, 2026-09-13

Result: KERNEL_RESTORED. The user authorized restoration of the previously
verified OP15 boot image. Only that restoration and read-only verification ran;
no inference, FFN load, trace, baseline, or scheduler change was performed.

## Exact restoration

- Device: OP15 / CPH2749, serial `3C15AU002CL00000`, desktop ADB port 5037.
- Image: `/home/zhihao/s41-ffs-dmabuf-fixed-v1/boot.img`, 67,108,864 bytes.
- Image SHA-256:
  `26e8d41808b10b70264d958582bb6f6c9fba634c34275e0bc28fc820a6d3fb8d`.
- Fastboot SHA-256:
  `76dde33fee8b1fd00bcaf2e7f94ddef6407f0beb5bc3a98a3d4127307af23f3a`.
- Method: the previously successful `fastboot boot` RAM-only procedure.
- Active slot: `_b` before and after; exact serial and slot checked in fastboot.
- Partition flashes, wipes, unlocks, and slot changes: zero.

Both image and fastboot hashes matched the saved restoration evidence. The
image was checked again by the existing recorder, including immediately before
boot. No phone workers, host inference jobs, or conflicting task locks were
observed before the targeted reboot. Battery was 100% and the bootloader was
already unlocked.

The original September 10 recorder was reused byte-for-byte, with only its
output root overridden. Its original `BOOT_PLAN.json` purpose string mentions
the old dev3 retest; `RESTORE_INVOCATION.json` explicitly records this invocation's
narrower restoration-only authorization. No old retest was started.

## Physical verification

| Check | Observed result |
| --- | --- |
| Restore invocation through verification | 59.452 s |
| Fastboot image transfer and boot command | 1.927 s wall time; 1.925 s reported by fastboot |
| Kernel | `6.12.23-android16-5-o-g227664cbe007-4k` |
| Android boot completed | `1` |
| USB | `ptp,adb`, 5,000 Mbps on desktop port 2-2 |
| Phone battery / temperature after boot | 100% / 26.2 C |
| Kernel crash persistence | Root check found empty pstore |
| Research phone workers | None observed |

Kernel BTF SHA-256 matches the saved qualified image:
`f3afcf985b24de3eb5a1d5453bf4ffd95963b806ce99a59430c5a8bf7d201d17`.

Boot ID changed from `649a2d68-ef7b-4ced-9a79-27b5bcb3f5bd` to
`03bc9013-c7d5-4c34-84cf-c6cc358a8613`. GDM PID 6350 and the original ADB
daemon PID 9886 remain running. No desktop reboot, driver change, process stop,
USB reset recovery, or operation on another phone was performed. The initial
unprivileged pstore read was denied; the subsequent root read succeeded and is
preserved separately, not silently reported as the first read succeeding.

This restoration is temporary. A normal reboot returns to the installed stock
kernel. Future execution must recheck current kernel/boot and session identities;
this record is not a new FFN execution or transport qualification. Maintenance
energy was not measured and is not included in an inference savings claim.

## Immutable evidence

Remote root:
`/mnt/storage/s42-phone-kernel-restore-20260913-v1-QGNw0k/`.

The [physical](physical) copy includes the unmodified recorder, invocation,
58 timestamped command receipts, boot plan, result, and post-boot checks.
`physical/ARTIFACTS.sha256` covers all copied evidence files except itself.
Earlier failed preflight and transport artifacts remain unchanged.

[RESTORE_RESULT.json](physical/run/RESTORE_RESULT.json) SHA-256:
`f1c6d90f9710b2ed88e2469e87fdf140d8f62237f5a9f806e6a1c10107533f9a`.

Only this report, its evidence copy, and the project log were added or updated.
No software tests were rerun for this device-only restoration. Nothing was
committed, pushed, or deployed into the scheduler runtime.
