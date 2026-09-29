# Qualified OP15 kernel restored with temporary RAM boot

The user authorized restoring the previously qualified phone kernel after the
unchanged dev3 retest stopped on the stock-kernel identity check.

Result: KERNEL_RESTORED. No partition was flashed, no boot slot was changed,
and no data was wiped. The phone was intentionally rebooted; no research worker
was active beforehand. Desktop processes and GDM were untouched.

## Verified inputs and procedure

The original FunctionFS transport report documents temporary `fastboot boot`
of this image, and documents that normal reboot returns the same phone to its
stock kernel. The stock DMA-BUF path previously caused a kernel panic, so its
strict exclusion was preserved rather than bypassed.

The saved rig manifest identifies:

```text
Phone serial: 3C15AU002CL00000
Boot image: /home/zhihao/s41-ffs-dmabuf-fixed-v1/boot.img
SHA-256: 26e8d41808b10b70264d958582bb6f6c9fba634c34275e0bc28fc820a6d3fb8d
Fastboot: /home/zhihao/s41-ffs-dmabuf-fixed-v1/fastboot
Fastboot SHA-256: 76dde33fee8b1fd00bcaf2e7f94ddef6407f0beb5bc3a98a3d4127307af23f3a
```

The image hash was checked twice, including immediately before boot. The phone
was already bootloader-unlocked, fully charged, and on slot `_b`. Exact serial
and slot were checked again in fastboot. No unlock command was issued.

Commands executed on the desktop, with bounded waits and command receipts
between steps:

```sh
/usr/bin/adb -P 5037 -s 3C15AU002CL00000 reboot bootloader
/home/zhihao/s41-ffs-dmabuf-fixed-v1/fastboot -s 3C15AU002CL00000 getvar serialno
/home/zhihao/s41-ffs-dmabuf-fixed-v1/fastboot -s 3C15AU002CL00000 getvar current-slot
/home/zhihao/s41-ffs-dmabuf-fixed-v1/fastboot -s 3C15AU002CL00000 boot /home/zhihao/s41-ffs-dmabuf-fixed-v1/boot.img
/usr/bin/adb -P 5037 -s 3C15AU002CL00000 shell uname -r
```

`fastboot boot` reported successful image transfer and boot in 1.946 s. Total
time from the reboot request through post-boot verification was 59.586 s.
This maintenance operation is separate from inference accounting; its energy
was not measured and is not claimed as free inference preparation.

## Post-boot checks

- Kernel: `6.12.23-android16-5-o-g227664cbe007-4k`.
- Kernel BTF SHA-256:
  `f3afcf985b24de3eb5a1d5453bf4ffd95963b806ce99a59430c5a8bf7d201d17`,
  matching the saved qualified transport evidence.
- Android boot completed, ADB usable, root identity confirmed.
- Active slot remains `_b`; USB remains `ptp,adb` at 5,000 Mbps.

This is intentionally temporary. A normal reboot boots the installed stock
image again. No persistent kernel installation is included in this task.

## Evidence

The recorder [restore.py](restore.py) performs only the bounded restore and
read-only checks above. It contains no scheduler policy or qualification change.
All 57 command receipts, the verified plan, and the terminal result are preserved
under [physical](physical), copied from
`/mnt/storage/s42-phone-kernel-restore-20260910-v1/`.

[RESTORE_RESULT.json](physical/RESTORE_RESULT.json) SHA-256:
`1f4f127dc237311977317de970dd27b08e4785274dff8efc49acd7690cd95eff`.

The earlier failed retest remains unchanged under
`20260910-layout-economics-clean-host`. The authorized small retest uses a new
artifact namespace, `s42-layout-economics-20260910-v3-restored-host`, with the
same frozen scheduler deployment and no baseline or longer trace.
