# OP15 DMA-BUF cancellation repair candidate

## Outcome

The existing fence-lifetime candidate was applied to an isolated copy of the
vendor kernel inputs and successfully built. Configuration, exported module
symbol versions, kernel release, built-in module list, boot header and ramdisk
match the previous restored image. This is a source/build result, not a
device-qualified cancellation fix. No new image has been booted or flashed.

The current direct-USB forced-abort quarantine remains enabled. Scheduler
phone-owner execution, owner-loss recovery and inference gates remain blocked
until transport cancellation is qualified. No trace or baseline was run.

## Defect and repair

The exported sync_file retains its dma_fence, but the reviewed kernel stores
that fence's spinlock in the separately reference-counted DMA-BUF attachment.
Completion cleanup and detach can release that attachment while the exported
fence remains accessible. A later fence operation can then use a freed lock.
The [upstream sync_file implementation](https://raw.githubusercontent.com/torvalds/linux/master/drivers/dma-buf/sync_file.c)
confirms that exported files retain and later operate on fences.

Reused, without rewriting, the existing candidate in the sibling project:

`moe-resident-routing-a6000/artifacts/op15_kernel_fix_20260913/fence_lifetime.patch`

- Move the lock into the fence, giving both the same lifetime.
- Free a raw, uninitialized fence allocation with `kfree()` on the two
  pre-initialization failure paths.
- Preserve the prior DMA direction, request-cancellation serialization and
  fence-reference backports. No USB UAPI, kernel configuration, worker protocol,
  FFN shard format or scheduler policy change.

This is a concrete source defect consistent with the crash timing, not a
proven identification of the crashing instruction. No panic stack has been
recovered. The vendor record still proves a kernel crash at 18:46:40; its
original FAIL is preserved and is not superseded by these software checks.

The sibling's userspace mitigation predates this turn. It waits for and closes
exported fences while attachments remain alive, then detaches completed
transfers. Pending idle receives instead time out into resource retention.
That is fail-closed, not successful forced cancellation or bounded recovery.
Cooperative STOP is a separate proof and must not be relabelled as cancellation.

## Isolated build and validation

Build root:

`/home/myid/zs89458/Documents/op15-cancel-kernel-20260913-QEC2Bo/`

Reference inputs, untouched:

`/home/myid/zs89458/Documents/s41-op15-kernel-rx-v1/`

Common revision: `227664cbe007bbad49aa74259179ac99608a2113`, plus the existing
backports. Only the isolated `drivers/usb/gadget/function/f_fs.c` receives the
additional patch. Original source and original boot-image hashes were checked
again after building and are unchanged.

Build command is recorded in `build.sh`. It uses the original vendor Kleaf
target `//common:kernel_aarch64_gki_artifacts`, four make jobs, two Bazel jobs,
CPU nice 10 and idle I/O priority. Build completed successfully in 628.872 s.
The original build's missing optional GKI-vmlinux and duplicate BTF-ID warnings
also appear here; the complete build log is preserved.

`verify_build.sh` compares the candidate with the original built outputs and
extracts the actual BTF and notes. Its first invocation had a wrong config
output pathname, not a build defect. The failed invocation and script are
preserved as `VERIFY.log` and `verify_build-v1.sh`. The corrected path is the
Kleaf configuration output, and `VERIFY-v2.log` ends with
`BUILD_COMPATIBILITY_CHECKS_PASSED; DEVICE_QUALIFICATION_NOT_PERFORMED`.
No comparison requirement was relaxed. The full boot-header comparison is
also identical, including kernel size.

The recorded final focused run completed in 4.82 s; `TESTS.xml` preserves its
76 passes and one skip. An earlier focused run in the same turn also passed
76 tests with one skip. The isolated Bazel server was shut down after the build
and checks; build outputs are retained.

| Check | Result |
| --- | --- |
| Focused direct-USB/lifecycle tests | 76 passed, 1 optional CPU-graph test skipped |
| Full ARM64 kernel and GKI image build | PASS |
| Kernel configuration | Byte-identical |
| Exported module ABI (`Module.symvers`) | Byte-identical |
| Kernel release and built-in module list | Byte-identical |
| Boot header, ramdisk and image/kernel consistency | PASS |
| Forced cancellation on the new kernel | NOT RUN |
| Real-phone FFN relocation/recovery | NOT RUN |

The focused tests exercise queue, protocol, shutdown acknowledgement, resource
retention on failure, the abort quarantine, and isolated application of the
fence patch. They do not execute a kernel cancellation or establish KASAN safety.
No scheduler logic was edited, so no scheduler replay was re-pinned.

## Artifact identities

| Artifact | SHA-256 |
| --- | --- |
| Existing candidate patch | `bf6702ce03b1165b047b33bc7fc940a64e7e87b29a91443b6894376a7c5eda7b` |
| Original vendor `f_fs.c` | `229d79e869516834161cd8e4e5cc6b619ecd0e8930c096acc33fde5aa2cf4f63` |
| Patched isolated `f_fs.c` | `f5882ee2939f5a74db307a3650fed2079c085f1ac23d65f5a4422829a651578a` |
| Original restored `boot.img` | `26e8d41808b10b70264d958582bb6f6c9fba634c34275e0bc28fc820a6d3fb8d` |
| New candidate `boot.img` | `f13c7c033ce74f32ea4aa6349cb531704407708f5fb35756a6c361ffb48322a3` |
| New candidate `Image` | `edf69b76e43bf41a17970f6c5c872fb0229dc2bb938a26a79bca456c2a897edd` |
| Unchanged kernel config | `9f03ed30a44329ebc6337dca7157f3eaa67c3143519883b026c51abd0d7dda43` |
| Unchanged module symbol versions | `0fabe11e297ba37c2069d73f2d74b99d014442226631c95c4704649b949429f3` |
| Candidate BTF | `77a8ce5acc215506e8dff1bf84e4561ef316680bcba803e9dfc16acecc449735` |
| Candidate kernel notes | `40bbacf73c0d35195693c566ae695803f59fbf7ec8ce817a1921fb44b3807b51` |

Candidate GNU build ID: `095dda29675da0a59a7c487cf77b8fc7775928ab`.
Unchanged release string: `6.12.23-android16-5-o-g227664cbe007-4k`.
The unchanged release string is NOT a sufficient identity check for this image.
Existing phone kernel/transport qualification must not be reused by merely
replacing saved hashes.

## Exact changes in this turn

- In the isolated vendor copy only: `kernel_platform/common/drivers/usb/gadget/function/f_fs.c`.
- New task-local files: `BUILD_PLAN.md`, `build.sh`, `verify_build.sh`,
  preserved `verify_build-v1.sh`, build/verification outputs and logs.
- In this repository: this report directory and the new `research_dev/talks.md`
  entry. No scheduler, native FFN worker, shard generator or sibling production
  source edits. Existing sibling repairs were reviewed and tested, not rewritten.

## Remaining physical step

Coordinate an exclusive idle-device window and authorization for a temporary
RAM boot of this new, unqualified image. No persistent partition flash.
Verify the exact boot-image, notes, BTF and config hashes; use a separately
recorded candidate test identity rather than bypassing the old kernel guard.

At 21:26:19 EDT, the phone was still on restored boot ID
`d03a2ef4-6392-4de3-a8a7-1f2376eb9b00`, Android boot complete, `ptp,adb`,
5,000 Mbps USB. A read-only desktop check found a new active lock holder:
PID 270652, started 21:23:40 EDT, `python -m src.run_gemma_direct_energy`,
working directory
`/home/zhihao/moe-resident-routing-4060ti-op15/experiments/gemma_direct_energy_20260913`.
Its FD 4 holds
`/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock`.
This is another experiment, not this turn's process. No signal, stop, reboot,
USB reset or attempted lock bypass was issued. An exclusive device window and
authorization for the new candidate's temporary boot are required before testing.

Then run bounded normal transfer/cleanup and cooperative-stop checks, followed
by explicitly reviewed forced cancellation of four pending receives. Capture
DMA/fence completion, cleanup acknowledgement, same-boot health, kernel logs
and resource counts. A timeout, reboot or missing completion is a failure;
never release buffers or reset USB to manufacture completion.

Only after cancellation passes may the existing scheduler phone-owner path be
completed with canonical preparation/READY and request-bound leases, followed
by short real-phone Gemma relocation and owner-loss recovery. The archived
empty-lease gate draft remains unvalidated and must not be deployed.

Evidence file SHA-256 values:

- `BUILD.log`: `3a2e5cf3363f9399648a580e6d42ab7d2f4b1e919be1239a3ad3ec69a4753fa3`.
- `VERIFY-v2.log`: `57e3ed2adad0e9bbc9a58e39bb8e575498812441407b665d1d6e85d3b96385d7`.
- `TESTS.xml`: `0c272e3852e1ecac02a11edf18d8164a8232486f0662d3719fef46f12ad2db60`.
- `KERNEL_CRASH_EVIDENCE.txt`: `2f0310138578e182872e2d27b8d8e7249e90ff815d803174b111b618bd355f9a`.
