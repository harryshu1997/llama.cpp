# OP15 candidate-kernel cancellation prerequisite

## Scope

The user authorized a temporary RAM boot and a bounded cancellation prerequisite
before actual FFN relocation. No partition flash, wipe, desktop reboot, driver
change, GDM stop, unrelated process signal, trace, or baseline run is authorized
or performed by this prerequisite.

The candidate is the already built fence-owned-lock repair recorded in
`../20260913-dmabuf-cancellation-repair/`. The original crash and failed attempts
remain unchanged. This test does not transfer the old kernel's qualification.

Remote artifact root:
`/mnt/storage/s42-dmabuf-cancel-20260914-v1-gBJdFx/`.

Local isolated test sources and binaries:
`/home/myid/zs89458/Documents/op15-cancel-kernel-20260913-QEC2Bo/transport-20260914-wbFjPs/`.

## Boot

The existing journalled RAM-boot recorder ran under the shared phone execution
lock after an idle GPU, phone-process/FD, camera, USB and serial preflight.
Post-boot verification checked actual notes, BTF and configuration, not just the
unchanged release string. Image and fastboot binary hashes were checked before
the boot command. No boot partition was written.

New boot ID: `fa74f551-fb1b-465e-b5a2-053b9869b1bb`.

| Artifact | SHA-256 |
| --- | --- |
| Candidate boot image | `f13c7c033ce74f32ea4aa6349cb531704407708f5fb35756a6c361ffb48322a3` |
| Actual kernel notes | `40bbacf73c0d35195693c566ae695803f59fbf7ec8ce817a1921fb44b3807b51` |
| Actual kernel BTF | `77a8ce5acc215506e8dff1bf84e4561ef316680bcba803e9dfc16acecc449735` |
| Actual config, unchanged | `9f03ed30a44329ebc6337dca7157f3eaa67c3143519883b026c51abd0d7dda43` |
| Isolated phone USB probe | `281f65477182909fd67e6c690ec6f44f670469dd893a35b2dabfcea2ae8b42f7` |
| Isolated session controller | `420ba1b965c209450c7b1d3c60af65702f588cce8bbecdabb3ce2f3c2387b075` |
| Reused host USB probe | `b017b8288efd036fb767d4b080e6bb48c97837528c3a78a6fc12ab0f52474cd6` |

## Test contract

Run one full-payload normal cycle, one cooperative STOP, then four pending
idle-receive cancellation. Each has a fresh phone session, artifact directory,
shared-lock acquisition and live preflight. A failure stops the sequence.

The opt-in cancellation probe permits only four 4096-byte receive buffers,
four unused transmit buffers, zero host bulk submissions, no loaded model and
no HTP computation. It first proves all four receive fences are pending.
It exports all eight fences, detaches, and requires receive cancellation
completion and settled transmit fences before releasing resources. It then
requires acknowledged USB restoration, endpoint closure, explicit buffer release,
60 seconds without a reboot, no new kernel fault diagnostics, and disappearance
of all eight individually identified kernel DMA allocations.

The candidate-native guard refuses the old kernel notes/BTF hashes. The active
sibling sources and forced-abort quarantine are unchanged. Only an isolated
Android USB test worker was rebuilt; HTP libraries, FFN worker and shard generator
were not rebuilt.

## Software checks

- Five Python test methods pass, including missing-event and ordering failures,
  invalid cancellation status, and refusal of the old binary bundle.
- Eighteen native cancellation-order/failure cases pass.
- Existing native direct-USB mocks, cooperative STOP, attached-fence lifetime and
  lifecycle-order checks pass.
- Shell syntax checks pass.

The smoke run used `src/run_direct_usb_device.py` as first deployed. Its final
kernel-log comparison rebuilt the same set for every line, delaying host-side
analysis after physical cleanup. `validator-v2/` hoists that set construction;
the membership check and all physical acceptance criteria are unchanged. The
original deployed source and smoke logs are retained, not overwritten.

## Results

All three bounded cases passed on the same candidate boot. The cancellation
case reproduced the four-pending-receive condition without the previous crash.

| Case | DMA drain | Drain through buffer release | Post-cleanup same-boot observation | Allocations remaining |
| --- | ---: | ---: | ---: | ---: |
| Normal full-payload transfer | 30.225 ms | 2.051 s | 61.316 s | 0 of 8 |
| Cooperative STOP, four acknowledgements | 52.005 ms | 1.974 s | 61.519 s | 0 of 8 |
| Four pending receive cancellation | 25.411 ms | 2.668 s | 61.491 s | 0 of 8 |

Native timestamps show 3.733 ms for all eight detach operations, 5.344 ms for
the exported-fence waits after detach, and 2.634 s waiting for the USB-restore
acknowledgement in the cancellation case. These are intervals in the phone's
own monotonic clock; the 25.411 ms drain includes event logging. No HTP time is
included. The four receive statuses are all `-104`; the four unused transmit
fences are settled with status `1`. The expected injected interruption exits
the worker with status 1 and controller with status 2, after acknowledged
cleanup. This is not an unexpected worker failure or an ordinary inference run.

The eight 4096-byte DMA allocations (32,768 bytes in total per case) were
checked in `/sys/kernel/dmabuf/buffers` before and after. All were absent after
exit. Android `ptp,adb`, 5,000 Mbps USB and the boot ID were retained; the saved
kernel-log differences contain no new BUG/KASAN/Oops/panic diagnostic matching
the strict test's fault patterns. This is bounded evidence, not a general
proof that every kernel or HTP cancellation is safe.

`analyze.py` reproduces `SUMMARY.json` from the saved physical receipts and
native events without contacting a device:

```sh
python3 research_dev/scheduler/campaigns/burstgpt/reports/20260914-dmabuf-cancellation-qualification/analyze.py
```

SUMMARY SHA-256:
`0bbb65b1c90461e256c3dee68d93c213cfab186219760e16923e6a648163b6b8`.
Cancellation RESULT SHA-256:
`4f983dffbdbbe6916a48722e56a53a25482557e3bfa95bd64380da4920133e9d`.
Cancellation native-events SHA-256:
`91ce5e35a77f8a6751f77381528319230f7470581313124ea79aba22748063cb`.

## Changes and limits

Only the isolated test copy changes: candidate kernel guards, an explicit
idle-receive cancellation probe, native fence/resource logging, strict receipt
validation, and the new boot/test recorder. The sibling's active implementation,
old kernel guard and forced-abort quarantine are unchanged. The `src-before/`
and `scripts-before/` copies, original candidate runner, `validator-v2/`, native
test sources, resolved host commands and boot command journal are preserved.

The two runner versions have SHA-256:

- Smoke: `ca64b98df1a62943b0fc9d9ddcca8b0b95b0dd990695a26a109ee3f05d308267`.
- STOP/cancellation: `40968ce8510d00dcfa0c03b68c7a67fe57ec3dbc20ac79bb4fe03cff19ffa77a`.

The first final read-only health command omitted root for kernel hashes and
reported permission denied. It is preserved in `FINAL_HEALTH.txt`; the corrected
rooted observer check is `FINAL_HEALTH-v2.json`. No USB or device change was
performed to correct that diagnostic invocation.

In the scheduler repository, changes are this report and its captured artifacts,
the offline analysis script, and the `research_dev/talks.md` entry. No scheduler
policy, lifecycle, FFN worker, shard generator, old physical result, baseline,
or replay golden is changed. Nothing committed or pushed.

Actual FFN relocation, model correctness, owner-loss recovery, useful KV
capacity and energy savings have not been tested here. This test uses the
isolated `own-direct-v4` USB worker, not the canonical FFN execution protocol.
Its cancellation proof must not be substituted for the canonical transport's
payload qualification or for an HTP-compute cancellation proof.

## Next relocation prerequisite

The active `remote_resident_gate.py` still supports phone preflight only. Its
request execution must be connected to the existing
`CanonicalOfflinePhoneResidencyPreloader`, authoritative READY publication and
request-bound physical adapter, with real leases and memory reservations. The
archived empty-lease draft must not be run. Re-measure the canonical transport
on this exact candidate image instead of replacing hashes in old receipts.

Then run the short real-phone omission/correctness/memory test, and separately
the supported owner-loss recovery. No 24/84-request trace. The candidate remains
RAM-booted; no persistent installation was made. A normal reboot will again use
the installed kernel.
