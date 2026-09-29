# Quiet-host dev3 retest: blocked before inference

Status: BLOCKED_PHONE_KERNEL_IDENTITY. This is a failed launch attempt, not an
inference or energy result. No request, phone shard load, or paid interval began.

The requested retest reused the previous deployment without rebuilding or
changing any of its 378 source/test files. Models, FFN shards, CUDA graph mode,
desktop placements, initial evidence, requests, arrivals and accounting remained
unchanged. Only artifact/session namespaces and read-only host-activity sampling
were new. No baseline or longer trace was run.

## Host was quiet; the phone identity changed

The pre-run five-second sample measured 0.080 busy CPU-core equivalents and no
compiler processes. GPU utilization was 0%, with 3,179 MiB used and 12,770 MiB
free. GDM and other users' processes were untouched.

The hardware/catalog preflight completed with status PASS, but that report also
explicitly has `phone_assistance_ready=false` and
`physical_inference_executed=false`. The subsequent direct-phone transport
preflight failed before `rig.start()`:

```text
Required: 6.12.23-android16-5-o-g227664cbe007-4k
Observed: 6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k

phone kernel is not qualified for direct DMA-BUF:
6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k
```

A separate read-only `adb shell uname -r` confirms the observed kernel. The
phone reported about 20 minutes of uptime when checked, so its boot state had
changed since the previous successful experiment. The frozen contract declares
boot image SHA-256
`26e8d41808b10b70264d958582bb6f6c9fba634c34275e0bc28fc820a6d3fb8d`;
the current boot image was not read or asserted to match it.

The strict check is in
`adapters/phone_session_ops/transport.py::_phone_kernel_release`, invoked first
by `DirectPhoneFfnSession.preflight`. It was not bypassed or relaxed. A changed
kernel requires separate qualification, not substitution of its name into the
previous contract. Restoring the previously qualified kernel is necessary for
an otherwise unchanged retest.

USB remains in normal `ptp,adb` mode at 5,000 Mbps. GPU memory and utilization
were unchanged after the failed launch. There was no phone reboot, boot-image
write, USB reset, worker restart, or process termination by this attempt.

## Preserved evidence and scope

- [FAILURE.json](physical/FAILURE.json): exact expected/observed kernel and
  launch phase, source verification, phone state, and null energy result.
- [RETEST_SUMMARY.json](RETEST_SUMMARY.json): failure plus the pre-run host sample.
- [RUN.log](physical/RUN.log): original transport-preflight traceback.
- [ARTIFACTS.json](ARTIFACTS.json): SHA-256 and sizes for all 26 physical files.
- `physical/HOST_ACTIVITY.jsonl`, the two before-run observation files,
  `HOST_AFTER_RUN.json`, and the partial Nsight report/export remain available.
  There is no completed RESULT.json and no new savings calculation.

An initial setup-only recorder error rejected floating-point diagnostic values
through the canonical configuration serializer. It was corrected to use ordinary
JSON for observations before launch, without scheduler changes. The initial
script is retained as `physical/PREPARE_FAILED_EXPERIMENT.py`; the exact launch
script is retained as `physical/EXPERIMENT_FINAL.py`.

New files are limited to this report directory: `experiment.py` (configuration
and observation/persistence wrapper), `summarize.py`, this README, JSON summaries
and copied physical artifacts. `research_dev/talks.md` records the blocker.
No scheduler tests were rerun: the unchanged deployment previously passed the
303 focused/replay tests with both goldens unchanged. The production source
hashes were checked against that deployment again before launch and after failure.
Nothing was committed or pushed, and older attempts were not overwritten.

Remote artifacts: `/mnt/storage/s42-layout-economics-20260910-v2-clean-host/`.
Reused deployment: `/mnt/storage/s42-layout-economics-20260910-v1-deploy/`.

```text
FAILURE.json SHA-256
64ec9ec3cf9ccfb019fc0f68f8d8a37b1ebc324043b81ad56cecc852a06351c4
RUN.log SHA-256
043dde10d49685df5c020488cf2bf46ceb7d73cc79871eab6cc47248afe4228b
inputs/SOURCE_MANIFEST_EXECUTION.json SHA-256
ebe029e79f8106a56eee9e264644ad1b43affdb581510bf037d3334c8f411e49
```
