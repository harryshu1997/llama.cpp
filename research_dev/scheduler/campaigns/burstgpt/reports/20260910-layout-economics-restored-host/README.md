# Kernel restored; unchanged dev3 retest blocked by clock-skew telemetry

The authorized temporary boot restored the exact qualified OP15 kernel and BTF
identity. [Restoration report](../20260910-phone-kernel-restore/README.md)
contains the image hashes and all boot commands. The strict direct-phone
preflight now passes. No production code, safety threshold, system clock, model,
binary, placement, initial evidence, arrival or token count was changed.

Only the existing adaptive Gemma36/Llama37/Qwen50 workload was retried, using
the same frozen deployment. All 378 source/test file hashes match the preceding
tested run. Artifact and phone-session directories are fresh. No baseline or
longer trace was run, and no previous results were overwritten.

## Result

The retest failed with `physical_execution_control_failed`, caused by
`phone power sampler is not ready`. It is not an energy or coverage result.
Desktop warmup and one phone shard load occurred, but no requested workload
request completed and no requested adaptive decode window was measured.

| Observation | Measured result |
| --- | --- |
| Pre-run host activity | 0.072 busy CPU cores; no compiler processes |
| GPU before run | 0% utilization; 12,770 MiB free; GDM preserved |
| Compiler processes in profiler capture | 0 |
| Completed phone loads | 1 Gemma shard, HTP0 generation 1 |
| Physical load receipt interval | 7.022249 to 24.765145 s |
| Explicit HTTP clock-skew rejections | 210 snapshots, 25.918 to 255.664 s |
| Reported snapshot age | -10.519 to -8.959 s |
| Power samples / unavailable events | 14 / 791 |
| Requested workload completions | 0 of 3; no valid savings comparison |

During FunctionFS operation, ADB was unavailable and the supported HTTP
telemetry endpoint responded. Its phone timestamps were about 10 seconds ahead
of the desktop. `adapters/probes.py` subtracts phone wall-clock time from host
wall-clock time and rejects absolute differences over five seconds. Runtime
snapshots explicitly record this rejection as `STALE`, not measured memory
exhaustion. The power probe uses the same cross-clock freshness condition.

Without a fresh power observation, `energy.py::wait_until_ready` did not allow
the requested execution measurement to begin and eventually raised the terminal
error. The 33 helper rematerialization failures include
`PHONE_TELEMETRY_UNAVAILABLE`; none is the old `opportunity is not exact` error.
This attempt cannot test the new economics under normal execution.

The power diagnostics collapse unavailable probes to `NO_SAMPLE`, without
preserving each raw HTTP response. They do not independently prove the reason
for every failed power read. Runtime snapshots do directly establish the clock
skew, and an independent post-cleanup ADB clock check confirms roughly ten
seconds of disagreement. Desktop NTP synchronization and phone automatic time
were both enabled. Neither clock was changed by this task.

## Preserved state and next correction

After cleanup, the phone still runs the qualified kernel, USB is back to
`ptp,adb`, and no research phone workers remain. GPU utilization is zero and
the GDM allocation remains intact. No phone reboot occurred during the gate.
The only reboot was the separately authorized RAM boot before this attempt.

The next narrow correction is telemetry clock-domain/freshness handling and its
measurement-readiness failure path. It must not accept expired observations,
fabricate sample times, or relax the five-second check to hide the skew. No
scheduler fix or additional run is included here.

The unchanged code retains its earlier 303 focused/replay PASS; tests were not
rerun for this hardware-only restoration. Replay goldens were not edited.

## Immutable evidence

- Remote: `/mnt/storage/s42-layout-economics-20260910-v3-restored-host/`.
- Local: [physical](physical), all 470 copied files including failed execution,
  decisions, snapshots, helper events, raw power samples and profiler capture.
- [DIAGNOSIS.json](DIAGNOSIS.json): derived counts, rejected snapshot paths,
  exact load receipt and post-cleanup clock/kernel checks.
- [ARTIFACTS.json](ARTIFACTS.json): per-file sizes and SHA-256 hashes.
- [experiment.py](experiment.py): bounded unchanged-retest wrapper; the exact
  executed copy is also preserved inside the physical artifact directory.

Key SHA-256 values:

```text
run/FAILURE.json
b8f900021de59e969163625736b49bf4e36e22cced30afbdfa78e3badc89b793
run/phone-power-diagnostics.json
f59f1950cb46570764f7b445f891eee1c2ed294be3f28908803ab9e1d884981c
inputs/SOURCE_MANIFEST_EXECUTION.json
2127aef65da96831e56b75ed6e8f38a3981a49ff14ee9ab05301ef9bd14abd64
```
