# Assumed-power measurement readiness fix

Fixed the inconsistency exposed by the restored-host v3 failure. When a phone
power profile and activity tracker are configured, `measure()` already uses
assumed active/idle power, but `prepare()` still waited for measured phone power.
Preparation now skips only that phone-sample wait in assumed-power mode.

Fresh CPU/RAPL and GPU/NVML samples are still required at preparation. Complete
server measurement coverage and gap checks remain unchanged. Phone activity
intervals, active/idle energy, diagnostic labeling, and unknown charging state
remain explicit. Measured-phone-power mode still waits for fresh phone samples
and requires complete sample coverage when measuring. Independent thermal,
battery, memory, session identity and lease checks were not changed.

## Verification

Six new regressions cover missing/stale phone samples in assumed mode, active
and idle accounting, fresh server requirements in both modes, measured-phone
readiness and coverage failure, measured-phone success, and server coverage
failures in assumed mode. Before the fix, the two assumed-mode preparation
regressions reproduced `phone power sampler is not ready`; after the fix all
six pass.

The focused run passes 46 tests in 13.288 s, including existing telemetry
recovery and assumed-power admission/low-battery protections. Command and output
are in [TESTS.txt](TESTS.txt). No broad harness or replay regeneration was run.
No adaptive controller or Qwen policy code changed.

Production change: `adapters/energy.py` only, plus regressions in
`tests/test_phone_power_probe.py`. [CODE_CHANGES.json](CODE_CHANGES.json) records
before/after hashes. This patch is local and has not been deployed.

## When the three-request retest can run

The power-readiness bug is fixed, but a separate runtime-health issue remains.
A read-only check at 10:19 EDT confirms the qualified kernel remains restored,
the normal-USB ADB health observation is valid, and the phone clock is at least
10.235 s ahead of the desktop. The phone has 11,899,387,904 available bytes,
33.3 C observed temperature and 100% battery. These are normal-mode observations,
not proof of FunctionFS-mode health freshness. See [LIVE_CHECK.json](LIVE_CHECK.json).

During FunctionFS execution, ADB is unavailable and the existing HTTP health
probe rejects clock differences greater than five seconds. The prior v3
snapshots prove that rejection. The live offset remains outside that bound, so
a power-only fix does not establish that phone assistance can be admitted.
FunctionFS execution was not restarted merely to reproduce this known condition.

Next: resolve the runtime-health clock-domain issue without inventing freshness
or weakening safety, deploy the tested changes in a fresh artifact namespace,
and pass preflight before running the same Gemma36/Llama37/Qwen50 experiment.
Qwen's coarse-probe exit, ACK/warmup budget overrun, and batch-1-to-1 context
events remain follow-up control-path issues, not repaired by this energy patch.

No new physical inference, baseline, longer trace, system-clock change, process
interference, commit or push. The previous failed physical artifacts and their
hashes remain unchanged; there is no new savings result.
