# S15 live launcher: one real coordinator-triggered B32 route

Status: `PHYSICAL_MECHANICS_PASS_ARRIVAL_FAITHFUL_SLO_BLOCKED`.

## Goal

Connect the S15 `MixedDispatchCoordinator` and `PhysicalExecutor` to one real,
already-certified OP15 `[0,8)` route. The run must consume the frozen observed
BurstGPT cohort, wait until all 32 compatible requests are present, launch one
B32 phone batch, continue the Gemma tail on the selected A6000, and fail closed
unless all 32 results are token-correct and owned exactly once.

This checkpoint does not measure energy. It does not claim arrival-faithful SLO
because the frozen host binary accepts the prompt on its command line before the
logical arrivals are replayed.

## Frozen inputs

- `s15_burst_cohort/cohort.json`, file digest
  `sha256:85e0614d31a6f35ad7d09c3b35b2ab4fa9ef2916bc4a1dcfc32d6f4c70a51ec4`
- `s15_burst_cohort/input_manifest.json`, file digest
  `sha256:ea1f2d47b8c6dec6096c8b95b6073181ecba6838d97741ca92ec2ceca6937858`
- 32 observed BurstGPT event identities over a one-second logical arrival span
- frozen OP15 route profile, host binary, Android binary, model, and shard

The cohort and input-manifest file digests are mandatory fields in the registry
lane, execution request, physical binding, and returned session record. A direct
fixed-id smoke manifest exists only as a subordinate parser fixture and cannot
authorize the physical result.

## Checkpoints

- [x] Bind cohort and input-manifest identities through the executor contract.
- [x] Preflight all frozen host, phone, model, shard, boot, and thermal evidence.
- [x] Prepare the phone and host processes before the paid coordinator exchange.
- [x] Replay all 32 requests through `MixedDispatchCoordinator` at their frozen
      logical arrival times.
- [x] Require 31 bounded `WAIT` decisions followed by one
      `target_batch_ready` B32 launch.
- [x] Validate 32 route records, exact CUDA-reference tokens, D2H completion,
      HTP placement, allowed CPU metadata work, epochs, and ownership.
- [x] Persist exact request/stdout/stderr/transport and underlying raw evidence.
- [x] Validate the persisted result with an independent fail-closed validator.
- [x] Preserve both failed attempts and their regression tests.
- [ ] Rebuild and separately certify the prompt-after-load seam before making an
      arrival-faithful SLO claim.
- [ ] Measure selected-GPU energy in a separately frozen matched-control run.
- [ ] Add OP12 and simultaneous independent phone lanes in a later checkpoint.

## Stop rules

- No partial cohort, direct fixed-id substitution, or fabricated completion.
- No launch on a cohort, input, route, profile, binary, boot, or epoch mismatch.
- No completion if the full transport reply exceeds any request's remaining
  deadline budget at the actual cohort launch time.
- No completion with a missing/extra stream, wrong token, missing D2H, CPU
  fallback beyond declared `GET_ROWS`, missing buffer, or invalid thermal sample.
- No energy, total-system, or arrival-faithful SLO claim from this checkpoint.
