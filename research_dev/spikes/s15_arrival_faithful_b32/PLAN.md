# S15 post-load B32 recertification

Status: `ARRIVAL_FAITHFUL_PHYSICAL_B32_PASS`.

## Goal

Recertify the OP15 `[0,8)` plus selected-A6000 `[8,48)` B32 route after the
host model and context are resident but before any prompt bytes are visible to
the host driver. The frozen BurstGPT cohort leaves exactly 4,000,000 us between
its B32 launch and its earliest synthetic deadline.

## Gate

- Seven fresh physical processes.
- Current host executable and all repo-local shared libraries bound by SHA-256.
- Frozen phone worker, model, shard, cohort, and input manifest bound by SHA-256.
- No `-p` or prompt text in the host command or pre-prompt stderr bytes.
- Exactly one `DRIVER_INPUT_READY` before submission and one
  `DRIVER_INPUT_ACCEPTED` after submission.
- 32/32 exact same-batch CUDA token streams.
- OP15 scheduled placement on HTP0; only `GET_ROWS` may use CPU.
- Complete prompt-submission-to-certified-result time at most 4,000,000 us in
  every process, with process CoV at most five percent.
- Valid continuous HMX thermal samples.

Energy is out of scope. Failure does not authorize lengthening the frozen SLO
or dropping the seven-process gate.

## Live integration gate

After the route profile passes, replay the frozen observed arrival timestamps
through `MixedDispatchCoordinator`. The replay is in logical trace time; it
does not sleep through the one-second trace interval. The physical route must
start only from the typed B32 EXECUTE action after the host emits
`DRIVER_INPUT_READY`.

- Produce exactly 31 bounded WAIT decisions followed by one exact B32 launch.
- Bind route epoch 13 and the post-load B32 profile digest.
- Give the physical executor only the 4,000,000 us remaining after launch.
- Require one terminal owner for every request.
- Include process exit, D2H completion, exact CUDA tokens, placement, and
  thermal evidence in the paid physical result.
- Validate the persisted result independently from its raw artifacts.

The payload, priority, and five-second SLO are synthetic sidecars. Only the
arrival timestamps and source request identities come from BurstGPT.
