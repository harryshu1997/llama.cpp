# S15 post-load B32 recertification results

Verdict: `ARRIVAL_FAITHFUL_PHYSICAL_B32_PASS`

Date: 2026-07-18. No commit or push. Energy scope: UNKNOWN.

## Arrival-faithful physical dispatch

The frozen BurstGPT cohort was replayed through the real
`MixedDispatchCoordinator` using its observed logical arrival timestamps. The
payload, priority, and five-second deadline were explicit synthetic sidecars.
The replay produced 31 bounded WAIT decisions and one exact B32 typed EXECUTE.
The host model and context were already resident, but the prompt was not in
the command line or preflight bytes. Prompt submission occurred only after the
typed launch.

| Metric | Result |
|---|---:|
| Observed arrival interval | 247,000,000 to 248,000,000 us |
| B32 logical launch | 248,000,000 us |
| Profile-derived latest wake | 248,031,633 us |
| Full typed-request-to-response time | 3,577,135 us |
| Prompt-to-process-exit paid window | 3,575,393 us |
| Logical completion | 251,577,135 us |
| Earliest synthetic deadline | 252,000,000 us |
| Deadline margin | 422,865 us |
| Completion ownership | 32/32 `completed_phone` |
| Correctness | 32/32 exact same-batch CUDA streams |
| Placement | 1,856 HTP compute nodes; CPU only 8 `GET_ROWS` nodes |
| End HMX temperature maximum | 39.1 C |

The real host marker order was `DRIVER_INPUT_READY`,
`DRIVER_INPUT_ACCEPTED`, `DRIVER_READY`, then `DRIVER_DONE`. The independent
validator re-derives scheduler decisions, the exact four-second post-launch
budget, all artifact hashes, terminal ownership, prompt ordering, token
correctness, placement, D2H completion, and the logical deadline:

```text
VALID_ARRIVAL_FAITHFUL_B32 transport_elapsed_us=3577135 deadline_margin_us=422865 energy=UNKNOWN
```

This is one physical dispatch backed by the separate seven-process profile
below. It is not seven arrival-trace repetitions, and the logical arrival
replay is not wall-clock paced. It establishes that the actual coordinator and
typed executor can wait for B32 and complete this frozen cohort within its
synthetic SLO without making the server wait for an unready phone.

## Seven-process route profile

Seven fresh real-device processes used the current CUDA host binary and the
frozen OP15 worker. The host model and context were resident before
`DRIVER_INPUT_READY`; the host command and pre-prompt stderr contained no
prompt. The harness submitted `Explain batching.` only after readiness and
timed through both process exits, placement evidence, and D2H completion.

| Metric | Result |
|---|---:|
| Batch / generated tokens | 32 / 8 |
| Layer route | OP15 `[0,8)` + A6000 `[8,48)` |
| Complete times, us | 3,944,399; 3,939,577; 3,901,898; 3,941,004; 3,946,008; 3,968,367; 3,943,267 |
| Median | 3,943,267 us |
| Conservative maximum | 3,968,367 us |
| Remaining cohort budget | 4,000,000 us |
| Conservative margin | 31,633 us |
| Process CoV | 0.462% |
| Correctness | 224/224 streams exact vs same-batch CUDA |
| Placement | all HTP0 compute; CPU only `GET_ROWS` |
| End HMX temperature maximum | 38.0 C |

The new profile ID is
`sha256:9fa84921ec2f90d2887d9ae0be6f785addd2ca64f2d99d98f4b446cd5e281a2d`.

The first acquisition is retained under `results_polling_wait_failure/`. Its
six normal processes were near 3.937 seconds, but one result was reported at
4.037 seconds because Python's timeout-based `Popen.wait()` polls at a coarse
interval. The route wall for that process was ordinary. The repair starts
blocking exit watchers for host and phone before prompt submission and records
the maximum observed exit time. It does not change the model, route, workload,
batch, SLO, or pass threshold.

Independent validation:

```text
VALID_POST_LOAD_B32 p50_us=3943267 conservative_us=3968367 profile_id=sha256:9fa84921ec2f90d2887d9ae0be6f785addd2ca64f2d99d98f4b446cd5e281a2d energy=UNKNOWN
```

## Claim boundary

The seven-process profile certifies the measured post-load B32 route. The live
dispatch additionally certifies one coordinator-triggered physical execution
of the frozen cohort. Neither result establishes selected-GPU energy relief,
phone energy, USB energy, total-system energy, another prompt shape, another
batch size, another device, or performance over the full source trace.
