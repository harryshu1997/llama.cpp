# S15 persistent OP15 B32 results

Verdict: `PERSISTENT_OP15_B32_MECHANICS_PASS_ENERGY_UNKNOWN`

Date: 2026-07-18. No commit or push. Energy scope: UNKNOWN.

## Result

One OP15 `[0,8)` stagenet worker served seven sequential exact B32 sessions.
Sessions 1 through 6 ended with DETACH and an acknowledged request-state reset;
session 7 ended with the legacy STOP command. The worker PID, boot nonce, device
boot ID, weights, backend contexts, and listening socket remained resident.

| Metric | Result |
|---|---:|
| Resident worker PID | 17070 |
| Session sequence | DETACH x6, STOP x1 |
| Requests | 224 total, 32 per session |
| Generated tokens | 1,792 total, 8 per request |
| Prompt-to-host-exit times | 3,497,118; 3,418,653; 3,418,709; 3,675,967; 3,642,362; 3,676,122; 3,620,283 us |
| Median | 3,620,283 us |
| Conservative maximum | 3,676,122 us |
| Process CoV | 3.020 percent |
| Four-second session gate | PASS, 323,878 us minimum margin |
| Correctness | 224/224 exact vs current same-batch CUDA |
| Placement | HTP0 compute; CPU only GET_ROWS |
| Worker steps | 384/session; totals 384 through 2,688 |
| HMX temperature | 29.8 C start, 38.4 C end |

Every session emits one strict `ls-stagenet-session-v2` certificate. Session
IDs are contiguous, reset is true only for DETACH, placement has no undeclared
fallback, and the worker identity is constant. The independent validator also
binds the raw host outputs, phone certificates, remote Android runtime hashes,
frozen host and Android binaries, shared libraries, source, and shard identity:

```text
VALID_PERSISTENT_OP15_B32 sessions=7 max_elapsed_us=3676122 energy=UNKNOWN
```

The result is installed as a new runtime profile at route epoch 14. It does not
re-pin the historical epoch-13 profile whose old host executable was not
retained. The S15 runtime suite passes 79/79 against epoch 14.

## Repair found during integration

The first persistence implementation sent DETACH whenever the option was set,
even after a host-session failure. That could leave a failed worker resident.
The driver now permits DETACH only after a clean session, bounds the ACK wait,
uses STOP on an already failed session, and attempts to stop a worker whose
DETACH acknowledgement fails. The batched single-phone pipedriver now supports
the same opt-in `--session-end detach|stop` behavior; B1 paths reject the option
until explicitly implemented.

## Claim boundary

This proves real phone-worker persistence and exact reset for the certified B32
route. It is not an arrival-trace replay: all seven sessions use the same
synthetic prompt. Each host process loads its A6000 tail before prompt
submission, so A6000-tail persistence and load energy are not measured. It does
not establish selected-GPU, phone, USB, server-wall, or total-system energy
savings. The next mechanics gate is a persistent host-tail process driven by
the typed multi-exchange transport, followed by a matched mixed-workload run.
