# S15 persistent OP15 B32 gate

Status: `PERSISTENT_OP15_B32_MECHANICS_PASS_ENERGY_UNKNOWN`.

## Goal

Connect the versioned stagenet DETACH/reset mechanism to the independently
certified OP15 `[0,8)` B32 route. Seven fresh host sessions must reuse one phone
worker and finish with legacy STOP.

## Gate

- One OP15 PID, boot nonce, and device boot ID across all seven sessions.
- Session IDs 1 through 7; DETACH x6 then STOP.
- Exact B32 same-batch CUDA tokens in every session.
- HTP0 compute placement; CPU only for GET_ROWS.
- Reset acknowledged before reuse and increasing worker step totals.
- Each prompt-to-host-exit interval at most 4,000,000 us.
- Valid end thermal state.

This is a phone-worker persistence gate. Each host session loads its A6000 tail
before prompt submission, so it does not yet prove a persistent server-tail
process or a mixed-workload energy benefit. Energy remains UNKNOWN.
