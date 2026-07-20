# S16 OP12 Persistent Lane Gate

## Question

Can OP12 serve as an independent READY B32 lane with exact reset semantics and
a lower-priority 12 s SLO?

Existing seven-process evidence places its non-persistent B32 p50 near 9.36 s.
It is therefore not eligible for the OP15 5 s class. The 12 s SLO is frozen
before this persistence run.

## Gate

- OP12 owns Gemma `[0,6)` on HTP0; one selected A6000 owns `[6,48)`.
- Same frozen B32 cohort and 8-token CUDA reference as the OP15 gate.
- Two exchanges in one process generation: DETACH then STOP.
- Same host PID, phone PID, boot nonce, and device boot ID.
- Session ids 1,2; DETACH applies a KV reset; STOP does not claim a reset.
- Every token is exact.
- Phone compute is HTP0 except declared `GET_ROWS` on CPU.
- Host tail compute is CUDA0 with no fallback.
- Both route walls are at most 12,000,000 us.

No energy claim is part of this gate.

