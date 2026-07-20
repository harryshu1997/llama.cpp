# S15 persistent host-tail results

Verdict: `PERSISTENT_OP15_B32_HOST_AND_PHONE_PASS_ENERGY_UNKNOWN`

Date: 2026-07-18. No commit or push. Energy scope: UNKNOWN.

## Physical result

One OP15 `[0,8)` stagenet worker and one selected-A6000 `[8,48)` tail process
served two separate B32 exchanges. Session 1 ended with DETACH and an
acknowledged phone KV reset. Session 2 ended with STOP. Neither model nor
context was reloaded between exchanges.

| Metric | Exchange 1 | Exchange 2 |
|---|---:|---:|
| Session end | DETACH | STOP |
| C++ exchange elapsed | 2,798,729 us | 2,598,388 us |
| Route wall | 2,726,200 us | 2,558,821 us |
| Requests | 32 | 32 |
| Generated tokens | 256 | 256 |
| Phone cumulative steps | 384 | 768 |
| Correctness | 32/32 exact | 32/32 exact |
| Host compute placement | 9,288 CUDA0 nodes | 9,288 CUDA0 nodes |

The A6000 tail PID was 3788992. The OP15 worker PID was 19868 with nonce
`984c31b0c384bce2`. OP15 placement was HTP0 for all compute except declared CPU
GET_ROWS. Both host exchanges observed zero missing compute buffers and no
fallback. HMX temperature rose from 30.2 C to 38.7 C.

The independent validator reopens the raw result and certificate bytes, checks
the frozen executed binaries/source, and rejects identity, reset, step,
placement, token, and latency mutations:

```text
VALID_PERSISTENT_HOST_TAIL sessions=2 host_pid=3788992 worker_pid=19868 energy=UNKNOWN
```

`test_gate.py` passes 9/9. The C++ input-seam suite passes 9/9, including
duplicate-key rejection and unsafe persistent-mode option gates. CPU, CUDA, and
Android builds pass.

## Implementation

`llama-layersplit --mode pipedriver --persistent-jsonl` now loads the model and
context once, then accepts bounded flat JSON commands. Each command carries a
strictly increasing launch ID, prompt, generated-token limit, exact request
count, and DETACH or STOP ending. The fixed batch comes from
`--driver-batch`; per-command request counts must match it.

The implementation reuses `run_pipebatchdriver`, clears host and phone KV at
the start of every exchange, suppresses human stdout, and emits one canonical
result line. A `PERSISTENT_DRIVER_EXCHANGE_END` marker closes the per-exchange
stderr interval after all diagnostics and placement evidence. Malformed input,
duplicate launch IDs, incomplete results, missing host placement, failed
compute, or DETACH failure stop the phone rather than releasing a reusable
lease.

## Claim boundary

This is a real persistent execution-mechanics pass, not an energy pass. The
runner writes directly to the C++ JSONL seam, so the typed scheduler transport
and session adapter have not yet admitted this physical result. The workload is
two copies of one synthetic B32 prompt, not the frozen mixed BGE+Gemma trace.
Phone, USB, host-wall, and total-system energy remain unknown.

The next gate is two physical exchanges through the typed persistent launcher.
Only after that passes should the repeated mixed-workload selected-GPU energy
experiment begin.
