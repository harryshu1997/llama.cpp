# W7 cold-promotion result

Verdict:
`NEW_REQUEST_PHONE_PREFILL_CANNOT_HIDE_PROCESS_COLD_CUDA_LOAD`.

Scope remains `MECHANICS_ONLY`. The Qwen2.5 Q8 route remains
scheduler-ineligible because the separate task-quality gate failed.

## Frozen attempts

W7 froze B8, a four-token minimum phone service window, a 24-token maximum,
a 120-second CUDA readiness bound, a 250-millisecond launch-delay bound, and
the W6 two-token concurrent delta plus eight-token CUDA continuation.

The first run is:

`results/w7_cold_promotion/run_20260725T005353Z/`

The request started before either CUDA model process. The phone route and both
fresh CUDA workers executed, all sequence state returned to zero, and the
ownership journal reached `COMPLETE`. The treatment failed its readiness-order
gate because no useful phone token was ready before CUDA.

R1 did not relax the gate. It added an explicit B8 two-token phone preparation
session, `DETACH` reset, and a same-resident-worker requirement. The R1 run is:

`results/w7_cold_promotion_r1/run_20260725T010308Z/`

Both phone workers prove the intended persistence:

| Device | Preparation | Paid treatment | Worker identity |
|---|---:|---:|---|
| OP15 | session 1, 72 steps, `DETACH` | session 2, 104 steps, `STOP` | same PID and nonce |
| OP12 | session 1, 72 steps, `DETACH` | session 2, 104 steps, `STOP` | same PID and nonce |

Preparation executed four B8 prefill calls and one B8 decode call. It took
8,901,039 us of measured route compute; the first token was ready 7,365,805 us
after preparation started. The paid run again failed the unchanged
readiness-order gate. Both result manifests revalidate byte-for-byte.

## Meaning

This is a negative regime result, not a protocol failure. On this A6000 host,
the Qwen checkpoint remains in the host page cache. Two process-cold CUDA
workers can load and become ready before the prepared collective-phone route
finishes a new B8 prompt prefill. Therefore a new request cannot receive a
phone token during this model-load interval.

The next bounded experiment is an ongoing-session promotion. Prompt history
and phone KV are established before the model switch; the paid interval begins
at the next decode token. That directly tests service continuity for live
sessions, which is the stateful catch-up use case. New-request prefill is not
claimed to hide the measured process-cold load.

No latency, scheduler, task-quality, or energy pass is claimed from W7.
