# S15 live launcher results

Verdict: `COORDINATOR_TRIGGERED_PHYSICAL_MECHANICS_PASS_ARRIVAL_FAITHFUL_SLO_BLOCKED`

Date: 2026-07-18. No commit or push. No model graph, backend scheduler, kernel,
S14 policy, `layersplit.cpp`, talks log, or memory file was edited by this work.

## Result

The actual S15 coordinator replayed the frozen observed cohort at its logical
arrival times. It issued 31 `bounded_batch_wait` decisions and then one
`target_batch_ready` launch containing exactly 32 requests. `PhysicalExecutor`
sent the typed request to the prepared live launcher. OP15 ran Gemma layers
`[0,8)` at B32 and the selected A6000 ran the tail.

| Check | Physical result |
|---|---:|
| Cohort | 32 observed BurstGPT identities, one-second arrival span |
| Scheduler decisions | 31 WAIT + 1 B32 LAUNCH |
| Coordinator paid exchange | 3,774,834 us |
| Full transport reply | 3,940,857 us |
| Earliest logical deadline margin | 59,143 us |
| Route wall maximum | 2,926,074 us |
| Terminal ownership | 32/32 `completed_phone`, no duplicate |
| Placement | `SCHEDULED_PLACEMENT_OK` |
| HTP compute | 1,856 nodes |
| Allowed CPU work | 8 `GET_ROWS` nodes only |
| Missing-buffer compute | 0 |
| Correctness | 32/32 exact same-batch CUDA token streams |
| D2H/identity/epoch | 32/32 true |
| End HMX thermal maximum | 35.6 C, continuous sample age 247,428 us |
| Energy | UNKNOWN |

The full transport time is load-bearing: the first request arrives at logical
247 seconds, the batch launches at 248 seconds, and the earliest five-second
deadline is at 252 seconds. The reply arrives 59,143 us before that boundary.
This is useful timing evidence, but it is not an arrival-faithful SLO result:
the frozen host binary saw the fixed prompt during preflight, before the logical
arrival replay. `arrival_faithful_slo_claim=false` is enforced by the validator.

## Evidence path

- `result.json` records all scheduler decisions, terminal states, paid window,
  scope, frozen input identities, and digests of every raw artifact.
- `transport/request.json` is the exact typed execution request.
- `transport/stdout.bin`, `stderr.bin`, and `transport.json` preserve the exact
  physical executor exchange and completion elapsed time.
- `raw/launch-1/` contains the frozen input copies, preflight identities,
  CUDA reference, host/phone output, route records, placement certificate,
  paid window, and continuous thermal stream.
- `validate_result.py` reopens and rehashes these artifacts, independently
  revalidates the cohort and input manifest, and rejects any identity, batch,
  ownership, correctness, placement, D2H, timing, or thermal mismatch. It
  derives the exact WAIT/LAUNCH timeline and every request deadline from the
  cohort; a generic five-second transport bound is not accepted after the
  first request has already waited one second.

Independent validator output:

```text
VALID_S15_PHYSICAL_B32 paid_elapsed_us=3774834 transport_elapsed_us=3940857 deadline_margin_us=59143 energy=UNKNOWN arrival_faithful_slo=false
```

## Failed attempts retained as red evidence

1. `failed_run_transport_timeout/`: all 32 streams completed correctly, but the
   transport timed out at 5,000,163 us with empty stdout. A post-compute thermal
   snapshot remained on the paid critical path. The repair uses a continuous
   on-device thermal stream and emits the typed session before cleanup.
2. `failed_preflight_stdin_inheritance/`: the thermal subprocess inherited the
   launcher's stdin and consumed the 1,591-byte execution request, causing
   `request has an invalid byte length`. No GO was sent and no paid model request
   ran. All preflight helpers now use `stdin=DEVNULL`; a regression scans both
   helper paths.

Neither failed attempt is accepted as a completion.

## Tests

- Live launcher: 26/26 pass.
- S15 runtime dispatch: 78/78 pass.
- Independent physical result validator: pass.
- Tests cover exact cohort/input binding, 31-wait/B32 behavior, strict request
  parsing, stale epochs, wrong route/profile/device, partial stream sets, token
  mismatch, CPU fallback, duplicate keys, transport timeout, setup exclusion,
  thermal sensor identity, and launcher-stdin isolation.

## Claim boundary and next gate

Established: the real coordinator can wait for a compatible B32 cohort, launch
one physical OP15/A6000 split route through the typed executor, and accept only
an exact, placement-certified, thermally valid, uniquely owned completion.

Not established: arrival-faithful SLO, repeated-run latency statistics, OP12 or
two-phone concurrent live dispatch, selected-GPU energy relief, phone/USB energy,
or total-system energy. The next executable gate is to build and independently
recertify the already-designed prompt-after-load seam, then repeat the observed
cohort run with the prompt becoming visible only at logical admission. Energy is
a separate matched-control experiment after that gate.
