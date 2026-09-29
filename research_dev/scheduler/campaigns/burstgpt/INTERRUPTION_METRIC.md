# Retained-session interruption metric v2

Schema: `s42-retained-session-call-gap-v2`.

The original pooled-call v1 result remains FAIL. V2 does not rewrite or
reinterpret that artifact. New gates also retain the v1 calculation as a
diagnostic, separately from v2 acceptance.

V2 joins each request's native FFN calls to the phone's exact session,
artifact, generation, and consecutive call counter. Physical identity and
terminal-proof checks are unchanged. Times used for gaps are phone monotonic
completion times, never desktop stderr arrival times.

Compare these classes separately:

- Within-token calls, keyed by the exact from-layer and to-layer pair.
- Same-layer calls on consecutive decode tokens, keyed by the layer.

Every class also matches active layer mask, column width, assistance fraction,
and batch width within the same request and desktop parent. A pair crossing
a control generation or mask/fraction change is not an equivalent interval.
Comparisons may cross control generations only when their physical masks
and widths are identical; individual intervals never cross such a boundary.

For every class overlapping LOAD_AUTHORIZED through READY, compare its
maximum with twice the median of 30 equivalent reference intervals outside
loading. Take the nearest preceding matching intervals, then the first
matching post-READY intervals if needed. Persist the exact reference rows
and the before/after counts. Fewer than 30 is INSUFFICIENT, never PASS.
No loading operation waits for a post-READY reference to be collected.
The forward transition can therefore start immediately after mask ACK.
Reverse probes collect 31 tokens' worth of retained-layer calls before the
replacement request to provide a pre-load reference when publication changes
the active mask. This measurement warmup is outside the drain latency.

Intervals crossing the start/end of loading are included when their class is
unchanged. Injected-failure measurements extend through physical rollback
READY, not just the failed target's first READY. All retained sessions must
pass. Missing identities, call events, or references fail closed.

Drain timing is separate: use the scheduler's requested/quiesced events,
direct decode-progress callbacks, and the HTTP control invocation timestamp.
All use the desktop measurement epoch. Physical load and READY retain their
original phone timestamps and the transition receipt's desktop timestamps;
do not subtract unsynchronized clocks. The immediate target is removal of
the adaptive measurement-window wait, not a faster physical weight load.
