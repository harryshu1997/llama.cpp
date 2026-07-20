# S16 OP12 Persistent Lane Results

Verdict: `OP12_PERSISTENT_B32_LANE_PASS`

OP12 is now an independently executable READY lane for a lower-priority 12 s
class. It is not eligible for the OP15 5 s class.

## Real route

- OP12: Gemma `[0,6)`, HTP0, B32, 8 generated tokens.
- Selected A6000: Gemma `[6,48)`, CUDA0.
- Exchange 1: DETACH, route wall 9,409,229 us.
- Exchange 2: STOP, route wall 9,143,931 us.
- Same host PID 3843832 and phone PID 12975 across both exchanges.
- Same worker boot nonce and device boot ID.
- Session ids 1,2 and cumulative steps 384,768.
- DETACH applies the request-local KV reset; STOP terminates without claiming a
  new reset.

All 64 request results exactly match the frozen CUDA reference. Phone compute
uses HTP0 except `GET_ROWS` on CPU. The v75 correctness containment works as
intended: attention is the explicit HTP path (`SOFT_MAX` is visible), not the
known-broken v75 fused flash-attention kernel. The host tail uses CUDA0 only.

HMX temperature rose from 31.8 C to 34.5 C. Energy was not measured.

The independent validator reopens the two host results, two host placement
certificates, two phone session certificates, reset sequence, process identity,
tokens, and executable/model digests. It passes with report SHA-256
`d748708c6934ff8cf19a350236a5235775df48f4af141d70990accae52e40b8b`.

Next: expose OP15 and OP12 as two independent scheduler lanes with different
SLO classes. Do not serially chain their layer ranges.

