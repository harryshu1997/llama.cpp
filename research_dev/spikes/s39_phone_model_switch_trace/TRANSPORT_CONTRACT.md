# S39 transport contract

Status: W2 direct mixed-batch mechanics pass; measured path and reservation
binding remain provisional.

S39 separates large, infrequent weight movement from latency-sensitive runtime
traffic.

## Frozen split

| Payload | Path | Timing |
|---|---|---|
| Model shards and runtime binaries | USB ADB | Before a paid serving window |
| Model reads during execution | Phone-local UFS into phone memory | Worker preparation |
| Commands and request metadata | WiFi TCP | Runtime |
| Intermediate hidden states | WiFi TCP | Runtime |
| Tokens and completion metadata | WiFi TCP | Runtime |

Weights must never be streamed over WiFi in the first proof of concept. A paid
run must report zero runtime weight bytes. USB provisioning ends only after the
phone artifact SHA-256 matches the frozen shard manifest.

The W0 Qwen run used this topology:

```text
USB provisioning:

A6000 host -- USB ADB --> OP15 UFS
A6000 host -- USB ADB --> OP12 UFS

Runtime activation path:

OP15 [0,30) -- WiFi TCP --> host coordinator -- WiFi TCP --> OP12 [30,40)
```

This W0 run is a host relay, not direct phone-to-phone transfer. It is retained
as the correctness baseline.

The target runtime data plane is:

```text
host -- reserve(batch, epoch, credits) --> OP15 and OP12
OP15 [0,30) -------- activation -------> OP12 [30,40)
OP12 ---------------- tokens ----------> host
```

The host remains authoritative for request identity, admission, downstream
credits, route epochs, and final token publication. OP15 cannot send work until
OP12 capacity is reserved. OP12 rejects an activation whose model, batch,
request, epoch, position, cut, shape, or integrity record differs from the
host-issued descriptor.

Direct transfer removes the host activation copy. At B32, total
application-level activation payload falls from 15,728,640 bytes across two
relay legs to 7,864,320 bytes across one direct leg, while activation payload
handled by the host falls to zero. This does not necessarily halve WiFi
airtime because an access point may still forward each phone-to-phone frame.
Host relay and direct transfer must therefore be measured as separate
controls.

## W1 executed direct path

W1 executes the target data path with an additive relay process on OP15:

```text
host -- token and lineage rows --> OP15 relay
OP15 head -- F32 activation --> OP12 tail
OP12 tail -- terminal tokens --> OP15 relay --> host
```

The relay validates both worker hellos and identities before listening to the
host. It rejects a non-contiguous cut, different model digest or GGUF type,
missing capability, capacity overflow, lineage mismatch, status mismatch, or
live sequence at drain. B1 and B32 completed on real devices with zero host
activation payload and exact terminal tokens.

W1 does not yet implement the host-issued descriptor described above. One host
connection implicitly owns the worker pair for one session. This is adequate
for the bounded mechanics proof, but not for concurrent multi-route dispatch
or stale-frame protection. The reducer therefore reports
`IMPLICIT_SINGLE_CLIENT_MECHANICS_ONLY`.

## W2 mixed-phase use

W2 keeps the same direct data plane and adds route-local continuous admission.
One physical 96-row call carried 16 existing decode rows and 80 new prefill
rows. The relay forwarded the same row order and lineage through both workers:

```text
host token rows -> OP15 [16 decode + 80 prefill]
OP15 activation rows -> OP12 [same 16 decode + 80 prefill]
OP12 token rows -> host
```

Weights remained on phone UFS and no weight bytes moved during service. Across
the nine calls, OP15 sent 384 F32 activation rows directly to OP12:

```text
384 x 5,120 x 4 = 7,864,320 bytes
```

The current relay is synchronous. It waits for OP12 to finish batch `k` before
accepting batch `k+1`, so W2 proves mixed-phase batching but not inter-stage
pipeline overlap. The next protocol must add a finite batch ID space,
downstream row credits, bounded outstanding activation bytes, ordered
completion ownership, and drain semantics before multiple batches can be in
flight.

## W0 evidence boundary

The three W0 reports bind worker model identity, layer ranges, batch events,
latency, and token output. Worker logs bind persistent process identity,
session reset, placement, and log-reported buffer sizes. The exact activation
payload is derived from the executed row counts and 5,120-element F32 boundary:

```text
bytes per row = 5,120 x 4 = 20,480
B32 rows      = 32 x (5 prompt + 7 decode) = 384
one WiFi leg  = 384 x 20,480 = 7,864,320 bytes
two WiFi legs = 15,728,640 bytes
```

`RUN_CONTEXT.json` records the endpoints used by the operator. It was written
after acquisition and no interface counter or socket-peer artifact was
captured. Therefore the reducer labels the path
`posthoc_operator_record_no_interface_counter` and cannot promote the route to
`PASS`.

## Next acquisition gate

The next wrapper must freeze these facts before starting the workers:

1. phone serial, boot ID, WiFi address, route, and interface name;
2. exact worker command and TCP endpoint for each stage;
3. absence of ADB forward/reverse tunnels;
4. interface byte counters immediately before and after the paid window;
5. USB provisioning completion and remote artifact hashes;
6. start/end monotonic timestamps shared by the route report and counters.
7. host-issued downstream credits and the direct-frame descriptor digest.

The verifier must reject endpoint changes, counter rollback, missing brackets,
unexpected USB runtime payload, a worker identity change, an unreserved direct
frame, or a completion that does not match the descriptor.
