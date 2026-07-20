# S12-V1 Asymmetric Dual-Path Scheduler

## Physical Topology

The host and both phones are on one WiFi LAN while each phone also has its own
USB data connection to the host:

```text
                              shared WiFi H2P domain
                         +----------------------------+
                         |                            |
server virtual queue ----+----> OP15 input buffer     +----> OP12 input buffer
                              | HTP / GPU |                 | HTP / GPU |
                              +-----+-----+                 +-----+-----+
                                    |                             |
                          USB P2H, Bus 008               USB P2H, Bus 006
                                    |                             |
                                    +------------+----------------+
                                                 v
                                      host result/tail queues
                                                 |
                                             A6000 tails
```

The WiFi server-to-phone path and USB phone-to-server path are different
physical resources. This makes overlap possible in the topology, but the
current runtime cannot safely expose it. Both paths can still contend for
phone CPU, LPDDR, thermals, and the server memory path.

The current executable S12-V1 slice has one eligible complete route, OP15
`A0_OP15`. OP12 remains in the topology plan but is not dispatchable by this
replay because there is no standalone complete-route profile for it. A future
fleet profile must bind every route to its phone and USB root-bus domain.

## Fast-Loop State Machine

For each admitted phone batch, the scheduler executes one prefill quantum and
then three decode quanta for the current S11 profile. Every quantum follows:

```text
HOST_VQ
  -> WIFI_WAIT
  -> WIFI_H2P
  -> PHONE_INPUT_BUFFER
  -> PHONE_EXEC
  -> PHONE_RESULT_BUFFER
  -> USB_P2H
  -> HOST_RESULT_BUFFER
  -> A6000_TAIL_WAIT
  -> A6000_TAIL
  -> COMPLETE
```

`COMPLETE` means the quantum is complete. Prefill tail completion produces the
first decode token and unlocks decode quantum 0; each decode tail unlocks the
next decode quantum. Request completion follows only the final decode tail.
Future decode inputs are never admitted into WiFi early. S12-V1 permits only
one KV-owning group on OP15, so different groups do not occupy phases
concurrently.

Completion is legal only after the USB result and A6000 tail both complete.
The scheduler reserves bounded credits for:

- phone in-flight groups;
- phone ingress bytes;
- phone result bytes;
- host returned-result bytes;
- the phone compute lane;
- the WiFi H2P domain;
- the per-phone USB P2H domain; and
- the A6000 lane and HBM used by the tail.

Returned results feed the A6000 tail immediately after the lane is available.
The A6000 is not compute-reserved while the group is on WiFi, on the phone, or
returning over USB. S12-V1 does not fill that idle time with full-model work,
because the tail-only and full-model measurements came from separate process
residencies.

## Executable Residency And KV Limits

Every V1 replay freezes one host residency for the complete horizon:

```text
server_only_optimized / causal_server_batch: FULL_MODEL, server routes only
fixed_phone:                              TAIL_ONLY, A0_OP15 routes only
```

The resident allocation is charged even while its compute lane is idle. V1
rejects the old `memory_admission_triggered` policy because switching between
these modes would require measured load, unload, synchronization, and durable
residency transitions. Keeping both allocations resident would erase the
claimed HBM relief.

The measured LayerSplit stage also has one `llama_context` and one KV
namespace. A new group resets that context and sequence IDs restart at zero.
Therefore `phone_inflight_group_limit` is exactly one. Cross-group link overlap
requires multiple leased contexts or an explicit KV spill/restore mechanism,
including capacity and epoch accounting. Neither exists today.

## Current Route Byte Semantics

The existing OP15 route owns layers `[0,2)`, including token embedding. Its
server-to-phone input is token IDs and sequence metadata, not a dense
activation. Its phone-to-server result is the dense F32 cut activation.
Therefore the two byte counts must not be copied from one field:

```text
WiFi input payload = prefill command/tokens + three decode command rows
USB result payload = 31 x B x 3840 x sizeof(float) + response headers
```

At `B=8`, these are 1,236 bytes and 3,809,312 bytes respectively. A middle
operator island can instead have dense activation input and output; its route
profile must carry both exact byte counts explicitly.

The payload formulas exclude TCP/IP, WiFi, USB, and ADB framing. The physical
measurement must count wire or socket bytes separately and say which boundary
it measures.

## Slow Loop And Weights

S12-V1 assumes weights are already resident. It does not yet compose the S9
weight-prefetch state machine with the fast loop.

The combined design uses explicit traffic classes:

| Traffic | Default path | Scheduler class |
|---|---|---|
| manifests, commands, token IDs, sequence metadata | WiFi H2P | latency-critical |
| large model segments and prepared-weight images | USB H2P | resumable background bulk |
| dense cut activations and final phone results | USB P2H | completion-critical |

This is a default route, not a byte-only rule. A route profile must still bind
the exact payload, queue delay, goodput, fixed latency, and deadline. A dense
middle-island input may belong on USB rather than WiFi if its measured WiFi
cost misses the request budget.

The intended priority is:

```text
USB result return > activation-critical control > background weight prefetch
```

If weight prefetch continues to use USB H2P, it shares a physical USB device
with result P2H even though the directions differ. Do not assume full-duplex
capacity. Result traffic must preempt or pause background weights until a
simultaneous directional measurement supports a less conservative rule. WiFi
activation does not preempt USB weight traffic because it uses another path.

The slow loop uses virtual-queue demand to prefetch reusable weight segments
before their islands reach the fast-loop frontier. Prefetch is opportunistic:
it consumes only unreserved USB credits, is resumable at verified chunk
boundaries, and never delays an admitted result return. A ready certificate is
published only after the complete weight set and prepared backend image pass
the S9 identity and durability checks.

## Evidence Boundary

The checked-in dual-path fixture uses assumed path rates and treats the old
S11 `phone_stage_us` as a conservative phone-phase proxy. That value includes
the old single-socket transaction and is not a decomposed compute measurement.
V1 divides that aggregate across prefill/decode quanta in proportion to their
activation rows only to exercise causality. It does not call the split a
measured kernel time.
Consequently every V1 output is:

```text
ASSUMED_MECHANICS_ONLY_UNMEASURED
PHONE_STAGE_WALL_PROXY_NOT_DECOMPOSED
ENERGY_NOT_RUN
```

Before a latency, capacity, or energy comparison, measure:

1. WiFi H2P latency and goodput per phone for the real input sizes;
2. shared-WiFi contention with both phones active;
3. native USB P2H result goodput per phone and root bus;
4. simultaneous WiFi H2P plus USB P2H;
5. phone compute slowdown under each path alone and both together; and
6. USB result interference with background USB weight streaming.

Runtime realization also needs two correlated connections. The current
LayerSplit stage uses one bidirectional socket and cannot receive on WiFi while
returning the corresponding result over an independently forwarded USB socket.
Both connections must bind the same request ID and boot, residency, route,
state, and sequence epochs. Exposing overlap additionally requires multiple
KV-isolated contexts or a measured state-switch mechanism; two sockets alone
are insufficient.
