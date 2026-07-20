# S10-E2A R4 Route Contract

R4 repairs two fail-open paths reproduced after the R3 review:

1. a route digest was an opaque label rather than a commitment to a route; and
2. decorative phone work could coexist with a server execution that actually
   produced the result.

The v1-v3 records remain historical. Active E2A records use schema version 4.
No physical measurement or energy claim is authorized.

## Route artifacts

The bundle resolves one control and one treatment `RouteSchedule` through the
same dirfd-relative, read-once artifact boundary used by all other evidence.
The anchored plan pins each route record digest.

A route record freezes:

- role, request manifest, model, and server/phone device sets;
- the exact action IDs and immutable action attributes;
- request coverage, byte requirements, duration requirements, and lease
  requirements; and
- the exact control/data dependency edges and payload identities.

Node IDs and edge IDs are unique, every endpoint resolves, every request is in
the manifest, and the graph is acyclic.

## Lifecycle matching

The realized lifecycle must contain exactly the route's action IDs. Missing and
extra actions fail separately. Each action must match its route node on kind,
execution domain, device, backend, operator island, model, and request set.
Lifecycle validation accepts only the resolver-issued route object, so a caller
cannot bypass route validation by handing it an unverified dictionary.
The resolver stores canonical route bytes rather than the caller's mutable
dictionary and reparses a private snapshot for each lifecycle validation.

Zero/positive byte requirements and lease presence are enforced. `EXEC` and
nonzero H2D/D2H work require positive duration. Every route edge is replayed as
an ordering constraint: the source ACK must precede the destination start.

The existing exact lease coverage, paid-window, request-arrival, output-timing,
drain, and clean-state checks remain active.

## Phone-result causality

The control route cannot contain a phone domain, phone device, HTP/OpenCL
backend, or phone-assisted request.

For every treatment request declared phone-assisted, the route must contain a
request-bound DATA path:

```text
phone H2D -> phone EXEC -> phone D2H -> optional server continuation -> result
```

The H2D, phone EXEC, and D2H use the same phone. The phone EXEC uses HTP or
OpenCL, all three nodes bind the request, and their byte/duration requirements
are positive. A server continuation is valid only as a descendant of the phone
D2H; it cannot replace the phone-result path.

This is an internal evidence contract. A witnessed launcher is still required
to prove that physical execution followed the committed route.
