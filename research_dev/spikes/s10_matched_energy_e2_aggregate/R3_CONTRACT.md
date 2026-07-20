# S10-E2A R3 Contract (Historical)

R3 repairs the claim-path defects found by independent review after R2. The v1
and v2 schemas remain historical. R3 is retained as review history and is
superseded by R4: a later audit proved that its route digests were opaque labels
and that decorative phone work could pass without contributing to the result.
Active E2A records use schema version 4; see R4_CONTRACT.md.

R3 is still not a physical measurement. It must fail closed until an independent
enumerable commitment verifier, a witnessed acquisition launcher, and a valid
power instrument exist.

## Bundle

`E2ABundle` is schema-checked before use. Its measured slots must:

- number exactly `2 * plan.n_pairs`;
- have integer indices exactly `0 .. 2N-1`;
- contain no duplicate index; and
- contain only the declared path and digest fields.

The bundle index is bootstrapped from one securely opened buffer. It is not
self-hashed.

## Artifact boundary

The resolver opens the trusted root once and retains its directory file
descriptor for the resolution. Each artifact component is opened relative to
the preceding directory descriptor with `O_NOFOLLOW`; the final object must be a
single-link regular file. The same descriptor supplies the bytes, hash, and
parser input. Metadata is checked before and after the read.

The frozen E2 timeline and wall-capability validators consume these buffers
through a locked read adapter. They do not reopen artifact paths.

## Same work

Every run-specific output artifact carries and binds:

- input digest;
- prompt-token digest;
- tokenizer and model digest;
- sampling mode and seed;
- decode-parameter digest;
- stop-set digest; and
- realized token events and stop reason.

The manifest, outcome record, and output artifact must agree. A manifest rewrite
cannot retain an old output artifact.

## Route evidence

The anchored plan pins distinct control and treatment route digests plus exact
server and phone device identities. Every lifecycle action binds the route
digest, execution domain, device, backend, byte counts, lease, and optional
operator-island identity.

Control refuses every phone action. Treatment requires at least one request-
covered phone `EXEC` on HTP or OpenCL, with nonzero input and output work,
bracketed by phone H2D and D2H actions. Every compute action binds the planned
model and an operator-island digest.

This makes a false server-only treatment structurally ineligible. It does not
make a fabricated acquisition artifact truthful; the future witnessed launcher
owns that boundary.

## Wall capability

A `SERVER_WALL` plan pins one E2 `ServerWallCapability` record. The bundle must
resolve it, and E2's complete-rail, proof, provenance, and uncertainty checks run
against every timeline. A GPU-board plan must not carry a wall capability.

The current E2 capability version deliberately refuses measured wall claims as
uncertified. R3 closes the bypass; it does not create calibration evidence.

## Warmups

R3 supports exactly zero warmups. Any nonzero `warmup_count` or lifecycle
`WARMUP` action is `E_WARMUP_UNBOUND`.

Supporting warmups later requires predeclared warmup slots and exact ledger,
lifecycle, route, and output evidence. Hidden or unlogged conditioning is never
accepted.

## Enumerable commitment

The anchor verifier result and commitment proof now bind:

- externally assigned experiment identity;
- commitment namespace and log identity;
- authenticated checkpoint;
- identity binding;
- completeness status;
- committed plan count; and
- digest of the complete enumerated plan set.

The resolved set must be exactly `[plan.record_sha256]`. Plan and close receipts
must agree on the experiment and enumeration identity.

No verifier is registered, no trust root is pinned, and UTC-to-monotonic
precedence remains unsupported. Therefore the production fixture still refuses
before any physical label.
