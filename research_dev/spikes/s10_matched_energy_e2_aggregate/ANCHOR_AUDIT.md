# S10-E2A Anchor Capability Audit

Measured first-hand on this host, 2026-07-15. This document is the E2A analogue of
E2's INSTRUMENT_AUDIT.md, and it plays the same role: it decides what the checkpoint
is ALLOWED to claim, before any code is written.

E2's audit found that no server-wall instrument exists, so SERVER_RELIEF is
unreachable without new hardware. This audit asks the parallel question for
integrity rather than physics:

    Is there an INDEPENDENT EXTERNAL ANCHOR that can bind the experiment plan
    before the runs, and the attempt ledger after them?

Answer: NO. See section 5.

---

## 0. Why an anchor is needed at all

E2A's aggregate is `SUM_ALL_PAIRS_V1` over EVERY planned pair. That rule is only
worth anything if "every planned pair" is fixed BEFORE the results are seen.
Otherwise the producer runs 20 pairs, keeps the best 8, and writes a plan that
declares exactly those 8. The sum is then honest arithmetic over a dishonest set:
the statistic is frozen, but WHICH RECORDS it is taken from is not. That is E1's
record-cherry-picking lesson, one level up.

So the plan needs a commitment with two properties:

- **P1 (precedence).** The plan digest existed before the first attempt ran.
- **P2 (uniqueness / non-selection).** Exactly ONE plan was committed for this
  experiment identity. The producer cannot commit many plans and reveal only the
  favorable one.

A local file cannot give either: the producer can write any bytes at any time and
restate any timestamp. The anchor must come from a party that does not answer to
the experiment.

## 1. What is actually on this host

| Candidate | Present? | Independent? | Proves | Enumerable? |
|---|---|---|---|---|
| RFC3161 TSA (`openssl ts` -> freetsa.org) | client yes; TSA reachable and **grants tokens** | **YES** | precedence (P1) | **NO** |
| TPM 2.0 (`/dev/tpm0`, `/dev/tpmrm0`) | device nodes exist, **permission denied** | no (custody) | order only | no |
| GPG | binary present, **no `user.signingkey`** | no | nothing | no |
| git remote | `github.com/harryshu1997/llama.cpp` | no (**own fork, force-pushable**) | nothing | no |
| Transparency log inclusion proof (CT / Rekor) | absent | yes | one-leaf precedence | **NO without an externally bound identity and completeness query** |
| Roughtime | absent | - | - | - |
| BMC / IPMI event log | absent (consistent with E2's instrument audit) | no (same admin domain) | - | - |
| NTP / chrony | synchronized, active | **not an anchor** (see 3) | - | - |

The two columns that matter are the last two, and they do not covary. Read section 2
before concluding anything from "Independent? YES".

### 1.1 TPM: present but inaccessible

    crw-rw---- 1 tss root  10,224   /dev/tpm0
    crw-rw---- 1 tss tss  252,65536 /dev/tpmrm0
    $ dd if=/dev/tpmrm0 bs=1 count=0
    dd: failed to open '/dev/tpmrm0': Permission denied

The experiment user is not in group `tss` (verified with `id`), and no tpm2-tools
binaries are installed. The TPM cannot be exercised at all from this account.

Worth stating even if access were granted: a TPM quote proves a measurement was
produced by a genuine TPM whose attestation key chains to the manufacturer's EK
certificate. It gives a monotonic counter, so it can enforce P1-style ordering. It
does NOT give P2: nothing stops the producer from incrementing the counter over
many candidate plans and disclosing one receipt. A TPM anchors *sequence*, not
*exclusivity*.

### 1.2 RFC3161: genuinely available, genuinely insufficient

This is the entry that matters, and an earlier draft of this audit got it wrong by
calling it merely "unprovisioned". Correction, recorded because the distinction is
the whole point of the checkpoint:

**A live probe during the design review confirmed freetsa.org returns `Status:
Granted`** (serial `0x0640067A`, `Ordering: yes`, `Accuracy: unspecified`), and the
reply chain **verifies** once `cacert.pem` is fetched from freetsa.org. So RFC3161
is not blocked by capability. It is roughly ten minutes of provisioning away:
fetch the CA, pin it, call `openssl ts`.

(Provenance note: that probe was made by a subagent during the design review, not
by the author of this document. It sent a hash of throwaway test data - not a plan
digest, not experiment data - to a third party. It is recorded here because it
corrects the audit, and disclosed because it was an unplanned external call.)

RFC3161 is also genuinely **independent**: the signing key belongs to a third party
the experimenter cannot compel, and the genTime is that party's assertion, not
ours. On the spec's own test - "local timestamps, self-hashes, writable files,
self-signed records, and experiment-owned HMACs are not independent anchors" -
RFC3161 passes. It is a real external verifier.

**It is still not enough, and section 2 explains why.** The short version: it proves
precedence (P1) and says nothing about exclusivity (P2). Recording this correctly
matters more than recording it pessimistically, because it produces a much more
useful conclusion for whoever plans the real experiment: *provisioning a TSA would
not unblock E2A*. Money and time spent on timestamping would buy the property that
was never in doubt.

Two remaining local hazards, worth keeping on the record:

- No TSA CA certificate is provisioned on the host today (`/etc/ssl/certs`
  contains no freetsa or TSA root). Without a **pinned** root, a timestamp reply
  cannot be verified against anything, and any party able to answer the request -
  including the producer running a TSA on localhost - would be accepted. The pin
  must be a code constant, not a `-CAfile` parameter.
- The only `[tsa]` section on the host is OpenSSL's stock example:

        [ tsa_config1 ]
        dir        = ./demoCA          # TSA root directory
        signer_cert    = $dir/tsacert.pem
        signer_key = $dir/private/tsakey.pem
        default_policy = tsa_policy1   # OID 1.2.3.4.1

  Those are placeholder OIDs (`1.2.3.4.1`) pointing at a local `./demoCA`. This
  config does not configure a client to trust an external authority; it configures
  the host to BE a timestamp authority. An experiment that timestamps its own plan
  with its own `demoCA` has produced an experiment-owned HMAC with extra steps -
  exactly what the E2A spec names as not an anchor.

### 1.3 git / GPG: self-owned

The remote is the experimenter's personal fork. A push can be force-rewritten or
the repo deleted; there is no external party that would notice or object. No
signing key is configured, so commits are not even self-signed. GitHub does record
push events, but a personal fork's history is not a public auditable commitment
and enumerating "every plan this user ever pushed" is not something a reviewer can
do. Self-owned: not an anchor.

## 2. The decisive question: does a non-auditable TSA close selective disclosure?

This is the question the whole checkpoint turns on, so it is worth being exact. It
is not hypothetical: section 1.2 establishes that a working, verifying RFC3161
token is about ten minutes of provisioning away.

So assume the strongest honest case: freetsa's CA pinned, `openssl ts` working, a
verified RFC3161 token over the plan digest, genTime genuinely before the runs.

That buys **P1 and only P1**. An RFC3161 token proves "this digest was presented to
the TSA before time T". It is a signed statement about ONE digest. The TSA does not
publish, enumerate, or cross-link the tokens it issues; there is no auditable log a
reviewer can query for "all tokens issued to this requester".

So the producer can:

1. Write 20 candidate plans (different pair counts, different workloads, different
   orderings).
2. Timestamp ALL 20 before running anything. Every token is genuine.
3. Run everything.
4. Reveal the single token whose plan happens to fit the favorable results.

Every check passes. The revealed token is authentic, its timestamp genuinely
precedes the runs, and the plan digest genuinely matches the disclosed plan. The
gate sees a perfect anchor. **P2 is untouched.**

Closing P2 needs a property RFC3161 does not have:

- an **auditable/enumerable log** plus an externally assigned experiment identity
  and a completeness query (a producer-selected inclusion proof shows one leaf,
  not every plan submitted under aliases), or
- **third-party pre-registration** where an outside party holds the plan and can
  attest it is the only one (OSF/AsPredicted-style registration), or
- a **witness on a machine the experimenter does not control** that records the
  submission.

None of these exist here. This matters beyond this host: it means "we timestamped
the plan" would NOT be sufficient even after provisioning a TSA. A reviewer should
treat an RFC3161-only anchor as evidence of precedence, never of exclusivity.

## 3. Things that look like anchors and are not

- **A synchronized clock.** `timedatectl` reports NTP active and the clock synced.
  Irrelevant: a correct clock does not stop a producer writing whatever integer it
  likes into a JSON field. Clock accuracy is not commitment.
- **A self-hash.** A record hashing itself proves internal consistency, nothing
  about time or exclusivity.
- **A writable file.** Every path under this spike is owned by the experiment user
  (`drwxr-xr-x zs89458 myid`). Anything the producer can rewrite, the producer can
  rewrite after seeing results.
- **An experiment-owned HMAC or signature.** Same key, same owner, same objection.
- **An "external verifier" name string.** E2's typed-capability lesson: a free-form
  label grants nothing. Independence must be a closed, typed property, not a name.

## 4. What would unblock a real anchor

In rough order of cost:

1. **Third-party pre-registration** of the plan digest (OSF, AsPredicted, or an
   institutional registry). Closes P1 and P2 - this is the only entry here that
   closes P2 on its own, because the registry is enumerable per identity.
2. **A transparency log with inclusion + consistency proofs, an externally bound
   experiment identity, and a completeness query** (Trillian/Sigstore Rekor is
   only a substrate). Closes P1; closes P2 only if aliases cannot hide entries.
3. **A witness process on a host the experimenter does not administer**, recording
   plan digests as they arrive. Closes P1; closes P2 if its log is append-only and
   readable by the reviewer.
4. **TPM access** (add the user to `tss`, install tpm2-tools). Closes P1 ordering
   only. Cheapest, weakest.
5. **A provisioned RFC3161 TSA** (pin freetsa's CA). Closes P1 only. Does NOT close
   P2 - see section 2.

Note the ranking: the cheap options buy precedence, and precedence is the property
that was never really in doubt. The expensive option buys exclusivity, which is the
one that actually protects the aggregate.

## 5. Frozen verdict

**No anchor with the property the aggregate requires exists on this host - and for
RFC3161, provisioning one would not change that.**

Stated precisely, because the imprecise version is misleading in both directions:

- **RFC3161 is independent and available.** It proves P1 (precedence). It is not
  enumerable, so it cannot prove P2 (exclusivity). `ORDERING_ONLY`.
- **TPM**: present, permission denied, no tools. Would prove order only, and fails
  on custody anyway - `tss` is grantable by a root the experimenter can become.
- **git/GPG**: experiment-owned, unsigned, force-pushable. Proves nothing.
- **Transparency log / registry / third-party witness**: absent. These are the only
  shapes that would close P2.

The aggregate requires `ORDERING_AND_ENUMERABLE`. Nothing on this host provides it.
The checked fixture currently reaches `E_ANCHOR_TRUST_ROOT` first because no root
is pinned. Supplying a root would not be enough: no cryptographic verifier is
registered, no witnessed launcher relates an authenticated UTC plan time to the
local monotonic attempt ledger, and neither RFC3161 nor a producer-selected log
inclusion proves exclusivity. No physical label is reachable. Maximum verdict for
this checkpoint:

    E2A_AGGREGATE_MECHANICS_PASS_EXTERNAL_ANCHOR_BLOCKED

This is not a defect in the mechanism. It is the mechanism reporting, correctly,
that the integrity precondition for a physical claim is absent - the same shape as
E2's finding that the physical precondition (a wall instrument) is absent.
