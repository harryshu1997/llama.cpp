# S10-E2A Results

## Current R2 verdict - 2026-07-16

    E2A_R2_TARGETED_MECHANICS_PASS_PHYSICAL_CLAIM_BLOCKED

This section supersedes the V0 status, open-findings list, and reproduction
counts retained below as historical review evidence. No measurement was run. No
physical energy or relief label is authorized.

The R2 repair closes the confirmed fail-open paths found after V0:

- v2 records bind the plan, exact timeline fields, request manifest, lifecycle,
  outcomes, ledger terminal entries, plan anchor, and full-ledger close receipt;
- the ledger is an exact `OPEN -> START -> END` state machine whose clock epoch,
  monotonic START/END, paid window, and drain timestamp join the measured run;
- run-specific output artifacts are parsed, so token count, token-ID digest,
  arrival, dispatch, last-token time, SLO outcome, and cross-role prefix
  certificate are recomputed instead of trusted;
- non-vacuous correspondence is frozen at prefix >=32 tokens, length difference
  <=4 tokens, and all requests SLO-met until a typed SLO policy is implemented;
- lifecycle replay requires arrival-bound queue submission, EXEC coverage through
  the final token, RESULT_EMIT after EXEC, a lease that covers every resource
  action, one final drain, and clean entry/exit state;
- lifecycle and same-work checks accept only request evidence issued by
  `resolve_requests`; a raw record or forged three-field wrapper cannot disable
  the per-request causal joins;
- one bundle resolution uses one immutable byte buffer per canonical path and
  rejects cross-run evidence-path reuse, overlapping windows, and reused evidence;
- E2 modules are loaded from pinned source. E2's transitive `e2_canon` pyc path is
  replaced by an adapter to E2A's already pinned, source-compiled E1 canon;
- the independent checker is arithmetic-only, rejects supplied physical labels,
  and never prints a physical result.

Verification after the repair:

```text
E2A unit/adversarial tests       183/183 PASS
R2 targeted regressions          19/19 PASS
E2A CLI negatives                32/32 PASS
E2 official suite               152 tests + 30 CLI negatives PASS
E1 official suite               201 tests + 39 evidence negatives
                                + 18 CLI negatives PASS
E1 differential                 1187 compared, 0 mismatches
E1/E2 pinned baseline files     45/28 unchanged before and after E2A
production fixture              refused at E_ANCHOR_TRUST_ROOT
```

The remaining blockers are external and explicit: no pinned independent trust
root, no registered cryptographic verifier, no witnessed launcher that orders an
authenticated plan time before each local attempt, no enumerable single-plan
commitment, and no server-wall power instrument. A plain RFC3161 token proves
plan age but not plan exclusivity. A producer-selected transparency-log inclusion
proof proves one leaf, not completeness under an externally assigned experiment
identity.

Passing these targeted tests is not a claim of complete scheduler-dispatch
certification. It means the confirmed evidence-chain defects are closed and the
current production path fails closed before a physical conclusion.

## Historical V0 verdict (superseded by R2)

    E2A_AGGREGATE_MECHANICS_PASS_EXTERNAL_ANCHOR_BLOCKED

The all-pairs aggregate mechanism works and fails closed. **No measurement was
run. No physical label is authorized, and none is reachable on this host.**

Two things make that verdict tighter than "mechanics pass", and both are findings
rather than caveats.

---

## 1. The anchor result

**Provisioning a timestamp authority would not unblock E2A.** That is the useful
sentence, and it is not the one the checkpoint set out to write.

`SUM_ALL_PAIRS_V1` sums every planned pair. That rule means something only if the
cohort was fixed before the results were seen; otherwise a producer runs 20 pairs,
keeps the best 8, writes a plan declaring exactly those 8, and the sum is honest
arithmetic over a chosen set. So the plan commitment needs two properties:

- **P1 PRECEDENCE** - the plan existed before the runs.
- **P2 EXCLUSIVITY** - exactly ONE plan was committed.

**They do not covary.** An RFC3161 timestamp is a genuine third-party attestation:
the key is not ours, the clock is not ours. It passes the spec's own independence
test completely, and a live probe during the design review confirmed freetsa.org
grants tokens whose chain verifies once `cacert.pem` is fetched -- roughly ten
minutes of provisioning away. It buys P1 in full and P2 not at all, because a TSA
is a **responder, not a log**: it does not publish or enumerate the tokens it
issues, and no reviewer can query "every token this requester obtained". Anchor 32
candidate plans, run everything, reveal the one that fits. Every per-record check
passes.

> An RFC3161 token is a lower bound on a plan's AGE.
> It is never an upper bound on a plan's COUNT.

Encoded as `ANCHOR_INDEPENDENT["RFC3161_TSA"] = True` beside
`ANCHOR_ENUMERABLE["RFC3161_TSA"] = False`. That pair of lines is the finding.
`REQUIRED_PROPERTY = ORDERING_AND_ENUMERABLE`; RFC3161 derives `ORDERING_ONLY`;
the gate refuses with `E_ANCHOR_UNENUMERABLE`.

**Anchor audit, measured first-hand** (ANCHOR_AUDIT.md):

| Candidate | Present | Independent | Proves | Enumerable |
|---|---|---|---|---|
| RFC3161 TSA | yes, grants tokens | **yes** | precedence | **no** |
| TPM 2.0 | `/dev/tpm0` exists, **permission denied** | no (custody) | order | no |
| GPG | binary, **no signing key** | no | nothing | no |
| git remote | **experiment's own fork** | no | nothing | no |
| Transparency log | absent | yes | precedence | yes |
| BMC / PDU / meter | absent | - | - | - |

The `[tsa]` section in the host's `openssl.cnf` points at `./demoCA` -- OpenSSL's
stock config configures you to be your *own* timestamp authority, which is the
experiment-owned anti-pattern made literal.

## 2. The adversarial review

Five red-team agents attacked the implementation and an independent verifier
adjudicated. **Three CRITICAL findings, all reproduced, all fixed.** Two were
verbatim recurrences of bugs this very codebase documents as fixed -- which is the
most useful thing the review said.

### Fixed

**C1. The independent checker printed a physical label on hand-written JSON.**
`checker.py --aggregate forged.json` emitted
`SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED` and exit 0 on twenty lines of fabricated
JSON: no bundle, no anchor, no artifacts, no privileges. `check_aggregate` derives
the label from the record's OWN declared `anchor_kind` and `pair_contributions`,
so self-consistency was the entire bar. This is E2's worst bug -- a component that
can be handed a conclusion -- recurring in the one artifact a reviewer actually
runs. It survived because the file is scrupulously honest about its scope in its
docstring, and the CLI ignored the docstring. *Fix:* the CLI reports only
`ARITHMETIC_CONSISTENT_ONLY` and exits non-zero; it can never print a physical
label. Only the resolver, against an anchored plan, produces labels.

**C2. `__pycache__` defeated the pinned canon.** `e2a_canon.py` hashed E1's
`canon.py` against a frozen digest and then called `spec.loader.exec_module()`,
which consults `__pycache__` and executes the **cached bytecode** when the pyc
header matches the source's mtime and size. A poisoned `canon.cpython-313.pyc`
with a forged header left the source digest at `b2a3bfde...` -- pin passing
cleanly -- while `is_int(1.0)` silently became `True`, killing the type gate under
every hash in the suite. Demonstrated before and after:

    source digest unchanged : b2a3bfde28a22033 (pin still passes)
    OLD exec_module()       : is_int(1.0) = True   <- poisoned bytecode ran
    NEW compile(data)       : is_int(1.0) = False  <- source ran, pyc ignored

This is E2's artifact lesson at module scope: hashing one thing and consuming
another is the same defect whether the thing is a power trace or the code that
hashes it. The pin was written specifically because that sibling directory is
untrusted, and it protected the wrong bytes. *Fix:* `exec(compile(data, ...))` --
execute the buffer that was hashed.

**C3. The label check was dead on arrival.** `validate_aggregate`, documented as
"what makes the label unfakeable", called `gate_record`, whose type gate rejects
booleans everywhere -- so it raised `E_TYPE` on its *own* sealed output and could
never have run once. No test called it, so 145 green tests said nothing. A gate
that rejects the thing it protects is not strict; it is absent. *Fix:*
`ALLOWED_BOOL_PATHS` enumerates the two real booleans by exact path, and a test
now round-trips `evaluate() -> validate_aggregate()` and rejects a forged label on
the same evidence.

**H4. The enumerable path was guarded by luck.** `TRANSPARENCY_LOG_INCLUSION`
derives `ORDERING_AND_ENUMERABLE` and passed `check_anchor_kind`. No token was
ever verified -- the only thing between a fabricated inclusion proof and a
physical label was `FROZEN_TSA_ROOT_SHA256` being `None`: a *timestamp-authority*
constant accidentally gating a *transparency-log* anchor. Two unrelated mechanisms
sharing one guard. Provisioning a TSA root for the RFC3161 path -- ten minutes of
work -- would have silently opened the log path. *Fix:* `ANCHOR_VERIFIERS` is an
explicit, **empty** registry; every kind is refused with `E_ANCHOR_NO_VERIFIER`;
`check_trust_root` applies only to CA-rooted kinds.

**M5. The additive guard was a no-op.** E2's `assert_not_additive_input` keys on
`{RealizedTimeline, MatchedComparison, RepetitionSet}`. `AggregateComparison` is
not in that set, so the guard did nothing for the most aggregated record E2A
produces -- the exact bug E2's docstring says it fixed, one kind later. *Fix:* a
local kind check that defers to E2 for E2's kinds. Also removed the call from
`evaluate()`: asserting "my own output is not an additive input" is either a no-op
or an unconditional self-refusal, and it was the former, which is how the missing
kind hid. The guard belongs at E1's front door.

**H6. The lifecycle window check was a name gate.** `ENERGY_BEARING_ACTIONS`
covered 7 of the schema's 12 action kinds; `QUEUE_SUBMIT`, `CANCEL`, `LEASE_*` and
`QUEUE_DRAIN` skipped it, so relabelling a `PREFETCH` as a `QUEUE_SUBMIT` moved
weight staging outside the paid window for free. What costs energy is not knowable
from a label the producer chooses. *Fix:* inverted to `FREE_ACTIONS = frozenset()`
-- everything is charged, and any future exemption is a claim to be argued.

### Refuted

- "Suite is not green" -- 145 tests OK, twice. The attacker was running its own
  stale bytecode. Ironically an instance of C2, but not a defect here.
- "A label can be emitted by editing `FROZEN_TSA_ROOT_SHA256`" -- that requires
  editing the gate's source. An attacker who owns the gate has already won; the
  threat model is a producer supplying evidence to a gate the reviewer runs.
- "Unbounded cross-multiplication" -- the direction is unfavourable; no false pass.

## 3. Open, not fixed

Listed because an undocumented limit is an overclaim. None is reachable today
(the anchor gate refuses everything), and all become load-bearing the day an
enumerable anchor exists.

1. **No window disjointness.** 16 slots could claim one board over overlapping
   windows. `check_match` compares window *durations* only.
2. **Realized work is asserted, not derived.** Output artifacts are hashed against
   their own declared hash but never parsed, so `realized_output_tokens` and
   `output_token_ids_sha256` are producer-typed integers nothing cross-checks.
   Section 7 of CONTRACT.md holds on *ordering* (work before energy), not on
   derivation.
3. **Dead required fields.** `anchor_time_utc_us` is never read, so P1 -- the one
   property RFC3161 genuinely buys -- is enforced nowhere in code. Also unread:
   `warmup_count`, `slo_deadline_us`, `corpus_sha256`, `work_vector`,
   `first_divergence_index`, `entry_state_digest`/`exit_state_digest`,
   `tsa_leaf_cert_sha256`, `tsa_policy_oid`. Each is a gate that looks real and is
   not.
4. **The outcome schema has no `realized_*` counterpart** for tokenizer, decode
   params, stop set or prompt, so those mismatches are undetectable *by
   construction* rather than by oversight.
5. **The bundle index self-hash is circular** -- `read_once(name,
   sha256_bytes(read_bytes()), root)` cannot fail. The index is not anchored.
6. **`ANCHOR_ENUMERABLE["TRANSPARENCY_LOG_INCLUSION"] = True` presumes an identity
   binding** this code does not check. Rekor-style logs accept submissions under
   any key, so anchor-many-reveal-one survives a transparency log unless requester
   identity is externally bound. The map's only `True` -- the only kind that can
   pass -- rests on the audit's weakest sentence. Unreachable today
   (`ANCHOR_VERIFIERS` is empty), so it costs nothing; it must be discharged
   before it ever could.
7. **Accepted limits** (CONTRACT.md section 10): anchor-many-reveal-one is refused
   rather than detected; an alternate ledger chain is undetectable without an
   enumerable anchor; a fabricated-but-consistent lifecycle is indistinguishable
   from a real one; semantic equivalence of outputs is not proven; SERVER_WALL is
   physically unreachable here.

## 4. A design result worth keeping

**Structural checks first, policy gates last.** The anchor gate originally ran
first. It was cheaper and it masked every structural check behind it: at CLI level
a broken ledger, a truncated cohort, a forged energy and a missing slot all
reported `E_ANCHOR_UNENUMERABLE`, so no negative could distinguish a working check
from a deleted one -- and those checks only become load-bearing on the day an
enumerable anchor exists, which is exactly the day nobody would notice they had
rotted. Moving it after resolution made 13 previously-masked CLI negatives
reachable.

It also sharpens what the refusal means. When the gate now fires, it fires on a
bundle that is otherwise impeccable: every slot resolved, every energy
re-integrated, every request matched. The refusal is not "your evidence is
broken". It is **"your evidence is fine and still cannot support this claim"**.

The same mistake recurred at smaller scale within the same session: a new
`check_verifier_available` was added early in `resolve_anchor` and immediately
masked six specific anchor checks. Moved last, for the same reason.

## 5. Reproduction

    bash research_dev/spikes/s10_matched_energy_e2_aggregate/scripts/run_tests.sh

    E1 baseline byte-identical (before): 45 files
    E2 baseline byte-identical (before): 28 files
    Ran 145 tests ... OK
    valid 8-pair bundle: fully resolved, then refused -> E_ANCHOR_UNENUMERABLE
    host anchor capability: best=ORDERING_ONLY required=ORDERING_AND_ENUMERABLE
                            sufficient=False
    S10_E2A_NEGATIVE_PASS (32 cases)
    E2A fixture determinism across 3 regenerations: c32ab63c...
    E1 baseline byte-identical (after): 45 files
    E2 baseline byte-identical (after): 28 files
    S10_E2A_AGGREGATE_TESTS_PASS

Baselines reproduced at CP0 with zero discrepancy and neither was re-pinned:

| Baseline | Expected | Observed |
|---|---|---|
| E1 tests | 201 | 201 OK |
| E1 evidence negatives | 39 | 39 |
| E1 CLI negatives | 18 | 18 |
| E1 differential | 1187, 0 mismatches | 294+298+297+298 = 1187, 0 |
| E1 manifest | 45 files | 45 |
| E2 tests | 152 | 152 OK |
| E2 CLI negatives | 30 | 30 |
| E2 replay | deterministic | `50b68e88...` |

## 6. A finding about E2 (CP5)

E2's `fixtures/repetition_set.json` declares 8 pairs and **validates cleanly under
E2's own rules**, but only pair 0 has timeline records on disk: 14 of its 16
declared record digests are digests of nothing. E2 could not tell, because it
never resolved the digests it listed -- a shape check cannot notice that seven
eighths of a cohort does not exist. E2A resolves every planned slot, so it can.
The gates were not lowered to admit it (`MIN_PAIRS=8`,
`MIN_INDEPENDENT_UPDATES=100`, `MIN_WINDOW_US=1000000` all unchanged).

Related: E2's `attempted_pairs == len(pairs)` check is a **tautology** -- the
producer writes both numbers, so a truncated set is self-consistent at any N. E2A
checks slot coverage against the anchored plan's 2N slots instead, because only an
external count makes truncation visible.
