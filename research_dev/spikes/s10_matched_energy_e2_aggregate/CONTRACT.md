# S10-E2A Contract

Frozen semantics for the all-pairs aggregate gate. Written before the
implementation and not edited to match it.

E2A is not a measurement. It is the machinery that would decide whether a
measurement, if one were ever taken, established server-side energy relief. On
this host it cannot reach a physical label, and section 1 says why.

---

## 0. What E2A is, and what it must never become

E1 is an ADDITIVE per-device temporal solver. E2 compares two REALIZED boundary
timelines post hoc. The rule that separates them is absolute and E2A inherits it:

> An aggregate boundary measurement must NEVER be fed back into E1's additive
> solver. A boundary sensor reports shared power once; an additive solver would
> attribute it per device and count it many times.

`assert_not_additive_input` makes this executable for RealizedTimeline,
MatchedComparison and RepetitionSet, and E2A calls it on every AggregateComparison
it emits.

E2A adds exactly one thing E2 deliberately left unbuilt: `SUM_ALL_PAIRS_V1`, the
evaluator that turns a complete predeclared repetition set into a result. E2
refused every single pair with `PAIR_ONLY_NO_AGGREGATE_CLAIM` precisely because
this did not exist.

E2A imports E1's `canon.py` (canonical JSON, SHA-256, integer type gate) and E2's
`comparator`/`integrator` (timeline validation, zero-order-hold integration, the
conservative decision) READ-ONLY and unmodified. It reuses no E1 decision logic:
no binder, no boundary, no validator.

## 1. The anchor, and why no physical label is reachable

`SUM_ALL_PAIRS_V1` sums EVERY planned pair. That rule is worth something only if
"every planned pair" was fixed before anyone saw the results. Otherwise a producer
runs 20 pairs, keeps the best 8, writes a plan declaring exactly those 8, and the
sum is honest arithmetic over a dishonest cohort. Freezing a statistic is not the
same as freezing which records it is taken from.

So a plan commitment needs two properties:

- **P1 PRECEDENCE** - the plan digest existed before the first attempt ran.
- **P2 EXCLUSIVITY** - exactly ONE plan was committed for this experiment identity.

**These do not covary, and conflating them is the trap.** An RFC3161 timestamp is
a real third-party attestation: the key is not ours, the clock is not ours, and it
passes the "not a local timestamp, not a self-hash, not an experiment-owned HMAC"
test completely. It buys P1 in full. It buys nothing of P2, because a TSA is a
responder, not a log: it does not publish, enumerate, or cross-link the tokens it
issues, and no reviewer can query "every token this requester obtained". Anchor 32
candidate plans, run everything, reveal the one that fits. Every check passes.

> An RFC3161 token is a lower bound on a plan's AGE.
> It is never an upper bound on a plan's COUNT.

Only an ENUMERABLE commitment closes P2: a transparency log with inclusion and
consistency proofs, or third-party pre-registration. Neither exists here. See
ANCHOR_AUDIT.md.

Therefore `REQUIRED_PROPERTY = ORDERING_AND_ENUMERABLE`, RFC3161 derives
`ORDERING_ONLY`, and the gate refuses with `E_ANCHOR_UNENUMERABLE`. This is a
finding about the world, not a defect in the mechanism.

Capability is TYPED, never named. `anchor_kind` is a closed enum; `is_independent`,
`proves` and `enumerable` are frozen maps; `anchor_property` is DERIVED. No schema
has an `anchor_property` field, because a value a producer can write is a value a
producer can lie about. This is E2's instrument-typing result applied to integrity:
a free-form `instrument_label` grants nothing, and neither does
`verifier_name: "freetsa.org"`.

## 2. Resolution

One trusted root, derived from the bundle file's own location. **Never a
parameter**: a caller-supplied root lets the producer point the gate at a
directory staged for the occasion, which turns every path check into decoration.

Every artifact is opened ONCE (`O_NOFOLLOW`, `st_nlink == 1`) and the same buffer
is both hashed and parsed. Hashing a path and re-opening it to parse is a
time-of-check/time-of-use gap the red team won 74 times out of 400 with no
privileges.

Absolute paths, `..` traversal, symlinks (at any component), and hardlinked
aliases are refused. Symlinks are refused even when they currently point inside
the root, because the target can be repointed after validation.

Order, per record: schema -> integer type gate -> self-digest. The type gate must
precede every comparison. JSON Schema draft6+ defines `integer` as any number with
a zero fractional part, so `900000.0` validates as an integer; Python then makes
`900000.0 == 900000`, and `bool` subclasses `int` so `True == 1`. Only
`type(x) is int` catches either.

Resolution is all-or-nothing. If any planned slot, record, anchor, request or
action fails to resolve, the caller gets an exception, never a shorter list. There
is no `except: continue` in the all-pairs loop: a refusal that becomes an omission
is a silently smaller and systematically more favourable cohort.

## 3. Gate ordering (structural first, policy last)

1. Structural: plan, ledger, manifest, all 2N slots (timelines re-integrated,
   lifecycles closed, requests matched).
2. Policy: anchor gate, then provenance.
3. Sum, decide, derive label.

The anchor gate ran FIRST in an earlier revision. That was cheaper and wrong: it
masked every structural check behind it, so at CLI level a broken ledger, a
truncated cohort, a forged energy and a missing slot all reported
`E_ANCHOR_UNENUMERABLE` and no negative could distinguish a working check from a
deleted one. Those checks become load-bearing only when an enumerable anchor
exists, which is exactly the day nobody would notice they had rotted.

Running the anchor gate last also sharpens what it says. When it fires, it fires on
a bundle that is otherwise impeccable. The refusal is not "your evidence is
broken"; it is "your evidence is fine and still cannot support this claim".

The anchor gate precedes the provenance gate because the anchor is host-level and
unfixable by any bundle, while provenance is a property of the bundle in hand.
Reporting the unfixable blocker first tells the reader that better evidence would
not help.

## 4. The plan

A `PreRunPlan` pins exact values only. No ranges, no options, no optional
decision-relevant fields: a plan that permits a choice is a plan that can be
resolved after the fact.

- `n_pairs >= MIN_PAIRS = 8`; exactly `2 * n_pairs` slots, contiguous from 0.
- ABBA rotation, checked against the DECLARED pair count. E2's repetition set
  derived the expected order from `len(pairs)`, which makes it a tautology: drop a
  pair and the expected order shrinks to match.
- `max_attempts_per_slot` is `const 1`. Re-running a slot until it looks good is
  what an all-pairs sum exists to prevent.
- `gate_constants_digest` pins MIN_PAIRS, MIN_INDEPENDENT_UPDATES,
  MAX_SAMPLE_GAP_US, MIN_WINDOW_US, WINDOW_TOLERANCE_US, MAX_SAMPLES, the 9/10
  ratio and the normalizer. A plan pinning different thresholds has retuned the
  gate -> `E_GATE_RETUNED`.
- The plan is scope-gated with E2's typed instrument map. The plan is the anchored
  commitment: if it can pair `NVML_BOARD` with `SERVER_WALL`, the whole set is
  designed around a promoted scope and every per-timeline refusal arrives after
  the experiment is spent.

## 5. The ledger

An append-only hash chain over every planned attempt, including failed, canceled,
crashed and ineligible ones.

**Truncation is caught by the PLAN, not by the ledger.** A chain dropped short is
still perfectly self-consistent: every link verifies and the unfavourable tail is
simply gone. E2 checked `attempted_pairs == len(pairs)`, which is a tautology --
the producer writes both numbers. Only an EXTERNAL count makes truncation visible,
so slot coverage is checked against the anchored plan's 2N slots.

Any non-OK status poisons the whole set. `INELIGIBLE` is named explicitly because
it is the drop channel nobody guards: it is neither OK nor FAILED, so a
`status == "FAILED"` check misses it and an unfavourable run relabelled INELIGIBLE
vanishes.

**ACCEPTED LIMIT.** A hash chain does not stop a producer building a whole
ALTERNATE chain from scratch. Nothing in this gate detects that. The plan anchor is
the only thing that would bind the chain to a prior commitment, and the anchor gate
refuses today. This is documented, not closed.

## 6. Lifecycle

Every action bound and closed. This is an ENERGY check, not bookkeeping: energy is
integrated over `[window_start, window_end]`, so work outside that interval is
real, is caused by the run, and is charged to nobody. Weight staging moved into a
PREFETCH before the window, an async D2H landing after it, a CLEANUP deferred past
the end marker -- each spends real joules the integral never sees. That is not a
smaller energy bill; it is a smaller window.

Marker end requires zero queued/running/in-flight actions and an acknowledged
drain. `outstanding(t)` is RECOMPUTED from per-action timestamps. There is no
`outstanding_actions` field and no `drained` flag, because a field read with
`.get()` is dead for `0`, `0.0`, `False`, and omission alike.

## 7. Same work

Per-request identical work is resolved BEFORE energy is examined. The order is not
stylistic: a number computed over an unvalidated cohort has already misled anyone
who stopped reading at the number. Work is a precondition of the comparison, not a
companion check to it.

**HONEST LIMIT, stated because it is load-bearing.** The treatment runs on
different hardware BY CONSTRUCTION, so bit-identical outputs are not expectable and
E2A never demands them. What it demands is that both roles ran the SAME REQUEST --
same input, tokenizer, model digest, seed, decode params, stop set, SLO -- and that
each realized output is bound to a hashed artifact whose length is within the
predeclared tolerance and whose stop reason agrees. That catches shorter
generations, truncation, model swaps, seed drift and substitution. **It does not
prove the outputs are semantically equivalent, and no field here should be read as
proving that.** `SAMPLED` decoding is refused outright rather than approximated: a
distribution match is not a work match.

Weight transforms other than `IDENTITY` are refused (`E_MODEL_TRANSFORM_
UNCERTIFIED`). E2A has no evidence that BF16->F16 or sharding preserves the
computation, so it refuses rather than assuming.

## 8. SUM_ALL_PAIRS_V1

    C  = sum(control_energy_nj)        Uc = sum(control_uncertainty_nj)
    T  = sum(treatment_energy_nj)      Ut = sum(treatment_uncertainty_nj)
    control_lower   = C - Uc
    treatment_upper = T + Ut
    relief          iff treatment_upper < control_lower
    ten percent     iff treatment_upper * 10 <= control_lower * 9

Every planned pair contributes. No trimming, no median, no retry selection, no
favourable-pair filter, no partial aggregation. A missing, failed, incorrect or
undrained attempt invalidates the ENTIRE result.

**Uncertainty is summed ELEMENTWISE.** Not quadrature, not `isqrt(sum of squares)`,
not a standard error, not divided by N or sqrt(N). This is the highest-motive bug
in the checkpoint: at N=8 quadrature shrinks U by ~2.83x and is very often the only
way relief appears at all. It would also be wrong. NVML's +/-5 W is a
vendor-stated SYSTEMATIC bound on each reading, not zero-mean noise. Systematic
error does not average down: if the sensor reads 3 W high, it reads 3 W high in all
eight pairs.

Integers only, no division. `t <= 0.9 * c` is evaluated as `t*10 <= c*9`. Values
and sums are bounded by 2^53-1 -- a CONTRACT bound, not a machine one, since Python
ints do not overflow, which is exactly why it must be checked explicitly.

Per-pair energies are RE-INTEGRATED from raw artifact bytes through E2's frozen
`validate_timeline`. They are never read from a sealed MatchedComparison:
`canon.seal()` is a public function, so a sealed record proves only its own
internal consistency. A signed number is never proof of itself.

`first_executed_role` is DERIVED from the validated windows, never read.

## 9. Labels

Allowed, and exhaustive:

| Label | Meaning |
|---|---|
| `GPU_BOARD_RELIEF_ONLY_TOTAL_ENERGY_BLOCKED` | NVML board comparison only |
| `SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED` | complete server-wall instrument only |
| `RELIEF_FAIL` | resolved, summed, no conservative relief |
| `MEASUREMENT_INVALID` | anything else |

**`SYSTEM_ENERGY_SAVING` is not expressible.** No schema contains it, and a
normalized scan refuses any record mentioning it. Phone, USB supply, charger and
external-device energy are UNKNOWN and permanently out of scope. A positive
server-side result is a BREAK-EVEN BUDGET, not a total-system saving.

A GPU board delta is not a server delta. A board sensor cannot see how CPU, DRAM,
fans or PSU losses moved, so `server_wall_delta_nj` and
`phone_plus_external_break_even_budget_nj` are NULL unless the scope is
`SERVER_WALL`.

No function accepts a label, a reason, a count, a sum, a boolean, a pair selection
or a trusted root. Everything is derived. E2's worst bug was a function that took a
`label` argument and checked only set membership: importing the module and calling
it stamped a sealed `SERVER_RELIEF_PASS` onto junk whose treatment burned 1e15 nJ
MORE. That function was this one's caller.

## 10. Accepted, documented limits

Listed because an undocumented limit is an overclaim.

1. **Anchor-many-reveal-one is not detected, only refused.** With an enumerable
   anchor it would be detectable. Today the gate declines to proceed at all.
2. **An alternate ledger chain is undetectable** without an enumerable anchor.
3. **A fabricated-but-consistent lifecycle** (plausible timestamps that hash
   correctly) is not distinguishable from a real one. The record pins and
   attributes the claim; it does not make it true.
4. **Semantic equivalence of outputs is not proven** (section 7).
5. **SERVER_WALL is unreachable on this host** for physical reasons that predate
   E2A: no wall instrument exists (E2's INSTRUMENT_AUDIT.md). Independent of, and
   stacking with, the anchor blocker.
6. **The independent checker verifies arithmetic, not reality.** It answers "given
   these contributions, is this the right sum, decision and label?" It does not
   re-resolve artifacts and is not a second opinion on whether the contributions
   describe anything.
