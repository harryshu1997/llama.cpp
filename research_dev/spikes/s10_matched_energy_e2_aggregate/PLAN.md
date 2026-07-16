# S10-E2A Plan

Independently anchored experiment planning, complete same-work/lifecycle
resolution, and SUM_ALL_PAIRS_V1 mechanics.

Status: **R2 targeted repair complete; physical experiment blocked.** No physical
measurement was run and no physical label is authorized.

---

## Why E2A exists

E2 refused every control/treatment pair with `PAIR_ONLY_NO_AGGREGATE_CLAIM`: one
pair is diagnostic, and a physical label needs the complete predeclared repetition
set aggregated by `SUM_ALL_PAIRS_V1`. That evaluator was deliberately not built,
which made physical labels unreachable by construction and left the honest
question open:

> If the aggregate existed, would a positive result mean anything?

E2A builds it, and answers: **not yet, and not for the reason anyone expected.**
The blocker is not arithmetic and not instrumentation. It is that an all-pairs sum
is only worth something if the cohort was committed before the results were seen,
and nothing on this host can commit it in a way a reviewer could check.

## Checkpoints

| CP | What | Result |
|---|---|---|
| CP0 | Reproduce and freeze E1 + E2 | both reproduce exactly; neither re-pinned |
| CP1 | Freeze strict schemas before implementation | 8 v1 + 8 v2 schemas, all local refs, closed enums |
| CP2 | One trusted-root read-once resolver | resolves all 2N slots; E2's fixture refused |
| CP3 | SUM_ALL_PAIRS_V1 | elementwise integer sums, integer 10% gate |
| CP4 | Independent verification | 183 tests + 32 CLI negatives + adversarial review |
| CP5 | R2 semantic evidence chain | 19 targeted regressions; all confirmed repros closed |

## What was built

    ANCHOR_AUDIT.md   the load-bearing result: no enumerable anchor exists
    CONTRACT.md       frozen semantics, written before the code
    schemas/          frozen v1 plus repaired v2 records, closed and local-only
    src/anchors.py    typed anchor capability. The verdict lives here.
    src/resolver.py   trusted-root read-once resolution of every planned slot
    src/aggregate.py  SUM_ALL_PAIRS_V1
    src/e2a_canon.py  pinned canonical JSON / SHA-256 / integer type gate
    checker/          independent checker, shares no code with src/
    tests/            183 tests, deterministic fixtures
    scripts/          32 CLI negatives; runner that pins E1 and E2 before and after

## The three questions E2A had to answer

**1. Can the cohort be fixed in advance?** No. See ANCHOR_AUDIT.md. RFC3161 is
independent and available, and buys precedence only; exclusivity needs an
enumerable commitment that does not exist here. `E_ANCHOR_UNENUMERABLE`.

**2. Can "identical work" be resolved?** For the current greedy-prefix contract,
yes. Both roles are held to the same request, arrival, model, tokenizer, seed,
decode mode, stop reason and SLO. Run-specific outputs are parsed; token counts,
token-ID digests, timings, SLO outcomes and the common-prefix certificate are
recomputed. The frozen floor is prefix >=32 tokens, length difference <=4, and
all requests SLO-met. This still does not prove semantic equivalence beyond that
explicit prefix contract. `SAMPLED` decoding is refused rather than approximated.

**3. Does the arithmetic hold?** Yes, and it is tested at the boundary: exact 10%
passes, 9.999% fails, seven favourable pairs cannot carry one unfavourable pair,
uncertainty is summed elementwise, floats and bools are refused before any
comparison, and an independent checker agrees on a corpus.

## Verdict

    E2A_R2_TARGETED_MECHANICS_PASS_PHYSICAL_CLAIM_BLOCKED

See RESULTS.md for the adversarial review, the findings it produced, and what is
still open.

## What would unblock the first controlled physical A/B

In dependency order. Note that (1) and (2) are independent walls: clearing either
alone changes nothing.

1. **An enumerable commitment** for the plan: third-party pre-registration, or a
   transparency log with an identity binding a reviewer can enumerate.
   Provisioning an RFC3161 TSA does NOT count and would not help.
2. **A cryptographic verifier.** E2A types anchor capability and implements no
   verifier at all: no token is parsed, no signature checked, no inclusion proof
   validated. `ANCHOR_VERIFIERS` is empty and every kind is refused with
   `E_ANCHOR_NO_VERIFIER`.
3. **An instrument.** GPU_BOARD is reachable today via NVML but demands a run
   design around ~1.7 Hz updates and a +/-5 W floor. SERVER_WALL needs hardware
   that does not exist here (E2's INSTRUMENT_AUDIT.md).
4. **A real acquisition run.** R2 closes window overlap and derived-work defects,
   but no physical timelines have been acquired under this repaired contract.
