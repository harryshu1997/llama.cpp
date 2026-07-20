# S10-E2A Plan

Independently anchored experiment planning, complete same-work/lifecycle
resolution, and SUM_ALL_PAIRS_V1 mechanics.

Status: **R4 route-DAG mechanics pass; physical experiment blocked.**
No physical measurement was run and no physical label is authorized.

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
| CP1 | Freeze strict schemas before implementation | v1-v3 historical; 10 active v4 schemas |
| CP2 | One trusted-root read-once resolver | resolves all 2N slots; E2's fixture refused |
| CP3 | SUM_ALL_PAIRS_V1 | elementwise integer sums, integer 10% gate |
| CP4 | Independent verification | 215 tests + 42 CLI negatives + adversarial review |
| CP5 | R2 semantic evidence chain | 19 targeted regressions; all confirmed repros closed |
| CP6 | R3 wall, identity, anchor, and artifact binding | 18 targeted regressions; later route audit found two blockers |
| CP7 | R4 exact route artifact and phone-result causality | 14 targeted regressions; both reproduced blockers plus post-validation mutation closed |

## What was built

    ANCHOR_AUDIT.md   the load-bearing result: no enumerable anchor exists
    CONTRACT.md       frozen semantics, written before the code
    schemas/          frozen v1-v3 history plus active v4 records
    src/anchors.py    typed anchor capability. The verdict lives here.
    src/resolver.py   dirfd-relative read-once resolution of every planned slot
    src/aggregate.py  SUM_ALL_PAIRS_V1
    src/e2a_canon.py  pinned canonical JSON / SHA-256 / integer type gate
    checker/          independent checker, shares no code with src/
    tests/            215 tests, including 14 focused R4 regressions
    scripts/          42 CLI negatives; runner that pins E1 and E2 before and after
    R3_CONTRACT.md    historical wall, identity, and artifact repair
    R4_CONTRACT.md    active exact-route and phone-result contract

## The four questions E2A had to answer

**1. Can the cohort be fixed in advance?** The v4 contract can represent and
verify an externally enumerated single-plan set, but no eligible external
provider or registered verifier exists. See ANCHOR_AUDIT.md and R3_CONTRACT.md.
RFC3161 remains insufficient because it buys precedence without exclusivity.

**2. Can "identical work" be resolved?** For the current greedy-prefix contract,
yes. Both roles are held to the same request, input, prompt tokens, model,
tokenizer, seed, decode parameters, stop set, stop reason, and SLO. Run-specific
outputs are parsed; token counts, token-ID digests, timings, SLO outcomes, and the
common-prefix certificate are recomputed. The frozen floor is prefix >=32
tokens, length difference <=4, and all requests SLO-met. This still does not
prove semantic equivalence beyond that explicit prefix contract. `SAMPLED`
decoding is refused rather than approximated.

**3. Does the claimed route exist in the evidence?** Internally, yes. The plan
pins two resolved `RouteSchedule` records, not opaque route labels. Each record
freezes the exact action IDs, devices, backends, operator islands, request sets,
byte/duration requirements, leases, and dependency edges. The lifecycle must
match that action DAG exactly. Every assisted request has a DATA path from phone
H2D through HTP/OpenCL execution and D2H into the result, optionally through a
server continuation. A witnessed launcher is still needed to connect those
records to physical reality.

**4. Does the arithmetic hold?** Yes, and it is tested at the boundary: exact 10%
passes, 9.999% fails, seven favourable pairs cannot carry one unfavourable pair,
uncertainty is summed elementwise, floats and bools are refused before any
comparison, and an independent checker agrees on a corpus.

## Verdict

    E2A_R4_ROUTE_DAG_MECHANICS_PASS_PHYSICAL_CLAIM_BLOCKED

See RESULTS.md for the adversarial review, the findings it produced, and what is
still open.

## What would unblock the first controlled physical A/B

In dependency order. Note that (1) and (2) are independent walls: clearing either
alone changes nothing.

1. **An external enumerable commitment** for the plan: third-party
   pre-registration, or a transparency log with a reviewable identity binding
   and complete-set proof. Provisioning an RFC3161 TSA does not count.
2. **A registered cryptographic verifier and trust root.** The v4 API carries
   the necessary enumeration fields, but no eligible verifier or pinned
   independent trust root is installed.
3. **A witnessed launcher.** It must prove that authenticated external
   commitment time precedes each local monotonic attempt and bind the launched
   route and devices to the resulting records.
4. **An instrument.** GPU_BOARD is reachable today via NVML but demands a run
   design around ~1.7 Hz updates and a +/-5 W floor. SERVER_WALL needs hardware
   that does not exist here (E2's INSTRUMENT_AUDIT.md).
5. **A real acquisition run.** R4 closes the internal route, work, wall-
   capability, bundle, and artifact-binding defects, but no physical timelines
   have been acquired under this contract.
