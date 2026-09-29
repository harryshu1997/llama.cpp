# Physical iteration I0_IMPORTED_REAL_PAIR

Declared change: enable dynamic Gemma FFN column offload to one OP15
Physical pairs: 1

| metric | control | treatment | change |
| --- | ---: | ---: | ---: |
| BurstGPT makespan | 327.704644 s | 197.321246 s | -39.79% |
| accounted fleet energy | MISSING | MISSING | MISSING |
| completed work | 74 req / 2175 tok | 74 req / 2175 tok | equal |
| SLO requests met | 57.0 | 59.0 | +2.0 |
| phone-routed requests | 0 / 74 | 17 / 74 | +17 |
| phone share of eligible cold FFN MACs | 0% | 62.62% | +62.62 points |

## Phone work

- Paid calls excluding warmup: 24240
- Phone MACs: 34161284874240
- Eligible cold FFN MACs: 54553529548800
- Host-to-phone paid bytes: 2367774720
- Phone-to-host paid bytes: 2367774720
- p50 branch imbalance: 3.35%
- p50 exposed join wait: 4.67% of island
- p50 phone path hidden: 95.20%
- Arithmetic-mean exposed join wait: MISSING
- Exact cold outputs: 8 / 17
- Equal cold token positions: 394 / 505

VERDICT: INCOMPLETE
BEST_REAL_RESULT: 197.321246 s, accounted_energy_j=missing, energy_change_pct=missing, phone_work_pct=62.62
BLOCKER: synchronized CPU, GPU, and phone energy receipts for both arms
NEXT_ONE_CHANGE: add synchronized CPU, GPU, and phone power plus mean overlap counters without changing execution
