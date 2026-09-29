# Physical pair R1 artifacts

`PAIR.json` is the canonical matched-pair record. It binds the accepted arm
results, execution plans, synchronized phone-energy summaries, exact workload,
placement, and pass/fail gates by SHA-256.

The `control` and `treatment` directories contain:

- `EXECUTION_PLAN.json`: scheduler decision, capacity contract, artifacts, and
  expected work;
- `RESULT.json`: completed requests, performance, observed phone work, resource
  summary, and server energy;
- `PHONE_ENERGY_V3.json`: synchronized whole-phone paid-interval energy;
- `phone-samples.tsv` and clock receipts: raw phone accounting inputs; and
- `resource-samples.jsonl`: raw CPU package and GPU board accounting samples.

The pair can be reproduced from the six canonical JSON inputs with
`analyze_gpu_overflow_pair.py`. The reproduced output is byte-identical to
`PAIR.json`.
