# S37 Results

## Verdict

`ARBITRARY_RESIDENT_LAYER_EXIT_PASS`

Both real phones completed the full finite handoff set `{4,5,6,7,8}` against
one resident A6000 tail. Every request retained one cut through prefill and
four output steps. All 20 physical B8 groups returned the common token sequence
`[532,532,532,532]` and drained all route pins, sequence leases, and worker KV.

## Results

Each row is the median of the two B8 group p95 request latencies. Latency is
reported only to identify the physical runs; S37 has no latency gate.

| Phone | Cut | Phone range | CUDA range | Median p95 |
| --- | ---: | --- | --- | ---: |
| OP12 | 4 | `[0,4)` | `[4,48)` | 1016.908 ms |
| OP12 | 5 | `[0,5)` | `[5,48)` | 1171.927 ms |
| OP12 | 6 | `[0,6)` | `[6,48)` | 1284.159 ms |
| OP12 | 7 | `[0,7)` | `[7,48)` | 1445.418 ms |
| OP12 | 8 | `[0,8)` | `[8,48)` | 1623.957 ms |
| OP15 | 4 | `[0,4)` | `[4,48)` | 801.083 ms |
| OP15 | 5 | `[0,5)` | `[5,48)` | 943.036 ms |
| OP15 | 6 | `[0,6)` | `[6,48)` | 1036.556 ms |
| OP15 | 7 | `[0,7)` | `[7,48)` | 1022.677 ms |
| OP15 | 8 | `[0,8)` | `[8,48)` | 1309.980 ms |

The small OP15 cut-6/cut-7 inversion is ordinary two-repetition variance and
is not interpreted as a performance result.

## What Passed

- ten request-level routes, covering both phones and every jointly resident
  boundary, executed twice;
- the phone graph used `[0,cut)` and the terminal graph used `[cut,48)` for
  every physical call;
- request identity, route epoch, position, and cut remained pinned through
  prompt and decode;
- terminal tokens matched across all cuts and devices;
- final active-sequence counts, route pins, and software leases were zero;
- workers detached and then stopped without reloading weights between cuts.

## Boundary

"Arbitrary" means any layer in the resident overlap. It does not mean any of
the 48 layers regardless of placement. With current images, cuts below 4 are
missing CUDA prefix weights and cuts above 8 are missing phone weights.

S37 proves placement mechanics only. S36 separately proves a real mixed
prefill/decode HTP batch. Neither result establishes GPU, phone, network, or
total-system energy savings.
