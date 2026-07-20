# Independent-phone B32 real-device gate

Verdict: `B32_INDEPENDENT_PHONE_GATE_PASS`

Date: 2026-07-18. Scope: B32 correctness, scheduled placement, latency
repeatability, and thermals on the selected A6000 plus one phone at a time.
Phone, USB, host-wall, and total-system energy are unknown.

## Method

The gate uses the read-only binaries and libraries frozen by the corrected S14
persistence gate. Each phone runs an independent Gemma head route rather than a
serial OP15-to-OP12 chain:

- OP15: `[0,8)` on HTP0, followed by CUDA `[8,48)`;
- OP12: `[0,6)` on HTP0, followed by CUDA `[6,48)`.

For each device, seven fresh processes execute 32 identical streams and eight
decode steps. Every routed stream must exactly match a full-model A6000 B32
reference for the same prompt. The gate also requires scheduled HTP placement,
zero missing-buffer compute nodes, only declared `GET_ROWS` work on CPU, valid
start/end thermal samples, and process CoV no greater than 5 percent.

The report binds the frozen runtime manifest, harness, reused Stage-B helper,
model, CUDA reference log, and every raw host and phone log by SHA-256. The
independent validator reconstructs tokens, timing summaries, placement totals,
thermals, and CoV from those raw bytes.

## Results

| Device route | B32 wall p50 | Conservative p95 | Process CoV | HTP nodes/process | CPU nodes/process | Max end temp |
|---|---:|---:|---:|---:|---:|---:|
| OP15 `[0,8)` | 2.903 s | 2.934 s | 0.489% | 1856 | 8 `GET_ROWS` | 31.8 C |
| OP12 `[0,6)` | 9.363 s | 9.383 s | 0.304% | 1488 | 8 `GET_ROWS` | 34.5 C |

All 448 routed streams (32 streams x 7 processes x 2 devices) match the B32
CUDA reference. All fourteen processes pass placement and thermal gates.

Independent replay:

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 validate_gate.py
```

Expected output:

```text
VALID_B32_GATE op15_p50_us=2902555 op12_p50_us=9363325
```

## Scheduler consequence

The current synthetic low-priority deadline budget is 5 seconds:

- OP15 B32 leaves 2.066 seconds of queueing slack at the conservative measured
  duration, so it is an eligible B32 target.
- OP12 B32 exceeds the deadline before queueing. The scheduler must launch a
  smaller certified batch, select a longer-SLO class, or use server fallback.

The median `mix-v1` trace cannot exercise OP15 B32: it has only 177 requests in
897 seconds and at most three total arrivals in any 2.066-second window. The
frozen real BurstGPT burst window has 11,550 requests in 898 seconds and up to
75 arrivals in the same window. That burst window is the correct next physical
trace for testing B32 formation without inventing arrival compression.

## Runtime integration

`batch32_profile_adapter.py` admits the result as explicit OP15 and OP12 route
variants. Existing B1 routes remain the default, so recorded mechanics cannot
silently masquerade as physical B32 execution. The S15 coordinator test proves
that 32 compatible requests launch as B32 on OP15 under a 5-second SLO, while
the same cohort does not launch as B32 on OP12.

This gate does not yet prove dynamic mixed-trace physical dispatch or energy
saving. The next real run must drive BurstGPT-burst arrivals through the typed
physical executor, keep BGE work isolated on the selected A6000, and report
batch formation, queue wait, fallback, SLO, and selected-GPU board energy.
