# Gemma decode split ratio: phone share of FFN columns versus the desktop (2026-09-16)

Bounded native calibration on the real OP15 over FunctionFS, one HTP session (the HTP0
shard, layers 0-7 of the 24 CPU layers, full 15,360-column width), the calibrated CUDA
desktop parent (23 GPU layers, context 8,192, batch 2,048, ubatch 512) and the 5,261-token
document with 96 greedy output tokens. Prefill always runs locally; the runtime control at
the first generated token gives the phone its column share of the split layers for decode
only (`decode-boundary-v1`). Two repetitions per arm, arms in the order 0, 100, 75, 50, 25, 0.
Not a scheduler qualification: no leases, no transport identity, direct worker path without
the resident router, one session of three.

Driver: `sweep_decode_split.py`; summary: `analyze_sweep.py`; artifacts: `physical-v1/`
(`SUMMARY.json`, `RESULT.json`, per-request `EXECUTION-*.json` with the full server response,
`SERVER_MEMORY.jsonl`, `PHONE_HEALTH.jsonl`, server logs, `sweep-v1.log`).
Remote: `/mnt/storage/s42-decode-split-ratio-20260916-v1-ae3a24/sweep-v1/`.
Server build: `/mnt/storage/s42-ffn-microbatch-20260916-v4-arwuw3/cuda-build` (the 2026-09-16
microbatch-attribution binary); phone worker `s42-ffn-shards-20260904-v1-bin`, max tokens 4,
column quantum 1,280, queue depth 4.

## Result

| Phone share of FFN columns (8 of 24 CPU layers) | Decode ms/token (2 runs) | vs desktop only | Prefill ms | Host energy per request J | Tokens equal to desktop output |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 % (desktop only, 4 runs) | 459.0 / 459.7 / 459.5 / 458.8 | - | 22,733 | 7,239 / 7,341 / 7,529 / 7,577 | - |
| 25 % | 438.7 / 439.2 | -4.4 % | 22,392 | 7,301 / 7,280 | 96 / 96 |
| 50 % | 415.9 / 416.6 | -9.4 % | 22,375 | 6,959 / 6,902 | 96 / 96 |
| 75 % | 411.4 / 412.3 | -10.3 % | 22,379 | 6,597 / 6,748 | 40 / 40 |
| 100 % | 432.8 / 434.5 | -5.6 % | 22,423 | 6,154 / 6,139 | 96 / 96 |

Decode time repeats within 0.2 % across repetitions. 75 % is the fastest tested share and
50 % is within 1 % of it; the phone-only share (100 %) gives back half of the gain because the
desktop then idles while the phone computes. Host CPU package plus GPU board energy per
request falls monotonically with the phone share (phone power is not measured here; at an
assumed 4.5 W the phone adds about 180 J per request). The 75 % outputs agree with the
desktop output for the first 40 tokens and then diverge (f16 phone arithmetic), as the
earlier relocation gates recorded; all arms produced 96 tokens.

Only one of the three sessions was split, so the absolute gain is one third of what the full
24-layer configuration can give; the per-layer balance point is what this sweep pins down.
The production three-session path adds about 1 ms of router forwarding per call, which moves
the balance slightly toward the desktop; 75 % remains the chosen ratio.

## Decision

Phone share 75 % (host columns 3,840 of 15,360) for decode-only relocation. Under a dormant
host share this releases the page-exact suffix of gate/up rows and the per-row suffix of
down: 1,981,743,104 bytes for the eight HTP0 layers (planning lower bound 1,918,828,544),
5.76 GB lower bound for all 24 layers, against 7.93 GB when the phone owns every column.
