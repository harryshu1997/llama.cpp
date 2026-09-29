# MoE edge-energy screen: can a phone-resident expert tier save host energy?

Date: 2026-09-19. Rig: desktop `172.20.74.85` (i9-12900K, 32 GB DDR5, RTX 4060 Ti 16 GB, NVMe root),
no phone attached to this screen. Model: `Qwen/Qwen3-30B-A3B-GGUF` `Qwen3-30B-A3B-Q4_K_M.gguf`
(SHA-256 `0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5`, matches the Hugging Face
LFS object), 48 layers, 128 experts, 8 used per token, routed-expert bank 16.35 GiB, 1,046 MiB of
expert weights per decoded token.

## Verdict

A phone-resident tier of exact experts can only ever pay back the **paging penalty** of a host whose
RAM cannot hold the expert bank. It never beats a host that holds the experts in RAM, and on this rig
the penalty it would recover is small until the host is starved:

| Host expert cache (cgroup cap) | Host decode | Host energy | Realistic net saving from a 3.75 GiB phone tier |
| --- | ---: | ---: | --- |
| All in RAM (reference) | 44 ms/tok | 6.28 J/tok | none: nothing to recover |
| ~8.4 GiB effective (10 GiB cap) | 145 ms/tok | 5.95 J/tok | **negative** (-2.6 to -4.9 J/tok): the host already waits for disk at 41 W, cheaper than adding phone latency |
| ~5.1 GiB effective (6 GiB cap) | 325 ms/tok | 12.24 J/tok | +0.7 to +2.8 J/tok (6 to 23 % of the paging arm), still 1.5x the reference |
| ~3.0 GiB effective (4 GiB cap) | 458 ms/tok | 17.37 J/tok | +3.8 to +5.9 J/tok (22 to 34 %), still 1.8x the reference |

The routing is skewed enough that a small tier covers a lot (a 3.75 GiB tier of the most-selected
experts serves 68 % of selections against 23 % for a uniform router), so **coverage is not the
problem**. The problem is that the paging penalty is idle-floor power times disk-wait time, and the
phone tier replaces disk-wait time with USB round-trip time charged at the same floor. The saving is
the difference of two waits, minus the phone's own energy.

Not run here: any phone execution. The phone enters the bound as two assumptions, 4.5 W active power
and 50 or 100 ms of added latency per token for 48 layers of expert calls. Those need the expert-select
worker mode on the phone before they can be measured; this screen decides whether building it is
justified. **Recommendation:** not for host-RAM-starved MoE serving on this rig. The VRAM warm tier
(below) is the tier to use when host RAM is short, and it is free.

## What was measured

Four host arms with an identical request (1,024 WikiText-2 prompt tokens, 128 greedy output tokens,
two requests per arm, the second one reported), `llama-server` with all non-expert weights on the GPU
and the experts on the host (`-ot exps=CPU`, mmap), 16 threads, batch 2048, ubatch 512, context 4096.
CPU-package energy from RAPL and GPU-board power from NVML at 10 Hz; the server's `/proc` read bytes
and major faults at the decode-window edges. Paging arms run in a `systemd-run --user --scope` with
`MemoryMax` and the model file evicted from the page cache first.

| Arm | Load | Prefill 1,024 tok | Decode | CPU pkg | GPU board | Host total | Host power in decode | Decode reads | Major faults / tok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference, 16 threads | 1.5 s | 3.59 s, 274 J | 45.9 ms/tok | 4.56 J/tok | 1.93 J/tok | 6.49 J/tok | 141 W | 0 | 0 |
| reference, 8 threads | 1.2 s | 3.60 s, 274 J | 44.2 ms/tok | 4.37 J/tok | 1.91 J/tok | 6.28 J/tok | 142 W | 0 | 0 |
| paging, 10 GiB cap | 19 s | 39.3 s, 1,351 J | 145.1 ms/tok | 1.47 J/tok | 4.48 J/tok | 5.95 J/tok | 41 W | 47.9 MiB/tok | 579 |
| paging, 6 GiB cap | 19 s | 46.7 s, 1,619 J | 325.2 ms/tok | 2.61 J/tok | 9.63 J/tok | 12.24 J/tok | 38 W | 221 MiB/tok | 2,359 |
| paging, 4 GiB cap | 19 s | 44.5 s, 1,544 J | 458.0 ms/tok | 3.94 J/tok | 13.43 J/tok | 17.37 J/tok | 38 W | 419 MiB/tok | 3,967 |

Readings:

- The reference decode is memory-bound on the host (1,046 MiB of expert weights per token in 44 ms is
  about 24 GB/s) and 8 versus 16 threads makes no difference. Decode power is 142 W: about 98 W CPU
  package plus 43 W GPU board.
- Paging makes decode 3.3x to 10x slower, but the host draws only 38 to 41 W while it waits on the
  NVMe (CPU package about 8 W, GPU idling at about 30 W with the model loaded). At the 10 GiB cap the
  paging arm therefore uses **less** energy per token than the reference. The penalty grows with the
  miss fraction: 4.6 % of expert bytes missed at 10 GiB, 21 % at 6 GiB, 40 % at 4 GiB.
- The misses are latency-bound, not bandwidth-bound: about 0.1 ms per major fault, taken synchronously
  by the compute threads (579 to 3,967 faults per token).
- Prefill under any cap re-reads most of the bank per ubatch through the used-expert copy path
  (20 to 52 GiB read per request) and takes 39 to 47 s instead of 3.6 s. A phone tier does not help
  prefill; the GPU already does the prefill math.

## Routing skew and tier coverage

`llama-moe-routing-histogram` (new tool, `examples/layersplit/moe-routing-histogram.cpp`) observed
the `ffn_moe_topk` tensor of every layer over 65,536 WikiText-2 test tokens in 2,048-token chunks
(3,145,728 layer-token rows; logits requested for every token so the last layer is observed for all
rows). `analyze_routing.py` turns the counts into a coverage curve with exact per-expert bytes from the
GGUF (2.72 MiB per expert per layer at Q4_K_M) and a greedy split of the tier's bytes across layers by
marginal hits per byte.

| Tier of most-selected experts | Coverage of selections | Uniform router would give | Experts resident (per layer) |
| ---: | ---: | ---: | ---: |
| 1 GiB | 31.8 % | 6.1 % | 381 (2 to 13) |
| 2 GiB | 47.9 % | 12.2 % | 757 (6 to 21) |
| 3.75 GiB (OP15 budget used by the resident-routing project) | 68.0 % | 22.9 % | 1,418 (17 to 38) |
| 6 GiB | 85.2 % | 36.7 % | 2,256 (37 to 59) |
| 8 GiB | 94.1 % | 48.9 % | 3,003 (50 to 91) |
| 12 GiB | 99.7 % | 73.4 % | 4,505 (86 to 117) |

Mean effective number of experts per layer (entropy) is 60.5 of 128; top-8 per layer covers 31.6 %,
top-32 covers 70.8 %. The curve also predicts the host's own page cache: the resident size that
reproduces each arm's measured miss fraction is 8.45, 5.05 and 2.98 GiB for the 10, 6 and 4 GiB caps,
which is the cap minus the server's non-expert footprint. The page cache behaves like a
frequency-ordered tier, which is why the phone tier's residents largely duplicate what the host
already keeps.

## The bound (`phone_upper_bound.py`)

Two estimates per paging arm, tier size and assumed phone latency:

- **Favorable:** every selection the tier serves is one the host would have missed; the penalty
  shrinks by min(coverage / miss, 1). This is an upper bound and it is what the "favorable" column
  reports.
- **Realistic:** the host cache is the frequency tier of the size that reproduces the measured miss;
  the phone tier adds its bytes to that tier; the new miss fraction comes off the curve. The host's
  measured wait-floor power (38 to 41 W) is charged for the phone's added latency, plus 4.5 W on the
  phone.

| Paging arm | Tier | Miss without / with tier | Phone +50 ms | Phone +100 ms | Favorable bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 GiB cap (5.95 J/tok) | 3.75 GiB | 4.6 % / 0.2 % | -2.59 J/tok | -4.87 J/tok | -2.61 J/tok |
| 6 GiB cap (12.24 J/tok) | 2 GiB | 21.1 % / 9.4 % | +1.19 J/tok (+9.7 %) | -0.92 J/tok | +3.85 J/tok |
| 6 GiB cap | 3.75 GiB | 21.1 % / 3.7 % | +2.80 J/tok (+22.9 %) | +0.69 J/tok (+5.6 %) | +3.85 J/tok |
| 6 GiB cap | 6 GiB | 21.1 % / 0.7 % | +3.64 J/tok (+29.7 %) | +1.53 J/tok (+12.5 %) | +3.85 J/tok |
| 4 GiB cap (17.37 J/tok) | 3.75 GiB | 40.0 % / 10.9 % | +5.94 J/tok (+34.2 %) | +3.82 J/tok (+22.0 %) | +8.97 J/tok |
| 4 GiB cap | 6 GiB | 40.0 % / 3.3 % | +8.04 J/tok (+46.3 %) | +5.92 J/tok (+34.1 %) | +8.97 J/tok |

Even the best realistic case (4 GiB cap, 6 GiB tier, 50 ms) lands at 9.3 J/tok, 1.5x the 6.28 J/tok
of a host that simply holds the experts in RAM. Full JSON: `physical/runs/phone_upper_bound.json`.

## VRAM warm tier (the free alternative, measured)

Same request, same 6 GiB host cap, experts of layers 0-23 placed in VRAM by
`-ot 'blk\.([0-9]|1[0-9]|2[0-3])\.ffn_.*_exps=CUDA0,exps=CPU'` (a layer-granular tier, 50 % of the
bank, not even frequency-ordered), plus the same placement without a cap. All arms produced the same
128 output tokens as the reference.

| Arm | Load | Prefill 1,024 tok | Decode | CPU pkg | GPU board | Host total | Host power in decode | Decode reads | Major faults / tok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference, all experts on host | 1.2 s | 3.60 s, 274 J | 44.2 ms/tok | 4.37 J/tok | 1.91 J/tok | 6.28 J/tok | 142 W | 0 | 0 |
| experts of 24 layers in VRAM, no cap | 15.6 s | 1.91 s, 155 J | 28.6 ms/tok | 1.94 J/tok | 1.62 J/tok | 3.56 J/tok | 125 W | 0 | 0 |
| experts of 24 layers in VRAM, 6 GiB host cap | 27.6 s | 13.3 s, 533 J | 56.4 ms/tok | 0.85 J/tok | 2.06 J/tok | 2.91 J/tok | 52 W | 5.6 MiB/tok | 67 |
| all experts on host, 6 GiB host cap (from above) | 19 s | 46.7 s, 1,619 J | 325 ms/tok | 2.61 J/tok | 9.63 J/tok | 12.24 J/tok | 38 W | 221 MiB/tok | 2,359 |

Under the same starved host, moving half the bank into free VRAM cuts decode energy from 12.24 to
2.91 J/tok (4.2x) and decode latency from 325 to 56 ms/tok, and it beats the unconstrained all-host
reference on energy (2.91 vs 6.28 J/tok) because the GPU serves its half at a fraction of the CPU's
memory-bound power. The best realistic phone-tier estimate for the same cap was 9.4 J/tok. Any rig
with free VRAM should spend it on experts before considering a phone; a phone tier is only conceivable
when VRAM, then host RAM, are both full of things that must stay resident, and even then it recovers
only part of the disk-wait floor.

## Bugs found on the way (fixed, recorded because they invalidated a first round)

- The first round's four arms all answered from one leftover `llama-server` of an earlier crashed
  gate run: its port stayed bound, later servers failed to bind, and the health check found the old
  one. Fixed by picking a free port per arm, refusing a port in use, verifying the answering process's
  command line, and cleaning the server up in a `finally`. The first round is preserved in
  `physical/runs/{reference-t16,reference-t8,paging-10g,paging-6g}` for the reference arms only; the
  `paging-*` directories of that round are invalid and superseded by `*-r2`.
- The histogram missed the last layer: only output rows reach it, so the tool now requests logits for
  every token.
- The gate read the server's cgroup before systemd moved it into the transient scope, so the
  `cgroup_final` fields of the `-r2` arms describe the login session, not the scope. The cap was in
  effect (reads and faults show it); the fix is in the script for the next run.
- The desktop's WikiText copy under `s33_sources` is an empty file.

## Files

- `moe_energy_gate.py`: one arm, RAPL + NVML + `/proc` sampling, request through `/completion`.
- `analyze_routing.py`: coverage curve and skew from the histogram and the GGUF.
- `phone_upper_bound.py`: favorable and realistic bounds.
- `physical/run_arms.sh`, `physical/run_paging_r2.sh`, `physical/run_gputier.sh`, `physical/segfetch.sh`: desktop scripts.
- `physical/runs/`: RESULT.json, SAMPLES.json and server logs per arm; `routing_histogram.json`,
  `coverage.json`, `phone_upper_bound.json`.
- Desktop deploy: `/mnt/storage/s43-moe-energy-20260919-v1` (source snapshot, CUDA build; `llama-server`
  SHA-256 prefix `e5f0e64a`), model and runs under `/home/zhihao/moe-energy-20260919`.

Single runs, two requests per arm, no statistics. No commit.
