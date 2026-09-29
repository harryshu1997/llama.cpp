# Long-tail trace: desktop-only baseline vs phone-assisted (OP15), 2026-09-23

Trace `burstgpt_longtail_v1` (desktop `/mnt/storage/burstgpt-source/longtail_v1`, see
`TRACE_longtail_v1.md`): first 30-min BurstGPT window whose share of requests and of output
tokens above 512 output tokens each match the whole log within 5 points
(log 18.6 % / 49.3 %, window 16.1 % / 49.4 %). 31 requests, output cap 1100, prompt cap 2048.

| run | status | duration | CPU kJ | GPU kJ | host kJ | host W | phone kJ (assumed) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (desktop-baseline) | PASS 31/31 | 4,901 s | 367.6 | 156.1 | 523.7 | 106.9 | 4.29 |
| treatment run 5 (energy-aware, OP15) | PASS 31/31 | 4,826 s | 275.8 | 155.2 | 431.0 | 89.3 | 6.20 |
| change | | -1.5 % | -25.0 % | -0.6 % | **-17.7 %** | | |

Phone assistance: Gemma 11/12 requests (29,672 phone calls), Qwen 16/16 (27,864), Llama overlay 0/3.
Output identity: 24/31 identical; 7 diverge mid-stream (first differing token 30-383, same
lengths): Qwen 001, 010, 013; Gemma 015, 021, 022, 025. Strict token-exact check FAIL (same
pattern as earlier real-window runs, consistent with batch-shape float near-ties; not investigated).
Phone energy is the assumed 4.5 W active / 0.875 W idle model, not measured.

Treatment runs 1-4 failed; four scheduler fixes were needed (runs 1, 3 and 4 hit scheduler bugs;
run 2 lost the OP15 USB link and also exposed the starvation) -- see research_dev/talks.md 2026-09-23:
1. projection ordering by causal queue order (`runtime_residency_projection.py`),
2. replan no longer starves behind its own dependents (`replan_commit.py`, `runtime_queue.py`, `runtime_controller_ops/replan.py`),
3. memory-rejected arrival baseline re-projects at its start (`automated_requests_ops/selection.py`),
4. dropped deferred verification candidate ends verification (`adaptive_decode_ops/sequencing.py`).
Run 2 also died on an OP15 USB link reset (not a scheduler bug). Open: m4a8b control-vs-completion race.

Inputs: `/home/zhihao/s42-trace-longtail-{baseline,treatment}-20260923-inputs`; baseline `run-baseline-1`,
treatment `run-treatment-5`. Pair analysis: `LONGTAIL_PAIR.json` (analyze_longdecode_pair.py).
