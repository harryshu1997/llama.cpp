# W8 live-session promotion result

Verdict:
`LIVE_SESSION_TRACE_MECHANICS_PASS; FULL_COMPLETION_SLOWER`.

Scope remains `MECHANICS_ONLY`. The Qwen2.5 Q8 phone route remains
scheduler-ineligible because the separate task-quality gate failed. W8 makes
no energy claim.

## Frozen W8 failure

The first W8 contract required a fresh CUDA-only greedy control to reproduce
every post-start phone and CUDA token. The real run is:

`results/w8_live_promotion/run_20260725T012609Z/`

The treatment itself completed with exact catch-up continuation, zero leaked
state, and valid OpenCL/CUDA placement. The outer gate correctly rejected the
independent greedy control: three sequences had a primary cross-backend token
decision difference, which cascaded to 29 mismatches among 104 generated
tokens. The validator then exposed a missing `ProtocolError` import while
reporting that refusal. The missing import is fixed; the failed manifest is
preserved and bound into R1.

## W8-R1 method

R1 did not relabel the failed greedy comparison. It froze a separate
same-token control before the next acquisition:

- the phone route owns B8 live KV before the paid promotion clock;
- fresh CUDA head and tail processes are absent at request start;
- the phones decode until a completed batch crosses CUDA readiness;
- CUDA replays the dynamic phone frontier while the phones produce a
  two-token-per-request delta;
- CUDA consumes that exact delta, commits ownership, and continues eight
  tokens;
- a second pair of fresh CUDA workers replays the exact treatment token trace;
- CUDA greedy predictions are recorded as a diagnostic, not substituted for
  the treatment tokens.

The passing run is:

`results/w8_live_promotion_r1/run_20260725T013802Z/`

Its manifest SHA-256 is
`f57d6b48ceedf2e36acae9a222dc8bcceb1c7004cbbdec33f8039e0cb8ecd5ad`.

## Real-device result

| Metric | Phone bridge + CUDA | Fresh CUDA trace control |
|---|---:|---:|
| Promotion trigger to next token | 1.357 s | 3.283 s |
| Promotion trigger to CUDA ready | 3.173 s | 3.039 s |
| Promotion trigger to completion | 7.256 s | 3.756 s |
| Post-start tokens per request | 13 | 13 |

The phone bridge reduces promotion-trigger-to-next-token latency by 58.7
percent and produces two useful decode rounds per request, 16 B8 tokens in
aggregate, before CUDA is ready. This is not new-request TTFT: the phone route
holds prompt KV and two committed output tokens per request before the paid
clock. It does not reduce full completion latency: completion is 93.2 percent
slower than the same-token CUDA control.

CUDA0 resolves to an NVIDIA RTX A6000 with 48,530 MiB. This process-cold,
host-page-cache-warm run is not a result on the target RTX 4060 Ti.
Request-to-CUDA-launch delay was 117.250 ms for treatment and effectively zero
for control, so W9 must match this offset before making a repeated performance
claim.

For the teacher-forced control, the 3.283-second endpoint is when replay of the
prompt and two preexisting tokens produces the first post-trigger prediction.
The first known post-trigger token is fed only afterward.

The cause is measured directly:

| Post-ready leg | Time |
|---|---:|
| CUDA replay of the dynamic frontier | 0.173 s |
| Two-token-per-request phone delta | 2.762 s |
| CUDA delta ingestion | 0.040 s |
| Seven CUDA continuation batches | 0.285 s |

The fixed two-token-per-request phone delta is therefore the wrong policy in
this regime. CUDA replay is much shorter than one additional phone decode
batch, so waiting for two post-ready decode rounds extends rather than hides
the cutover.

## Correctness and placement

- all eight autonomous post-cutover CUDA continuations match the autonomous
  same-frontier CUDA control;
- the fresh teacher-forced CUDA trace control evaluates all 104 treatment token
  positions and feeds the first 96 post-start treatment token IDs exactly; the
  final position needs no feed because no later decision is evaluated;
- that control executes 176 CUDA rows: 80 pre-trigger history rows and 96
  post-start teacher-forced rows across 17 physical calls;
- teacher-forced CUDA predictions agree on 101 of 104 token decisions;
- all phone and CUDA session certificates report scheduled placement and zero
  missing buffers;
- OP15 executes `[0,30)`, OP12 executes `[30,48)`, and both CUDA routes use
  the same layer split;
- all phone, treatment CUDA, and control CUDA sequence-state counts return to
  zero;
- the six-record ownership journal reaches `COMPLETE`;
- every file in the result manifest revalidates.

The 101/104 diagnostic does not repair the larger 128-prompt task-quality
failure. R1 proves state continuity and trace replay mechanics, not
cross-backend greedy equivalence or model quality.

## Next bounded gate

W9 must freeze a profile-driven extra-batch rule before acquisition:

```text
k_extra = 0 unless a bounded positive candidate hides
    predicted in-flight residual + phone extra
        behind CUDA replay with a nonzero margin
    and does not increase predicted completion
```

`k_extra` excludes an optional zero-or-one phone batch already in flight at
CUDA readiness. For the measured W8 profile this rule selects `k_extra=0`. At
CUDA readiness, W9 should snapshot the last committed frontier `F0`,
immediately replay `F0` on CUDA, and freeze further phone submission after the
optional in-flight batch. CUDA then ingests the exact realized `F1 - F0`,
including an explicit zero-delta path, commits ownership, and continues to the
same 13-token budget. It must not wait for `F1` before starting replay. The
full prospective contract is in `PLAN.md`.

W9 may claim a promotion-next-token/completion tradeoff only after repeated
prospective runs. W8 does not authorize scheduler integration, the full
two-model trace, or energy measurement.
