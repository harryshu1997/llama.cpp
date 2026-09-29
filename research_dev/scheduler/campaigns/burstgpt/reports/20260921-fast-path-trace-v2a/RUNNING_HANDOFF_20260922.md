# Active Task 1 checkpoint - 2026-09-22 13:11 UTC

Current run: /home/zhihao/s42-trace-v2a-m4a6-r2-20260922-inputs/run-treatment-1/run.
Preflight and transport admission PASS; 24-request acceptance RUNNING under flock.
Wait on RESULT.json / FAILURE.json and parent RUN_EXIT.txt. Do not run hardware
concurrently. m4a6 initial startup failed before inference because the builder
omitted the matching rig boot hash; fixed and freshly regenerated in r2.
No inference result yet. Full details in talks.md and README. Older checkpoints
below are historical and superseded where marked.

# Active Task 1 checkpoint - 2026-09-22 12:58 UTC

User renewed Task 1 authorization, including "continue". Work is active.
Fresh inputs: /home/zhihao/s42-trace-v2a-m4a6-20260922-inputs.
Transport 9/9 and catalog capacity admission PASS at 40,960 bytes/depth 4;
154 tests PASS. No production controller/native change in this resumed work.
The owner-handoff diagnostic below is corrected: it omitted the live batch
membership change; a realistic regression passes. Campaign preflight is
pending. Do not start a second rig run concurrently. Updated details in README.

Historical checkpoint follows; its no-third-attempt instruction was superseded
by the user's renewed request. Retain all old results and diagnostics.

# Completed-work checkpoint - 2026-09-22 06:26 UTC

No runs remain active. Never commit, push, write PR text, force-kill a phone
worker, or use adb outside 5037. The original shared-hardware rules remain.

Task 1 is STOPPED/FAIL after two physical runs. Do not launch a third Task 1
attempt without renewed direction.

| Arm | Host kJ | Saving vs 132.253 kJ | Qwen assisted | Exact outputs |
| --- | ---: | ---: | ---: | ---: |
| m4a4 | 109.561 | 17.158% | 5/15 | 22/24 |
| m4a5 | 94.909 | 28.237% | 0/15 | 22/24 |

The same Gemma 88132/token 2 and 88139/token 36 mismatches repeat (zero-based).
M4a4 made 352 two-row native FFN forwards, but split-row transport made all USB
calls one row. Its batch>=2 phone windows had median 57.141 J/token.
M4a5's Qwen coalesced candidates were rejected as TRANSPORT_PROFILE_INCOMPLETE;
no Qwen helper was admitted. Its saving occurred with Gemma-only assistance.
First-token assistance and coalesced Qwen execution remain unverified.

The experimental server_policy_coherence implementation is off by default.
245 targeted tests passed, but an extra local diagnostic failed: an unprobed
short owner completes and its long follower remains EXPLOITING despite
_can_probe=True. Preserve this known limitation; it did not cause m4a5's
transport admission rejection. Source: physical/m4a5-source.

Task 2 matched pair PASS, one successful pair:

- Host 203.348 -> 164.290 kJ: 19.208% saving.
- Duration 2057.388 -> 2094.451 s: 1.801% longer.
- Phone 1.800 -> 2.773 kJ, assumed at 0.875 W idle / 4.5 W active.
- 19/19 requests complete, 3267 output tokens, all token-identical.
- Qwen 11/13 and Gemma 4/4 assisted; Llama 0/2.
- The exact supplied builder selected 19 requests, 10578 input tokens,
  p50/p90/max output 152/446/472, zero clipping, no outputs above 512.
  The old cap's removal had no effect in this window; long-tail benefit is
  unverified. No filtered best-case trace was substituted.

Completed remote results:

- /home/zhihao/s42-trace-longdecode-baseline-20260922-inputs/run-baseline-1/run/RESULT.json
- /home/zhihao/s42-trace-longdecode-treatment-r2-20260922-inputs/run-treatment-1/run/RESULT.json

The first treatment attempt failed at 05:36:18 UTC. Earlier Gemma request 002
had 2496 complete FFN records but only 2495 parsed: an interleaved timestamp
without a severity letter prefixed call 1610/layer 1. The parser now accepts
that exact timestamp form; all structured fields remain strictly matched.
79 adapter/KV/reference tests and pyflakes passed. Saved-log parsing recovers
2496/2496 calls, 104 per layer. No native rebuild was needed. Baseline made no
phone calls and is unaffected. Source: physical/longdecode-parser-fix.

Attempt 1 has FAILURE.json, no RESULT.json, and no matched energy saving. It
completed 17 streams, of which 13 were exact. Differences: Gemma 000/token 52;
Qwen 008/token 115, 013/token 7, 014/token 36. Combined stream index 017
(source request 015) stopped after 81/339 tokens; index 018 never started.
The parser fix is not established as a numerical fix. The successful retry
does not explain these differences; repeatability remains unverified.

Both Task 2 arms use m4a3 split-row inputs, server_policy_coherence=false. The
retry only changes parser recognition and campaign ID/paths from attempt 1.
Rig, model, evidence and transport identity manifests otherwise match. Native
library unchanged; 16 FFN markers were verified before the runs. The final
retry exited zero at 06:21:57 UTC, with no FAILURE or CLEANUP_FAILURE file;
a read-only check found the shared rig lock free.

Local evidence: physical/LONGDECODE_PAIR_COMPARISON.json,
physical/LONGDECODE_ENERGY_COMPARISON.json, physical/longdecode-baseline,
physical/longdecode-treatment (failed attempt 1), physical/longdecode-treatment-r2,
and physical/longdecode-treatment-r2-inputs. Native logs and SSE streams are
saved. Use analyze_longdecode_pair.py for exact output comparison;
stream hashes include timings and do not establish token equality.

README.md, research_dev/talks.md, the plan status and NEXT_AGENT_PROMPT.md are
current. Task 3 / OP11 was not started under Task 1's stop condition; its
checklist was not read and the second phone was not touched in this session.
No commit, push, native rebuild, kernel change or manual worker kill occurred.
