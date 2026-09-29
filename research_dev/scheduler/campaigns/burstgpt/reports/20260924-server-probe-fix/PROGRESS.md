# server-probe-fix PROGRESS

- 15:50 UTC copy made: base/ and root/ = main research_dev/scheduler (rsync -a, no __pycache__, reports/ and
  baselines/ excluded except reports/20260917-decode-relocation-kv-headroom which a test imports);
  shared/gguf-py copied from evidence-fixes, shared/spikes -> evidence-fixes/shared/spikes (read-only).
- ~15:52 UTC read AGENTS.md, coherent-policy README 4.4, phone-rejection README + EVIDENCE_FIXES_README; copied
  coherentEF/plainEF run JSONs (read-only scp) to runs/. Confirmed from events:
  * 002 @153.5 s: W1 host 73.02 J/605 ms, W4 P100 61.78 J/618 ms -> verdict[1]=host SERVER_PAIR_NOT_IMPROVED;
    means favor phone, bounds overlap (67.96 > 65.06).
  * (b) 56 batch-1 host decisions (004 x23 at 295-349 s, 006 x33) have server_policy.reason None because the
    batch-2 phone verdict at 235.7 s cleared the single group.reason.
  * (a) plainEF 003 x4 SERVER_POLICY_COHERENCE (tok 46/50 P75, 62/66 P25) from `_follow_coherent_policy`
    coherence-off branch; that follow rule is the intended default (docstring + AdaptiveCoherenceTests) ->
    relabel, not remove.
  * (c) C++: examples/layersplit/ffn-split-client.cpp summary counts `tokens == 1` as decode, else prefill
    (server.cpp only sums). Python does not parse decode/prefill_calls.
- Design (server probe): pair fails the unchanged phone test ->
  1. means favor phone (energy mean <= host mean*(1-min saving), latency mean <= limit, LEARNING paired gate)
     = inconclusive (F1a): keep measuring; side below `comparable_measurement_targets`, host first
     (as advance_comparable_measurements); targets None/met -> side with fewer windows (host on ties);
  2. means against, < 2 host windows and bounds overlap (phone lower < host upper) (F2): one more host window;
  3. else decisive -> host verdict SERVER_PAIR_NOT_IMPROVED.
  Shared budget exhaustion / attempt cap unchanged and now also bound (1)/(2); reason for host windows of (1)/(2)
  = SERVER_PAIR_INCONCLUSIVE (comparison window before any host record keeps SERVER_COMPARISON_HOST_WINDOW).
- (b) group.reason -> group.reasons per batch + server_reason(); snapshot "reason" = active batch, + "reasons".
- (a) coherence-off follow -> reason CO_TENANT_POLICY_FOLLOW.
- ~16:00 UTC coherence.py + adaptive_decode_state.py edited in root. Design revisions after exploring
  (scratch/explore.py, synthetic controller runs):
  * ending an attempt when `comparable_measurement_targets` is None made the same owner burn the whole
    attempt cap within 2 windows (re-probe is evaluated at its ack with no new phone window) -> dropped;
  * "next square for both sides" fallback looped on host windows forever (host windows are not charged)
    -> fallback is now "side with the wider band (isqrt count), host on ties": sides take turns one band
    step at a time, phone windows charged, budget exhaustion -> attempt -> cap -> verdict (verified).
  * scenarios verified root vs base: inconclusive 73/61.78 -> root keeps measuring, phone verdict after
    host n=4 (base: host verdict at once); worse-on-1-host (80) -> second reference then host verdict;
    bound-resolved (95) and clear saving (40) identical on both; per-batch reason scenario reproduces the
    coherentEF None (base) vs SERVER_PAIR_NOT_IMPROVED (root).
- next: new test file tests/test_adaptive_server_probe.py (base-importable, string literals), update the
  4 affected assertions in test_adaptive_coherence.py, then C++ (c).
- ~16:05 UTC new tests/test_adaptive_server_probe.py (9 tests): root 9/9 pass; against base 6 FAIL (label (a),
  inconclusive pair, second reference, attempt cap, 2x per-batch reason (b)), 3 guards pass on both
  (bound-resolved rejection, clear saving, resolution host windows don't charge the budget).
  test_adaptive_coherence.py: 3 assertions updated (CO_TENANT_REASON for the coherence-off follow;
  server_reason(group, 1); batch-2 rejection now needs a second host reference). 30/30 on root.
  Full suite on root started (run_scheduler_tests.py root tests_root.json).
- ~16:08 UTC C++ (c): SERVER_PROBE_FIX_CPP.diff (examples/layersplit/ffn-split-client.{cpp,h}: per-call decode flag
  = every runtime-context slot has 1 row, else tokens==1; summary + decode_rpc/compute/overlap p50 use it);
  g++ -fsyntax-only clean with/without FFN_SPLIT_USB_TRANSPORT; git apply --check OK. Not built/run.
- ~16:09 UTC closed-loop replay (EF harness, unchanged) of coherentEF 002: base reproduces the recording
  (host 106/phone 11); root diverges at token 35 -> verdict[1]=P100 after the recorded host W6-W8, host 26 /
  phone 91 tokens, window energy 7441.7 vs 8408.5 J (ASSUMED_4P5W diagnostic).
- Coordinator FYI: dispatcher change merged into MAIN; must rebase + re-run coherence/adaptive tests at end.
- 16:11 UTC SERVER_PROBE_FIX.diff built (make_diff.sh; new file as /dev/null creation), git apply --check
  against CURRENT main OK (dispatcher merge present; my 3 touched files unchanged in main). rebased/ = fresh
  main copy + git apply: adaptive_coherence 30, adaptive_decode 89, adaptive_evidence_fixes 12,
  adaptive_runtime 44, adaptive_server_probe 9, dispatch_policy 22, prepare_trace_inputs_v2 5 all OK.
  Full suite on rebased/ started (tests_rebased.json); root/ full run still going.
- ~16:14 UTC final code tweak: the inconclusive check is now exactly the phone test on means (always requires
  `_learning_probe_improves`, as the phone test does); line wraps. Stale rebased run killed; root run continues.
  Replays: 002 root unchanged (diverges tok 35, host 26/phone 91, 7441.7 J); plainEF 000-006 base==root
  (coherence off; single-request replays do not exercise the co-tenant follow); coherentEF Gemma
  000/001/005/007 base==root.
- 16:16 UTC final diff rebuilt (sha256 42a7850b...), CPP diff (fa5e1fb0...); both `git apply --check` OK on
  current main; rebased/ re-synced from current main + diff (4 files byte-identical to root/); full suite on
  rebased/ started.
- ~16:35 UTC DONE. root full suite: 106 files / 1,566 tests, fails only admission (flaky 10 ms) + resident_router_subset
  (env). rebased full suite (current main + diff): 107 files / 1,589 tests, same two; admission reruns 2/3 pass.
  pyflakes clean. Both diffs `git apply --check` OK against current main (stdin closed). README.md written.
