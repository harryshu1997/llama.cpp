# Coherent server probe: keep measuring an inconclusive pair, plus three logging fixes (2026-09-24)

Fixes the defect found in `reports/20260924-coherent-policy-coalesced/README.md` section 4.4 ("New defect: one
noisy single-window pair decides a batch size for the server") and the three logging issues listed in that
section's caveats. Built in an isolated copy; the main tree was not edited, nothing was committed, no hardware
was used. Run artifacts were read from the desktop (scp, read-only).

| file | content |
| --- | --- |
| `SERVER_PROBE_FIX.diff` | scheduler change (Python): `_internal/adaptive_decode_ops/coherence.py`, `_internal/adaptive_decode_state.py`, `tests/test_adaptive_coherence.py` (3 assertions updated), new `tests/test_adaptive_server_probe.py` (9 tests). Paths `a/research_dev/scheduler/...`; apply from the repo root with `git apply`. |
| `SERVER_PROBE_FIX_CPP.diff` | issue (c), C++: `examples/layersplit/ffn-split-client.{cpp,h}`. Needs a llama-server rebuild and a transport identity re-materialization, so it can wait (see (c)). |
| `base/`, `root/` | pristine snapshot of main (15:50 UTC, before the dispatcher merge) and the working copy |
| `rebased/` | the CURRENT main (dispatcher merge included) + `git apply SERVER_PROBE_FIX.diff`; the final test runs use it |
| `PROGRESS.md` | checkpoints; `make_diff.sh` rebuilds the diff; `replay/` closed-loop replay of coherentEF 002 |

`git apply --check SERVER_PROBE_FIX.diff` and `git apply --check SERVER_PROBE_FIX_CPP.diff` pass against the
current main tree (repo root, stdin closed). The three scheduler files the diff modifies are byte-identical in
main and in the snapshot, so the dispatcher merge does not overlap it.

## 1. The defect

coherentEF, Qwen 002 alone at batch 1 [M]: W1 host 73.02 J/token, 604.8 ms; W4 P100 61.78 J/token, 618.0 ms
(a 15.4 % saving). At token 20 (153.5 s) `_server_probe_policy` evaluated the pair: phone upper
61.78 x 1.10 = 67.96 > host lower x 0.99 = 73.02 x 0.90 x 0.99 = 65.06, so the unchanged phone test failed, and
the code turned every failure into `verdict[1] = host`, `SERVER_PAIR_NOT_IMPROVED`, final for the layout
generation. The means favored the phone; only the +/-10 % bands (n < 4) overlapped. F1a/F2 made the
per-request controller keep measuring in exactly this case; the server probe had no such rule.

## 2. The fix (`coherence.py`)

When the pair fails the phone test (unchanged: `_learning_probe_improves`, phone energy upper <= host energy
lower x (1 - minimum saving), phone latency upper <= host latency upper x maximum latency), the new
`_server_pair_next_measurement` classifies the failure:

| pair after the failed test | action | group reason for the batch | mirrors |
| --- | --- | --- | --- |
| means favor the phone (the phone test on means: `_learning_probe_improves`, energy mean <= host mean x (1 - min saving), latency mean <= host mean x max latency) but the bounds overlap | keep the proposal, no verdict; measure the side below `comparable_measurement_targets` (host first, as `advance_comparable_measurements`); without such a target, the side with the wider band (fewer windows by `isqrt`, host on ties), so the sides take turns one band step at a time | `SERVER_PAIR_INCONCLUSIVE` | F1a (`_incumbent_inconclusive` -> `_reserve_comparable_measurements`) |
| means against the phone, fewer than 2 host windows (current + historical groups), energy bounds still overlap (phone lower < host upper) | one more host window, then re-judge | `SERVER_REFERENCE_BASELINE` | F2 (`_single_reference_rejection`) |
| anything else: means against with >= 2 host windows, bound-resolved (phone lower >= host upper), missing bounds | host verdict for this batch composition (unchanged) | `SERVER_PAIR_NOT_IMPROVED` | decisive rejection |

What bounds the extra measurement: phone windows still charge the shared `probe_tokens` budget while the
proposal is pending (host windows do not, as before); when it is spent the existing exhaustion path runs: the
attempt is counted, the proposal cleared, and only the `maximum_probe_attempts_per_context`-th exhaustion
(campaign: 4) becomes the host verdict `SERVER_PROBE_BUDGET_EXHAUSTED`. The exhaustion check now also runs
for a measured but inconclusive pair (before, a measured pair always decided at once). Host windows of the
resolution are at most one band step ahead of the phone side, so they are bounded too.

Unchanged: the phone verdict test, `_set_verdict` for phone verdicts, the elimination -> `SERVER_<reason>`
path, `server_policy_failed` (any failed phone window/control -> host verdict, whole server follows), the
comparison host window before any host record exists (`SERVER_COMPARISON_HOST_WINDOW`), budgets, bands,
`server_window_is_comparable`, the per-request controller. Everything is still inert with
`server_policy_coherence=false` (no group is ever created); the only coherence-off change is the label in (a).
Decisions are deterministic functions of the recorded windows, as before.

Two designs were tried and dropped (synthetic runs, `scratch/explore.py`): ending the attempt when no
resolving target exists made the same owner burn the whole attempt cap in two windows (its re-probe is judged
at the acknowledgement, before any new phone window), and "next square for both sides" could run host
windows forever (they are not charged).

## 3. The three small issues

**(a) `SERVER_POLICY_COHERENCE` with coherence off.** Path: `coherence._follow_coherent_policy`, the
coherence-off branch (the request-local rule: a session exploiting the host follows a co-tenant's running phone
policy at its next boundary). Following is intended with the flag off: the module docstring says "the default
path retains the earlier request-local coherence rule" and `AdaptiveCoherenceTests` (default config) test it.
So it is relabelled, not removed: the directive reason is now `CO_TENANT_POLICY_FOLLOW` (`CO_TENANT_REASON`).
plainEF's 4 decisions (003 at tokens 46/50 -> P75 and 62/66 -> P25, following 004's probes) were this path.
`campaigns/burstgpt/reports/20260924-coherent-policy-coalesced/analyze_coherent_arm.py` counts
`SERVER_POLICY_COHERENCE`; coherence-off follows no longer land in that count.

**(b) Reason per batch size.** `_AdaptiveServerPolicy.reason` -> `reasons: ((active_batch, reason), ...)`
(tuple, checkpoint-safe like `verdicts`), `server_reason(group, batch)`. `_set_verdict` clears only its own
batch's reason on a phone verdict. The snapshot keeps the key `reason` (now: the reason at the session's
active batch) and adds `reasons` (all batches); followers' `zero_assistance_reason` uses the active batch. In
coherentEF the batch-2 phone verdict at 235.7 s cleared the single group reason, so 56 later batch-1 host
decisions (004 x 23 at 295-349 s, 006 x 33) reported `None`; with the fix they report
`SERVER_PAIR_NOT_IMPROVED` (test below reproduces that sequence).

**(c) 2-row decode calls counted as `prefill_calls`.** The Python side does not parse these counters. The
server's `S41SERVERFFN` summary only sums the client summaries; the classification is in
`examples/layersplit/ffn-split-client.cpp` (`client::summary`): `tokens == 1` -> decode, else prefill, so a
coalesced 2-slot decode call (2 rows) counted as prefill. Fix (`SERVER_PROBE_FIX_CPP.diff`): each completed
call records a decode flag = every runtime-context slot contributed exactly one row (no runtime context:
`tokens == 1`, the old rule); `decode_calls`/`prefill_calls` and the `decode_rpc/compute/overlap_p50_ms`
percentiles use it. A 1-token prompt chunk stays counted as decode (as before). Checked with
`g++ -std=c++17 -fsyntax-only -Wall -Wextra` (with and without `FFN_SPLIT_USB_TRANSPORT`): clean; not built or
run. `ffn-split-client.cpp/.h` are untracked in git (`??`); the diff applies to the working-tree files. It
changes the llama-server binary (and `llama-layersplit`, `ffn-remote-resident-probe`, which link the same
client), so deploying it needs a rebuild and a transport identity re-materialization (runtime library hashes);
nothing in the scheduler depends on it. Observation, not changed: the macro tail fence runs only for 1-row
calls (`pending_tokens_ == 1`), so coalesced decode calls never use it.

## 4. Tests

New `tests/test_adaptive_server_probe.py` (campaign-like bands: `uncertainty_ppm=100000`,
`maximum_latency_ppm=1250000`, attempt cap 2). "base" = the same file run against the pristine snapshot.

| test | covers | base | root |
| --- | --- | --- | --- |
| `test_inconclusive_single_pair_keeps_measuring_until_the_phone_qualifies` | 73.0 vs 61.78 (002): no verdict, proposal kept, `SERVER_PAIR_INCONCLUSIVE`, 3 host windows, then `verdict[1]=phone`, no attempt used | FAIL | pass |
| `test_rejection_resting_on_one_host_window_takes_a_second_reference` | phone 80 vs host 73: `SERVER_REFERENCE_BASELINE`, second host window, then host verdict `SERVER_PAIR_NOT_IMPROVED` | FAIL | pass |
| `test_unresolved_pair_becomes_the_host_verdict_at_the_attempt_cap` | 2.7 % saving: measures until the budget is spent, attempt 1 (no verdict), next owner re-probes, attempt 2 = cap -> host verdict `SERVER_PROBE_BUDGET_EXHAUSTED` | FAIL | pass |
| `test_bound_resolved_rejection_is_final_at_once` | phone 95 vs 73 (phone lower >= host upper): host verdict at once | pass | pass |
| `test_clear_saving_qualifies_the_phone_at_once` | phone 40 vs 73: phone verdict at once | pass | pass |
| `test_host_windows_of_the_resolution_do_not_consume_the_probe_budget` | resolution host windows leave `probe_tokens` unchanged | pass | pass |
| `test_host_reason_survives_a_phone_verdict_at_another_batch_size` | (b), the coherentEF sequence: `verdict[1]=host`, then `verdict[2]=phone`, back at batch 1: reason and `zero_assistance_reason` = `SERVER_PAIR_NOT_IMPROVED`, `reasons={"1": ...}` | FAIL (`None`) | pass |
| `test_host_reason_of_another_batch_size_is_not_reported_on_the_phone` | (b), inverse: batch-2 host reason is not reported at a batch-1 phone verdict | FAIL | pass |
| `test_request_local_follow_is_not_labelled_as_server_coherence` | (a): coherence off, the follow still happens, reason `CO_TENANT_POLICY_FOLLOW` | FAIL | pass |

Updated in `tests/test_adaptive_coherence.py` (30 tests, all pass): the coherence-off follow expects
`CO_TENANT_REASON`; `group.reason` -> `server_reason(group, 1)`; `test_batch_verdicts_are_per_composition`
(phone 40 vs host 40 at batch 2, one host window) now expects `SERVER_REFERENCE_BASELINE` and one more host
window before the host verdict (the F2 rule), later windows shifted by 1 ms.

Full suite and pyflakes: see section 6 (filled in from `tests_root.json` / `tests_rebased.json`).

## 5. Replay of coherentEF 002 (estimate)

The evidence-fixes closed-loop harness (`replay/replay_decisions.py`, copied unchanged) drives the real
controller of a tree with 002's recorded windows until the first different decision, then synthesizes windows
from 002's own recorded windows of the same policy. 002 was the only member of its server group (the other
requests of that time were Gemma/Llama), so a single-request replay with coherence on is faithful up to the
divergence.

| tree | diverges at | final | host / phone tokens | window energy (ASSUMED_4P5W, diagnostic) |
| --- | --- | --- | --- | --- |
| base | none (reproduces the recording) | host | 106 / 11 | 8,185 J simulated (8,408 J recorded) |
| root | token 35 | P100 (`verdict[1]=phone`) | 26 / 91 | 7,442 J (-967 J vs recorded, -11.5 %) |

Root path: token 20 pair inconclusive -> host windows W6-W8 (the ones the run actually recorded, 74.4/73.1/73.1
J/token) -> at token 35 host n = 4 (band 5 %): phone upper 67.96 <= 73.4 x 0.95 x 0.99 = 69.0 -> phone verdict.
With `verdict[1] = phone` the batch-1 host tokens of 004 (86) and 006 (129) would also have run P100 [I]; at the
coherentEF batch-1 rates (host 75.0, phone 60.5 J/token) that is about 3.1 kJ more, about 4.1 kJ window energy
in total. This is fleet window energy with the assumed 4.5 W phone, not host RAPL+NVML; the section-4.4 gap it
addresses is the +3.4 kJ host energy of coherentEF's batch-1 phase vs plainEF. Not a measurement.

## 6. Full suite and pyflakes

`run_scheduler_tests.py TREE OUT.json` (the evidence-fixes runner: every `tests/test_*.py` in its own
`/usr/bin/python3` process, cwd = TREE, module mode for files with relative imports such as
`test_split_kv_attention.py`, `gguf-py` and `research_dev/spikes` linked from `shared/`):

| tree | files | tests | failing files |
| --- | ---: | ---: | --- |
| `root/` (pre-dispatcher main + fix) | 106 | 1,566 | `test_automated_runtime_admission.py` (known flaky 10 ms timing test: 10.05 ms), `test_resident_router_subset.py` (environmental: compiles `examples/layersplit/ffn-split-resident-router.cpp`, absent in a copy) |
| `rebased/` (CURRENT main incl. dispatcher merge + `git apply SERVER_PROBE_FIX.diff`) | 107 | 1,589 | the same two; the admission file then passed 2 of 3 isolated reruns (third: 10.28 ms) |

On `rebased/` individually: `test_adaptive_coherence` 30, `test_adaptive_decode` 89, `test_adaptive_evidence_fixes`
12, `test_adaptive_runtime` 44, `test_adaptive_server_probe` 9, `test_dispatch_policy` 22,
`test_prepare_trace_inputs_v2` 5: all OK. `test_two_phone_server_native` exits 0 with its native tests skipped
in the copy. The other named flaky test (`test_kv_touch_occupies_exactly_the_cache_pages`) passed.
`python3 -m pyflakes` on the four changed files: clean; all ASCII.

Coherence off, closed-loop replays of plainEF 000-006 (single-request, so the co-tenant follow is not
exercised): base and root identical. coherentEF Gemma 000/001/005/007: base and root identical.

## 7. Caveats

- Behavior change beyond the reported case: a server-level rejection that rests on one host window with
  overlapping energy bounds now takes a second host window first (F2 mirror), in any batch composition. A
  bound-resolved rejection is still immediate.
- An inconclusive server probe keeps the whole server on the probe (host comparison windows for all members,
  then the phone) until it resolves or the shared budget is spent; each switch costs a transition window. With
  campaign settings a 15 % saving resolves after 3 more host windows; a saving too small to ever pass the bands
  spends up to 80 phone tokens per attempt and up to 4 attempts before the host verdict.
- The inconclusive rule uses plain means; like F1a it only delays or re-tests a negative decision. A phone
  verdict still needs the unchanged bounds.
- The replay after divergence repeats 002's single eligible P100 window (61.78 J/token); the 004/006 figure is
  arithmetic on recorded rates.
- (c) is untested beyond a syntax check and needs a rebuild + transport identity before it shows up in runs.

## 8. Reproduce

```sh
W=/tmp/claude-1761612022/-home-myid-zs89458-Documents-llama-cpp-release/6fd7e39c-bd31-4e92-b800-5784f94be902/scratchpad/server-probe-fix
cd <repo root> && git apply --check $W/SERVER_PROBE_FIX.diff < /dev/null      # and SERVER_PROBE_FIX_CPP.diff
cd $W/rebased && PYTHONPATH=$PWD:$W/shared/gguf-py /usr/bin/python3 -m research_dev.scheduler.tests.test_adaptive_server_probe -v
cd $W/base && PYTHONPATH=$PWD:$W/shared/gguf-py /usr/bin/python3 $W/scratch/test_adaptive_server_probe.py   # 6 FAIL on the pristine tree
cd $W && /usr/bin/python3 run_scheduler_tests.py rebased tests_rebased.json
cd $W && PYTHONPATH=$W/shared/gguf-py /usr/bin/python3 replay/replay_decisions.py --source root/research_dev runs/coherentEF burstgpt_longtail_dev_v2:002
./make_diff.sh    # rebuilds SERVER_PROBE_FIX.diff from base/ and root/
```

`runs/coherentEF`, `runs/plainEF`: read-only copies of `ADAPTIVE_DECODE_OBSERVATIONS.json`, `RESULT.json`,
`SCHEDULER_DECISION_LOG.json`, `adaptive-timing-events.json` of the two desktop runs.
