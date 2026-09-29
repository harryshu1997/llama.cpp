# Scheduler change #1: one phone policy per server, coalesced multi-row phone calls (2026-09-24)

Implements change #1 of the offload accounting (`../20260923-offload-accounting/README.md`, section 8):
decode slots of one server share one FFN split policy, so a pair batches into one forward pass and one
multi-row phone call per layer instead of alternating passes, and the controller compares the phone against
the host per batch composition instead of rejecting batch-2 phone probes against batch-1 host history.

Status: code + tests done (sections 1-3). Physical A/B on the dev trace: section 4.

## 1. What was already there, and what was wrong with it

`server_policy_coherence` (default off, `_internal/adaptive_decode_contracts.py:78`) already elects one owner per
(model, desktop placement, phone layout generation, geometry) in `_internal/adaptive_decode_ops/coherence.py`,
makes followers execute the owner's policy at their first decode boundary, shares the HTP window lease
(`_unified/helper_preparation_ops/common.py:shared_helper_attachments`), pins READY helpers to the server's USB batch
plan, and marks mixed-policy windows non-comparable. On the 24-request trace the last arm with it (m4a8b, both batch
plans declared, coalesced qualified) produced 221 two-row + 221 three-row Qwen calls, but the second Qwen burst ran on
the host: five requests (88125-88130), 120 `SERVER_POLICY_COHERENCE` decisions, all fraction 0, all `exploitation`.

Root cause, reconstructed from the archived m4a8b windows and helper events
(`/home/zhihao/s42-trace-v2a-m4a8b-20260922-inputs/run-treatment-1/run/FAILURE_*`): the shared probe started at
80.4 s (owner 88121 at batch 2), membership went 2 -> 3 -> 2 and the desktop external-activity identity changed three
times, so the phone windows kept being reset or made incomparable: only 3 eligible phone windows (all at batch 3,
90.9-92.0 s) and no eligible host window at batch 3 (the owner's like-for-like host window at batch 3 was cut by 88119
completing at 96.3 s). By 103.2 s the members had recorded 85 window tokens (57 phone + 28 host) against the shared
80-token budget, and `_server_probe_policy` turned `SERVER_PROBE_BUDGET_EXHAUSTED` into `qualified=True,
policy=host` for layout generation 3 - a permanent host decision without a single like-for-like pair. The defects:

1. **Budget exhaustion was a permanent host decision** although nothing was measured.
2. **Host windows consumed the phone probe budget** (28 of the 85 tokens: followers still attaching, the comparison
   host window).
3. **One decision for all batch compositions**: a decision taken at one batch size applied to every other one (a
   batch-2 rejection, where the host's per-token energy halves, would also stop single-tenant assistance, i.e. the
   coherent controller could do worse than the per-request one).
4. **A follower's failure split the server**: a follower whose phone window or control failed went to the host alone
   while the owner stayed on the phone (mixed passes, the state coherence exists to avoid).
5. **Not diagnosable**: every host decision under coherence was logged as `SERVER_POLICY_COHERENCE` with fraction 0,
   whether it came from a measured pair, an exhausted budget or a comparison window.

(An earlier draft of this README blamed the owner's comparison host window for a permanent host decision. That is
wrong: `server_directive` sets `group.policy` before issuing the control, so `publish_server_policy` saw no change.)

## 2. Change set (scheduler only; the server needs nothing: `apply_dormant_host_share` releases when all slots agree)

| file | change |
| --- | --- |
| `_internal/adaptive_decode_state.py` | `_AdaptiveServerPolicy.qualified: bool` -> `verdicts: ((active_batch, policy), ...)`, `attempts: ((active_batch, count), ...)`, `reason` (tuples keep the `copy.copy` checkpoint safe) |
| `_internal/adaptive_decode_ops/coherence.py` | decisions per batch composition (`server_verdict`); a phone verdict from another batch size becomes the owner's proposal when the batch changes (it keeps running while one like-for-like host window is collected, bounded by the shared budget); a proposal left over from another composition is cleared while the owner runs a decided batch; `publish_server_policy` never records the host as a decision (comparison window, paused probe, recovery keep the pending proposal); `_server_probe_policy` stores a measured pair as the verdict, treats an owner elimination at this batch (`LATENCY_BOUND_EXCEEDED`, `ENERGY_DOMINATED`, ...) as a host verdict, and turns an exhausted budget into an attempt (proposal cleared, the next owner may probe again) until the `maximum_probe_attempts_per_context`-th exhaustion becomes the host verdict; `record_server_window` charges only phone windows to the shared budget; new `server_policy_failed` (any member's failed phone window or control -> host verdict for that batch, the whole server follows); `server_helper_window_eligible` = the server runs this policy by a verdict or a bounded probe; new `server_policy_snapshot` |
| `_internal/adaptive_decode_ops/windows.py`, `completion.py` | the failure branch of `record_window` and `control_failed` call `server_policy_failed` |
| `_internal/adaptive_decode_ops/reporting.py`, `_unified/adaptive_decode_control.py` | the controller snapshot and every `ASSISTANCE_DECISION` event carry `server_policy` (owner, running policy, proposal, per-batch verdicts, attempts, probe tokens, reason); followers report the server's reason as `zero_assistance_reason` (`SERVER_PAIR_NOT_IMPROVED`, `SERVER_PROBE_BUDGET_EXHAUSTED`, `SERVER_COMPARISON_HOST_WINDOW`, `SERVER_<elimination>`, `SERVER_PHONE_POLICY_FAILED`) |
| `campaigns/burstgpt/prepare_trace_inputs_v2.py` | declare the arm instead of hand-editing inputs: `--adaptive-decode-overrides-json`, `--qualify-phone-batch-plan MODEL_KEY=PLAN` (keeps every declared plan, removes a common `usb_batch_plan` override: two helper identities that differ only by executor id cannot share a server policy, lesson of m4a6/m4a7), `--transport-receipts-dir`, `--phone-boot-image/--phone-boot-image-sha256` |

Everything is inert with `server_policy_coherence=false` (no group is ever created), so the plain treatment and the
baseline are unaffected by the deploy sync. Not changed: the 1.25x latency bound, the minimum saving, the
per-request probe budget, `server_window_is_comparable`, helper lease sharing, `can_batch_with` in the server.

Mixed passes that remain possible (all pre-existing, all transient, all non-comparable and therefore excluded from
evidence): a member whose helper becomes unavailable (helper not yet READY at start, incompatible batch change, a
lease yield) or whose execution context is unknown runs the host until it rejoins; the last 2-3 tokens of a request
(`server_release_guard` tail). Making the whole server follow these would need a server-wide hold state that knows
members outside the group; not done.

## 3. Tests

`tests/test_adaptive_coherence.py`, new class `AdaptiveServerBatchVerdictTests` (8 tests). Fail-before: running the
file against the pre-change copies of the four controller files (deploy tree) with a one-line `server_verdict` shim
(`group.policy if group.qualified`) gives 9 failed / 21 passed (the 8 new tests + the updated budget assertion);
after the change 30 passed.

| test | behavior |
| --- | --- |
| `test_comparison_host_window_keeps_the_pending_proposal` | owner qualified alone; a follower joins at batch 2 and follows; both measure the phone; the owner opens one host window at batch 2 (both slots follow); the pair at batch 2 (phone 40 vs host 100 J/token) decides `verdict[2]=phone` and the server returns to the phone |
| `test_batch_verdicts_are_per_composition` | same with host 40 J/token: `SERVER_PAIR_NOT_IMPROVED`, `verdict[2]=host`, `verdict[1]` stays phone; the follower reports the server's reason and the snapshot carries `{"1": 1000000, "2": 0}`; the partner completes and the owner returns to the phone at batch 1 |
| `test_exhausted_probe_is_retried_by_the_next_owner_until_the_attempt_cap` | exhaustion is not a verdict: the next owner probes again; the second exhaustion (cap 2 in the test) is the host verdict |
| `test_follower_phone_failure_moves_the_whole_server_to_the_host` | a follower's failed phone window ends the phone for that batch and the owner follows (`SERVER_PHONE_POLICY_FAILED`) |
| `test_failed_follower_control_decides_the_batch_for_the_host` | same for a failed phone control |
| `test_host_windows_do_not_consume_the_shared_phone_probe_budget` | the comparison host window leaves the shared budget unchanged |
| `test_decided_batch_clears_a_proposal_from_another_composition` | a batch-2 proposal does not survive (and does not consume budget) once the owner is back at a decided batch 1 |
| `test_owner_elimination_decides_the_shared_probe` | an owner elimination at this batch ends the shared probe with `SERVER_LATENCY_BOUND_EXCEEDED`; the policy is no longer helper-window eligible |

Updated: `test_unknown_context_pauses_without_qualifying_the_host_policy` (the paused host window no longer charges
the phone budget) and three `group.qualified` assertions -> `server_verdict(group, 1)`.

`tests/test_prepare_trace_inputs_v2.py` (new, 3 tests): the plain derivation changes only paths/ids/cache policy;
the coherent derivation declares `adaptive_decode_overrides.server_policy_coherence=true`, qualifies only
`coalesced-batch` for `hot` while keeping both plans declared and removing the common override, leaves Gemma on
split-row, repoints receipts and boot identity; undeclared plans, unknown model keys, unpaired boot flags and
non-object overrides are rejected.

After the coordinator merged the evidence fixes (F1a/F1b/F2/F3/F4) into the same files, this change set is intact
(`verdicts/attempts/reason`, `server_policy_failed` in `windows.py`/`completion.py`, `server_policy` snapshot) and
coherence + prepare + evidence-fix + adaptive decode tests pass together (135 passed), pyflakes clean. The rig runs
reported here used the pre-merge tree (this change set only).

Runs before the merge: coherence + adaptive decode + prepare tests 122 passed; `test_adaptive_runtime` (44), `test_session_cow_transaction`
(45), `test_catalog_materialization`, `test_campaign_inputs` OK. `python3 -m pyflakes` on all touched files: clean.
Full `/usr/bin/python3 tests/run_all.py` on the final code: 128 test files, 127 OK, 1 FAILED =
`test_automated_runtime_admission.py::test_cached_synthetic_refinement_is_below_ten_milliseconds` (10.4 ms vs the
10 ms bound, the known timing-flaky test; fails the same way in isolation and does not touch this code). The other
known flaky test, `test_kv_touch_occupies_exactly_the_cache_pages`, passed.

## 4. Physical A/B on the dev trace

Trace `burstgpt_longtail_dev_v1` (`/mnt/storage/burstgpt-source/longtail_dev_v1`): 6 requests (3 Qwen, 2 Gemma,
1 Llama overlay), 1,327 prompt / 1,071 output tokens, replay span 392 s. Arms (desktop inputs
`/home/zhihao/s42-trace-longtaildev-<arm>-2026092{3,4}-inputs`):

| arm | inputs | phone stack | differences |
| --- | --- | --- | --- |
| baseline | other agent, `...-baseline-20260923` | idle (dual-stack rig.json, desktop-baseline selection) | - |
| dual treatment | other agent, `...-treatment-20260923` | dual-engine worker (HTP + OpenCL secondary) | not the standard stack; reported, not the A/B control |
| **plain** | `...-plain-20260924` | standard OP15 stack | the 2026-09-23 long-tail treatment inputs on the dev trace; deploy transport identity copied unchanged (as the 09-23 arms) |
| **coherent** | `...-coherent-20260924` | standard OP15 stack | plain + `adaptive_decode_overrides={"server_policy_coherence": true}` + Qwen `qualified_phone_batch_plans=["coalesced-batch"]` (both declared) + the transport identity those need (below) |

Why the coherent arm cannot reuse the deploy transport identity: a 4-row coalesced Qwen call carries
4 x 5,120 x 2 = 40,960 bytes and the deploy identity's receipts are 7,680/10,240 bytes, so the coalesced helper would
be rejected with `TRANSPORT_PROFILE_INCOMPLETE` (exactly the m4a5 failure). The coherent arm binds the 2026-09-22 task1
receipts (7,680 / 10,240 / 40,960 bytes, depth 4; the m4a8b materialize command with a new identity id) and the boot
image they were measured under (`f13c7c03...`; kernel notes/BTF verified 2026-09-22, phone boot_id unchanged since,
checked 2026-09-24). The plain arm keeps the old receipts on purpose: with 40,960-byte receipts its SHADOW coalesced
variant would become transport-admissible and could be picked as the dormant runtime hint
(`_dormant_phone_ffn_parent_parameters` does not filter by maturity), which would change the plain arm's server
launches. Gemma stays split-row in both arms: its server runs `parallel=8`, a coalesced call would need 61,440-byte
receipts. Under coherence a Gemma pair still shares one forward pass; only its USB calls stay one row each.

Rig tooling (this directory; the desktop copies are in `/home/zhihao/s42-coherent-20260924-rig/`):

```sh
bash derive_inputs.sh plain coherent          # outside the lock: writes only the two input directories
python3 run_arms.py coherent plain            # flock -w 7200: rsync -c staging -> deploy, then per arm
                                              # preflight, check_admission.py, run; battery + notify code
                                              # before/after each stage, stop on notify 512
python3 analyze_coherent_arm.py --arm baseline=<run> --arm plain=<run> --arm coherent=<run> --out ANALYSIS.json
```

### 4.1 Run log

- 03:46 first chain: coherent preflight PASS, then my admission check crashed (`ModuleNotFoundError: research_dev`,
  the checker ran without `PYTHONPATH`); chain stopped, lock released, nothing ran. Fixed (checker runs with
  `PYTHONPATH=<deploy source>`, `PYTHONDONTWRITEBYTECODE=1`; a passed preflight is reused).
- 03:55 chain restarted: sync no-op, coherent preflight-1 reused, admission PASS (one qualified coalesced Qwen helper
  per parent, 40,960 bytes at depth 4; Gemma split-row), coherent run 03:56-04:11 exit 0, plain preflight 04:11.
- Coordinator decision: after dev_v1 run the new pair-bearing trace `longtail_dev_v2` instead of v2a (v2a inputs kept,
  not run): baseline, plain, coherent, plus one coherence-only arm (section 4.3), each derived by `derive_inputs.sh`.

### 4.2 dev_v1 (6 requests, never concurrent)

Filled in below as arms finish (`results/` on the desktop: `/home/zhihao/s42-coherent-20260924-rig/results/`).

| arm | status | duration s | host kJ (CPU + GPU) | vs baseline | Qwen phone-policy tokens | Gemma phone-policy tokens | calls by rows | mixed passes | outputs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| baseline (other agent) | PASS 6/6 | 897.5 | 76.67 (51.49 + 25.18) | - | 0 / 388 | 0 / 636 | - | 0 | - |
| dual treatment (other agent, dual-engine stack) | PASS 6/6 | 905.4 (+0.9 %) | 71.76 (45.61 + 26.16) | -6.4 % | 240 / 388 (61.9 %) | 52 / 636 (8.2 %) | Qwen 2,940 x 1, Gemma 312 x 1 | 0 | 6/6 identical |
| coherent (Qwen coalesced) | PASS 6/6 | 852.7 (-5.0 %) | 68.95 (44.59 + 24.36) | -10.1 % | 349 / 388 (89.9 %) | **0 / 636** | Qwen 5,427 x 1 | 0 | 6/6 identical |
| plain (standard stack, deploy identity) | PASS 6/6 | 916.1 (+2.1 %) | 60.47 (36.14 + 24.33) | **-21.1 %** | 342 / 388 (88.1 %) | 565 / 636 (88.8 %) | Qwen 4,008 x 1, Gemma 4,536 x 1 | 0 | 6/6 identical |

`active_slots_peak = 1` in every dev_v1 arm: no pairs, so no multi-row calls, no mixed passes and no batch-2
windows are possible, and coherence cannot change a pass. What it does change on a serial trace: the second and third
Qwen requests start on the server's phone verdict instead of re-probing (`SERVER_POLICY_COHERENCE` x90 at fraction
1.0, 0 host decisions under coherence), which is why Qwen's phone share is 89.9 % vs 88.1 % in plain and Qwen phone
windows average 43.8 J/token (545 ms) vs 48.2 (551 ms) in plain (host windows 71.2 vs 69.5). The whole-run energy
difference between plain (-21.1 %) and coherent (-10.1 %) is Gemma: 88.8 % phone-assisted in plain, 0 % in coherent
(next paragraph).

**Gemma was never probed in the coherent arm (not "probed and lost").** Diagnosis from `run-coherent-1` vs `run-plain-1`:

| | plain | coherent |
| --- | --- | --- |
| phone layout events | 3 PROPOSED / 3 READY / 3 SESSION_READY | 133 PROPOSED, 3 READY (Qwen), **130 SESSION_FAILED + 130 TRANSITION_FAILED** |
| Gemma helper events (001, 003) | 139 WINDOW_LEASE_SELECTED, 138 PHYSICAL_WORK_RECORDED | 130 PREPARATION_STARTED -> 130 PREPARATION_FAILED, no attach |
| Gemma decisions | probes and exploitation (001 ends on the phone, 003 `PROBE_INCOMPLETE` -> host) | 156 x `PHONE_HELPER_UNAVAILABLE`, no candidate ever reached the controller |

Every failure reads "physical_helper_preparation_failed: exact partial phone residency transition is unavailable"
(`adapters/heterogeneous_rig.py:_transition_conflicting_executors`). The direct phone session was launched for the
first helper (Qwen, `coalesced-batch`); switching HTP2 to the Gemma shards is a partial reconfiguration, and
`supports_partial_reconfiguration` requires `PhoneTransportContract.shares_resident_session_with`, which compares
`batch_plan`. Rebuilding both helpers' contracts from the run records with the adapter's own code: they differ only in
`batch_plan`, `max_payload_bytes` and `profile_id`; `shares_resident_session_with` is False, and True if Gemma's plan
were `coalesced-batch` (the other two fields are not compared). m4a8b had the same failure (206 helper events with that
message, all Gemma requests `PHONE_HELPER_UNAVAILABLE`), so no arm with "Qwen coalesced + Gemma split-row" has ever
assisted Gemma.

Ruled out: transport identity/admission (Gemma's split-row helper is QUALIFIED and admitted, one per parent; the plain
arm's Gemma helper uses the same catalog rules), duplicate helper identities (one qualified helper per parent), a host
verdict leaking across models or batch sizes (coherence groups are keyed by model artifact, desktop placement and layout
generation; Gemma never had a group because its helper never became available), and a controller decision not to probe.

Not a defect in the #1 controller code, and not something to fix by relaxing the adapter check (that would weaken a
fail-closed identity rule for the phone session). It is a configuration conflict: one phone session serves every model
and carries one batch plan, and with the current receipts (<= 40,960 bytes) coalesced is admissible only for Qwen
(Gemma `parallel=8` needs 61,440 bytes). Fix in this change set: `prepare_trace_inputs_v2.py` now refuses assisted models
with different qualified phone batch plans unless `--allow-mixed-phone-batch-plans` is passed (new test
`test_mixed_qualified_batch_plans_across_models_are_refused_by_default`; fail-before on the previous builder, 4/4 after).
The coherent arms requested here pass the flag explicitly. The dev_v2 coherent arm (same mixed configuration) will lose
Gemma the same way and does not need repeating for that reason; its Qwen coalesced/multi-row data stay valid, and the
coherence-only arm `dev2coherentsr` (both split-row) is the clean comparison for the controller change on both models.

Pattern check requested by the coordinator (dev requests ending on the host despite cheaper phone windows): in the
dual arm 000, 001 and 003 end on the host (000 `MEASURED_REJECTION`, 001 `INCUMBENT_NO_LONGER_BENEFICIAL` x130, 003
`PROBE_INCOMPLETE`). The receipt flag `measurement_eligible` is true for most of their windows (000: 27/36,
001: 134/146), but every window has `energy_attribution_kind=diagnostic` and carries `ASSUMED_4P5W`, so
`energy_measurement_eligible` is false for all of them and operational selection runs only on the
assumed-phone-power path (`allow_assumed_phone_power_for_operational_selection`). The same holds in my arms
(all windows `diagnostic` + `ASSUMED_4P5W`). Plain dev_v1: 000 and 001 end on the phone; only 003 (Gemma, 58 tokens)
shows the pattern (`PROBE_INCOMPLETE` x12 -> host; phone windows 48.3 vs host 56.4 J/token). Coherent dev_v1: no Qwen
request shows it (all three end on the phone, 83 of 88 phone windows eligible); Gemma cannot be judged there (never
probed, see above).

### 4.3 dev_v2 (9 requests with same-model overlapping pairs)

Arms, all from `prepare_trace_inputs_v2.py` with the dev_v2 files; plain, coherent and coherence-only share the
coalesced-receipt transport identity (file sha256 `23b7f356...`), so they differ only by the #1 flags:

| arm | derived from | differences vs plain |
| --- | --- | --- |
| dev2base | 09-23 long-tail baseline inputs (desktop-baseline selection, deploy identity) | reference |
| dev2plain | 09-23 long-tail treatment inputs | - |
| dev2coherent | same | `server_policy_coherence` + Qwen `coalesced-batch` qualified |
| dev2coherentsr | same | `server_policy_coherence` only |

Admission for these arms runs `launch.py --resolve-only` (catalog only, no hardware) plus `check_admission.py`; the rig
was physically preflighted twice in the same lock hold.

dev2base: PASS 9/9, 1,154.3 s, host 96.75 kJ (64.90 CPU + 31.85 GPU), `active_slots_peak = 2`. Even with the trace's
8 overlapping same-model arrival pairs, today's dispatcher formed exactly one decode pair: Qwen 003 (353-434 s) with
Qwen 004 (356-492 s), about 78 s together; every other request ran alone (Gemma 000 -> 001 -> Llama -> Qwen 002 ->
003/004 -> Gemma 005 (617 tokens) -> Qwen 006 -> Gemma 007). Results of the other arms: below.

dev2plain: run PASS 9/9, 963.8 s (-16.5 %), host 81.81 kJ (51.16 + 30.65, **-15.4 %**); outputs 8/9 identical
(Gemma 005 diverges at token 146 of 617, strict check FAIL). The Qwen pair formed again (003/004 at batch 2) and ran
with mixed policies: **109 mixed passes** on the Qwen server, no shared 2-row forwards, all 672 Qwen USB calls one row.
Batch-2 Qwen windows: host 95.8 J/slot-token (47.9 per produced token) at 786 ms, phone-under-mix 144.3 (72.1) at
1,194 ms, so both 003 and 004 were rejected at batch 2 (`MEASURED_REJECTION`) and stayed on the host after the partner
left. Qwen phone share only 18.8 % (112/595): 002 and 006 ran alone and still ended on the host with cheaper phone
windows (002 `INCUMBENT_NO_LONGER_BENEFICIAL`, 006 `INCONCLUSIVE`), the evidence pattern the separate diagnosis covers.
Gemma phone share 83.6 % (666/797; 005 on the phone at 34.6 vs 57.2 J/token, 000/001 short requests stayed on the host).

dev2coherent (Qwen coalesced + Gemma split-row): run PASS 9/9, 1,088.3 s (-5.7 %), host 77.33 kJ (44.60 + 32.74,
**-20.1 %**, i.e. 5.5 % below plain); outputs 8/9 identical, the same Gemma 005 divergence at token 146 as in plain
(so a phone-path near-tie, not #1). **The batch-plan conflict hit Qwen this time**: dev_v2 starts with Gemma, so the
phone session was launched split-row, every Qwen helper preparation failed and all 4 Qwen requests were
`PHONE_HELPER_UNAVAILABLE` (0 Qwen phone calls, 0 multi-row calls). The Qwen pair ran both-host in shared passes
(0 mixed passes; batch-2 host windows 78.8 J/slot-token = 39.4 per produced token at 637 ms, vs 47.9 per produced token
at 786 ms for the plain arm's batch-2 host windows). Gemma under coherence: 82.9 % phone share (661/797), 005 and 007
started on the server's phone verdict (`SERVER_POLICY_COHERENCE` x151, x7) and Gemma phone windows averaged
26.1 J/token vs 35.6 in plain. So this arm cannot show coalesced calls on this trace, and repeating it would not change
that: whichever model starts the phone session decides the plan. The coherence-only arm is the clean test.

**Lock-out, both directions (answer to "the coherent path locks one model out").** The locked-out model is whichever one
did *not* start the phone session: dev_v1 starts with Qwen (session launched `coalesced-batch`, Gemma locked out, 130
failed transitions), dev_v2 starts with Gemma (session launched `split-row`, Qwen locked out, 43 failed transitions), both
with "exact partial phone residency transition is unavailable". It is caused by the mixed qualified batch plans, not by
the coherence controller: no coherence group or verdict exists for the locked-out model (its helper never becomes
available), the failing check is the adapter's session-sharing rule, and the coherence-only arm below runs the same
controller with one batch plan.

**"No pair formed in the phone arms" is a measurement artifact.** `active_slots_peak` is the maximum of one `/slots` probe
taken at each request's first token (`adapters/http_backend.py:315-330`), and a request under adaptive control has its
`on_active_batch` replaced by the controller's `update_active_batch` (`http_backend.py:1716-1720`), so it never reaches
the runner's sample list: 003/004 have no sample at all in either phone arm. The decode intervals and the adaptive windows show the Qwen pair in every dev_v2 arm: 003+004 overlapped
78 s (desktop), 109 s (plain), 80 s (coherent), with 119/126 and 119/124 window tokens at `active_batch = 2` in plain and
coherent. So the #1 path was exercised by this one pair: plain = 109 mixed passes and phone rejected at batch 2;
coherent = both slots host in shared passes (0 mixed) because Qwen was locked out.

**What serializes same-model requests: the dispatcher, identically in all arms, not the phone.** Every queued request
is released by a `replanned` wake at its predecessor's completion, although its first planned start is far later:
dev2base Gemma 001 arrives at 1 s with 000, is first planned at 431 s, and is released at 161 s when 000 ends; 003 and
004 are both released at 002's end (350/351 s), which is the only reason they pair. Same pattern in plain (released at
70, 207/207 s) and coherent (73, 271/272 s). This is the transition-lease plus calendar serialization of accounting
section 4 (followers planned after the leader's predicted end, woken by completion replans), i.e. change #2
(work-conserving admission), outside #1's scope and not caused by the phone (the desktop arm shows it too).

**Non-identical outputs:** in both phone arms the one mismatch is Gemma 005 (617 tokens), first differing token 146
(zero-based) against dev2base, the same position in plain and coherent: a phone-assisted Gemma near-tie, not a #1
effect. Neither Qwen nor the other Gemma requests differ.

### 4.4 dev_v2 after the evidence fixes: plainEF vs coherentEF, both models coalesced (2026-09-24)

**Setup.** The current main tree (this change set + the merged evidence fixes F1a/F1b/F2/F3/F4 + the mixed-plan guard)
was synced into the deploy source under the rig lock (exactly the 15 files of `SYNC_MANIFEST_EF.sha256`; sync, both
physical preflights and both runs in one lock hold, 14:52-15:38 UTC; checkpoints in `PROGRESS_EF.md`). Both arms were
derived by `derive_inputs_ef.sh` (`prepare_trace_inputs_v2.py`, 09-23 long-tail treatment inputs + dev_v2 trace) and
bound to the same both-models identity `s43-coalesced-both-20260924` (file sha256 `e95f0b94...`, 15 receipts,
boot `f13c7c03`), so the input diff is exactly the flags:

| arm | inputs (`/home/zhihao/...`) | flags | admission on the physical preflight catalog |
| --- | --- | --- | --- |
| dev2plainEF | `s42-trace-longtaildev2-plainEF-20260924-inputs` | none (split-row qualified for both, coherence off) | one QUALIFIED split-row helper per desktop parent, both models (coalesced SHADOW) |
| dev2coherentEF | `s42-trace-longtaildev2-coherentEF-20260924-inputs` | `server_policy_coherence` + `hot=coalesced-batch` + `cold=coalesced-batch` (no mixed plans, no `--allow-mixed-phone-batch-plans`) | one QUALIFIED coalesced-batch helper per parent, both models; Qwen 4 rows = 40,960 B -> `payload-40960`; Gemma helpers at the catalog `parallel=2` -> 15,360 B (`payload-38400`) |

Battery notify code 0 at every check (level 79-80 %, USB 500 mA, charging from 15:16). Tooling: `run_arms_ef.py`,
`analyze_ef.py`, `tables_ef.py`, `phase_energy.py`; result JSONs and full tables in `data-ef/` (desktop copies
`/home/zhihao/s42-coherent-20260924-rig/results/`). Run dirs: `<inputs>/run-dev2plainEF-1/run`,
`<inputs>/run-dev2coherentEF-1/run`. dev2base (desktop only, deploy identity) is the reference; dev2plain, dev2coherent
and dev2coherentsr ran before the evidence fixes with the 09-22 task1 identity (9 receipts) and are shown for context.

**Energy, duration, exactness** (host = RAPL package + NVML board; outputs compared token by token with dev2base):

| arm | code | duration s | host kJ (CPU + GPU) | vs dev2base | vs plainEF | outputs identical to dev2base |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dev2base | - | 1,154.3 | 96.75 (64.90 + 31.85) | - | - | - |
| **dev2plainEF** | fixes | 1,035.1 | 76.31 (46.37 + 29.93) | **-21.1 %** kJ, -10.3 % s | - | 6/9 (Qwen 003 @103, Qwen 004 @210, Gemma 005 @146) |
| **dev2coherentEF** | fixes | 924.5 | 71.86 (43.50 + 28.36) | **-25.7 %** kJ, -19.9 % s | **-5.8 %** kJ, -10.7 % s | **9/9** |
| dev2plain | pre-fix | 963.8 | 81.81 | -15.4 % | | 8/9 (005 @146) |
| dev2coherent (Qwen coalesced, Gemma split-row) | pre-fix | 1,088.3 | 77.33 | -20.1 % | | 8/9 (005 @146) |
| dev2coherentsr (coherence, split-row) | pre-fix | 1,107.5 | 77.48 | -19.9 % | | 7/9 (003 @103, 004 @210) |

**Phone share, calls by rows, mixed passes, lock-out** (phone share = decode-window tokens under a phone policy; calls
from `S41SERVERFFNSHAPE` `tokens`, all processes of a server role; mixed passes = the server's
"release skipped: mixed slot policies" counter):

| arm | Qwen phone share | Gemma phone share | Qwen calls by rows | Gemma calls by rows | Qwen mixed passes | helper prep. / phone transition failures |
| --- | ---: | ---: | --- | --- | ---: | --- |
| dev2plainEF | 278/595 (46.7 %) | 692/797 (86.8 %) | 1,692 x 1 | 11,760 x 1 | 121 | 0 / 0 |
| dev2coherentEF | 234/595 (39.3 %) | 696/797 (87.3 %) | 96 x 1 + **660 x 2** | 11,856 x 1 | **1** | 0 / 0 |
| dev2plain | 112 (18.8 %) | 666 (83.6 %) | 672 x 1 | 10,488 x 1 | 109 | 0 / 0 |
| dev2coherent | 0 | 661 (82.9 %) | 0 | 15,400 x 1 | 0 | 43 / 43 (Qwen locked out) |
| dev2coherentsr | 338 (56.8 %) | 692 (86.8 %) | 1,932 x 1 + 66 x 2 forward, all 2,064 USB calls 1-row | 10,856 x 1 | 5 | 0 / 0 |

The Qwen pair 003/004 overlapped 111.8 s (plainEF) and 72.7 s (coherentEF) in decode (window spans), 246 / 242 Qwen
window tokens at `active_batch = 2`; no Gemma pair formed in any arm (dispatcher serialization, change #2), so every
Gemma call is 1-row and this A/B says nothing about 2-row Gemma calls. RESULT `active_slots_peak` is 1 in both EF arms
(the under-report of 4.3).

**J/token by batch composition, Qwen** (window receipts: `energy_per_token_uj` = whole-fleet energy incl. the assumed
4.5 W phone per slot token, divided by `active_batch` = per produced token; `latency_per_token_us` per slot token;
token-weighted over all valid windows, tokens in brackets; eligible-only rows in `data-ef/TABLES_EF.md`, same picture):

| arm | batch 1 host | batch 1 phone | batch 2 host | batch 2 phone |
| --- | --- | --- | --- | --- |
| dev2plainEF | 74.1 J, 614 ms (117) | 60.4 J, 576 ms (219) | 48.3 J, 806 ms (187, mixed passes) | 72.6 J, 1,232 ms (59, mixed passes) |
| dev2coherentEF | 75.0 J, 617 ms (321) | 60.5 J, 591 ms (18) | 37.0 J, 592 ms (26, shared passes) | **30.9 J, 596 ms (216, 2-row calls)** |
| dev2coherentsr | 71.3 J, 597 ms (21) | 60.4 J, 582 ms (314) | 39.4 J, 636 ms (224, shared passes) | 39.3 J, 754 ms (24) |
| dev2plain | 76.9 J, 626 ms (279) | 65.8 J, 581 ms (58) | 48.2 J, 786 ms (191, mixed passes) | 72.7 J, 1,194 ms (54, mixed passes) |

Gemma (batch 1 only): host 56.7 / 57.0 J, phone 33.9 / 32.9 J per token (plainEF / coherentEF). Mean phone RPC per
call: Qwen 1-row 9.90 ms, 2-row 10.23 ms (+3 %; coherentEF), Gemma full-width 1-row calls 6.7-8.4 ms in both arms.

**Wall-clock host energy by phase** (`phase_energy.py`: RAPL + NVML from `resource-samples.jsonl` integrated over the
union of that model's adaptive windows at that batch size; host only; totals reproduce RESULT to 0.01 kJ):

| arm | Qwen batch-2 phase | Qwen batch-1 phase | Gemma phase | all decode windows | outside decode windows |
| --- | --- | --- | --- | --- | --- |
| dev2plainEF | 113.0 s, 13.36 kJ (54.3 J/token) | 199.8 s, 21.91 kJ (64.6) | 337.1 s, 28.04 kJ (35.7) | 648.7 s, 63.15 kJ | 386.4 s, 13.16 kJ |
| dev2coherentEF | **72.7 s, 7.62 kJ (31.5)** | 211.8 s, 25.34 kJ (73.9) | 338.4 s, 27.25 kJ (34.7) | 621.7 s, 60.04 kJ | 302.8 s, 11.82 kJ |
| dev2coherentsr | 82.0 s, 9.89 kJ (39.9) | 195.3 s, 20.21 kJ (60.3) | 382.2 s, 31.30 kJ (39.9) | 658.3 s, 61.23 kJ | 449.2 s, 16.25 kJ |
| dev2plain | 108.8 s, 13.24 kJ (54.0) | 210.2 s, 25.31 kJ (74.4) | 347.1 s, 30.74 kJ (39.2) | 664.9 s, 69.12 kJ | 298.9 s, 12.69 kJ |

plainEF -> coherentEF (-4.45 kJ): Qwen pair phase -5.74 kJ (-40 s), Qwen batch-1 phase +3.43 kJ, Gemma -0.79 kJ,
outside decode windows -1.34 kJ (-84 s).

**Evidence-fix markers:**

| marker | dev2plainEF | dev2coherentEF | pre-fix arms |
| --- | --- | --- | --- |
| `HELPER_PHONE_SESSION_LOAD` decisions (F4) | 16 (Gemma 000 at run start, fraction 0 while the phone session loads) | 15 (same) | 0 |
| `TOKEN_STREAM_CATCH_UP` windows (F3, `measurement_ineligible_reason`) | 1 (Gemma) | 2 (Gemma) | 0 |
| `CANDIDATE_REQUALIFIED` decisions (F1b) | 0 | 0 | 0 |
| windows with any `measurement_ineligible_reason` | 1 | 2 | 0 |

F1a/F2 have no reason code of their own; their effect shows in plainEF: Qwen 002 and 006 (alone) now end on the phone
(pre-fix plain: host by `INCUMBENT_NO_LONGER_BENEFICIAL` / `INCONCLUSIVE`), Qwen share 18.8 -> 46.7 %, host -6.7 % vs
dev2plain. Qwen 003/004 are still rejected at batch 2 in plainEF (`MEASURED_REJECTION`,
`INCUMBENT_NO_LONGER_BENEFICIAL`; phone under mixed passes 72.6 vs host 48.3 J per produced token), a correct rejection.

**Interpretation.**

1. The coherent pair path works: coherentEF ran the Qwen pair as one forward pass with one 2-row USB call per phone
   layer (660 two-row calls; 1 mixed pass vs 121). At batch 2 the phone costs 30.9 J per produced token at 596 ms/slot
   token, vs 37.0 J / 592 ms for the same server both-host in shared passes (its comparison windows) and 48.3 J /
   806 ms (host) or 72.6 J / 1,232 ms (phone) under the mixed passes of plainEF. The pair phase took 7.62 kJ in 72.7 s
   vs 13.36 kJ in 113.0 s in plainEF (-43 % energy, -36 % time). A 2-row call costs 3 % more RPC time than a 1-row call.
2. No lock-out: with both models qualified for `coalesced-batch` the phone session carries one plan; 0 helper
   preparation and 0 transition failures (dev2coherent: 43 / 43), Gemma phone share 87.3 %, Gemma 000/001/005/007 all on
   the phone verdict.
3. **New defect: one noisy single-window pair decides a batch size for the server.** Qwen 002 (alone) collected one
   eligible host window (73.0 J/token fleet, 605 ms) and one eligible phone window (61.8 J, 618 ms; a 15.4 % saving);
   at 153 s `_server_probe_policy` set `verdict[1] = host` with `SERVER_PAIR_NOT_IMPROVED`, final for the layout
   generation. So 002 after token 23, 004 after 003 left (295-351 s) and all of 006 (663-832 s, not one phone window)
   ran on the host: 18 Qwen batch-1 phone tokens vs 219 (plainEF) / 314 (coherentsr), batch-1 phase 73.9 vs 64.6 /
   60.3 J/token, about +3.4 kJ vs plainEF (at the coherentsr batch-1 rate coherentEF would be near 67 kJ, -30 % vs
   dev2base; an estimate, not measured). Mechanism: the server probe decides as soon as both sides have one valid record,
   and `_bounds` uses uncertainty max(2 %, `uncertainty_ppm` / isqrt(n)) = 10 % until n = 4, so the test
   `phone_mean x 1.10 <= host_mean x 0.90 x (1 - 0.01)` needs a measured saving of >= 19 % at n = 1 (>= 10.4 % at
   n = 4); an inconclusive pair is recorded as a measured host verdict. The same rule set coherentsr's batch-2 verdict to
   the host and coherentEF's batch-2 verdict to the phone (decision-time windows at 235 s: phone 60.6/61.6 vs host
   76.3/75.5 J per slot token, owner + follower, a 19-21 % saving, just above the threshold), i.e. at n = 1 the verdict is noise-limited in both directions. F1a/F2 made the per-request controller
   keep measuring in exactly this case; the coherent server probe has no such rule. Suggested fix (not implemented
   here): when the bounds overlap, keep the proposal and collect more alternating windows within the shared probe
   budget (the existing exhaustion/attempt path bounds it) instead of `_set_verdict(..., "SERVER_PAIR_NOT_IMPROVED")`;
   reserve the final host verdict for a pair whose phone lower bound exceeds the host upper bound, or for the attempt
   cap.
4. Duration: of the 110.6 s between plainEF and coherentEF, 27 s is decode (window union 648.7 -> 621.7 s: pair -40 s,
   batch 1 +12 s) and 84 s is outside decode windows (per-request launch, load and prefill before the first window,
   e.g. Qwen 002 91 vs 56 s, Gemma 005 72 vs 46 s, Qwen 006 102 vs 88 s); these phases vary by tens of seconds per request
   between runs (coherentsr: 85, 85, 132 s) and are not a #1 effect.
5. Exactness: coherentEF matches dev2base on all 9 requests. The plainEF mismatches sit at the same positions as in
   other arms (Qwen 003 @103 and 004 @210 as in coherentsr, Gemma 005 @146 as in dev2plain/dev2coherent), and 003 @103
   is a host batch-2 window in both plainEF and coherentsr: near-ties that flip with the earlier phone/host history of
   the request, not an error of coherence or coalescing.

**Repeat (2026-09-24, `../20260924-phone-reprovision/RIG_RESULTS.md` section 7).** An exact repeat of coherentEF
(dev2coherentEF2) gave 69.64 kJ / 1,042.8 s (-3.1 % / +12.8 %), 9/9 identical, and inverted both Qwen verdicts
(batch 1 phone, batch 2 host by `SERVER_PAIR_NOT_IMPROVED`), confirming that the n = 1 server verdict of point 3 is
noise-limited in both directions. The same report has change #4 on this configuration (53.69 kJ).

**Caveats.** One run per arm. Per-request pre-decode phases vary by tens of seconds between runs, so the whole-run
duration and part of the 4.45 kJ gap are noise; the phase split above isolates the pair effect (-5.74 kJ, -40 s), which
is large against it. The pre-fix arms differ in code and identity, so comparisons with them are indicative. Phone
energy is the assumed 4.5 W model (included in window J/token, not in host kJ). The 660 two-row calls appear as
`prefill_calls` in the server's transport summary line (`decode_calls` 96): the summary counts any multi-row call as
prefill; they are decode passes of the pair (216 phone slot tokens = 108 two-slot passes x about 6 phone layers). With
coherence off, a request that follows a co-tenant's running phone policy (`_follow_coherent_policy`) is logged with the
same reason `SERVER_POLICY_COHERENCE` (plainEF: 4 decisions for 003; the RESULT configuration has
`server_policy_coherence=false`). The server snapshot's `reason` is group-level and is overwritten by the batch-2
decision, so 56 later batch-1 host decisions in coherentEF report reason `None` although `verdict[1]` came from
`SERVER_PAIR_NOT_IMPROVED`; a per-batch reason would make that visible.

## 5. Caveats

- Fail-closed choices that cost phone coverage: any member's failed phone window/control is a host verdict for that
  batch composition for the rest of the layout generation; a measured `SERVER_PAIR_NOT_IMPROVED` or an owner
  elimination at a batch size is final for that size; `maximum_probe_attempts_per_context` exhausted shared probes at
  a batch size are final (the campaign sets 4).
- The owner's like-for-like comparison costs about one host window pair per new batch composition per layout
  generation plus two transitions.
- A phone verdict from batch k runs at a new batch size j before j is decided (bounded by the shared budget); that is
  the optimistic choice, and the measured pair at j decides.
- Energy per window is the whole host's energy over the window divided by the slot's own tokens (the existing
  convention), so batch-2 windows are only comparable with batch-2 windows, which is what the verdicts enforce.
