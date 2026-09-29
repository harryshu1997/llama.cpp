# Elastic phones — phones drop or join at runtime, the scheduler re-places (SPEC, 2026-09-25)

User direction (2026-09-25 18:20 UTC): "implement and test that if some phones will drop or join, the scheduler
should be able to adjust the placement accordingly." This is P1 (S1 + S3) of `research_dev/ROBUSTNESS_PLAN.md`.

## 0. Ground truth (two read-only code maps, 2026-09-25)

- A helper RPC failure makes `llama_decode` return 2 → `tools/server/server-context.cpp:4120-4136` errors every
  processing slot and `throw`s → **the llama-server process dies**. A second death path: a runtime-policy apply
  failure at batch time (`server-context.cpp:4049-4066`).
- The scheduler **never reaps a dead server**: `adapters/llama_server.py` only polls `process.poll()` during load
  and in `begin_execution` (raises `PhysicalAdapterError("… endpoint is not active")`); `_live_executors`
  (`adapters/heterogeneous_rig.py:504`) keeps the dead entry, `_snapshot_once` keeps emitting a hot residency
  sample, route generation therefore plans no reload and every route on it is `EXECUTOR_NOT_READY`
  (`route_generation/feasibility.py:606-655`). A planned transition WOULD relaunch it
  (`heterogeneous_rig_ops/transitions.py:94-107` checks `poll()`).
- The **request fallback path exists**: `adapters/runtime.py::_recover` →
  `_unified/automated_requests_ops/failure.py::fail_automated_request` → `_prepare_automated_failure_fallback`
  → `_commit_automated_attempt(event_kind="FALLBACK")` → the adapter re-executes the same payload
  (`tests/test_physical_adapter.py::test_adapter_owns_failure_replan_and_logs_the_fallback`). Blockers:
  (a) mid-stream failures are `execution_started=True, retry_safe=False` → `fallback_allowed` false
  (`_internal/runtime_execution.py:63-65`); (b) nothing names the failed device (`coherence.py:539-541`);
  (c) `LlamaCppCompletionPayload` rejects an existing `stream_path`, the stream is opened `"xb"`
  (`adapters/http_backend.py:122,277`); (d) the SSE error body is discarded (`http_backend.py:288-291`);
  (e) every physical failure quarantines the ROUTE for the rest of the run
  (`_internal/runtime_controller_ops/completion.py:20-45`); (f) adaptive recovery targets the paired desktop
  baseline on the same composite executor (`automated_selection_ops/objectives.py:390-428`) → if that executor is
  dead → `FALLBACK_UNAVAILABLE`.
- **Device-set drops** (`_internal/adaptive_decode_ops/coherence.py`): `server_policy_failed` (536-572) drops the
  failed set + supersets at batch 0 (= all compositions) but only in ONE server-policy group
  (model × placement × layout × geometry); groups are never removed; `device_drops`/`device_attempts`
  (`adaptive_decode_state.py:40-42`) are append-only; **no clear/readmit API**.
- **Quarantine primitives**: `RuntimeController._quarantined_routes/_quarantined_resources`
  (`_internal/runtime_controller.py:162-163, 373-386`, applied in `runtime_controller_ops/admission.py:21-40`,
  `automated_selection_ops/adaptive.py:134-153`, `objectives.py:349-420`); no clear API. Phone resources are NOT in
  a desktop-parent ticket's binding (late-attached helpers) → never pass them as `failed_resource_ids`
  (`completion.py:144-152`).
- **Telemetry quarantine is viable**: `feasibility.py:528-533` → `PHONE_TELEMETRY_UNAVAILABLE` from
  `snapshot.telemetry_unavailable_reason`; the rig fills `telemetry_observations` only for the primary phone
  (`heterogeneous_rig.py:1994-2000`); a co-helper row with `validity="UNAVAILABLE"` (shape
  `PhoneRuntimeObservation.to_json()`, validator `capability_contracts/snapshots.py:448-466`) would keep new
  requests off that device.
- **Join**: the server already DEFERS co-helpers when `RUNTIME_CONTROL=1` (`tools/server/server.cpp:535-536`,
  log `connection=deferred`) and connects lazily in `apply_policy` (871-910) — the server starts even if a helper
  is unreachable; no C++ change for join. Scheduler side: `adapters/co_helper_lifecycle.py::start_trace` (64-87)
  raises on any helper failure (no try), `heterogeneous_rig.py::begin_trace` (808-823) propagates → trace abort;
  `adapters/phone_tcp_session.py:272` sets `_process` before the readiness wait (a failed start leaves
  `active=True`); no liveness method; `campaigns/burstgpt/preflight.py:795-845` + `helper_phone_evidence.py:40-50`
  + `launch.py:669-672` require every declared co-helper present; a cold co-helper rejects the whole split
  (`tests/test_two_phone_gaps.py::test_cold_co_helper_is_rejected_not_prepared`).
- **Periodic hook**: `BackgroundRuntimeMonitor` (`_internal/runtime_queue.py:2575-2690`, one daemon thread per
  probe, 0.5 s refresh, probes fixed at construction in `heterogeneous_rig.py:485-500`; existing
  `helper-runtime:<device>` probe → `probe_android_phone_runtime`).
- **Phone session cleanup blocks**: OP15 `phone_session_ops/completion.py::abort/finish` waits up to ~195 s
  synchronously on the transition thread; `phone_tcp_session.stop` sends SIGTERM, never SIGKILL.

## 1. Scope

Opt-in campaign field `elastic_phones` (default absent → behaviour identical to today; every golden test must stay
byte-identical with the field absent):

```json
"elastic_phones": {"drop_recovery": true, "join": true, "join_probe_interval_s": 10,
                   "readmission_cooldown_s": 60, "max_readmissions_per_device": 3}
```

Non-goals (later): S2 server-side step retry (no process death), per-helper column fractions, layer migration.

## 2. Contracts (shared by both implementation slices)

- **Failure kinds** (structured, on `PhysicalBackendFailure` / `RuntimeExecutionFailure` and the ticket):
  `phase="helper_lost"` with `failed_device_id: str` (device id from the rig: `op15-phone`, `pixel10pro-phone`),
  and `phase="server_exited"` with `executor_id`. Both: `retry_safe=True`; `helper_lost` has
  `execution_started=True` but `fallback_allowed` is True for it (recovery route is desktop-only and the request
  is re-executed from its prompt, so it is exact); `server_exited` raised from `begin_execution` has
  `execution_started=False`.
- **Device id ↔ helper label**: `S41SERVERFFNERROR helper=<label>` labels come from the co-helper declaration
  (`RuntimeCoHelperPhone`); a single-helper server names the primary phone. Provide one helper
  `device_id_for_helper_label(rig, label)`.
- **Scheduler API (facade `research_dev/scheduler/__init__.py` / `_unified` scheduler)**:
  `quarantine_device(device_id: str, *, reason: str, at_us: int) -> None` and
  `readmit_device(device_id: str, *, at_us: int, identity_sha256: str) -> None`. Idempotent. Both record
  `device_membership_events` rows `{kind: DEVICE_QUARANTINED|DEVICE_READMITTED|DEVICE_ABSENT_AT_START,
  device_id, reason, at_us, identity_sha256?}` exported in RESULT.
- **Events in RESULT**: `SERVER_EXITED{executor_id, returncode, at_us}`, `REQUEST_RECOVERED{request, failed_ticket,
  new_ticket, failure_kind, device_id, tokens_discarded, penalty_us}` (runner rows already carry
  `attempt_ticket_ids`/`recoveries`; extend, don't replace), `device_membership_events`.
- **Stream files**: on re-execution the partial `streams/request-NNN.raw` is renamed to
  `request-NNN.raw.attempt<k>`; the canonical path always holds the terminal attempt (the offline identity check
  reads the canonical path; `validate_attempt_chains` already accepts FALLBACK chains).

## 3. Slice 1 — DROP: recover the request, reload a dead server (owner: agent 1)

Files owned: `adapters/llama_server.py`, `adapters/heterogeneous_rig_ops/observations.py`,
`adapters/heterogeneous_rig_ops/lifecycle.py`, `adapters/runtime.py`, `adapters/http_backend.py`,
`_internal/runtime_execution.py`, `_internal/runtime_controller_ops/completion.py`,
`_unified/automated_requests_ops/failure.py`, `_unified/automated_selection_ops/objectives.py`,
`campaigns/burstgpt/runner.py` (recovery rows only), `adapters/contracts.py` (failure fields),
tests: `tests/test_physical_adapter.py`, `tests/test_physical_residency.py`, `tests/test_automated_runtime_routes.py`,
new `tests/test_elastic_drop_recovery.py`.

1. **Server liveness.** `ManagedLlamaServer.exit_code()`; `begin_execution` raises
   `PhysicalBackendFailure(phase="server_exited", retry_safe=True, execution_started=False)` when the process is
   gone. `_executor_samples` (observations.py) detects `poll() is not None` and calls a new
   `lifecycle.reap_exited_executor(rig, executor_id)` = `_stop_executor(…, terminate_phone_session=False)` +
   `_dormant_forget_server` + `SERVER_EXITED` event → residency goes cold → route generation plans a load →
   the existing `_begin_transition_execution` relaunches. Only under `elastic_phones.drop_recovery`.
2. **Classification.** In `http_backend.py` keep the SSE error message; after a failed stream, read the server
   stderr captured since `marker.stderr_index` (`adapters/llama_server.py` `stderr_lines`); if it contains
   `S41SERVERFFNERROR`, `LIBUSB_ERROR`, `Compute aborted` with a helper context, or the phone session raised a
   transport-kind `PhysicalAdapterError`, raise `PhysicalBackendFailure(phase="helper_lost",
   failed_device_id=…)`. Otherwise today's behaviour.
3. **Fallback allowance.** `_internal/runtime_execution.py`: `fallback_allowed = retry_safe and (not
   execution_started or phase in {"helper_lost"})`. `completion.py::_route_violation`: no route quarantine for
   `helper_lost` / `server_exited` (the DEVICE is quarantined by slice 2; the route is fine).
4. **Recovery through a reload.** `failure.py::_prepare_automated_failure_fallback` +
   `objectives.py::_select_adaptive_desktop_recovery`: when the paired desktop executor is dead/reaped, the recovery
   candidate may require a load transition (accept it; the load time is the penalty). Recovery candidates never
   include a quarantined device (consult slice 2's `quarantined_devices` via the runtime controller; if the API is
   not merged yet, filter by `failed_device_id`).
5. **Quarantine call.** In `adapters/runtime.py::_execute_once` on `helper_lost`: call
   `scheduler.quarantine_device(device_id, reason="HELPER_LOST", at_us=…)` (guard with `hasattr` until slice 2
   lands), then `_recover`. Move the partial stream aside before re-execution (`_runtime_progress_payload`).
6. **Fan-out.** Co-tenant requests on the dead server fail with the same stderr context → same classification →
   same FALLBACK path; one `SERVER_EXITED` event, N `REQUEST_RECOVERED` rows.
7. **Tests** (recorded, no hardware): fake backend that streams k tokens then raises `helper_lost` →
   FALLBACK on a desktop route, canonical stream = recovered tokens, `.attempt1` holds the partial, coordinator
   does not abort, events present; `server_exited` from `begin_execution` → reaped → next attempt plans a load;
   fail-fast still raises for a non-helper exception; with `elastic_phones` absent every existing test passes
   unchanged.

## 4. Slice 2 — QUARANTINE / JOIN: device membership, co-helper lifecycle (owner: agent 2)

Files owned: `_internal/adaptive_decode_ops/coherence.py`, `_internal/adaptive_decode_state.py`,
`_internal/adaptive_decode.py`, `_unified/adaptive_decode_control.py`, `_internal/runtime_controller.py`
(quarantine API only), `adapters/heterogeneous_rig.py`, `adapters/co_helper_lifecycle.py`,
`adapters/phone_tcp_session.py`, `_internal/runtime_queue.py` (probe registration), `_internal/route_generation/*`
and `_internal/adaptive_decode_planning.py` (cold co-helper → primary-only sets), `campaigns/burstgpt/preflight.py`,
`campaigns/burstgpt/helper_phone_evidence.py`, `campaigns/burstgpt/launch.py`, `campaigns/burstgpt/arguments.py`,
`configuration/campaign.py`, `campaigns/burstgpt/prepare_trace_inputs_v2.py` (`--elastic-phones-json`),
`research_dev/scheduler/__init__.py` (facade), new `campaigns/burstgpt/tools/inject_helper_loss.py`,
tests: `tests/test_per_device_policies.py`, `tests/test_two_phone_gaps.py`, `tests/test_two_phone_helpers.py`,
`tests/test_campaign_inputs.py`, new `tests/test_elastic_join.py`.

1. **Config.** `elastic_phones` field (validated, serialized, CLI `--elastic-phones-json`, runner → coordinator →
   rig/adapters). Absent → today's behaviour everywhere.
2. **Quarantine in coherence.** `quarantine_device(controller, device_id, reason)`: in EVERY group of
   `controller._server_policies`, `_drop_device_set(group, 0, devices, reason)` for each set containing the device,
   remove verdicts containing it, `_mirror_device_set_drops`, mark `eliminated_policy_reasons` in the affected
   sessions. `readmit_device(controller, device_id)`: remove `device_drops`/`device_attempts` rows whose set
   contains the device, clear its `DEVICE_SET_FAILED`/quarantine eliminations, so `_eligible_device_sets` offers
   it again and the normal probe (primary first) decides on evidence. Wrappers in `_internal/adaptive_decode.py`,
   `_unified/adaptive_decode_control.py`, facade.
3. **Quarantine in the runtime controller.** `quarantine_device` adds the device's compute + transport resource
   ids (rig topology / co-helper declaration) to `_quarantined_resources` (action `device_lost`);
   `readmit_device` removes them and any routes quarantined with that action.
4. **Quarantine for new requests via telemetry.** `heterogeneous_rig.py` `_snapshot_once`: emit
   `telemetry_observations[device]` for co-helpers (today only the primary); while a device is quarantined or
   absent, `validity="UNAVAILABLE"` → `PHONE_TELEMETRY_UNAVAILABLE`; primary-only device sets must still be
   generated when a co-helper is cold/absent (change `test_cold_co_helper_is_rejected_not_prepared` semantics
   ONLY under the flag).
5. **Co-helper lifecycle.** `phone_tcp_session.py`: reset `_process` on a failed start; `alive()` (worker pid
   present, adb forward listed, optional TCP connect); `terminate_for_fault_injection()` = SIGTERM to the worker
   pids, logged, only when `elastic_phones` is set and the caller passes `authorized=True`.
   `co_helper_lifecycle.py`: per-helper try in `start_trace` → `absent` set + `DEVICE_ABSENT_AT_START`;
   `join(device)` (preflight + start); `stop(device)` on a background thread; `end_trace` continues past failures.
   `heterogeneous_rig.py::begin_trace` tolerates absence under the flag.
6. **Liveness + join watcher.** New `BackgroundRuntimeMonitor` probe `helper-membership:<device>`: for a started
   device, `alive()` false → `quarantine_device(HELPER_LOST)` + background stop; for an absent/quarantined device,
   every `join_probe_interval_s`: adb device present, identity pins verify (`helper_phone_evidence.live_preflight`
   equivalent: USB identity, kernel release, worker/shard sha) → `join(device)` → `readmit_device` (respect
   `readmission_cooldown_s`, `max_readmissions_per_device`; identity mismatch → stay quarantined, event with
   reason). The OP15 (FunctionFS primary) is covered for loss by slice 1's classification; its re-join = same
   watcher with `adb get-state` + identity pins (session relaunch through the existing transition path).
7. **Preflight/launch.** Under the flag an absent co-helper is recorded, not fatal
   (`preflight.py::_helper_phone_checks`, `helper_phone_evidence.py`, `launch.py:669`); evidence stays mandatory
   for identity verification at join time.
8. **Hardware fault tool.** `tools/inject_helper_loss.py --device pixel10pro-phone --when request=NNN|t=SECONDS
   --signal TERM --run-dir DIR` : waits for the condition in the run's decision log, SIGTERMs the worker via adb
   (root), writes `FAULT_INJECTED.json` with timestamps. Never SIGKILL. Used only with the user's authorization.
9. **Tests**: quarantine drops in all groups + verdict removal; readmit restores eligibility and probe order; runtime
   controller filters quarantined resources; absent at start → primary-only sets, run proceeds; join → readmitted →
   probe admits on fake evidence; cooldown/max/identity-mismatch; flag absent → unchanged.

## 5. Rules for both agents

- Work ONLY inside `<scratchpad>/iso-p1/research_dev/scheduler` (the isolated copy). Never touch the main tree,
  the deploy, the rig, git, or files owned by the other slice (coordinate through the contracts in §2; if you need
  a hook in the other slice's file, write the call behind `hasattr`/`getattr` and note it in your report).
- Match the surrounding code style; keep everything fail-closed; every new behaviour behind `elastic_phones`.
- Run targeted tests continuously (`python3 -m unittest research_dev.scheduler.tests.test_x` from the iso root) and
  the full suite at the end (`python3 research_dev/scheduler/tests/run_all.py` from the iso root; baseline =
  every module passes). Report: files changed, tests added, suite result, open issues.

## 6. Acceptance

Recorded: T1 drop mid-stream → recovered + quarantined + run continues; T2 server exit → reload → recovered;
T3 absent at start → primary-only; join → readmitted → admitted on evidence; T4 cooldown/max/identity; T5 flag
absent → byte-identical. Hardware (eval_v2, two-phone arm): G1 Pixel SIGTERM mid-run → run PASS, outputs
identical/near-tie, saving still ≫ baseline, `REQUEST_RECOVERED` + `DEVICE_QUARANTINED`; G1b worker restarted →
`DEVICE_READMITTED`, Pixel calls resume; G2 OP15 loss (user unplug / `adb reboot`) later.

## 7. Status (2026-09-26 09:30 UTC)

Implemented in the working checkout and deployed (`elastic_phones` opt-in; nothing committed).
2026-09-27 audit: 463 scheduler Python files match checkout, stage and deploy. Fresh full runner FAIL
on one 10.475 ms timing check against 10 ms; its isolated rerun PASS. Recovery and thermal-policy
tests pass. See [results audit](../20260927-results-audit/README.md) for the current verification scope.
Hardware gates on `longtail_eval_v2`, two-phone per-device arm:

| gate | result |
| --- | --- |
| G1 (idle kill, first tool version) | PASS; quarantine + identity-verified re-join; loss detected late (probe: 409 s) |
| G1b (mid-call kill) | FAIL — control-path (`StalePhysicalSlotError`) not classified → fix 2 |
| G1c (kill at t=520) | PASS; loss at the first liveness check (client_exited), re-join +65 s |
| G1d (mid-call, `active` trigger) | FAIL — server survives a helper loss with a poisoned client; never retired → fix 3 (retire + reload) |
| G1e | FAIL — recovery planned at failed_at against a later snapshot ("system snapshot is stale") → fix 3b |
| G1f | FAIL at the recovered attempt's start: stale adaptive registration → fix 3c |
| **G1g** | **PASS**: `SERVER_EXITED{HELPER_LOST}` → `REQUEST_RECOVERED{001, 95 tokens discarded, penalty 60.6 s}` → `DEVICE_QUARANTINED` +5 s → `DEVICE_READMITTED` +65 s; 112.9 kJ (−50.6 % vs baseline; undisturbed −58.7 %); 14/14 served |
| g9 (S2a `helper_loss_recovery: mask_out`, rebuilt server) | PASS mechanically: `HELPER_MASKED_OUT` → `REQUEST_RECOVERED{001, same_server_mask_out, 60 tokens, penalty 40 s}`, no `SERVER_EXITED` — but the dispatcher stopped the freed server 4 s later for the waiting Gemma switch (`HELPER_MASK_ENDED cause=SERVER_STOPPED`): arrival ORDER, the recovery re-entered the queue as a new admission → dispatcher fix |
| **g11 (mask_out + dispatcher fix)** | **PASS**: `HELPER_MASKED_OUT` 505.5 s → `REQUEST_RECOVERED` ×2 (003, 004; 71/72 tokens; penalty 47 s) with `dispatch_policy: RECOVERY_RETAINED_QUEUE_PLACE`, NO server stop/exit → `HELPER_REATTACHED mode=live_reconnect` 570.3 s + `DEVICE_READMITTED`; 121.4 kJ (−46.9 %); 14/14 served |

Also new: `THERMAL_DEFERRAL`/`THERMAL_DEFERRAL_CLEARED` rows (`thermal_deferral_events` in RESULT, elastic only) —
g11 shows the OP15 thermally excluded for 529 s (Android thermal status ≠ 0, rule `thermal_qualified = status == 0`);
an opt-in per-device `maximum_thermal_status` (default 0 = byte-identical) is merged but not enabled.

Corrections to §0: the llama-server does NOT die on a helper loss (it catches the decode error and keeps serving
with the helper client latched failed); `request=NNN` in the fault tool fires at request completion (streams are
written at request end) — use `active=SECONDS`. Open: G2 (OP15 loss, needs the user), G3 re-plug of the OP15,
repeats for confidence intervals (three two-phone runs: 94.4 / 120.6 / 112.2 kJ). The scheduler's thermal
gate reduces OP15 coverage; provisioning defects also affected the second run. The still-unrun thermal
intervention is needed to establish how much of the remaining energy gap it explains. S2a recovery
mechanics pass on g11; strict output identity remains FAIL, 12/14. Reported recovery `penalty_us` is
discarded attempt time, not recovery downtime or a matched latency delta.
