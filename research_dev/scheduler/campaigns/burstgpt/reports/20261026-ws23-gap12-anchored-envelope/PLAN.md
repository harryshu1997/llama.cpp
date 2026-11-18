# WS23 gap 12: anchored helper envelope (Pixel-led route) — implementation plan

Context (2026-10-26): on the second desktop (RTX 5070 Ti) the OnePlus hits the strict thermal gate ~6 min into every
derived fleet arm; Qwen requests arriving afterwards get `PHONE_HELPER_UNAVAILABLE` (device_set empty) although the
Pixel holds Qwen FFN layers 1-23. `elastic_phones.helper_independence` (WS23) selects a Pixel-led composite, but the
desktop ticket never gets a helper envelope (gap 12, parked 2026-10-05). Plan produced by a read-only planning pass.

## 1. Call chain and the two death points
1. `submit_automated_request` -> `_unified/automated_requests_ops/selection.py:413-500` selects; then always
   `controller._desktop_with_async_phone_helper(...)` (`selection.py:502-512`, `replan_selection.py:301-311`).
2. `automated_selection_ops/helpers.py:35-162 _desktop_with_async_phone_helper`: `adaptive_decode_policies`
   (`_internal/adaptive_decode_planning.py:345-544`) returns the led `operator_split` envelope (its
   `anchored_resident_parameters`, `_internal/route_generation/costing_parameters.py:43-81`, carry the resident
   superset). Dormant desktop parameters from `automated_selection_ops/dormant.py:122-171` (`phone_device_id=pixel`,
   `ffn_transport=adb-tcp`, `static_helper_anchor=1`).
3. **Death point A** `helpers.py:133-137`: `layout = planning_phone_layout()` (HTP target-or-ready layout) is None or the
   OP15's Gemma layout -> `_async_helper_layout_matches` (`attachment.py:28-45`) fails -> `_defer_unready_phone_selection`
   (`helpers.py:165-192`) -> desktop baseline with no `helper_envelope`. Even if reached, `_build_helper_envelope`
   (`helper_envelopes_ops/policies.py:25-53`) needs `layout.generation/geometry` and `RuntimeHelperExecutionEnvelope`
   (`_internal/plan_contracts/execution.py:615-720`) requires non-empty `phone_shards` (:655), generation >= 1 (:672),
   geometry match (:677); the led plan has none.
4. Commit (`automated_requests_ops/commit.py:750-771`): `_dispatched_helper_envelope_binding` (:387-429) needs
   `phone_layout()`; `_compact_helper_opportunities` (`attachment.py:217-294`) returns () without
   `phone_shard_set_geometry_sha256`.
5. **Death point B** `_unified/adaptive_decode_control.py:384-522 start_adaptive_decode`: `_request_helper_envelope`
   -> None; `_ready_request_helper` (`late_attachment.py:337-416`) needs `phone_layout(generation)`;
   `_anchored_helpers_available` (:592-600) -> `_adaptive_policies_from_ticket` (:268-381) -> no opportunity ->
   `helper_available=False` (`adaptive_decode_ops/admission.py:94-113`); every window baseline.
6. Downstream needing the envelope: completion proof `adapters/http_backend.py:2749-2795`
   (`runtime_attached_helper_envelope`, `_unified/runtime_requests.py:912-1025`); `phone_calls_by_session` from
   `helper_plan.execution_contract.phone_shards` (`adapters/llama_server_ops/proofs.py:1094-1124`); energy
   `http_backend.py:1109-1159 _record_phone_window` reads `static_helper_anchor`/`phone_device_id` from the desktop
   command's top-level parameters where they never are (`adapters/ticket.py:470-476`) -> would charge the OP15.

## 2. Minimal design: static-anchor content convention on the existing envelope type
Predicate: `helper_plan.adapter_parameters.get(STATIC_HELPER_ANCHOR_PARAMETER) == 1` (already on led plans).
Existing pieces: `anchored_resident_parameters` (resident superset + `ffn_resident_geometry_sha256` = layout identity),
`_anchored_helper_residency` (`adapters/snapshot.py:1077-1121`), `anchored_policy_devices/anchored_devices_are_members`
(`adaptive_decode_control.py:77-98`, readiness = TCP worker alive and not quarantined).
Missing, in order:
- **Layout identity + synthetic shard** at route generation (`costing_parameters.py:518-522`; `envelopes.py:601/714`):
  `phone_shard_set_geometry_sha256 = anchored["ffn_resident_geometry_sha256"]`, `phone_session_count=1`, one
  `RuntimePhoneShard` (`plan_contracts/phone.py:14-43`): session_id = declaration row session id (PIXEL10PRO0),
  endpoint `adb-tcp://<serial>/<session_id>`, layer_mask = row mask, maximum_columns = feed_forward_length,
  resident_bytes, resident_geometry_sha256 = anchored geometry, operator_plan_sha256 = row shard sha, generation 1.
  At route generation so `plan_sha256` stays stable (`adaptive_decode_control.py:338-349`).
- **Envelope builder** `_build_anchored_helper_envelope` (`helper_envelopes_ops/policies.py`): constant
  `ANCHORED_HELPER_LAYOUT_GENERATION` (never a real generation), geometry from the plan parameter, no
  `_authorize_phone_helper_plan`; `_bind_anchored_phone_helper` in `automated_selection_ops/attachment.py` mirroring
  `_bind_async_phone_helper` :139-167.
- **Selection** `helpers.py:133-149`: anchored branch before the `layout is None` test; readiness =
  `anchored_devices_are_members`; preparation authorization with the layout state treated as READY.
- **Commit/binding** `commit.py:387-429`: anchored `RequestHelperEnvelopeBinding` (`phone_session_ids=(session,)`,
  identities empty) without `phone_layout`; `model_placement_ops/requests.py:246-260` and `attach_request_helper`
  skip `phone_layout()`/`_session_identities_are_ready` for the anchored generation.
- **Readiness at start/attach** `late_attachment.py:337-416` anchored branch (membership check);
  `helper_preparation_ops/attachment.py:18` tolerates `layout is None` for anchored; tighten
  `_anchored_helpers_available` (:582-587) to require an anchored envelope or late context.
- **Late producer** `_materialize_late_request_helper` (`late_attachment.py:60-212`) anchored branch from the stored
  `HelperOpportunity`.
- **Window accounting + energy** `_record_phone_window` reads anchor/device via `_phone_ffn_command(command)`
  (`llama_server_contracts.py:93-106`) with dormant fallback; anchored `device_layer_masks=((pixel, mask),)` in
  `adaptive_decode_planning.py:37-48/158` so quarantine elimination and `record_helper_phone_window` see the Pixel.

## 3. Alternative rejected: union route with the OP15 quarantined
Union envelope/shards come from OP15 HTP sessions holding Qwen (cold while Gemma is resident; transition forbidden
under THERMAL_LIMIT); readiness keys on OP15 session identities; the Pixel-primary catalog has no union family
(`primary_phone_stores_model=False`). The led envelope is strictly smaller.

## 4. Steps, tests, risks, hardware
Steps: (1) synthetic shard + geometry params; (2) anchored builder/binder + predicate in `plan_contracts/co_helpers.py`;
(3) selection branch; (4) commit binding + requests tolerance; (5) readiness at start/attach + tighten availability;
(6) late producer; (7) energy/window masks; (8) tests; (9) hardware.
Tests (extend `tests/test_ws23_helper_independence.py`, harness `tests/two_phone_harness.py`, `_submit` from
`test_two_phone_gaps.py:96`): led candidate carries one shard with the declaration session id + geometry; OP15-hot
energy-aware submit -> desktop ticket with anchored `helper_envelope`, `READY_DESKTOP_WITH_ASYNC_PHONE_HELPER`,
`interpret_runtime_ticket`/`validate_physical_execution_command` pass, `S41_SERVER_FFN_TRANSPORT=tcp`;
`start_adaptive_decode` -> `helper_available=True`, non-baseline candidates on the Pixel only; quarantine before start
-> unavailable, readmit -> late producer attaches; `_record_phone_window` charges only the helper;
`runtime_attached_helper_envelope` resolves. Suites: test_ws23_helper_independence, test_two_phone_gaps,
test_two_phone_helpers, test_adaptive_layout_identity.
Risks: energy double counting (Pixel tracker + rig `_co_helper_activity`, as today; never charge the deferred OP15);
stale windows (fix 11 reverted; Pixel windows on Qwen make the race reachable — monitor); readiness races (liveness
cadence vs `_sync_deferred_helpers` only on candidate generation); plan-sha coupling (shard must be added at route
generation).
Hardware: one derived arm on the second desktop with a helper-independence template. Expect
`quarantined_devices=={op15-phone: HELPER_DEFERRED}` after the THERMAL_DEFERRAL row, Qwen tickets with
`static_helper_anchor=1` envelopes, proofs with endpoint `adb-tcp://5A040DLCH004ES/PIXEL10PRO0` layers 1-23, no
`PHONE_HELPER_UNAVAILABLE` for Qwen, generality_gap.py showing Qwen tokens on phones with Pixel kJ up / OP15 flat.
