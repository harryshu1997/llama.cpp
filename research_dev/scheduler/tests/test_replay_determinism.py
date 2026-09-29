#!/usr/bin/env python3

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
import sys
import unittest

from research_dev.scheduler import (
    HeterogeneousRuntimeSnapshot,
    Request,
    RuntimeCapabilityCatalog,
    RuntimeTransitionReceipt,
    UnifiedScheduler,
)
from research_dev.scheduler._internal.model_manifest import ModelManifest
from research_dev.scheduler._internal.runtime_capabilities import (
    PhoneSessionResidencyObservation,
)
from research_dev.scheduler._internal.types import canonical_json


DATA_PATH = (
    Path(__file__).resolve().parent
    / "data/replay/s42_saved_runs_v1.json.gz.b64"
)
FIXTURE_JSON_SHA256 = (
    "sha256:873066db3c7af982919ae857626dee5f44e7e12ae72fc5e7b380139a0c2d21f1"
)
GOLDEN_SHA256_BY_CASE = {
    "session_cow_gate_v3": (
        "sha256:5d52e8673fc60f974ca167964b3ad579e59abea28feb3d66357cf39c1ca7caf4"
    ),
    "sparse_locality24_v8": (
        "sha256:241924463ac27ead0c2d7bcb5da214e85097049c91d5800848e0b29effd2b917"
    ),
}
GOLDEN_CASE_ID_BY_CASE = {
    "session_cow_gate_v3": "v3",
    "sparse_locality24_v8": "v8",
}
VOLATILE_EVENT_KEYS = frozenset({
    "arrival_us",
    "busy_until_us",
    "captured_at_us",
    "deadline_us",
    "event_index",
    "event_sha256",
    "finish_us",
    "finished_us",
    "start_us",
    "started_us",
    "valid_until_us",
})
LIFECYCLE_REQUEST_INDICES_BY_CASE = {
    "session_cow_gate_v3": (36, 44),
    "sparse_locality24_v8": (48, 49, 52, 57),
}


class ReplayDeterminismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        encoded = base64.b64decode(DATA_PATH.read_bytes())
        fixture_bytes = gzip.decompress(encoded)
        actual_sha256 = "sha256:" + hashlib.sha256(fixture_bytes).hexdigest()
        if actual_sha256 != FIXTURE_JSON_SHA256:
            raise AssertionError(
                f"replay fixture hash {actual_sha256} != {FIXTURE_JSON_SHA256}"
            )
        cls.fixture = json.loads(fixture_bytes)
        if (
            cls.fixture["schema"]
            != "research-scheduler-replay-determinism-fixture-v1"
        ):
            raise AssertionError("unexpected replay fixture schema")

    def _new_scheduler(self, case: Mapping[str, object]) -> UnifiedScheduler:
        common = self.fixture["common"]
        scheduler = UnifiedScheduler.for_runtime_discovery(
            "enforce",
            maximum_phone_sessions=case["maximum_phone_sessions"],
        )
        scheduler.register_runtime_capabilities(
            RuntimeCapabilityCatalog.from_json(case["runtime_catalog"])
        )
        for value in common["model_manifests"]:
            scheduler.register_model_manifest(ModelManifest.from_json(value))
        source_catalog = RuntimeCapabilityCatalog.from_json(
            common["observation_source_catalog"]
        )
        scheduler.load_automated_observations(
            common["automated_observations"],
            source_catalog=source_catalog,
        )
        scheduler.load_adaptive_decode_observations(
            common["adaptive_decode_observations"],
            source_catalog=source_catalog,
        )
        return scheduler

    def _recognize_saved_ready_layout(
        self,
        scheduler: UnifiedScheduler,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> bool:
        result = scheduler.replay_observed_phone_layout(snapshot)
        return result["status"] == "READY"

    @classmethod
    def _canonical_event(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {
                key: cls._canonical_event(item)
                for key, item in sorted(value.items())
                if key not in VOLATILE_EVENT_KEYS
                and not key.endswith((
                    "_at_us",
                    "_time_us",
                    "_timestamp_us",
                    "_until_us",
                ))
            }
        if isinstance(value, (list, tuple)):
            return [cls._canonical_event(item) for item in value]
        return value

    @staticmethod
    def _request(row: Mapping[str, object]) -> Request:
        return Request(
            request_id=row["request_id"],
            workload_id="physical:" + row["model_id"],
            arrival_us=row["replay_arrival_us"],
            deadline_us=(
                row["replay_arrival_us"] + row["source_slo_us"]
            ),
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            quality_requirement="semantic",
        )

    def _submit(
        self,
        scheduler: UnifiedScheduler,
        case: Mapping[str, object],
        row: Mapping[str, object],
    ) -> tuple[Request, object]:
        snapshot = HeterogeneousRuntimeSnapshot.from_json(row["snapshot"])
        request = self._request(row)
        ticket = scheduler.submit_automated_request(
            request,
            row["model_id"],
            snapshot,
            observed_at_us=snapshot.captured_at_us,
            selection_mode=case["selection_mode"],
        )
        return request, ticket

    def _bootstrap_ready_layout(
        self,
        scheduler: UnifiedScheduler,
        case: Mapping[str, object],
        rows_by_index: Mapping[int, Mapping[str, object]],
        bootstrap: Mapping[str, object],
    ) -> tuple[
        Mapping[str, object],
        HeterogeneousRuntimeSnapshot,
        object,
        Mapping[str, object],
    ]:
        proposal_row = rows_by_index[
            bootstrap["proposal_request_index"]
        ]
        request, ticket = self._submit(scheduler, case, proposal_row)
        plan = ticket.execution_plan
        helper = None if plan is None else plan.helper_envelope
        self.assertIsNotNone(helper)
        proof_row = rows_by_index[bootstrap["proof_snapshot_index"]]
        proof_snapshot = HeterogeneousRuntimeSnapshot.from_json(
            proof_row["snapshot"]
        )
        observed = scheduler.replay_observed_phone_layout(proof_snapshot)
        if observed["status"] == "NOT_OBSERVED":
            proposed = observed.get("layout")
            self.assertIsInstance(proposed, Mapping, observed)
            self.assertEqual(proposed["state"], "PROPOSED", observed)
            phone_device_id = (
                helper.helper_plan.execution_contract.phone_device_id
            )
            self.assertIsNotNone(phone_device_id)
            proof_snapshot = self._phone_layout_snapshot(
                proof_snapshot,
                proposed["layout"],
                at_us=proof_snapshot.captured_at_us,
                label=(
                    "replay-current-policy-bootstrap:"
                    + case["case_id"]
                ),
                device_id=phone_device_id,
                executor_id=helper.helper_binding.executor_id,
            )
            observed = scheduler.replay_observed_phone_layout(
                proof_snapshot
            )
        self.assertEqual(
            observed["status"], "READY", observed
        )
        scheduler.cancel_runtime_request(
            request.request_id,
            proof_snapshot.captured_at_us,
            "DECISION_REPLAY_BOOTSTRAP",
        )
        return observed["layout"], proof_snapshot, helper, observed

    @staticmethod
    def _retime_snapshot(
        snapshot: HeterogeneousRuntimeSnapshot,
        at_us: int,
        label: str,
    ) -> HeterogeneousRuntimeSnapshot:
        valid_until_us = at_us + 300_000_000
        return replace(
            snapshot,
            snapshot_id=label,
            captured_at_us=at_us,
            valid_until_us=valid_until_us,
            memory=replace(
                snapshot.memory,
                snapshot_id=label + ":memory",
                captured_at_us=at_us,
                valid_until_us=valid_until_us,
            ),
        )

    @classmethod
    def _phone_layout_snapshot(
        cls,
        snapshot: HeterogeneousRuntimeSnapshot,
        layout: Mapping[str, object],
        *,
        at_us: int,
        label: str,
        device_id: str,
        executor_id: str,
    ) -> HeterogeneousRuntimeSnapshot:
        snapshot = cls._retime_snapshot(snapshot, at_us, label)
        snapshot = replace(
            snapshot,
            executors={
                key: replace(
                    value,
                    healthy=True,
                    ready=True,
                    temperature_millic=30_000,
                    battery_ppm=900_000,
                    free_slots=max(1, value.free_slots),
                    busy_until_us=0,
                    thermal_qualified=True,
                )
                for key, value in snapshot.executors.items()
            },
            links={
                key: replace(value, ready=True, busy_until_us=0)
                for key, value in snapshot.links.items()
            },
            phone_session_residency=tuple(
                PhoneSessionResidencyObservation(
                    session_id=shard["session_id"],
                    device_id=device_id,
                    executor_id=executor_id,
                    endpoint=shard["endpoint"],
                    artifact_sha256=shard["artifact_sha256"],
                    resident_geometry_sha256=(
                        shard["resident_geometry_sha256"]
                    ),
                    operator_plan_sha256=(
                        shard["operator_plan_sha256"]
                    ),
                    session_generation=layout[
                        "session_generation_by_id"
                    ][shard["session_id"]],
                    resident_bytes=shard["resident_bytes"],
                )
                for shard in layout["shards"]
            ),
        )
        return snapshot

    @staticmethod
    def _completed_preparation_receipts(
        request_id: str,
        helper: object,
        preparation: Mapping[str, object],
    ) -> tuple[RuntimeTransitionReceipt, ...]:
        participants = {
            row.device_id: row
            for row in helper.helper_binding.participants
        }
        return tuple(
            RuntimeTransitionReceipt(
                ticket_id=preparation["preparation_ticket_id"],
                request_id=request_id,
                artifact_sha256=helper.artifact_sha256,
                operator_plan_sha256=helper.operator_plan_sha256,
                transition_id=transition.transition_id,
                executor_id=participants[
                    transition.device_id
                ].executor_id,
                endpoint=participants[transition.device_id].endpoint,
                device_id=transition.device_id,
                source_state=transition.source_state,
                target_state=transition.target_state,
                resource_ids=transition.resource_ids,
                resource_slots=transition.resource_slots,
                started_us=preparation["started_at_us"],
                finished_us=preparation["ready_at_us"],
                status="COMPLETED",
                evicted_artifact_sha256s=tuple(sorted({
                    row.artifact_sha256
                    for row in transition.evictions
                })),
            )
            for transition in helper.preparation_transitions
        )

    def _generation_two_preparation(
        self,
        case: Mapping[str, object],
    ) -> tuple[
        UnifiedScheduler,
        str,
        object,
        Mapping[str, object],
        tuple[RuntimeTransitionReceipt, ...],
        HeterogeneousRuntimeSnapshot,
        int,
        int,
    ]:
        rows_by_index = {
            row["combined_request_index"]: row
            for row in case["requests"]
        }
        scheduler = self._new_scheduler(case)
        source_state, proof_snapshot, source_helper, source_view = (
            self._bootstrap_ready_layout(
                scheduler,
                case,
                rows_by_index,
                {
                    "proof_snapshot_index": 36,
                    "proposal_request_index": 34,
                },
            )
        )
        phone_event_start = len(scheduler.phone_residency_events())
        helper_event_start = len(source_view["request_helper_events"])
        active_terminal_us: dict[str, int] = {}
        submitted = []
        for request_index in LIFECYCLE_REQUEST_INDICES_BY_CASE[
            case["case_id"]
        ]:
            row = rows_by_index[request_index]
            if case["case_id"] == "sparse_locality24_v8":
                source = self._retime_snapshot(
                    proof_snapshot,
                    row["snapshot"]["captured_at_us"],
                    "replay-lifecycle-source:" + str(request_index),
                )
                replay_row = {**row, "snapshot": source.to_json()}
            else:
                source = HeterogeneousRuntimeSnapshot.from_json(
                    row["snapshot"]
                )
                self._recognize_saved_ready_layout(scheduler, source)
                for request_id, terminal_us in sorted(
                    tuple(active_terminal_us.items()),
                    key=lambda item: (item[1], item[0]),
                ):
                    if terminal_us > source.captured_at_us:
                        continue
                    scheduler.cancel_runtime_request(
                        request_id,
                        terminal_us,
                        "DECISION_REPLAY_LIFECYCLE_TERMINAL",
                    )
                    del active_terminal_us[request_id]
                replay_row = row
            request, ticket = self._submit(
                scheduler, case, replay_row
            )
            submitted.append((request, ticket, row, source))
            active_terminal_us[request.request_id] = (
                row["terminal_actual_end_us"]
            )

        request, ticket, row, source = submitted[-1]
        source_layout = source_state["layout"]
        phone_device_id = (
            source_helper.helper_plan.execution_contract.phone_device_id
        )
        self.assertIsNotNone(phone_device_id)
        observed = None
        for confirmation_index in range(3):
            source_snapshot = self._phone_layout_snapshot(
                proof_snapshot,
                source_layout,
                at_us=source.captured_at_us,
                label="replay-lifecycle-generation-2-source",
                device_id=phone_device_id,
                executor_id=source_helper.helper_binding.executor_id,
            )
            observed = scheduler.replay_observed_phone_layout(
                source_snapshot
            )
            if observed["status"] != "NOT_REQUIRED":
                break
            ready_after_us = max(
                (
                    value["minimum_resident_until_us"]
                    for value in scheduler.phone_residency_session_states()
                    if value["state"] == "READY"
                ),
                default=source.captured_at_us,
            )
            confirmation_at_us = max(
                source.captured_at_us + 1,
                ready_after_us + 1,
            )
            source = self._retime_snapshot(
                source,
                confirmation_at_us,
                (
                    "replay-lifecycle-selection-confirmation:"
                    + str(confirmation_index + 1)
                ),
            )
            replay_row = {
                **row,
                "request_id": (
                    row["request_id"]
                    + ":selection-confirmation:"
                    + str(confirmation_index + 1)
                ),
                "replay_arrival_us": confirmation_at_us,
                "snapshot": source.to_json(),
            }
            request, ticket = self._submit(
                scheduler, case, replay_row
            )
            submitted.append((request, ticket, replay_row, source))
        self.assertIsNotNone(observed)
        self.assertEqual(observed["status"], "NOT_OBSERVED", observed)
        target_state = observed["layout"]
        self.assertEqual(target_state["generation"], 2, target_state)
        self.assertEqual(target_state["state"], "PROPOSED", target_state)
        changed_artifacts = {
            shard["artifact_sha256"]
            for shard in target_state["layout"]["shards"]
            if shard["session_id"]
                in target_state["layout"]["changed_session_ids"]
        }
        owner = next(
            (request_row, ticket_row)
            for request_row, ticket_row, _row, _snapshot
                in reversed(submitted)
            if ticket_row.model.artifact_sha256 in changed_artifacts
        )
        request, ticket = owner
        helper = scheduler.runtime_request_helper_preparation_envelope(
            request.request_id,
            expected_ticket_id=ticket.ticket_id,
            observed_at_us=source_snapshot.captured_at_us,
            snapshot=source_snapshot,
        )
        self.assertIsNotNone(helper)
        self.assertEqual(helper.phone_layout_generation, 2)
        preparation = scheduler.begin_request_helper_preparation(
            request.request_id,
            observed_at_us=source_snapshot.captured_at_us,
            snapshot=source_snapshot,
            expected_phone_layout_generation=(
                helper.phone_layout_generation
            ),
            expected_phone_layout_geometry_sha256=(
                helper.phone_layout_geometry_sha256
            ),
            expected_operator_plan_sha256=helper.operator_plan_sha256,
        )
        self.assertEqual(preparation["status"], "OWNER", preparation)
        receipts = self._completed_preparation_receipts(
            request.request_id, helper, preparation
        )
        self.assertTrue(receipts)
        target_snapshot = self._phone_layout_snapshot(
            proof_snapshot,
            target_state["layout"],
            at_us=preparation["ready_at_us"],
            label="replay-lifecycle-generation-2-target",
            device_id=phone_device_id,
            executor_id=helper.helper_binding.executor_id,
        )
        return (
            scheduler,
            request.request_id,
            helper,
            preparation,
            receipts,
            target_snapshot,
            phone_event_start,
            helper_event_start,
        )

    def _preparation_lifecycle_replay(
        self,
        case: Mapping[str, object],
        *,
        inject_post_load_failure: bool,
    ) -> Mapping[str, object]:
        (
            scheduler,
            request_id,
            helper,
            preparation,
            receipts,
            target_snapshot,
            phone_event_start,
            helper_event_start,
        ) = self._generation_two_preparation(case)
        if inject_post_load_failure:
            target_layout = next(
                event["layout"]
                for event in reversed(scheduler.phone_residency_events())
                if event["kind"] == "PREPARING"
                and event["generation"] == 2
            )
            restored_generations = {
                session_id: (
                    target_layout["session_generation_by_id"][session_id]
                    + 1
                )
                for session_id in target_layout["changed_session_ids"]
            }
            scheduler.fail_request_helper_preparation(
                request_id,
                preparation["preparation_ticket_id"],
                failed_at_us=preparation["ready_at_us"],
                reason="DECISION_REPLAY_POST_LOAD_FAILURE",
                restored_session_generations=restored_generations,
            )
            outcome = "FAILED"
            final_view = scheduler.replay_observed_phone_layout(
                target_snapshot
            )
            request_events = tuple(
                event
                for event in final_view["request_helper_events"]
                if event["request_id"] == request_id
            )
            self.assertEqual(
                request_events[-1]["kind"],
                "PREPARATION_FAILED",
            )
        else:
            result = scheduler.complete_request_helper_preparation(
                request_id,
                preparation["preparation_ticket_id"],
                receipts,
                snapshot=target_snapshot,
            )
            self.assertEqual(result["state"], "READY", result)
            outcome = "READY"
            final_view = scheduler.replay_observed_phone_layout(
                target_snapshot
            )
        return self._canonical_event({
            "outcome": outcome,
            "phone_layout_events": scheduler.phone_residency_events()[
                phone_event_start:
            ],
            "request_helper_events": final_view["request_helper_events"][
                helper_event_start:
            ],
            "simulated_transition_receipts": [
                row.to_json() for row in receipts
            ],
        })

    def _replay(self, case: Mapping[str, object]) -> bytes:
        rows_by_index = {
            row["combined_request_index"]: row
            for row in case["requests"]
        }
        source_indices = [
            row["combined_request_index"] for row in case["requests"]
        ]
        selected_routes = []
        phone_layout_events = []
        request_helper_events = []

        for segment in case["segments"]:
            scheduler = self._new_scheduler(case)
            bootstrap = segment.get("bootstrap_ready_layout")
            if bootstrap is not None:
                self._bootstrap_ready_layout(
                    scheduler, case, rows_by_index, bootstrap
                )
            event_start = len(scheduler.phone_residency_events())
            active_terminal_us: dict[str, int] = {}
            segment_routes = []
            last_snapshot = None
            for request_index in segment["request_indices"]:
                row = rows_by_index[request_index]
                snapshot = HeterogeneousRuntimeSnapshot.from_json(
                    row["snapshot"]
                )
                last_snapshot = snapshot
                self._recognize_saved_ready_layout(
                    scheduler, snapshot
                )
                for request_id, terminal_us in sorted(
                    tuple(active_terminal_us.items()),
                    key=lambda item: (item[1], item[0]),
                ):
                    if terminal_us > snapshot.captured_at_us:
                        continue
                    scheduler.cancel_runtime_request(
                        request_id,
                        terminal_us,
                        "DECISION_REPLAY_TERMINAL",
                    )
                    del active_terminal_us[request_id]

                request, ticket = self._submit(scheduler, case, row)
                route = {
                    "combined_request_index": request_index,
                    "request_id": request.request_id,
                    "route_id": ticket.decision.route_id,
                }
                selected_routes.append(route)
                segment_routes.append(route)
                active_terminal_us[request.request_id] = (
                    row["terminal_actual_end_us"]
                )

            self.assertIsNotNone(last_snapshot)
            replay_view = scheduler.replay_observed_phone_layout(
                last_snapshot
            )
            for event in scheduler.phone_residency_events()[event_start:]:
                phone_layout_events.append(self._canonical_event({
                    **dict(event),
                    "segment_id": segment["segment_id"],
                }))
            for route in segment_routes:
                request_helper_events.extend(
                    self._canonical_event(event)
                    for event in replay_view["request_helper_events"]
                    if event["request_id"] == route["request_id"]
                )

        self.assertEqual(
            [row["combined_request_index"] for row in selected_routes],
            source_indices,
        )
        preparation_lifecycle = {
            "completed": self._preparation_lifecycle_replay(
                case, inject_post_load_failure=False
            ),
            "post_load_failure": self._preparation_lifecycle_replay(
                case, inject_post_load_failure=True
            ),
        }
        return canonical_json({
            "case_id": GOLDEN_CASE_ID_BY_CASE[case["case_id"]],
            "phone_layout_events": phone_layout_events,
            "preparation_lifecycle": preparation_lifecycle,
            "request_helper_events": request_helper_events,
            "selected_route_ids": selected_routes,
        }).encode("ascii")

    def test_preparation_publication_survives_owner_ticket_replacement(self) -> None:
        case = next(
            row for row in self.fixture["cases"]
            if row["case_id"] == "session_cow_gate_v3"
        )
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                (
                    scheduler, request_id, helper, preparation, receipts,
                    snapshot, _phone_start, _helper_start,
                ) = self._generation_two_preparation(case)
                original = scheduler.runtime_ticket(request_id)
                replacement = replace(
                    original,
                    ticket_id=f"{request_id}:attempt:{original.attempt_index + 1}",
                    previous_ticket_id=original.ticket_id,
                    attempt_index=original.attempt_index + 1,
                )
                # Re-admission replaces the nonterminal ticket, not the load.
                scheduler._runtime_controller._tickets[request_id] = replacement
                with self.assertRaisesRegex(ValueError, "runtime ticket is unknown"):
                    scheduler.runtime_ticket_by_id(original.ticket_id)
                if cancelled:
                    scheduler.cancel_runtime_request(
                        request_id, snapshot.captured_at_us - 1,
                        "preparation-owner-cancelled",
                    )
                ready_before = scheduler._model_placement_controller.ready_phone_layout()
                target = scheduler._model_placement_controller.preparing_phone_layout()
                selected = target.layout.changed_session_ids
                self.assertEqual(len(selected), 1)
                invalid = replace(
                    snapshot,
                    phone_session_residency=tuple(
                        replace(row, session_generation=row.session_generation + 1)
                        if row.session_id in selected else row
                        for row in snapshot.phone_session_residency
                    ),
                )
                with self.assertRaisesRegex(ValueError, "not physically ready"):
                    scheduler.complete_request_helper_preparation(
                        request_id, preparation["preparation_ticket_id"],
                        receipts, snapshot=invalid,
                    )
                self.assertEqual(
                    scheduler._model_placement_controller.ready_phone_layout(),
                    ready_before,
                )
                result = scheduler.complete_request_helper_preparation(
                    request_id, preparation["preparation_ticket_id"],
                    receipts, snapshot=snapshot,
                )
                self.assertEqual(result["state"], "READY")
                ready = scheduler._model_placement_controller.ready_phone_layout()
                self.assertEqual(ready.layout, target.layout)
                self.assertEqual(result["request_ticket_id"], original.ticket_id)
                self.assertEqual(scheduler.runtime_ticket(request_id).ticket_id,
                                 replacement.ticket_id)
                before_generations = ready_before.layout.session_generation_by_id
                self.assertTrue(all(
                    generation == before_generations[session_id]
                    for session_id, generation in ready.layout.session_generation_by_id.items()
                    if session_id not in selected
                ))

    def test_saved_decision_replays_match_golden_hashes(self) -> None:
        cases = {
            case["case_id"]: case for case in self.fixture["cases"]
        }
        self.assertEqual(set(cases), set(GOLDEN_SHA256_BY_CASE))
        for case_id, expected_sha256 in sorted(
            GOLDEN_SHA256_BY_CASE.items()
        ):
            with self.subTest(case_id=case_id):
                first = self._replay(cases[case_id])
                second = self._replay(cases[case_id])
                self.assertEqual(first, second)
                self.assertEqual(
                    "sha256:" + hashlib.sha256(first).hexdigest(),
                    expected_sha256,
                )


def _write_golden_candidates(output_directory: Path) -> None:
    ReplayDeterminismTests.setUpClass()
    test = ReplayDeterminismTests(
        "test_saved_decision_replays_match_golden_hashes"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for case in sorted(
        test.fixture["cases"], key=lambda row: row["case_id"]
    ):
        payload = test._replay(case)
        case_id = case["case_id"]
        hashes[case_id] = (
            "sha256:" + hashlib.sha256(payload).hexdigest()
        )
        decoded = json.loads(payload)
        (output_directory / (case_id + ".json")).write_text(
            json.dumps(
                decoded,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="ascii",
        )
    (output_directory / "SHA256.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(hashes, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--regenerate-goldens":
        _write_golden_candidates(Path(sys.argv[2]))
    else:
        unittest.main()
