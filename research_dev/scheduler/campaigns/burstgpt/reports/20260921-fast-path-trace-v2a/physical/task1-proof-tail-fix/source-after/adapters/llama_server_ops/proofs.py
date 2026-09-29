"""Managed llama-server: execution call verification and execution proof construction."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Mapping, Sequence
from ..._internal.adaptive_decode_contracts import AdaptiveDecodeGroupedObservation
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_plan import RuntimeHelperExecutionEnvelope
from ..._internal.types import canonical_sha256
from ..contracts import PhysicalAdapterError
from ..ticket import PhysicalExecutionCommand, static_decode_policy
from ..llama_server_contracts import (
    _phone_ffn_command as _phone_ffn_command,
    _adaptive_phone_ffn as _adaptive_phone_ffn,
    _launch_contract_supports_execution as _launch_contract_supports_execution,
    PhoneFfnExecutionContract as PhoneFfnExecutionContract,
    LlamaServerFfnCall as LlamaServerFfnCall,
    LlamaServerPhoneSessionProof as LlamaServerPhoneSessionProof,
    LlamaServerExecutionMarker as LlamaServerExecutionMarker,
    LlamaServerExecutionProof as LlamaServerExecutionProof,
    _TAIL_RELEASE_REASONS as _TAIL_RELEASE_REASONS,
    parse_llama_server_ffn_call as parse_llama_server_ffn_call,
    phone_ffn_execution_contract as phone_ffn_execution_contract,
    phone_ffn_resident_contract as phone_ffn_resident_contract,
    llama_server_launch_contract as llama_server_launch_contract,
)


class ManagedServerProofMixin:
    """Managed llama-server: execution call verification and execution proof construction."""

    @staticmethod
    def _validate_execution_finish(
        marker: LlamaServerExecutionMarker,
        command: PhysicalExecutionCommand,
        output_tokens: int,
    ) -> None:
        if (
            not isinstance(marker, LlamaServerExecutionMarker)
            or not isinstance(command, PhysicalExecutionCommand)
            or type(output_tokens) is not int
            or output_tokens <= 0
            or (
                marker.ticket_id,
                marker.artifact_sha256,
                marker.operator_plan_sha256,
                marker.executor_id,
            ) != (
                command.ticket_id,
                command.artifact_sha256,
                command.operator_plan_sha256,
                command.executor_id,
            )
        ):
            raise PhysicalAdapterError(
                "llama-server execution marker differs from the ticket"
            )

    def _execution_phone_contracts(
        self,
        marker: LlamaServerExecutionMarker,
        command: PhysicalExecutionCommand,
        manifest: ModelManifest,
        helper_envelopes: Sequence[RuntimeHelperExecutionEnvelope],
    ):
        phone_source = _phone_ffn_command(command)
        remote = command.execution_contract.remote_resident_ffn
        remote_identity = None if remote is None else canonical_sha256(remote.to_json())
        if marker.remote_resident_identity_sha256 != remote_identity:
            raise PhysicalAdapterError(
                "remote-resident owner identity changed during execution"
            )
        helper_rows = tuple(helper_envelopes)
        if any(
            not isinstance(row, RuntimeHelperExecutionEnvelope)
            for row in helper_rows
        ):
            raise PhysicalAdapterError(
                "llama-server helper history is invalid"
            )
        current_helper = command.helper_envelope
        if (
            current_helper is not None
            and current_helper.operator_plan_sha256 not in {
                row.operator_plan_sha256 for row in helper_rows
            }
        ):
            helper_rows += (current_helper,)
        helper_by_plan = {
            row.operator_plan_sha256: row for row in helper_rows
        }
        if len(helper_by_plan) != len(helper_rows) or any(
            row.artifact_sha256 != command.artifact_sha256
            or row.desktop_parent_route_id != command.route_id
            or row.desktop_placement_sha256
                != command.operator_plan.get(
                    "desktop_placement_sha256"
                )
            for row in helper_rows
        ):
            raise PhysicalAdapterError(
                "llama-server helper history differs from the base ticket"
            )
        contract_by_plan = {
            row.operator_plan_sha256: phone_ffn_resident_contract(
                replace(command, helper_envelope=row), manifest
            )
            for row in helper_rows
        }
        expected_contract = None
        if (
            self.launch_contract.phone_device_id is not None
            and (
                phone_source.execution_contract.execution_mode != "desktop"
                or phone_source.execution_contract.remote_resident_ffn is not None
            )
        ):
            expected_contract = (
                phone_ffn_resident_contract(command, manifest)
                if _adaptive_phone_ffn(command)
                else phone_ffn_execution_contract(command, manifest)
            )
            contract_by_plan.setdefault(
                phone_source.operator_plan_sha256, expected_contract
            )
        if (
            not _launch_contract_supports_execution(
                self.launch_contract,
                llama_server_launch_contract(command, manifest),
            )
            or marker.phone_contract != expected_contract
        ):
            raise PhysicalAdapterError(
                "llama-server phone contract changed during execution"
            )
        return (
            phone_source,
            helper_rows,
            current_helper,
            contract_by_plan,
            expected_contract,
        )

    @staticmethod
    def _static_execution_ack(
        command: PhysicalExecutionCommand,
        output_tokens: int,
        static_control_ack: Mapping[str, object] | None,
    ):
        policy = static_decode_policy(command)
        if policy is None:
            if static_control_ack is not None:
                raise PhysicalAdapterError(
                    "static FFN runtime acknowledgement is unexpected"
                )
            return None, None, None
        if not isinstance(static_control_ack, Mapping):
            raise PhysicalAdapterError(
                "static FFN runtime acknowledgement is absent"
            )
        applied_token = static_control_ack.get("applied_token_index")
        slot_id = static_control_ack.get("slot_id")
        if (
            static_control_ack.get("request_id") != command.request_id
            or type(slot_id) is not int
            or slot_id < 0
            or static_control_ack.get("plan_generation") != 1
            or static_control_ack.get("policy_hash") != policy.policy_hash
            or type(applied_token) is not int
            or not 0 <= applied_token <= output_tokens
        ):
            raise PhysicalAdapterError(
                "static FFN runtime acknowledgement differs"
            )
        return policy, applied_token, slot_id

    def _execution_stderr_calls(
        self,
        marker: LlamaServerExecutionMarker,
        expected_scope_ids: set[str],
    ) -> tuple[tuple[str, ...], tuple[LlamaServerFfnCall, ...]]:
        lines = tuple(self.stderr_lines[marker.stderr_index:])
        parsed = tuple(
            call for call in (
                parse_llama_server_ffn_call(line) for line in lines
            )
            if call is not None
        )
        scoped = tuple(call for call in parsed if call.contexts)
        if scoped and len(scoped) != len(parsed):
            raise PhysicalAdapterError(
                "llama-server mixed scoped and legacy FFN proof"
            )
        calls = (
            parsed
            if not scoped
            else tuple(
                call for call in scoped
                if any(
                    row.scheduler_request_id in expected_scope_ids
                    for row in call.contexts
                )
            )
        )
        return lines, calls

    @staticmethod
    def _logical_call_rows(
        call: LlamaServerFfnCall,
        expected_scope_ids: set[str],
    ) -> int:
        if not call.contexts:
            return call.tokens
        return sum(
            row.rows for row in call.contexts
            if row.scheduler_request_id in expected_scope_ids
        )

    @staticmethod
    def _stale_adaptive_baseline_tail(
        observation: AdaptiveDecodeGroupedObservation,
    ) -> tuple[tuple[object, ...], bool, bool]:
        stale_windows = tuple(
            row for row in observation.windows
            if row.failure_reason in {
                "stale_slot_stats_discarded", "released_slot_baseline_tail",
            }
        )
        stale_tail = (
            len(stale_windows) == 1
            and stale_windows[0] is observation.windows[-1]
            and not stale_windows[0].measurement_eligible
            and stale_windows[0].policy.baseline
            and stale_windows[0].applied_ack is None
            and stale_windows[0].completed_phone_calls == 0
            and stale_windows[0].completed_phone_input_rows == 0
            and observation.final_policy.baseline
            and observation.unmeasured_tail_tokens > 0
            and observation.unmeasured_tail_reason
                == "stale_slot_stats_discarded"
            and all(row.policy.baseline for row in observation.windows)
        )
        last_ack_index, last_ack_window = next(
            ((index, row) for index, row in reversed(tuple(enumerate(observation.windows)))
             if row.applied_ack is not None),
            (0, None),
        )
        zero_ack = (
            last_ack_window.applied_ack
            if last_ack_window is not None
            and last_ack_window.policy == observation.final_policy
            and last_ack_window.policy.baseline else None
        )
        released_tail = (
            len(stale_windows) == 1
            and stale_windows[0] is observation.windows[-1]
            and stale_windows[0].failure_reason == "released_slot_baseline_tail"
            and not stale_windows[0].measurement_eligible
            and stale_windows[0].policy.baseline
            and stale_windows[0].completed_phone_calls == 0
            and stale_windows[0].completed_phone_input_rows == 0
            and "physical:terminal-release-confirmed" in stale_windows[0].evidence_ids
            and observation.final_policy == stale_windows[0].policy
            and observation.unmeasured_tail_tokens == 0
            and (zero_ack is not None or all(
                row.policy.baseline for row in observation.windows))
            and all(row.policy == observation.final_policy
                    for row in observation.windows[last_ack_index:])
            and all(
                row.applied_ack is None or row.policy.baseline or (
                    zero_ack is not None
                    and row.applied_ack.plan_generation
                        < zero_ack.plan_generation
                ) for row in observation.windows[:-1]
            )
        )
        return stale_windows, stale_tail or released_tail, stale_tail

    @staticmethod
    def _validate_adaptive_observation(
        observation: AdaptiveDecodeGroupedObservation,
        command: PhysicalExecutionCommand,
        cohort,
        cohort_leader,
        desktop_placement,
        stale_windows,
        stale_baseline_tail: bool,
    ) -> None:
        if (
            observation.request_id != (
                command.request_id if cohort is None else cohort_leader
            )
            or (
                cohort is None
                and observation.ticket_id != command.ticket_id
            )
            or observation.model_artifact_sha256
                != command.artifact_sha256
            or observation.desktop_placement_sha256
                != desktop_placement
            or observation.terminal_status != "COMPLETED"
            or any(
                not row.output_valid
                or (
                    row.failure_reason is not None
                    and row not in stale_windows
                )
                for row in observation.windows
            )
            or (bool(stale_windows) and not stale_baseline_tail)
        ):
            raise PhysicalAdapterError(
                "adaptive observation differs from the execution ticket"
                + ": " + json.dumps({
                    "request_id": observation.request_id,
                    "expected_request_id": command.request_id if cohort is None else cohort_leader,
                    "ticket_id": observation.ticket_id,
                    "expected_ticket_id": command.ticket_id,
                    "artifact_sha256": observation.model_artifact_sha256,
                    "expected_artifact_sha256": command.artifact_sha256,
                    "desktop_placement_sha256": observation.desktop_placement_sha256,
                    "expected_desktop_placement_sha256": desktop_placement,
                    "terminal_status": observation.terminal_status,
                    "invalid_windows": [
                        {"window_index": row.window_index,
                         "failure_reason": row.failure_reason,
                         "output_valid": row.output_valid}
                        for row in observation.windows
                        if not row.output_valid or row.failure_reason is not None
                    ],
                    "stale_baseline_tail": stale_baseline_tail,
                }, sort_keys=True, separators=(",", ":"))
            )

    @staticmethod
    def _adaptive_window_call_rows(
        observation: AdaptiveDecodeGroupedObservation,
        contract_by_plan,
        desktop_placement,
        cohort,
    ) -> list[tuple[int, int]]:
        rows: list[tuple[int, int]] = []
        for window in observation.windows:
            policy = window.policy
            contract = (
                None if policy.baseline
                else contract_by_plan.get(policy.operator_plan_sha256)
            )
            if (
                policy.desktop_placement_sha256 != desktop_placement
                or (
                    policy.baseline
                    and (
                        policy.layer_indices
                        or policy.columns != 0
                        or policy.layer_mask != 0
                    )
                )
                or (
                    not policy.baseline
                    and (
                        contract is None
                        or not set(policy.layer_indices).issubset(
                            set(contract.layer_indices)
                        )
                        or policy.columns > contract.columns
                    )
                )
            ):
                raise PhysicalAdapterError(
                    "adaptive window exceeds the resident operator plan"
                )
            if policy.baseline:
                continue
            if window.completed_phone_calls is not None:
                layer_count = len(policy.layer_indices)
                completed_rows = window.completed_phone_input_rows
                if (
                    completed_rows is None
                    or layer_count == 0
                    or window.completed_phone_calls % layer_count
                    or completed_rows % layer_count
                ):
                    raise PhysicalAdapterError(
                        "adaptive completed phone rows are invalid"
                    )
                window_rows = completed_rows // layer_count
            elif cohort is None:
                window_rows = window.token_count
            else:
                layer_count = len(policy.layer_indices)
                if (
                    layer_count == 0
                    or window.transfer_subrequests % layer_count
                ):
                    raise PhysicalAdapterError(
                        "adaptive cohort transfer rows are invalid"
                    )
                window_rows = (
                    window.transfer_subrequests // layer_count
                ) * window.active_batch
            for _ in range(window_rows):
                rows.extend(
                    (layer, policy.columns)
                    for layer in policy.layer_indices
                )
        return rows

    @staticmethod
    def _adaptive_tail_call_rows(
        observation: AdaptiveDecodeGroupedObservation,
        rows: list[tuple[int, int]],
        contract_by_plan,
        desktop_placement,
        cohort,
    ) -> None:
        tail_policy = observation.final_policy
        if (
            not observation.unmeasured_tail_tokens
            or tail_policy.baseline
        ):
            return
        tail_contract = contract_by_plan.get(
            tail_policy.operator_plan_sha256
        )
        if (
            cohort is not None
            and observation.windows[-1].active_batch != 1
        ):
            raise PhysicalAdapterError(
                "adaptive cohort tail requires singleton ownership"
            )
        if (
            tail_contract is None
            or tail_policy.desktop_placement_sha256 != desktop_placement
            or not set(tail_policy.layer_indices).issubset(
                set(tail_contract.layer_indices)
            )
            or tail_policy.columns > tail_contract.columns
        ):
            raise PhysicalAdapterError(
                "adaptive tail exceeds the resident operator plan"
            )
        tail_tokens = observation.unmeasured_tail_tokens
        if (observation.unmeasured_tail_reason in _TAIL_RELEASE_REASONS
                and observation.final_policy_ack is None):
            accounted_ahead = 1
            ack_index = next((index for index in range(len(observation.windows) - 1, -1, -1)
                              if observation.windows[index].applied_ack is not None), None)
            segment = () if ack_index is None else observation.windows[ack_index:]
            if (cohort is None and segment
                    and all(window.policy == tail_policy
                            and window.completed_phone_calls is not None
                            and window.completed_phone_input_rows is not None
                            for window in segment)):
                # Statistics can include forwards beyond the requested token boundary.
                measured_rows = sum(window.completed_phone_input_rows for window in segment)
                measured_tokens = (segment[-1].token_end
                                   - segment[0].applied_ack.applied_token_index)
                accounted_ahead = measured_rows // len(tail_policy.layer_indices) - measured_tokens
                if not 0 <= accounted_ahead <= tail_tokens:
                    raise PhysicalAdapterError("adaptive tail counters differ from acknowledged tokens")
            tail_tokens -= accounted_ahead
        for _ in range(max(0, tail_tokens)):
            rows.extend(
                (layer, tail_policy.columns)
                for layer in tail_policy.layer_indices
            )

    def _adaptive_expected_calls(
        self,
        observation: AdaptiveDecodeGroupedObservation | None,
        command: PhysicalExecutionCommand,
        phone_source,
        contract_by_plan,
        cohort,
        cohort_leader,
        output_tokens: int,
    ) -> tuple[tuple[tuple[int, int], ...] | None, bool]:
        if observation is None:
            return None, False
        desktop_placement = phone_source.operator_plan.get(
            "desktop_placement_sha256"
        )
        stale_windows, stale_tail, ignore_unattributed_calls = (
            self._stale_adaptive_baseline_tail(observation)
        )
        self._validate_adaptive_observation(
            observation,
            command,
            cohort,
            cohort_leader,
            desktop_placement,
            stale_windows,
            stale_tail,
        )
        rows = self._adaptive_window_call_rows(
            observation, contract_by_plan, desktop_placement, cohort
        )
        if cohort is None and (
            observation.windows[-1].token_end
            + observation.unmeasured_tail_tokens
            != output_tokens
        ):
            raise PhysicalAdapterError(
                "adaptive window coverage differs from output"
            )
        self._adaptive_tail_call_rows(
            observation,
            rows,
            contract_by_plan,
            desktop_placement,
            cohort,
        )
        return tuple(rows), ignore_unattributed_calls

    def _verify_single_request_adaptive_calls(
        self,
        calls: tuple[LlamaServerFfnCall, ...],
        expected_calls: tuple[tuple[int, int], ...],
        observation: AdaptiveDecodeGroupedObservation,
        command: PhysicalExecutionCommand,
        manifest: ModelManifest,
        contract_by_plan,
        expected_scope_ids: set[str],
    ) -> tuple[bool, dict[int, str]]:
        expected_rows: dict[tuple[int, int], int] = {}
        for row in expected_calls:
            expected_rows[row] = expected_rows.get(row, 0) + 1
        actual_rows: dict[tuple[int, int], int] = {}
        policies = {
            (
                row.applied_ack.slot_id,
                row.applied_ack.plan_generation,
            ): row.policy
            for row in observation.windows
            if row.applied_ack is not None
        }
        if observation.final_policy_ack is not None:
            ack = observation.final_policy_ack
            policies[(ack.slot_id, ack.plan_generation)] = observation.final_policy
        invalid = len({call.request_id for call in calls}) != len(calls)
        plan_by_request: dict[int, str] = {}
        for call in calls:
            rows = self._logical_call_rows(call, expected_scope_ids)
            key = (call.layer, call.columns)
            actual_rows[key] = actual_rows.get(key, 0) + rows
            own_contexts = tuple(
                row for row in call.contexts
                if row.scheduler_request_id == command.request_id
            )
            own_policy = (
                None if not own_contexts
                else policies.get((
                    own_contexts[0].server_slot_id,
                    own_contexts[0].plan_generation,
                ))
            )
            own_contract = (
                None if own_policy is None
                else contract_by_plan.get(
                    own_policy.operator_plan_sha256
                )
            )
            invalid = invalid or (
                rows <= 0
                or call.payload_bytes
                    != manifest.embedding_length * call.tokens * 2
                or (
                    bool(call.contexts)
                    and (
                        len(own_contexts) != 1
                        or own_policy is None
                        or own_policy.baseline
                        or own_contract is None
                        or call.layer not in own_contract.layer_indices
                        or call.columns > own_contract.columns
                    )
                )
            )
            if own_policy is not None and not own_policy.baseline:
                plan_by_request[call.request_id] = (
                    own_policy.operator_plan_sha256
                )
        return invalid or actual_rows != expected_rows, plan_by_request

    def _verify_cohort_adaptive_calls(
        self,
        calls: tuple[LlamaServerFfnCall, ...],
        expected_calls: tuple[tuple[int, int], ...],
        current_helper,
        manifest: ModelManifest,
        expected_contract: PhoneFfnExecutionContract,
        expected_scope_ids: set[str],
    ) -> tuple[bool, dict[int, str]]:
        expected_rows: dict[tuple[int, int], int] = {}
        for row in expected_calls:
            expected_rows[row] = expected_rows.get(row, 0) + 1
        actual_rows: dict[tuple[int, int], int] = {}
        invalid = len({call.request_id for call in calls}) != len(calls)
        plan_by_request = {}
        for call in calls:
            if call.contexts and any(
                row.scheduler_request_id not in expected_scope_ids
                for row in call.contexts
            ):
                invalid = True
            key = (call.layer, call.columns)
            actual_rows[key] = actual_rows.get(
                key, 0
            ) + self._logical_call_rows(call, expected_scope_ids)
            invalid = invalid or (
                call.tokens <= 0
                or call.tokens > expected_contract.max_tokens
                or call.payload_bytes
                    != manifest.embedding_length * call.tokens * 2
            )
            if current_helper is not None:
                plan_by_request[call.request_id] = (
                    current_helper.operator_plan_sha256
                )
        return invalid or actual_rows != expected_rows, plan_by_request

    def _verify_adaptive_execution_calls(
        self,
        calls: tuple[LlamaServerFfnCall, ...],
        expected_calls: tuple[tuple[int, int], ...],
        observation: AdaptiveDecodeGroupedObservation,
        command: PhysicalExecutionCommand,
        manifest: ModelManifest,
        contract_by_plan,
        expected_contract: PhoneFfnExecutionContract,
        expected_scope_ids: set[str],
        cohort,
        current_helper,
    ) -> tuple[dict[int, int], dict[int, str]]:
        if cohort is None:
            invalid, plan_by_request = (
                self._verify_single_request_adaptive_calls(
                    calls,
                    expected_calls,
                    observation,
                    command,
                    manifest,
                    contract_by_plan,
                    expected_scope_ids,
                )
            )
        else:
            invalid, plan_by_request = (
                self._verify_cohort_adaptive_calls(
                    calls,
                    expected_calls,
                    current_helper,
                    manifest,
                    expected_contract,
                    expected_scope_ids,
                )
            )
        if invalid:
            expected_by_shape = {}
            for row in expected_calls:
                expected_by_shape[row] = expected_by_shape.get(row, 0) + 1
            actual_by_shape = {}
            for call in calls:
                row = (call.layer, call.columns)
                actual_by_shape[row] = (
                    actual_by_shape.get(row, 0)
                    + self._logical_call_rows(call, expected_scope_ids)
                )
            windows = [
                (
                    row.token_start,
                    row.token_end,
                    row.policy.columns,
                    row.completed_phone_calls,
                    row.completed_phone_input_rows,
                )
                for row in observation.windows
            ]
            raise PhysicalAdapterError(
                "llama-server phone calls differ from adaptive windows: "
                f"expected_rows={len(expected_calls)} "
                f"actual_rows={sum(self._logical_call_rows(call, expected_scope_ids) for call in calls)} "
                f"physical_calls={len(calls)} "
                f"unique={len({call.request_id for call in calls})} "
                f"scoped={bool(calls and calls[0].contexts)} "
                f"expected_by_shape={expected_by_shape} "
                f"actual_by_shape={actual_by_shape} "
                f"windows={windows} "
                f"tail_tokens={observation.unmeasured_tail_tokens} "
                f"tail_reason={observation.unmeasured_tail_reason} "
                f"final_columns={observation.final_policy.columns}"
            )
        by_layer: dict[int, int] = {}
        for call in calls:
            by_layer[call.layer] = by_layer.get(call.layer, 0) + 1
        return by_layer, plan_by_request

    def _verify_unmeasured_cohort_calls(
        self,
        calls: tuple[LlamaServerFfnCall, ...],
        expected_contract: PhoneFfnExecutionContract,
        phone_source,
        manifest: ModelManifest,
        expected_scope_ids: set[str],
        current_helper,
    ) -> tuple[dict[int, int], dict[int, str]]:
        expected_layers = set(expected_contract.layer_indices)
        column_quantum = phone_source.adapter_parameters.get(
            "ffn_column_quantum", 1
        )
        if (
            type(column_quantum) is not int
            or column_quantum <= 0
            or len({call.request_id for call in calls}) != len(calls)
            or any(
                call.layer not in expected_layers
                or call.tokens <= 0
                or call.tokens > expected_contract.max_tokens
                or self._logical_call_rows(
                    call, expected_scope_ids
                ) <= 0
                or (
                    bool(call.contexts)
                    and any(
                        row.scheduler_request_id not in expected_scope_ids
                        for row in call.contexts
                    )
                )
                or call.columns <= 0
                or call.columns > expected_contract.columns
                or call.columns % column_quantum
                or call.payload_bytes
                    != manifest.embedding_length * call.tokens * 2
                for call in calls
            )
        ):
            raise PhysicalAdapterError(
                "llama-server cohort calls exceed the resident envelope"
            )
        by_layer = {}
        plan_by_request = {}
        for call in calls:
            by_layer[call.layer] = by_layer.get(call.layer, 0) + 1
            if current_helper is not None:
                plan_by_request[call.request_id] = (
                    current_helper.operator_plan_sha256
                )
        return by_layer, plan_by_request

    def _verify_static_execution_calls(
        self,
        calls: tuple[LlamaServerFfnCall, ...],
        expected_contract: PhoneFfnExecutionContract,
        manifest: ModelManifest,
        expected_scope_ids: set[str],
        output_tokens: int,
        static_applied_token: int | None,
    ) -> tuple[dict[int, int], dict[int, str]]:
        expected_layers = set(expected_contract.layer_indices)
        if (
            not calls
            or len({call.request_id for call in calls}) != len(calls)
            or any(
                call.layer not in expected_layers
                or call.tokens > expected_contract.max_tokens
                or self._logical_call_rows(
                    call, expected_scope_ids
                ) <= 0
                or call.columns != expected_contract.columns
                or call.payload_bytes
                    != manifest.embedding_length * call.tokens * 2
                for call in calls
            )
        ):
            raise PhysicalAdapterError(
                "llama-server phone calls differ from the operator plan"
            )
        by_layer = {}
        rows_by_layer = {}
        for call in calls:
            by_layer[call.layer] = by_layer.get(call.layer, 0) + 1
            rows_by_layer[call.layer] = (
                rows_by_layer.get(call.layer, 0)
                + self._logical_call_rows(call, expected_scope_ids)
            )
        minimum_rows = (
            output_tokens
            if static_applied_token is None
            else max(0, output_tokens - static_applied_token)
        )
        if any(
            rows_by_layer.get(layer, 0) < minimum_rows
            for layer in expected_layers
        ):
            raise PhysicalAdapterError(
                "llama-server phone call coverage is incomplete"
            )
        return by_layer, {}

    def _verify_execution_calls(
        self,
        *,
        calls: tuple[LlamaServerFfnCall, ...],
        expected_contract,
        expected_calls,
        observation,
        command,
        manifest,
        contract_by_plan,
        expected_scope_ids,
        cohort,
        adaptive,
        phone_source,
        current_helper,
        output_tokens,
        static_applied_token,
    ) -> tuple[dict[int, int], dict[int, str]]:
        if expected_contract is None:
            if calls:
                raise PhysicalAdapterError(
                    "desktop llama-server emitted phone FFN calls"
                )
            return {}, {}
        if expected_calls is not None:
            return self._verify_adaptive_execution_calls(
                calls,
                expected_calls,
                observation,
                command,
                manifest,
                contract_by_plan,
                expected_contract,
                expected_scope_ids,
                cohort,
                current_helper,
            )
        if adaptive and cohort is not None:
            return self._verify_unmeasured_cohort_calls(
                calls,
                expected_contract,
                phone_source,
                manifest,
                expected_scope_ids,
                current_helper,
            )
        return self._verify_static_execution_calls(
            calls,
            expected_contract,
            manifest,
            expected_scope_ids,
            output_tokens,
            static_applied_token,
        )

    def _execution_session_proofs(
        self,
        command: PhysicalExecutionCommand,
        phone_source,
        helper_rows,
        observation,
        calls,
        call_plan_by_request,
        manifest: ModelManifest,
        expected_scope_ids: set[str],
    ) -> tuple[LlamaServerPhoneSessionProof, ...]:
        shards_by_plan = {
            row.operator_plan_sha256:
                row.helper_plan.execution_contract.phone_shards
            for row in helper_rows
        }
        remote = phone_source.execution_contract.remote_resident_ffn
        if remote is not None:
            shards_by_plan[phone_source.operator_plan_sha256] = (
                phone_source.execution_contract.resident_phone_shards(
                    manifest.feed_forward_length
                )
            )
        if (
            not shards_by_plan
            and phone_source.execution_contract.phone_shards
        ):
            shards_by_plan[phone_source.operator_plan_sha256] = (
                phone_source.execution_contract.phone_shards
            )
        used_plans = set()
        if observation is not None:
            used_plans.update(
                row.policy.operator_plan_sha256
                for row in observation.windows
                if not row.policy.baseline
            )
            if (
                observation.unmeasured_tail_tokens
                and not observation.final_policy.baseline
            ):
                used_plans.add(
                    observation.final_policy.operator_plan_sha256
                )
        if calls and not used_plans:
            used_plans.add(phone_source.operator_plan_sha256)
        totals = {}
        for call in calls:
            if not shards_by_plan:
                break
            plan_sha256 = call_plan_by_request.get(call.request_id)
            if plan_sha256 is None:
                available = tuple(sorted(
                    used_plans.intersection(shards_by_plan)
                ))
                if len(available) != 1:
                    raise PhysicalAdapterError(
                        "llama-server call lacks an exact helper generation"
                    )
                plan_sha256 = available[0]
            owners = tuple(
                row for row in shards_by_plan.get(plan_sha256, ())
                if row.layer_mask & (1 << call.layer)
            )
            if len(owners) != 1:
                raise PhysicalAdapterError(
                    "llama-server phone call has no unique shard owner"
                )
            shard = owners[0]
            key = (
                shard.session_id,
                shard.endpoint,
                shard.artifact_sha256 or command.artifact_sha256,
                shard.resident_geometry_sha256,
                shard.operator_plan_sha256,
                shard.session_generation,
            )
            row = totals.setdefault(key, {
                "calls": 0, "layer_mask": 0,
                "payload_bytes": 0, "rows": 0,
            })
            logical_rows = self._logical_call_rows(
                call, expected_scope_ids
            )
            row["calls"] += 1
            row["layer_mask"] |= shard.layer_mask
            row["payload_bytes"] += (
                manifest.embedding_length * logical_rows * 2
            )
            row["rows"] += logical_rows
        required_sessions = {
            (
                shard.session_id,
                shard.endpoint,
                shard.artifact_sha256 or command.artifact_sha256,
                shard.resident_geometry_sha256,
                shard.operator_plan_sha256,
                shard.session_generation,
            )
            for plan_sha256 in used_plans
            for shard in shards_by_plan.get(plan_sha256, ())
        }
        if calls and any(
            totals.get(key, {}).get("calls", 0) <= 0
            or totals.get(key, {}).get("rows", 0) <= 0
            for key in required_sessions
        ):
            raise PhysicalAdapterError(
                "llama-server phone shard received no physical calls"
            )
        return tuple(
            LlamaServerPhoneSessionProof(
                session_id=session_id,
                endpoint=endpoint,
                artifact_sha256=artifact_sha256,
                resident_geometry_sha256=geometry_sha256,
                operator_plan_sha256=plan_sha256,
                session_generation=generation,
                layer_mask=row["layer_mask"],
                calls=row["calls"],
                rows=row["rows"],
                payload_bytes=row["payload_bytes"],
            )
            for (
                session_id,
                endpoint,
                artifact_sha256,
                geometry_sha256,
                plan_sha256,
                generation,
            ), row in sorted(totals.items())
        )

    def _execution_proof(
        self,
        *,
        marker,
        command,
        lines,
        calls,
        by_layer,
        session_proofs,
        observation,
        static_policy,
        static_applied_token,
        static_slot_id,
        manifest,
        expected_scope_ids,
    ) -> LlamaServerExecutionProof:
        return LlamaServerExecutionProof(
            ticket_id=command.ticket_id,
            artifact_sha256=command.artifact_sha256,
            operator_plan_sha256=command.operator_plan_sha256,
            executor_id=command.executor_id,
            phone_call_count=len(calls),
            phone_calls_by_layer=tuple(sorted(by_layer.items())),
            phone_calls_sha256=canonical_sha256([
                {
                    "columns": call.columns,
                    "layer": call.layer,
                    "payload_bytes": call.payload_bytes,
                    "request_id": call.request_id,
                    "runtime_context": [
                        {
                            "plan_generation": row.plan_generation,
                            "request_id": row.scheduler_request_id,
                            "rows": row.rows,
                            "slot_id": row.server_slot_id,
                        }
                        for row in call.contexts
                    ],
                    "tokens": call.tokens,
                }
                for call in calls
            ]),
            phone_first_request_id=(
                None if not calls else calls[0].request_id
            ),
            phone_last_request_id=(
                None if not calls else calls[-1].request_id
            ),
            phone_payload_bytes=sum(
                manifest.embedding_length
                * self._logical_call_rows(call, expected_scope_ids)
                * 2
                for call in calls
            ),
            stderr_start_index=marker.stderr_index,
            stderr_end_index=marker.stderr_index + len(lines),
            phone_calls_by_session=tuple(session_proofs),
            adaptive_grouped_observation_sha256=(
                None if observation is None
                else observation.grouped_observation_sha256
            ),
            adaptive_group_owner_request_id=(
                None if observation is None else observation.request_id
            ),
            adaptive_window_count=(
                0 if observation is None else len(observation.windows)
            ),
            static_policy_applied_token_index=static_applied_token,
            static_policy_hash=(
                None if static_policy is None
                else static_policy.policy_hash
            ),
            static_policy_plan_generation=(
                None if static_policy is None else 1
            ),
            static_policy_slot_id=static_slot_id,
        )
