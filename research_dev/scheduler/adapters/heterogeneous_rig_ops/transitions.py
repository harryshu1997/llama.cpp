"""Physical rig: transition preparation, publication, cleanup and helper rollback."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import replace
from types import MappingProxyType
from typing import Callable, Mapping
from ..._internal.model_manifest import ModelManifest
from ..android_llama_server import ANDROID_LLAMA_SERVER_ADAPTER
from ..contracts import PhysicalAdapterError
from ..http_backend import LlamaCppCompletionPayload
from ..phone_transport import phone_transport_contract
from ..residency import (
    physical_residency_parameters_match,
    physical_residency_supports_execution_plan,
)
from ..ticket import PhysicalTransitionCommand, validate_phone_session_replacement_command
from .common import (
    _LiveExecutorResidency as _LiveExecutorResidency,
    _HelperReconfiguration as _HelperReconfiguration,
    _TransitionExecutionState as _TransitionExecutionState,
)


class RigTransitionMixin:
    """Physical rig: transition preparation, publication, cleanup and helper rollback."""

    def _transition_manifest(
        self,
        command: PhysicalTransitionCommand,
        payload: object,
    ) -> ModelManifest:
        validate_phone_session_replacement_command(command)
        if not isinstance(payload, LlamaCppCompletionPayload):
            raise PhysicalAdapterError("transition payload is invalid")
        manifest = next((
            row
            for row in self.configuration.manifests.values()
            if row.artifact_sha256 == command.artifact_sha256
        ), None)
        if manifest is None:
            raise PhysicalAdapterError(
                "transition model artifact is not registered"
            )
        return manifest

    def _reject_legacy_transition(
        self,
        command: PhysicalTransitionCommand,
        manifest: ModelManifest,
        control_check: Callable[[], None],
    ) -> None:
        if hasattr(self, "_live_executors"):
            return
        with self._lock:
            executor_id = getattr(self, "_current_executor_id", None)
            legacy_manifest = getattr(self, "_current_manifest", None)
            generation = getattr(self, "_generation", -1)
        artifact = getattr(legacy_manifest, "artifact_sha256", None)
        if any(
            eviction.artifact_sha256 != artifact
            or eviction.generation != generation
            or (
                eviction.executor_id is not None
                and eviction.executor_id != executor_id
            )
            for eviction in command.transition.evictions
        ):
            raise PhysicalAdapterError(
                "transition eviction source differs from physical endpoint"
            )
        control_check()
        raise PhysicalAdapterError(
            "legacy physical transition lacks a capability registry"
        )

    def _begin_transition_execution(
        self,
        command: PhysicalTransitionCommand,
        manifest: ModelManifest,
        transition_started_ns: int,
    ) -> _TransitionExecutionState | None:
        target_executor_id = command.participant.executor_id
        helper_only = bool(getattr(command, "helper_only", False))
        resources = (
            self._phone_residency_resources(target_executor_id)
            if helper_only
            else self._residency_resources(target_executor_id)
        )
        target_devices, target_replacement, target_session = resources
        with self._lock:
            live = dict(self._live_executors)
            previous_phone = getattr(self, "_phone_residency", None)
            current = live.get(target_executor_id)
            if current is not None and (
                current.manifest == manifest
                and physical_residency_parameters_match(
                    current.parameters, command.adapter_parameters
                )
                and physical_residency_supports_execution_plan(
                    current.operator_plan, command.operator_plan
                )
                and current.server.process is not None
                and current.server.process.poll() is None
            ):
                return None
        phone = command.adapter_parameters.get("phone_device_id")
        direct_transport = (
            None
            if phone is None
            else phone_transport_contract(command.adapter_parameters)
        )
        direct_phone_reused = False
        direct_phone_reconfigurable = False
        direct_phone_partial_requested = False
        if (
            direct_transport is not None
            and direct_transport.transport == "functionfs-usb"
            and self._direct_phone_session.active
        ):
            direct_phone_reused = self._direct_phone_session.supports(
                command, manifest, direct_transport
            )
            direct_phone_reconfigurable = (
                helper_only
                and not direct_phone_reused
                and self._direct_phone_session
                    .supports_partial_reconfiguration(
                        command, manifest, direct_transport
                    )
            )
            direct_phone_partial_requested = bool(
                helper_only
                and not direct_phone_reused
                and command.transition.changed_phone_session_ids
            )
            if getattr(self, "_direct_phone_relaunch_required", None) is not None:
                # S2a mask_out: the primary was lost since its session started; relaunch, never reuse
                direct_phone_reused = direct_phone_reconfigurable = False
        ticket_id = getattr(command, "ticket_id", None)
        state = _TransitionExecutionState(
            manifest=manifest,
            target_executor_id=target_executor_id,
            helper_only=helper_only,
            target_devices=target_devices,
            target_replacement=target_replacement,
            target_session=target_session,
            live=live,
            previous_phone_residency=previous_phone,
            phone=phone,
            direct_transport=direct_transport,
            direct_phone_reused=direct_phone_reused,
            direct_phone_reconfigurable=direct_phone_reconfigurable,
            direct_phone_partial_requested=direct_phone_partial_requested,
            transition_started_ns=transition_started_ns,
            phone_activity=getattr(self, "_phone_activity", None),
            phone_activity_id=(
                None if ticket_id is None
                else "transition:" + ticket_id
            ),
        )
        self._begin_transition_phone_activity(command, state)
        return state

    def _whole_phone_device(self, command) -> str | None:
        parameters = command.adapter_parameters
        if parameters.get("execution_adapter") != ANDROID_LLAMA_SERVER_ADAPTER:
            return None
        participant = getattr(command, "participant", None)
        executor_id = command.executor_id if participant is None else participant.executor_id
        endpoint = command.endpoint if participant is None else participant.endpoint
        capability = self.configuration.catalog.executor_by_id.get(executor_id)
        phone = parameters.get("gpu_device_id")
        if (
            capability is None
            or phone != self.configuration.phone_device_id
            or capability.device_id != phone
            or capability.endpoint != endpoint
            or capability.adapter_parameters.get("execution_adapter") != ANDROID_LLAMA_SERVER_ADAPTER
            or capability.adapter_parameters.get("gpu_device_id") != phone
            or (participant is not None and participant.device_id != phone)
        ):
            raise PhysicalAdapterError("whole-phone physical device differs from the ticket")
        return phone

    def _begin_transition_phone_activity(
        self,
        command: PhysicalTransitionCommand,
        state: _TransitionExecutionState,
    ) -> None:
        contract = getattr(command, "execution_contract", None)
        phone = None if contract is None else contract.phone_device_id or self._whole_phone_device(command)
        if (
            state.phone_activity is None
            or state.phone_activity_id is None
            or contract is None
            or phone is None
            or phone not in command.transition.prepares_device_ids
            or state.direct_phone_reused
        ):
            return
        state.phone_activity.begin(
            state.phone_activity_id,
            "endpoint_preparation",
            state.transition_started_ns,
        )
        state.phone_activity_started = True

    def _start_transition_mutation(
        self,
        command: PhysicalTransitionCommand,
        state: _TransitionExecutionState,
        control_check: Callable[[], None],
    ) -> None:
        conflicts = self._transition_conflicting_executors(
            command, state
        )
        self._validate_phone_transition_observation(command, state, control_check)
        control_check()
        with self._lock:
            self._launch_attempt += 1
        state.mutation_started = True
        for executor_id in conflicts:
            self._stop_executor(
                executor_id, terminate_phone_session=False
            )
        control_check()

    def _direct_phone_failure_resources(
        self, command: PhysicalTransitionCommand, phone: str
    ) -> tuple[str, ...]:
        capability = (
            self.configuration.catalog.executor_by_device[phone]
        )
        transition_resources = set(command.transition.resource_ids)
        resources = tuple(sorted(
            (
                set(capability.execution_resource_ids)
                | {
                    resource_id
                    for resource_id in transition_resources
                    if self.configuration.catalog.resources[
                        resource_id
                    ].kind == "transport"
                }
            )
            & transition_resources
        ))
        if not resources:
            raise PhysicalAdapterError(
                "direct phone failure domain is absent"
            )
        return resources

    def _prepare_functionfs_phone(
        self,
        command: PhysicalTransitionCommand,
        state: _TransitionExecutionState,
        control_check: Callable[[], None],
    ) -> None:
        direct_phone = self._direct_phone_session
        state.direct_failure_resources = (
            self._direct_phone_failure_resources(
                command, str(state.phone)
            )
        )
        if state.direct_phone_reused:
            direct_phone.bind(
                command, state.manifest, state.direct_transport
            )
        elif state.direct_phone_reconfigurable:
            receipt = direct_phone.reconfigure(
                command, state.manifest, state.direct_transport
            )
            state.direct_phone_reconfiguration_receipt = receipt
            state.direct_phone_reconfigured = True
            with self._lock:
                self._direct_phone_receipts.append({
                    **receipt.to_json(),
                    "command_changed_session_ids": list(
                        command.transition.changed_phone_session_ids
                    ),
                    "phone_layout_generation": (
                        command.phone_layout_generation
                    ),
                    "transition_id": command.transition.transition_id,
                })
        else:
            relaunch = getattr(self, "_direct_phone_relaunch_required", None) is not None
            if direct_phone.active:
                # a session the primary's loss left behind cannot finish its execution proofs
                self._stop_phone_session(
                    terminate_phone_session=True,
                    **({"allow_incomplete_direct_phone": True} if relaunch else {}),
                )
            android_launcher = getattr(self, "_android_phone_launcher", None)
            if android_launcher is not None:
                android_launcher.prepare_ncm_control(direct_phone.configuration.functionfs_gadget_path)
            direct_phone.start(
                command,
                state.manifest,
                state.direct_transport,
                control_check=control_check,
            )
            if relaunch:
                self._direct_phone_relaunched(state.target_executor_id)
        state.direct_phone_used = True
        with self._lock:
            self._current_direct_phone = direct_phone
            self._phone_executor_id = state.target_executor_id
            self._phone_parameters = dict(command.adapter_parameters)
            self._last_phone_parameters = dict(
                command.adapter_parameters
            )
            self._phone_session_active = True
            self._phone_terminal_sent = False

    def _prepare_transition_phone(
        self,
        command: PhysicalTransitionCommand,
        state: _TransitionExecutionState,
        control_check: Callable[[], None],
    ) -> None:
        if state.phone is None:
            return
        assert state.direct_transport is not None
        if state.direct_transport.transport == "functionfs-usb":
            self._prepare_functionfs_phone(
                command, state, control_check
            )
            return
        bridge = self._start_bridge(command)
        with self._lock:
            self._current_bridge = bridge
            self._phone_executor_id = state.target_executor_id
            self._phone_parameters = dict(command.adapter_parameters)
            self._last_phone_parameters = dict(
                command.adapter_parameters
            )
            self._phone_session_active = True
            self._phone_terminal_sent = False

    def _publish_helper_transition(
        self,
        command: PhysicalTransitionCommand,
        state: _TransitionExecutionState,
    ) -> None:
        if not state.direct_phone_used:
            raise PhysicalAdapterError(
                "request helper transition lacks direct phone"
            )
        direct_phone = self._direct_phone_session
        with self._lock:
            self._generation += 1
            generation = self._generation
            previous_phone = self._phone_residency
            if state.direct_phone_reused and previous_phone is None:
                raise PhysicalAdapterError(
                    "persistent phone residency state is absent"
                )
            self._phone_residency = (
                previous_phone
                if state.direct_phone_reused
                else self._persistent_phone_residency_state(
                    command,
                    state.manifest,
                    direct_phone,
                    fallback_generation=generation,
                    previous=previous_phone,
                )
            )
            self._helper_reconfigurations[command.ticket_id] = (
                _HelperReconfiguration(
                    transition_id=command.transition.transition_id,
                    changed_session_ids=tuple(
                        command.transition.changed_phone_session_ids
                    ),
                    receipt=(
                        state.direct_phone_reconfiguration_receipt
                        if state.direct_phone_reconfigured else None
                    ),
                    previous_phone_residency=(
                        state.previous_phone_residency
                    ),
                    fresh_start=(
                        not state.direct_phone_reused
                        and not state.direct_phone_reconfigured
                    ),
                )
            )

    def _launch_transition_server(
        self,
        command: PhysicalTransitionCommand,
        state: _TransitionExecutionState,
        control_check: Callable[[], None],
    ):
        label = (
            "large-model-"
            + str(self._launch_attempt)
            + "-"
            + command.participant.executor_id.replace(":", "-")
        )
        if (
            command.adapter_parameters.get("execution_adapter")
            == ANDROID_LLAMA_SERVER_ADAPTER
        ):
            if self._android_phone_launcher is None:
                raise PhysicalAdapterError(
                    "Android llama-server launcher is absent"
                )
            return self._android_phone_launcher.launch(
                command,
                state.manifest,
                label=label,
                control_check=control_check,
            )
        return self._launcher.launch(
            command,
            state.manifest,
            label=label,
            control_check=control_check,
        )

    def _publish_transition_server(
        self,
        command: PhysicalTransitionCommand,
        state: _TransitionExecutionState,
        server,
    ) -> int:
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._live_executors[state.target_executor_id] = (
                _LiveExecutorResidency(
                    executor_id=state.target_executor_id,
                    endpoint=command.participant.endpoint,
                    server=server,
                    manifest=state.manifest,
                    parameters=dict(command.adapter_parameters),
                    operator_plan=dict(command.operator_plan),
                    generation=generation,
                    participant_device_ids=state.target_devices,
                    replacement_resource_ids=state.target_replacement,
                    session_resource_ids=state.target_session,
                    owns_phone_session=state.direct_phone_used,
                )
            )
            self._dormant_book_server(command.participant.endpoint, server, state.manifest, command.adapter_parameters)
            if state.direct_phone_used:
                previous_phone = self._phone_residency
                if state.direct_phone_reused and previous_phone is None:
                    raise PhysicalAdapterError(
                        "persistent phone residency state is absent"
                    )
                self._phone_residency = (
                    previous_phone
                    if state.direct_phone_reused
                    else self._persistent_phone_residency_state(
                        command,
                        state.manifest,
                        self._direct_phone_session,
                        fallback_generation=generation,
                        previous=previous_phone,
                    )
                )
        return generation

    def _warm_transition_server(
        self,
        command: PhysicalTransitionCommand,
        payload: LlamaCppCompletionPayload,
        generation: int,
        control_check: Callable[[], None],
    ) -> None:
        warm_path = payload.stream_path.with_name(
            payload.stream_path.stem
            + "-transition-warm-"
            + str(generation)
            + payload.stream_path.suffix
        )
        self._client.complete(
            command.participant.endpoint,
            replace(
                payload,
                input_tokens=1,
                output_tokens=2,
                prompt_tokens=(payload.prompt_tokens[0],),
                quality_mode="accounting-only",
                stream_path=warm_path,
                on_first_token=lambda _value: None,
            ),
            control_check,
        )

    def _cleanup_transition_failure(
        self,
        state: _TransitionExecutionState,
        primary_error: BaseException,
    ) -> None:
        if not state.mutation_started:
            return
        try:
            if state.helper_only:
                if (
                    state.direct_phone_used
                    and not state.direct_phone_reused
                    and not state.direct_phone_reconfigurable
                ):
                    self._stop_phone_session(
                        terminate_phone_session=True,
                        allow_incomplete_direct_phone=True,
                    )
            else:
                self._stop_executor(
                    state.target_executor_id,
                    terminate_phone_session=(
                        not state.direct_phone_reused
                        and not state.direct_phone_reconfigurable
                    ),
                    allow_incomplete_direct_phone=True,
                )
            with self._lock:
                orphan_phone = (
                    not state.helper_only
                    and state.direct_phone_used
                    and self._phone_executor_id
                        == state.target_executor_id
                )
            if (
                orphan_phone
                and not state.direct_phone_reused
                and not state.direct_phone_reconfigurable
            ):
                self._stop_phone_session(
                    terminate_phone_session=True,
                    allow_incomplete_direct_phone=True,
                )
            elif (
                state.direct_phone_reused
                or (
                    state.direct_phone_reconfigurable
                    and not state.direct_phone_reconfigured
                )
            ):
                self._restore_previous_phone_residency(
                    state.previous_phone_residency
                )
            elif state.direct_phone_reconfigured:
                receipt = state.direct_phone_reconfiguration_receipt
                if receipt is None:
                    raise PhysicalAdapterError(
                        "partial phone transition receipt is absent"
                    )
                try:
                    self._direct_phone_session.rollback_reconfiguration(
                        receipt
                    )
                except BaseException:
                    with self._lock:
                        self._phone_residency = None
                    raise
                with self._lock:
                    self._phone_residency = replace(
                        state.previous_phone_residency,
                        load_count_by_session=(
                            self._direct_phone_session
                                .load_count_by_session
                        ),
                        column_quantum_by_session=(
                            self._direct_phone_session
                                .column_quantum_by_session
                        ),
                        max_tokens_by_session=(
                            self._direct_phone_session
                                .max_tokens_by_session
                        ),
                    )
        except BaseException as cleanup_error:
            primary_error.add_note(
                "transition cleanup failed: " + str(cleanup_error)
            )

    @contextmanager
    def _transition_scope(self, command: PhysicalTransitionCommand):
        phone_device_id = self.configuration.phone_device_id
        devices = set(command.transition.prepares_device_ids) | {
            row.device_id for row in command.transition.evictions
        }
        helper_only = bool(getattr(command, "helper_only", False))
        if helper_only and devices != {phone_device_id}:
            raise PhysicalAdapterError(
                "helper transition includes non-phone residency changes"
            )
        uses_phone = (
            phone_device_id in devices
            or command.adapter_parameters.get("phone_device_id") is not None
        )
        uses_desktop = not helper_only and (
            bool(devices - {phone_device_id}) or not uses_phone
        )
        with ExitStack() as stack:
            if uses_phone:
                stack.enter_context(self._transition_lock)
            if uses_desktop:
                stack.enter_context(self._desktop_transition_lock)
            with self._lock:
                self._active_transition_count += 1
                self._transition_active = True
            power = getattr(self, "_device_power", None)
            if power is not None:
                power.note_transition_active(True)
            try:
                yield
            finally:
                with self._lock:
                    self._active_transition_count -= 1
                    self._transition_active = self._active_transition_count > 0
                    transition_active = self._transition_active
                if power is not None:
                    power.note_transition_active(transition_active)

    def rollback_helper_transition(
        self, command: PhysicalTransitionCommand
    ) -> Mapping[str, object]:
        """Physically restore the source shard of one completed helper load.

        The scheduler calls this when its commit fails after the phone was
        already reconfigured. Success is verified against the captured
        source layout; any failure leaves the phone residency unknown so the
        replaced session is never published as resident again.
        """

        with self._transition_lock:
            with self._lock:
                record = self._helper_reconfigurations.pop(
                    command.ticket_id, None
                )
                direct_phone = self._current_direct_phone
            transition_id = command.transition.transition_id
            if record is None or record.transition_id != transition_id:
                raise PhysicalAdapterError(
                    "helper transition rollback has no reconfiguration receipt"
                )
            base = {
                "command_changed_session_ids": list(
                    command.transition.changed_phone_session_ids
                ),
                "kind": "helper_transition_rollback",
                "phone_layout_generation": command.phone_layout_generation,
                "schema": "research-scheduler-phone-session-rollback-v1",
                "ticket_id": command.ticket_id,
                "transition_id": transition_id,
            }
            if record.fresh_start:
                # The load created the whole phone residency; undoing it
                # means stopping that session so nothing stays resident.
                try:
                    self._stop_phone_session(
                        terminate_phone_session=True,
                        allow_incomplete_direct_phone=True,
                    )
                finally:
                    with self._lock:
                        self._phone_residency = None
                        self._phone_executor_id = None
                        self._phone_parameters = None
                receipt = {
                    **base,
                    "physical_change": True,
                    "restored_shards": [],
                }
                with self._lock:
                    self._direct_phone_receipts.append(receipt)
                return MappingProxyType(receipt)
            if record.receipt is None:
                receipt = {**base, "physical_change": False}
                with self._lock:
                    self._direct_phone_receipts.append(receipt)
                return MappingProxyType(receipt)
            previous = record.previous_phone_residency
            if (
                direct_phone is None
                or not direct_phone.active
                or previous is None
            ):
                with self._lock:
                    self._phone_residency = None
                raise PhysicalAdapterError(
                    "helper transition rollback has no active phone session"
                )
            try:
                direct_phone.rollback_reconfiguration(record.receipt)
            except BaseException:
                with self._lock:
                    self._phone_residency = None
                raise
            restored = tuple(sorted(
                direct_phone.phone_shards, key=lambda row: row.session_id
            ))
            changed_session_id = record.receipt.changed_session_id
            expected_restored = (
                previous.phone_shards
                if record.receipt.previous_shard is None else
                tuple(
                    replace(
                        row,
                        session_generation=(
                            record.receipt.target_shard.session_generation + 1
                        ),
                    )
                    if row.session_id == changed_session_id else row
                    for row in previous.phone_shards
                )
            )
            if restored != expected_restored:
                with self._lock:
                    self._phone_residency = None
                raise PhysicalAdapterError(
                    "helper transition rollback did not restore the source"
                )
            restored_shard = next((
                row for row in restored
                if row.session_id == changed_session_id
            ), None)
            with self._lock:
                self._phone_residency = replace(
                    previous,
                    phone_shards=restored,
                    load_count_by_session=(
                        direct_phone.load_count_by_session
                    ),
                    column_quantum_by_session=(
                        direct_phone.column_quantum_by_session
                    ),
                    max_tokens_by_session=(
                        direct_phone.max_tokens_by_session
                    ),
                )
                receipt = {
                    **base,
                    "changed_session_id": changed_session_id,
                    "load_count": direct_phone.load_count_by_session.get(
                        changed_session_id, 0
                    ),
                    "physical_change": True,
                    "residency_generation": direct_phone.residency_generation,
                    "restored_session_generations": {
                        row.session_id: row.session_generation
                        for row in restored
                    },
                    "restored_empty_session_ids": (
                        [changed_session_id]
                        if restored_shard is None else []
                    ),
                    "restored_shard": (
                        None if restored_shard is None else
                        restored_shard.to_json()
                    ),
                    "restored_shards": [
                        row.to_json() for row in restored
                    ],
                    "reverted_shard": record.receipt.target_shard.to_json(),
                }
                self._direct_phone_receipts.append(receipt)
            return MappingProxyType(receipt)
