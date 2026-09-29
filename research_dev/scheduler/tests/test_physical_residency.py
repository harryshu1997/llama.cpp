#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from research_dev.scheduler import RuntimeResidencyEviction
from research_dev.scheduler import RuntimePhoneShard, UnifiedScheduler
from research_dev.scheduler._internal.runtime_plan import (
    PhoneSessionReplacementAuthorization,
)
from research_dev.scheduler.adapters import (
    DirectPhoneFfnSession,
    EndpointRuntimeSample,
    HeterogeneousPhysicalRig,
    LlamaCppCompletionPayload,
    PhysicalAdapterError,
)
from research_dev.scheduler.adapters.phone_transport import (
    PhoneTransportContract,
)
from research_dev.scheduler.adapters.heterogeneous_rig import (
    _LiveExecutorResidency,
    _PersistentPhoneResidency,
)
from research_dev.scheduler.adapters.llama_server import (
    PhoneFfnExecutionContract,
)
from research_dev.scheduler.adapters.residency import (
    PhysicalPhoneSessionEndpoint,
    PhysicalResidentEndpoint,
    physical_transition_stop_set,
)


ARTIFACT_A = "sha256:" + "a" * 64
ARTIFACT_B = "sha256:" + "b" * 64


def resident(
    executor_id: str,
    artifact_sha256: str,
    generation: int,
    *,
    endpoint: str,
    devices: tuple[str, ...],
    replacement: tuple[str, ...] = (),
    session: tuple[str, ...] = (),
) -> PhysicalResidentEndpoint:
    return PhysicalResidentEndpoint(
        executor_id=executor_id,
        endpoint=endpoint,
        artifact_sha256=artifact_sha256,
        generation=generation,
        participant_device_ids=devices,
        replacement_resource_ids=replacement,
        session_resource_ids=session,
    )


class PhysicalResidencyTests(unittest.TestCase):
    @staticmethod
    def shard(
        session_id: str,
        artifact_sha256: str,
        layer_mask: int,
        resident_bytes: int,
        session_generation: int = 1,
    ) -> RuntimePhoneShard:
        suffix = session_id[-1].lower()
        return RuntimePhoneShard(
            session_id=session_id,
            endpoint="session://phone/" + session_id,
            layer_mask=layer_mask,
            maximum_columns=128,
            resident_bytes=resident_bytes,
            resident_geometry_sha256="sha256:" + suffix * 64,
            operator_plan_sha256="sha256:" + (
                "d" if artifact_sha256 == ARTIFACT_A else "e"
            ) * 64,
            artifact_sha256=artifact_sha256,
            session_generation=session_generation,
        )

    @staticmethod
    def ffn_contract(max_tokens: int) -> PhoneFfnExecutionContract:
        return PhoneFfnExecutionContract(
            device_id="phone",
            n_embd=128,
            layer_indices=(0,),
            layer_mask=1,
            columns=128,
            max_tokens=max_tokens,
            activation="gelu",
        )

    @staticmethod
    def phone_transport(
        *,
        maximum: int,
        profile: str,
        identity: str = "sha256:" + "1" * 64,
    ) -> PhoneTransportContract:
        return PhoneTransportContract(
            transport="functionfs-usb",
            allocator="devmem",
            queue_depth=4,
            concurrent_streams=4,
            max_payload_bytes=maximum,
            full_duplex=True,
            split_h2d=False,
            generation="functionfs-dmabuf-async-ring-v2",
            profile_id=profile,
            usbfs_available_bytes=16_777_216,
            slot_safety_bytes=65_536,
            vendor_id=0x18D1,
            product_id=0x2D00,
            control_host="direct-functionfs",
            control_port=0,
            batch_plan="split-row",
            qualification_identity_sha256=identity,
        )

    def test_resident_router_accepts_payload_specific_contracts(self) -> None:
        launch = self.phone_transport(maximum=15_360, profile="payload-7680")
        target = self.phone_transport(maximum=20_480, profile="payload-10240")

        self.assertTrue(launch.shares_resident_session_with(target))
        self.assertFalse(launch.shares_resident_session_with(
            self.phone_transport(
                maximum=20_480,
                profile="payload-10240",
                identity="sha256:" + "2" * 64,
            )
        ))

    def test_one_session_replacement_reuses_resident_router(self) -> None:
        launch_transport = self.phone_transport(
            maximum=15_360, profile="payload-7680"
        )
        target_transport = self.phone_transport(
            maximum=20_480, profile="payload-10240"
        )
        retained = tuple(
            replace(
                self.shard(
                    "HTP" + str(index),
                    ARTIFACT_A,
                    1 << index,
                    100,
                ),
                maximum_columns=15_360,
            )
            for index in range(2)
        )
        previous = replace(
            self.shard("HTP2", ARTIFACT_A, 1 << 2, 100),
            maximum_columns=15_360,
        )
        replacement = replace(
            self.shard(
                "HTP2",
                ARTIFACT_B,
                (1 << 6) - 1,
                120,
                session_generation=2,
            ),
            maximum_columns=17_408,
        )
        source_shards = (*retained, previous)
        target_shards = (*retained, replacement)
        session = object.__new__(DirectPhoneFfnSession)
        session._launch = SimpleNamespace(
            phone_shards=source_shards,
            transport=launch_transport,
            execution=SimpleNamespace(max_tokens=2),
        )
        session._remote_root = "/data/local/tmp/resident"
        session.configuration = SimpleNamespace(
            diagnostic_host="192.0.2.1",
            diagnostic_port=20_000,
            model_paths_by_artifact={ARTIFACT_B: "/data/model-b.gguf"},
        )
        session._verified_remote_hash_by_path = {
            "/data/model-b.gguf": ARTIFACT_B,
        }
        command = SimpleNamespace(
            artifact_sha256=ARTIFACT_B,
            replacement_authorization=(
                PhoneSessionReplacementAuthorization.create(
                    selected_session_id="HTP2",
                    source_shards=source_shards,
                    target_shards=target_shards,
                )
            ),
            transition=SimpleNamespace(
                phone_shards=target_shards,
                changed_phone_session_ids=("HTP2",),
            ),
        )
        manifest = SimpleNamespace(artifact_sha256=ARTIFACT_B)

        with mock.patch(
            "research_dev.scheduler.adapters.phone_session_ops.replacement."
            "phone_ffn_resident_contract",
            return_value=SimpleNamespace(
                layer_mask=(1 << 6) - 1,
                columns=17_408,
                max_tokens=4,
            ),
        ):
            self.assertTrue(session.supports_partial_reconfiguration(
                command, manifest, target_transport
            ))

        with mock.patch(
            "research_dev.scheduler.adapters.phone_session_ops.replacement."
            "phone_ffn_resident_contract",
            return_value=SimpleNamespace(
                layer_mask=(1 << 6) - 1,
                columns=17_409,
                max_tokens=4,
            ),
        ):
            self.assertFalse(session.supports_partial_reconfiguration(
                command, manifest, target_transport
            ))

    def test_partial_add_requires_eviction_only_for_resident_session(
        self,
    ) -> None:
        retained = self.shard("HTP0", ARTIFACT_A, 1, 100)
        previous = _PersistentPhoneResidency(
            executor_id="phone-helper",
            endpoint="http://desktop.invalid:1",
            phone_shards=(retained,),
            layout_geometry_sha256="sha256:" + "8" * 64,
            manifests_by_artifact={
                ARTIFACT_A: SimpleNamespace(
                    artifact_sha256=ARTIFACT_A
                ),
            },
            parameters_by_artifact={ARTIFACT_A: {}},
            operator_plans_by_artifact={ARTIFACT_A: {}},
            executions_by_artifact={
                ARTIFACT_A: self.ffn_contract(2),
            },
            load_count_by_session={"HTP0": 1},
            column_quantum_by_session={"HTP0": 32},
            max_tokens_by_session={"HTP0": 2},
            generation=1,
            participant_device_ids=("phone",),
            replacement_resource_ids=("phone-memory",),
            session_resource_ids=("phone-session",),
        )
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig.configuration = SimpleNamespace(phone_device_id="phone")
        rig._phone_executor_id = "phone-helper"
        command = SimpleNamespace(
            artifact_sha256=ARTIFACT_A,
            participant=SimpleNamespace(
                endpoint="http://desktop.invalid:1"
            ),
            transition=SimpleNamespace(
                changed_phone_session_ids=("HTP1",),
                evictions=(),
            ),
        )
        state = SimpleNamespace(
            direct_phone_partial_requested=True,
            direct_phone_reconfigurable=True,
            live={},
            previous_phone_residency=previous,
            target_executor_id="phone-helper",
            target_replacement=(),
            target_session=(),
        )

        with mock.patch(
            "research_dev.scheduler.adapters.heterogeneous_rig."
            "physical_transition_stop_set",
            return_value=(),
        ) as stop_set:
            self.assertEqual(
                rig._transition_conflicting_executors(command, state), ()
            )
        self.assertEqual(
            set(stop_set.call_args.kwargs["phone_sessions"]), {"HTP0"}
        )

        command.artifact_sha256 = ARTIFACT_B
        state.target_executor_id = "another-model-phone-helper"
        state.target_replacement = ("phone-memory",)
        self.assertEqual(rig._transition_conflicting_executors(command, state), ())
        state.direct_phone_reconfigurable = False
        with self.assertRaisesRegex(PhysicalAdapterError, "exact partial phone residency transition is unavailable"):
            rig._transition_conflicting_executors(command, state)
        state.direct_phone_partial_requested = False
        with self.assertRaisesRegex(PhysicalAdapterError, "exclusive residency replacement lacks an exact eviction"):
            rig._transition_conflicting_executors(command, state)
        state.direct_phone_partial_requested = True
        state.direct_phone_reconfigurable = True

        command.transition.changed_phone_session_ids = ("HTP0",)
        with self.assertRaisesRegex(
            PhysicalAdapterError,
            "partial phone transition lacks exact session eviction",
        ):
            rig._transition_conflicting_executors(command, state)

    def test_ggg_to_ggq_preserves_retained_shard_objects(self) -> None:
        old = tuple(
            self.shard("HTP" + str(index), ARTIFACT_A, 1 << index, 100)
            for index in range(3)
        )
        replacement = self.shard(
            "HTP2", ARTIFACT_B, 1, 120, session_generation=2
        )
        previous = _PersistentPhoneResidency(
            executor_id="phone-helper",
            endpoint="http://desktop.invalid:1",
            phone_shards=old,
            layout_geometry_sha256="sha256:" + "8" * 64,
            manifests_by_artifact={
                ARTIFACT_A: SimpleNamespace(
                    artifact_sha256=ARTIFACT_A
                ),
            },
            parameters_by_artifact={ARTIFACT_A: {"old": 1}},
            operator_plans_by_artifact={ARTIFACT_A: {"old": 1}},
            executions_by_artifact={ARTIFACT_A: self.ffn_contract(2)},
            load_count_by_session={
                "HTP0": 1, "HTP1": 1, "HTP2": 1,
            },
            column_quantum_by_session={
                "HTP0": 32, "HTP1": 32, "HTP2": 32,
            },
            max_tokens_by_session={
                "HTP0": 2, "HTP1": 2, "HTP2": 2,
            },
            generation=1,
            participant_device_ids=("phone",),
            replacement_resource_ids=("phone-memory",),
            session_resource_ids=("htp", "usb"),
        )
        direct_phone = SimpleNamespace(
            phone_shards=(*old[:2], replacement),
            load_count_by_session={
                "HTP0": 1, "HTP1": 1, "HTP2": 2,
            },
            column_quantum_by_session={
                "HTP0": 32, "HTP1": 32, "HTP2": 32,
            },
            max_tokens_by_session={
                "HTP0": 2, "HTP1": 2, "HTP2": 4,
            },
        )
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._phone_residency_resources = lambda _executor: (
            ("phone",), ("phone-memory",), ("htp", "usb")
        )
        command = SimpleNamespace(
            adapter_parameters={
                "phone_shard_set_geometry_sha256": (
                    "sha256:" + "9" * 64
                ),
            },
            operator_plan={"new": 1},
            participant=SimpleNamespace(
                executor_id="phone-helper",
                endpoint="http://desktop.invalid:1",
            ),
            phone_layout_generation=2,
        )
        manifest = SimpleNamespace(artifact_sha256=ARTIFACT_B)

        with mock.patch(
            "research_dev.scheduler.adapters.heterogeneous_rig_ops.residency."
            "phone_ffn_resident_contract",
            return_value=self.ffn_contract(4),
        ):
            current = rig._persistent_phone_residency_state(
                command,
                manifest,
                direct_phone,
                fallback_generation=2,
                previous=previous,
            )

        self.assertIs(current.phone_shards[0], old[0])
        self.assertIs(current.phone_shards[1], old[1])
        self.assertIs(current.phone_shards[2], replacement)
        self.assertEqual(current.load_count_by_session["HTP0"], 1)
        self.assertEqual(current.load_count_by_session["HTP1"], 1)
        self.assertEqual(current.load_count_by_session["HTP2"], 2)
        self.assertEqual(
            set(current.covered_artifacts), {ARTIFACT_A, ARTIFACT_B}
        )

    def test_failed_partial_replacement_keeps_launch_identity(self) -> None:
        previous = self.shard("HTP2", ARTIFACT_A, 1, 100)
        replacement = self.shard(
            "HTP2", ARTIFACT_B, 1, 120, session_generation=2
        )
        retained = (
            self.shard("HTP0", ARTIFACT_A, 2, 100),
            self.shard("HTP1", ARTIFACT_A, 4, 100),
        )
        launch = SimpleNamespace(
            phone_shards=(*retained, previous),
            shard_manifest_sha256="sha256:" + "1" * 64,
        )
        session = object.__new__(DirectPhoneFfnSession)
        session._launch = launch
        session._remote_root = "/data/local/tmp/resident"
        session._load_count_by_session = {
            "HTP0": 1, "HTP1": 1, "HTP2": 1,
        }
        session._column_quantum_by_session = {
            "HTP0": 32, "HTP1": 32, "HTP2": 32,
        }
        session._max_tokens_by_session = {
            "HTP0": 2, "HTP1": 2, "HTP2": 2,
        }
        session.configuration = SimpleNamespace(
            diagnostic_host="192.0.2.1",
            diagnostic_port=20_000,
            launch_timeout_s=1,
        )
        command = SimpleNamespace(
            adapter_parameters={"ffn_column_quantum": 32},
            ticket_id="ticket-b",
            replacement_authorization=(
                PhoneSessionReplacementAuthorization.create(
                    selected_session_id="HTP2",
                    source_shards=(*retained, previous),
                    target_shards=(*retained, replacement),
                )
            ),
            transition=SimpleNamespace(
                phone_shards=(*retained, replacement),
                changed_phone_session_ids=("HTP2",),
            ),
        )
        with mock.patch.object(
            session, "supports_partial_reconfiguration", return_value=True
        ), mock.patch.object(
            session,
            "_multi_session_manifest",
            return_value=("manifest", "sha256:" + "2" * 64),
        ), mock.patch(
            "research_dev.scheduler.adapters.phone_session_ops.replacement."
            "phone_ffn_resident_contract",
            return_value=self.ffn_contract(4),
        ), mock.patch(
            "research_dev.scheduler.adapters.phone_session."
            "socket.create_connection",
            side_effect=OSError("synthetic replacement failure"),
        ):
            with self.assertRaisesRegex(
                PhysicalAdapterError, "phone residency control failed"
            ):
                session.reconfigure(
                    command,
                    SimpleNamespace(artifact_sha256=ARTIFACT_B),
                    self.phone_transport(maximum=1024, profile="small"),
                )

        self.assertIs(session._launch, launch)
        self.assertEqual(session._load_count_by_session["HTP2"], 1)

    def test_mixed_layout_close_uses_current_artifact_identity(self) -> None:
        shards = (
            self.shard("HTP0", ARTIFACT_A, 1, 100),
            self.shard("HTP1", ARTIFACT_A, 2, 100),
            self.shard("HTP2", ARTIFACT_B, 3, 120),
        )
        session = object.__new__(DirectPhoneFfnSession)
        session._launch = SimpleNamespace(
            phone_shards=shards,
            execution=self.ffn_contract(2),
            transport=self.phone_transport(maximum=1024, profile="a"),
            artifact_sha256=ARTIFACT_A,
        )
        session._execution_by_artifact = {
            ARTIFACT_A: self.ffn_contract(2),
            ARTIFACT_B: self.ffn_contract(4),
        }
        session._transport_by_artifact = {
            ARTIFACT_A: self.phone_transport(maximum=1024, profile="a"),
            ARTIFACT_B: self.phone_transport(maximum=2048, profile="b"),
        }
        session._max_tokens_by_session = {
            "HTP0": 2, "HTP1": 2, "HTP2": 4,
        }

        execution, transport, artifact = session._current_close_identity()

        self.assertEqual(artifact, ARTIFACT_B)
        self.assertEqual(execution.layer_mask, 3)
        self.assertEqual(execution.max_tokens, 4)
        self.assertEqual(transport.profile_id, "b")

    def test_mixed_layout_close_tries_each_current_identity(self) -> None:
        shards = (
            self.shard("HTP0", ARTIFACT_A, 1, 100),
            self.shard("HTP1", ARTIFACT_A, 2, 100),
            self.shard("HTP2", ARTIFACT_B, 3, 120),
        )
        session = object.__new__(DirectPhoneFfnSession)
        session._launch = SimpleNamespace(
            phone_shards=shards,
            execution=self.ffn_contract(2),
            transport=self.phone_transport(maximum=1024, profile="a"),
            artifact_sha256=ARTIFACT_A,
        )
        session._execution_by_artifact = {
            ARTIFACT_A: self.ffn_contract(2),
            ARTIFACT_B: self.ffn_contract(4),
        }
        session._transport_by_artifact = {
            ARTIFACT_A: self.phone_transport(maximum=1024, profile="a"),
            ARTIFACT_B: self.phone_transport(maximum=2048, profile="b"),
        }
        session._max_tokens_by_session = {
            "HTP0": 2, "HTP1": 2, "HTP2": 4,
        }
        session._close_direct_usb = mock.Mock(side_effect=(
            PhysicalAdapterError("synthetic identity rejection"),
            None,
        ))

        session._close_current_usb()

        calls = session._close_direct_usb.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args[2], ARTIFACT_B)
        self.assertEqual(calls[1].args[2], ARTIFACT_A)
        self.assertEqual(calls[1].args[0].layer_mask, 3)

    def test_mixed_layout_requires_separate_artifact_rows(self) -> None:
        geometry = "sha256:" + "3" * 64
        shard_a = RuntimePhoneShard(
            session_id="HTP0",
            endpoint="session://phone/HTP0",
            layer_mask=1,
            maximum_columns=128,
            resident_bytes=100,
            resident_geometry_sha256="sha256:" + "4" * 64,
            operator_plan_sha256="sha256:" + "5" * 64,
            artifact_sha256=ARTIFACT_A,
        )
        shard_b = RuntimePhoneShard(
            session_id="HTP2",
            endpoint="session://phone/HTP2",
            layer_mask=2,
            maximum_columns=128,
            resident_bytes=200,
            resident_geometry_sha256="sha256:" + "6" * 64,
            operator_plan_sha256="sha256:" + "7" * 64,
            artifact_sha256=ARTIFACT_B,
        )
        scheduler = object.__new__(UnifiedScheduler)
        scheduler._runtime_capabilities = SimpleNamespace(
            executor_by_device={
                "phone": SimpleNamespace(phone_sessions=()),
            },
        )
        state = SimpleNamespace(
            generation=2,
            covered_artifact_sha256s=(ARTIFACT_A, ARTIFACT_B),
            layout=SimpleNamespace(
                shards=(shard_a, shard_b),
                geometry_sha256=geometry,
                resident_bytes=300,
            ),
        )
        plan = SimpleNamespace(
            execution_contract=SimpleNamespace(
                phone_device_id="phone",
                phone_shards=(shard_a,),
            ),
            memory_demands=(),
            resource_ids=(),
        )
        observed = tuple(
            SimpleNamespace(
                artifact_sha256=artifact,
                device_id="phone",
                executor_id="phone-helper",
                generation=2,
                model_id=model_id,
                resident_bytes=resident_bytes,
                resident_geometry_sha256=geometry,
                resident_tensor_ids=(),
                state="hot",
                to_json=lambda artifact=artifact,
                    resident_bytes=resident_bytes: {
                        "artifact_sha256": artifact,
                        "device_id": "phone",
                        "resident_bytes": resident_bytes,
                        "resident_geometry_sha256": geometry,
                    },
            )
            for artifact, model_id, resident_bytes in (
                (ARTIFACT_A, "model-a", 100),
                (ARTIFACT_B, "model-b", 200),
            )
        )
        snapshot = SimpleNamespace(
            captured_at_us=10,
            executors={
                "desktop": SimpleNamespace(healthy=True, ready=True),
            },
            links={},
            residency=observed,
            snapshot_id="mixed-layout-snapshot",
        )

        verification = scheduler._phone_layout_snapshot_verification(
            state,
            model_id="model-a",
            artifact_sha256=ARTIFACT_A,
            plan=plan,
            binding=SimpleNamespace(executor_id="phone-helper"),
            base_executor_id="desktop",
            snapshot=snapshot,
        )

        self.assertIsNotNone(verification)

        aggregate_only = SimpleNamespace(
            captured_at_us=snapshot.captured_at_us,
            executors=snapshot.executors,
            links=snapshot.links,
            residency=(SimpleNamespace(
                **{
                    **vars(observed[1]),
                    "resident_bytes": 300,
                },
            ),),
            snapshot_id=snapshot.snapshot_id,
        )
        self.assertIsNone(scheduler._phone_layout_snapshot_verification(
            state,
            model_id="model-a",
            artifact_sha256=ARTIFACT_A,
            plan=plan,
            binding=SimpleNamespace(executor_id="phone-helper"),
            base_executor_id="desktop",
            snapshot=aggregate_only,
        ))

    def test_desktop_transition_preserves_direct_phone_residency(self) -> None:
        direct_phone = SimpleNamespace(active=True)
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.RLock()
        rig._current_direct_phone = direct_phone
        rig._finish_direct_phone = mock.Mock()

        rig._stop_phone_session(terminate_phone_session=False)

        rig._finish_direct_phone.assert_not_called()
        self.assertIs(rig._current_direct_phone, direct_phone)

    def test_desktop_campaign_restart_preserves_exact_phone_map(self) -> None:
        shard = SimpleNamespace(to_json=lambda: {
            "artifact_sha256": ARTIFACT_A,
            "endpoint": "session://phone/HTP1",
            "resident_bytes": 128,
            "resident_geometry_sha256": "sha256:" + "1" * 64,
            "operator_plan_sha256": "sha256:" + "2" * 64,
            "session_generation": 1,
            "session_id": "HTP1",
        })
        direct_phone = SimpleNamespace(
            active=True,
            load_count_by_session={"HTP1": 1},
            column_quantum_by_session={"HTP1": 32},
            phone_shards=(shard,),
            residency_generation=1,
            weight_sources=(),
        )
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.RLock()
        rig._transition_active = False
        rig._execution_markers = {}
        rig._execution_backend = object()
        rig._direct_phone_session = direct_phone
        rig._stop_dynamic_executors = mock.Mock()

        receipt = rig.restart_desktop_campaign()

        rig._stop_dynamic_executors.assert_called_once_with(
            terminate_phone_session=False
        )
        self.assertIsNone(rig._execution_backend)
        self.assertEqual(
            receipt["phone_residency_before"],
            receipt["phone_residency_after"],
        )
        self.assertTrue(direct_phone.active)

    def test_stopping_whole_phone_endpoint_preserves_ffn_router(self) -> None:
        server = SimpleNamespace(stop=mock.Mock())
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.RLock()
        rig._phone_executor_id = "physical:op15-phone"
        rig._stop_phone_session = mock.Mock()
        rig._live_executors = {
            "physical:op15-phone": _LiveExecutorResidency(
                executor_id="physical:op15-phone",
                endpoint="http://127.0.0.1:29382",
                server=server,
                manifest=SimpleNamespace(artifact_sha256=ARTIFACT_A),
                parameters={"execution_adapter": "android-llama-server-v1"},
                operator_plan={},
                generation=3,
                participant_device_ids=("op15-phone",),
                replacement_resource_ids=(),
                session_resource_ids=(),
                owns_phone_session=False,
            ),
        }

        rig._stop_executor("physical:op15-phone")

        server.stop.assert_called_once_with()
        rig._stop_phone_session.assert_not_called()
        self.assertEqual(rig._phone_executor_id, "physical:op15-phone")

    def test_phone_preparation_does_not_serialize_desktop_loading(self) -> None:
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.RLock()
        rig._transition_lock = threading.Lock()
        rig._desktop_transition_lock = threading.Lock()
        rig._transition_active = False
        rig._active_transition_count = 0
        rig.configuration = SimpleNamespace(phone_device_id="phone")
        rig._transition_manifest = mock.Mock(return_value=mock.sentinel.manifest)
        rig._reject_legacy_transition = mock.Mock()
        rig._begin_transition_execution = lambda command, *_args: SimpleNamespace(
            helper_only=command.helper_only, phone_activity_started=False,
        )
        rig._start_transition_mutation = mock.Mock()
        rig._publish_helper_transition = mock.Mock()
        rig._publish_transition_server = mock.Mock(return_value=1)
        rig._warm_transition_server = mock.Mock()
        phone_started = threading.Event()
        finish_phone = threading.Event()
        desktop_loaded = threading.Event()
        failures = []

        def prepare_phone(_command, state, _check):
            if state.helper_only:
                phone_started.set()
                if not finish_phone.wait(5):
                    raise AssertionError("test phone load was not released")

        rig._prepare_transition_phone = prepare_phone
        rig._launch_transition_server = lambda *_args: desktop_loaded.set()
        phone = SimpleNamespace(
            helper_only=True, adapter_parameters={"phone_device_id": "phone"},
            transition=SimpleNamespace(prepares_device_ids=("phone",), evictions=()),
        )
        desktop = SimpleNamespace(
            helper_only=False, adapter_parameters={},
            transition=SimpleNamespace(prepares_device_ids=("cpu", "gpu"), evictions=()),
        )

        def execute(command):
            try:
                rig._execute_transition(command, object(), lambda: None)
            except BaseException as error:
                failures.append(error)

        phone_thread = threading.Thread(target=execute, args=(phone,))
        desktop_thread = threading.Thread(target=execute, args=(desktop,))
        phone_thread.start()
        try:
            self.assertTrue(phone_started.wait(2))
            desktop_thread.start()
            self.assertTrue(desktop_loaded.wait(2))
            desktop_thread.join(2)
            self.assertFalse(desktop_thread.is_alive())
            self.assertTrue(rig._transition_active)
        finally:
            finish_phone.set()
            phone_thread.join(2)
            if desktop_thread.ident is not None:
                desktop_thread.join(2)
        self.assertFalse(failures)
        self.assertFalse(rig._transition_active)

    def test_conflicting_transition_domains_remain_serialized(self) -> None:
        for first_devices, second_devices in (
            (("phone",), ("phone",)),
            (("phone",), ("cpu", "phone")),
            (("cpu",), ("cpu", "phone")),
        ):
            with self.subTest(first=first_devices, second=second_devices):
                rig = object.__new__(HeterogeneousPhysicalRig)
                rig._lock = threading.RLock()
                rig._transition_lock = threading.Lock()
                rig._desktop_transition_lock = threading.Lock()
                rig._transition_active = False
                rig._active_transition_count = 0
                rig.configuration = SimpleNamespace(phone_device_id="phone")

                def command(devices):
                    return SimpleNamespace(
                        helper_only=devices == ("phone",), adapter_parameters={},
                        transition=SimpleNamespace(prepares_device_ids=devices, evictions=()),
                    )

                attempted = threading.Event()
                entered = threading.Event()

                def second():
                    attempted.set()
                    with rig._transition_scope(command(second_devices)):
                        entered.set()

                worker = threading.Thread(target=second)
                with rig._transition_scope(command(first_devices)):
                    worker.start()
                    self.assertTrue(attempted.wait(2))
                    self.assertFalse(entered.wait(0.02))
                worker.join(2)
                self.assertTrue(entered.is_set())
                self.assertFalse(worker.is_alive())
                self.assertFalse(rig._transition_active)

    def test_transition_warmup_uses_accounting_quality(self) -> None:
        manifest = SimpleNamespace(artifact_sha256=ARTIFACT_A)
        observed = []

        class Launcher:
            @staticmethod
            def launch(*_args, **_kwargs):
                return SimpleNamespace(process=SimpleNamespace(poll=lambda: None))

        class Client:
            @staticmethod
            def complete(_endpoint, payload, _control_check):
                observed.append(payload)

        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.RLock()
        rig._transition_lock = threading.Lock()
        rig._desktop_transition_lock = threading.Lock()
        rig._live_executors = {}
        rig._transition_active = False
        rig._active_transition_count = 0
        rig._launch_attempt = 0
        rig._generation = 0
        rig._launcher = Launcher()
        rig._client = Client()
        rig._residency_resources = lambda _executor_id: (("cpu-a",), (), ())
        rig.configuration = SimpleNamespace(
            manifests={"synthetic-model": manifest},
            phone_device_id="phone-a",
        )
        command = SimpleNamespace(
            adapter_parameters={},
            artifact_sha256=ARTIFACT_A,
            operator_plan={},
            participant=SimpleNamespace(
                endpoint="http://desktop.invalid:1",
                executor_id="executor:desktop",
            ),
            transition=SimpleNamespace(evictions=(), prepares_device_ids=("cpu-a",)),
        )
        payload = LlamaCppCompletionPayload(
            request_id="synthetic-request",
            expected_model_alias="synthetic-model",
            input_tokens=2,
            output_tokens=4,
            prompt_tokens=(1, 2),
            seed=7,
            stream_path=Path("synthetic-transition.raw"),
            on_first_token=lambda _value: None,
        )

        rig._execute_transition(command, payload, lambda: None)

        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0].quality_mode, "accounting-only")

    def test_cold_executor_snapshot_preserves_raw_endpoint_fields(self) -> None:
        sample = EndpointRuntimeSample(
            "unavailable", "failed", 0, busy_until_us=123
        )

        class Monitor:
            @staticmethod
            def snapshot(_name):
                return SimpleNamespace(
                    stale=False, error=None, value=sample
                )

        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.RLock()
        rig._runtime_monitor = Monitor()
        rig._live_executors = {}
        rig._active_large = {}
        rig.configuration = SimpleNamespace(catalog=SimpleNamespace(
            executors=(SimpleNamespace(executor_id="executor:cold", adapter_parameters={}),),
            transitions=(SimpleNamespace(executor_id="executor:cold"),),
            composite_executors=(),
        ))

        projected = rig._executor_samples()["executor:cold"]

        self.assertEqual(projected.health, sample.health)
        self.assertEqual(projected.slots_probe, sample.slots_probe)
        self.assertEqual(projected.busy_until_us, sample.busy_until_us)
        self.assertTrue(projected.transition_available)

    def test_parallel_server_accepts_scheduler_leased_desktop_work(self) -> None:
        manifest = SimpleNamespace(artifact_sha256=ARTIFACT_A)

        class Server:
            def begin_execution(self, command, selected_manifest):
                self.asserted_manifest = selected_manifest
                return "marker:" + command.ticket_id

        class Activity:
            def start(self, *_args, **_kwargs):
                pass

        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.RLock()
        rig._request_shapes = {"request-a": (8, 4), "request-b": (8, 4)}
        rig._resident_server = Server()
        rig._live_executors = {}
        rig._execution_markers = {}
        rig._execution_proofs = {}
        rig._active_large = {}
        rig._activity = Activity()
        rig.configuration = SimpleNamespace(
            resident_executor_id="executor:desktop",
            resident_model_id="synthetic-model",
            manifests={"synthetic-model": manifest},
            large_phase_id_by_model={},
        )

        def command(request_id: str):
            return SimpleNamespace(
                request_id=request_id,
                ticket_id=request_id + ":attempt:0",
                model_id="synthetic-model",
                artifact_sha256=ARTIFACT_A,
                executor_id="executor:desktop",
                endpoint="synthetic://desktop",
                adapter_parameters={},
                operator_plan={},
            )

        rig._execution_start(command("request-a"))
        rig._execution_start(command("request-b"))

        self.assertEqual(len(rig._execution_markers), 2)

    def test_resident_server_accepts_new_request_memory_plan(self) -> None:
        manifest = SimpleNamespace(artifact_sha256=ARTIFACT_A)

        class Server:
            def begin_execution(self, command, selected_manifest):
                self.asserted_manifest = selected_manifest
                return "marker:" + command.ticket_id

        class Activity:
            def start(self, *_args, **_kwargs):
                pass

        def plan(
            *,
            residency_variant: str,
            kv_bytes: int,
            operator_device: str = "gpu-a",
        ) -> dict[str, object]:
            return {
                "adapter_parameters": {"gpu_layers": 7},
                "assisted_operator_kind": None,
                "baseline_executor_id": "executor:desktop",
                "desktop_placement_sha256": "sha256:" + "c" * 64,
                "device_ids": ["cpu-a", "gpu-a"],
                "memory_demands": [
                    {
                        "additional_bytes": kv_bytes,
                        "demand_id": "kv:gpu-a",
                        "device_id": "gpu-a",
                        "kind": "kv_cache",
                        "lifetime": "request",
                        "required_bytes": kv_bytes,
                        "resident_bytes": 0,
                        "resource_id": "gpu-memory",
                    },
                    {
                        "additional_bytes": 0,
                        "demand_id": "weights:gpu-a",
                        "device_id": "gpu-a",
                        "kind": "model_weights",
                        "lifetime": "resident",
                        "required_bytes": 4096,
                        "resident_bytes": 4096,
                        "resource_id": "gpu-memory",
                        "share_key": ARTIFACT_A + ":gpu-a",
                    },
                ],
                "operators": [
                    {
                        "candidate_id": "candidate:desktop",
                        "device_ids": [operator_device],
                        "operator_id": "layer.0.ffn",
                        "operator_kind": "ffn",
                        "split_axis": "none",
                        "split_fraction_ppm": 0,
                    },
                ],
                "overlap_kind": "serial_layer_islands",
                "plan_sha256": "sha256:" + (
                    "d" if residency_variant == "cold" else "e"
                ) * 64,
                "residency_variant": residency_variant,
                "resource_ids": ["cpu-a", "gpu-a"],
                "resource_slots": {
                    "cpu-a": 4 if residency_variant == "cold" else 1,
                    "gpu-a": 8 if residency_variant == "cold" else 1,
                },
                "route_family": "layer_placement",
                "route_id": "route:desktop:" + residency_variant,
                "route_profile_id": "profile:" + residency_variant,
                "schema": "research-scheduler-execution-plan-v1",
                "split_axis": "none",
                "split_fraction_ppm": 0,
                "transitions": (
                    [{"transition_id": "load:desktop"}]
                    if residency_variant == "cold" else []
                ),
            }

        server = Server()
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.RLock()
        rig._request_shapes = {
            "request-a": (8, 4),
            "request-b": (32, 16),
        }
        rig._resident_server = None
        rig._live_executors = {
            "executor:desktop": SimpleNamespace(
                endpoint="http://desktop.invalid:1",
                manifest=manifest,
                operator_plan=plan(
                    residency_variant="cold", kv_bytes=1024
                ),
                parameters={"gpu_layers": 7},
                server=server,
            ),
        }
        rig._execution_markers = {}
        rig._execution_proofs = {}
        rig._active_large = {}
        rig._activity = Activity()
        rig.configuration = SimpleNamespace(
            resident_executor_id="executor:preloaded",
            manifests={"synthetic-model": manifest},
            large_phase_id_by_model={},
        )

        def command(request_id: str, operator_plan: dict[str, object]):
            return SimpleNamespace(
                request_id=request_id,
                ticket_id=request_id + ":attempt:0",
                model_id="synthetic-model",
                artifact_sha256=ARTIFACT_A,
                executor_id="executor:desktop",
                endpoint="http://desktop.invalid:1",
                adapter_parameters={"gpu_layers": 7},
                operator_plan=operator_plan,
            )

        rig._execution_start(command(
            "request-a",
            plan(residency_variant="hot", kv_bytes=8192),
        ))
        self.assertIn("request-a:attempt:0", rig._execution_markers)

        with self.assertRaisesRegex(
            PhysicalAdapterError, "physical execution residency differs"
        ):
            rig._execution_start(command(
                "request-b",
                plan(
                    residency_variant="hot",
                    kv_bytes=8192,
                    operator_device="cpu-a",
                ),
            ))

    def test_cpu_phone_transition_preserves_nonconflicting_gpu(self) -> None:
        live = {
            "executor:gpu-a": resident(
                "executor:gpu-a",
                ARTIFACT_A,
                7,
                endpoint="http://gpu-a.invalid:1",
                devices=("cpu-a", "gpu-a"),
                replacement=("residency:gpu-a",),
            ),
        }

        stopped = physical_transition_stop_set(
            live,
            target_artifact_sha256=ARTIFACT_B,
            target_executor_id="executor:cpu-phone-b",
            target_endpoint="http://cpu-phone-b.invalid:1",
            target_replacement_resource_ids=(),
            target_session_resource_ids=("usb:a", "compute:phone-a"),
            evictions=(),
        )

        self.assertEqual(stopped, ())

    def test_phone_transition_replaces_only_conflicting_phone_session(
        self,
    ) -> None:
        live = {
            "executor:gpu-a": resident(
                "executor:gpu-a",
                ARTIFACT_A,
                7,
                endpoint="http://gpu-a.invalid:1",
                devices=("cpu-a", "gpu-a"),
                replacement=("residency:gpu-a",),
            ),
            "executor:cpu-phone-a": resident(
                "executor:cpu-phone-a",
                ARTIFACT_A,
                8,
                endpoint="http://cpu-phone-a.invalid:1",
                devices=("cpu-a", "phone-a"),
                session=("usb:a", "compute:phone-a"),
            ),
        }

        stopped = physical_transition_stop_set(
            live,
            target_artifact_sha256=ARTIFACT_B,
            target_executor_id="executor:cpu-phone-b",
            target_endpoint="http://cpu-phone-b.invalid:1",
            target_replacement_resource_ids=(),
            target_session_resource_ids=("usb:a", "compute:phone-a"),
            evictions=(),
        )

        self.assertEqual(stopped, ("executor:cpu-phone-a",))

    def test_gpu_replacement_requires_exact_scheduler_eviction(self) -> None:
        live = {
            "executor:gpu-a": resident(
                "executor:gpu-a",
                ARTIFACT_A,
                7,
                endpoint="http://gpu-a.invalid:1",
                devices=("cpu-a", "gpu-a"),
                replacement=("residency:gpu-a",),
            ),
        }
        arguments = {
            "target_artifact_sha256": ARTIFACT_B,
            "target_executor_id": "executor:gpu-b",
            "target_endpoint": "http://gpu-b.invalid:1",
            "target_replacement_resource_ids": ("residency:gpu-a",),
            "target_session_resource_ids": (),
        }
        with self.assertRaisesRegex(
            PhysicalAdapterError, "lacks an exact eviction"
        ):
            physical_transition_stop_set(live, evictions=(), **arguments)

        eviction = RuntimeResidencyEviction(
            model_id="model-a",
            artifact_sha256=ARTIFACT_A,
            device_id="gpu-a",
            resident_bytes=1,
            generation=7,
            executor_id="executor:gpu-a",
            replacement_group="residency:gpu-a",
        )
        self.assertEqual(
            physical_transition_stop_set(
                live, evictions=(eviction,), **arguments
            ),
            ("executor:gpu-a",),
        )
        with self.assertRaisesRegex(
            PhysicalAdapterError, "differs from physical endpoint"
        ):
            physical_transition_stop_set(
                live,
                evictions=(
                    RuntimeResidencyEviction(
                        model_id="model-a",
                        artifact_sha256=ARTIFACT_A,
                        device_id="gpu-a",
                        resident_bytes=1,
                        generation=6,
                        executor_id="executor:gpu-a",
                        replacement_group="residency:gpu-a",
                    ),
                ),
                **arguments,
            )

    def test_phone_replacement_uses_exact_physical_session(self) -> None:
        session = PhysicalPhoneSessionEndpoint(
            session_id="HTP2",
            executor_id="phone-helper",
            endpoint="session://phone/HTP2",
            artifact_sha256=ARTIFACT_A,
            resident_geometry_sha256="sha256:" + "c" * 64,
            operator_plan_sha256="sha256:" + "d" * 64,
            session_generation=3,
            device_id="phone",
            resident_bytes=100,
        )
        eviction = RuntimeResidencyEviction(
            model_id="model-a",
            artifact_sha256=ARTIFACT_A,
            device_id="phone",
            resident_bytes=100,
            generation=3,
            executor_id="phone-helper",
            session_id="HTP2",
            resident_geometry_sha256=session.resident_geometry_sha256,
            operator_plan_sha256=session.operator_plan_sha256,
        )
        arguments = {
            "target_artifact_sha256": ARTIFACT_B,
            "target_executor_id": "phone-helper",
            "target_endpoint": "http://phone-helper.invalid:1",
            "target_replacement_resource_ids": (),
            "target_session_resource_ids": (),
            "phone_sessions": {"HTP2": session},
        }
        self.assertEqual(
            physical_transition_stop_set(
                {}, evictions=(eviction,), **arguments
            ),
            ("phone-helper",),
        )
        with self.assertRaisesRegex(
            PhysicalAdapterError,
            "differs from physical phone session",
        ):
            physical_transition_stop_set(
                {},
                evictions=(replace(eviction, generation=2),),
                **arguments,
            )

    def test_phone_eviction_stops_authoritative_session_endpoint(self) -> None:
        session = PhysicalPhoneSessionEndpoint(
            session_id="HTP0",
            executor_id="physical:session-worker:HTP0",
            endpoint="session://phone/HTP0",
            artifact_sha256=ARTIFACT_A,
            resident_geometry_sha256="sha256:" + "c" * 64,
            operator_plan_sha256="sha256:" + "d" * 64,
            session_generation=5,
            device_id="phone",
            resident_bytes=320,
        )
        eviction = RuntimeResidencyEviction(
            model_id="model-a",
            artifact_sha256=ARTIFACT_A,
            device_id="phone",
            resident_bytes=320,
            generation=5,
            executor_id="physical:logical:phone-assisted",
            session_id="HTP0",
            resident_geometry_sha256=session.resident_geometry_sha256,
            operator_plan_sha256=session.operator_plan_sha256,
        )
        arguments = {
            "target_artifact_sha256": ARTIFACT_B,
            "target_executor_id": "physical:logical:phone-assisted",
            "target_endpoint": "http://phone-helper.invalid:1",
            "target_replacement_resource_ids": (),
            "target_session_resource_ids": (),
            "phone_sessions": {"HTP0": session},
        }

        self.assertEqual(
            physical_transition_stop_set(
                {}, evictions=(eviction,), **arguments
            ),
            ("physical:session-worker:HTP0",),
        )

        mismatches = (
            replace(eviction, session_id="HTP1"),
            replace(eviction, artifact_sha256=ARTIFACT_B),
            replace(eviction, generation=4),
            replace(eviction, device_id="other-phone"),
            replace(eviction, resident_bytes=319),
            replace(
                eviction,
                resident_geometry_sha256="sha256:" + "e" * 64,
            ),
            replace(
                eviction,
                operator_plan_sha256="sha256:" + "f" * 64,
            ),
        )
        for mismatch in mismatches:
            with self.subTest(eviction=mismatch.to_json()):
                with self.assertRaisesRegex(
                    PhysicalAdapterError,
                    "differs from physical phone session",
                ):
                    physical_transition_stop_set(
                        {}, evictions=(mismatch,), **arguments
                    )

    def test_progressive_manifest_keeps_retained_session_port(self) -> None:
        session = object.__new__(DirectPhoneFfnSession)
        session.configuration = SimpleNamespace(
            multi_session_port_base=21_000,
            multi_session_device_count=3,
            model_paths_by_artifact={
                ARTIFACT_A: "/data/qwen.gguf",
                ARTIFACT_B: "/data/gemma.gguf",
            },
        )
        htp2 = self.shard("HTP2", ARTIFACT_A, 4, 100)
        htp0 = self.shard("HTP0", ARTIFACT_B, 1, 120)

        first, _first_hash = session._multi_session_manifest((htp2,))
        second, _second_hash = session._multi_session_manifest((htp0, htp2))
        first_port = first.split(",")[6]
        second_ports = {
            fields[0]: fields[6]
            for fields in (
                row.split(",") for row in second.split(";")
            )
        }

        self.assertEqual(first_port, second_ports["HTP2"])
        self.assertNotEqual(second_ports["HTP0"], second_ports["HTP2"])

    def test_same_executor_replacement_requires_exact_eviction(self) -> None:
        live = {
            "executor:gpu-a": resident(
                "executor:gpu-a",
                ARTIFACT_A,
                7,
                endpoint="http://gpu-a.invalid:1",
                devices=("cpu-a", "gpu-a"),
                replacement=("residency:gpu-a",),
            ),
        }

        with self.assertRaisesRegex(
            PhysicalAdapterError, "lacks an exact eviction"
        ):
            physical_transition_stop_set(
                live,
                target_artifact_sha256=ARTIFACT_B,
                target_executor_id="executor:gpu-a",
                target_endpoint="http://gpu-a.invalid:1",
                target_replacement_resource_ids=("residency:gpu-a",),
                target_session_resource_ids=(),
                evictions=(),
            )


if __name__ == "__main__":
    unittest.main()
