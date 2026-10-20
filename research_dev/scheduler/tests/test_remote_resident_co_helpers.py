"""Quantized remote-resident FFN owners next to a static co-helper phone.

A phone session may own the only copy of Q4_0 / Q8_0 FFN weights (one dtype per group, matching
the manifest and the owner's declared data types), and a second phone may serve other layers as
an ordinary static co-helper of the same server. The co-helper never serves a remote-resident
layer: every overlap fails closed, from the catalog through route generation, the ticket and the
several-helper launch environment, where the remote-resident mask belongs to the owner's client.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from research_dev.scheduler import ModelManifest, RuntimeCapabilityCatalog, UnifiedScheduler  # noqa: E402
from research_dev.scheduler._internal.capability_contracts.common import (  # noqa: E402
    RuntimeCapabilityError,
)
from research_dev.scheduler._internal.lifecycle import UnifiedScheduleError  # noqa: E402
from research_dev.scheduler._internal.phone_shards import (  # noqa: E402
    PhoneFfnResidencyLayout,
    PhoneFfnShardPlacement,
)
from research_dev.scheduler._internal.plan_contracts.co_helpers import (  # noqa: E402
    PHONE_CO_HELPERS_PARAMETER,
    PHONE_HELPERS_PARAMETER,
    RuntimeCoHelperDeclaration,
    RuntimeCoHelperPhone,
    phone_helper_layer_masks,
)
from research_dev.scheduler._internal.route_generation.common import (  # noqa: E402
    RouteGenerationError,
)
from research_dev.scheduler._internal.route_generation.remote_resident import (  # noqa: E402
    REMOTE_RESIDENT_OWNER_NOT_READY,
    remote_resident_co_helper_parameters,
)
from research_dev.scheduler._internal.runtime_capabilities import (  # noqa: E402
    RuntimePhoneSessionCapability,
)
from research_dev.scheduler._internal.runtime_plan import (  # noqa: E402
    RuntimeExecutionContract,
    RuntimeRemoteResidentFfn,
    RuntimeRemoteResidentSession,
    remote_resident_tensor_ids,
)
from research_dev.scheduler._internal.types import canonical_sha256  # noqa: E402
from research_dev.scheduler.adapters import (  # noqa: E402
    ManagedLlamaServer,
    PhysicalAdapterError,
    PhysicalParticipantCommand,
    llama_server_launch_contract,
)
from research_dev.scheduler.adapters.phone_helpers import (  # noqa: E402
    PhoneHelperBinding,
    helper_server_environment,
    helper_transport_contracts,
    remote_resident_layer_mask,
)
from research_dev.scheduler.adapters.ticket import validate_physical_execution_command  # noqa: E402
import test_remote_resident_launch as launch_fixture  # noqa: E402
import test_remote_resident_routes as route_fixture  # noqa: E402
import two_phone_harness as two_phone  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GEMMA_JSON = json.loads(
    (ROOT / "campaigns/burstgpt/data/GEMMA_MANIFEST.json").read_text(encoding="ascii")
)
MASK_0_7 = 0xFF
PIXEL_MASK_8_15 = 0xFF00
PIXEL_TRANSPORT = {
    "adb_port": 5037, "adb_serial": "PIXEL10", "ffn_transport": "adb-tcp",
    "ffn_worker_host": "127.0.0.1", "ffn_worker_port": 40317, "phone_worker_port": 26990,
}
_BLOCKS = {"F16": (1, 2), "Q4_0": (32, 18), "Q8_0": (32, 34), "Q4_K": (256, 144)}


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def quantized_manifest(quantization: str, *, overrides: dict[str, str] | None = None) -> ModelManifest:
    """The Gemma manifest with every dense FFN weight in ``quantization`` (per-tensor overrides)."""
    value = json.loads(json.dumps(GEMMA_JSON))
    value["artifact_sha256"] = _sha("gemma-" + quantization + json.dumps(overrides, sort_keys=True))
    value.pop("tensor_bytes", None)
    ffn_ids = set(remote_resident_tensor_ids((1 << value["block_count"]) - 1))
    for row in value["tensors"]:
        if row["tensor_id"] not in ffn_ids:
            continue
        kind = (overrides or {}).get(row["tensor_id"], quantization)
        block, size = _BLOCKS[kind]
        elements = 1
        for extent in row["shape"]:
            elements *= extent
        row.update(quantization=kind, quantization_block_size=block,
                   quantization_type_size=size, nbytes=elements // block * size)
    return ModelManifest.from_json(value)


def omitted_bytes(manifest: ModelManifest, mask: int = MASK_0_7) -> int:
    return sum(manifest.tensor_by_id[name].nbytes for name in remote_resident_tensor_ids(mask))


def declaration(manifest: ModelManifest, dtype: str, *, omitted: int | None = None) -> dict:
    omitted = omitted_bytes(manifest) if omitted is None else omitted
    payload = RuntimeRemoteResidentFfn(
        parent_artifact_sha256=manifest.artifact_sha256,
        layer_mask=MASK_0_7,
        dtype=dtype,
        omitted_bytes=omitted,
        tensor_ids=remote_resident_tensor_ids(MASK_0_7),
        shard_index_sha256=route_fixture.INDEX_SHA,
        sessions=(RuntimeRemoteResidentSession(
            session_id="HTP0",
            endpoint="session://op15/HTP0",
            layer_mask=MASK_0_7,
            shard_sha256=route_fixture.SHARD_SHA,
            resident_geometry_sha256=route_fixture.GEOMETRY,
            resident_bytes=omitted,
            remote_path="/data/local/tmp/qwen32/HTP0.ffn.gguf",
        ),),
    ).to_json()
    payload.pop("geometry_sha256", None)
    return payload


def co_helpers(*, layer_mask: int = PIXEL_MASK_8_15, device_id: str = "pixel-phone",
               max_tokens: int = 512) -> RuntimeCoHelperDeclaration:
    return RuntimeCoHelperDeclaration(
        primary_label="op15",
        primary_serial="OP15",
        helpers=(RuntimeCoHelperPhone(
            device_id=device_id, serial="PIXEL10", label="pixel", session_id="PIXEL0",
            layer_mask=layer_mask, column_quantum=32, max_tokens=max_tokens,
            shard_sha256=_sha("pixel-shard"), resident_bytes=1 << 20,
            transport_parameters=PIXEL_TRANSPORT,
        ),),
    )


def ready_layout(manifest: ModelManifest, generation: int = 3) -> PhoneFfnResidencyLayout:
    """The routes fixture's READY layout for another artifact."""
    shard = PhoneFfnShardPlacement(
        artifact_sha256=manifest.artifact_sha256,
        session_id="HTP0",
        endpoint="session://op15/HTP0",
        memory_resource_id="op15-ram:session:HTP0",
        operator_ids=tuple(f"layer:{index}:ffn" for index in range(8)),
        layer_mask=MASK_0_7,
        maximum_columns=manifest.feed_forward_length,
        resident_bytes=omitted_bytes(manifest),
        resident_geometry_sha256=route_fixture.GEOMETRY,
        operator_plan_sha256="sha256:" + "7" * 64,
    )
    return PhoneFfnResidencyLayout(
        shards=(shard,),
        queued_work_by_artifact={manifest.artifact_sha256: 1},
        queue_benefit_by_artifact={manifest.artifact_sha256: 1},
        queue_benefit_by_session={"HTP0": 1},
        queue_benefit=1,
        transition_cost=0,
        transition_cost_by_session={},
        objective=-1,
        objective_kind="queue_energy_delta_uj",
        changed_session_ids=(),
        geometry_sha256=canonical_sha256({
            "artifact_sha256": manifest.artifact_sha256,
            "shards": [{"geometry_sha256": shard.resident_geometry_sha256, "session_id": "HTP0"}],
        }),
        session_generation_by_id={} if generation == 0 else {"HTP0": generation},
    )


def compiler(manifest: ModelManifest, dtype: str, *, data_types=("F16", "Q4_0", "Q8_0"),
             omitted: int | None = None, helpers: RuntimeCoHelperDeclaration | None = None,
             **overrides):
    parameters = route_fixture.coordinator_parameters(
        declaration(manifest, dtype, omitted=omitted),
        **{
            "ffn_resident_layer_mask": MASK_0_7,
            **({} if helpers is None else {PHONE_CO_HELPERS_PARAMETER: helpers.encode()}),
            **overrides,
        },
    )
    capability = route_fixture.session_capability(
        supported_data_types=data_types,
        resident_artifact_sha256=manifest.artifact_sha256,
    )
    return route_fixture.Compiler(parameters, sessions=(capability,), layout=ready_layout(manifest))


class QuantizedGroupTests(unittest.TestCase):
    def test_q4_0_and_q8_0_groups_are_accepted(self) -> None:
        for quantization, dtype in (("Q4_0", "q4_0"), ("Q8_0", "q8_0")):
            with self.subTest(dtype=dtype):
                manifest = quantized_manifest(quantization)
                route = compiler(manifest, dtype)
                group = route._remote_resident_group(manifest, route_fixture.pattern())
                self.assertEqual(group.dtype, dtype)
                self.assertEqual(group.omitted_bytes, omitted_bytes(manifest))
                self.assertLess(group.omitted_bytes, route_fixture.omitted_bytes())
                bound, reasons = route._remote_resident_owner_status(manifest, group)
                self.assertEqual(reasons, ())
                self.assertEqual(bound.session_generation_by_id, {"HTP0": 3})
                demand, = route._remote_resident_memory_demands(manifest, group, ())
                self.assertEqual(demand.required_bytes, omitted_bytes(manifest))
                self.assertEqual(len(route._remote_resident_tensor_ids(group)), 24)

    def test_group_round_trips_its_quantized_dtype(self) -> None:
        manifest = quantized_manifest("Q4_0")
        group = RuntimeRemoteResidentFfn.from_json(declaration(manifest, "q4_0"))
        self.assertEqual(group.dtype, "q4_0")
        self.assertEqual(RuntimeRemoteResidentFfn.from_json(group.to_json()), group)
        self.assertNotEqual(
            group.geometry_sha256,
            RuntimeRemoteResidentFfn.from_json(declaration(manifest, "q8_0")).geometry_sha256,
        )

    def test_mixed_or_mismatched_dtypes_fail_closed(self) -> None:
        mixed = quantized_manifest("Q4_0", overrides={"blk.3.ffn_down.weight": "Q8_0"})
        cases = (
            ("differs from the group dtype q4_0: blk.3.ffn_down.weight", mixed, "q4_0"),
            ("differs from the group dtype q8_0", mixed, "q8_0"),
            ("differs from the group dtype f16", quantized_manifest("Q4_0"), "f16"),
            ("unsupported: blk.0.ffn_down.weight Q4_K", quantized_manifest("Q4_K"), "q4_0"),
        )
        for message, manifest, dtype in cases:
            with self.subTest(message=message):
                route = compiler(manifest, dtype)
                with self.assertRaisesRegex(RouteGenerationError, message):
                    route._remote_resident_group(manifest, route_fixture.pattern())

    def test_omitted_bytes_are_the_quantized_manifest_bytes(self) -> None:
        manifest = quantized_manifest("Q4_0")
        f16_bytes = omitted_bytes(quantized_manifest("F16"))
        route = compiler(manifest, "q4_0", omitted=f16_bytes)
        with self.assertRaisesRegex(RouteGenerationError, "omitted bytes differ"):
            route._remote_resident_group(manifest, route_fixture.pattern())

    def test_owner_without_the_quantized_type_is_rejected(self) -> None:
        manifest = quantized_manifest("Q4_0")
        for data_types in (("F16",), ("F16", "Q8_0"), ("q4_0",)):
            with self.subTest(data_types=data_types):
                route = compiler(manifest, "q4_0", data_types=data_types)
                with self.assertRaisesRegex(RouteGenerationError, "does not serve Q4_0: HTP0"):
                    route._remote_resident_group(manifest, route_fixture.pattern())

    def test_existing_gates_still_apply_to_quantized_groups(self) -> None:
        manifest = quantized_manifest("Q4_0")
        for message, overrides in (
            ("resident layer mask", {"ffn_resident_layer_mask": 0x0F}),
            ("complete FFN columns", {"ffn_resident_columns": 8192}),
            ("ffn_max_tokens", {"ffn_max_tokens": 64}),
        ):
            with self.subTest(message=message):
                route = compiler(manifest, "q4_0", **overrides)
                with self.assertRaisesRegex(RouteGenerationError, message):
                    route._remote_resident_group(manifest, route_fixture.pattern())
        route = compiler(manifest, "q4_0")
        route._phone_residency_layout = ready_layout(manifest, generation=0)
        group = route._remote_resident_group(manifest, route_fixture.pattern())
        bound, reasons = route._remote_resident_owner_status(manifest, group)
        self.assertIsNone(bound)
        self.assertTrue(reasons[0].startswith(REMOTE_RESIDENT_OWNER_NOT_READY + ":HTP0"))


class CoHelperRouteTests(unittest.TestCase):
    def test_disjoint_co_helper_on_a_second_phone_is_accepted(self) -> None:
        manifest = quantized_manifest("Q4_0")
        route = compiler(manifest, "q4_0", helpers=co_helpers())
        group = route._remote_resident_group(manifest, route_fixture.pattern())
        self.assertEqual(group.layer_mask, MASK_0_7)
        coordinator = next(iter(route.catalog.composite_executor_by_id.values()))
        parameters = dict(coordinator.adapter_parameters)
        remote_resident_co_helper_parameters(parameters)
        self.assertEqual(parameters["ffn_resident_layer_mask"], MASK_0_7 | PIXEL_MASK_8_15)
        self.assertEqual(dict(phone_helper_layer_masks(parameters[PHONE_HELPERS_PARAMETER])),
                         {"op15-phone": MASK_0_7, "pixel-phone": PIXEL_MASK_8_15})
        plain = dict(compiler(manifest, "q4_0").catalog.composite_executor_by_id[
            coordinator.executor_id].adapter_parameters)
        before = dict(plain)
        remote_resident_co_helper_parameters(plain)
        self.assertEqual(plain, before, "no co-helper declaration changes nothing")

    def test_co_helpers_serving_remote_or_owner_layers_fail_closed(self) -> None:
        manifest = quantized_manifest("Q4_0")
        for message, helpers, overrides in (
            ("overlaps remote-resident layers", co_helpers(layer_mask=0x180), {}),
            ("owner phone's resident layers", co_helpers(), {"ffn_resident_layer_mask": 0xFFF}),
            ("cannot also be a static co-helper", co_helpers(device_id="op15-phone"), {}),
            ("batch capacity", co_helpers(max_tokens=4), {}),
            ("block count", co_helpers(layer_mask=0xFF << 48), {}),
        ):
            with self.subTest(message=message):
                route = compiler(manifest, "q4_0", helpers=helpers, **overrides)
                with self.assertRaisesRegex(RouteGenerationError, message):
                    route._remote_resident_group(manifest, route_fixture.pattern())


class RemoteOwnerWithCoHelperCatalogTests(unittest.TestCase):
    """Two-phone harness (Q8_0 FFN weights): OP15 owns layers 0-3, the Pixel serves 4-5."""

    REMOTE = 0b1111

    def setUp(self) -> None:
        scope = two_phone.TemporaryModel()
        self.model = scope.__enter__()
        self.addCleanup(scope.__exit__)
        self.base = two_phone.runtime_catalog(self.model, declaration=two_phone.co_helpers())

    def remote_catalog(self, *, remote_mask: int = REMOTE, owner_as_co_helper: bool = False):
        model, catalog = self.model, self.base
        omitted = omitted_bytes(model, remote_mask)
        session = RuntimePhoneSessionCapability(
            session_id="HTP0", device_id=two_phone.OP15, endpoint="session://op15-phone/HTP0",
            worker_identity_sha256=_sha("worker"), memory_resource_id="op15-ram",
            resident_memory_limit_bytes=1 << 30, shared_compute_resource_id="op15-htp",
            shared_transport_resource_ids=("op15-functionfs", "usb-root"), supported_layer_mask=0xFF,
            maximum_columns=model.feed_forward_length, column_quantum=16,
            supported_data_types=("F16", "Q8_0"), batch_plans=("split-row",), ready=True,
            residency_state="cold",
        )
        group = RuntimeRemoteResidentFfn(
            parent_artifact_sha256=model.artifact_sha256, layer_mask=remote_mask, dtype="q8_0",
            omitted_bytes=omitted, tensor_ids=remote_resident_tensor_ids(remote_mask),
            shard_index_sha256=_sha("index"),
            sessions=(RuntimeRemoteResidentSession(
                session_id="HTP0", endpoint=session.endpoint, layer_mask=remote_mask,
                shard_sha256=_sha("shard"), resident_geometry_sha256=_sha("geometry"),
                resident_bytes=omitted, remote_path="/data/local/tmp/q8/HTP0.ffn.gguf",
            ),),
        )
        desktop = catalog.composite_executor_by_id[catalog.desktop_control_profiles[0].executor_id]
        helper = catalog.composite_executor_by_id["physical:two:phone-assisted:operator_split"]
        parameters = {
            **helper.adapter_parameters, **desktop.adapter_parameters,
            "remote_resident_ffn_v1": json.dumps(group.to_json(), sort_keys=True),
            "ffn_assistance_phase": "decode", "ffn_max_tokens": desktop.adapter_parameters["ubatch_size"],
            "ffn_resident_layer_mask": remote_mask, "ffn_resident_columns": model.feed_forward_length,
            "ffn_weight_buffer_layout": "resident-superset", "ffn_resident_weight_bytes": omitted,
            "resident_model_identity_sha256": model.artifact_sha256,
        }
        if owner_as_co_helper:
            declared = two_phone.co_helpers()
            parameters[PHONE_CO_HELPERS_PARAMETER] = replace(declared, helpers=(replace(
                declared.helpers[0], device_id=two_phone.OP15),)).encode()
        phones = (two_phone.OP15,) if owner_as_co_helper else (two_phone.OP15, two_phone.PIXEL)
        remote_id = desktop.executor_id + ":remote-owner"
        remote = replace(
            desktop, executor_id=remote_id, maturity="CALIBRATION_PENDING", adapter_parameters=parameters,
            participant_device_ids=(*desktop.participant_device_ids, *phones),
            participant_resource_ids={**desktop.participant_resource_ids,
                                      **{phone: helper.participant_resource_ids[phone] for phone in phones}},
            resource_ids=tuple(sorted(set(desktop.resource_ids) | {
                resource for phone in phones for resource in helper.participant_resource_ids[phone]})),
        )
        transitions = tuple(
            replace(row, transition_id=row.transition_id + ":remote-owner", executor_id=remote_id)
            for row in catalog.transitions if row.executor_id == desktop.executor_id
        )
        value = replace(
            catalog,
            executors=tuple(replace(row, phone_sessions=(session,)) if row.device_id == two_phone.OP15
                            else row for row in catalog.executors),
            composite_executors=(*catalog.composite_executors, remote),
            transitions=(*catalog.transitions, *transitions),
        )
        return RuntimeCapabilityCatalog.from_json(value.to_json()), remote

    def candidates(self, catalog):
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(catalog)
        scheduler.register_model_manifest(self.model)
        return scheduler.generate_automated_candidates(
            two_phone.request("remote-owner"), self.model.model_id,
            two_phone.snapshot(self.model, catalog),
        )

    def test_generated_route_carries_the_owner_and_the_co_helper(self) -> None:
        catalog, remote = self.remote_catalog()
        candidate, = (row for row in self.candidates(catalog).candidates
                      if row.binding.executor_id == remote.executor_id)
        plan, parameters = candidate.plan, candidate.plan.adapter_parameters
        self.assertEqual(plan.device_ids, (two_phone.CPU, two_phone.GPU, two_phone.OP15, two_phone.PIXEL))
        self.assertEqual(plan.execution_contract.execution_mode, "desktop")
        self.assertEqual(plan.execution_contract.remote_resident_ffn.dtype, "q8_0")
        self.assertEqual(parameters["ffn_resident_layer_mask"], self.REMOTE | two_phone.PIXEL_MASK)
        self.assertEqual(dict(phone_helper_layer_masks(parameters[PHONE_HELPERS_PARAMETER])),
                         {two_phone.OP15: self.REMOTE, two_phone.PIXEL: two_phone.PIXEL_MASK})
        self.assertTrue({"pixel-adb", "pixel-gpu"} <= set(plan.resource_ids))
        # only owner readiness and the (unqualified) synthetic transport hold this candidate back
        self.assertEqual(set(candidate.rejection_reasons), {
            REMOTE_RESIDENT_OWNER_NOT_READY + ":HTP0", "ROUTE_NOT_QUALIFIED", "TRANSPORT_PROFILE_INCOMPLETE",
        })

    def test_catalog_refuses_co_helpers_on_remote_or_owner_layers(self) -> None:
        with self.assertRaisesRegex(RuntimeCapabilityError, "overlaps remote-resident layers"):
            self.remote_catalog(remote_mask=0b111111)
        with self.assertRaisesRegex(RuntimeCapabilityError, "co-helper phones"):
            self.remote_catalog(owner_as_co_helper=True)

    def test_route_generation_refuses_a_co_helper_on_owner_resident_layers(self) -> None:
        catalog, remote = self.remote_catalog()
        widened = replace(remote, adapter_parameters={
            **remote.adapter_parameters, "ffn_resident_layer_mask": 0b111111})
        catalog = replace(catalog, composite_executors=tuple(
            widened if row.executor_id == remote.executor_id else row for row in catalog.composite_executors))
        with self.assertRaisesRegex(UnifiedScheduleError, "owner phone's resident layers"):
            self.candidates(catalog)


def binding(owner_mask: int = MASK_0_7, pixel_mask: int = PIXEL_MASK_8_15) -> str:
    return json.dumps([
        {"device_id": "op15-phone", "label": "op15", "layer_mask": owner_mask, "serial": "OP15",
         "transport_parameters": {}},
        {"device_id": "pixel-phone", "label": "pixel", "layer_mask": pixel_mask, "serial": "PIXEL10",
         "transport_parameters": PIXEL_TRANSPORT},
    ], sort_keys=True, separators=(",", ":"))


def two_phone_command(*, helpers: str | None = None, resident_mask: int = MASK_0_7 | PIXEL_MASK_8_15,
                      pixel_participant: bool = True):
    parameters = launch_fixture.parameters(
        ffn_resident_layer_mask=resident_mask,
        **{PHONE_HELPERS_PARAMETER: binding() if helpers is None else helpers},
    )
    command = launch_fixture.command(
        RuntimeExecutionContract.desktop(launch_fixture.group()), parameters
    )
    if pixel_participant:
        command = replace(command, participants=(*command.participants, PhysicalParticipantCommand(
            executor_id="physical:hot:desktop-remote", device_id="pixel-phone",
            endpoint="http://127.0.0.1:18573", backend="llama-server-cuda-cpu",
            resource_ids=("pixel-adb",),
        )))
    return command


class RemoteOwnerWithCoHelperLaunchTests(unittest.TestCase):
    def test_owner_client_alone_carries_the_remote_resident_mask(self) -> None:
        command = two_phone_command()
        validate_physical_execution_command(command)
        contract = llama_server_launch_contract(command, launch_fixture.MANIFEST)
        environment = contract.ffn_environment
        self.assertEqual(contract.phone_device_id, "op15-phone")
        self.assertEqual(environment["S41_SERVER_FFN_HELPERS"], "2")
        self.assertEqual(environment["S41_SERVER_FFN_LAYER_MASK"], str(MASK_0_7 | PIXEL_MASK_8_15))
        self.assertEqual(environment["S41_SERVER_FFN_HELPER0_LAYER_MASK"], str(MASK_0_7))
        self.assertEqual(environment["S41_SERVER_FFN_HELPER0_REMOTE_RESIDENT_LAYER_MASK"], str(MASK_0_7))
        self.assertEqual(environment["S41_SERVER_FFN_HELPER0_TRANSPORT"], "functionfs-usb")
        self.assertEqual(environment["S41_SERVER_FFN_HELPER1_LAYER_MASK"], str(PIXEL_MASK_8_15))
        self.assertEqual(environment["S41_SERVER_FFN_HELPER1_TRANSPORT"], "tcp")
        self.assertEqual(environment["S41_SERVER_FFN_HELPER1_PORT"], "40317")
        self.assertNotIn("S41_SERVER_FFN_HELPER1_REMOTE_RESIDENT_LAYER_MASK", environment)
        self.assertNotIn("S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK", environment)
        self.assertNotIn("S41_SERVER_FFN_TRANSPORT", environment)
        self.assertEqual(environment["S41_SERVER_FFN_MAX_TOKENS"], "512")
        self.assertEqual(environment["S41_SERVER_FFN_RUNTIME_CONTROL"], "1")
        self.assertEqual(remote_resident_layer_mask(environment), MASK_0_7)

    def test_remote_parent_without_co_helpers_keeps_its_environment(self) -> None:
        plain = launch_fixture.command(
            RuntimeExecutionContract.desktop(launch_fixture.group()), launch_fixture.parameters()
        )
        environment = llama_server_launch_contract(plain, launch_fixture.MANIFEST).ffn_environment
        self.assertEqual(environment["S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK"], str(MASK_0_7))
        self.assertNotIn("S41_SERVER_FFN_HELPERS", environment)
        self.assertEqual(remote_resident_layer_mask(environment), MASK_0_7)

    def test_co_helper_on_remote_layers_fails_closed(self) -> None:
        overlapping = binding(owner_mask=0x7F, pixel_mask=0xFF80)
        command = two_phone_command(helpers=overlapping, resident_mask=0xFFFF)
        with self.assertRaisesRegex(PhysicalAdapterError, "co-helper phone overlaps remote-resident"):
            validate_physical_execution_command(command)
        with self.assertRaisesRegex(PhysicalAdapterError, "offered to another phone helper"):
            llama_server_launch_contract(command, launch_fixture.MANIFEST)

    def test_binding_must_match_the_ticket(self) -> None:
        for label, command in (
            ("pixel not a participant", two_phone_command(pixel_participant=False)),
            ("union differs", two_phone_command(resident_mask=0xFFFFF)),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(PhysicalAdapterError, "differs from the remote-resident ticket"):
                    validate_physical_execution_command(command)
        with self.assertRaisesRegex(PhysicalAdapterError, "differs from the server layer mask"):
            llama_server_launch_contract(two_phone_command(resident_mask=0xFFFFF), launch_fixture.MANIFEST)

    def test_owner_only_execution_proof_still_binds_the_owner_session(self) -> None:
        command = two_phone_command()
        contract = llama_server_launch_contract(command, launch_fixture.MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            server = ManagedLlamaServer(("server",), {}, Path(directory), "reduced", contract)
            server.process = SimpleNamespace(poll=lambda: None)
            marker = server.begin_execution(command, launch_fixture.MANIFEST)
            self.assertEqual(marker.phone_contract.layer_mask, MASK_0_7)
            for token in range(2):
                for layer in range(8):
                    server.stderr_lines.append(
                        f"S41SERVERFFNCALL request={1 + token * 8 + layer} layer={layer} "
                        f"tokens=1 columns={launch_fixture.MANIFEST.feed_forward_length} "
                        f"payload_bytes={launch_fixture.MANIFEST.embedding_length * 2}"
                    )
            proof = server.finish_execution(marker, command, launch_fixture.MANIFEST, output_tokens=2)
            self.assertEqual(proof.phone_call_count, 16)
            shard, = proof.phone_calls_by_session
            self.assertEqual(shard.layer_mask, MASK_0_7)


class HelperEnvironmentTests(unittest.TestCase):
    OWNER = PhoneHelperBinding(device_id="op15-phone", serial="OP15", layer_mask=MASK_0_7, label="op15")
    PIXEL = PhoneHelperBinding(device_id="pixel-phone", serial="PIXEL10", layer_mask=PIXEL_MASK_8_15,
                               label="pixel", transport_parameters=PIXEL_TRANSPORT)
    SHARED = {
        "S41_SERVER_FFN_ARTIFACT_SHA256": "sha256:" + "a" * 64,
        "S41_SERVER_FFN_LAYER_MASK": str(MASK_0_7 | PIXEL_MASK_8_15),
        "S41_SERVER_FFN_MAX_TOKENS": "4",
        "S41_SERVER_FFN_N_EMBD": "64",
        "S41_SERVER_FFN_RUNTIME_CONTROL": "1",
    }

    def environment(self, rows, remote: str | None):
        shared = dict(self.SHARED)
        if remote is not None:
            shared["S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK"] = remote
        if len(rows) == 1:
            shared["S41_SERVER_FFN_LAYER_MASK"] = str(rows[0].layer_mask)
        contracts = helper_transport_contracts(rows, launch_fixture.parameters())
        return dict(helper_server_environment(rows, shared, contracts))

    def test_remote_mask_moves_to_its_owner_only(self) -> None:
        environment = self.environment((self.OWNER, self.PIXEL), "3")
        self.assertEqual(environment["S41_SERVER_FFN_HELPER0_REMOTE_RESIDENT_LAYER_MASK"], "3")
        self.assertNotIn("S41_SERVER_FFN_HELPER1_REMOTE_RESIDENT_LAYER_MASK", environment)
        self.assertNotIn("S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK", environment)
        self.assertEqual(remote_resident_layer_mask(environment), 3)
        self.assertNotIn("REMOTE_RESIDENT", " ".join(self.environment((self.OWNER, self.PIXEL), None)))
        # the owning client, wherever it sits in helper order
        tcp_owner = replace(self.OWNER, transport_parameters={
            **PIXEL_TRANSPORT, "adb_serial": "OP15", "ffn_worker_port": 40318})
        swapped = self.environment((replace(self.PIXEL, transport_parameters={}), tcp_owner), "3")
        self.assertEqual(swapped["S41_SERVER_FFN_HELPER1_REMOTE_RESIDENT_LAYER_MASK"], "3")
        self.assertNotIn("S41_SERVER_FFN_HELPER0_REMOTE_RESIDENT_LAYER_MASK", swapped)
        single = self.environment((self.OWNER,), str(MASK_0_7))
        self.assertEqual(single["S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK"], str(MASK_0_7))
        self.assertNotIn("S41_SERVER_FFN_HELPERS", single)

    def test_remote_mask_spanning_or_outside_helpers_fails_closed(self) -> None:
        fused_owner = replace(self.OWNER, fused_layer_mask=1, fused_n_q=64, fused_n_kv=32)
        for rows, remote in (
            ((self.OWNER, self.PIXEL), str(0x180)),
            ((self.OWNER, self.PIXEL), str(1 << 20)),
            ((self.OWNER, self.PIXEL), "0"),
            ((self.OWNER, self.PIXEL), "0255"),
            ((fused_owner, self.PIXEL), "1"),
        ):
            with self.subTest(remote=remote), self.assertRaisesRegex(PhysicalAdapterError, "exactly one"):
                self.environment(rows, remote)

    def test_environment_mask_reader_fails_closed(self) -> None:
        self.assertEqual(remote_resident_layer_mask({}), 0)
        self.assertEqual(remote_resident_layer_mask({"S41_SERVER_FFN_HELPERS": "2"}), 0)
        for environment in (
            {"S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK": "x"},
            {"S41_SERVER_FFN_HELPER0_REMOTE_RESIDENT_LAYER_MASK": "01"},
            {"S41_SERVER_FFN_HELPER0_REMOTE_RESIDENT_LAYER_MASK": "3",
             "S41_SERVER_FFN_HELPER1_REMOTE_RESIDENT_LAYER_MASK": "2"},
        ):
            with self.subTest(environment=environment), self.assertRaises(PhysicalAdapterError):
                remote_resident_layer_mask(environment)


if __name__ == "__main__":
    unittest.main()
