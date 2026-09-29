"""Two-phone Stage A (OP15 scheduled + Pixel static co-helper): gaps 1-3 end to end, gap 4 hook.

Gap 1: catalog materialization and route generation bind the co-helper as a participant with its
own executor, resources and operator assignments. Gap 2: ``phone_helpers`` flows through the
dormant desktop contract, residency matching and ticket validation into the launch environment.
Gap 3: adaptive policies use the union mask on the co-helper's column grid, carry per-device masks,
and proofs account every co-helper call to its own shard and request-id range. Nothing here runs a
phone; the executors are fakes.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from research_dev.scheduler import RuntimeCapabilityCatalog, UnifiedScheduler
from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeError,
    AdaptiveDecodePolicy,
)
from research_dev.scheduler._internal.adaptive_decode_planning import adaptive_decode_policies
from research_dev.scheduler._internal.capability_contracts.common import RuntimeCapabilityError
from research_dev.scheduler._internal.plan_contracts.common import RuntimePlanError
from research_dev.scheduler._internal.plan_contracts.phone import RuntimePhoneShard
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler._unified.automated_selection_ops.attachment import (
    _dormant_phone_ffn_runtime_supports,
)
from research_dev.scheduler._unified.automated_selection_ops.dormant import (
    _dormant_phone_ffn_storage_superset,
)
from research_dev.scheduler.adapters import (
    CatalogMaterializationError,
    interpret_runtime_ticket,
    llama_server_launch_contract,
)
from research_dev.scheduler.adapters.contracts import (
    DORMANT_PHONE_FFN_RUNTIME_PARAMETER,
    PhysicalAdapterError,
    dormant_phone_ffn_parameters,
)
from research_dev.scheduler.adapters.llama_server_contracts import LlamaServerFfnCall
from research_dev.scheduler.adapters.llama_server_ops.proofs import ManagedServerProofMixin
from research_dev.scheduler.adapters.residency import physical_residency_parameters_match
from research_dev.scheduler.adapters.ticket import validate_physical_execution_command
from research_dev.scheduler.campaigns.burstgpt import catalog as campaign_catalog
from research_dev.scheduler.configuration.rig import RigManifest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import two_phone_harness as h  # noqa: E402
from test_two_phone_helpers import two_phone_rig_json  # noqa: E402

# literal on purpose: the module imports on a tree without co-helper support, so every gap test
# fails there on its own behaviour rather than at import
PHONE_CO_HELPERS_PARAMETER = "phone_co_helpers_v1"
PHONE_HELPERS_PARAMETER = "phone_helpers"


def _co_helpers():
    from research_dev.scheduler._internal.plan_contracts import co_helpers

    return co_helpers


def co_helper_declaration(parameters):
    return _co_helpers().co_helper_declaration(parameters)


def phone_helper_layer_masks(value):
    return _co_helpers().phone_helper_layer_masks(value)


def phone_helpers_support(resident, requested):
    return _co_helpers().phone_helpers_support(resident, requested)


def RuntimeCoHelperDeclaration(*arguments):
    return _co_helpers().RuntimeCoHelperDeclaration(*arguments)


def CoHelperLifecycle(*arguments):
    from research_dev.scheduler.adapters.co_helper_lifecycle import CoHelperLifecycle

    return CoHelperLifecycle(*arguments)


def _submit(catalog, model, snapshot, request_id: str, mode: str):
    scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
    scheduler.register_runtime_capabilities(catalog)
    scheduler.register_model_manifest(model)
    ticket = scheduler.submit_automated_request(
        h.request(request_id), model.model_id, snapshot, selection_mode=mode
    )
    return scheduler.wait_runtime_request(
        request_id, time.monotonic_ns() - ticket.decision.start_us * 1_000
    )


def _dormant(mask: int, helpers: str | None, **overrides) -> str:
    contract = {
        **h.FUNCTIONFS_PARAMETERS,
        "ffn_activation": "swiglu", "ffn_assistance_phase": "decode", "ffn_max_tokens": 4,
        "ffn_n_embd": 32, "ffn_resident_columns": 128, "ffn_resident_layer_mask": mask,
        "ffn_runtime_control_protocol": "decode-boundary-v1", "ffn_timeout_ms": 1000,
        "phone_device_id": h.OP15,
        **({} if helpers is None else {PHONE_HELPERS_PARAMETER: helpers}),
        **overrides,
    }
    return json.dumps(contract, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


class CoHelperContractTests(unittest.TestCase):
    def test_declaration_is_canonical_and_widths_use_the_common_quantum(self) -> None:
        declaration = h.co_helpers()
        self.assertEqual(declaration.encode(), h.co_helpers_json())
        self.assertEqual(declaration.phone_helpers(h.OP15, 0b1111), h.phone_helpers_json(0b1111))
        parameters = {PHONE_CO_HELPERS_PARAMETER: declaration.encode()}
        self.assertEqual(co_helper_declaration(parameters), declaration)
        self.assertEqual(declaration.column_quantum(16), 32)
        # the real rig: OP15's Qwen quantum 2176 and the Pixel's 4352 give 25 % steps only
        self.assertEqual(h.co_helpers(column_quantum=4352).column_quantum(2176), 4352)
        with self.assertRaises(RuntimePlanError):
            co_helper_declaration({PHONE_CO_HELPERS_PARAMETER: json.dumps(declaration.to_json())})
        self.assertIsNone(co_helper_declaration({}))

    def test_declaration_rejects_foreign_forwards_and_shared_identities(self) -> None:
        row = h.co_helpers().helpers[0]
        for change in (
            {"transport_parameters": {**h.PIXEL_TRANSPORT, "adb_serial": h.OP15_SERIAL}},
            {"transport_parameters": {**h.PIXEL_TRANSPORT, "ffn_transport": "tcp"}},
            {"transport_parameters": {**h.PIXEL_TRANSPORT, "ffn_worker_port": 0}},
            {"column_quantum": 48},
        ):
            with self.subTest(change=change), self.assertRaises(RuntimePlanError):
                replace(row, **change)
        with self.assertRaises(RuntimePlanError):
            RuntimeCoHelperDeclaration("op15", h.PIXEL_SERIAL, (row,))
        with self.assertRaises(RuntimePlanError):
            RuntimeCoHelperDeclaration("op15", h.OP15_SERIAL, (row, replace(row, device_id="other")))

    def test_launch_binding_and_live_server_superset(self) -> None:
        declaration = h.co_helpers()
        resident = declaration.phone_helpers(h.OP15, 0b1111)
        self.assertEqual(dict(phone_helper_layer_masks(resident)), {h.OP15: 0b1111, h.PIXEL: h.PIXEL_MASK})
        self.assertTrue(phone_helpers_support(resident, declaration.phone_helpers(h.OP15, 0b0011)))
        self.assertFalse(phone_helpers_support(resident, declaration.phone_helpers(h.OP15, 1 << 6)))
        self.assertTrue(phone_helpers_support(
            resident, h.co_helpers(layer_mask=1 << 4).phone_helpers(h.OP15, 0b1111)))
        moved = h.co_helpers(layer_mask=1 << 6).phone_helpers(h.OP15, 0b1111)
        self.assertFalse(phone_helpers_support(resident, moved))
        other_forward = RuntimeCoHelperDeclaration("op15", h.OP15_SERIAL, (replace(
            declaration.helpers[0], transport_parameters={**h.PIXEL_TRANSPORT, "ffn_worker_port": 26992}),))
        self.assertFalse(phone_helpers_support(resident, other_forward.phone_helpers(h.OP15, 0b0011)))
        self.assertFalse(phone_helpers_support(resident, None))
        with self.assertRaises(RuntimePlanError):
            declaration.phone_helpers(h.OP15, h.PIXEL_MASK)


class CatalogCoHelperTests(unittest.TestCase):
    """Gap 1: the catalog marks every capability by device."""

    def setUp(self) -> None:
        self.model_scope = h.TemporaryModel()
        self.model = self.model_scope.__enter__()
        self.addCleanup(self.model_scope.__exit__)

    def test_phone_families_carry_both_phones_with_device_marked_capabilities(self) -> None:
        catalog = h.runtime_catalog(self.model, declaration=h.co_helpers())
        pixel_ffn = {row.operator_id for row in self.model.operators
                     if row.kind == "ffn" and row.layer_id in {"layer:4", "layer:5"}}
        assisted = [row for row in catalog.composite_executors if row.assisted_operator_kind == "ffn"]
        self.assertEqual({row.route_family for row in assisted}, {"operator_offload", "operator_split"})
        for row in assisted:
            self.assertEqual(row.participant_device_ids, (h.CPU, h.GPU, h.OP15, h.PIXEL))
            self.assertEqual(row.helper_device_id, h.OP15)
            self.assertEqual(row.participant_resource_ids[h.PIXEL], ("usb-root", "pixel-adb", "pixel-gpu"))
            self.assertTrue({"pixel-adb", "pixel-gpu"} <= set(row.resource_ids))
            self.assertEqual(co_helper_declaration(row.adapter_parameters), h.co_helpers())
            self.assertEqual(row.adapter_parameters["ffn_column_quantum"], 32)
            self.assertFalse(pixel_ffn & set(row.operator_ids))
            self.assertEqual(len(row.operator_ids), 6)
        pixel = catalog.executor_by_device[h.PIXEL]
        self.assertEqual((pixel.executor_id, pixel.backend, pixel.maturity), ("physical:pixel-phone", "phone", "QUALIFIED"))
        self.assertEqual(set(pixel.kernel_profiles), {"ffn"})
        self.assertEqual(pixel.memory_resource_id, "pixel-ram")
        self.assertFalse(pixel.supports_operator_placement or pixel.supports_whole_model
                         or pixel.supports_layer_placement or pixel.phone_sessions)
        self.assertEqual(catalog.resources["pixel-gpu"].kind, "compute")
        self.assertEqual(catalog.executor_by_device[h.OP15].phone_sessions, ())
        # a static co-helper is never prepared by a model transition
        self.assertTrue(all(h.PIXEL not in row.prepares_device_ids for row in catalog.transitions))
        self.assertEqual(RuntimeCapabilityCatalog.from_json(catalog.to_json()).to_json(), catalog.to_json())
        unqualified = h.runtime_catalog(self.model, declaration=h.co_helpers(), qualified_pixel=False)
        self.assertEqual(unqualified.executor_by_device[h.PIXEL].maturity, "SHADOW")

    def test_co_helper_layers_must_be_cpu_parent_layers(self) -> None:
        with self.assertRaisesRegex(CatalogMaterializationError, "CPU-parent layers"):
            h.runtime_catalog(self.model, declaration=h.co_helpers(layer_mask=0b11 << 5))

    def test_catalog_rejects_a_co_helper_that_is_not_a_phone_participant(self) -> None:
        catalog = h.runtime_catalog(self.model, declaration=h.co_helpers())
        value = catalog.to_json()
        for row in value["composite_executors"]:
            if row["executor_id"].endswith(":operator_split"):
                row["participant_device_ids"].remove(h.PIXEL)
                del row["participant_resource_ids"][h.PIXEL]
        with self.assertRaisesRegex(RuntimeCapabilityError, "co-helper phones"):
            RuntimeCapabilityCatalog.from_json(value)

    def test_single_phone_catalog_carries_no_co_helper_state(self) -> None:
        catalog = h.runtime_catalog(self.model)
        self.assertNotIn(h.PIXEL, catalog.executor_by_device)
        self.assertTrue(all(PHONE_CO_HELPERS_PARAMETER not in row.adapter_parameters
                            for row in catalog.composite_executors))


class TwoPhoneRouteTests(unittest.TestCase):
    """Gaps 1 and 3: generated plans and adaptive policies over both phones."""

    def setUp(self) -> None:
        self.model_scope = h.TemporaryModel()
        self.model = self.model_scope.__enter__()
        self.addCleanup(self.model_scope.__exit__)
        self.catalog = h.runtime_catalog(self.model, declaration=h.co_helpers())
        self.snapshot = h.snapshot(self.model, self.catalog)

    def _candidates(self, catalog=None, snapshot=None):
        catalog = self.catalog if catalog is None else catalog
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(catalog)
        scheduler.register_model_manifest(self.model)
        return scheduler.generate_automated_candidates(
            h.request("routes"), self.model.model_id,
            h.snapshot(self.model, catalog) if snapshot is None else snapshot,
        )

    def _split(self, values):
        return next(row for row in values.candidates
                    if row.binding.executor_id.endswith(":operator_split"))

    def test_assisted_plan_assigns_co_helper_layers_to_their_phone(self) -> None:
        split = self._split(self._candidates())
        plan, parameters = split.plan, split.plan.adapter_parameters
        # the desktop parent is cold; the only reason is the adaptive-tolerated break-even
        self.assertEqual(split.rejection_reasons, ("COLD_RESIDENCY_BREAK_EVEN",))
        self.assertEqual(plan.device_ids, (h.CPU, h.GPU, h.OP15, h.PIXEL))
        devices = {
            int(row.operator_id.split(":")[1]): row.device_ids
            for row in plan.operators if row.operator_kind == "ffn"
        }
        self.assertEqual({layer: devices[layer] for layer in range(6)},
                         {**{layer: (h.CPU, h.OP15) for layer in range(4)}, 4: (h.CPU, h.PIXEL), 5: (h.CPU, h.PIXEL)})
        self.assertEqual({devices[6], devices[7]}, {(h.GPU,)})
        self.assertEqual(parameters["ffn_resident_layer_mask"], 0b111111)
        self.assertEqual(dict(phone_helper_layer_masks(parameters[PHONE_HELPERS_PARAMETER])),
                         {h.OP15: 0b1111, h.PIXEL: h.PIXEL_MASK})
        self.assertEqual(parameters["ffn_column_quantum"], 32)
        self.assertEqual(parameters["phone_device_id"], h.OP15)
        self.assertEqual(parameters["ffn_transport"], "functionfs-usb")
        self.assertTrue({"pixel-adb", "pixel-gpu"} <= set(plan.resource_ids))
        self.assertEqual(plan.execution_contract.phone_device_id, h.OP15)
        self.assertEqual(plan.execution_contract.execution_mode, "adaptive-split")
        self.assertEqual(plan.execution_contract.allowed_adaptive_fractions_ppm,
                         (0, 250_000, 500_000, 750_000, 1_000_000))
        self.assertTrue(all(h.PIXEL not in row.prepares_device_ids for row in plan.transitions))

    def test_primary_minimum_does_not_become_a_fixed_helper_grid(self) -> None:
        original = h.model_endpoint
        with mock.patch.object(h, "model_endpoint", side_effect=lambda model, declaration: replace(
            original(model, declaration), ffn_column_quantum=24
        )):
            catalog = h.runtime_catalog(self.model, declaration=h.co_helpers())
        values = self._candidates(catalog)
        _baseline, policies, envelope = adaptive_decode_policies(values, self.model, catalog, 30)
        self.assertIsNotNone(envelope)
        self.assertEqual(envelope.plan.adapter_parameters["ffn_column_quantum"], 32)
        both = [row for row in policies if len(row.device_layer_masks) == 2]
        self.assertEqual([row.columns for row in both], [32, 64, 96, 128])
        self.assertTrue(all(row.device_layer_masks == (
            (h.OP15, 0b1111), (h.PIXEL, h.PIXEL_MASK)
        ) for row in both))
        # each phone alone on the same grid (per-device policies)
        self.assertEqual(sorted((row.device_layer_masks, row.columns) for row in policies
                                if len(row.device_layer_masks) == 1),
                         sorted([(((h.OP15, 0b1111),), columns) for columns in (32, 64, 96, 128)]
                                + [(((h.PIXEL, h.PIXEL_MASK),), columns) for columns in (32, 64, 96, 128)]))

    def test_adaptive_policies_cover_both_phones_on_the_union_grid(self) -> None:
        values = self._candidates()
        _baseline, policies, envelope = adaptive_decode_policies(values, self.model, self.catalog, 30)
        self.assertIsNotNone(envelope)
        both = [row for row in policies if len(row.device_layer_masks) == 2]
        self.assertEqual([(row.columns, row.split_fraction_ppm) for row in both],
                         [(32, 250_000), (64, 500_000), (96, 750_000), (128, 1_000_000)])
        for policy in both:
            self.assertEqual(policy.layer_mask, 0b111111)
            self.assertEqual(policy.device_layer_masks, ((h.OP15, 0b1111), (h.PIXEL, h.PIXEL_MASK)))
            self.assertEqual(AdaptiveDecodePolicy.from_json(policy.to_json()), policy)
        for policy in (row for row in policies if len(row.device_layer_masks) == 1):
            (device, mask), = policy.device_layer_masks
            self.assertEqual(policy.layer_mask, mask)
            self.assertIn(device, (h.OP15, h.PIXEL))
            self.assertEqual(AdaptiveDecodePolicy.from_json(policy.to_json()), policy)
        self.assertEqual(len(policies), 12)
        # the same model with the primary phone alone keeps its finer 12.5 % grid
        single = h.runtime_catalog(self.model)
        _baseline, single_policies, _ = adaptive_decode_policies(self._candidates(single), self.model, single, 30)
        self.assertIn(16, {row.columns for row in single_policies})
        self.assertTrue(all(row.device_layer_masks == () for row in single_policies))

    def test_unqualified_co_helper_yields_no_two_phone_policy(self) -> None:
        catalog = h.runtime_catalog(self.model, declaration=h.co_helpers(), qualified_pixel=False)
        values = self._candidates(catalog)
        self.assertEqual(self._split(values).maturity, "SHADOW")
        _baseline, policies, envelope = adaptive_decode_policies(values, self.model, catalog, 30)
        self.assertEqual((policies, envelope), ((), None))

    def test_cold_co_helper_is_rejected_not_prepared(self) -> None:
        values = self._candidates(snapshot=h.snapshot(self.model, self.catalog, pixel_hot=False))
        split = self._split(values)
        self.assertFalse(split.admitted)
        self.assertTrue(all(h.PIXEL not in row.prepares_device_ids for row in split.plan.transitions))
        _baseline, policies, _ = adaptive_decode_policies(values, self.model, self.catalog, 30)
        self.assertEqual(policies, ())

    def test_elastic_absent_co_helper_keeps_the_primary_only_sets(self) -> None:
        """Elastic phones only: the rig marks an absent co-helper with a membership telemetry row. The
        split is still not prepared for it (nor admitted as a static route), but the adaptive envelope
        keeps its primary-only device sets; the controller keeps the absent phone's sets eliminated."""
        snapshot = h.snapshot(self.model, self.catalog, pixel_hot=False)
        snapshot = replace(snapshot, telemetry_observations={h.PIXEL: {
            "failure_reason": "co-helper absent", "membership": "ABSENT", "source": "helper-membership",
            "valid": False, "validity": "UNAVAILABLE"}})
        values = self._candidates(snapshot=snapshot)
        split = self._split(values)
        self.assertFalse(split.admitted)
        self.assertEqual(split.rejection_reasons, ("COLD_RESIDENCY_BREAK_EVEN", "CO_HELPER_UNAVAILABLE"))
        self.assertTrue(all(h.PIXEL not in row.prepares_device_ids for row in split.plan.transitions))
        _baseline, policies, envelope = adaptive_decode_policies(values, self.model, self.catalog, 30)
        self.assertIsNotNone(envelope)
        self.assertEqual(sorted(row.columns for row in policies
                                if tuple(device for device, _ in row.device_layer_masks) == (h.OP15,)),
                         [32, 64, 96, 128])

    def test_co_helper_batch_capacity_must_match_the_server(self) -> None:
        catalog = h.runtime_catalog(self.model, declaration=h.co_helpers(max_tokens=2))
        split = self._split(self._candidates(catalog))
        self.assertIn("TRANSPORT_PROFILE_INCOMPLETE", split.rejection_reasons)


class TwoPhoneTicketLaunchTests(unittest.TestCase):
    """Gap 2: ``phone_helpers`` from plan to ticket checks to the launch environment."""

    def setUp(self) -> None:
        self.model_scope = h.TemporaryModel()
        self.model = self.model_scope.__enter__()
        self.addCleanup(self.model_scope.__exit__)
        self.catalog = h.runtime_catalog(self.model, declaration=h.co_helpers())
        self.snapshot = h.snapshot(self.model, self.catalog)

    def _assert_two_helper_environment(self, environment) -> None:
        self.assertEqual(environment["S41_SERVER_FFN_HELPERS"], "2")
        self.assertEqual(environment["S41_SERVER_FFN_LAYER_MASK"], "63")
        self.assertEqual(
            {key: environment[key] for key in environment if key.startswith("S41_SERVER_FFN_HELPER1_")},
            {"S41_SERVER_FFN_HELPER1_HOST": "127.0.0.1", "S41_SERVER_FFN_HELPER1_LABEL": "pixel",
             "S41_SERVER_FFN_HELPER1_LAYER_MASK": "48", "S41_SERVER_FFN_HELPER1_PORT": "26991",
             "S41_SERVER_FFN_HELPER1_TRANSPORT": "tcp"},
        )
        self.assertEqual(environment["S41_SERVER_FFN_HELPER0_LAYER_MASK"], "15")
        self.assertEqual(environment["S41_SERVER_FFN_HELPER0_TRANSPORT"], "functionfs-usb")
        self.assertNotIn("S41_SERVER_FFN_TRANSPORT", environment)

    def test_adaptive_ticket_launches_one_server_with_both_helpers(self) -> None:
        ticket = _submit(self.catalog, self.model, self.snapshot, "adaptive", "adaptive-decode")
        self.assertEqual(ticket.execution_plan.execution_contract.execution_mode, "adaptive-split")
        command = interpret_runtime_ticket(ticket)
        validate_physical_execution_command(command)
        contract = llama_server_launch_contract(command, self.model)
        self.assertEqual(contract.gpu_layers, 2)
        self.assertEqual(contract.phone_device_id, h.OP15)
        self._assert_two_helper_environment(contract.ffn_environment)

    def test_desktop_parent_carries_both_phones_in_its_dormant_runtime(self) -> None:
        ticket = _submit(self.catalog, self.model, self.snapshot, "parent", "energy-aware")
        self.assertEqual(ticket.execution_plan.execution_contract.execution_mode, "desktop")
        dormant = dormant_phone_ffn_parameters(ticket.execution_plan.adapter_parameters)
        self.assertEqual(dormant["ffn_resident_layer_mask"], 63)
        self.assertEqual(dict(phone_helper_layer_masks(dormant[PHONE_HELPERS_PARAMETER])),
                         {h.OP15: 0b1111, h.PIXEL: h.PIXEL_MASK})
        command = interpret_runtime_ticket(ticket)
        validate_physical_execution_command(command)
        self._assert_two_helper_environment(llama_server_launch_contract(command, self.model).ffn_environment)

    def test_ticket_rejects_a_binding_that_differs_from_the_plan(self) -> None:
        command = interpret_runtime_ticket(
            _submit(self.catalog, self.model, self.snapshot, "tamper", "adaptive-decode"))
        declaration = h.co_helpers()
        for helpers in (
            declaration.phone_helpers(h.OP15, 0b0111),
            h.co_helpers(layer_mask=1 << 6).phone_helpers(h.OP15, 0b1111),
            "[]",
        ):
            with self.subTest(helpers=helpers), self.assertRaises(PhysicalAdapterError):
                validate_physical_execution_command(replace(command, adapter_parameters={
                    **command.adapter_parameters, PHONE_HELPERS_PARAMETER: helpers}))
        participants = tuple(row for row in command.participants if row.device_id != h.PIXEL)
        with self.assertRaisesRegex(PhysicalAdapterError, "binding differs"):
            validate_physical_execution_command(replace(command, participants=participants))


class DormantContractTests(unittest.TestCase):
    """Gap 2: the dormant desktop contract, residency match and storage superset carry ``phone_helpers``."""

    def setUp(self) -> None:
        self.model_scope = h.TemporaryModel()
        self.model = self.model_scope.__enter__()
        self.addCleanup(self.model_scope.__exit__)

    def test_dormant_contract_whitelists_and_checks_phone_helpers(self) -> None:
        helpers = h.phone_helpers_json(0b1111)
        contract = dormant_phone_ffn_parameters({DORMANT_PHONE_FFN_RUNTIME_PARAMETER: _dormant(63, helpers)})
        self.assertEqual(contract[PHONE_HELPERS_PARAMETER], helpers)
        for mask, value in ((0b1111, helpers), (63, h.phone_helpers_json(0b0111))):
            with self.assertRaises(PhysicalAdapterError):
                dormant_phone_ffn_parameters({DORMANT_PHONE_FFN_RUNTIME_PARAMETER: _dormant(mask, value)})
        with self.assertRaises(PhysicalAdapterError):
            dormant_phone_ffn_parameters({DORMANT_PHONE_FFN_RUNTIME_PARAMETER: _dormant(
                63, helpers, phone_device_id=h.PIXEL)})

    def test_live_server_serves_a_per_device_subset(self) -> None:
        resident = {DORMANT_PHONE_FFN_RUNTIME_PARAMETER: _dormant(63, h.phone_helpers_json(0b1111))}
        subset = {DORMANT_PHONE_FFN_RUNTIME_PARAMETER: _dormant(0b110011, h.phone_helpers_json(0b11))}
        single = {DORMANT_PHONE_FFN_RUNTIME_PARAMETER: _dormant(0b11, None)}
        moved = {DORMANT_PHONE_FFN_RUNTIME_PARAMETER: _dormant(0b110011, h.phone_helpers_json(
            0b11, forward_port=26992))}
        self.assertFalse(physical_residency_parameters_match(resident, moved))
        self.assertTrue(physical_residency_parameters_match(resident, subset))
        self.assertTrue(physical_residency_parameters_match(resident, resident))
        self.assertFalse(physical_residency_parameters_match(subset, resident))
        self.assertFalse(physical_residency_parameters_match(resident, single))
        self.assertFalse(physical_residency_parameters_match(single, subset))
        plan = SimpleNamespace(adapter_parameters=resident)
        self.assertTrue(_dormant_phone_ffn_runtime_supports(plan, subset[DORMANT_PHONE_FFN_RUNTIME_PARAMETER]))
        self.assertFalse(_dormant_phone_ffn_runtime_supports(plan, single[DORMANT_PHONE_FFN_RUNTIME_PARAMETER]))

    def test_stored_shards_grow_only_the_primary_phone(self) -> None:
        parameters = json.loads(_dormant(0b110011, h.phone_helpers_json(0b11)))
        controller = SimpleNamespace(_phone_ffn_shard_storage=(SimpleNamespace(
            parent_artifact_sha256=self.model.artifact_sha256, maximum_columns=128, layer_mask=0xFF),))
        desktop = SimpleNamespace(adapter_parameters={"cpu_device_id": h.CPU}, operators=tuple(
            SimpleNamespace(operator_id=row.operator_id, operator_kind=row.kind, split_axis="none",
                            device_ids=(h.CPU if int(row.layer_id[6:]) < 6 else h.GPU,))
            for row in self.model.operators if row.kind == "ffn"))
        result = _dormant_phone_ffn_storage_superset(controller, parameters, desktop, self.model)
        self.assertEqual(result["ffn_resident_layer_mask"], 0b111111)
        self.assertEqual(dict(phone_helper_layer_masks(result[PHONE_HELPERS_PARAMETER])),
                         {h.OP15: 0b1111, h.PIXEL: h.PIXEL_MASK})


class AdaptivePolicyOwnerTests(unittest.TestCase):
    """Gap 3: per-device masks enter the policy hash only for several phones."""

    def _policy(self, **changes) -> AdaptiveDecodePolicy:
        return AdaptiveDecodePolicy(**{
            "route_id": "route", "executor_id": "executor", "operator_plan_sha256": "sha256:" + "2" * 64,
            "desktop_parent_route_id": "parent", "desktop_placement_sha256": "sha256:" + "3" * 64,
            "layer_indices": tuple(range(6)), "layer_mask": 63, "columns": 32, "split_fraction_ppm": 250_000,
            "resource_ids": ("compute",), **changes,
        })

    def test_single_phone_policy_hash_is_unchanged(self) -> None:
        policy = self._policy()
        self.assertEqual(getattr(policy, "device_layer_masks", ()), ())
        self.assertNotIn("device_layer_masks", policy.to_json())
        # the digest the scheduler computed before per-device owners existed
        self.assertEqual(policy.policy_hash,
                         "sha256:e75a05351429f96d3326e494e161962daec88cf1acf83fbb2b6f3e7c78ed0c15")

    def test_owners_are_hashed_normalized_and_narrowed_with_the_policy(self) -> None:
        owners = ((h.OP15, 0b1111), (h.PIXEL, h.PIXEL_MASK))
        policy = self._policy(device_layer_masks=owners)
        self.assertNotEqual(policy.policy_hash, self._policy().policy_hash)
        self.assertEqual(AdaptiveDecodePolicy.from_json(policy.to_json()), policy)
        swapped = self._policy(device_layer_masks=((h.OP15, 0b0011_1100), (h.PIXEL, 0b11)))
        self.assertNotEqual(swapped.policy_hash, policy.policy_hash)
        narrowed = replace(policy, layer_mask=0b1111, layer_indices=(0, 1, 2, 3))
        self.assertEqual(narrowed.device_layer_masks, ((h.OP15, 0b1111),))
        for bad in (((h.OP15, 0b1111),), ((h.OP15, 63), (h.PIXEL, 48)), ((h.OP15, 15), (h.OP15, 48))):
            with self.subTest(owners=bad), self.assertRaises(AdaptiveDecodeError):
                self._policy(device_layer_masks=bad)
        with self.assertRaises(AdaptiveDecodeError):
            self._policy(route_id="parent", layer_indices=(), layer_mask=0, columns=0, split_fraction_ppm=0,
                         baseline=True, device_layer_masks=owners)


class TwoPhoneProofTests(unittest.TestCase):
    """Gap 3: every co-helper call is accounted to its own proof shard and id range."""

    def setUp(self) -> None:
        self.model_scope = h.TemporaryModel()
        self.model = self.model_scope.__enter__()
        self.addCleanup(self.model_scope.__exit__)
        self.plan_sha256 = "sha256:" + "4" * 64
        self.op15_shard = RuntimePhoneShard(
            "HTP0", "functionfs://op15/HTP0", 0b1111, 128, 1 << 20, "sha256:" + "5" * 64,
            self.plan_sha256, self.model.artifact_sha256, 3)

    def _proofs(self, calls, *, with_helper: bool = True):
        parameters = {
            PHONE_CO_HELPERS_PARAMETER: h.co_helpers_json(),
            PHONE_HELPERS_PARAMETER: h.phone_helpers_json(0b1111),
        } if with_helper else {}
        source = SimpleNamespace(
            operator_plan_sha256=self.plan_sha256, adapter_parameters=parameters,
            execution_contract=SimpleNamespace(remote_resident_ffn=None, phone_shards=(self.op15_shard,)))
        command = SimpleNamespace(artifact_sha256=self.model.artifact_sha256)
        return ManagedServerProofMixin()._execution_session_proofs(
            command, source, (), None, calls, {}, self.model, set())

    @staticmethod
    def _call(request_id: int, layer: int) -> LlamaServerFfnCall:
        return LlamaServerFfnCall(request_id=request_id, layer=layer, tokens=1, columns=32, payload_bytes=64)

    def test_co_helper_calls_land_on_its_proof_shard(self) -> None:
        proofs = self._proofs((self._call(1, 0), self._call(2, 3), self._call(1 + (1 << 24), 4),
                               self._call(2 + (1 << 24), 5)))
        by_session = {row.session_id: row for row in proofs}
        self.assertEqual(set(by_session), {"HTP0", "PIXEL0"})
        self.assertEqual((by_session["HTP0"].calls, by_session["PIXEL0"].calls), (2, 2))
        self.assertEqual(by_session["PIXEL0"].layer_mask, h.PIXEL_MASK)
        self.assertEqual(by_session["PIXEL0"].endpoint, "adb-tcp://" + h.PIXEL_SERIAL + "/PIXEL0")
        self.assertEqual(by_session["PIXEL0"].resident_geometry_sha256, h.SHARD_SHA)

    def test_calls_outside_their_owners_id_range_or_unserved_shards_fail(self) -> None:
        for calls in (
            (self._call(1, 0), self._call(2, 4)),
            (self._call(1 + (1 << 24), 0), self._call(2 + (1 << 24), 4)),
            (self._call(1, 0),),
        ):
            with self.subTest(calls=calls), self.assertRaises(PhysicalAdapterError):
                self._proofs(calls)

    def test_single_phone_proofs_are_unchanged(self) -> None:
        proofs = self._proofs((self._call(1, 0), self._call(2, 3)), with_helper=False)
        self.assertEqual([(row.session_id, row.calls) for row in proofs], [("HTP0", 2)])
        with self.assertRaises(PhysicalAdapterError):
            self._proofs((self._call(1, 0), self._call(2, 4)), with_helper=False)


class CampaignCoHelperTests(unittest.TestCase):
    """Gap 1 inputs: the rig helper phone and its shard index give the catalog declaration."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = h.manifest(self.directory.name)
        self.index = Path(self.directory.name) / "FFN_SHARDS.json"
        self._write_index(self.model.artifact_sha256, 128)

    def _write_index(self, parent: str, columns: int, mask: int = h.PIXEL_MASK, path: Path | None = None) -> Path:
        path = self.index if path is None else path
        path.write_text(json.dumps({"parent_sha256": parent, "schema": "s42-ffn-shard-index-v1", "shards": [{
            "columns": columns, "layer_mask": "%016x" % mask, "n_ff": 128, "parent_sha256": parent,
            "path": "HTP0.ffn.gguf", "session_id": "HTP0", "shard_bytes": 4096, "shard_sha256": h.SHARD_SHA,
            "weight_type": "F16"}]}), encoding="ascii")
        return path

    def _rig(self, forward_port: int = 26991) -> RigManifest:
        row = two_phone_rig_json()
        row["helper_phones"][0]["forward_port"] = forward_port
        return RigManifest.from_json(row, Path("/"))

    def _config(self, primary_index=None):
        return SimpleNamespace(
            helper_phone_ffn_shards={
                "pixel10pro-phone": (self.index, "/data/local/tmp/s42-pixel10pro-qualification-20260922-v1")},
            phone_ffn_shard_index_path=primary_index, phone_ffn_shard_directory="/data/local/tmp/op15-shards")

    def test_rig_helper_phone_and_shard_index_give_the_declaration(self) -> None:
        primary = self._write_index(self.model.artifact_sha256, 128, 0b1111, Path(self.directory.name) / "op15.json")
        declaration = campaign_catalog.helper_phone_co_helpers(self._rig(), self._config(primary), self.model)
        row, = declaration.helpers
        self.assertEqual((declaration.primary_label, declaration.primary_serial), ("op15", h.OP15_SERIAL))
        self.assertEqual((row.device_id, row.label, row.session_id), ("pixel10pro-phone", "pixel10pro", "PIXEL10PRO0"))
        self.assertEqual((row.layer_mask, row.column_quantum, row.max_tokens, row.shard_sha256, row.resident_bytes),
                         (h.PIXEL_MASK, 4352, 4, h.SHARD_SHA, 4096))
        self.assertEqual(dict(row.transport_parameters), {
            "adb_port": 5037, "adb_serial": h.PIXEL_SERIAL, "ffn_transport": "adb-tcp",
            "ffn_worker_host": "127.0.0.1", "ffn_worker_port": 26991, "phone_worker_port": 26990})
        self.assertIsNone(campaign_catalog.helper_phone_co_helpers(self._rig(), SimpleNamespace(helper_phone_ffn_shards={}), self.model))

    def test_unfixed_forward_or_foreign_shard_is_refused(self) -> None:
        with self.assertRaisesRegex(campaign_catalog.MaterializationError, "fixed host forward port"):
            campaign_catalog.helper_phone_co_helpers(self._rig(forward_port=0), self._config(), self.model)
        overlapping = self._write_index(self.model.artifact_sha256, 128, 0b11_1111, Path(self.directory.name) / "o.json")
        with self.assertRaisesRegex(campaign_catalog.MaterializationError, "overlap co-helper layers"):
            campaign_catalog.helper_phone_co_helpers(self._rig(), self._config(overlapping), self.model)
        self._write_index("sha256:" + "9" * 64, 128)
        with self.assertRaisesRegex(campaign_catalog.MaterializationError, "full-width shard"):
            campaign_catalog.helper_phone_co_helpers(self._rig(), self._config(), self.model)
        self._write_index(self.model.artifact_sha256, 64)
        with self.assertRaisesRegex(campaign_catalog.MaterializationError, "full-width shard"):
            campaign_catalog.helper_phone_co_helpers(self._rig(), self._config(), self.model)


class _Receipt:
    def __init__(self, value) -> None:
        self.value = value

    def to_json(self):
        return dict(self.value)


class _FakeWorker:
    def __init__(self, **changes) -> None:
        row = h.co_helpers().helpers[0]
        self.configuration = SimpleNamespace(**{
            "device_id": row.device_id, "serial": row.serial, "layer_mask": row.layer_mask,
            "column_quantum": row.column_quantum, "max_tokens": row.max_tokens, "phone_port": 26990,
            "forward_port": 26991, **changes})
        self.calls = []

    def preflight(self):
        self.calls.append("preflight")
        return _Receipt({"kind": "preflight"})

    def start(self, log_path):
        self.calls.append(("start", log_path.name))
        return _Receipt({"kind": "start"})

    def transport_parameters(self):
        return dict(h.PIXEL_TRANSPORT)


class _RecordingStop:
    name = "recording"

    def __init__(self) -> None:
        self.calls = []

    def stop(self, session, *, served_calls):
        self.calls.append((session.configuration.device_id, served_calls))
        return {"stopped": session.configuration.device_id}


class CoHelperLifecycleHookTests(unittest.TestCase):
    """Gap 4 (designed only): the hook refuses to run until a stop policy is chosen."""

    def test_no_stop_policy_means_no_start(self) -> None:
        worker = _FakeWorker()
        lifecycle = CoHelperLifecycle(h.co_helpers(), {h.PIXEL: worker})
        with self.assertRaisesRegex(PhysicalAdapterError, "stop policy"):
            lifecycle.start_trace(Path("/nonexistent"))
        self.assertEqual(worker.calls, [])

    def test_a_plugged_stop_policy_ends_the_trace(self) -> None:
        worker, stop = _FakeWorker(), _RecordingStop()
        lifecycle = CoHelperLifecycle(h.co_helpers(), {h.PIXEL: worker}, stop)
        with tempfile.TemporaryDirectory() as directory:
            model = h.manifest(directory)
        lifecycle.start_trace(Path("/logs"))
        self.assertEqual(worker.calls, ["preflight", ("start", "PIXEL0-worker.log")])
        residency, = lifecycle.residency_observations(model)
        self.assertEqual((residency.device_id, residency.state, len(residency.resident_tensor_ids)),
                         (h.PIXEL, "hot", 6))
        op15 = SimpleNamespace(session_id="HTP0")
        pixel = SimpleNamespace(session_id="PIXEL0")
        self.assertEqual(lifecycle.split_session_proofs((op15, pixel)), ((op15,), {h.PIXEL: (pixel,)}))
        self.assertEqual(lifecycle.end_trace({h.PIXEL: 12}), ({"stopped": h.PIXEL},))
        self.assertEqual(stop.calls, [(h.PIXEL, 12)])
        self.assertEqual(lifecycle.residency_observations(model), ())

    def test_worker_must_match_the_catalog_declaration(self) -> None:
        for change in ({"forward_port": 0}, {"layer_mask": 1 << 4}, {"max_tokens": 1}):
            with self.subTest(change=change), self.assertRaises(PhysicalAdapterError):
                CoHelperLifecycle(h.co_helpers(), {h.PIXEL: _FakeWorker(**change)}, _RecordingStop())


class SinglePhoneUnchangedTests(unittest.TestCase):
    """The single-phone catalog, candidates, policies, plans and launches of the base tree."""

    GOLDEN = {
        "candidates": "sha256:bc585e565fc0d908a00c11a85f7efce26073de74d0192cea424d26cd6421665b",
        "catalog": "sha256:95fc2b3c8f30de297e05d463a897d3cc853cab226d553b14efe3458994930eae",
        "launch-adaptive-decode": "sha256:67308b45b9fa283faa154177327ca8dd0243ff3dca48949a68949138b1c5b886",
        "launch-energy-aware": "sha256:67308b45b9fa283faa154177327ca8dd0243ff3dca48949a68949138b1c5b886",
        "plan-adaptive-decode": "sha256:e37a307712d21ee0e648a9a0463943ebc165be07df4d4bd85537bb1a7a524f3c",
        "plan-energy-aware": "sha256:597039df3a3874aff9f8bcc13604f71e8f3385a84ef7f08194f54a83c9f2bb9b",
        "policies": "sha256:71fb74a4f9617afa0bbcbe410d954f40f30ad28988e668b3996118a9e4dc8e3b",
    }

    def test_single_phone_digests_match_the_base_tree(self) -> None:
        with h.TemporaryModel() as model:
            catalog = h.runtime_catalog(model)
            snapshot = h.snapshot(model, catalog)
            scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
            scheduler.register_runtime_capabilities(catalog)
            scheduler.register_model_manifest(model)
            values = scheduler.generate_automated_candidates(h.request("golden"), model.model_id, snapshot)
            baseline, policies, _ = adaptive_decode_policies(values, model, catalog, 30)
            digests = {
                "catalog": canonical_sha256(catalog.to_json()),
                "candidates": canonical_sha256([
                    [row.candidate_id, row.plan.to_json(), list(row.rejection_reasons)]
                    for row in values.candidates
                ]),
                "policies": canonical_sha256([row.to_json() for row in (baseline, *policies)]),
            }
            for mode in ("adaptive-decode", "energy-aware"):
                ticket = _submit(catalog, model, snapshot, "golden-" + mode, mode)
                contract = llama_server_launch_contract(interpret_runtime_ticket(ticket), model)
                digests["plan-" + mode] = ticket.execution_plan.plan_sha256
                digests["launch-" + mode] = canonical_sha256(dict(contract.ffn_environment))
        self.assertEqual(digests, self.GOLDEN)


if __name__ == "__main__":
    unittest.main()
