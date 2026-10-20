"""WS23: every helper phone is its own resource (opt-in ``per_helper_outage``).

OP15 thermally deferred -> the route keeps running on the Pixel; Pixel lost -> the OP15 keeps its
layers; both back -> the union is restored. Defaults stay byte-identical.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest import mock

from research_dev.scheduler import UnifiedScheduler
from research_dev.scheduler._internal.adaptive_decode_planning import adaptive_decode_policies

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import two_phone_harness as h  # noqa: E402


def catalog_for(model, *, independent: bool):
    original = h.model_endpoint
    with mock.patch.object(h, "model_endpoint", side_effect=lambda m, d: replace(
        original(m, d), per_helper_outage=independent
    )):
        return h.runtime_catalog(model, declaration=h.co_helpers())


def hot(snapshot, device_executor: str, millic: int = 95_000):
    executors = dict(snapshot.executors)
    executors[device_executor] = replace(executors[device_executor], temperature_millic=millic)
    return replace(snapshot, executors=executors)


def lost(snapshot, device: str):
    return replace(snapshot, telemetry_observations={**snapshot.telemetry_observations, device: {
        "failure_reason": "helper absent", "membership": "ABSENT", "source": "helper-membership",
        "valid": False, "validity": "UNAVAILABLE"}})


class HelperIndependenceTests(unittest.TestCase):
    def setUp(self) -> None:
        scope = h.TemporaryModel()
        self.model = scope.__enter__()
        self.addCleanup(scope.__exit__)

    def candidates(self, catalog, snapshot, scheduler=None):
        scheduler = scheduler or UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(catalog)
        scheduler.register_model_manifest(self.model)
        return scheduler, scheduler.generate_automated_candidates(h.request("ws23"), self.model.model_id, snapshot)

    @staticmethod
    def split(values):
        return next(row for row in values.candidates if row.binding.executor_id.endswith(":operator_split"))

    def test_defaults_keep_route_wide_thermal_rejection(self) -> None:
        catalog = catalog_for(self.model, independent=False)
        _, values = self.candidates(catalog, hot(h.snapshot(self.model, catalog), "physical:" + h.OP15))
        reasons = self.split(values).rejection_reasons
        self.assertIn("THERMAL_LIMIT", reasons)
        self.assertNotIn("CO_HELPER_UNAVAILABLE", reasons)
        _baseline, policies, envelope = adaptive_decode_policies(values, self.model, catalog, 30)
        self.assertEqual((policies, envelope), ((), None))

    def test_deferred_op15_keeps_the_route_and_quarantines_only_the_op15(self) -> None:
        catalog = catalog_for(self.model, independent=True)
        snapshot = h.snapshot(self.model, catalog)
        scheduler, values = self.candidates(catalog, hot(snapshot, "physical:" + h.OP15))
        split = self.split(values)
        self.assertNotIn("THERMAL_LIMIT", split.rejection_reasons)
        self.assertIn("CO_HELPER_UNAVAILABLE", split.rejection_reasons)
        _baseline, policies, envelope = adaptive_decode_policies(values, self.model, catalog, 30)
        self.assertIsNotNone(envelope)
        self.assertTrue(any(tuple(d for d, _ in row.device_layer_masks) == (h.PIXEL,) for row in policies))
        self.assertEqual(dict(scheduler.quarantined_devices()), {h.OP15: "HELPER_DEFERRED"})
        # the deferral is still recorded as the rig/route gate records it, per device
        events = scheduler.thermal_deferral_events()
        self.assertEqual([(row["kind"], row["device_id"]) for row in events], [("THERMAL_DEFERRAL", h.OP15)])
        # no route-wide resource withholding: the Pixel's own layers keep their route
        self.assertEqual(scheduler.runtime_controller_snapshot()["quarantined_resources"], [])

    def test_both_back_restores_the_union(self) -> None:
        catalog = catalog_for(self.model, independent=True)
        snapshot = h.snapshot(self.model, catalog)
        scheduler, _ = self.candidates(catalog, hot(snapshot, "physical:" + h.OP15))
        self.assertEqual(set(scheduler.quarantined_devices()), {h.OP15})
        _, values = self.candidates(catalog, snapshot, scheduler)
        self.assertEqual(dict(scheduler.quarantined_devices()), {})
        self.assertEqual(self.split(values).rejection_reasons, ("COLD_RESIDENCY_BREAK_EVEN",))
        kinds = [row["kind"] for row in scheduler.device_membership_events()]
        self.assertEqual(kinds, ["DEVICE_QUARANTINED", "DEVICE_READMITTED"])
        self.assertEqual([row["kind"] for row in scheduler.thermal_deferral_events()],
                         ["THERMAL_DEFERRAL", "THERMAL_DEFERRAL_CLEARED"])

    def test_lost_pixel_leaves_the_op15_its_layers_and_op15_hot_too_falls_back(self) -> None:
        catalog = catalog_for(self.model, independent=True)
        snapshot = lost(h.snapshot(self.model, catalog, pixel_hot=False), h.PIXEL)
        _, values = self.candidates(catalog, snapshot)
        self.assertEqual(self.split(values).rejection_reasons, ("COLD_RESIDENCY_BREAK_EVEN", "CO_HELPER_UNAVAILABLE"))
        _baseline, policies, envelope = adaptive_decode_policies(values, self.model, catalog, 30)
        self.assertIsNotNone(envelope)
        # nobody left: both gates apply as today (route-wide)
        both = hot(snapshot, "physical:" + h.OP15)
        _, values = self.candidates(catalog, both)
        self.assertIn("THERMAL_LIMIT", self.split(values).rejection_reasons)

    def test_deferred_pixel_leaves_the_op15_its_layers(self) -> None:
        catalog = catalog_for(self.model, independent=True)
        scheduler, values = self.candidates(catalog, hot(h.snapshot(self.model, catalog), "physical:" + h.PIXEL))
        self.assertNotIn("THERMAL_LIMIT", self.split(values).rejection_reasons)
        self.assertEqual(dict(scheduler.quarantined_devices()), {h.PIXEL: "HELPER_DEFERRED"})
        _baseline, policies, envelope = adaptive_decode_policies(values, self.model, catalog, 30)
        self.assertTrue(any(tuple(d for d, _ in row.device_layer_masks) == (h.OP15,) for row in policies))

    def test_default_catalog_has_no_per_helper_parameter(self) -> None:
        for independent, expected in ((False, False), (True, True)):
            catalog = catalog_for(self.model, independent=independent)
            self.assertEqual(
                any("per_helper_outage" in row.adapter_parameters for row in catalog.composite_executors
                    if row.executor_id.startswith("physical:two:phone-assisted")), expected)


class HelperAnchoredRouteTests(unittest.TestCase):
    """A co-helper anchors a phone-assisted route of its own (no other phone in it)."""

    def setUp(self) -> None:
        scope = h.TemporaryModel()
        self.model = scope.__enter__()
        self.addCleanup(scope.__exit__)

    def catalog(self, led: bool, independent: bool | None = None):
        original = h.model_endpoint
        with mock.patch.object(h, "model_endpoint", side_effect=lambda m, d: replace(
            original(m, d), per_helper_outage=led if independent is None else independent,
            led_helper_device_ids=(h.PIXEL,) if led else ()
        )):
            return h.runtime_catalog(self.model, declaration=h.co_helpers())

    def test_default_catalog_has_no_anchored_family(self) -> None:
        ids = [row.executor_id for row in self.catalog(False).composite_executors]
        self.assertFalse(any("-led:" in value for value in ids))

    def test_primary_without_shards_gets_no_primary_anchored_family(self) -> None:
        """2026-10-05 rig: the Pixel-primary arm dropped the OP15's Qwen shard index, yet the OP15-anchored families
        were still materialized with every layer the Pixel does not own (the GPU layers) as the OP15's candidates,
        and that launch won: the server released the FFN weights of all 40 layers. Without shards there is no
        primary-anchored route; the led routes and the desktop route remain."""
        original = h.model_endpoint
        with mock.patch.object(h, "model_endpoint", side_effect=lambda m, d: replace(
            original(m, d), per_helper_outage=True, led_helper_device_ids=(h.PIXEL,),
            primary_phone_stores_model=False)):
            catalog = h.runtime_catalog(self.model, declaration=h.co_helpers())
        ids = [row.executor_id for row in catalog.composite_executors]
        self.assertFalse(any(value.startswith("physical:two:phone-assisted:") for value in ids), ids)
        self.assertTrue(any("-led:" + h.PIXEL in value for value in ids))
        self.assertIn("physical:two:desktop", ids)
        # the default (primary stores the model) keeps today's families
        ids_default = [row.executor_id for row in self.catalog(True).composite_executors]
        self.assertTrue(any(value.startswith("physical:two:phone-assisted:operator_offload") for value in ids_default))

    def test_measured_profile_binds_to_the_led_family_when_the_primary_family_is_absent(self) -> None:
        from research_dev.scheduler.campaigns.burstgpt.catalog import assisted_profile_targets
        devices = ("desktop-cpu", "desktop-cuda", h.OP15)
        present = frozenset({"physical:hot:phone-assisted:operator_offload", "physical:hot:desktop"})
        self.assertEqual(assisted_profile_targets("hot", "physical:hot:phone-assisted:operator_offload", devices, present),
                         (("physical:hot:phone-assisted:operator_offload", devices, ""),))
        led_only = frozenset({"physical:hot:desktop", f"physical:hot:phone-assisted-led:{h.PIXEL}:operator_offload",
                              f"physical:hot:phone-assisted-led:{h.PIXEL}:operator_split"})
        self.assertEqual(assisted_profile_targets("hot", "physical:hot:phone-assisted:operator_offload", devices, led_only),
                         ((f"physical:hot:phone-assisted-led:{h.PIXEL}:operator_offload",
                           ("desktop-cpu", "desktop-cuda", h.PIXEL), ":led:" + h.PIXEL),))
        # no composites known (older callers) or nothing led: unchanged
        self.assertEqual(assisted_profile_targets("hot", "physical:hot:phone-assisted:operator_offload", devices, None),
                         (("physical:hot:phone-assisted:operator_offload", devices, ""),))

    def test_lifecycle_discovery_finds_the_declaration_on_led_only_catalogs(self) -> None:
        """2026-10-05 rig: with the primary-anchored Qwen families gone, campaign_co_helper_lifecycles found no declaration
        ("catalog and helper campaign evidence differ") and the Pixel worker would never have been launched."""
        from research_dev.scheduler.campaigns.burstgpt.helper_phone_evidence import catalog_co_helper_declarations
        original = h.model_endpoint
        with mock.patch.object(h, "model_endpoint", side_effect=lambda m, d: replace(
            original(m, d), per_helper_outage=True, led_helper_device_ids=(h.PIXEL,),
            primary_phone_stores_model=False)):
            catalog = h.runtime_catalog(self.model, declaration=h.co_helpers())
        found = catalog_co_helper_declarations(catalog)
        self.assertEqual({a: d.encode() for a, d in found.items()}, {self.model.artifact_sha256: h.co_helpers().encode()})
        # the regular catalog finds the same declaration through phone_co_helpers_v1
        found_default = catalog_co_helper_declarations(self.catalog(True))
        self.assertEqual({a: d.encode() for a, d in found_default.items()}, {self.model.artifact_sha256: h.co_helpers().encode()})

    def test_anchored_route_gets_a_resident_superset_from_the_helper_declaration(self) -> None:
        """2026-10-05 rig: the Pixel-led arm ran 15 min with 0 Pixel calls: without HTP sessions the route had no
        resident envelope, so no resident-superset layout and no decode policy. The helper's declared slice is it."""
        from research_dev.scheduler._internal.route_generation.costing_parameters import anchored_resident_parameters
        from research_dev.scheduler._internal.plan_contracts.co_helpers import ANCHORED_CO_HELPERS_PARAMETER
        declaration = h.co_helpers()
        row = next(r for r in declaration.helpers if r.device_id == h.PIXEL)
        parameters = {"static_helper_anchor": 1, "phone_device_id": h.PIXEL, "ffn_column_quantum": 16,
                      ANCHORED_CO_HELPERS_PARAMETER: declaration.encode()}
        result = anchored_resident_parameters(parameters, self.model.feed_forward_length, 16)
        self.assertEqual((result["ffn_weight_buffer_layout"], result["ffn_resident_layer_mask"], result["ffn_resident_columns"],
                          result["ffn_resident_weight_bytes"], result["resident_model_identity_sha256"],
                          result["ffn_runtime_partition_count"], result["ffn_column_quantum"]),
                         ("resident-superset", row.layer_mask, self.model.feed_forward_length, row.resident_bytes,
                          row.shard_sha256, 1, 16))
        self.assertTrue(result["ffn_resident_geometry_sha256"].startswith("sha256:"))
        # not anchored, or the declaration absent: nothing (today's behaviour)
        self.assertIsNone(anchored_resident_parameters({"phone_device_id": h.PIXEL}, 64, 16))
        self.assertIsNone(anchored_resident_parameters({"static_helper_anchor": 1, "phone_device_id": h.PIXEL}, 64, 16))

    def test_led_resident_limit_admits_every_declared_layer_in_manifest_bytes(self) -> None:
        """2026-10-05 rig: the Pixel's packed shard (4.7 GB) capped the route costing to 8 of its 24 declared layers because
        allocations are compared in manifest (f16) bytes; the anchored limit is the declared layers' manifest bytes."""
        catalog = self.catalog(True)
        led = [row for row in catalog.composite_executors if "-led:" + h.PIXEL in row.executor_id][0]
        declared = sum(self.model.tensor_by_id[t].nbytes for operator in self.model.operators
                       if operator.kind == "ffn" and int(str(operator.layer_id).rsplit(":", 1)[-1]) in h.PIXEL_LAYERS
                       for t in operator.tensor_ids)
        self.assertGreaterEqual(led.adapter_parameters["maximum_helper_resident_weight_bytes"], declared)

    def test_anchored_helpers_count_as_available_while_members(self) -> None:
        from research_dev.scheduler._unified.adaptive_decode_control import (
            anchored_policy_devices, anchored_devices_are_members, DEVICE_ABSENT_AT_START)
        catalog = self.catalog(True)
        led = [row for row in catalog.composite_executors if "-led:" + h.PIXEL in row.executor_id][0]
        class Policy:
            def __init__(self, executor_id): self.executor_id = executor_id
        self.assertEqual(anchored_policy_devices(catalog, (Policy(led.executor_id), Policy("physical:two:desktop"))), (h.PIXEL,))
        self.assertEqual(anchored_policy_devices(catalog, (Policy("physical:two:desktop"),)), ())
        self.assertTrue(anchored_devices_are_members((h.PIXEL,), []))
        self.assertFalse(anchored_devices_are_members((h.PIXEL,), [{"device_id": h.PIXEL, "kind": "DEVICE_QUARANTINED"}]))
        self.assertFalse(anchored_devices_are_members((h.PIXEL,), [{"device_id": h.PIXEL, "kind": DEVICE_ABSENT_AT_START}]))
        self.assertTrue(anchored_devices_are_members((h.PIXEL,), [{"device_id": h.PIXEL, "kind": "DEVICE_QUARANTINED"},
                                                                  {"device_id": h.PIXEL, "kind": "DEVICE_READMITTED"}]))

    def test_led_family_has_only_the_helper_and_its_layers(self) -> None:
        catalog = self.catalog(True)
        rows = [row for row in catalog.composite_executors if "-led:" + h.PIXEL in row.executor_id]
        self.assertEqual(sorted(row.route_family for row in rows), ["operator_offload", "operator_split"])
        for row in rows:
            self.assertEqual(row.helper_device_id, h.PIXEL)
            self.assertNotIn(h.OP15, row.participant_device_ids)
            self.assertEqual(row.adapter_parameters["phone_device_id"], h.PIXEL)
            self.assertEqual(row.adapter_parameters["ffn_transport"], "adb-tcp")
            self.assertEqual(row.adapter_parameters["static_helper_anchor"], 1)
            self.assertNotIn("phone_co_helpers_v1", row.adapter_parameters)
            self.assertEqual({int(op.split(":")[1]) for op in row.operator_ids}, set(h.PIXEL_LAYERS))
            # the lifecycle carrier: the declaration, decodable, but not under the key the runtime reads
            from research_dev.scheduler._internal.plan_contracts.co_helpers import anchored_co_helper_declaration
            self.assertEqual(anchored_co_helper_declaration(row.adapter_parameters).encode(), h.co_helpers().encode())

    def test_anchored_composite_preloaded_residency_is_the_helper_layers(self) -> None:
        """2026-10-05 rig: the first preflight of the Pixel-led route failed with "runtime composite residency
        contract is incomplete": the anchored composite declares the Pixel preloaded but carries no exact operator
        placements (a placement record cannot say "whole operator on the helper"). The residency is the helper's layers."""
        from research_dev.scheduler.adapters.snapshot import (
            catalog_preloaded_residency_samples, model_residency_observations)
        catalog = self.catalog(True)
        manifest = self.model
        samples = catalog_preloaded_residency_samples(catalog, {manifest.model_id: manifest})
        led = [row for row in samples if "-led:" + h.PIXEL in row.executor_id]
        self.assertEqual(len(led), 2)
        def layer(operator):
            tail = str(operator.layer_id).rsplit(":", 1)[-1]
            return int(tail) if tail.isdigit() else None
        expected = {tensor_id for operator in manifest.operators if layer(operator) in h.PIXEL_LAYERS
                    for tensor_id in operator.tensor_ids if "ffn" in tensor_id}
        for sample in led:
            rows = model_residency_observations(catalog, manifest, sample.executor_id, generation=sample.generation,
                                                state=sample.state, resident_device_ids=sample.resident_device_ids)
            pixel = [row for row in rows if row.device_id == h.PIXEL]
            self.assertEqual(len(pixel), 1)
            self.assertTrue(set(pixel[0].resident_tensor_ids) >= expected)
            self.assertEqual(pixel[0].resident_bytes,
                             sum(manifest.tensor_by_id[t].nbytes for t in pixel[0].resident_tensor_ids))
            self.assertFalse(any(row.device_id == h.OP15 for row in rows))
            # shared device residency: not one endpoint's allocation (the second anchored composite must not conflict)
            self.assertIsNone(pixel[0].executor_id)
            # the full (unrestricted) residency of the anchored composite: the desktop keeps every tensor the helper
            # does not own, placed as the baseline desktop composite places it (GPU layers on the GPU), none of the helper's
            full = model_residency_observations(catalog, manifest, sample.executor_id, generation=1)
            desktop = [row for row in full if row.device_id != h.PIXEL]
            self.assertTrue(desktop and all(row.resident_bytes > 0 and row.executor_id == sample.executor_id
                                            for row in desktop))
            self.assertEqual(sorted(row.device_id for row in desktop), sorted(
                device for device in catalog.composite_executor_by_id[sample.executor_id].participant_device_ids
                if device != h.PIXEL))
            desktop_tensors = {t for row in desktop for t in row.resident_tensor_ids}
            self.assertTrue(desktop_tensors.isdisjoint(pixel[0].resident_tensor_ids))
            self.assertEqual(desktop_tensors | set(pixel[0].resident_tensor_ids), set(manifest.tensor_by_id))
        from research_dev.scheduler.adapters.snapshot import _merge_residency_rows
        first, second = (
            [row for row in model_residency_observations(catalog, manifest, sample.executor_id, generation=1,
                                                         resident_device_ids=sample.resident_device_ids)
             if row.device_id == h.PIXEL][0] for sample in led)
        merged = _merge_residency_rows(first, second)
        self.assertEqual(set(merged.resident_tensor_ids), set(first.resident_tensor_ids) | set(second.resident_tensor_ids))

    def test_op15_deferred_selects_the_pixel_anchored_route_and_launches_it_alone(self) -> None:
        from research_dev.scheduler.adapters import interpret_runtime_ticket, llama_server_launch_contract
        from research_dev.scheduler.adapters.ticket import validate_physical_execution_command
        from test_two_phone_gaps import _submit

        # led family alone: the OP15's union route is THERMAL_LIMIT (route-wide, as today), so the Pixel's own
        # route is the only phone-assisted one left; with ``per_helper_outage`` too the choice is cost-based
        catalog = self.catalog(True, independent=False)
        snapshot = hot(h.snapshot(self.model, catalog), "physical:" + h.OP15)
        for mode in ("adaptive-decode", "energy-aware"):
            ticket = _submit(catalog, self.model, snapshot, "led-" + mode, mode)
            command = interpret_runtime_ticket(ticket)
            validate_physical_execution_command(command)
            contract = llama_server_launch_contract(command, self.model)
            self.assertEqual(contract.phone_device_id, h.PIXEL)
            environment = contract.ffn_environment
            self.assertEqual(environment["S41_SERVER_FFN_TRANSPORT"], "tcp")
            self.assertEqual(environment["S41_SERVER_FFN_PORT"], "26991")
            self.assertEqual(environment["S41_SERVER_FFN_LAYER_MASK"], str(h.PIXEL_MASK))
            self.assertNotIn("S41_SERVER_FFN_HELPERS", environment)
            if mode == "adaptive-decode":
                self.assertIn("-led:" + h.PIXEL, ticket.decision.route_id)
                self.assertEqual(ticket.execution_plan.execution_contract.phone_device_id, h.PIXEL)

    def test_op15_healthy_keeps_the_union_route(self) -> None:
        from test_two_phone_gaps import _submit

        catalog = self.catalog(True)
        ticket = _submit(catalog, self.model, h.snapshot(self.model, catalog), "union", "adaptive-decode")
        self.assertNotIn("-led:", ticket.decision.route_id)
        self.assertEqual(ticket.execution_plan.execution_contract.phone_device_id, h.OP15)

    def test_led_family_needs_a_declared_co_helper(self) -> None:
        from research_dev.scheduler.adapters import CatalogMaterializationError

        original = h.model_endpoint
        with mock.patch.object(h, "model_endpoint", side_effect=lambda m, d: replace(
            original(m, d), led_helper_device_ids=("nobody",)
        )), self.assertRaises(CatalogMaterializationError):
            h.runtime_catalog(self.model, declaration=h.co_helpers())


class PlannerOutageTests(unittest.TestCase):
    def test_helper_anchored_inventory_lets_the_pixel_own_every_cpu_layer(self) -> None:
        from research_dev.scheduler._internal.layer_placement import (
            DESKTOP_CPU, price_single_outages, restrict_to_executable, solve_placement,
        )
        from research_dev.scheduler._internal.layer_placement_io import problem_from_json
        from research_dev.scheduler.campaigns.burstgpt import layer_placement_rig as rig
        from research_dev.scheduler.campaigns.burstgpt.tools.layer_placement import override_inventory

        profile = rig.measured_profile_v1()
        full = (f"{rig.PIXEL}:{rig.QWEN}=0-23",)
        results = {}
        for runtime in ("co-helper", "helper-anchored"):
            inventory = override_inventory(
                rig.rig_inventory_v1(profile=profile, pixel_runtime=runtime),
                stored=full, qualified=full, busy_envelope=(f"{rig.PIXEL}=1000.0",))
            problem = restrict_to_executable(problem_from_json(inventory, profile))
            plan = solve_placement(problem)
            results[runtime] = (problem, plan)
        anchored = results["helper-anchored"][1].models[rig.QWEN]
        self.assertEqual(anchored.mask(rig.OP15), 0)
        self.assertEqual(anchored.mask(rig.PIXEL), (1 << 24) - 1)
        # the co-helper inventory keeps the forced anchor layer on the OP15
        coupled = results["co-helper"][1].models[rig.QWEN]
        self.assertGreaterEqual(coupled.mask(rig.OP15).bit_count(), 1)
        # single-outage pricing: the Pixel down puts its 24 layers back on the CPU, the Gemma layers stay
        problem, plan = results["helper-anchored"]
        prices = {(row.model_id, row.device_id): row for row in price_single_outages(problem, plan)}
        qwen = prices[(rig.QWEN, rig.PIXEL)]
        self.assertEqual(len(qwen.layers), 24)
        self.assertGreater(qwen.delta_j_per_token, 0.0)
        self.assertAlmostEqual(qwen.energy_j_per_token, anchored.cpu_only_energy_j_per_token, places=6)
        self.assertAlmostEqual(qwen.saving_kept, 0.0, places=6)
        gemma = plan.models[rig.GEMMA]
        if gemma.mask(rig.OP15):
            self.assertGreater(prices[(rig.GEMMA, rig.OP15)].delta_j_per_token, 0.0)
        self.assertNotIn(DESKTOP_CPU, {device for _model, device in prices})

    def test_default_report_has_no_outage_rows(self) -> None:
        from research_dev.scheduler._internal.layer_placement import placement_report
        from research_dev.scheduler._internal.layer_placement_io import problem_from_json
        from research_dev.scheduler.campaigns.burstgpt import layer_placement_rig as rig

        profile = rig.measured_profile_v1()
        problem = problem_from_json(rig.rig_inventory_v1(profile=profile), profile)
        self.assertNotIn("single_outage_prices", placement_report(problem))
        report = placement_report(problem, outages=True)
        self.assertEqual(set(report["single_outage_prices"]), {"ideal", "executable"})


class RuntimeAccountingTests(unittest.TestCase):
    def test_static_helper_anchor_is_only_a_co_helper_session_device(self) -> None:
        from research_dev.scheduler.adapters.heterogeneous_rig import HeterogeneousPhysicalRig

        rig = object.__new__(HeterogeneousPhysicalRig)
        self.assertFalse(rig._phone_is_static_helper(h.PIXEL))          # no co-helper sessions: today's rig
        rig._co_helper_sessions = {h.PIXEL: object()}
        self.assertTrue(rig._phone_is_static_helper(h.PIXEL))
        self.assertFalse(rig._phone_is_static_helper(h.OP15))
        self.assertFalse(rig._phone_is_static_helper(None))

    def test_anchored_window_charges_the_helper_not_the_scheduled_phone(self) -> None:
        from types import SimpleNamespace
        from research_dev.scheduler.adapters.http_backend import _AdaptivePayloadController

        def controller(parameters):
            primary, helper = [], []
            value = _AdaptivePayloadController.__new__(_AdaptivePayloadController)
            value.backend = SimpleNamespace(_epoch_ns=1000, _energy_meter=SimpleNamespace(
                record_phone_activity_duration=lambda *row: primary.append(row),
                record_helper_phone_window=lambda *row: helper.append(row)))
            value.command = SimpleNamespace(adapter_parameters=parameters)
            return value, primary, helper

        delta = {"rpc_us": 5, "usb_transfer_us": 1, "phone_compute_us": 4}
        value, primary, helper = controller({"phone_device_id": h.PIXEL, "static_helper_anchor": 1})
        value._record_phone_window("qwen", 10, 20, delta)
        self.assertEqual(primary, [])
        self.assertEqual(helper[0][-1], (h.PIXEL,))
        value, primary, helper = controller({"phone_device_id": h.OP15})
        value._record_phone_window("qwen", 10, 20, delta)
        self.assertEqual((primary[0][-1], helper[0][-1]), (5, ()))


class MaterializeAnchoredTests(unittest.TestCase):
    """prepare_measured_placement: a Pixel-only Qwen plan is refused by default, accepted when anchored."""

    def setUp(self) -> None:
        from test_layer_placement_materialize import MaterializeTests

        self.fixture = MaterializeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def test_default_refuses_and_opt_in_drops_the_primary_index(self) -> None:
        import json
        from research_dev.scheduler.campaigns.burstgpt import layer_placement_rig as rig
        from research_dev.scheduler.campaigns.burstgpt.tools.prepare_measured_placement import (
            MaterializationRefused, materialize)

        f = self.fixture
        full = (1 << 24) - 1
        plan = json.loads(json.dumps(f.plan))
        owners = plan["models"][rig.QWEN]["owners"]
        owners[rig.PIXEL] = {"layers": "0-23", "layer_mask": f"{full:016x}", "count": 24}
        owners[rig.OP15] = {"layers": "", "layer_mask": f"{0:016x}", "count": 0}
        shard = json.loads((f.tmp / "pixel.json").read_text())
        shard["shards"][0]["layer_mask"] = f"{full:016x}"
        (f.tmp / "pixel.json").write_text(json.dumps(shard))
        evidence = json.loads(json.dumps(f.evidence))
        evidence["worker"]["layer_mask"] = full
        (f.tmp / "evidence.json").write_text(json.dumps(evidence))
        with self.assertRaisesRegex(MaterializationRefused, "primary phone op15-phone none"):
            materialize(f.base, plan, f.tmp / "d1", **f.kwargs)
        receipt = materialize(f.base, plan, f.tmp / "d2", helper_anchored=True, **f.kwargs)
        self.assertIn("PRIMARY_OWNS_NONE_INDEX_DROPPED", [row["status"] for row in receipt["checks"]])
        out = f.tmp / "d2"
        models = {row["model_id"]: row for row in json.loads((out / "models.json").read_text())["models"]}
        self.assertNotIn("phone_ffn_shard_index_path", models[rig.QWEN])
        self.assertIn("phone_ffn_shard_index_path", models[rig.GEMMA])        # the OP15 keeps Gemma
        campaign = json.loads((out / "campaign.json").read_text())
        self.assertIs(campaign["elastic_phones"]["helper_independence"], True)
        from research_dev.scheduler.configuration.campaign import CampaignManifest
        CampaignManifest.from_json(campaign, out)


class CampaignOptInTests(unittest.TestCase):
    def test_helper_independence_is_opt_in_and_validated(self) -> None:
        from research_dev.scheduler.configuration.campaign import elastic_phones_configuration

        base = elastic_phones_configuration({"drop_recovery": True, "join": True})
        self.assertNotIn("helper_independence", base)
        on = elastic_phones_configuration({"drop_recovery": True, "helper_independence": True})
        self.assertIs(on["helper_independence"], True)
        with self.assertRaises(Exception):
            elastic_phones_configuration({"drop_recovery": True, "helper_independence": "yes"})


if __name__ == "__main__":
    unittest.main()
