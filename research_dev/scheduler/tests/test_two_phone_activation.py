"""Two-phone activation (OP15 scheduled + Pixel static co-helper).

1. The ticket's own phone owns only its layers. A two-phone plan's resident FFN slice is the union of
   both phones' layers (the server serves all of them), while OP15's session, its shards and its stored
   residency hold only the first ``phone_helpers`` row. Comparing the union with OP15's shards refused
   every OP15 shard replacement of the two-phone model ("exact partial phone residency transition is
   unavailable", 225 times in the 2026-09-24 v7 trace).
2. The bounded rough frontier keeps the ``operator_split`` representative of every resident-envelope
   group the adaptive controller probes, even when the group's ``operator_offload`` visit ranks lower.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from research_dev.scheduler import RuntimePhoneShard, UnifiedScheduler
from research_dev.scheduler._internal.adaptive_decode_planning import adaptive_decode_policies
from research_dev.scheduler._internal.route_generation import compiler as route_compiler
from research_dev.scheduler._internal.route_generation.costing_rough import RouteRoughCostMixin
from research_dev.scheduler._internal.runtime_plan import PhoneSessionReplacementAuthorization
from research_dev.scheduler.adapters import (
    DirectPhoneFfnSession,
    HeterogeneousPhysicalRig,
    PhysicalAdapterError,
    interpret_runtime_ticket,
)
from research_dev.scheduler.adapters.heterogeneous_rig import _PersistentPhoneResidency
from research_dev.scheduler.adapters.llama_server import (
    PhoneFfnExecutionContract,
    phone_ffn_resident_contract,
)
from research_dev.scheduler.adapters.phone_transport import phone_transport_contract

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import two_phone_harness as h  # noqa: E402

GEMMA = "sha256:" + "a" * 64
QWEN = "sha256:" + "b" * 64
PRIMARY_MASK = 0b1111
UNION_MASK = PRIMARY_MASK | h.PIXEL_MASK
REPLACEMENT = "research_dev.scheduler.adapters.phone_session_ops.replacement.phone_ffn_resident_contract"
IDENTITY = "research_dev.scheduler.adapters.phone_session_ops.identity.phone_ffn_resident_contract"
PREFLIGHT = "research_dev.scheduler.adapters.phone_session_ops.preflight.phone_ffn_resident_contract"
RESIDENCY = "research_dev.scheduler.adapters.heterogeneous_rig_ops.residency.phone_ffn_resident_contract"


def primary_phone_ffn_contract(command, contract):
    from research_dev.scheduler.adapters.llama_server import primary_phone_ffn_contract as project

    return project(command, contract)


def shard(session_id: str, artifact: str, layer_mask: int, generation: int = 1) -> RuntimePhoneShard:
    return RuntimePhoneShard(
        session_id=session_id,
        endpoint="session://op15/" + session_id,
        layer_mask=layer_mask,
        maximum_columns=128,
        resident_bytes=100,
        resident_geometry_sha256="sha256:" + session_id[-1] * 64,
        operator_plan_sha256="sha256:" + ("d" if artifact == GEMMA else "e") * 64,
        artifact_sha256=artifact,
        session_generation=generation,
    )


def union_contract(mask: int = UNION_MASK) -> PhoneFfnExecutionContract:
    return PhoneFfnExecutionContract(
        device_id=h.OP15,
        n_embd=32,
        layer_indices=tuple(index for index in range(8) if mask >> index & 1),
        layer_mask=mask,
        columns=128,
        max_tokens=4,
        activation="swiglu",
    )


def two_phone_parameters(primary_mask: int = PRIMARY_MASK) -> dict[str, object]:
    return {
        **h.FUNCTIONFS_PARAMETERS,
        "ffn_column_quantum": 32,
        "phone_device_id": h.OP15,
        "phone_helpers": h.phone_helpers_json(primary_mask),
    }


class PrimaryPhoneContractTests(unittest.TestCase):
    """OP15 residency checks use OP15's layers; the server keeps the union."""

    def setUp(self) -> None:
        # v7: OP15 held Gemma on HTP0-2 and replaces HTP0 with the Qwen layers it owns.
        self.source = (shard("HTP0", GEMMA, 0b0001), shard("HTP1", GEMMA, 0b0010), shard("HTP2", GEMMA, 0b0100))
        self.target = (shard("HTP0", QWEN, PRIMARY_MASK, generation=2), *self.source[1:])
        self.transport = phone_transport_contract(two_phone_parameters())
        self.session = object.__new__(DirectPhoneFfnSession)
        self.session._launch = SimpleNamespace(
            phone_shards=self.source, transport=self.transport,
            remote_hashes={"model:" + QWEN: QWEN}, execution=union_contract(0b0111),
        )
        self.session._remote_root = "/data/local/tmp/resident"
        self.session.configuration = SimpleNamespace(
            diagnostic_host="192.0.2.1", diagnostic_port=20_000,
            model_paths_by_artifact={QWEN: "/data/qwen.gguf"},
        )
        self.session._verified_remote_hash_by_path = {"/data/qwen.gguf": QWEN}
        self.manifest = SimpleNamespace(artifact_sha256=QWEN)

    def command(self, primary_mask: int = PRIMARY_MASK, target=None):
        target = self.target if target is None else target
        return SimpleNamespace(
            artifact_sha256=QWEN,
            ticket_id="phone-helper-layout-4",
            adapter_parameters=two_phone_parameters(primary_mask),
            replacement_authorization=PhoneSessionReplacementAuthorization.create(
                selected_session_id="HTP0", source_shards=self.source, target_shards=target,
            ),
            transition=SimpleNamespace(phone_shards=target, changed_phone_session_ids=("HTP0",)),
            execution_contract=SimpleNamespace(
                phone_shards=target, remote_resident_ffn=None, phone_device_id=h.OP15),
        )

    def test_partial_replacement_compares_the_primary_phone_layers(self) -> None:
        with mock.patch(REPLACEMENT, return_value=union_contract()):
            self.assertTrue(self.session.supports_partial_reconfiguration(
                self.command(), self.manifest, self.transport))
        # the primary row still has to match OP15's shards exactly
        with mock.patch(REPLACEMENT, return_value=union_contract(0b110011)):
            self.assertFalse(self.session.supports_partial_reconfiguration(
                self.command(primary_mask=0b0011), self.manifest, self.transport))
        # a helper binding that does not cover the resident slice is refused, never projected
        with mock.patch(REPLACEMENT, return_value=union_contract(0b111111 | 1 << 6)):
            with self.assertRaisesRegex(PhysicalAdapterError, "phone helpers"):
                self.session.supports_partial_reconfiguration(self.command(), self.manifest, self.transport)

    def test_single_phone_replacement_is_unchanged(self) -> None:
        command = self.command()
        command.adapter_parameters = {
            key: value for key, value in command.adapter_parameters.items() if key != "phone_helpers"}
        with mock.patch(REPLACEMENT, return_value=union_contract(PRIMARY_MASK)):
            self.assertTrue(self.session.supports_partial_reconfiguration(command, self.manifest, self.transport))
        with mock.patch(REPLACEMENT, return_value=union_contract()):
            self.assertFalse(self.session.supports_partial_reconfiguration(command, self.manifest, self.transport))

    def test_reuse_compares_the_primary_phone_layers(self) -> None:
        self.session._launch.phone_shards = self.target
        with mock.patch(IDENTITY, return_value=union_contract()):
            self.assertTrue(self.session.supports(self.command(), self.manifest, self.transport))
        with mock.patch(IDENTITY, return_value=union_contract(0b110011)):
            self.assertFalse(self.session.supports(
                self.command(primary_mask=0b0011), self.manifest, self.transport))

    def test_session_start_launches_only_the_primary_layers(self) -> None:
        session = object.__new__(DirectPhoneFfnSession)
        session._launch = None
        session._phone_kernel_release = lambda: "kernel"
        session._multi_session_manifest = lambda shards: ("rows", "sha256:" + "4" * 64)
        session.configuration = SimpleNamespace(
            resident_workers_path="/w", resident_router_path="/r",
            worker_paths_by_artifact={QWEN: "/worker"}, model_paths_by_artifact={QWEN: "/model"},
            serial=h.OP15_SERIAL, adb_port=5037, minimum_usb_speed_mbps=5000,
        )
        with mock.patch(PREFLIGHT, return_value=union_contract()), mock.patch(
            "research_dev.scheduler.adapters.phone_session_ops.preflight.verify_android_usb_restored"
        ):
            execution = session._start_contract(self.command(), self.manifest, self.transport, None)[2]
        self.assertEqual((execution.layer_mask, execution.layer_indices, execution.layers),
                         (PRIMARY_MASK, (0, 1, 2, 3), "0-3"))

    def test_rig_accepts_the_v7_gemma_to_qwen_shard_replacement(self) -> None:
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.Lock()
        rig._live_executors = {}
        rig._phone_residency = None
        rig._phone_activity = None
        rig._direct_phone_session = self.session
        rig._phone_residency_resources = lambda _executor: ((h.OP15,), ("op15-ram",), ("op15-htp",))
        command = self.command()
        command.helper_only = True
        command.participant = SimpleNamespace(executor_id="physical:op15-phone", endpoint="http://desktop:1")
        command.transition.evictions = ()
        command.transition.prepares_device_ids = (h.OP15,)
        with mock.patch(IDENTITY, return_value=union_contract()), mock.patch(
            REPLACEMENT, return_value=union_contract()
        ), mock.patch(
            "research_dev.scheduler.adapters.heterogeneous_rig.physical_transition_stop_set", return_value=()
        ):
            state = rig._begin_transition_execution(command, self.manifest, 0)
            self.assertTrue(state.direct_phone_partial_requested)
            self.assertTrue(state.direct_phone_reconfigurable)
            self.assertEqual(rig._transition_conflicting_executors(command, state), ())

    def test_persistent_residency_records_the_primary_phone_layers(self) -> None:
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._phone_residency_resources = lambda _executor: ((h.OP15,), ("op15-ram",), ("op15-htp",))
        command = self.command()
        command.operator_plan = {"plan": 1}
        command.participant = SimpleNamespace(executor_id="physical:op15-phone", endpoint="http://desktop:1")
        command.phone_layout_generation = 2
        command.adapter_parameters = {**command.adapter_parameters,
                                      "phone_shard_set_geometry_sha256": "sha256:" + "9" * 64}
        direct_phone = SimpleNamespace(
            phone_shards=self.target,
            load_count_by_session={"HTP0": 2, "HTP1": 1, "HTP2": 1},
            column_quantum_by_session={"HTP0": 32, "HTP1": 32, "HTP2": 32},
            max_tokens_by_session={"HTP0": 4, "HTP1": 4, "HTP2": 4},
        )
        previous = _PersistentPhoneResidency(
            executor_id="physical:op15-phone", endpoint="http://desktop:1", phone_shards=self.source,
            layout_geometry_sha256="sha256:" + "8" * 64,
            manifests_by_artifact={GEMMA: SimpleNamespace(artifact_sha256=GEMMA)},
            parameters_by_artifact={GEMMA: {}}, operator_plans_by_artifact={GEMMA: {}},
            executions_by_artifact={GEMMA: union_contract(0b0111)},
            load_count_by_session={"HTP0": 1, "HTP1": 1, "HTP2": 1},
            column_quantum_by_session={"HTP0": 32, "HTP1": 32, "HTP2": 32},
            max_tokens_by_session={"HTP0": 4, "HTP1": 4, "HTP2": 4},
            generation=1, participant_device_ids=(h.OP15,), replacement_resource_ids=("op15-ram",),
            session_resource_ids=("op15-htp",),
        )
        with mock.patch(RESIDENCY, return_value=union_contract()):
            current = rig._persistent_phone_residency_state(
                command, SimpleNamespace(artifact_sha256=QWEN, block_count=8), direct_phone,
                fallback_generation=2, previous=previous)
        self.assertEqual(current.executions_by_artifact[QWEN].layer_mask, PRIMARY_MASK)
        self.assertEqual(current.executions_by_artifact[QWEN].layer_indices, (0, 1, 2, 3))


class PrimaryPhoneTicketTests(unittest.TestCase):
    """The projection on real tickets: two-phone narrows to OP15, single-phone is the identity."""

    def setUp(self) -> None:
        self.model_scope = h.TemporaryModel()
        self.model = self.model_scope.__enter__()
        self.addCleanup(self.model_scope.__exit__)

    def _command(self, catalog):
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(catalog)
        scheduler.register_model_manifest(self.model)
        ticket = scheduler.submit_automated_request(
            h.request("activation"), self.model.model_id, h.snapshot(self.model, catalog),
            selection_mode="adaptive-decode")
        ticket = scheduler.wait_runtime_request(
            "activation", time.monotonic_ns() - ticket.decision.start_us * 1_000)
        return interpret_runtime_ticket(ticket)

    def test_two_phone_ticket_narrows_to_the_primary_phone(self) -> None:
        command = self._command(h.runtime_catalog(self.model, declaration=h.co_helpers()))
        resident = phone_ffn_resident_contract(command, self.model)
        self.assertEqual(resident.layer_mask, UNION_MASK)
        primary = primary_phone_ffn_contract(command, resident)
        self.assertEqual((primary.layer_mask, primary.layer_indices), (PRIMARY_MASK, (0, 1, 2, 3)))
        self.assertEqual(replace(primary, layer_mask=resident.layer_mask,
                                 layer_indices=resident.layer_indices), resident)

    def test_single_phone_ticket_is_unchanged(self) -> None:
        command = self._command(h.runtime_catalog(self.model))
        resident = phone_ffn_resident_contract(command, self.model)
        self.assertIs(primary_phone_ffn_contract(command, resident), resident)


class AdaptiveSplitFrontierTests(unittest.TestCase):
    """The split representative of a resident-envelope group survives a tight refinement budget."""

    def setUp(self) -> None:
        self.model_scope = h.TemporaryModel()
        self.model = self.model_scope.__enter__()
        self.addCleanup(self.model_scope.__exit__)

    def _generate(self, catalog, refinement_budget=None):
        visits = {}
        original = RouteRoughCostMixin._rough_visits
        bounded = route_compiler.BoundedPlacementCompiler

        def spy(compiler, *arguments, **keywords):
            visits["rows"] = original(compiler, *arguments, **keywords)
            visits["patterns"] = {row.route_key: row for row in compiler._patterns(self.model)}
            return visits["rows"]

        def budgeted(search_budget, refinement_budget_default):
            return bounded(search_budget=search_budget, refinement_budget=(
                refinement_budget_default if refinement_budget is None else refinement_budget))

        with mock.patch.object(RouteRoughCostMixin, "_rough_visits", spy), mock.patch.object(
            route_compiler, "BoundedPlacementCompiler",
            lambda search_budget, refinement_budget: budgeted(search_budget, refinement_budget),
        ):
            scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
            scheduler.register_runtime_capabilities(catalog)
            scheduler.register_model_manifest(self.model)
            values = scheduler.generate_automated_candidates(
                h.request("frontier"), self.model.model_id, h.snapshot(self.model, catalog))
        return values, visits

    def _assert_split_survives(self, catalog, owners) -> None:
        _values, visits = self._generate(catalog)
        mandatory = [row for row in visits["rows"] if row.mandatory]
        groups = {row.required_group for row in mandatory
                  if str(row.required_group).startswith("resident-envelope:")}
        self.assertTrue(groups)
        for group in groups:
            members = [row for row in visits["rows"] if row.required_group == group]
            families = {visits["patterns"][row.route_key].route_family for row in members}
            self.assertEqual(families, {"operator_offload", "operator_split"})
            # the offload visit is cheaper, so energy alone would make it the representative
            self.assertLess(
                min(row.rough_energy_uj for row in members
                    if visits["patterns"][row.route_key].route_family == "operator_offload"),
                min(row.rough_energy_uj for row in members
                    if visits["patterns"][row.route_key].route_family == "operator_split"))
            chosen = [row for row in members if row.mandatory]
            self.assertEqual([visits["patterns"][row.route_key].route_family for row in chosen],
                             ["operator_split"])
        # no budget beyond the mandatory visits: only mandatory representatives are refined
        values, _ = self._generate(catalog, refinement_budget=len(mandatory))
        _baseline, policies, envelope = adaptive_decode_policies(values, self.model, catalog, 30)
        self.assertIsNotNone(envelope)
        self.assertEqual(envelope.route_family, "operator_split")
        self.assertTrue(policies)
        # the full device set on every width; per-device subsets drive a strict part of it
        full = [row for row in policies if row.device_layer_masks == owners]
        self.assertEqual({row.split_fraction_ppm for row in full},
                         {row.split_fraction_ppm for row in policies})
        self.assertTrue(all(set(row.device_layer_masks) < set(owners) for row in policies
                            if row not in full))

    def test_two_phone_split_envelope_survives(self) -> None:
        self._assert_split_survives(
            h.runtime_catalog(self.model, declaration=h.co_helpers()),
            ((h.OP15, PRIMARY_MASK), (h.PIXEL, h.PIXEL_MASK)))

    def test_single_phone_split_envelope_survives(self) -> None:
        self._assert_split_survives(h.runtime_catalog(self.model), ())


class ReprovisioningExclusionTests(unittest.TestCase):
    """Re-provisioning layouts of a two-phone model never place the co-helper's fixed layers on OP15."""

    def test_follow_layouts_use_only_the_primary_phone_operators(self) -> None:
        from research_dev.scheduler._internal.phone_shards import (
            PhoneFfnResidencyDemand, generate_mixed_ffn_residency_layouts,
        )
        from test_multi_session_phone import manifest as session_manifest, session

        with h.TemporaryModel() as model:
            catalog = h.runtime_catalog(model, declaration=h.co_helpers())
            families = [row for row in catalog.composite_executors
                        if row.helper_device_id == h.OP15 and row.assisted_operator_kind == "ffn"]
            self.assertTrue(families)
            owned = {row.operator_ids for row in families}
            self.assertEqual(len(owned), 1)
            owned = next(iter(owned))
        # the residency-evidence demand takes a family's operator ids as its allowed set
        synthetic = session_manifest(8)
        allowed = tuple(operator_id for operator_id in owned
                        if int(operator_id.split(":")[1]) < 8)
        demand = PhoneFfnResidencyDemand(synthetic, 100, 8, "split-row", allowed_operator_ids=allowed)
        sessions = tuple(replace(session(index), resident_memory_limit_bytes=1 << 20) for index in range(3))
        layouts = generate_mixed_ffn_residency_layouts((demand,), sessions, phone_wide_limit_bytes=1 << 30)
        self.assertTrue(layouts)
        covered = 0
        for layout in layouts:
            for row in layout.shards:
                covered |= row.layer_mask
        self.assertFalse(covered & h.PIXEL_MASK)
        self.assertTrue(covered)


if __name__ == "__main__":
    unittest.main()
