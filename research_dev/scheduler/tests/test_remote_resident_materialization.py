"""Catalog materialization of a remote-resident owner parent (Phase 2, target A).

The parent is derived from the model's desktop control and its primary phone family; its owner
geometries must be the ones the shard packer gives the same sessions, so the READY fixed-residency
layout binds it. Every inconsistency fails closed, including a transport identity that does not
qualify a whole-ubatch payload.
"""
from __future__ import annotations

from dataclasses import replace
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler.adapters import (  # noqa: E402
    CatalogMaterializationError,
    add_phone_session_memory_pools,
    materialize_remote_resident_owner,
    remote_resident_owner_executor_id,
    remote_resident_payload_qualification,
)
from research_dev.scheduler.adapters.ffn_shards import FfnShardIndex  # noqa: E402
from research_dev.scheduler.config import FixedPhoneResidencyConfiguration  # noqa: E402
from research_dev.scheduler._internal.runtime_capabilities import RuntimePhoneSessionCapability  # noqa: E402
from research_dev.scheduler._internal.route_generation.remote_resident import (  # noqa: E402
    REMOTE_RESIDENT_FFN_PARAMETER,
    parse_remote_resident_declaration,
)
from research_dev.scheduler._internal.runtime_plan import remote_resident_tensor_ids  # noqa: E402

try:
    from . import test_offline_phone_residency as offline_fixture
    from . import two_phone_harness as two_phone
except ImportError:
    import test_offline_phone_residency as offline_fixture
    import two_phone_harness as two_phone

IDENTITY = "sha256:" + "d" * 64
# two of the fixture's three sessions; 6 F16 layers of the 14B model fill one 3.2 GB session
OWNER_MASKS = {"HTP0": 0x3F, "HTP1": 0x3F << 6}


def shard_index(manifest, masks=OWNER_MASKS, *, weight_type="F16", columns=None, **row_overrides):
    """A real FFN shard index with one full-width shard per owner session."""
    rows = []
    for session_id, mask in sorted(masks.items()):
        rows.append({
            "path": session_id + ".ffn.gguf", "parent_sha256": manifest.artifact_sha256,
            "shard_sha256": "sha256:" + format(mask, "064x"), "layer_mask": format(mask, "x"),
            "columns": columns or manifest.feed_forward_length, "n_ff": manifest.feed_forward_length,
            "shard_bytes": 1 << 30, "weight_type": weight_type, "session_id": session_id,
            **row_overrides,
        })
    return FfnShardIndex.from_index({
        "schema": "s42-ffn-shard-index-v1", "parent_sha256": manifest.artifact_sha256, "shards": rows,
    }, "/data/local/tmp/qwen-owner")


def owner_fixture(*, payload_bytes: int = 16 << 20, usbfs_bytes: int = 256 << 20):
    """The replay rig (Qwen 14B F16, three op15 sessions) with an identity-bound FunctionFS transport."""
    offline_fixture.OfflinePhoneResidencyTests.setUpClass()
    scheduler, requests, snapshot = offline_fixture.OfflinePhoneResidencyTests.replay_runtime(
        load_evidence=False)
    catalog = scheduler._runtime_capabilities
    catalog = replace(catalog, placement_profile=replace(catalog.placement_profile, links=tuple(
        replace(row, maximum_payload_bytes=payload_bytes, usbfs_available_bytes=usbfs_bytes,
                qualification_identity_sha256=IDENTITY, evidence_ids=(IDENTITY,))
        if row.transport_generation.startswith("functionfs") else row
        for row in catalog.placement_profile.links)))
    manifest = scheduler.runtime_model_manifest(next(iter(requests)))
    return SimpleNamespace(scheduler=scheduler, requests=requests, snapshot=snapshot,
                           catalog=catalog, manifest=manifest)


def fixed_residency(manifest, masks=OWNER_MASKS) -> FixedPhoneResidencyConfiguration:
    return FixedPhoneResidencyConfiguration("remote-owner-test", tuple(
        (session_id, manifest.artifact_sha256, mask, manifest.feed_forward_length)
        for session_id, mask in masks.items()
    ))


class RemoteResidentMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = owner_fixture()
        self.manifest = self.fixture.manifest
        self.control = self.fixture.catalog.desktop_control_by_artifact[self.manifest.artifact_sha256]

    def materialize(self, catalog=None, masks=OWNER_MASKS, index=None):
        return materialize_remote_resident_owner(
            self.fixture.catalog if catalog is None else catalog, self.manifest, masks,
            shard_index(self.manifest, masks) if index is None else index,
        )

    def test_parent_is_the_desktop_control_with_the_phone_owner_declared(self) -> None:
        catalog = self.materialize()
        source = catalog.composite_executor_by_id[self.control.executor_id]
        remote = catalog.composite_executor_by_id[remote_resident_owner_executor_id(self.control.executor_id)]
        group = parse_remote_resident_declaration(remote.adapter_parameters[REMOTE_RESIDENT_FFN_PARAMETER])
        self.assertEqual(remote.maturity, "CALIBRATION_PENDING")
        self.assertIsNone(remote.assisted_operator_kind)
        self.assertEqual(remote.operator_placements, source.operator_placements)
        self.assertEqual(remote.participant_device_ids, (*source.participant_device_ids, "op15-phone"))
        self.assertEqual((group.dtype, group.layer_mask, group.session_ids), ("f16", 0xFFF, ("HTP0", "HTP1")))
        self.assertEqual(group.omitted_bytes, sum(
            self.manifest.tensor_by_id[key].nbytes for key in group.tensor_ids))
        parameters = remote.adapter_parameters
        self.assertEqual(
            (parameters["ffn_max_tokens"], parameters["ffn_resident_layer_mask"],
             parameters["ffn_resident_columns"], parameters["ffn_weight_buffer_layout"]),
            (source.adapter_parameters["ubatch_size"], 0xFFF, self.manifest.feed_forward_length,
             "resident-superset"))
        self.assertTrue(all(row.energy_maturity == "CALIBRATION_PENDING"
                            for row in catalog.transitions if row.executor_id == remote.executor_id))
        # the desktop control and every other composite are unchanged
        self.assertEqual(catalog.desktop_control_profiles, self.fixture.catalog.desktop_control_profiles)
        self.assertEqual(
            {row.executor_id: row for row in catalog.composite_executors if row.executor_id != remote.executor_id},
            {row.executor_id: row for row in self.fixture.catalog.composite_executors},
        )

    def test_owner_geometry_is_the_packed_fixed_residency_geometry(self) -> None:
        catalog = self.materialize()
        remote = catalog.composite_executor_by_id[remote_resident_owner_executor_id(self.control.executor_id)]
        group = parse_remote_resident_declaration(remote.adapter_parameters[REMOTE_RESIDENT_FFN_PARAMETER])
        scheduler = self.fixture.scheduler
        scheduler.register_runtime_capabilities(catalog)
        scheduler.configure_fixed_phone_residency(fixed_residency(self.manifest))
        snapshot = self.fixture.snapshot
        plan = scheduler.plan_offline_phone_residency(
            scheduler.fixed_phone_residency_requests(), snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        self.assertEqual(
            {row.session_id: (row.layer_mask, row.resident_bytes, row.resident_geometry_sha256)
             for row in plan.target_layout.shards},
            {row.session_id: (row.layer_mask, row.resident_bytes, row.resident_geometry_sha256)
             for row in group.sessions},
        )

    def test_materialization_is_idempotent_and_refuses_a_different_parent(self) -> None:
        catalog = self.materialize()
        self.assertIs(self.materialize(catalog), catalog)
        with self.assertRaisesRegex(CatalogMaterializationError, "different remote-resident"):
            self.materialize(catalog, masks={"HTP0": 0x3F})

    def test_inconsistent_declarations_fail_closed(self) -> None:
        manifest = self.manifest
        cases = (
            ("not discovered", {"HTP9": 0x3F}, None),
            ("overlap", {"HTP0": 0x3F, "HTP1": 0x30}, None),
            ("CPU FFN layers", {"HTP0": 1 << 30}, None),
            ("shard is absent", OWNER_MASKS, shard_index(manifest, {"HTP0": 0x3F})),
            ("shard is absent", OWNER_MASKS, shard_index(manifest, weight_type="Q4_0")),
            ("shard is absent", OWNER_MASKS, shard_index(manifest, columns=manifest.feed_forward_length // 2)),
            ("layer masks are invalid", {"HTP0": 0}, shard_index(manifest)),
        )
        for message, masks, index in cases:
            with self.subTest(message=message), self.assertRaisesRegex(CatalogMaterializationError, message):
                self.materialize(masks=masks, index=index or shard_index(manifest, masks))

    def test_unsupported_owner_types_fail_closed(self) -> None:
        catalog = self.fixture.catalog
        f16_only = replace(catalog, executors=tuple(
            replace(row, phone_sessions=tuple(
                replace(session, supported_data_types=("Q4_0",)) for session in row.phone_sessions))
            for row in catalog.executors))
        with self.assertRaisesRegex(CatalogMaterializationError, "cannot hold its complete shard"):
            self.materialize(f16_only)

    def test_assisted_copy_relocation_keys_are_not_inherited(self) -> None:
        catalog = self.fixture.catalog
        share = {"ffn_host_share_release": 1, "ffn_host_share_drop_cache": 0, "ffn_host_share_populate": 1}
        catalog = replace(catalog, composite_executors=tuple(
            replace(row, adapter_parameters={**row.adapter_parameters, **share})
            if row.baseline_executor_id == self.control.executor_id else row
            for row in catalog.composite_executors))
        remote = self.materialize(catalog).composite_executor_by_id[
            remote_resident_owner_executor_id(self.control.executor_id)]
        self.assertFalse(set(share) & set(remote.adapter_parameters))


class QuantizedOwnerTests(unittest.TestCase):
    """Two-phone harness (Q8_0 FFN weights): OP15 session HTP0 owns layers 0-3."""

    def catalog(self, *, declaration=None):
        catalog = two_phone.runtime_catalog(self.model, declaration=declaration)
        session = RuntimePhoneSessionCapability(
            session_id="HTP0", device_id=two_phone.OP15, endpoint="session://op15-phone/HTP0",
            worker_identity_sha256="sha256:" + "9" * 64, memory_resource_id="op15-ram",
            resident_memory_limit_bytes=1 << 30, shared_compute_resource_id="op15-htp",
            shared_transport_resource_ids=("op15-functionfs", "usb-root"), supported_layer_mask=0xFF,
            maximum_columns=self.model.feed_forward_length, column_quantum=16,
            supported_data_types=("F16", "Q8_0"), batch_plans=("split-row",), ready=True,
            residency_state="cold",
        )
        links = tuple(
            replace(row, transport_generation="functionfs-dmabuf-async-ring-v2", allocator="devmem",
                    full_duplex=True, queue_depth=4, concurrent_streams=4, maximum_payload_bytes=1 << 20,
                    usbfs_available_bytes=1 << 30, slot_safety_bytes=65536, transport_profile_id=row.link_id,
                    qualification_identity_sha256=IDENTITY, evidence_ids=(IDENTITY,))
            if two_phone.OP15 in (row.source_device, row.target_device) else row
            for row in catalog.placement_profile.links)
        return replace(
            catalog,
            placement_profile=replace(catalog.placement_profile, links=links),
            executors=tuple(replace(row, phone_sessions=(session,)) if row.device_id == two_phone.OP15
                            else row for row in catalog.executors),
        )

    def setUp(self) -> None:
        scope = two_phone.TemporaryModel()
        self.model = scope.__enter__()
        self.addCleanup(scope.__exit__)

    def test_q8_0_owner_declares_its_quantized_group(self) -> None:
        catalog = materialize_remote_resident_owner(
            self.catalog(), self.model, {"HTP0": 0b1111},
            shard_index(self.model, {"HTP0": 0b1111}, weight_type="Q8_0"))
        remote = catalog.composite_executor_by_id["physical:two:desktop:remote-owner"]
        group = parse_remote_resident_declaration(remote.adapter_parameters[REMOTE_RESIDENT_FFN_PARAMETER])
        self.assertEqual((group.dtype, group.layer_mask), ("q8_0", 0b1111))
        # the harness family's decode-only relocation keys stay on the assisted copies
        self.assertNotIn("ffn_host_share_release", remote.adapter_parameters)
        with self.assertRaisesRegex(CatalogMaterializationError, "shard is absent"):
            materialize_remote_resident_owner(
                self.catalog(), self.model, {"HTP0": 0b1111}, shard_index(self.model, {"HTP0": 0b1111}))

    def test_three_q8_0_owner_sessions_account_their_exact_layers(self) -> None:
        # the production shape on the 32B Q8_0 model: three HTP sessions, consecutive layer blocks
        masks = {"HTP0": 0b11, "HTP1": 0b1100, "HTP2": 0b110000}
        sessions = tuple(
            RuntimePhoneSessionCapability(
                session_id=session_id, device_id=two_phone.OP15, endpoint="session://op15-phone/" + session_id,
                worker_identity_sha256="sha256:" + "9" * 64, memory_resource_id="op15-ram:session:" + session_id,
                resident_memory_limit_bytes=1 << 30, shared_compute_resource_id="op15-htp",
                shared_transport_resource_ids=("op15-functionfs", "usb-root"), supported_layer_mask=0xFF,
                maximum_columns=self.model.feed_forward_length, column_quantum=16,
                supported_data_types=("Q8_0",), batch_plans=("split-row",), ready=True, residency_state="cold")
            for session_id in masks)
        catalog = self.catalog()
        catalog = replace(
            catalog,
            placement_profile=add_phone_session_memory_pools(catalog.placement_profile, sessions),
            executors=tuple(replace(row, phone_sessions=sessions) if row.device_id == two_phone.OP15 else row
                            for row in catalog.executors))
        catalog = materialize_remote_resident_owner(
            catalog, self.model, masks, shard_index(self.model, masks, weight_type="Q8_0"))
        remote = catalog.composite_executor_by_id["physical:two:desktop:remote-owner"]
        group = parse_remote_resident_declaration(remote.adapter_parameters[REMOTE_RESIDENT_FFN_PARAMETER])
        self.assertEqual((group.dtype, group.layer_mask, group.session_ids), ("q8_0", 0b111111, tuple(masks)))
        def dense_bytes(mask):  # gate/up/down only: the norms stay on the desktop
            return sum(self.model.tensor_by_id[key].nbytes for key in remote_resident_tensor_ids(mask))

        for row in group.sessions:
            self.assertEqual(row.resident_bytes, dense_bytes(masks[row.session_id]))
            self.assertEqual(self.model.tensor_by_id[f"blk.{row.layer_indices[0]}.ffn_up.weight"].quantization, "Q8_0")
        self.assertEqual(group.omitted_bytes, dense_bytes(0b111111))
        self.assertEqual(len({row.resident_geometry_sha256 for row in group.sessions}), 3)

    def test_static_co_helpers_fail_closed(self) -> None:
        with self.assertRaisesRegex(CatalogMaterializationError, "co-helper"):
            materialize_remote_resident_owner(
                self.catalog(declaration=two_phone.co_helpers()), self.model, {"HTP0": 0b1111},
                shard_index(self.model, {"HTP0": 0b1111}, weight_type="Q8_0"))


class RemoteResidentPayloadTests(unittest.TestCase):
    def test_whole_ubatch_payload_must_be_qualified(self) -> None:
        # 5120 x 512 tokens x 2 bytes: the 32B owner's prefill call
        needed = 5120 * 512 * 2
        for payload, usbfs, qualified in ((needed, 256 << 20, True), (needed - 1, 256 << 20, False),
                                          (61_440, 256 << 20, False), (needed, 16 << 20, False)):
            with self.subTest(payload=payload, usbfs=usbfs):
                fixture = owner_fixture(payload_bytes=payload, usbfs_bytes=usbfs)
                manifest = fixture.manifest
                if qualified:
                    catalog = materialize_remote_resident_owner(
                        fixture.catalog, manifest, OWNER_MASKS, shard_index(manifest))
                    control = catalog.desktop_control_by_artifact[manifest.artifact_sha256]
                    remote = catalog.composite_executor_by_id[
                        remote_resident_owner_executor_id(control.executor_id)]
                    value = remote_resident_payload_qualification(catalog, remote.adapter_parameters)
                    self.assertEqual((value["required_payload_bytes"], value["qualified"]), (needed, True))
                    continue
                with self.assertRaisesRegex(CatalogMaterializationError, "needs " + str(needed) + " bytes"):
                    materialize_remote_resident_owner(
                        fixture.catalog, manifest, OWNER_MASKS, shard_index(manifest))


class CampaignCatalogTests(unittest.TestCase):
    def test_campaign_glue_is_a_no_op_without_the_opt_in(self) -> None:
        from research_dev.scheduler.campaigns.burstgpt.catalog import remote_resident_owner_catalog

        catalog = owner_fixture().catalog
        configuration = SimpleNamespace(campaign=SimpleNamespace(remote_resident_owner=None))
        self.assertIs(remote_resident_owner_catalog(catalog, configuration), catalog)

    def test_campaign_glue_uses_the_fixed_assignment_and_the_model_shard_index(self) -> None:
        import tempfile

        from research_dev.scheduler.campaigns.burstgpt.catalog import (
            MaterializationError,
            remote_resident_owner_catalog,
        )
        from research_dev.scheduler.config import RemoteResidentOwnerConfiguration

        fixture = owner_fixture()
        manifest = fixture.manifest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "MANIFEST.json").write_text(json.dumps(manifest.to_json()), encoding="ascii")
            index = shard_index(manifest)
            (root / "FFN_SHARDS.json").write_text(json.dumps({
                "schema": "s42-ffn-shard-index-v1", "parent_sha256": manifest.artifact_sha256,
                "shards": [{
                    "path": row.remote_path.rsplit("/", 1)[1], "parent_sha256": manifest.artifact_sha256,
                    "shard_sha256": row.shard_sha256, "layer_mask": format(row.layer_mask, "x"),
                    "columns": row.columns, "n_ff": row.n_ff, "shard_bytes": row.shard_bytes,
                    "weight_type": row.weight_type, "session_id": row.session_hint,
                } for row in index.records],
            }), encoding="ascii")
            model = SimpleNamespace(
                model_id=manifest.model_id, checked_manifest_path=root / "MANIFEST.json",
                phone_ffn_shard_index_path=root / "FFN_SHARDS.json",
                phone_ffn_shard_directory="/data/local/tmp/qwen-owner",
            )

            def configuration(fixed):
                return SimpleNamespace(
                    campaign=SimpleNamespace(
                        remote_resident_owner=RemoteResidentOwnerConfiguration(manifest.model_id),
                        fixed_phone_residency=fixed),
                    models=SimpleNamespace(assisted_models=(model,)),
                )

            catalog = remote_resident_owner_catalog(fixture.catalog, configuration(fixed_residency(manifest)))
            control = catalog.desktop_control_by_artifact[manifest.artifact_sha256]
            remote = catalog.composite_executor_by_id[remote_resident_owner_executor_id(control.executor_id)]
            group = parse_remote_resident_declaration(remote.adapter_parameters[REMOTE_RESIDENT_FFN_PARAMETER])
            self.assertEqual(group.shard_index_sha256, FfnShardIndex.load(
                root / "FFN_SHARDS.json", "/data/local/tmp/qwen-owner").index_sha256)
            partial = FixedPhoneResidencyConfiguration("t", ((
                "HTP0", manifest.artifact_sha256, 0x3F, manifest.feed_forward_length // 2),))
            with self.assertRaisesRegex(MaterializationError, "full-width owner shards"):
                remote_resident_owner_catalog(fixture.catalog, configuration(partial))


if __name__ == "__main__":
    unittest.main()
