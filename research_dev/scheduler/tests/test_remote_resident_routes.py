"""Route generation for remote-resident desktop parents: fail-closed declaration checks,
owner binding from the READY phone layout, and memory demands that omit the phone-owned
tensors while pinning the owning sessions."""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import ModelManifest  # noqa: E402
from research_dev.scheduler._internal.phone_shards import (  # noqa: E402
    PhoneFfnResidencyLayout,
    PhoneFfnShardPlacement,
)
from research_dev.scheduler._internal.route_generation.common import (  # noqa: E402
    RouteGenerationError,
    _Pattern,
)
from research_dev.scheduler._internal.route_generation.remote_resident import (  # noqa: E402
    REMOTE_RESIDENT_FFN_PARAMETER,
    REMOTE_RESIDENT_OWNER_NOT_READY,
    RouteRemoteResidentMixin,
    parse_remote_resident_declaration,
)
from research_dev.scheduler._internal.runtime_capabilities import (  # noqa: E402
    RuntimePhoneSessionCapability,
)
from research_dev.scheduler._internal.runtime_cost import RuntimeMemoryDemand  # noqa: E402
from research_dev.scheduler._internal.runtime_plan import (  # noqa: E402
    RuntimeRemoteResidentFfn,
    RuntimeRemoteResidentSession,
    remote_resident_tensor_ids,
)
from research_dev.scheduler._internal.types import canonical_sha256  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ModelManifest.from_json(json.loads(
    (ROOT / "campaigns/burstgpt/data/GEMMA_MANIFEST.json").read_text(encoding="ascii")
))
MASK_0_7 = 0xFF
GEOMETRY = "sha256:" + hashlib.sha256(b"HTP0-geometry").hexdigest()
SHARD_SHA = "sha256:" + hashlib.sha256(b"HTP0-shard").hexdigest()
INDEX_SHA = "sha256:" + hashlib.sha256(b"index").hexdigest()


def omitted_bytes(mask: int = MASK_0_7) -> int:
    tensor_by_id = MANIFEST.tensor_by_id
    return sum(tensor_by_id[name].nbytes for name in remote_resident_tensor_ids(mask))


def declaration(**overrides) -> dict:
    payload = RuntimeRemoteResidentFfn(
        parent_artifact_sha256=MANIFEST.artifact_sha256,
        layer_mask=MASK_0_7,
        dtype="f16",
        omitted_bytes=omitted_bytes(),
        tensor_ids=remote_resident_tensor_ids(MASK_0_7),
        shard_index_sha256=INDEX_SHA,
        sessions=(RuntimeRemoteResidentSession(
            session_id="HTP0",
            endpoint="session://op15/HTP0",
            layer_mask=MASK_0_7,
            shard_sha256=SHARD_SHA,
            resident_geometry_sha256=GEOMETRY,
            resident_bytes=omitted_bytes(),
            remote_path="/data/local/tmp/gemma24/HTP0.ffn.gguf",
        ),),
    ).to_json()
    payload.update(overrides)
    payload.pop("geometry_sha256", None)
    return payload


def session_capability(**overrides) -> RuntimePhoneSessionCapability:
    values = dict(
        session_id="HTP0",
        device_id="op15-phone",
        endpoint="session://op15/HTP0",
        worker_identity_sha256="sha256:" + "9" * 64,
        memory_resource_id="op15-ram:session:HTP0",
        resident_memory_limit_bytes=3_000_000_000,
        shared_compute_resource_id="op15-htp",
        shared_transport_resource_ids=("op15-functionfs",),
        supported_layer_mask=(1 << 24) - 1,
        maximum_columns=MANIFEST.feed_forward_length,
        column_quantum=512,
        supported_data_types=("F16",),
        batch_plans=("split-row",),
        ready=True,
        residency_state="hot",
        resident_artifact_sha256=MANIFEST.artifact_sha256,
        resident_geometry_sha256=GEOMETRY,
    )
    values.update(overrides)
    return RuntimePhoneSessionCapability(**values)


def ready_layout(generation: int = 3, **shard_overrides) -> PhoneFfnResidencyLayout:
    values = dict(
        artifact_sha256=MANIFEST.artifact_sha256,
        session_id="HTP0",
        endpoint="session://op15/HTP0",
        memory_resource_id="op15-ram:session:HTP0",
        operator_ids=tuple(f"layer:{index}:ffn" for index in range(8)),
        layer_mask=MASK_0_7,
        maximum_columns=MANIFEST.feed_forward_length,
        resident_bytes=omitted_bytes(),
        resident_geometry_sha256=GEOMETRY,
        operator_plan_sha256="sha256:" + "7" * 64,
    )
    values.update(shard_overrides)
    shard = PhoneFfnShardPlacement(**values)
    return PhoneFfnResidencyLayout(
        shards=(shard,),
        queued_work_by_artifact={MANIFEST.artifact_sha256: 1},
        queue_benefit_by_artifact={MANIFEST.artifact_sha256: 1},
        queue_benefit_by_session={"HTP0": 1},
        queue_benefit=1,
        transition_cost=0,
        transition_cost_by_session={},
        objective=-1,
        objective_kind="queue_energy_delta_uj",
        changed_session_ids=(),
        geometry_sha256=canonical_sha256({
            "artifact_sha256": MANIFEST.artifact_sha256,
            "shards": [{"geometry_sha256": shard.resident_geometry_sha256, "session_id": "HTP0"}],
        }),
        session_generation_by_id={} if generation == 0 else {"HTP0": generation},
    )


class Compiler(RouteRemoteResidentMixin):
    """The mixin with only the state it reads."""

    def __init__(self, coordinator_parameters, *, sessions, layout):
        coordinator = SimpleNamespace(
            executor_id="physical:hot:desktop-remote",
            adapter_parameters=coordinator_parameters,
        )
        self.catalog = SimpleNamespace(
            executor_by_device={
                "desktop-cpu": SimpleNamespace(phone_sessions=()),
                "op15-phone": SimpleNamespace(phone_sessions=tuple(sessions)),
            },
            composite_executor_by_id={coordinator.executor_id: coordinator},
        )
        self._phone_residency_layout = layout


def pattern() -> _Pattern:
    return _Pattern(
        route_key="coordinated:physical:hot:desktop-remote",
        route_family="layer_placement",
        assignments={},
        assisted_operator_kind=None,
        split_axis="none",
        split_fraction_ppm=0,
        overlap_kind="serial",
        coordinator_device_id="desktop-cpu",
        coordinator_executor_id="physical:hot:desktop-remote",
    )


def coordinator_parameters(group_json: dict, **overrides) -> dict:
    values = {
        REMOTE_RESIDENT_FFN_PARAMETER: json.dumps(group_json, sort_keys=True),
        "phone_device_id": "op15-phone",
        "ffn_resident_layer_mask": (1 << 24) - 1,
        "ffn_resident_columns": MANIFEST.feed_forward_length,
        "ffn_max_tokens": 512,
        "ubatch_size": 512,
    }
    values.update(overrides)
    return values


class DeclarationTests(unittest.TestCase):
    def test_session_capacity_can_exceed_exact_resident_ffn_width(self) -> None:
        compiler = Compiler(
            coordinator_parameters(declaration()),
            sessions=(session_capability(maximum_columns=17408),),
            layout=ready_layout(),
        )
        group = compiler._remote_resident_group(MANIFEST, pattern())
        bound, reasons = compiler._remote_resident_owner_status(MANIFEST, group)
        self.assertEqual(reasons, ())
        self.assertIsNotNone(bound)
        compiler._phone_residency_layout = ready_layout(maximum_columns=8192)
        bound, reasons = compiler._remote_resident_owner_status(MANIFEST, group)
        self.assertIsNone(bound)
        self.assertTrue(reasons)

    def test_declaration_round_trip_and_generation_rule(self) -> None:
        group = parse_remote_resident_declaration(json.dumps(declaration()))
        self.assertEqual(group.layer_mask, MASK_0_7)
        bound = declaration()
        bound["sessions"][0]["session_generation"] = 2
        with self.assertRaisesRegex(RouteGenerationError, "session generations"):
            parse_remote_resident_declaration(json.dumps(bound))
        with self.assertRaisesRegex(RouteGenerationError, "JSON text"):
            parse_remote_resident_declaration(5)
        with self.assertRaisesRegex(RouteGenerationError, "invalid"):
            parse_remote_resident_declaration("{not json")

    def test_manifest_and_session_mismatches_fail_closed(self) -> None:
        cases = {
            "omitted bytes": (declaration(omitted_bytes=omitted_bytes() - 1), {}),
            "another artifact": (
                declaration(parent_artifact_sha256="sha256:" + "b" * 64), {}
            ),
            "resident layer mask": (declaration(), {"ffn_resident_layer_mask": 0x0F}),
            "complete FFN columns": (declaration(), {"ffn_resident_columns": 8192}),
            "ffn_max_tokens": (declaration(), {"ffn_max_tokens": 64}),
            "phone runtime parameters": (declaration(), {"phone_device_id": "other"}),
        }
        for message, (group_json, overrides) in cases.items():
            with self.subTest(message=message):
                compiler = Compiler(
                    coordinator_parameters(group_json, **overrides),
                    sessions=(session_capability(),), layout=ready_layout(),
                )
                with self.assertRaisesRegex(RouteGenerationError, message):
                    compiler._remote_resident_group(MANIFEST, pattern())
        unknown = Compiler(coordinator_parameters(declaration()), sessions=(), layout=None)
        with self.assertRaisesRegex(RouteGenerationError, "unknown"):
            unknown._remote_resident_group(MANIFEST, pattern())
        narrow = Compiler(
            coordinator_parameters(declaration()),
            sessions=(session_capability(maximum_columns=8192, column_quantum=512),),
            layout=None,
        )
        with self.assertRaisesRegex(RouteGenerationError, "complete shards"):
            narrow._remote_resident_group(MANIFEST, pattern())

    def test_absent_declaration_means_no_group(self) -> None:
        compiler = Compiler({"cuda_graph_mode": "default"}, sessions=(), layout=None)
        self.assertIsNone(compiler._remote_resident_group(MANIFEST, pattern()))
        self.assertEqual(compiler._remote_resident_tensor_ids(None), frozenset())


class OwnerBindingTests(unittest.TestCase):
    def test_ready_owner_binds_its_generation(self) -> None:
        compiler = Compiler(
            coordinator_parameters(declaration()),
            sessions=(session_capability(),), layout=ready_layout(generation=4),
        )
        group = compiler._remote_resident_group(MANIFEST, pattern())
        bound, reasons = compiler._remote_resident_owner_status(MANIFEST, group)
        self.assertEqual(reasons, ())
        self.assertEqual(bound.session_generation_by_id, {"HTP0": 4})
        self.assertEqual(
            bound.sessions[0].operator_plan_sha256,
            compiler._phone_residency_layout.shards[0].operator_plan_sha256,
        )
        self.assertEqual(bound.geometry_sha256, group.geometry_sha256)
        demands = compiler._remote_resident_memory_demands(MANIFEST, group, ())
        self.assertEqual(len(demands), 1)
        demand = demands[0]
        self.assertEqual(demand.demand_id, "phone-session:HTP0:weights")
        self.assertEqual(demand.kind, "session_residency_constraint")
        self.assertEqual(demand.resource_id, "op15-ram:session:HTP0")
        self.assertEqual(demand.required_bytes, omitted_bytes())
        self.assertEqual(demand.resident_bytes, omitted_bytes(), "READY owner is credited")
        self.assertEqual(demand.additional_bytes, 0)

    def test_missing_or_stale_owner_blocks_without_credit(self) -> None:
        for label, sessions, layout in (
            ("no layout", (session_capability(),), None),
            ("generation zero", (session_capability(),), ready_layout(generation=0)),
            ("other geometry", (session_capability(),), ready_layout(
                resident_geometry_sha256="sha256:" + "c" * 64)),
            ("session cold", (session_capability(residency_state="cold", ready=False,
                                                 resident_artifact_sha256=None,
                                                 resident_geometry_sha256=None,
                                                 unavailable_reason="reloading"),),
             ready_layout()),
        ):
            with self.subTest(label=label):
                compiler = Compiler(
                    coordinator_parameters(declaration()), sessions=sessions, layout=layout,
                )
                group = compiler._remote_resident_group(MANIFEST, pattern())
                bound, reasons = compiler._remote_resident_owner_status(MANIFEST, group)
                self.assertIsNone(bound)
                self.assertEqual(len(reasons), 1)
                self.assertTrue(reasons[0].startswith(REMOTE_RESIDENT_OWNER_NOT_READY + ":HTP0"))
                if label == "session cold":
                    self.assertTrue(reasons[0].endswith(":RELOADING"))
                demands = compiler._remote_resident_memory_demands(MANIFEST, group, ())
                self.assertEqual(demands[0].resident_bytes, 0, "no credit before verification")
                self.assertEqual(demands[0].additional_bytes, omitted_bytes())

    def test_owner_cannot_double_as_helper_shard(self) -> None:
        compiler = Compiler(
            coordinator_parameters(declaration()),
            sessions=(session_capability(),), layout=ready_layout(),
        )
        group = compiler._remote_resident_group(MANIFEST, pattern())
        helper_demand = RuntimeMemoryDemand(
            demand_id="phone-session:HTP0:weights",
            resource_id="op15-ram:session:HTP0",
            kind="session_residency_constraint",
            required_bytes=1, resident_bytes=0, lifetime="resident",
            share_key="x:y", device_id="op15-phone",
        )
        with self.assertRaisesRegex(RouteGenerationError, "helper shard"):
            compiler._remote_resident_memory_demands(MANIFEST, group, (helper_demand,))

    def test_omitted_tensor_ids_are_the_dense_ffn_groups(self) -> None:
        compiler = Compiler(
            coordinator_parameters(declaration()),
            sessions=(session_capability(),), layout=ready_layout(),
        )
        group = compiler._remote_resident_group(MANIFEST, pattern())
        omitted = compiler._remote_resident_tensor_ids(group)
        self.assertEqual(len(omitted), 24)
        self.assertIn("blk.0.ffn_gate.weight", omitted)
        self.assertNotIn("blk.8.ffn_gate.weight", omitted)
        self.assertNotIn("blk.0.attn_q.weight", omitted)


if __name__ == "__main__":
    unittest.main()
