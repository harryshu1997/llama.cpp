"""Remote-resident FFN contract: identity, fail-closed validation, hash stability."""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler._internal.runtime_plan import (  # noqa: E402
    RuntimeExecutionContract,
    RuntimePlanError,
    RuntimeRemoteResidentFfn,
    RuntimeRemoteResidentSession,
    remote_resident_tensor_ids,
)
from research_dev.scheduler._internal.runtime_capabilities import (  # noqa: E402
    RuntimeCapabilityError,
    RuntimeCompositeOperatorPlacement,
    RuntimeDesktopControlProfile,
    desktop_control_placement_payload,
)


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


ARTIFACT = _sha("gemma")
MASK_0_7 = 0xFF
MASK_8_15 = 0xFF00


def session(session_id: str, mask: int, generation: int = 0) -> RuntimeRemoteResidentSession:
    return RuntimeRemoteResidentSession(
        session_id=session_id,
        endpoint=f"usb:{session_id}",
        layer_mask=mask,
        shard_sha256=_sha("shard-" + session_id),
        resident_geometry_sha256=_sha("geometry-" + session_id),
        resident_bytes=2_831_157_472,
        remote_path=f"/data/local/tmp/gemma24/{session_id}.ffn.gguf",
        session_generation=generation,
    )


def group(*sessions: RuntimeRemoteResidentSession, mask: int | None = None) -> RuntimeRemoteResidentFfn:
    layer_mask = mask if mask is not None else sum(row.layer_mask for row in sessions)
    return RuntimeRemoteResidentFfn(
        parent_artifact_sha256=ARTIFACT,
        layer_mask=layer_mask,
        dtype="f16",
        omitted_bytes=3 * 8 * 3840 * 15360 * 2 * len(sessions),
        tensor_ids=remote_resident_tensor_ids(layer_mask),
        shard_index_sha256=_sha("index"),
        sessions=sessions,
    )


def placements() -> list[RuntimeCompositeOperatorPlacement]:
    return [
        RuntimeCompositeOperatorPlacement("layer:0:attention", "desktop-cpu", None, "none", 0),
        RuntimeCompositeOperatorPlacement("layer:0:ffn", "desktop-cpu", None, "none", 0),
    ]


class RemoteResidentGroupTests(unittest.TestCase):
    def test_tensor_dependencies_are_the_complete_dense_groups(self) -> None:
        self.assertEqual(
            remote_resident_tensor_ids(0b101),
            ("blk.0.ffn_down.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight",
             "blk.2.ffn_down.weight", "blk.2.ffn_gate.weight", "blk.2.ffn_up.weight"),
        )

    def test_group_round_trips_and_binds_generations(self) -> None:
        template = group(session("HTP0", MASK_0_7), session("HTP1", MASK_8_15))
        self.assertFalse(template.bound)
        self.assertEqual(template.session_ids, ("HTP0", "HTP1"))
        again = RuntimeRemoteResidentFfn.from_json(json.loads(json.dumps(template.to_json())))
        self.assertEqual(again, template)
        self.assertEqual(again.geometry_sha256, template.geometry_sha256)
        bound = template.with_session_generations({"HTP0": 3, "HTP1": 1})
        self.assertTrue(bound.bound)
        self.assertEqual(bound.session_generation_by_id, {"HTP0": 3, "HTP1": 1})
        self.assertEqual(bound.geometry_sha256, template.geometry_sha256,
                         "generations bind execution, not the placement identity")
        self.assertNotEqual(bound.to_json(), template.to_json())
        with self.assertRaises(RuntimePlanError):
            template.with_session_generations({"HTP0": 2})
        with self.assertRaises(RuntimePlanError):
            template.with_session_generations({"HTP0": 0, "HTP1": 1})

    def test_operator_binding_round_trips_without_changing_placement(self) -> None:
        template = group(session("HTP0", MASK_0_7))
        bound = template.with_session_generations(
            {"HTP0": 3}, operator_plan_by_id={"HTP0": _sha("loaded-plan")}
        )
        self.assertEqual(bound.sessions[0].operator_plan_sha256, _sha("loaded-plan"))
        self.assertEqual(bound.geometry_sha256, template.geometry_sha256)
        self.assertEqual(RuntimeRemoteResidentFfn.from_json(bound.to_json()), bound)
        self.assertEqual(
            bound.with_session_generations({"HTP0": 4}).sessions[0].operator_plan_sha256,
            _sha("loaded-plan"),
        )

    def test_incomplete_or_overlapping_coverage_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimePlanError, "cover the layer mask exactly"):
            group(session("HTP0", MASK_0_7), mask=MASK_0_7 | MASK_8_15)
        with self.assertRaisesRegex(RuntimePlanError, "overlap"):
            group(session("HTP0", MASK_0_7), session("HTP1", 0x0F), mask=MASK_0_7)
        with self.assertRaisesRegex(RuntimePlanError, "complete dense"):
            RuntimeRemoteResidentFfn(
                parent_artifact_sha256=ARTIFACT, layer_mask=MASK_0_7, dtype="f16",
                omitted_bytes=1, tensor_ids=remote_resident_tensor_ids(MASK_0_7)[:-1],
                shard_index_sha256=_sha("index"), sessions=(session("HTP0", MASK_0_7),),
            )
        with self.assertRaisesRegex(RuntimePlanError, "dtype"):
            RuntimeRemoteResidentFfn(
                parent_artifact_sha256=ARTIFACT, layer_mask=MASK_0_7, dtype="q4_0",
                omitted_bytes=1, tensor_ids=remote_resident_tensor_ids(MASK_0_7),
                shard_index_sha256=_sha("index"), sessions=(session("HTP0", MASK_0_7),),
            )
        with self.assertRaisesRegex(RuntimePlanError, "requires phone sessions"):
            RuntimeRemoteResidentFfn(
                parent_artifact_sha256=ARTIFACT, layer_mask=MASK_0_7, dtype="f16",
                omitted_bytes=1, tensor_ids=remote_resident_tensor_ids(MASK_0_7),
                shard_index_sha256=_sha("index"), sessions=(),
            )
        with self.assertRaisesRegex(RuntimePlanError, "absolute device path"):
            RuntimeRemoteResidentSession(
                session_id="HTP0", endpoint="usb:HTP0", layer_mask=1,
                shard_sha256=_sha("s"), resident_geometry_sha256=_sha("g"),
                resident_bytes=1, remote_path="relative.gguf",
            )

    def test_tampered_geometry_hash_is_rejected(self) -> None:
        payload = group(session("HTP0", MASK_0_7)).to_json()
        payload["geometry_sha256"] = _sha("other")
        with self.assertRaisesRegex(RuntimePlanError, "geometry hash differs"):
            RuntimeRemoteResidentFfn.from_json(payload)


class ExecutionContractTests(unittest.TestCase):
    def test_plain_desktop_contract_json_is_unchanged(self) -> None:
        plain = RuntimeExecutionContract.desktop()
        self.assertNotIn("remote_resident_ffn", plain.to_json())
        self.assertIsNone(plain.remote_resident_ffn)

    def test_remote_resident_requires_the_desktop_mode(self) -> None:
        remote = group(session("HTP0", MASK_0_7, 1))
        contract = RuntimeExecutionContract.desktop(remote)
        self.assertEqual(contract.to_json()["remote_resident_ffn"], remote.to_json())
        with self.assertRaisesRegex(RuntimePlanError, "desktop parent contract"):
            RuntimeExecutionContract(
                execution_mode="adaptive-split", initial_split_fraction_ppm=0,
                allowed_adaptive_fractions_ppm=(0, 500_000), batch_plan="split-row",
                maximum_batch_size=4, queue_depth=1, phone_device_id="op15",
                phone_endpoint="usb", operator_kind="ffn", remote_resident_ffn=remote,
            )


class DesktopPlacementIdentityTests(unittest.TestCase):
    def test_placement_hash_moves_only_when_the_group_is_present(self) -> None:
        base = desktop_control_placement_payload(ARTIFACT, placements())
        self.assertNotIn("remote_resident_ffn", base)
        remote = group(session("HTP0", MASK_0_7))
        reduced = desktop_control_placement_payload(ARTIFACT, placements(), "default", remote)
        self.assertEqual(reduced["remote_resident_ffn"], remote.placement_payload())
        self.assertNotIn("session_generation", json.dumps(reduced))
        bound = remote.with_session_generations({"HTP0": 5})
        self.assertEqual(
            desktop_control_placement_payload(ARTIFACT, placements(), "default", bound),
            reduced,
            "generations are execution state, not placement identity",
        )
        with self.assertRaisesRegex(RuntimeCapabilityError, "another artifact"):
            desktop_control_placement_payload(_sha("qwen"), placements(), "default", remote)

    def test_control_profile_round_trip_and_hash_distinctness(self) -> None:
        remote = group(session("HTP0", MASK_0_7))
        full = RuntimeDesktopControlProfile(
            profile_id="desktop-control:full", artifact_sha256=ARTIFACT,
            executor_id="physical:hot:desktop", operator_placements=tuple(placements()),
            maturity="QUALIFIED", evidence_ids=(_sha("e"),),
        )
        reduced = RuntimeDesktopControlProfile(
            profile_id="desktop-control:reduced", artifact_sha256=ARTIFACT,
            executor_id="physical:hot:desktop-remote", operator_placements=tuple(placements()),
            maturity="QUALIFIED", evidence_ids=(_sha("e"),), remote_resident_ffn=remote,
        )
        self.assertNotEqual(full.placement_sha256, reduced.placement_sha256)
        self.assertNotIn("remote_resident_ffn", full.to_json())
        again = RuntimeDesktopControlProfile.from_json(json.loads(json.dumps(reduced.to_json())))
        self.assertEqual(again.placement_sha256, reduced.placement_sha256)
        self.assertEqual(again.remote_resident_ffn, remote)
        tampered = reduced.to_json()
        tampered["remote_resident_ffn"]["omitted_bytes"] += 1
        with self.assertRaisesRegex(RuntimePlanError, "geometry hash differs"):
            RuntimeDesktopControlProfile.from_json(tampered)
        swapped = reduced.to_json()
        swapped["remote_resident_ffn"] = group(session("HTP1", MASK_8_15)).to_json()
        with self.assertRaisesRegex(RuntimeCapabilityError, "placement hash differs"):
            RuntimeDesktopControlProfile.from_json(swapped)


if __name__ == "__main__":
    unittest.main()
