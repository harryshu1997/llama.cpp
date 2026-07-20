#!/usr/bin/env python3
"""Regressions for the S10-E2A v3 claim-path repair."""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"

import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import aggregate  # noqa: E402
import anchors  # noqa: E402
import e2a_canon as canon  # noqa: E402
import resolver  # noqa: E402
from test_e2a import allow_anchor, bundle, load_route  # noqa: E402


def resolved_slot(slot):
    plan = canon.load_strict(FIXTURES / "plan.json")
    manifest = canon.load_strict(FIXTURES / "manifest.json")
    timeline = canon.load_strict(
        FIXTURES / f"timelines/tl.slot{slot:02d}.json")
    outcomes = canon.load_strict(
        FIXTURES / f"outcomes/outcomes.slot{slot:02d}.json")
    lifecycle = canon.load_strict(
        FIXTURES / f"lifecycle/lc.slot{slot:02d}.json")
    resolved = resolver.resolve_requests(
        manifest, outcomes, timeline, FIXTURES, f"slot{slot}", slot // 2)
    return plan, manifest, timeline, outcomes, lifecycle, resolved


class BundleContract(unittest.TestCase):

    def test_duplicate_slot_is_rejected_before_dictionary_conversion(self):
        with bundle() as work:
            index = work.load("bundle.json")
            index["slots"].append(dict(index["slots"][0]))
            work.write("bundle.json", index, reseal=False)
            with self.assertRaises(aggregate.AggregateError) as ctx:
                work.evaluate()
            self.assertEqual(ctx.exception.code, "E_SLOT_DUPLICATE")

    def test_float_slot_index_is_rejected_by_the_type_gate(self):
        with bundle() as work:
            index = work.load("bundle.json")
            index["slots"][1]["slot_index"] = 1.0
            work.write("bundle.json", index, reseal=False)
            with self.assertRaises(aggregate.AggregateError) as ctx:
                work.evaluate()
            self.assertEqual(ctx.exception.code, "E_TYPE")


class RealizedRequestIdentity(unittest.TestCase):

    def test_manifest_rewrite_cannot_reuse_an_old_output_artifact(self):
        fields = {
            "input_sha256": "realized_input_sha256",
            "prompt_token_ids_sha256":
                "realized_prompt_token_ids_sha256",
            "decode_params_digest": "realized_decode_params_digest",
            "stop_set_digest": "realized_stop_set_digest",
        }
        for manifest_field, outcome_field in fields.items():
            with self.subTest(field=manifest_field):
                manifest = canon.load_strict(FIXTURES / "manifest.json")
                outcomes = canon.load_strict(
                    FIXTURES / "outcomes/outcomes.slot00.json")
                timeline = canon.load_strict(
                    FIXTURES / "timelines/tl.slot00.json")
                entry = manifest["entries"][0]
                entry[manifest_field] = canon.digest({
                    "r3-rewrite": manifest_field,
                })
                entry["entry_sha256"] = canon.digest({
                    key: value for key, value in entry.items()
                    if key != "entry_sha256"
                })
                canon.seal(manifest)
                outcomes["manifest_sha256"] = manifest["record_sha256"]
                item = outcomes["outcomes"][0]
                item["manifest_entry_sha256"] = entry["entry_sha256"]
                item[outcome_field] = entry[manifest_field]
                canon.seal(outcomes)
                with self.assertRaises(resolver.ResolveError) as ctx:
                    resolver.resolve_requests(
                        manifest, outcomes, timeline, FIXTURES,
                        "request-rewrite", 0)
                self.assertEqual(ctx.exception.code, "E_OUTPUT_BINDING")


class RouteEvidence(unittest.TestCase):

    def test_treatment_requires_a_phone_exec(self):
        plan, _manifest, timeline, _outcomes, lifecycle, resolved = \
            resolved_slot(1)
        execution = next(action for action in lifecycle["actions"]
                         if action["action_kind"] == "EXEC")
        execution["execution_domain"] = "SERVER"
        execution["device_identity"] = plan["server_device_ids"][0]
        execution["backend_kind"] = "CUDA"
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(
                lifecycle, timeline, plan["slots"][1], resolved, plan,
                load_route(timeline["role"]))
        self.assertEqual(ctx.exception.code, "E_ROUTE_NODE_MISMATCH")

    def test_control_cannot_use_a_phone(self):
        plan, _manifest, timeline, _outcomes, lifecycle, resolved = \
            resolved_slot(0)
        execution = next(action for action in lifecycle["actions"]
                         if action["action_kind"] == "EXEC")
        execution["execution_domain"] = "PHONE"
        execution["device_identity"] = plan["phone_device_ids"][0]
        execution["backend_kind"] = "HTP"
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(
                lifecycle, timeline, plan["slots"][0], resolved, plan,
                load_route(timeline["role"]))
        self.assertEqual(ctx.exception.code, "E_ROUTE_NODE_MISMATCH")

    def test_phone_exec_requires_h2d_and_d2h(self):
        plan, _manifest, timeline, _outcomes, lifecycle, resolved = \
            resolved_slot(1)
        execution = next(action for action in lifecycle["actions"]
                         if action["action_kind"] == "EXEC")
        d2h = next(action for action in lifecycle["actions"]
                   if action["action_kind"] == "D2H")
        d2h["enqueue_us"] = execution["start_us"]
        d2h["start_us"] = execution["start_us"]
        d2h["end_us"] = execution["start_us"]
        d2h["ack_us"] = execution["start_us"]
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(
                lifecycle, timeline, plan["slots"][1], resolved, plan,
                load_route(timeline["role"]))
        self.assertEqual(ctx.exception.code, "E_ACTION_DURATION")

    def test_timeline_route_digest_is_plan_bound(self):
        plan = canon.load_strict(FIXTURES / "plan.json")
        timeline = canon.load_strict(
            FIXTURES / "timelines/tl.slot01.json")
        timeline["route_schedule_digest"] = "f" * 64
        with self.assertRaises(aggregate.AggregateError) as ctx:
            aggregate._bind_timeline_to_plan(
                timeline, plan, plan["slots"][1], 1)
        self.assertEqual(ctx.exception.code, "E_ROUTE_BINDING")


class LifecycleAndWarmup(unittest.TestCase):

    def test_lifecycle_set_id_is_bound(self):
        plan, _manifest, timeline, _outcomes, lifecycle, resolved = \
            resolved_slot(0)
        lifecycle["set_id"] = "set.unrelated"
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(
                lifecycle, timeline, plan["slots"][0], resolved, plan,
                load_route(timeline["role"]))
        self.assertEqual(ctx.exception.code, "E_LIFECYCLE_BINDING")

    def test_work_action_must_name_the_covering_lease(self):
        plan, _manifest, timeline, _outcomes, lifecycle, resolved = \
            resolved_slot(0)
        execution = next(action for action in lifecycle["actions"]
                         if action["action_kind"] == "EXEC")
        execution["lease_id"] = "lease.unrelated"
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(
                lifecycle, timeline, plan["slots"][0], resolved, plan,
                load_route(timeline["role"]))
        self.assertEqual(ctx.exception.code, "E_LEASE_COVERAGE")

    def test_nonzero_warmup_count_is_fail_closed(self):
        plan = canon.load_strict(FIXTURES / "plan.json")
        plan["warmup_count"] = 1
        canon.seal(plan)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_plan(plan, FIXTURES)
        self.assertEqual(ctx.exception.code, "E_WARMUP_UNBOUND")


class WallAndAnchorEvidence(unittest.TestCase):

    def test_server_wall_bundle_requires_a_capability(self):
        plan = canon.load_strict(FIXTURES / "plan.json")
        plan["scope"] = "SERVER_WALL"
        plan["instrument_kind"] = "EXTERNAL_WALL_METER"
        plan["board_uuids"] = []
        plan["included_rails"] = ["SERVER/AC_INPUT"]
        plan["excluded_rails"] = []
        plan["server_wall_capability_record_sha256"] = "a" * 64
        canon.seal(plan)
        resolver.resolve_plan(plan, FIXTURES)
        index = {
            "server_wall_capability_path": None,
            "server_wall_capability_sha256": None,
        }
        with self.assertRaises(aggregate.AggregateError) as ctx:
            aggregate._resolve_wall_capability(index, plan, FIXTURES)
        self.assertEqual(ctx.exception.code, "E_INCOMPLETE_WALL")

    def test_resolved_wall_capability_is_checked_for_each_timeline(self):
        plan = canon.load_strict(FIXTURES / "plan.json")
        anchor = canon.load_strict(FIXTURES / "plan_anchor.json")
        ledger = canon.load_strict(FIXTURES / "ledger.json")
        manifest = canon.load_strict(FIXTURES / "manifest.json")
        index = canon.load_strict(FIXTURES / "bundle.json")
        terminal = resolver.resolve_ledger(ledger, plan, anchor)
        routes = {
            "OPTIMIZED_SERVER_ONLY_CONTROL":
                load_route("OPTIMIZED_SERVER_ONLY_CONTROL"),
            "Q_PIM_TREATMENT": load_route("Q_PIM_TREATMENT"),
        }
        refusal = ["E_CAPABILITY_UNCERTIFIED: calibration is not certified"]
        with resolver.resolution_context(FIXTURES), \
                mock.patch.object(
                    aggregate.e2_comparator, "check_wall_capability",
                    return_value=refusal) as check:
            with self.assertRaises(aggregate.AggregateError) as ctx:
                aggregate._resolve_slots(
                    index, plan, terminal, manifest, FIXTURES,
                    {"record_sha256": "a" * 64}, routes)
        self.assertEqual(ctx.exception.code, "E_CAPABILITY_UNCERTIFIED")
        check.assert_called()

    def test_plan_anchor_binds_external_experiment_identity(self):
        plan = canon.load_strict(FIXTURES / "plan.json")
        receipt = canon.load_strict(FIXTURES / "plan_anchor.json")
        receipt["experiment_identity"] = "experiment.other"
        canon.seal(receipt)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.bind_plan_anchor(receipt, plan)
        self.assertEqual(ctx.exception.code, "E_ANCHOR_IDENTITY")

    def test_tsa_leaf_certificate_pin_is_enforced(self):
        receipt = canon.load_strict(FIXTURES / "plan_anchor.json")
        with mock.patch.multiple(
                anchors,
                FROZEN_TSA_ROOT_SHA256=receipt["trust_root_sha256"],
                FROZEN_TSA_LEAF_SHA256="f" * 64,
                FROZEN_TSA_POLICY_OID=receipt["tsa_policy_oid"]):
            with self.assertRaises(anchors.AnchorError) as ctx:
                anchors.check_trust_root(receipt)
        self.assertEqual(ctx.exception.code, "E_ANCHOR_TRUST_ROOT")

    def test_tsa_policy_pin_is_enforced(self):
        receipt = canon.load_strict(FIXTURES / "plan_anchor.json")
        with mock.patch.multiple(
                anchors,
                FROZEN_TSA_ROOT_SHA256=receipt["trust_root_sha256"],
                FROZEN_TSA_LEAF_SHA256=receipt["tsa_leaf_cert_sha256"],
                FROZEN_TSA_POLICY_OID="1.2.3.4.999"):
            with self.assertRaises(anchors.AnchorError) as ctx:
                anchors.check_trust_root(receipt)
        self.assertEqual(ctx.exception.code, "E_ANCHOR_TRUST_ROOT")

    def test_enumeration_must_contain_exactly_one_plan(self):
        with allow_anchor(), bundle() as work:
            plan = work.load("plan.json")
            proof_path = work.path("anchors/commitment_proof.json")
            proof = json.loads(proof_path.read_text(encoding="ascii"))
            proof["committed_plan_sha256s"].append("f" * 64)
            proof["committed_plan_count"] = 2
            proof["committed_plan_set_sha256"] = canon.digest(
                proof["committed_plan_sha256s"])
            proof_text = json.dumps(proof, indent=2, sort_keys=True) + "\n"
            proof_path.write_text(proof_text, encoding="ascii")
            proof_digest = canon.sha256_bytes(proof_text.encode("ascii"))
            for name in ("plan_anchor.json", "ledger_close.json"):
                receipt = work.load(name)
                receipt["commitment_proof_sha256"] = proof_digest
                receipt["committed_plan_count"] = 2
                receipt["committed_plan_set_sha256"] = \
                    proof["committed_plan_set_sha256"]
                work.write(name, receipt)
            work.rebind_anchor_chain()
            with self.assertRaises(resolver.ResolveError) as ctx:
                work.evaluate()
            self.assertEqual(ctx.exception.code, "E_ANCHOR_COMPLETENESS")
            self.assertEqual(plan["record_sha256"],
                             proof["committed_plan_sha256s"][0])


class SecureArtifactOpen(unittest.TestCase):

    def test_intermediate_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            root = base / "root"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            (outside / "secret").write_bytes(b"outside")
            (root / "link").symlink_to(outside, target_is_directory=True)
            with resolver.resolution_context(root):
                with self.assertRaises(resolver.ResolveError) as ctx:
                    resolver.read_once(
                        "link/secret", canon.sha256_bytes(b"outside"), root)
            self.assertEqual(ctx.exception.code, "E_PATH")

    def test_context_stays_bound_to_original_root_inode(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            root = base / "root"
            original = base / "original"
            (root / "sub").mkdir(parents=True)
            (root / "sub/file").write_bytes(b"original")
            digest = canon.sha256_bytes(b"original")
            with resolver.resolution_context(root):
                os.rename(root, original)
                (root / "sub").mkdir(parents=True)
                (root / "sub/file").write_bytes(b"replacement")
                _path, data = resolver.read_once("sub/file", digest, root)
            self.assertEqual(data, b"original")


if __name__ == "__main__":
    unittest.main(verbosity=2)
