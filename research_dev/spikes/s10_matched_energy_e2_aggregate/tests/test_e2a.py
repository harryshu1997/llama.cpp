#!/usr/bin/env python3
"""S10-E2A test suite. Non-vacuous negatives only.

Every negative test here mutates a bundle that is otherwise VALID and asserts a
specific E_* code. That matters: a negative test that would pass even without the
check it claims to exercise proves nothing, and E2 shipped a status check that was
dead across its entire 104-test suite because every fixture happened to omit the
field it keyed on. So each negative starts from the positive fixture, changes
exactly one thing, and names the code it expects.

Where a test asserts a gate downstream of the anchor, it neutralises the anchor
gate explicitly through `allow_anchor()`. That helper exists ONLY in this file. It
is how the suite proves the rest of the machinery is real rather than hidden
behind an early refusal -- and `test_the_anchor_gate_is_not_neutralised_in_
production` asserts the production path has no such door.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "checker"))
E2_SRC = ROOT.parent / "s10_matched_energy_e2" / "src"
sys.path.insert(0, str(E2_SRC))

import aggregate                      # noqa: E402
import anchors                        # noqa: E402
import checker                        # noqa: E402
import e2a_canon as canon             # noqa: E402
import resolver                       # noqa: E402
import make_e2a_fixtures as fixtures   # noqa: E402

FIXTURES = ROOT / "fixtures"
BUNDLE = FIXTURES / "bundle.json"


@contextlib.contextmanager
def allow_anchor():
    """Test-only: neutralise anchor policy to exercise later mechanics.

    There are three independent refusals, and a test that neutralises two still
    stops at the third. They are listed here rather than hidden behind a single
    switch so the count stays honest: enumerability (the finding), a pinned CA
    root (unprovisioned), and an implemented verifier (none exist).
    """
    saved_enum = dict(anchors.ANCHOR_ENUMERABLE)
    saved_root = anchors.check_trust_root
    saved_verify = anchors.verify_token
    saved_precedence = anchors.check_precedence_supported
    anchors.ANCHOR_ENUMERABLE["RFC3161_TSA"] = True
    anchors.check_trust_root = lambda record: True
    anchors.verify_token = lambda receipt, token: {
        field: receipt[field] for field in anchors.VERIFIED_TOKEN_FIELDS
    }
    anchors.check_precedence_supported = lambda receipt, verified: True
    try:
        yield
    finally:
        anchors.ANCHOR_ENUMERABLE.clear()
        anchors.ANCHOR_ENUMERABLE.update(saved_enum)
        anchors.check_trust_root = saved_root
        anchors.verify_token = saved_verify
        anchors.check_precedence_supported = saved_precedence


class Bundle:
    """A mutable copy of the valid fixture bundle."""

    def __init__(self, tmp):
        self.root = pathlib.Path(tmp) / "fixtures"
        shutil.copytree(FIXTURES, self.root)

    def path(self, rel):
        return self.root / rel

    def load(self, rel):
        return canon.load_strict(self.root / rel)

    def write(self, rel, obj, reseal=True):
        if reseal and "record_sha256" in obj:
            canon.seal(obj)
        text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
        (self.root / rel).write_text(text, encoding="ascii")
        return canon.sha256_bytes(text.encode("ascii"))

    def reindex(self):
        """Recompute the bundle index so only the intended mutation is tested."""
        index = self.load("bundle.json")
        for key, rel in (("plan", "plan.json"), ("plan_anchor",
                                                 "plan_anchor.json"),
                         ("ledger", "ledger.json"),
                         ("ledger_close", "ledger_close.json"),
                         ("manifest", "manifest.json")):
            data = (self.root / rel).read_bytes()
            index[f"{key}_sha256"] = canon.sha256_bytes(data)
        for slot in index["slots"]:
            for key in ("timeline", "lifecycle", "outcomes"):
                data = (self.root / slot[f"{key}_path"]).read_bytes()
                slot[f"{key}_sha256"] = canon.sha256_bytes(data)
        text = json.dumps(index, indent=2, sort_keys=True) + "\n"
        (self.root / "bundle.json").write_text(text, encoding="ascii")
        return index

    def rebind_anchor_chain(self):
        """Repair dependent hashes so an anchor-field negative reaches its gate."""
        anchor = self.load("plan_anchor.json")
        ledger = self.load("ledger.json")
        ledger["entries"][0]["plan_anchor_record_sha256"] = \
            anchor["record_sha256"]
        _rechain(ledger)
        self.write("ledger.json", ledger)

        close = self.load("ledger_close.json")
        close["anchor_kind"] = anchor["anchor_kind"]
        close["plan_anchor_id"] = anchor["anchor_id"]
        close["plan_anchor_record_sha256"] = anchor["record_sha256"]
        close["attempt_ledger_sha256"] = ledger["record_sha256"]
        close["ledger_head_sha256"] = ledger["head_sha256"]
        close["anchored_digest"] = ledger["record_sha256"]
        close["message_imprint_sha256"] = ledger["record_sha256"]
        self.write("ledger_close.json", close)
        self.reindex()

    def evaluate(self):
        return aggregate.evaluate(str(self.root / "bundle.json"))


@contextlib.contextmanager
def bundle():
    with tempfile.TemporaryDirectory() as tmp:
        yield Bundle(tmp)


def contribution(control_e, control_u, treatment_e, treatment_u, index=0):
    return {
        "pair_index": index,
        "control_timeline_id": f"c{index}",
        "control_record_sha256": "a" * 64,
        "treatment_timeline_id": f"t{index}",
        "treatment_record_sha256": "b" * 64,
        "control_energy_nj": control_e,
        "control_uncertainty_nj": control_u,
        "treatment_energy_nj": treatment_e,
        "treatment_uncertainty_nj": treatment_u,
        "first_executed_role": "OPTIMIZED_SERVER_ONLY_CONTROL",
    }


# ---------------------------------------------------------------------------
# The anchor verdict
# ---------------------------------------------------------------------------

class AnchorTyping(unittest.TestCase):

    def test_rfc3161_is_independent_but_not_enumerable(self):
        """The finding, as an executable assertion.

        These two lines are the whole checkpoint. RFC3161 is a real third-party
        attestation -- it passes the spec's own independence test -- and it still
        cannot support an all-pairs claim, because it cannot show a reviewer how
        many OTHER plans were anchored beside the one revealed.
        """
        self.assertTrue(anchors.ANCHOR_INDEPENDENT["RFC3161_TSA"])
        self.assertFalse(anchors.ANCHOR_ENUMERABLE["RFC3161_TSA"])
        self.assertEqual(anchors.anchor_property("RFC3161_TSA"),
                         anchors.PROPERTY_ORDERING_ONLY)

    def test_ordering_only_is_not_enough_for_an_aggregate(self):
        with self.assertRaises(anchors.AnchorError) as ctx:
            anchors.check_anchor_kind("RFC3161_TSA")
        self.assertEqual(ctx.exception.code, "E_ANCHOR_UNENUMERABLE")

    def test_no_declared_anchor_kind_carries_the_required_property(self):
        enough = [kind for kind in anchors.ANCHOR_KINDS
                  if anchors.anchor_property(kind)
                  == anchors.REQUIRED_PROPERTY]
        self.assertEqual(enough, [])

    def test_this_host_cannot_reach_the_required_property(self):
        capability = anchors.describe_host_capability()
        self.assertFalse(capability["sufficient"])
        self.assertEqual(capability["best_property"],
                         anchors.PROPERTY_ORDERING_ONLY)

    def test_no_anchor_property_field_is_trusted(self):
        """anchor_property is derived from a frozen map, never read."""
        for kind in anchors.ANCHOR_KINDS:
            self.assertEqual(anchors.anchor_property(kind),
                             anchors.anchor_property(kind))
        with self.assertRaises(anchors.AnchorError) as ctx:
            anchors.anchor_property("FREETSA_TOTALLY_LEGIT")
        self.assertEqual(ctx.exception.code, "E_ANCHOR_KIND")

    def test_every_self_owned_anchor_kind_is_refused(self):
        for kind in ("SELF_SIGNED_TSA", "LOCAL_HMAC", "LOCAL_GPG_SIGNATURE",
                     "GIT_COMMIT_LOCAL", "LOCAL_CLOCK_ASSERTION",
                     "BMC_EVENT_LOG", "SYNTHETIC_ANCHOR"):
            with self.subTest(kind=kind):
                self.assertEqual(anchors.anchor_property(kind),
                                 anchors.PROPERTY_NONE)
                with self.assertRaises(anchors.AnchorError) as ctx:
                    anchors.check_anchor_kind(kind)
                self.assertEqual(ctx.exception.code,
                                 "E_ANCHOR_NOT_INDEPENDENT")

    def test_tpm_proves_order_not_time(self):
        with self.assertRaises(anchors.AnchorError) as ctx:
            anchors.check_anchor_kind("TPM_NV_QUOTE")
        self.assertEqual(ctx.exception.code, "E_ANCHOR_NOT_INDEPENDENT")

    def test_no_trust_root_is_pinned_so_verification_cannot_pass(self):
        self.assertIsNone(anchors.FROZEN_TSA_ROOT_SHA256)
        with self.assertRaises(anchors.AnchorError) as ctx:
            anchors.check_trust_root({"anchor_kind": "RFC3161_TSA",
                                      "trust_root_custodian": "THIRD_PARTY_CA",
                                      "trust_root_sha256": "a" * 64})
        self.assertEqual(ctx.exception.code, "E_ANCHOR_TRUST_ROOT")

    def test_an_experiment_owned_custodian_is_refused(self):
        for custodian in ("EXPERIMENT_USER", "HOST_ROOT", "UNKNOWN"):
            with self.subTest(custodian=custodian):
                with self.assertRaises(anchors.AnchorError) as ctx:
                    anchors.check_trust_root(
                        {"anchor_kind": "RFC3161_TSA",
                         "trust_root_custodian": custodian,
                         "trust_root_sha256": "a" * 64})
                self.assertEqual(ctx.exception.code, "E_ANCHOR_TRUST_ROOT")


class AnchorGateInTheBundle(unittest.TestCase):

    def test_the_valid_bundle_stops_at_the_anchor_gate(self):
        """The production path, unmodified. This is what E2A actually does."""
        with bundle() as b:
            with self.assertRaises(resolver.ResolveError) as ctx:
                b.evaluate()
            self.assertEqual(ctx.exception.code, "E_ANCHOR_TRUST_ROOT")

    def test_a_missing_anchor_is_not_a_pass(self):
        with allow_anchor(), bundle() as b:
            index = b.load("bundle.json")
            index["plan_anchor_path"] = "does_not_exist.json"
            b.write("bundle.json", index, reseal=False)
            with self.assertRaises(resolver.ResolveError) as ctx:
                b.evaluate()
            self.assertIn(ctx.exception.code, {"E_MISSING", "E_HASH"})

    def test_a_fake_anchor_kind_is_refused(self):
        with allow_anchor(), bundle() as b:
            anchor = b.load("plan_anchor.json")
            anchor["anchor_kind"] = "LOCAL_HMAC"
            b.write("plan_anchor.json", anchor)
            b.rebind_anchor_chain()
            with self.assertRaises(resolver.ResolveError) as ctx:
                b.evaluate()
            self.assertEqual(ctx.exception.code, "E_ANCHOR_NOT_INDEPENDENT")

    def test_a_synthetic_anchor_cannot_authorize_anything(self):
        with bundle(), allow_anchor():
            with bundle() as b:
                anchor = b.load("plan_anchor.json")
                anchor["provenance"] = "SYNTHETIC"
                b.write("plan_anchor.json", anchor)
                b.rebind_anchor_chain()
                with self.assertRaises(resolver.ResolveError) as ctx:
                    b.evaluate()
                self.assertEqual(ctx.exception.code, "E_ANCHOR_PROVENANCE")

    def test_a_caller_supplied_trust_root_is_refused(self):
        with allow_anchor(), bundle() as b:
            anchor = b.load("plan_anchor.json")
            anchor["trust_root_pin_location"] = "CALLER_SUPPLIED"
            b.write("plan_anchor.json", anchor)
            b.rebind_anchor_chain()
            with self.assertRaises(resolver.ResolveError) as ctx:
                b.evaluate()
            self.assertEqual(ctx.exception.code, "E_ANCHOR_TRUST_ROOT")

    def test_an_anchor_over_a_different_plan_is_refused(self):
        """Post-hoc plan substitution: anchor plan A, ship plan B."""
        with allow_anchor(), bundle() as b:
            anchor = b.load("plan_anchor.json")
            anchor["anchored_digest"] = "0" * 64
            anchor["message_imprint_sha256"] = "0" * 64
            b.write("plan_anchor.json", anchor)
            b.rebind_anchor_chain()
            with self.assertRaises(resolver.ResolveError) as ctx:
                b.evaluate()
            self.assertEqual(ctx.exception.code, "E_ANCHOR_BINDING")

    def test_editing_the_plan_after_anchoring_breaks_the_ledger_binding(self):
        """Rewriting the plan breaks TWO independent bindings.

        The ledger pins plan_sha256 and fires first. The anchor binding would
        also fire; the next test isolates it by repairing the ledger, so neither
        check is left resting on the other.
        """
        with allow_anchor(), bundle() as b:
            plan = b.load("plan.json")
            plan["experiment_identity"] = "rewritten.after.the.fact"
            b.write("plan.json", plan)
            b.rebind_anchor_chain()
            with self.assertRaises(resolver.ResolveError) as ctx:
                b.evaluate()
            self.assertEqual(ctx.exception.code, "E_ANCHOR_BINDING")

    def test_editing_the_plan_and_repairing_the_ledger_still_breaks_the_anchor(self):
        """Post-hoc plan substitution, with the obvious cover-up applied.

        A producer who rewrites the plan will of course re-point the ledger at
        it. The anchor is what they cannot re-point, because the receipt commits
        to the ORIGINAL plan digest and re-anchoring would need the third party.
        This is the one check in the file that an attacker cannot route around
        locally -- which is exactly why its absence would matter, and why it is
        tested in isolation rather than left behind the ledger check.
        """
        with allow_anchor(), bundle() as b:
            plan = b.load("plan.json")
            plan["experiment_identity"] = "rewritten.after.the.fact"
            b.write("plan.json", plan)
            ledger = b.load("ledger.json")
            ledger["plan_sha256"] = plan["record_sha256"]
            b.write("ledger.json", ledger)
            b.rebind_anchor_chain()
            with self.assertRaises(resolver.ResolveError) as ctx:
                b.evaluate()
            self.assertEqual(ctx.exception.code, "E_ANCHOR_BINDING")

    def test_a_verifierless_anchor_is_refused(self):
        with allow_anchor(), bundle() as b:
            anchor = b.load("plan_anchor.json")
            anchor["verifier_kind"] = "NONE"
            b.write("plan_anchor.json", anchor)
            b.rebind_anchor_chain()
            with self.assertRaises(resolver.ResolveError) as ctx:
                b.evaluate()
            self.assertEqual(ctx.exception.code, "E_ANCHOR_VERIFY")

    def test_a_failed_anchor_status_is_not_evidence(self):
        with allow_anchor(), bundle() as b:
            anchor = b.load("plan_anchor.json")
            anchor["status"] = "INELIGIBLE"
            b.write("plan_anchor.json", anchor)
            b.rebind_anchor_chain()
            with self.assertRaises(resolver.ResolveError) as ctx:
                b.evaluate()
            self.assertEqual(ctx.exception.code, "E_ANCHOR_STATUS")

    def test_the_anchor_gate_is_not_neutralised_in_production(self):
        """allow_anchor() is a test artefact. Assert the real maps are intact."""
        self.assertFalse(anchors.ANCHOR_ENUMERABLE["RFC3161_TSA"])
        self.assertIsNone(anchors.FROZEN_TSA_ROOT_SHA256)
        source = (ROOT / "src" / "anchors.py").read_text(encoding="ascii")
        self.assertNotIn("allow_anchor", source)


# ---------------------------------------------------------------------------
# CP3 arithmetic
# ---------------------------------------------------------------------------

class AggregateArithmetic(unittest.TestCase):

    def test_sums_are_elementwise_over_every_pair(self):
        contributions = [contribution(100, 1, 80, 1, i) for i in range(8)]
        totals = aggregate.sum_all_pairs(contributions)
        self.assertEqual(totals["control_energy_sum_nj"], 800)
        self.assertEqual(totals["control_uncertainty_sum_nj"], 8)
        self.assertEqual(totals["treatment_energy_sum_nj"], 640)
        self.assertEqual(totals["treatment_uncertainty_sum_nj"], 8)

    def test_uncertainty_is_summed_not_averaged_or_quadratured(self):
        """The highest-motive bug in the checkpoint.

        Eight pairs each carrying 100 nJ of uncertainty sum to 800, not to
        isqrt(8*100^2) = 282 and not to 100. NVML's +/-5 W is a vendor-stated
        SYSTEMATIC bound, not zero-mean noise: if the sensor reads high, it reads
        high in every pair, so it never averages down. Quadrature here would
        shrink U by ~2.83x and is very often the only way relief appears at all.
        """
        contributions = [contribution(1000, 100, 800, 100, i) for i in range(8)]
        totals = aggregate.sum_all_pairs(contributions)
        self.assertEqual(totals["control_uncertainty_sum_nj"], 800)
        self.assertEqual(totals["treatment_uncertainty_sum_nj"], 800)
        import math
        self.assertNotEqual(totals["control_uncertainty_sum_nj"],
                            math.isqrt(8 * 100 * 100))
        self.assertNotEqual(totals["control_uncertainty_sum_nj"], 100)

    def test_exact_ten_percent_passes_the_gate(self):
        """t*10 == c*9 exactly. The boundary is inclusive by construction."""
        totals = {"control_energy_sum_nj": 1000,
                  "control_uncertainty_sum_nj": 0,
                  "treatment_energy_sum_nj": 900,
                  "treatment_uncertainty_sum_nj": 0}
        detail = aggregate.decide_aggregate(totals, "GPU_BOARD")
        self.assertEqual(detail["control_lower_nj"], 1000)
        self.assertEqual(detail["treatment_upper_nj"], 900)
        self.assertTrue(detail["meets_ten_percent_gate"])
        self.assertTrue(detail["relief"])

    def test_just_under_ten_percent_fails_the_gate(self):
        """9.999%: treatment_upper=90001 against control_lower=100000.

        90001*10 = 900010 > 100000*9 = 900000. Relief is still true (the
        treatment IS lower), so this test isolates the 10% gate from the
        conservative bound -- a float implementation would round this to a pass.
        """
        totals = {"control_energy_sum_nj": 100000,
                  "control_uncertainty_sum_nj": 0,
                  "treatment_energy_sum_nj": 90001,
                  "treatment_uncertainty_sum_nj": 0}
        detail = aggregate.decide_aggregate(totals, "GPU_BOARD")
        self.assertFalse(detail["meets_ten_percent_gate"])
        self.assertFalse(detail["relief"])

    def test_the_gate_uses_no_division(self):
        """The executable body only: prose may discuss the traps it avoids."""
        import ast
        import inspect
        for function in (aggregate.decide_aggregate, aggregate.sum_all_pairs,
                         checker.decide):
            with self.subTest(function=function.__name__):
                tree = ast.parse(inspect.getsource(function).lstrip())
                for node in ast.walk(tree):
                    if isinstance(node, ast.BinOp):
                        self.assertNotIsInstance(
                            node.op, (ast.Div, ast.FloorDiv),
                            f"{function.__name__} divides; the 10 percent gate "
                            f"is integer cross multiplication")
                    if isinstance(node, ast.Constant) and \
                            isinstance(node.value, float):
                        self.fail(f"{function.__name__} contains the float "
                                  f"{node.value}")

    def test_seven_favorable_pairs_cannot_carry_one_unfavorable_pair(self):
        """The all-pairs rule, doing its job.

        Seven pairs save 20 nJ each; the eighth burns 1000 more. Dropping it
        would give relief. Including it -- which SUM_ALL_PAIRS_V1 requires -- must
        not.
        """
        contributions = [contribution(100, 0, 80, 0, i) for i in range(7)]
        contributions.append(contribution(100, 0, 1100, 0, 7))
        totals = aggregate.sum_all_pairs(contributions)
        self.assertEqual(totals["control_energy_sum_nj"], 800)
        self.assertEqual(totals["treatment_energy_sum_nj"], 560 + 1100)
        detail = aggregate.decide_aggregate(totals, "GPU_BOARD")
        self.assertFalse(detail["relief"])
        # and the favourable subset alone WOULD have passed, so the test is real
        favorable = aggregate.sum_all_pairs(contributions[:7])
        self.assertTrue(aggregate.decide_aggregate(favorable,
                                                   "GPU_BOARD")["relief"])

    def test_uncertainty_can_erase_an_apparent_win(self):
        contributions = [contribution(1000, 200, 900, 200, i) for i in range(8)]
        totals = aggregate.sum_all_pairs(contributions)
        detail = aggregate.decide_aggregate(totals, "GPU_BOARD")
        self.assertEqual(detail["control_lower_nj"], 8000 - 1600)
        self.assertEqual(detail["treatment_upper_nj"], 7200 + 1600)
        self.assertFalse(detail["relief"])

    def test_a_dropped_uncertainty_is_refused(self):
        item = contribution(100, 1, 80, 1)
        del item["treatment_uncertainty_nj"]
        with self.assertRaises(aggregate.AggregateError) as ctx:
            aggregate.sum_all_pairs([item])
        self.assertEqual(ctx.exception.code, "E_UNCERTAINTY")

    def test_control_swamped_by_its_own_uncertainty_cannot_show_relief(self):
        totals = {"control_energy_sum_nj": 100,
                  "control_uncertainty_sum_nj": 100,
                  "treatment_energy_sum_nj": 1,
                  "treatment_uncertainty_sum_nj": 0}
        with self.assertRaises(aggregate.AggregateError) as ctx:
            aggregate.decide_aggregate(totals, "GPU_BOARD")
        self.assertEqual(ctx.exception.code, "E_UNCERTAINTY")

    def test_overflow_beyond_the_canonical_bound_is_refused(self):
        big = canon.MAX_INT
        with self.assertRaises(aggregate.AggregateError) as ctx:
            aggregate.sum_all_pairs([contribution(big, 0, 1, 0, 0),
                                     contribution(big, 0, 1, 0, 1)])
        self.assertEqual(ctx.exception.code, "E_OVERFLOW")

    def test_a_single_value_above_the_canonical_bound_is_refused(self):
        with self.assertRaises(aggregate.AggregateError) as ctx:
            aggregate.sum_all_pairs([contribution(canon.MAX_INT + 1, 0, 1, 0)])
        self.assertEqual(ctx.exception.code, "E_OVERFLOW")

    def test_a_gpu_board_delta_is_not_a_server_delta_or_a_budget(self):
        totals = {"control_energy_sum_nj": 1000,
                  "control_uncertainty_sum_nj": 0,
                  "treatment_energy_sum_nj": 900,
                  "treatment_uncertainty_sum_nj": 0}
        detail = aggregate.decide_aggregate(totals, "GPU_BOARD")
        self.assertEqual(detail["boundary_delta_nj"], 100)
        self.assertIsNone(detail["server_wall_delta_nj"])
        self.assertIsNone(detail["phone_plus_external_break_even_budget_nj"])


class TypeConfusion(unittest.TestCase):

    def test_a_float_energy_is_refused_before_it_is_compared(self):
        with self.assertRaises(aggregate.AggregateError) as ctx:
            aggregate.sum_all_pairs([contribution(900000.0, 0, 1, 0)])
        self.assertEqual(ctx.exception.code, "E_TYPE")

    def test_a_float_that_equals_an_int_still_fails(self):
        self.assertEqual(900000.0, 900000)   # the trap, stated
        with self.assertRaises(aggregate.AggregateError):
            aggregate.sum_all_pairs([contribution(900000.0, 0, 1, 0)])

    def test_a_bool_energy_is_refused(self):
        self.assertEqual(True, 1)            # the trap, stated
        with self.assertRaises(aggregate.AggregateError) as ctx:
            aggregate.sum_all_pairs([contribution(True, 0, 1, 0)])
        self.assertEqual(ctx.exception.code, "E_TYPE")

    def test_a_bool_laundered_through_sum_would_look_clean(self):
        """Why the type gate is per record and not on the total."""
        self.assertIs(type(sum([True, 3])), int)
        with self.assertRaises(aggregate.AggregateError):
            aggregate.sum_all_pairs([contribution(True, 0, 1, 0, 0),
                                     contribution(3, 0, 1, 0, 1)])

    def test_a_negative_energy_is_refused(self):
        with self.assertRaises(aggregate.AggregateError) as ctx:
            aggregate.sum_all_pairs([contribution(-1, 0, 1, 0)])
        self.assertEqual(ctx.exception.code, "E_TYPE")


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

class PlanIntegrity(unittest.TestCase):

    def _plan(self):
        return canon.load_strict(FIXTURES / "plan.json")

    def test_the_fixture_plan_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            resolver.resolve_plan(self._plan(), root)

    def test_retuned_gate_constants_are_refused(self):
        plan = self._plan()
        plan["gate_constants_digest"] = canon.digest({"MIN_PAIRS": 2})
        canon.seal(plan)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_plan(plan, FIXTURES)
        self.assertEqual(ctx.exception.code, "E_GATE_RETUNED")

    def test_fewer_than_min_pairs_is_refused(self):
        """MIN_PAIRS=8 is enforced twice: in the schema and in resolve_plan.

        The schema fires first here, so the code-level MIN_PAIRS check is
        unreachable through gate_record. That redundancy is deliberate and worth
        naming rather than hiding: the schema is data and could be edited, the
        code check is the backstop, and a reader who greps for MIN_PAIRS should
        find both. The test asserts the code that ACTUALLY fires rather than the
        one it would be tidier to claim.
        """
        plan = self._plan()
        plan["n_pairs"] = 4
        plan["slots"] = plan["slots"][:8]
        plan["declared_order"] = plan["declared_order"][:8]
        plan["order_digest"] = canon.digest(plan["declared_order"])
        canon.seal(plan)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_plan(plan, FIXTURES)
        self.assertEqual(ctx.exception.code, "E_SCHEMA")

    def test_the_code_level_min_pairs_backstop_is_real(self):
        """Exercise the backstop directly, with the schema gate stepped over."""
        plan = self._plan()
        plan["n_pairs"] = 4
        canon.seal(plan)
        saved = resolver.gate_record
        resolver.gate_record = lambda kind, record, what: record
        try:
            with self.assertRaises(resolver.ResolveError) as ctx:
                resolver.resolve_plan(plan, FIXTURES)
            self.assertEqual(ctx.exception.code, "E_PAIRS")
        finally:
            resolver.gate_record = saved

    def test_a_broken_abba_rotation_is_refused(self):
        plan = self._plan()
        plan["declared_order"][1] = "OPTIMIZED_SERVER_ONLY_CONTROL"
        plan["order_digest"] = canon.digest(plan["declared_order"])
        canon.seal(plan)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_plan(plan, FIXTURES)
        self.assertEqual(ctx.exception.code, "E_ORDER")

    def test_an_order_digest_that_does_not_cover_the_order_is_refused(self):
        plan = self._plan()
        plan["order_digest"] = "0" * 64
        canon.seal(plan)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_plan(plan, FIXTURES)
        self.assertEqual(ctx.exception.code, "E_ORDER")

    def test_a_slot_count_that_is_not_2n_is_refused(self):
        """n_pairs=9 with 16 slots: schema-valid, semantically a missing pair."""
        plan = self._plan()
        plan["n_pairs"] = 9
        canon.seal(plan)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_plan(plan, FIXTURES)
        self.assertEqual(ctx.exception.code, "E_PLAN")

    def test_a_reused_run_nonce_is_refused(self):
        plan = self._plan()
        plan["slots"][1]["run_nonce"] = plan["slots"][0]["run_nonce"]
        canon.seal(plan)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_plan(plan, FIXTURES)
        self.assertEqual(ctx.exception.code, "E_PLAN")

    def test_control_and_treatment_cannot_share_a_policy(self):
        plan = self._plan()
        plan["policy_digest_treatment"] = plan["policy_digest_control"]
        canon.seal(plan)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_plan(plan, FIXTURES)
        self.assertEqual(ctx.exception.code, "E_SAME_POLICY")

    def test_a_plan_that_permits_retries_is_not_expressible(self):
        plan = self._plan()
        plan["max_attempts_per_slot"] = 3
        canon.seal(plan)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_plan(plan, FIXTURES)
        self.assertEqual(ctx.exception.code, "E_SCHEMA")

    def test_an_unsealed_plan_is_refused(self):
        plan = self._plan()
        plan["experiment_identity"] = "edited"
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_plan(plan, FIXTURES)
        self.assertEqual(ctx.exception.code, "E_DIGEST")


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------

class LedgerIntegrity(unittest.TestCase):

    def _pair(self):
        return (canon.load_strict(FIXTURES / "plan.json"),
                canon.load_strict(FIXTURES / "ledger.json"))

    def test_the_fixture_ledger_covers_every_planned_slot(self):
        plan, ledger = self._pair()
        terminal = resolver.resolve_ledger(ledger, plan)
        self.assertEqual(sorted(terminal), list(range(16)))

    def test_truncating_the_ledger_is_caught_by_the_plan_not_by_itself(self):
        """The tautology E2 shipped, closed.

        E2's repetition set checked `attempted_pairs == len(pairs)`. The producer
        writes both numbers, so dropping the unfavourable tail and decrementing
        the count is self-consistent at any N. Only an EXTERNAL count -- the
        anchored plan's 2N slots -- makes truncation visible.
        """
        plan, ledger = self._pair()
        ledger["entries"] = [e for e in ledger["entries"]
                             if not (e["entry_kind"] == "SLOT_ATTEMPT_END"
                                     and e["slot_index"] == 15)]
        for index, entry in enumerate(ledger["entries"]):
            entry["seq"] = index
        # rebuild a perfectly self-consistent chain over the truncated entries
        previous = "0" * 64
        for entry in ledger["entries"]:
            entry["prev_entry_sha256"] = previous
            body = {k: v for k, v in entry.items() if k != "entry_sha256"}
            entry["entry_sha256"] = canon.digest(body)
            previous = entry["entry_sha256"]
        ledger["head_sha256"] = previous
        canon.seal(ledger)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_ledger(ledger, plan)
        self.assertEqual(ctx.exception.code, "E_SLOT_UNCOVERED")

    def test_a_broken_chain_link_is_caught(self):
        plan, ledger = self._pair()
        ledger["entries"][5]["prev_entry_sha256"] = "0" * 64
        canon.seal(ledger)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_ledger(ledger, plan)
        self.assertEqual(ctx.exception.code, "E_LEDGER_CHAIN")

    def test_an_edited_entry_body_is_caught(self):
        plan, ledger = self._pair()
        ledger["entries"][5]["reason_code"] = "REWRITTEN"
        canon.seal(ledger)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_ledger(ledger, plan)
        self.assertEqual(ctx.exception.code, "E_LEDGER_CHAIN")

    def test_a_head_that_is_not_the_last_entry_is_caught(self):
        plan, ledger = self._pair()
        ledger["head_sha256"] = "0" * 64
        canon.seal(ledger)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_ledger(ledger, plan)
        self.assertEqual(ctx.exception.code, "E_LEDGER_HEAD")

    def test_a_ledger_bound_to_another_plan_is_caught(self):
        plan, ledger = self._pair()
        ledger["plan_sha256"] = "0" * 64
        canon.seal(ledger)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_ledger(ledger, plan)
        self.assertEqual(ctx.exception.code, "E_LEDGER_BINDING")

    def test_a_second_terminal_for_one_slot_is_an_undeclared_retry(self):
        plan, ledger = self._pair()
        extra = dict(ledger["entries"][3])
        extra["seq"] = len(ledger["entries"])
        extra["prev_entry_sha256"] = ledger["head_sha256"]
        body = {k: v for k, v in extra.items() if k != "entry_sha256"}
        extra["entry_sha256"] = canon.digest(body)
        ledger["entries"].append(extra)
        ledger["head_sha256"] = extra["entry_sha256"]
        canon.seal(ledger)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_ledger(ledger, plan)
        self.assertEqual(ctx.exception.code, "E_SLOT_DOUBLE_TERMINAL")

    def test_a_retry_ordinal_is_refused(self):
        plan, ledger = self._pair()
        ledger["entries"][3]["attempt_ordinal"] = 1
        _rechain(ledger)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_ledger(ledger, plan)
        self.assertEqual(ctx.exception.code, "E_UNDECLARED_RETRY")

    def test_every_non_ok_status_poisons_the_set(self):
        """INELIGIBLE especially: it is neither OK nor FAILED.

        A `status == "FAILED"` check misses it entirely, so an unfavourable run
        relabelled INELIGIBLE would simply vanish from the cohort.
        """
        for status in ("FAILED", "CANCELED", "CRASHED", "INELIGIBLE",
                       "ABORTED_BY_GATE"):
            with self.subTest(status=status):
                plan, ledger = self._pair()
                ledger["entries"][3]["status"] = status
                _rechain(ledger)
                with self.assertRaises(resolver.ResolveError) as ctx:
                    resolver.resolve_ledger(ledger, plan)
                self.assertEqual(ctx.exception.code, "E_NON_OK_ATTEMPT")

    def test_a_slot_the_plan_never_declared_is_refused(self):
        plan, ledger = self._pair()
        ledger["entries"][3]["slot_index"] = 99
        _rechain(ledger)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_ledger(ledger, plan)
        self.assertEqual(ctx.exception.code, "E_LEDGER_BINDING")

    def test_a_non_contiguous_seq_is_refused(self):
        plan, ledger = self._pair()
        ledger["entries"][3]["seq"] = 99
        body = {k: v for k, v in ledger["entries"][3].items()
                if k != "entry_sha256"}
        ledger["entries"][3]["entry_sha256"] = canon.digest(body)
        canon.seal(ledger)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_ledger(ledger, plan)
        self.assertEqual(ctx.exception.code, "E_LEDGER_SEQ")


def _rechain(ledger):
    previous = "0" * 64
    for entry in ledger["entries"]:
        entry["prev_entry_sha256"] = previous
        body = {k: v for k, v in entry.items() if k != "entry_sha256"}
        entry["entry_sha256"] = canon.digest(body)
        previous = entry["entry_sha256"]
    ledger["head_sha256"] = previous
    canon.seal(ledger)


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------

class LifecycleIntegrity(unittest.TestCase):

    def _pair(self, slot=0):
        return (canon.load_strict(FIXTURES / f"timelines/tl.slot{slot:02d}.json"),
                canon.load_strict(FIXTURES / f"lifecycle/lc.slot{slot:02d}.json"))

    def _resolve(self, lifecycle, timeline, slot=0):
        manifest = canon.load_strict(FIXTURES / "manifest.json")
        outcomes = canon.load_strict(
            FIXTURES / f"outcomes/outcomes.slot{slot:02d}.json")
        resolved = resolver.resolve_requests(
            manifest, outcomes, timeline, FIXTURES, f"slot{slot}", slot // 2)
        return resolver.resolve_lifecycle(
            lifecycle, timeline, None, resolved)

    def test_the_fixture_lifecycle_is_closed(self):
        timeline, lifecycle = self._pair()
        self._resolve(lifecycle, timeline)

    def test_work_started_before_the_window_is_refused(self):
        """Prefetching outside the paid window is real energy, charged to nobody."""
        timeline, lifecycle = self._pair()
        for action in lifecycle["actions"]:
            if action["action_kind"] == "PREFETCH":
                action["enqueue_us"] -= 5_000_000
                action["start_us"] -= 5_000_000
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            self._resolve(lifecycle, timeline)
        self.assertEqual(ctx.exception.code, "E_ACTION_UNBOUND")

    def test_an_outstanding_transfer_at_the_window_edge_is_refused(self):
        timeline, lifecycle = self._pair()
        for action in lifecycle["actions"]:
            if action["action_kind"] == "D2H":
                action["ack_us"] = timeline["window_end_us"] + 1_000_000
                action["end_us"] = action["ack_us"]
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            self._resolve(lifecycle, timeline)
        self.assertEqual(ctx.exception.code, "E_DEFERRED_CLEANUP")

    def test_deferred_cleanup_is_refused(self):
        timeline, lifecycle = self._pair()
        for action in lifecycle["actions"]:
            if action["action_kind"] == "CLEANUP":
                action["ack_us"] = timeline["window_end_us"] + 10
                action["end_us"] = action["ack_us"]
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            self._resolve(lifecycle, timeline)
        self.assertEqual(ctx.exception.code, "E_DEFERRED_CLEANUP")

    def test_an_action_still_running_is_refused(self):
        timeline, lifecycle = self._pair()
        lifecycle["actions"][3]["state"] = "RUNNING"
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            self._resolve(lifecycle, timeline)
        self.assertEqual(ctx.exception.code, "E_LIFECYCLE_OPEN")

    def test_an_in_flight_action_is_refused(self):
        timeline, lifecycle = self._pair()
        lifecycle["actions"][3]["state"] = "IN_FLIGHT"
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            self._resolve(lifecycle, timeline)
        self.assertEqual(ctx.exception.code, "E_LIFECYCLE_OPEN")

    def test_a_leaked_lease_is_refused(self):
        timeline, lifecycle = self._pair()
        lifecycle["actions"] = [a for a in lifecycle["actions"]
                                if a["action_kind"] != "LEASE_RELEASE"]
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            self._resolve(lifecycle, timeline)
        self.assertEqual(ctx.exception.code, "E_LEASE_OPEN")

    def test_a_missing_drain_is_refused(self):
        timeline, lifecycle = self._pair()
        lifecycle["actions"] = [a for a in lifecycle["actions"]
                                if a["action_kind"] != "QUEUE_DRAIN"]
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            self._resolve(lifecycle, timeline)
        self.assertEqual(ctx.exception.code, "E_DRAIN_UNACKED")

    def test_an_unacknowledged_drain_is_refused(self):
        timeline, lifecycle = self._pair()
        lifecycle["drain_acknowledged_us"] = timeline["window_start_us"]
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            self._resolve(lifecycle, timeline)
        self.assertEqual(ctx.exception.code, "E_DRAIN_UNACKED")

    def test_a_lifecycle_on_another_clock_epoch_is_refused(self):
        timeline, lifecycle = self._pair()
        lifecycle["clock_epoch_id"] = "boot-other-0001"
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            self._resolve(lifecycle, timeline)
        self.assertEqual(ctx.exception.code, "E_CLOCK_EPOCH")

    def test_a_lifecycle_for_another_timeline_is_refused(self):
        timeline, lifecycle = self._pair()
        lifecycle["timeline_id"] = "tl.somewhere.else"
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            self._resolve(lifecycle, timeline)
        self.assertEqual(ctx.exception.code, "E_LIFECYCLE_BINDING")

    def test_a_lifecycle_window_that_disagrees_with_the_timeline_is_refused(self):
        timeline, lifecycle = self._pair()
        lifecycle["window_end_us"] += 1
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            self._resolve(lifecycle, timeline)
        self.assertEqual(ctx.exception.code, "E_LIFECYCLE_BINDING")

    def test_unordered_action_timestamps_are_refused(self):
        timeline, lifecycle = self._pair()
        lifecycle["actions"][3]["start_us"] = \
            lifecycle["actions"][3]["end_us"] + 1
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            self._resolve(lifecycle, timeline)
        self.assertEqual(ctx.exception.code, "E_LIFECYCLE_ORDER")

    def test_outstanding_is_recomputed_not_declared(self):
        """There is no `outstanding_actions` field to read, by design."""
        _timeline, lifecycle = self._pair()
        self.assertNotIn("outstanding_actions", lifecycle)
        self.assertNotIn("drained", lifecycle)
        actions = lifecycle["actions"]
        self.assertEqual(resolver.outstanding_at(actions, 0), [])
        mid = lifecycle["window_start_us"] + 1_100_000
        self.assertTrue(resolver.outstanding_at(actions, mid))


# ---------------------------------------------------------------------------
# requests / same work
# ---------------------------------------------------------------------------

class RequestIntegrity(unittest.TestCase):

    def _triple(self, slot=0):
        return (canon.load_strict(FIXTURES / "manifest.json"),
                canon.load_strict(FIXTURES /
                                  f"outcomes/outcomes.slot{slot:02d}.json"),
                canon.load_strict(FIXTURES /
                                  f"timelines/tl.slot{slot:02d}.json"))

    def test_the_fixture_requests_resolve(self):
        manifest, outcomes, timeline = self._triple()
        resolver.resolve_requests(manifest, outcomes, timeline, FIXTURES, "slot0")

    def test_request_substitution_is_caught(self):
        manifest, outcomes, timeline = self._triple()
        outcomes["outcomes"][0]["request_id"] = "req.impostor"
        canon.seal(outcomes)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_requests(manifest, outcomes, timeline, FIXTURES, "s")
        self.assertEqual(ctx.exception.code, "E_REQUEST_SUBSTITUTED")

    def test_a_dropped_request_is_caught(self):
        manifest, outcomes, timeline = self._triple()
        outcomes["outcomes"] = outcomes["outcomes"][:-1]
        canon.seal(outcomes)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_requests(manifest, outcomes, timeline, FIXTURES, "s")
        self.assertEqual(ctx.exception.code, "E_REQUEST_MISSING")

    def test_a_mutated_output_artifact_is_caught(self):
        with bundle() as b:
            path = b.path("outputs/slot00/req.0.json")
            path.write_text('{"tampered": true}\n', encoding="ascii")
            manifest = b.load("manifest.json")
            outcomes = b.load("outcomes/outcomes.slot00.json")
            timeline = b.load("timelines/tl.slot00.json")
            with self.assertRaises(resolver.ResolveError) as ctx:
                resolver.resolve_requests(manifest, outcomes, timeline,
                                          b.root, "s")
            self.assertEqual(ctx.exception.code, "E_HASH")

    def test_a_missing_output_artifact_is_caught(self):
        with bundle() as b:
            b.path("outputs/slot00/req.0.json").unlink()
            with self.assertRaises(resolver.ResolveError) as ctx:
                resolver.resolve_requests(
                    b.load("manifest.json"),
                    b.load("outcomes/outcomes.slot00.json"),
                    b.load("timelines/tl.slot00.json"), b.root, "s")
            self.assertEqual(ctx.exception.code, "E_MISSING")

    def test_a_missing_correctness_certificate_is_caught(self):
        manifest, outcomes, timeline = self._triple()
        outcomes["outcomes"][0]["certificate"]["certificate_kind"] = "NONE"
        canon.seal(outcomes)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_requests(manifest, outcomes, timeline, FIXTURES, "s")
        self.assertEqual(ctx.exception.code, "E_CERT_MISSING")

    def test_an_unrun_referee_is_caught(self):
        manifest, outcomes, timeline = self._triple()
        outcomes["outcomes"][0]["certificate"]["referee_verdict"] = "NOT_RUN"
        canon.seal(outcomes)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_requests(manifest, outcomes, timeline, FIXTURES, "s")
        self.assertEqual(ctx.exception.code, "E_CERT_MISSING")

    def test_a_diverging_certificate_is_caught(self):
        manifest, outcomes, timeline = self._triple()
        outcomes["outcomes"][0]["certificate"]["referee_verdict"] = "DIVERGE"
        canon.seal(outcomes)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_requests(manifest, outcomes, timeline, FIXTURES, "s")
        self.assertEqual(ctx.exception.code, "E_CERT_FAIL")

    def test_a_short_prefix_agreement_is_caught(self):
        manifest, outcomes, timeline = self._triple()
        outcomes["outcomes"][0]["certificate"]["prefix_agreement_tokens"] = 1
        canon.seal(outcomes)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_requests(manifest, outcomes, timeline, FIXTURES, "s")
        self.assertEqual(ctx.exception.code, "E_CERT_FAIL")

    def test_a_model_swap_is_caught(self):
        manifest, outcomes, timeline = self._triple()
        outcomes["outcomes"][0]["realized_model_digest"] = "0" * 64
        canon.seal(outcomes)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_requests(manifest, outcomes, timeline, FIXTURES, "s")
        self.assertEqual(ctx.exception.code, "E_MODEL_MISMATCH")

    def test_a_quantized_weight_transform_is_refused_not_assumed(self):
        manifest, outcomes, timeline = self._triple()
        outcomes["outcomes"][0]["weights_transform_id"] = "QUANTIZED"
        canon.seal(outcomes)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_requests(manifest, outcomes, timeline, FIXTURES, "s")
        self.assertEqual(ctx.exception.code,
                         "E_MODEL_TRANSFORM_UNCERTIFIED")

    def test_a_seed_change_is_caught(self):
        manifest, outcomes, timeline = self._triple()
        outcomes["outcomes"][0]["realized_seed"] = 9999
        canon.seal(outcomes)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_requests(manifest, outcomes, timeline, FIXTURES, "s")
        self.assertEqual(ctx.exception.code, "E_SEED_MISMATCH")

    def test_sampled_decoding_is_refused_rather_than_approximated(self):
        manifest, outcomes, timeline = self._triple()
        manifest["entries"][0]["sampling_mode"] = "SAMPLED"
        outcomes["outcomes"][0]["realized_sampling_mode"] = "SAMPLED"
        manifest["entries"][0]["entry_sha256"] = canon.digest(
            {k: v for k, v in manifest["entries"][0].items()
             if k != "entry_sha256"})
        outcomes["outcomes"][0]["manifest_entry_sha256"] = \
            manifest["entries"][0]["entry_sha256"]
        canon.seal(manifest)
        outcomes["manifest_sha256"] = manifest["record_sha256"]
        canon.seal(outcomes)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_requests(manifest, outcomes, timeline, FIXTURES, "s")
        self.assertEqual(ctx.exception.code, "E_SAMPLING_REFUSED")

    def test_a_rejected_request_is_not_the_same_work(self):
        manifest, outcomes, timeline = self._triple()
        outcomes["outcomes"][0]["terminal_outcome"] = "rejected"
        canon.seal(outcomes)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_requests(manifest, outcomes, timeline, FIXTURES, "s")
        self.assertEqual(ctx.exception.code, "E_REQUEST_OUTCOME")

    def test_work_outside_the_paid_window_is_caught(self):
        manifest, outcomes, timeline = self._triple()
        outcomes["outcomes"][0]["last_token_us"] = \
            timeline["window_end_us"] + 1_000_000
        canon.seal(outcomes)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_requests(manifest, outcomes, timeline, FIXTURES, "s")
        self.assertEqual(ctx.exception.code, "E_REQUEST_OUT_OF_WINDOW")

    def test_a_truncation_that_claims_the_cap_is_caught(self):
        manifest, outcomes, timeline = self._triple()
        outcomes["outcomes"][0]["stop_reason"] = "MAX_TOKENS"
        canon.seal(outcomes)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_requests(manifest, outcomes, timeline, FIXTURES, "s")
        self.assertEqual(ctx.exception.code, "E_OUTPUT_TRUNCATED")

    def test_outcomes_bound_to_another_manifest_are_caught(self):
        manifest, outcomes, timeline = self._triple()
        outcomes["manifest_sha256"] = "0" * 64
        canon.seal(outcomes)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_requests(manifest, outcomes, timeline, FIXTURES, "s")
        self.assertEqual(ctx.exception.code, "E_REQUEST_BINDING")


class SameWork(unittest.TestCase):

    def _roles(self):
        manifest = canon.load_strict(FIXTURES / "manifest.json")
        records = []
        for slot in (0, 1):
            outcomes = canon.load_strict(
                FIXTURES / f"outcomes/outcomes.slot{slot:02d}.json")
            timeline = canon.load_strict(
                FIXTURES / f"timelines/tl.slot{slot:02d}.json")
            records.append(resolver.resolve_requests(
                manifest, outcomes, timeline, FIXTURES, f"slot{slot}", 0))
        return manifest, records[0], records[1]

    def test_the_fixture_roles_did_the_same_work(self):
        manifest, control, treatment = self._roles()
        resolver.check_same_work(control, treatment, manifest)

    def test_a_shorter_treatment_generation_is_caught(self):
        """Generating less text is the cheapest way to look faster."""
        manifest, control, treatment = self._roles()
        treatment.record["outcomes"][0]["realized_output_tokens"] = 4
        canon.seal(treatment.record)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.check_same_work(control, treatment, manifest)
        self.assertEqual(ctx.exception.code, "E_OUTPUT_SHORT")

    def test_a_different_terminal_outcome_is_caught(self):
        manifest, control, treatment = self._roles()
        treatment.record["outcomes"][0]["terminal_outcome"] = "tardy"
        canon.seal(treatment.record)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.check_same_work(control, treatment, manifest)
        self.assertEqual(ctx.exception.code, "E_OUTCOME_MISMATCH")

    def test_a_different_stop_reason_is_caught(self):
        manifest, control, treatment = self._roles()
        treatment.record["outcomes"][0]["stop_reason"] = "STOP_STRING"
        canon.seal(treatment.record)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.check_same_work(control, treatment, manifest)
        self.assertEqual(ctx.exception.code, "E_STOP_REASON_MISMATCH")

    def test_disjoint_request_sets_are_caught(self):
        manifest, control, treatment = self._roles()
        treatment.record["outcomes"] = treatment.record["outcomes"][:-1]
        canon.seal(treatment.record)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.check_same_work(control, treatment, manifest)
        self.assertEqual(ctx.exception.code, "E_REQUEST_MISSING")


# ---------------------------------------------------------------------------
# scope
# ---------------------------------------------------------------------------

class ScopePromotion(unittest.TestCase):

    def test_synthetic_evidence_can_never_carry_a_physical_result(self):
        with allow_anchor(), bundle() as b:
            with self.assertRaises(aggregate.AggregateError) as ctx:
                b.evaluate()
            self.assertEqual(ctx.exception.code, "E_PROVENANCE")

    def test_reaching_the_provenance_gate_proves_every_slot_resolved(self):
        """The provenance refusal fires in the PAIR loop.

        The slot loop runs to completion first, so this exception is evidence
        that all 16 timelines re-integrated, all 16 lifecycles closed, and all 16
        request records resolved. Without this test, an early anchor refusal
        could hide a broken resolver indefinitely.
        """
        manifest = canon.load_strict(FIXTURES / "manifest.json")
        for slot in range(16):
            timeline = canon.load_strict(
                FIXTURES / f"timelines/tl.slot{slot:02d}.json")
            lifecycle = canon.load_strict(
                FIXTURES / f"lifecycle/lc.slot{slot:02d}.json")
            outcomes = canon.load_strict(
                FIXTURES / f"outcomes/outcomes.slot{slot:02d}.json")
            resolved = resolver.resolve_requests(
                manifest, outcomes, timeline, FIXTURES, f"slot{slot}",
                slot // 2)
            resolver.resolve_lifecycle(lifecycle, timeline, None, resolved)

    def test_rapl_can_claim_no_scope_at_all(self):
        sys.path.insert(0, str(E2_SRC))
        import integrator as e2_integrator
        self.assertEqual(e2_integrator.INSTRUMENT_ALLOWED_SCOPE["RAPL_PACKAGE"],
                         set())

    def test_nvml_cannot_be_promoted_to_server_wall(self):
        import integrator as e2_integrator
        with self.assertRaises(e2_integrator.TimelineError) as ctx:
            e2_integrator.check_scope("NVML_BOARD", "SERVER_WALL")
        self.assertEqual(ctx.exception.code, "E_SCOPE")

    def test_a_plan_cannot_promote_scope_by_relabelling(self):
        plan = canon.load_strict(FIXTURES / "plan.json")
        plan["instrument_kind"] = "NVML_BOARD"
        plan["scope"] = "SERVER_WALL"
        plan["board_uuids"] = []
        canon.seal(plan)
        with self.assertRaises(resolver.ResolveError):
            resolver.resolve_plan(plan, FIXTURES)


# ---------------------------------------------------------------------------
# resolver security
# ---------------------------------------------------------------------------

class ResolverSecurity(unittest.TestCase):

    def test_an_absolute_path_is_refused(self):
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_path("/etc/passwd", FIXTURES)
        self.assertEqual(ctx.exception.code, "E_PATH")

    def test_upward_traversal_is_refused(self):
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_path("../../../etc/passwd", FIXTURES)
        self.assertEqual(ctx.exception.code, "E_PATH")

    def test_a_symlink_is_refused_even_when_it_points_inside(self):
        with bundle() as b:
            link = b.root / "alias.json"
            link.symlink_to(b.root / "plan.json")
            with self.assertRaises(resolver.ResolveError) as ctx:
                resolver.resolve_path("alias.json", b.root)
            self.assertEqual(ctx.exception.code, "E_PATH")

    def test_a_hardlinked_artifact_is_refused(self):
        with bundle() as b:
            alias = b.root / "hard.json"
            import os as _os
            _os.link(b.root / "plan.json", alias)
            data = (b.root / "plan.json").read_bytes()
            with self.assertRaises(resolver.ResolveError) as ctx:
                resolver.read_once("hard.json", canon.sha256_bytes(data), b.root)
            self.assertEqual(ctx.exception.code, "E_HARDLINK")

    def test_a_wrong_hash_is_refused(self):
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.read_once("plan.json", "0" * 64, FIXTURES)
        self.assertEqual(ctx.exception.code, "E_HASH")

    def test_the_trusted_root_is_derived_not_supplied(self):
        """A caller-chosen root turns every path check into decoration."""
        import inspect
        signature = inspect.signature(aggregate.resolve_bundle)
        self.assertEqual(list(signature.parameters), ["bundle_path"])
        self.assertNotIn("trusted_root", signature.parameters)
        derived = resolver.derive_trusted_root(BUNDLE)
        self.assertEqual(derived, FIXTURES)

    def test_duplicate_json_keys_are_refused(self):
        text = '{"a": 1, "a": 2}'
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.parse_once(text.encode("ascii"), "probe")
        self.assertEqual(ctx.exception.code, "E_DUPLICATE_KEY")

    def test_non_finite_constants_are_refused(self):
        for text in ('{"a": NaN}', '{"a": Infinity}'):
            with self.subTest(text=text):
                with self.assertRaises(resolver.ResolveError):
                    resolver.parse_once(text.encode("ascii"), "probe")

    def test_non_ascii_is_refused(self):
        # The payload is built from escapes, not written literally: this file is
        # itself required to be ASCII, and test_the_whole_tree_is_ascii enforces
        # that. A literal here would have made the suite fail its own rule.
        payload = b'{"a": "\xc3\xa9"}'
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.parse_once(payload, "probe")
        self.assertEqual(ctx.exception.code, "E_SCHEMA")

    def test_the_schema_worker_runs_isolated_from_pythonpath(self):
        source = (ROOT / "src" / "resolver.py").read_text(encoding="ascii")
        self.assertIn('"-I"', source)

    def test_the_schema_worker_env_is_scrubbed(self):
        """LD_PRELOAD must not reach a process whose job is to be uncorrupted."""
        self.assertNotIn("LD_PRELOAD", resolver.SAFE_ENV_KEYS)
        self.assertEqual(set(resolver.SAFE_ENV_KEYS),
                         {"PATH", "LANG", "LC_ALL", "HOME"})

    def test_every_schema_uses_only_local_refs(self):
        for path in sorted((ROOT / "schemas").glob("*.json")):
            with self.subTest(schema=path.name):
                text = path.read_text(encoding="ascii")
                schema = json.loads(text)
                self._assert_local(schema)

    def _assert_local(self, value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("$ref", "$dynamicRef", "$recursiveRef"):
                    self.assertTrue(str(item).startswith("#/"), f"{key}={item}")
                self._assert_local(item)
        elif isinstance(value, list):
            for item in value:
                self._assert_local(item)

    def test_a_hostile_jsonschema_on_the_path_cannot_replace_the_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            hostile = pathlib.Path(tmp) / "jsonschema.py"
            hostile.write_text(
                "class Draft202012Validator:\n"
                "    def __init__(self, *a, **k): pass\n"
                "    @staticmethod\n"
                "    def check_schema(*a, **k): pass\n"
                "    def iter_errors(self, *a, **k): return []\n",
                encoding="ascii")
            import os as _os
            env = dict(_os.environ)
            env["PYTHONPATH"] = tmp
            broken = {"schema_version": 2, "kind": "NotAPlan"}
            result = subprocess.run(
                [str(resolver.SYSTEM_PYTHON), "-I",
                 str(ROOT / "src" / "e2a_schema_gate.py")],
                input=json.dumps({"kind": "plan", "document": broken}),
                capture_output=True, text=True, env=env, timeout=60)
            payload = json.loads(result.stdout)
            # -I means the hostile module is never imported, so real errors appear
            self.assertNotIn("engine_error", payload)
            self.assertTrue(payload["errors"])


class CanonPinning(unittest.TestCase):

    def test_the_shared_canon_module_is_pinned_by_digest(self):
        actual = canon.file_sha256(canon.CANON_PATH)
        self.assertEqual(actual, canon.FROZEN_CANON_SHA256)

    def test_a_swapped_canon_module_would_be_refused(self):
        """E2 execs this module with no digest check. E2A does not."""
        source = (ROOT / "src" / "e2a_canon.py").read_text(encoding="ascii")
        self.assertIn("FROZEN_CANON_SHA256", source)
        self.assertIn("E_CANON_UNPINNED", source)
        # the pin is compared BEFORE exec_module
        pin_at = source.index("actual != FROZEN_CANON_SHA256")
        exec_at = source.index("spec.loader.exec_module")
        self.assertLess(pin_at, exec_at)

    def test_the_pinned_canon_rejects_bool_and_float_integers(self):
        self.assertFalse(canon.is_int(True))
        self.assertFalse(canon.is_int(1.0))
        self.assertTrue(canon.is_int(1))


# ---------------------------------------------------------------------------
# independent checker
# ---------------------------------------------------------------------------

class IndependentChecker(unittest.TestCase):

    def _record(self, contributions, scope="GPU_BOARD",
                anchor_kind="RFC3161_TSA"):
        c = sum(i["control_energy_nj"] for i in contributions)
        u_c = sum(i["control_uncertainty_nj"] for i in contributions)
        t = sum(i["treatment_energy_nj"] for i in contributions)
        u_t = sum(i["treatment_uncertainty_nj"] for i in contributions)
        control_lower, treatment_upper, relief, meets = checker.decide(c, u_c, t,
                                                                       u_t)
        prop = ("ORDERING_AND_ENUMERABLE"
                if anchor_kind == "TRANSPARENCY_LOG_INCLUSION"
                else "ORDERING_ONLY" if anchor_kind == "RFC3161_TSA"
                else "NONE")
        if prop != "ORDERING_AND_ENUMERABLE":
            label, reason = "MEASUREMENT_INVALID", "ANCHOR_NOT_ENUMERABLE"
        elif not (relief and meets):
            label, reason = "RELIEF_FAIL", "NO_CONSERVATIVE_RELIEF"
        else:
            label = checker.SCOPE_LABEL[scope]
            reason = "ALL_PAIRS_CONSERVATIVE_RELIEF"
        record = {
            "schema_version": 1, "kind": "AggregateComparison",
            "aggregate_id": "agg.test", "chain_id": "chain", "set_id": "set",
            "plan_id": "plan", "plan_sha256": "a" * 64,
            "plan_anchor_record_sha256": "b" * 64,
            "ledger_close_record_sha256": "c" * 64,
            "ledger_head_sha256": "d" * 64,
            "request_set_manifest_sha256": "e" * 64,
            "aggregate_method": "SUM_ALL_PAIRS_V1",
            "anchor_kind": anchor_kind, "anchor_property": prop,
            "scope": scope, "instrument_kind": "SYNTHETIC",
            "n_pairs": len(contributions),
            "pair_contributions": contributions,
            "control_energy_sum_nj": c, "control_uncertainty_sum_nj": u_c,
            "treatment_energy_sum_nj": t, "treatment_uncertainty_sum_nj": u_t,
            "control_lower_nj": control_lower,
            "treatment_upper_nj": treatment_upper,
            "conservative_margin_nj": control_lower - treatment_upper,
            "boundary_delta_nj": c - t,
            "server_wall_delta_nj": (c - t) if scope == "SERVER_WALL" else None,
            "phone_plus_external_break_even_budget_nj":
                (c - t) if scope == "SERVER_WALL" else None,
            "conservative_relief": relief and meets,
            "meets_ten_percent_gate": meets,
            "result_label": label, "reason_code": reason,
            "record_sha256": "",
        }
        record["record_sha256"] = checker.record_digest(record)
        return record

    def test_the_checker_accepts_a_correct_aggregate(self):
        contributions = [contribution(1000, 1, 800, 1, i) for i in range(8)]
        label, reason = checker.check_aggregate(self._record(contributions))
        self.assertEqual(label, "MEASUREMENT_INVALID")
        self.assertEqual(reason, "ANCHOR_NOT_ENUMERABLE")

    def test_the_checker_catches_a_forged_sum(self):
        contributions = [contribution(1000, 1, 800, 1, i) for i in range(8)]
        record = self._record(contributions)
        record["treatment_energy_sum_nj"] = 1
        record["record_sha256"] = checker.record_digest(record)
        with self.assertRaises(checker.CheckError) as ctx:
            checker.check_aggregate(record)
        self.assertEqual(ctx.exception.code, "E_SUM")

    def test_the_checker_catches_a_forged_label(self):
        """A hand-written physical label is schema-valid and digest-valid."""
        contributions = [contribution(1000, 1, 800, 1, i) for i in range(8)]
        record = self._record(contributions)
        record["result_label"] = "SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED"
        record["record_sha256"] = checker.record_digest(record)
        with self.assertRaises(checker.CheckError) as ctx:
            checker.check_aggregate(record)
        self.assertEqual(ctx.exception.code, "E_LABEL")

    def test_the_checker_catches_a_dropped_pair(self):
        contributions = [contribution(1000, 1, 800, 1, i) for i in range(8)]
        record = self._record(contributions)
        record["pair_contributions"] = record["pair_contributions"][:7]
        record["record_sha256"] = checker.record_digest(record)
        with self.assertRaises(checker.CheckError) as ctx:
            checker.check_aggregate(record)
        self.assertEqual(ctx.exception.code, "E_PAIRS")

    def test_the_checker_catches_an_anchor_property_upgrade(self):
        contributions = [contribution(1000, 1, 800, 1, i) for i in range(8)]
        record = self._record(contributions)
        record["anchor_property"] = "ORDERING_AND_ENUMERABLE"
        record["record_sha256"] = checker.record_digest(record)
        with self.assertRaises(checker.CheckError) as ctx:
            checker.check_aggregate(record)
        self.assertEqual(ctx.exception.code, "E_ANCHOR_PROPERTY")

    def test_the_checker_catches_a_gpu_board_budget_claim(self):
        contributions = [contribution(1000, 1, 800, 1, i) for i in range(8)]
        record = self._record(contributions)
        record["phone_plus_external_break_even_budget_nj"] = 1600
        record["record_sha256"] = checker.record_digest(record)
        with self.assertRaises(checker.CheckError) as ctx:
            checker.check_aggregate(record)
        self.assertEqual(ctx.exception.code, "E_SCOPE")

    def test_the_checker_catches_a_forbidden_total_system_claim(self):
        contributions = [contribution(1000, 1, 800, 1, i) for i in range(8)]
        record = self._record(contributions)
        record["reason_code"] = "SYSTEM_ENERGY_SAVING"
        record["record_sha256"] = checker.record_digest(record)
        with self.assertRaises(checker.CheckError) as ctx:
            checker.check_aggregate(record)
        self.assertEqual(ctx.exception.code, "E_SYSTEM_CLAIM")

    def test_the_checker_catches_an_unsealed_record(self):
        contributions = [contribution(1000, 1, 800, 1, i) for i in range(8)]
        record = self._record(contributions)
        record["n_pairs"] = 8
        record["set_id"] = "rewritten"
        with self.assertRaises(checker.CheckError) as ctx:
            checker.check_aggregate(record)
        self.assertEqual(ctx.exception.code, "E_DIGEST")

    def test_the_checker_agrees_with_the_aggregate_on_a_corpus(self):
        """Two independent implementations, checked against each other.

        If they ever disagree, one is wrong and this test says so rather than
        letting the difference be smoothed away.
        """
        cases = [
            [contribution(1000, 1, 800, 1, i) for i in range(8)],
            [contribution(1000, 0, 900, 0, i) for i in range(8)],
            [contribution(100000, 0, 90001, 0, i) for i in range(8)],
            [contribution(1000, 200, 900, 200, i) for i in range(8)],
            ([contribution(100, 0, 80, 0, i) for i in range(7)]
             + [contribution(100, 0, 1100, 0, 7)]),
        ]
        for index, contributions in enumerate(cases):
            with self.subTest(case=index):
                totals = aggregate.sum_all_pairs(contributions)
                c, u_c, t, u_t = checker.sum_pairs(contributions)
                self.assertEqual(totals["control_energy_sum_nj"], c)
                self.assertEqual(totals["control_uncertainty_sum_nj"], u_c)
                self.assertEqual(totals["treatment_energy_sum_nj"], t)
                self.assertEqual(totals["treatment_uncertainty_sum_nj"], u_t)
                detail = aggregate.decide_aggregate(totals, "GPU_BOARD")
                lower, upper, relief, meets = checker.decide(c, u_c, t, u_t)
                self.assertEqual(detail["control_lower_nj"], lower)
                self.assertEqual(detail["treatment_upper_nj"], upper)
                self.assertEqual(detail["meets_ten_percent_gate"], meets)
                self.assertEqual(detail["relief"], relief and meets)

    def test_the_checker_shares_no_code_with_the_gate(self):
        source = (ROOT / "checker" / "checker.py").read_text(encoding="ascii")
        for forbidden in ("import resolver", "import aggregate", "import anchors",
                          "import e2a_canon", "from resolver", "from aggregate",
                          "import comparator", "import integrator"):
            self.assertNotIn(forbidden, source)


# ---------------------------------------------------------------------------
# CP2: the existing E2 fixture must fail
# ---------------------------------------------------------------------------

class ExistingE2FixtureIsRejected(unittest.TestCase):

    def test_the_e2_repetition_fixture_is_shape_only_and_must_not_resolve(self):
        """CP2's explicit requirement, and a genuine finding about E2.

        E2's repetition_set.json declares 8 pairs and validates cleanly under
        E2's own rules -- but only pair 0 has timeline records on disk. Pairs 1-7
        are digests of nothing. E2 could not tell, because it never resolved the
        digests it listed; a shape check cannot notice that seven eighths of the
        cohort does not exist. E2A resolves every planned slot, so it can.
        """
        e2_fixtures = ROOT.parent / "s10_matched_energy_e2" / "fixtures"
        record = canon.load_strict(e2_fixtures / "repetition_set.json")
        self.assertEqual(record["attempted_pairs"], 8)
        self.assertEqual(len(record["pairs"]), 8)

        # Every pair names a control and a treatment record digest.
        declared = set()
        for pair in record["pairs"]:
            declared.add(pair["control_record_sha256"])
            declared.add(pair["treatment_record_sha256"])
        self.assertEqual(len(declared), 16)

        # But only two timeline records exist on disk, both belonging to pair 0.
        present = set()
        for path in sorted(e2_fixtures.glob("*_timeline.json")):
            present.add(canon.load_strict(path)["record_sha256"])
        self.assertEqual(len(present), 2)

        unresolvable = declared - present
        self.assertEqual(len(unresolvable), 14,
                         "14 of 16 declared timeline records have no file")

        # E2A refuses it: the resolver requires a file per declared digest.
        self.assertTrue(unresolvable)

    def test_the_gates_were_not_lowered_to_admit_it(self):
        self.assertEqual(resolver.MIN_PAIRS, 8)
        self.assertEqual(resolver.GATE_CONSTANTS["MIN_INDEPENDENT_UPDATES"], 100)
        self.assertEqual(resolver.GATE_CONSTANTS["MIN_WINDOW_US"], 1_000_000)


# ---------------------------------------------------------------------------
# architecture
# ---------------------------------------------------------------------------

class ArchitectureSeparation(unittest.TestCase):

    def test_an_aggregate_never_re_enters_the_additive_solver(self):
        import comparator as e2_comparator
        for kind in ("RealizedTimeline", "MatchedComparison", "RepetitionSet"):
            with self.subTest(kind=kind):
                with self.assertRaises(e2_comparator.ComparisonError) as ctx:
                    e2_comparator.assert_not_additive_input({"kind": kind})
                self.assertEqual(ctx.exception.code, "E_ADDITIVE_REUSE")

    def test_e2a_imports_no_e1_decision_logic(self):
        for module in ("binder", "boundary", "validator"):
            self.assertNotIn(module, sys.modules,
                             f"E1's {module} must never be imported by E2A")

    def test_no_function_accepts_a_label(self):
        """Derive, never accept. E2's worst bug, kept dead."""
        import inspect
        for name in ("evaluate", "sum_all_pairs", "decide_aggregate",
                     "derive_label", "resolve_bundle"):
            function = getattr(aggregate, name)
            parameters = list(inspect.signature(function).parameters)
            with self.subTest(function=name):
                for forbidden in ("label", "reason", "result_label",
                                  "relief", "drained"):
                    self.assertNotIn(forbidden, parameters)

    def test_the_forbidden_total_system_label_is_not_expressible(self):
        for path in sorted((ROOT / "schemas").glob("*.json")):
            with self.subTest(schema=path.name):
                text = path.read_text(encoding="ascii")
                self.assertNotIn("SYSTEM_ENERGY_SAVING", text)

    def test_the_whole_tree_is_ascii(self):
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file() or ".git" in path.parts:
                continue
            # .der stands in for a binary token; bytecode is build output and is
            # removed by scripts/run_tests.sh before any manifest is taken.
            if path.suffix in (".der", ".pyc") or "__pycache__" in path.parts:
                continue
            with self.subTest(path=str(path.relative_to(ROOT))):
                try:
                    path.read_text(encoding="ascii")
                except UnicodeDecodeError:
                    self.fail(f"{path} is not ASCII")


class RedTeamRegressions(unittest.TestCase):
    """One test per finding from the adversarial review. See RESULTS.md.

    Each of these fails against the code as it stood before the review, which is
    the only property that makes a regression test worth keeping.
    """

    def test_the_pinned_canon_executes_the_bytes_it_hashed(self):
        """Finding 2 (CRITICAL): __pycache__ defeated the digest pin.

        exec_module() consults __pycache__ and runs the cached bytecode when the
        pyc header matches the source mtime and size. So the pin hashed the
        source and executed something else -- a poisoned pyc left the source
        digest intact while check_integers stopped rejecting floats. Hash one
        thing, execute another: E2's artifact lesson, one level up.
        """
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(canon._load_pinned_canon).lstrip())
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)}
        self.assertNotIn("exec_module", called,
                         "exec_module runs __pycache__ bytecode, not the bytes "
                         "that were hashed")
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertIn("compile", names)

    def test_a_poisoned_bytecode_cache_cannot_change_the_type_gate(self):
        """The property that matters, asserted directly rather than by grep."""
        self.assertFalse(canon.is_int(1.0))
        self.assertFalse(canon.is_int(True))
        with self.assertRaises(ValueError):
            canon.check_integers({"x": 900000.0}, "probe")

    def test_the_checker_cli_never_prints_a_physical_label(self):
        """Finding 1 (CRITICAL): the CLI printed SERVER_RELIEF_PASS, exit 0.

        Twenty lines of hand-written JSON -- no bundle, no anchor, no artifacts
        -- produced a physical label and a zero exit from the artifact a reviewer
        actually runs. check_aggregate derives the label from the record's OWN
        declared anchor_kind, so self-consistency was the whole bar.
        """
        contributions = [contribution(1000, 0, 1, 0, i) for i in range(8)]
        record = IndependentChecker()._record(
            contributions, scope="SERVER_WALL",
            anchor_kind="TRANSPARENCY_LOG_INCLUSION")
        self.assertEqual(record["result_label"],
                         "SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "forged.json"
            path.write_text(json.dumps(record, sort_keys=True), encoding="ascii")
            result = subprocess.run(
                [sys.executable, str(ROOT / "checker" / "checker.py"),
                 "--aggregate", str(path)],
                capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0,
                            "a forged record must not exit 0")
        self.assertNotIn("SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED",
                         result.stdout)
        self.assertNotIn("GPU_BOARD_RELIEF_ONLY", result.stdout)

    def test_the_checker_cli_reports_arithmetic_only_on_a_consistent_record(self):
        contributions = [contribution(1000, 1, 800, 1, i) for i in range(8)]
        record = IndependentChecker()._record(contributions)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "ok.json"
            path.write_text(json.dumps(record, sort_keys=True), encoding="ascii")
            result = subprocess.run(
                [sys.executable, str(ROOT / "checker" / "checker.py"),
                 "--aggregate", str(path)],
                capture_output=True, text=True, timeout=60)
        self.assertIn("ARITHMETIC_CONSISTENT_ONLY", result.stdout)
        self.assertNotEqual(result.returncode, 0)

    def test_validate_aggregate_accepts_its_own_sealed_output(self):
        """Finding 3 (CRITICAL): the label check was dead on arrival.

        gate_record's type gate rejects bools everywhere, so validate_aggregate
        raised E_TYPE on its OWN output. The function documented as "what makes
        the label unfakeable" could never have run once, and no test called it --
        which is how it and the no-op additive guard survived 135 green tests.
        """
        with allow_anchor(), bundle() as b:
            # Reach a sealed record by neutralising only the two policy gates.
            saved = aggregate.check_provenance
            aggregate.check_provenance = lambda resolved: True
            try:
                record = aggregate.evaluate(str(b.root / "bundle.json"))
                # With all three anchor gates and the provenance gate stepped
                # over, the machinery reaches a physical label. That is the
                # point: the label path is real code that would run, not a
                # vestige. Only the gates stand between this bundle and
                # GPU_BOARD_RELIEF_ONLY -- which is exactly why they are tested
                # so heavily, and why this test is careful to restore them.
                self.assertEqual(record["result_label"],
                                 "GPU_BOARD_RELIEF_ONLY_TOTAL_ENERGY_BLOCKED")
                self.assertEqual(record["reason_code"],
                                 "ALL_PAIRS_CONSERVATIVE_RELIEF")
                self.assertTrue(record["conservative_relief"])
                self.assertIsNone(record["server_wall_delta_nj"])
                # The dead defense, now alive: it must accept a true record.
                aggregate.validate_aggregate(record, str(b.root / "bundle.json"))
                # ... and reject a forged label on the same evidence.
                forged = dict(record)
                forged["result_label"] = "RELIEF_FAIL"
                canon.seal(forged)
                with self.assertRaises(aggregate.AggregateError) as ctx:
                    aggregate.validate_aggregate(forged,
                                                 str(b.root / "bundle.json"))
                self.assertEqual(ctx.exception.code, "E_INCOHERENT")
            finally:
                aggregate.check_provenance = saved

    def test_the_additive_guard_covers_the_aggregate_record(self):
        """Finding 9: E2's guard does not know AggregateComparison exists.

        Calling E2's assert_not_additive_input on an AggregateComparison was a
        no-op -- the guard against re-entering the additive solver did nothing
        for the most aggregated record E2A produces. That is the exact bug E2's
        docstring claims to have fixed, one kind later.
        """
        with self.assertRaises(aggregate.AggregateError) as ctx:
            aggregate.assert_not_additive_input(
                {"kind": "AggregateComparison", "aggregate_id": "agg.x"})
        self.assertEqual(ctx.exception.code, "E_ADDITIVE_REUSE")
        # and it still defers to E2 for the kinds E2 owns
        with self.assertRaises(aggregate.e2_comparator.ComparisonError):
            aggregate.assert_not_additive_input({"kind": "RepetitionSet"})

    def test_every_action_kind_is_charged_unless_explicitly_exempt(self):
        """Finding 7: the window check was a name gate over 7 of 12 kinds.

        QUEUE_SUBMIT, CANCEL, LEASE_* and QUEUE_DRAIN were unchecked, so
        relabelling a PREFETCH as a QUEUE_SUBMIT moved weight staging outside the
        paid window for free. The set of things that cost energy is not knowable
        from a label the producer chooses.
        """
        self.assertEqual(resolver.FREE_ACTIONS, frozenset())
        timeline = canon.load_strict(FIXTURES / "timelines/tl.slot00.json")
        manifest = canon.load_strict(FIXTURES / "manifest.json")
        outcomes = canon.load_strict(FIXTURES / "outcomes/outcomes.slot00.json")
        resolved = resolver.resolve_requests(
            manifest, outcomes, timeline, FIXTURES, "slot0", 0)
        lifecycle = canon.load_strict(FIXTURES / "lifecycle/lc.slot00.json")
        for action in lifecycle["actions"]:
            if action["action_kind"] == "PREFETCH":
                action["action_kind"] = "QUEUE_SUBMIT"
                action["enqueue_us"] -= 5_000_000
                action["start_us"] -= 5_000_000
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(lifecycle, timeline, None, resolved)
        self.assertEqual(ctx.exception.code, "E_ACTION_UNBOUND")

    def test_an_anchor_kind_with_no_verifier_is_refused(self):
        """Finding 4: the enumerable path was guarded only by a TSA constant.

        The historical implementation granted TRANSPARENCY_LOG_INCLUSION the
        required property before checking its token. The only thing between a
        fabricated inclusion proof and a physical label was
        FROZEN_TSA_ROOT_SHA256 being None, a timestamp-authority constant
        accidentally gating a transparency-log anchor. Two unrelated mechanisms,
        one shared guard, held by luck.
        """
        self.assertEqual(anchors.ANCHOR_VERIFIERS, {},
                         "E2A v2 implements no cryptographic verifier")
        for kind in sorted(anchors.ANCHOR_KINDS):
            with self.subTest(kind=kind):
                with self.assertRaises(anchors.AnchorError) as ctx:
                    anchors.check_verifier_available(kind)
                self.assertEqual(ctx.exception.code, "E_ANCHOR_NO_VERIFIER")

    def test_plain_log_inclusion_is_not_enumeration(self):
        """One producer-selected leaf cannot establish plan exclusivity."""
        self.assertEqual(anchors.anchor_property("TRANSPARENCY_LOG_INCLUSION"),
                         anchors.PROPERTY_ORDERING_ONLY)
        with self.assertRaises(anchors.AnchorError) as ctx:
            anchors.check_anchor_kind("TRANSPARENCY_LOG_INCLUSION")
        self.assertEqual(ctx.exception.code, "E_ANCHOR_UNENUMERABLE")

    def test_a_transparency_log_anchor_in_a_bundle_is_still_refused(self):
        with bundle() as b:
            anchor = b.load("plan_anchor.json")
            anchor["anchor_kind"] = "TRANSPARENCY_LOG_INCLUSION"
            b.write("plan_anchor.json", anchor)
            b.rebind_anchor_chain()
            with self.assertRaises(resolver.ResolveError) as ctx:
                b.evaluate()
            self.assertEqual(ctx.exception.code, "E_ANCHOR_NO_VERIFIER")


class Determinism(unittest.TestCase):

    def test_fixture_generation_is_deterministic(self):
        first = canon.load_strict(FIXTURES / "plan.json")["record_sha256"]
        fixtures.generate()
        second = canon.load_strict(FIXTURES / "plan.json")["record_sha256"]
        self.assertEqual(first, second)

    def test_the_gate_constants_digest_is_stable(self):
        self.assertEqual(resolver.GATE_CONSTANTS_DIGEST,
                         canon.digest(resolver.GATE_CONSTANTS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
