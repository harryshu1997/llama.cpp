#!/usr/bin/env python3
"""Regressions for the E2A v2 evidence-chain repair."""

from __future__ import annotations

import json
import importlib.util
import marshal
import os
import pathlib
import struct
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import aggregate  # noqa: E402
import anchors  # noqa: E402
import e2a_canon as canon  # noqa: E402
import e2a_e2  # noqa: E402
import resolver  # noqa: E402
from test_e2a import _rechain, bundle  # noqa: E402

FIXTURES = ROOT / "fixtures"


class EvidenceJoinRepair(unittest.TestCase):

    def test_every_plan_timeline_field_is_bound(self):
        plan = canon.load_strict(FIXTURES / "plan.json")
        original = canon.load_strict(FIXTURES / "timelines/tl.slot00.json")
        fields = aggregate.PLAN_TIMELINE_FIELDS + \
            aggregate.PLAN_TIMELINE_SET_FIELDS
        for field in fields:
            with self.subTest(field=field):
                timeline = dict(original)
                if isinstance(timeline[field], list):
                    timeline[field] = list(timeline[field]) + ["unexpected"]
                else:
                    timeline[field] = "unexpected"
                with self.assertRaises(aggregate.AggregateError) as ctx:
                    aggregate._bind_timeline_to_plan(
                        timeline, plan, plan["slots"][0], 0)
                self.assertEqual(ctx.exception.code, "E_PLAN_TIMELINE_BINDING")

    def test_policy_is_bound_by_role(self):
        plan = canon.load_strict(FIXTURES / "plan.json")
        timeline = canon.load_strict(FIXTURES / "timelines/tl.slot00.json")
        timeline["policy_digest"] = plan["policy_digest_treatment"]
        with self.assertRaises(aggregate.AggregateError) as ctx:
            aggregate._bind_timeline_to_plan(timeline, plan, plan["slots"][0], 0)
        self.assertEqual(ctx.exception.code, "E_PLAN_TIMELINE_BINDING")

    def test_lifecycle_rewrite_is_not_authorized_by_reindexing(self):
        with bundle() as work:
            lifecycle = work.load("lifecycle/lc.slot00.json")
            lifecycle["actions"][0]["reason"] = "not-schema-valid"
            work.write("lifecycle/lc.slot00.json", lifecycle)
            work.reindex()
            with self.assertRaises((aggregate.AggregateError,
                                    resolver.ResolveError)) as ctx:
                work.evaluate()
            self.assertEqual(ctx.exception.code, "E_SLOT_BINDING")

    def test_close_anchors_the_complete_ledger_not_only_its_head(self):
        with bundle() as work:
            ledger = work.load("ledger.json")
            close = work.load("ledger_close.json")
            close["anchored_digest"] = ledger["head_sha256"]
            close["message_imprint_sha256"] = ledger["head_sha256"]
            work.write("ledger_close.json", close)
            work.reindex()
            with self.assertRaises(resolver.ResolveError) as ctx:
                work.evaluate()
            self.assertEqual(ctx.exception.code, "E_ANCHOR_BINDING")

    def test_run_specific_output_paths_are_not_reused(self):
        paths = []
        for path in sorted((FIXTURES / "outcomes").glob("*.json")):
            record = canon.load_strict(path)
            paths.extend(item["output_artifact_path"]
                         for item in record["outcomes"])
        self.assertEqual(len(paths), 48)
        self.assertEqual(len(set(paths)), 48)


class LedgerStateMachineRepair(unittest.TestCase):

    def _records(self):
        return (canon.load_strict(FIXTURES / "plan.json"),
                canon.load_strict(FIXTURES / "ledger.json"),
                canon.load_strict(FIXTURES / "plan_anchor.json"))

    def test_terminal_only_ledger_is_refused(self):
        plan, ledger, anchor = self._records()
        ledger["entries"] = [entry for entry in ledger["entries"]
                             if entry["entry_kind"] == "SLOT_ATTEMPT_END"]
        for index, entry in enumerate(ledger["entries"]):
            entry["seq"] = index
        _rechain(ledger)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_ledger(ledger, plan, anchor)
        self.assertEqual(ctx.exception.code, "E_SLOT_UNCOVERED")

    def test_clock_epoch_change_is_refused(self):
        plan, ledger, anchor = self._records()
        ledger["entries"][5]["clock_epoch_id"] = "another-boot"
        _rechain(ledger)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_ledger(ledger, plan, anchor)
        self.assertEqual(ctx.exception.code, "E_CLOCK_EPOCH")

    def test_backward_time_is_refused(self):
        plan, ledger, anchor = self._records()
        ledger["entries"][5]["monotonic_us"] = 0
        _rechain(ledger)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_ledger(ledger, plan, anchor)
        self.assertEqual(ctx.exception.code, "E_LEDGER_TIME")

    def test_terminal_binds_all_three_evidence_records(self):
        plan, ledger, anchor = self._records()
        ledger["entries"][3]["lifecycle_record_sha256"] = None
        _rechain(ledger)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_ledger(ledger, plan, anchor)
        self.assertEqual(ctx.exception.code, "E_LEDGER_BINDING")


class RealizedWorkRepair(unittest.TestCase):

    def test_output_count_is_recomputed_from_token_events(self):
        with bundle() as work:
            manifest = work.load("manifest.json")
            outcomes = work.load("outcomes/outcomes.slot00.json")
            timeline = work.load("timelines/tl.slot00.json")
            item = outcomes["outcomes"][0]
            output = work.load(item["output_artifact_path"])
            output["token_events"] = output["token_events"][:1]
            item["output_artifact_sha256"] = work.write(
                item["output_artifact_path"], output, reseal=False)
            canon.seal(outcomes)
            with self.assertRaises(resolver.ResolveError) as ctx:
                resolver.resolve_requests(
                    manifest, outcomes, timeline, work.root, "slot0", 0)
            self.assertEqual(ctx.exception.code, "E_OUTPUT_COUNT")

    def test_slo_outcome_is_derived_from_latency(self):
        manifest = canon.load_strict(FIXTURES / "manifest.json")
        outcomes = canon.load_strict(FIXTURES / "outcomes/outcomes.slot00.json")
        timeline = canon.load_strict(FIXTURES / "timelines/tl.slot00.json")
        item = outcomes["outcomes"][0]
        item["last_token_us"] = item["dispatched_us"] + 6_000_000
        canon.seal(outcomes)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_requests(
                manifest, outcomes, timeline, FIXTURES, "slot0", 0)
        self.assertEqual(ctx.exception.code, "E_SLO_OUTCOME")

    def test_prefix_certificate_is_recomputed_from_both_outputs(self):
        with bundle() as work:
            manifest = work.load("manifest.json")
            control_record = work.load("outcomes/outcomes.slot00.json")
            treatment_record = work.load("outcomes/outcomes.slot01.json")
            control_timeline = work.load("timelines/tl.slot00.json")
            treatment_timeline = work.load("timelines/tl.slot01.json")

            item = treatment_record["outcomes"][0]
            output = work.load(item["output_artifact_path"])
            output["token_events"][0]["token_id"] += 1
            token_ids = [event["token_id"] for event in output["token_events"]]
            item["output_token_ids_sha256"] = canon.digest(token_ids)
            item["output_artifact_sha256"] = work.write(
                item["output_artifact_path"], output, reseal=False)
            canon.seal(treatment_record)

            control = resolver.resolve_requests(
                manifest, control_record, control_timeline, work.root, "control", 0)
            treatment = resolver.resolve_requests(
                manifest, treatment_record, treatment_timeline, work.root,
                "treatment", 0)
            with self.assertRaises(resolver.ResolveError) as ctx:
                resolver.check_same_work(control, treatment, manifest)
            self.assertEqual(ctx.exception.code, "E_CERT_FAIL")

    def test_only_drain_is_not_a_lifecycle(self):
        manifest = canon.load_strict(FIXTURES / "manifest.json")
        outcomes = canon.load_strict(FIXTURES / "outcomes/outcomes.slot00.json")
        timeline = canon.load_strict(FIXTURES / "timelines/tl.slot00.json")
        resolved = resolver.resolve_requests(
            manifest, outcomes, timeline, FIXTURES, "slot0", 0)
        lifecycle = canon.load_strict(FIXTURES / "lifecycle/lc.slot00.json")
        lifecycle["actions"] = [action for action in lifecycle["actions"]
                                if action["action_kind"] == "QUEUE_DRAIN"]
        lifecycle["entry_state_digest"] = resolver.lifecycle_state_digest(
            lifecycle["actions"], timeline["window_start_us"])
        lifecycle["exit_state_digest"] = resolver.lifecycle_state_digest(
            lifecycle["actions"], timeline["window_end_us"])
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(lifecycle, timeline, None, resolved)
        self.assertEqual(ctx.exception.code, "E_LIFECYCLE_CAUSAL")


class ExecutedCodeAndAnchorRepair(unittest.TestCase):

    def test_transitive_e1_pyc_cannot_replace_pinned_e2_canon(self):
        source_path = pathlib.Path(canon.CANON_PATH)
        cache_path = pathlib.Path(importlib.util.cache_from_source(str(source_path)))
        original_cache = cache_path.read_bytes() if cache_path.exists() else None
        source = source_path.read_text(encoding="ascii")
        hostile = source.replace(
            "return type(value) is int",
            "return isinstance(value, (int, float))",
            1)
        self.assertNotEqual(hostile, source)
        code = compile(hostile, str(source_path), "exec")
        stat = source_path.stat()
        pyc = (importlib.util.MAGIC_NUMBER + struct.pack(
            "<III", 0, int(stat.st_mtime), stat.st_size) + marshal.dumps(code))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache_path.write_bytes(pyc)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(e2a_e2.E2_SRC)
            vulnerable = subprocess.run(
                [sys.executable, "-c",
                 "import e2_canon; print(e2_canon.is_int(1.0))"],
                cwd=str(ROOT), env=env, capture_output=True, text=True,
                timeout=30, check=False)
            self.assertEqual(vulnerable.returncode, 0, vulnerable.stderr)
            self.assertEqual(vulnerable.stdout.strip(), "True")

            env["PYTHONPATH"] = str(ROOT / "src")
            result = subprocess.run(
                [sys.executable, "-c",
                 "import e2a_e2; print(e2a_e2.e2_canon.is_int(1.0))"],
                cwd=str(ROOT), env=env, capture_output=True, text=True,
                timeout=30, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "False")
            self.assertIs(e2a_e2.e2_canon.is_int, canon.is_int)
        finally:
            if original_cache is None:
                cache_path.unlink(missing_ok=True)
            else:
                cache_path.write_bytes(original_cache)

    def test_hostile_preloaded_e2_modules_are_ignored(self):
        program = f"""
import sys, types
evil_i = types.ModuleType('integrator')
evil_i.check_scope = lambda *args: None
evil_c = types.ModuleType('comparator')
evil_c.decide = lambda *args: {{'relief': True}}
sys.modules['integrator'] = evil_i
sys.modules['comparator'] = evil_c
sys.path.insert(0, {str(ROOT / 'src')!r})
import aggregate
assert aggregate.e2_integrator is not evil_i
assert aggregate.e2_comparator is not evil_c
assert aggregate.e2_integrator.__name__.startswith('_s10_e2a_pinned_')
assert aggregate.e2_comparator.__name__.startswith('_s10_e2a_pinned_')
"""
        result = subprocess.run([sys.executable, "-c", program],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_verifier_is_invoked_but_p1_still_fails_closed(self):
        plan = canon.load_strict(FIXTURES / "plan.json")
        receipt = canon.load_strict(FIXTURES / "plan_anchor.json")
        token = (FIXTURES / receipt["token_der_path"]).read_bytes()
        calls = []

        def verifier(value):
            calls.append(value)
            return {field: receipt[field]
                    for field in anchors.VERIFIED_TOKEN_FIELDS}

        key = (receipt["anchor_kind"], receipt["verifier_kind"])
        saved_registry = dict(anchors.ANCHOR_VERIFIERS)
        saved_root = anchors.check_trust_root
        anchors.ANCHOR_VERIFIERS[key] = verifier
        anchors.check_trust_root = lambda record: True
        try:
            with self.assertRaises(resolver.ResolveError) as ctx:
                resolver.resolve_plan_anchor(receipt, plan, FIXTURES)
            self.assertEqual(ctx.exception.code,
                             "E_ANCHOR_PRECEDENCE_UNSUPPORTED")
            self.assertEqual(calls, [token])
        finally:
            anchors.ANCHOR_VERIFIERS.clear()
            anchors.ANCHOR_VERIFIERS.update(saved_registry)
            anchors.check_trust_root = saved_root


if __name__ == "__main__":
    unittest.main(verbosity=2)
