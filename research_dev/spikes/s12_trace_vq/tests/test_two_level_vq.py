#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


SPIKE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPIKE))

from s12lib import S12Error, canonical_json, load_json, load_jsonl  # noqa: E402
from two_level_vq import (  # noqa: E402
    DeviceView,
    Replica,
    ReplicaView,
    SCOPE,
    Simulator,
    SlowSnapshot,
    build_catalog,
    plan_slow,
    run_config,
    validate_config,
    validate_cross_inputs,
    validate_profile,
    validate_trace,
)


CONFIG_PATH = SPIKE / "configs" / "two_level_fixture.json"
PROFILE_PATH = SPIKE / "profiles" / "two_model_mechanics.json"
TRACE_PATH = SPIKE / "fixtures" / "two_model_arrivals.jsonl"
MIB = 1024 * 1024


def fixture_inputs():
    cfg = validate_config(copy.deepcopy(load_json(CONFIG_PATH)))
    profile = validate_profile(copy.deepcopy(load_json(PROFILE_PATH)))
    catalog = build_catalog(profile)
    validate_cross_inputs(cfg, catalog)
    trace = validate_trace(copy.deepcopy(load_jsonl(TRACE_PATH)), catalog)
    return cfg, catalog, trace


def fixture_result(policy: str):
    cfg, catalog, trace = fixture_inputs()
    return Simulator(policy, cfg, catalog, trace).run()


def policy_result(manifest, policy: str):
    return next(result for result in manifest["results"] if result["policy"] == policy)


def write_run(tmp: Path, cfg: dict, trace: list[dict]):
    trace_path = tmp / "trace.jsonl"
    trace_path.write_text(
        "".join(canonical_json(record) + "\n" for record in trace),
        encoding="ascii",
    )
    cfg = copy.deepcopy(cfg)
    cfg["profile_path"] = str(PROFILE_PATH.resolve())
    cfg["trace_path"] = str(trace_path.resolve())
    config_path = tmp / "config.json"
    config_path.write_text(canonical_json(cfg) + "\n", encoding="ascii")
    return run_config(config_path)


class TwoLevelContractTests(unittest.TestCase):
    def test_bool_cannot_impersonate_integer(self):
        cfg = copy.deepcopy(load_json(CONFIG_PATH))
        cfg["horizon_us"] = True
        with self.assertRaises(S12Error):
            validate_config(cfg)

    def test_unknown_config_key_is_rejected(self):
        cfg = copy.deepcopy(load_json(CONFIG_PATH))
        cfg["future_hint"] = []
        with self.assertRaisesRegex(S12Error, "key mismatch"):
            validate_config(cfg)

    def test_trace_must_be_ordered_and_model_bound(self):
        _, catalog, trace = fixture_inputs()
        unordered = copy.deepcopy(trace)
        unordered[1]["arrival_us"] = 200000
        with self.assertRaisesRegex(S12Error, "nondecreasing"):
            validate_trace(unordered, catalog)
        wrong_model = copy.deepcopy(trace)
        wrong_model[0]["model_id"] = "other"
        with self.assertRaisesRegex(S12Error, "does not own"):
            validate_trace(wrong_model, catalog)

    def test_initial_residency_must_fit_and_be_compatible(self):
        cfg, catalog, _ = fixture_inputs()
        cfg["devices"][0]["dynamic_initial_islands"] = ["island_a_head", "island_b_encoder"]
        with self.assertRaisesRegex(S12Error, "exceeds capacity"):
            validate_cross_inputs(cfg, catalog)
        cfg, catalog, _ = fixture_inputs()
        cfg["devices"][1]["dynamic_initial_islands"] = ["island_b_encoder"]
        with self.assertRaisesRegex(S12Error, "incompatible"):
            validate_cross_inputs(cfg, catalog)

    def test_profile_is_explicitly_synthetic(self):
        profile = validate_profile(copy.deepcopy(load_json(PROFILE_PATH)))
        self.assertEqual(profile["scope"], SCOPE)
        self.assertEqual(profile["energy_status"], "NOT_RUN")
        for island in profile["islands"]:
            for route in island["device_routes"]:
                self.assertEqual(route["correctness_status"], "ASSUMED_SYNTHETIC_ONLY")

    def test_scheduler_score_overflow_fails_closed(self):
        cfg, catalog, trace = fixture_inputs()
        catalog["island_a_head"]["synthetic_server_full_us"] = 9007199254740991
        replay = Simulator("dynamic_two_level", cfg, catalog, trace)
        for request in replay.requests[:4]:
            replay._handle_arrival(request)
        with self.assertRaisesRegex(S12Error, "score exceeds"):
            replay._run_slow_loop()

    def test_cross_device_aggregate_bytes_cannot_overflow(self):
        cfg, catalog, trace = fixture_inputs()
        for device in cfg["devices"]:
            device["capacity_bytes"] = 9007199254740991
        for island in catalog.values():
            island["weight_bytes"] = 9007199254740991
        with self.assertRaisesRegex(S12Error, "initial_residency_bytes: integer overflow"):
            Simulator("static_two_phone", cfg, catalog, trace)

    def test_illegal_replica_transition_fails_closed(self):
        replica = Replica("op12", "island_a_head", 1, 1, "READY")
        with self.assertRaisesRegex(AssertionError, "invalid replica transition"):
            replica.transition("VERIFYING")

    def test_global_score_wins_shared_prefetch_slot(self):
        cfg, catalog, trace = fixture_inputs()
        cfg["devices"][1]["dynamic_initial_islands"] = []
        catalog["island_a_head"]["routes"].pop("op15")
        replay = Simulator("dynamic_two_level", cfg, catalog, trace)
        for request_id in ("a0", "a1", "b0", "b1"):
            replay._handle_arrival(replay.request_by_id[request_id])
        placements = [
            action
            for action in plan_slow(replay._snapshot(), catalog, cfg["scheduler"])
            if action["action"] != "KEEP"
        ]
        self.assertEqual(len(placements), 1)
        self.assertEqual(placements[0]["island_id"], "island_b_encoder")
        self.assertEqual(placements[0]["score_us"], 80000)

    def test_plan_never_evicts_last_two_demanded_replicas(self):
        def island(island_id, routes):
            return {
                "island_id": island_id,
                "model_id": "m",
                "weight_identity": f"synthetic:{island_id}",
                "routes": {
                    device_id: {
                        "backend": "HTP0",
                        "synthetic_phone_us": 10,
                        "transfer_us": 10,
                        "verify_us": 10,
                        "prepare_us": 10,
                    }
                    for device_id in routes
                },
                "synthetic_server_full_us": 100,
                "synthetic_server_tail_us": 10,
                "weight_bytes": 64,
            }

        catalog = {
            "a_new": island("a_new", ["d1"]),
            "b_new": island("b_new", ["d2"]),
            "z_old": island("z_old", ["d1", "d2"]),
        }
        replica1 = ReplicaView("d1", "z_old", "READY", 0, 1, 1, 1)
        replica2 = ReplicaView("d2", "z_old", "READY", 0, 1, 1, 1)
        snapshot = SlowSnapshot(
            now_us=0,
            queued_by_island=(("a_new", 2), ("b_new", 2), ("z_old", 1)),
            observed_by_island=(("a_new", 2), ("b_new", 2), ("z_old", 1)),
            devices=(
                DeviceView("d1", 1, 64, 64, 0, None, (replica1,)),
                DeviceView("d2", 1, 64, 64, 0, None, (replica2,)),
            ),
        )
        scheduler = {
            "min_prefetch_score_us": 0,
            "prefetch_min_observations": 2,
            "replicate_min_queue": 2,
            "reuse_horizon_requests": 4,
        }
        actions = plan_slow(snapshot, catalog, scheduler)
        evictions = [action for action in actions if action["victim_island_id"] == "z_old"]
        keeps = [action for action in actions if action["action"] == "KEEP"]
        self.assertEqual(len(evictions), 1)
        self.assertEqual(len(keeps), 1)
        self.assertNotEqual(evictions[0]["device_id"], keeps[0]["device_id"])

    def test_two_device_assignment_maximizes_total_score(self):
        def island(island_id, phone_by_device):
            return {
                "island_id": island_id,
                "model_id": "m",
                "weight_identity": f"synthetic:{island_id}",
                "routes": {
                    device_id: {
                        "backend": "HTP0",
                        "synthetic_phone_us": phone_us,
                        "transfer_us": 10,
                        "verify_us": 10,
                        "prepare_us": 10,
                    }
                    for device_id, phone_us in phone_by_device.items()
                },
                "synthetic_server_full_us": 100,
                "synthetic_server_tail_us": 10,
                "weight_bytes": 64,
            }

        catalog = {
            "a": island("a", {"d1": 10, "d2": 20}),
            "b": island("b", {"d1": 25}),
        }
        snapshot = SlowSnapshot(
            now_us=0,
            queued_by_island=(("a", 2), ("b", 2)),
            observed_by_island=(("a", 2), ("b", 2)),
            devices=(
                DeviceView("d1", 1, 64, 0, 0, None, ()),
                DeviceView("d2", 1, 64, 0, 0, None, ()),
            ),
        )
        scheduler = {
            "min_prefetch_score_us": 0,
            "prefetch_min_observations": 2,
            "replicate_min_queue": 2,
            "reuse_horizon_requests": 4,
        }
        placements = [
            action for action in plan_slow(snapshot, catalog, scheduler) if action["action"] != "KEEP"
        ]
        self.assertEqual(
            {(action["island_id"], action["device_id"]) for action in placements},
            {("a", "d2"), ("b", "d1")},
        )
        self.assertEqual(sum(action["score_us"] for action in placements), 210)

    def test_planner_input_has_no_future_collection(self):
        cfg, catalog, trace = fixture_inputs()
        replay = Simulator("dynamic_two_level", cfg, catalog, trace)
        snapshot = replay._snapshot()
        self.assertFalse(hasattr(snapshot, "future"))
        self.assertFalse(hasattr(snapshot, "events"))
        self.assertEqual(plan_slow(snapshot, catalog, cfg["scheduler"]), [])


class TwoLevelMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = run_config(CONFIG_PATH)
        cls.dynamic = policy_result(cls.manifest, "dynamic_two_level")
        cls.static = policy_result(cls.manifest, "static_two_phone")
        cls.server = policy_result(cls.manifest, "server_only")
        cls.fixed = policy_result(cls.manifest, "fixed_static_phone")

    def test_manifest_scope_and_energy_are_fail_closed(self):
        self.assertEqual(self.manifest["scope"], SCOPE)
        self.assertEqual(self.manifest["energy"], {"status": "NOT_RUN"})
        encoded = canonical_json(self.manifest).lower()
        for forbidden in ("joule", "energy_saving", "server_relief_pass", "total_wall_nj"):
            self.assertNotIn(forbidden, encoded)
        for result in self.manifest["results"]:
            self.assertEqual(result["energy"], {"status": "NOT_RUN"})

    def test_dynamic_actions_replicate_then_drain_and_prefetch(self):
        actions = self.dynamic["slow_actions"]
        replicate = next(action for action in actions if action["action"] == "REPLICATE")
        self.assertEqual((replicate["time_us"], replicate["device_id"]), (0, "op12"))
        self.assertEqual(replicate["island_id"], "island_a_head")
        replacement = next(
            action for action in actions if action["action"] == "DEFERRED_EVICT_PREFETCH"
        )
        self.assertEqual(replacement["time_us"], 161000)
        self.assertEqual(replacement["device_id"], "op12")
        self.assertEqual(replacement["victim_island_id"], "island_a_head")
        self.assertEqual(replacement["reason"], "victim_pinned_drain_started")

    def test_cold_miss_uses_server_without_waiting_for_ready(self):
        b0 = next(row for row in self.dynamic["outcomes"] if row["request_id"] == "b0")
        b1 = next(row for row in self.dynamic["outcomes"] if row["request_id"] == "b1")
        self.assertEqual((b0["route"], b0["start_us"]), ("server", 190000))
        self.assertEqual((b1["route"], b1["device_id"], b1["start_us"]), ("phone", "op12", 250000))

    def test_dispatch_requires_ready_generation(self):
        phone_actions = [
            action for action in self.dynamic["fast_actions"] if action["action"] == "DISPATCH_PHONE"
        ]
        self.assertTrue(phone_actions)
        self.assertTrue(all(action["status_before"] == "READY" for action in phone_actions))
        b_actions = [action for action in phone_actions if action["island_id"] == "island_b_encoder"]
        self.assertTrue(all(action["start_us"] >= 250000 for action in b_actions))
        self.assertTrue(all(action["generation"] == 2 for action in b_actions))
        self.assertTrue(all(action["boot_epoch"] == 1 for action in phone_actions))
        self.assertTrue(all(action["status_seq_before"] >= 1 for action in phone_actions))
        self.assertTrue(all(action["batch_size"] == 1 for action in self.dynamic["fast_actions"]))
        self.assertTrue(all(len(action["request_ids"]) == 1 for action in self.dynamic["fast_actions"]))

    def test_draining_replica_receives_no_new_dispatch(self):
        op12_a = [
            action
            for action in self.dynamic["fast_actions"]
            if action["action"] == "DISPATCH_PHONE"
            and action["device_id"] == "op12"
            and action["island_id"] == "island_a_head"
        ]
        self.assertEqual([action["start_us"] for action in op12_a], [60000])

    def test_final_placement_and_independent_replica_charges(self):
        ledgers = {row["device_id"]: row for row in self.dynamic["device_ledgers"]}
        self.assertEqual(ledgers["op12"]["peak_resident_bytes"], 64 * MIB)
        self.assertEqual(ledgers["op15"]["peak_resident_bytes"], 64 * MIB)
        self.assertEqual(
            [row["island_id"] for row in ledgers["op12"]["final_residency"]],
            ["island_b_encoder"],
        )
        self.assertEqual(
            [row["island_id"] for row in ledgers["op15"]["final_residency"]],
            ["island_a_head"],
        )
        self.assertEqual(ledgers["op12"]["resident_bytes_final"], 64 * MIB)
        self.assertEqual(ledgers["op15"]["resident_bytes_final"], 64 * MIB)

    def test_transfer_eviction_and_useful_ledgers_reconcile(self):
        self.assertEqual(self.dynamic["transfer_bytes_completed"], 128 * MIB)
        self.assertEqual(self.dynamic["useful_transfer_bytes"], 128 * MIB)
        self.assertEqual(self.dynamic["wasted_transfer_bytes"], 0)
        self.assertEqual(self.dynamic["evicted_bytes"], 64 * MIB)
        self.assertEqual(self.dynamic["cancelled_prefetch_reservation_bytes"], 0)

    def test_two_phones_overlap_but_each_phone_serializes(self):
        self.assertGreater(self.dynamic["two_phone_compute_overlap_us"], 0)
        for ledger in self.dynamic["device_ledgers"]:
            intervals = ledger["compute_intervals"]
            for previous, current in zip(intervals, intervals[1:]):
                self.assertLessEqual(previous["finish_us"], current["start_us"])
            self.assertLessEqual(ledger["peak_activation_used"], ledger["activation_slots"])

    def test_every_policy_has_exact_terminal_conservation(self):
        for result in self.manifest["results"]:
            self.assertEqual(result["terminal_conservation"], result["request_count"])
            self.assertEqual(sum(result["terminal_counts"].values()), result["request_count"])
            self.assertLessEqual(result["max_queue_depth"], 4)

    def test_static_control_never_moves_weights(self):
        self.assertEqual(self.static["slow_actions"], [])
        self.assertEqual(self.static["transfer_bytes_completed"], 0)
        self.assertEqual(self.static["evicted_bytes"], 0)
        final = {
            row["device_id"]: [replica["island_id"] for replica in row["final_residency"]]
            for row in self.static["device_ledgers"]
        }
        self.assertEqual(final, {"op12": ["island_b_encoder"], "op15": ["island_a_head"]})

    def test_server_control_has_no_phone_state(self):
        self.assertEqual(self.server["slow_actions"], [])
        self.assertEqual(self.server["fast_actions"][0]["action"], "SERVER_FALLBACK")
        self.assertTrue(
            all(not row["final_residency"] for row in self.server["device_ledgers"])
        )

    def test_fixed_static_phone_runs_only_on_phone(self):
        self.assertEqual(self.fixed["terminal_counts"]["completed_server"], 0)
        self.assertEqual(self.fixed["slow_actions"], [])
        tc = self.fixed["terminal_counts"]
        self.assertEqual(
            tc["completed_phone"] + tc["tardy_phone"] + tc["tardy_server"] + tc["rejected_queue_full"] + tc["timed_out"],
            self.fixed["terminal_conservation"],
        )
        self.assertTrue(len(self.fixed["server_intervals"]) == 0)
        self.assertTrue(
            all(action["action"] == "DISPATCH_PHONE" for action in self.fixed["fast_actions"])
        )
        self.assertFalse(
            any(action["action"] == "SERVER_FALLBACK" or action["action"] == "BATCH_TAIL"
                   for action in self.fixed["fast_actions"])
        )

    def test_repeated_replay_is_identical(self):
        other = run_config(CONFIG_PATH)
        self.assertEqual(other, self.manifest)

    def test_weight_identity_is_bound_into_dispatch_and_result(self):
        cfg, catalog, trace = fixture_inputs()
        original = Simulator("dynamic_two_level", cfg, catalog, trace).run()
        catalog["island_a_head"]["weight_identity"] = "synthetic:model-a:changed:v2"
        changed = Simulator("dynamic_two_level", cfg, catalog, trace).run()
        self.assertNotEqual(original["result_digest"], changed["result_digest"])
        dispatch = next(
            action
            for action in changed["fast_actions"]
            if action["action"] == "DISPATCH_PHONE"
            and action["island_id"] == "island_a_head"
        )
        self.assertEqual(dispatch["weight_identity"], "synthetic:model-a:changed:v2")
        final = next(
            replica
            for device in changed["device_ledgers"]
            for replica in device["final_residency"]
            if replica["island_id"] == "island_a_head"
        )
        self.assertEqual(final["weight_identity"], "synthetic:model-a:changed:v2")


class TwoLevelAdversarialTests(unittest.TestCase):
    def test_future_mutation_cannot_change_prefix_actions(self):
        cfg, _, trace = fixture_inputs()
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            before = write_run(tmp, cfg, trace)
            mutated = copy.deepcopy(trace)
            mutated[5]["event_id"] = "future-mutated-b1"
            after = write_run(tmp, cfg, mutated)
        before_dynamic = policy_result(before, "dynamic_two_level")
        after_dynamic = policy_result(after, "dynamic_two_level")
        before_slow = [row for row in before_dynamic["slow_actions"] if row["time_us"] <= 160000]
        after_slow = [row for row in after_dynamic["slow_actions"] if row["time_us"] <= 160000]
        before_fast = [row for row in before_dynamic["fast_actions"] if row["start_us"] <= 160000]
        after_fast = [row for row in after_dynamic["fast_actions"] if row["start_us"] <= 160000]
        self.assertEqual(before_slow, after_slow)
        self.assertEqual(before_fast, after_fast)

    def test_replay_digest_does_not_depend_on_bundle_path(self):
        cfg, _, trace = fixture_inputs()
        profile_bytes = PROFILE_PATH.read_bytes()
        manifests = []
        with tempfile.TemporaryDirectory() as root:
            for name in ("left", "right"):
                directory = Path(root) / name
                directory.mkdir()
                (directory / "profile.json").write_bytes(profile_bytes)
                (directory / "trace.jsonl").write_text(
                    "".join(canonical_json(record) + "\n" for record in trace),
                    encoding="ascii",
                )
                local_cfg = copy.deepcopy(cfg)
                local_cfg["profile_path"] = "profile.json"
                local_cfg["trace_path"] = "trace.jsonl"
                if name == "right":
                    local_cfg["profile_path"] = str((directory / "profile.json").resolve())
                    local_cfg["trace_path"] = str((directory / "trace.jsonl").resolve())
                config_path = directory / "config.json"
                config_path.write_text(canonical_json(local_cfg) + "\n", encoding="ascii")
                manifests.append(run_config(config_path))
        self.assertNotEqual(manifests[0]["inputs"]["config_path"], manifests[1]["inputs"]["config_path"])
        self.assertEqual(
            manifests[0]["deterministic_replay_sha256"],
            manifests[1]["deterministic_replay_sha256"],
        )

    def test_queue_overflow_is_explicit_and_conserved(self):
        cfg, _, trace = fixture_inputs()
        cfg["queue_limit"] = 1
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_run(Path(directory), cfg, trace)
        for result in manifest["results"]:
            self.assertGreater(result["terminal_counts"]["rejected_queue_full"], 0)
            self.assertEqual(result["terminal_conservation"], len(trace))

    def test_horizon_cancels_prefetch_and_releases_all_credits(self):
        cfg, _, trace = fixture_inputs()
        cfg["horizon_us"] = 50000
        trace = trace[:4]
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_run(Path(directory), cfg, trace)
        dynamic = policy_result(manifest, "dynamic_two_level")
        self.assertEqual(dynamic["terminal_counts"]["timed_out"], 4)
        self.assertEqual(dynamic["cancelled_prefetch_reservation_bytes"], 64 * MIB)
        for ledger in dynamic["device_ledgers"]:
            self.assertEqual(ledger["activation_used_final"], 0)
            self.assertEqual(ledger["prefetch_active_final"], 0)
            self.assertTrue(all(row["finish_us"] <= 50000 for row in ledger["compute_intervals"]))
        self.assertTrue(all(row["finish_us"] <= 50000 for row in dynamic["server_intervals"]))

    def test_unfinished_phone_dispatch_is_not_counted_as_useful_transfer(self):
        cfg, _, trace = fixture_inputs()
        cfg["horizon_us"] = 70000
        trace = trace[:4]
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_run(Path(directory), cfg, trace)
        dynamic = policy_result(manifest, "dynamic_two_level")
        self.assertEqual(dynamic["transfer_bytes_completed"], 64 * MIB)
        self.assertEqual(dynamic["useful_transfer_bytes"], 0)
        self.assertEqual(dynamic["wasted_transfer_bytes"], 64 * MIB)

    def test_stale_slow_action_is_rejected_without_mutation(self):
        cfg, catalog, trace = fixture_inputs()
        replay = Simulator("dynamic_two_level", cfg, catalog, trace)
        for request in replay.requests[:4]:
            replay._handle_arrival(request)
        actions = plan_slow(replay._snapshot(), catalog, cfg["scheduler"])
        replicate = next(action for action in actions if action["action"] == "REPLICATE")
        replay.devices["op12"].boot_epoch += 1
        replay._apply_residency_intent(replicate)
        self.assertEqual(replay.slow_actions[-1]["reason"], "stale_boot_epoch")
        self.assertFalse(replay.devices["op12"].replicas)

    def test_stale_residency_event_cannot_advance_state(self):
        cfg, catalog, trace = fixture_inputs()
        replay = Simulator("dynamic_two_level", cfg, catalog, trace)
        for request in replay.requests[:4]:
            replay._handle_arrival(request)
        replicate = next(
            action
            for action in plan_slow(replay._snapshot(), catalog, cfg["scheduler"])
            if action["action"] == "REPLICATE"
        )
        replay._apply_residency_intent(replicate)
        replica = replay.devices["op12"].replicas["island_a_head"]
        replay._handle_residency_transition(
            {
                "boot_epoch": replica.boot_epoch,
                "device_id": "op12",
                "expected_status_seq": replica.status_seq + 1,
                "generation": replica.generation,
                "island_id": replica.island_id,
                "target": "VERIFYING",
            }
        )
        self.assertEqual(replica.status, "RECEIVING")
        self.assertEqual(replay.stale_residency_events, 1)

    def test_corrupt_ready_identity_is_never_dispatched(self):
        cfg, catalog, trace = fixture_inputs()
        replay = Simulator("dynamic_two_level", cfg, catalog, trace)
        replay.devices["op15"].replicas["island_a_head"].weight_identity = "stale-alias"
        with self.assertRaisesRegex(AssertionError, "identity mismatch"):
            replay.run()
        self.assertFalse(
            any(action["action"] == "DISPATCH_PHONE" for action in replay.fast_actions)
        )

    def test_no_new_work_starts_exactly_at_horizon(self):
        cfg, _, trace = fixture_inputs()
        cfg["horizon_us"] = 10
        one = copy.deepcopy(trace[:1])
        one[0]["arrival_us"] = 10
        one[0]["deadline_us"] = 10
        with tempfile.TemporaryDirectory() as directory:
            at_arrival = write_run(Path(directory), cfg, one)
        for result in at_arrival["results"]:
            self.assertEqual(result["fast_actions"], [])
            self.assertEqual(result["terminal_counts"]["timed_out"], 1)

        cfg, _, trace = fixture_inputs()
        cfg["horizon_us"] = 60000
        with tempfile.TemporaryDirectory() as directory:
            at_ready = write_run(Path(directory), cfg, trace[:4])
        dynamic = policy_result(at_ready, "dynamic_two_level")
        op12_dispatch = [
            action
            for action in dynamic["fast_actions"]
            if action["action"] == "DISPATCH_PHONE" and action["device_id"] == "op12"
        ]
        self.assertEqual(op12_dispatch, [])


if __name__ == "__main__":
    unittest.main()
