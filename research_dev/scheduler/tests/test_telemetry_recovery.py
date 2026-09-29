"""Telemetry absence defers new residency without invalidating READY work."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import threading
import subprocess
import time
import unittest
from unittest.mock import Mock, patch

from research_dev.scheduler._internal.lifecycle import PhoneTelemetryUnavailable
from research_dev.scheduler._internal.capacity import DeviceMemoryCapacity
from research_dev.scheduler.adapters.probes import (
    PhoneRuntimeObservation, PhoneRuntimeProbe, probe_phone_runtime,
    probe_phone_runtime_with_adb_fallback, probe_phone_power, http_json,
)
from research_dev.scheduler.adapters.offline_phone_residency import (
    CanonicalOfflinePhoneResidencyPreloader,
)
from research_dev.scheduler.adapters.heterogeneous_rig import HeterogeneousPhysicalRig
from research_dev.scheduler.campaigns.burstgpt import offline_residency_gate as gate

from research_dev.scheduler.tests import test_offline_phone_residency as offline_fixture


PROBES = "research_dev.scheduler.adapters.probes."


def observation(snapshot, *, validity="VALID", age_us=0):
    return replace(snapshot, telemetry_observations={
        "op15-phone": {
            "source": "synthetic-adb", "sample_timestamp_ns": 123,
            "age_us": age_us, "maximum_age_us": 5_000_000,
            "validity": validity, "valid": validity == "VALID",
            "failure_reason": None if validity == "VALID" else "injected " + validity,
        },
    })


class PhoneObservationTests(unittest.TestCase):
    def test_functionfs_hal_producer_matches_supported_adb_probe(self):
        script = (Path(__file__).parents[1] / "adapters/native/direct_phone_ffn_session.sh").read_text()
        function = script.split("probe_android_thermal_state() {", 1)[1].split("\n}\n", 1)[0]
        command = "dumpsys() { cat; }\nprobe_android_thermal_state() {" + function + "\n}\nprobe_android_thermal_state"
        for status, sensor_status, expected in ((0, 0, "26700 0"), (0, 2, "26700 2"), (3, 0, "26700 3")):
            raw = (
                f"Thermal Status: {status}\nCached temperatures:\n"
                "Temperature{mValue=98.0, mType=0, mName=stale, mStatus=4}\n"
                "Current temperatures from HAL:\n"
                "Temperature{mValue=72.0, mType=8, mName=virtual, mStatus=4}\n"
                f"Temperature{{mValue=26.7, mType=0, mName=CPU0, mStatus={sensor_status}}}\n"
                "Current cooling devices from HAL:\n"
            )
            with self.subTest(status=status, sensor_status=sensor_status):
                result = subprocess.run(["sh", "-c", command], input=raw, text=True,
                                        capture_output=True, check=True)
                self.assertEqual(result.stdout.strip(), expected)
        result = subprocess.run(["sh", "-c", command], input="service unavailable",
                                text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), "0 -1")

    def test_http_hal_probe_preserves_source_and_throttling(self):
        with patch(PROBES + "http_json", return_value={
            **self.payload(), "temperature_source": "android-hal",
            "android_thermal_status": 2,
        }), patch(PROBES + "time.time", return_value=101):
            observed = probe_phone_runtime("http://192.0.2.1:18383", diagnostic=True)
        self.assertEqual(observed.to_json()["temperature_source"], "android-hal")
        self.assertFalse(observed.value.thermal_qualified)

    def rig(self, available_bytes=2_000_000_000):
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.RLock()
        rig._direct_phone_receipts = []
        rig.configuration = SimpleNamespace(catalog=SimpleNamespace(executor_by_device={
            "phone": SimpleNamespace(maximum_temperature_millic=80_000,
                                      minimum_battery_ppm=50_000,
                                      memory_resource_id="phone-ram"),
        }))
        sample = PhoneRuntimeProbe(10_000_000_000, available_bytes,
                                   35_000, 800_000, True, True)
        rig.phone_runtime_observation = Mock(return_value=(sample, {"valid": True}))
        command = SimpleNamespace(
            ticket_id="new-load", operator_plan={},
            transition=SimpleNamespace(
                changed_phone_session_ids=("selected",), evictions=(),
                phone_shards=(SimpleNamespace(session_id="selected", resident_bytes=1_000_000_000),),
            ),
        )
        return rig, command, SimpleNamespace(phone="phone", direct_phone_reused=False)

    def test_preload_revalidates_live_memory_before_mutation(self):
        rig, command, state = self.rig(0)
        with self.assertRaisesRegex(Exception, "live memory capacity is insufficient"):
            rig._validate_phone_transition_observation(command, state, lambda: None)
        self.assertEqual(rig._direct_phone_receipts, [])

    def test_preload_waits_for_fresh_observation_without_resetting_sessions(self):
        rig, command, state = self.rig()
        valid = rig.phone_runtime_observation.return_value
        rig.phone_runtime_observation.side_effect = [(None, {"failure_reason": "STALE"}), valid]
        check = Mock()
        rig._validate_phone_transition_observation(command, state, check)
        self.assertEqual(check.call_count, 1)
        self.assertEqual(len(rig._direct_phone_receipts), 1)

    def test_interruption_reference_wait_does_not_force_a_policy(self):
        scheduler = Mock()
        scheduler.request_helper_events.return_value = [
            {"request_id": "request", "kind": "FRACTION_APPLIED"},
        ]
        scheduler.adaptive_decode_snapshot.side_effect = [{"state": "PROBING"}, {"state": "EXPLOITING"}]
        with patch.object(gate, "_request_completed", return_value=False):
            result = gate._wait_interruption_reference_phase(scheduler, "request")
        self.assertEqual(result["state"], "EXPLOITING")
        self.assertEqual(scheduler.method_calls, [
            unittest.mock.call.request_helper_events(),
            unittest.mock.call.adaptive_decode_snapshot("request"),
            unittest.mock.call.request_helper_events(),
            unittest.mock.call.adaptive_decode_snapshot("request"),
        ])

    def test_attachment_event_does_not_imply_adaptive_session_has_started(self):
        scheduler = Mock()
        scheduler.request_helper_events.side_effect = [
            [{"request_id": "request", "kind": "ATTACHED"}],
            [{"request_id": "request", "kind": "FRACTION_APPLIED"}],
        ]
        scheduler.adaptive_decode_snapshot.return_value = {"state": "EXPLOITING"}
        with patch.object(gate, "_request_completed", return_value=False):
            result = gate._wait_interruption_reference_phase(scheduler, "request")
        self.assertEqual(result["state"], "EXPLOITING")
        scheduler.adaptive_decode_snapshot.assert_called_once_with("request")

    def payload(self):
        return {
            "schema": "s42-op15-live-snapshot-v1",
            "captured_epoch_s": 100, "mem_available_kib": 10,
            "mem_total_kib": 20, "temperature_max_millic": 35_000,
            "battery_level_pct": 80, "android_thermal_status": 0,
            "task_server_alive": True,
        }

    def test_missing_stale_timeout_and_malformed_are_distinct(self):
        cases = (
            ({}, None, "MISSING"),
            ({**self.payload(), "captured_epoch_s": 90}, None, "STALE"),
            ({**self.payload(), "mem_available_kib": "bad"}, None, "MALFORMED"),
            (None, TimeoutError("socket exceeded 1 s"), "TIMED_OUT"),
        )
        for payload, error, expected in cases:
            with self.subTest(expected=expected), patch(
                PROBES + "http_json", return_value=payload, side_effect=error,
            ), patch(PROBES + "time.time", return_value=101):
                result = probe_phone_runtime("http://192.0.2.1:18383", diagnostic=True)
            self.assertEqual(result.validity, expected)
            self.assertIsNone(result.to_json()["available_bytes"])
            self.assertTrue(result.failure_reason)
            self.assertIn("snapshot.json", result.source)

    def test_zero_free_memory_is_a_valid_measurement(self):
        with patch(PROBES + "http_json", return_value={
            **self.payload(), "mem_available_kib": 0,
        }), patch(PROBES + "time.time", return_value=101):
            result = probe_phone_runtime("http://192.0.2.1:18383", diagnostic=True)
        self.assertTrue(result.to_json()["valid"])
        self.assertEqual(result.value.available_bytes, 0)

    def test_http_phone_clock_bounds_age_without_host_clock_agreement(self):
        def response(_endpoint, _path, _timeout, *, response_headers):
            response_headers.update(Date="Thu, 01 Jan 1970 00:01:41 GMT", Age=None)
            return self.payload()

        ticks = iter((10_000_000_000, 10_200_000_000))
        with patch(PROBES + "http_json", side_effect=response), patch(
            PROBES + "time.time", return_value=90,
        ) as wall, patch(PROBES + "time.monotonic_ns", side_effect=lambda: next(ticks, 10_200_000_000)):
            result = probe_phone_runtime("http://192.0.2.1:18383", diagnostic=True)
            detail = result.to_json()
        wall.assert_not_called()
        self.assertTrue(detail["valid"])
        self.assertEqual(detail["freshness_basis"], "phone-http-date")
        self.assertEqual(detail["age_us"], 2_200_000)
        self.assertFalse(result.to_json(now_ns=14_000_000_000)["valid"])

    def test_phone_clock_rejects_expired_future_slow_or_malformed_samples(self):
        cases = (
            ("Thu, 01 Jan 1970 00:01:50 GMT", None, 200_000_000, "STALE"),
            ("Thu, 01 Jan 1970 00:01:30 GMT", None, 200_000_000, "STALE"),
            ("Thu, 01 Jan 1970 00:01:41 GMT", None, 5_000_000_000, "STALE"),
            ("Thu, 01 Jan 1970 00:01:41 GMT", "6", 200_000_000, "STALE"),
            ("Thu, 01 Jan 1970 00:01:41 GMT", "bad", 200_000_000, "MALFORMED"),
            ("not a date", None, 200_000_000, "MALFORMED"),
        )
        for date, age, elapsed, validity in cases:
            def response(_endpoint, _path, _timeout, *, response_headers):
                response_headers.update(Date=date, Age=age)
                return self.payload()
            ticks = iter((10_000_000_000, 10_000_000_000 + elapsed))
            with self.subTest(date=date, age=age, elapsed=elapsed), patch(
                PROBES + "http_json", side_effect=response,
            ), patch(PROBES + "time.monotonic_ns", side_effect=lambda: next(ticks, 20_000_000_000)):
                result = probe_phone_runtime("http://192.0.2.1:18383", diagnostic=True)
            self.assertEqual(result.validity, validity)
            self.assertIsNone(result.value)

    def test_phone_power_uses_the_same_response_clock_age_bound(self):
        payload = {
            "captured_epoch_s": 100, "uptime_s": 50.0,
            "schema": "s42-op15-live-power-v1",
            "battery_charge_counter_uah": 4_100_000,
            "battery_current_ma": 250, "battery_voltage_uv": 4_000_000,
            "usb_current_ua": 400_000, "usb_voltage_uv": 5_000_000,
        }
        for date, valid in (("Thu, 01 Jan 1970 00:01:41 GMT", True),
                            ("Thu, 01 Jan 1970 00:01:50 GMT", False)):
            def response(_endpoint, _path, _timeout, *, response_headers):
                response_headers.update(Date=date)
                return payload
            with self.subTest(date=date), patch(PROBES + "http_json", side_effect=response), patch(
                PROBES + "time.time", side_effect=AssertionError("host wall clock used"),
            ), patch(PROBES + "time.monotonic_ns", side_effect=(10_000_000_000, 10_200_000_000)):
                result = probe_phone_power("http://192.0.2.1:18383")
            self.assertEqual(result is not None, valid)

    def test_http_json_preserves_response_clock_headers(self):
        response = Mock(status=200)
        response.read.return_value = b'{"sequence":1}'
        response.getheader.side_effect = lambda name: {"Date": "date", "Age": None}[name]
        connection = Mock()
        connection.getresponse.return_value = response
        with patch(PROBES + "http.client.HTTPConnection", return_value=connection):
            headers = {}
            self.assertEqual(http_json("http://192.0.2.1:18383", "/snapshot.json", 1,
                                       response_headers=headers), {"sequence": 1})
        self.assertEqual(headers, {"Date": "date", "Age": None})
        connection.close.assert_called_once()

    def test_adb_fallback_preserves_http_failure_and_its_own_timestamp(self):
        sample = PhoneRuntimeProbe(20, 10, 35_000, 800_000, True, True)
        adb = PhoneRuntimeObservation("adb:phone:5037", time.monotonic_ns(),
                                     time.monotonic_ns(), "VALID", None, sample)
        with patch(PROBES + "http_json", side_effect=TimeoutError("HTTP timeout")), patch(
            PROBES + "probe_android_phone_runtime", return_value=adb,
        ):
            result = probe_phone_runtime_with_adb_fallback(
                "http://192.0.2.1:18383", "phone", 5037, diagnostic=True,
            )
        self.assertEqual(result.value, sample)
        self.assertEqual(result.captured_at_ns, adb.captured_at_ns)
        self.assertEqual(result.attempts[0]["validity"], "TIMED_OUT")
        self.assertTrue(result.to_json()["valid"])

    def test_persistent_failure_has_both_reasons(self):
        with patch(PROBES + "http_json", side_effect=TimeoutError("HTTP timeout")), patch(
            PROBES + "subprocess.run", side_effect=subprocess.TimeoutExpired("adb", 3),
        ):
            result = probe_phone_runtime_with_adb_fallback(
                "http://192.0.2.1:18383", "phone", 5037, diagnostic=True,
            )
        self.assertFalse(result.to_json()["valid"])
        self.assertEqual([row["validity"] for row in result.attempts], ["TIMED_OUT"] * 2)
        self.assertIn("ADB shell", result.failure_reason)


class ResidencyTelemetryRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        offline_fixture.OfflinePhoneResidencyTests.setUpClass()

    def runtime(self):
        scheduler, requests, snapshot = offline_fixture.OfflinePhoneResidencyTests.replay_runtime()
        return scheduler, requests, observation(snapshot)

    def test_unavailable_planning_is_deferred_not_unchanged_and_resumes_once(self):
        scheduler, requests, snapshot = self.runtime()
        backend = offline_fixture._OfflinePhoneBackend(snapshot)
        preloader = CanonicalOfflinePhoneResidencyPreloader(
            scheduler, backend, epoch_ns=time.monotonic_ns(),
            snapshot_provider=backend.snapshot,
        )
        plan = scheduler.plan_offline_phone_residency(
            requests, snapshot=snapshot, observed_at_us=snapshot.captured_at_us,
        )
        materialized, _ = scheduler._offline_phone_materialization_snapshot(plan, snapshot)
        self.assertEqual(materialized.telemetry_observations, snapshot.telemetry_observations)
        ready = preloader.preload(plan, lambda _: object(), initial_snapshot=snapshot,
                                  observed_at_us=snapshot.captured_at_us).plan
        hot = backend.snapshot(ready.current_stage, ready.finished_at_us)
        states_before = scheduler.phone_residency_session_states()
        physical_before = dict(backend.phone_shards)
        historical = ready.to_json()
        replay_fixture = offline_fixture.replay_fixture
        case = next(row for row in replay_fixture.ReplayDeterminismTests.fixture["cases"]
                    if row["case_id"] == "session_cow_gate_v3")
        replay = replay_fixture.ReplayDeterminismTests(methodName="runTest")
        other = {}
        for row in case["requests"]:
            if row["model_id"] not in requests:
                other.setdefault(row["model_id"], []).append(replay._request(row))
        for status in ("MISSING", "STALE", "TIMED_OUT", "MALFORMED"):
            with self.subTest(status=status), self.assertRaises(PhoneTelemetryUnavailable):
                scheduler.plan_offline_phone_residency(
                    other, snapshot=observation(hot, validity=status),
                    observed_at_us=hot.captured_at_us,
                )
            self.assertEqual(scheduler.phone_residency_session_states(), states_before)
            self.assertEqual(backend.phone_shards, physical_before)
        refreshed = Mock()
        replacement = preloader.plan_with_observation_refresh(
            scheduler, other, snapshot=observation(hot, validity="MISSING"),
            snapshot_provider=lambda _: hot, refresh_observation=refreshed,
            epoch_ns=time.monotonic_ns(), observation_timeout_s=1,
        )
        self.assertEqual(refreshed.call_count, 1)
        selected = replacement.current_stage.selected_session_id
        deferred = scheduler.begin_offline_phone_residency_stage(
            replacement.plan_id, snapshot=observation(hot, age_us=6_000_000),
            observed_at_us=hot.captured_at_us,
        )
        self.assertEqual(deferred["status"], "DEFERRED")
        self.assertEqual(backend.phone_shards, physical_before)
        done = preloader.execute_next_stage(replacement.plan_id, object(),
                                           snapshot=hot, observed_at_us=hot.captured_at_us)
        self.assertEqual(done.plan.state, "READY")
        self.assertEqual(len(backend.commands), 4)
        for session_id, shard in physical_before.items():
            if session_id != selected:
                self.assertEqual(backend.phone_shards[session_id], shard)
        self.assertEqual(ready.to_json(), historical)
        self.assertEqual(backend.phone_shards[selected].session_generation, 2)

    def test_persistent_missing_does_not_change_layout(self):
        scheduler, requests, snapshot = self.runtime()
        missing = observation(snapshot, validity="MISSING")
        refresh = Mock()
        with self.assertRaises(PhoneTelemetryUnavailable):
            CanonicalOfflinePhoneResidencyPreloader.plan_with_observation_refresh(
                scheduler, requests, snapshot=missing, snapshot_provider=lambda _: missing,
                refresh_observation=refresh, epoch_ns=time.monotonic_ns(),
                observation_timeout_s=0.01,
            )
        self.assertEqual(scheduler.phone_residency_session_states(), ())
        self.assertGreater(refresh.call_count, 0)

    def test_measured_insufficient_memory_is_not_telemetry_deferral(self):
        scheduler, requests, snapshot = self.runtime()
        plan = scheduler.plan_offline_phone_residency(
            requests, snapshot=snapshot, observed_at_us=snapshot.captured_at_us,
        )
        pool = snapshot.memory.capacities["op15-ram"]
        full = replace(snapshot, memory=replace(snapshot.memory, capacities={
            **snapshot.memory.capacities,
            "op15-ram": DeviceMemoryCapacity("op15-ram", pool.capacity_bytes,
                                             pool.capacity_bytes, 0),
        }))
        with self.assertRaisesRegex(Exception, "memory capacity is insufficient") as caught:
            scheduler.begin_offline_phone_residency_stage(
                plan.plan_id, snapshot=full, observed_at_us=full.captured_at_us,
            )
        self.assertNotIsInstance(caught.exception, PhoneTelemetryUnavailable)
        self.assertEqual(scheduler.offline_phone_residency_stage(plan.plan_id).state, "PROPOSED")


if __name__ == "__main__":
    unittest.main()
