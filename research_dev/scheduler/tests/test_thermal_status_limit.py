#!/usr/bin/env python3
"""Opt-in per-device ``maximum_thermal_status`` (Android thermal status NONE 0 .. SHUTDOWN 6).

The OnePlus 15 sits at Android status 1 (LIGHT, UX not impacted) for minutes under a sustained
run; the platform rule ``status == 0`` excluded it for a whole model window. A campaign
``phone_thermal_status_limits`` row raises the limit for one phone. Default 0 keeps the platform
rule and every serialized artifact (capability, executor state, snapshot, campaign manifest,
THERMAL_DEFERRAL row) byte-identical; only a device with a non-default limit carries its raw
status into the runtime snapshot, where feasibility compares it against the limit.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from research_dev.scheduler._internal.route_generation import _static_executor_identity
from research_dev.scheduler._internal.runtime_capabilities import (
    RuntimeCapabilityError,
    RuntimeExecutorState,
)
from research_dev.scheduler._internal.types import canonical_json, canonical_sha256
from research_dev.scheduler.adapters import (
    DeviceRuntimeTelemetry,
    EndpointRuntimeSample,
    PhoneRuntimeProbe,
    PhysicalAdapterError,
    RuntimeSnapshotBuilder,
    probe_android_phone_runtime,
    probe_phone_runtime,
)
from research_dev.scheduler.adapters.preflight import _check_model_phone_state
from research_dev.scheduler.campaigns.burstgpt import arguments, runner
from research_dev.scheduler.campaigns.burstgpt.common import UnifiedTraceError
from research_dev.scheduler.config import (
    CAMPAIGN_MANIFEST_SCHEMA,
    CampaignManifest,
    PhoneThermalStatusLimitConfiguration,
    SchedulerConfigurationError,
)

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import test_automated_runtime as automated  # noqa: E402

HELPER = "helper-c"
HELPER_EXECUTOR = "executor:" + HELPER
PROBES = "research_dev.scheduler.adapters.probes."
# not an int under this repo's typed checks: too high, negative, text, float, bool
INVALID_LIMITS = (7, -1, "1", 1.0, True)


def catalog_with_limit(limit: int):
    """The synthetic three-device catalog with ``maximum_thermal_status`` on the phone only."""
    base = automated.catalog()
    return replace(base, executors=tuple(
        replace(row, maximum_thermal_status=limit) if row.device_id == HELPER else row
        for row in base.executors
    ))


def phone_at(snapshot, *, status=None, qualified=None, temperature_millic=40_000):
    helper = snapshot.executors[HELPER_EXECUTOR]
    return replace(snapshot, executors={**snapshot.executors, HELPER_EXECUTOR: replace(
        helper, temperature_millic=temperature_millic, thermal_qualified=qualified,
        thermal_status=status)})


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class CapabilityContractTests(unittest.TestCase):
    def test_default_capability_serializes_without_the_key_and_round_trips(self):
        capability = automated.capability(HELPER, "phone")
        self.assertEqual(capability.maximum_thermal_status, 0)
        self.assertNotIn("maximum_thermal_status", capability.to_json())
        self.assertEqual(type(capability).from_json(capability.to_json()), capability)
        limited = replace(capability, maximum_thermal_status=1)
        self.assertEqual(limited.to_json()["maximum_thermal_status"], 1)
        self.assertEqual(type(capability).from_json(limited.to_json()), limited)

    def test_capability_rejects_invalid_limits(self):
        capability = automated.capability(HELPER, "phone")
        for value in INVALID_LIMITS:
            with self.subTest(value=value), self.assertRaises(RuntimeCapabilityError):
                replace(capability, maximum_thermal_status=value)
        self.assertEqual(replace(capability, maximum_thermal_status=6).maximum_thermal_status, 6)

    def test_overlays_combine_with_min_like_the_temperature_limit(self):
        capability = automated.capability(HELPER, "phone")
        loose, strict = (replace(capability, maximum_thermal_status=value) for value in (2, 1))
        self.assertEqual(loose.with_runtime_overlay(strict).maximum_thermal_status, 1)
        self.assertEqual(strict.with_runtime_overlay(loose).maximum_thermal_status, 1)
        self.assertEqual(loose.with_runtime_overlay(capability).maximum_thermal_status, 0)

    def test_executor_state_carries_the_raw_status_only_when_given(self):
        state = automated.executor_state(HELPER_EXECUTOR)
        self.assertIsNone(state.thermal_status)
        self.assertNotIn("thermal_status", state.to_json())
        observed = replace(state, thermal_qualified=False, thermal_status=1)
        self.assertEqual(observed.to_json()["thermal_status"], 1)
        self.assertEqual(RuntimeExecutorState.from_json(observed.to_json()), observed)
        for value in (-1, "1", 1.0, True):
            with self.subTest(value=value), self.assertRaises(RuntimeCapabilityError):
                replace(state, thermal_status=value)

    def test_qualification_under_a_limit_prefers_the_raw_status(self):
        state = automated.executor_state(HELPER_EXECUTOR)
        # no raw status: the probe verdict stands (None stays None)
        self.assertIsNone(state.thermal_qualified_under(1))
        self.assertIs(replace(state, thermal_qualified=False).thermal_qualified_under(3), False)
        light = replace(state, thermal_qualified=False, thermal_status=1)
        self.assertIs(light.thermal_qualified_under(0), False)
        self.assertIs(light.thermal_qualified_under(1), True)
        self.assertIs(replace(light, thermal_status=2).thermal_qualified_under(1), False)
        self.assertIs(replace(light, thermal_status=0).thermal_qualified_under(0), True)


class ConfigurationTests(unittest.TestCase):
    ROW = {"schema": CAMPAIGN_MANIFEST_SCHEMA, "campaign_id": "thermal-limit-test",
           "rig_manifest_path": "rig.json", "models_manifest_path": "models.json",
           "evidence_manifest_path": "evidence.json", "selection_mode": "energy-aware",
           "maximum_latency_ppm": 1_100_000, "energy_attribution_kind": "diagnostic",
           "trace": {"large_requests_path": "large.json", "overlay_requests_path": "overlay.json",
                     "trace_manifest_path": "trace.json"}}
    LIMIT = {"phone_device_id": "op15-phone", "maximum_thermal_status": 1}

    def test_limit_configuration_validates_fail_closed(self):
        limit = PhoneThermalStatusLimitConfiguration("op15-phone", 1)
        self.assertEqual(PhoneThermalStatusLimitConfiguration.from_json(limit.to_json()), limit)
        self.assertEqual(limit.to_json(), self.LIMIT)
        for value in INVALID_LIMITS:
            with self.subTest(value=value), self.assertRaises(SchedulerConfigurationError):
                PhoneThermalStatusLimitConfiguration("op15-phone", value)
            with self.subTest(value=value, via="json"), self.assertRaises(SchedulerConfigurationError):
                PhoneThermalStatusLimitConfiguration.from_json({**self.LIMIT, "maximum_thermal_status": value})
        for invalid in ({"phone_device_id": "op15-phone"}, {**self.LIMIT, "extra": 1},
                        {**self.LIMIT, "phone_device_id": ""}, [self.LIMIT], "op15-phone"):
            with self.subTest(invalid=invalid), self.assertRaises(SchedulerConfigurationError):
                PhoneThermalStatusLimitConfiguration.from_json(invalid)

    def test_campaign_manifest_omits_the_key_by_default_and_round_trips_when_set(self):
        plain = CampaignManifest.from_json(dict(self.ROW), Path("/inputs"))
        self.assertEqual(plain.phone_thermal_status_limits, ())
        self.assertNotIn("phone_thermal_status_limits", plain.to_json())
        configured = CampaignManifest.from_json(
            {**self.ROW, "phone_thermal_status_limits": [self.LIMIT]}, Path("/inputs"))
        self.assertEqual(configured.phone_thermal_status_limits,
                         (PhoneThermalStatusLimitConfiguration("op15-phone", 1),))
        self.assertEqual(configured.to_json()["phone_thermal_status_limits"], [self.LIMIT])
        self.assertEqual(CampaignManifest.from_json(configured.to_json(), Path("/inputs")), configured)
        self.assertEqual({**configured.to_json(), "phone_thermal_status_limits": None}.keys()
                         - plain.to_json().keys(), {"phone_thermal_status_limits"})
        for invalid in ([self.LIMIT, self.LIMIT], {"op15-phone": 1}, [{"maximum_thermal_status": 1}],
                        [{**self.LIMIT, "maximum_thermal_status": 7}]):
            with self.subTest(invalid=invalid), self.assertRaises(SchedulerConfigurationError):
                CampaignManifest.from_json({**self.ROW, "phone_thermal_status_limits": invalid}, Path("/inputs"))

    def test_runner_argument_exists_and_applies_the_limit_to_the_loaded_catalog(self):
        parser = arguments._build_parser()
        self.assertIn("phone_thermal_status_limits_json", {action.dest for action in parser._actions})
        catalog = automated.catalog()
        self.assertIs(runner._apply_phone_thermal_status_limits(catalog, None), catalog)
        applied = runner._apply_phone_thermal_status_limits(
            catalog, canonical([{"phone_device_id": HELPER, "maximum_thermal_status": 1}]))
        self.assertEqual(applied.executor_by_device[HELPER].maximum_thermal_status, 1)
        self.assertEqual({row.device_id: row.maximum_thermal_status for row in applied.executors
                          if row.device_id != HELPER}, {"host-a": 0, "accelerator-b": 0})
        self.assertEqual(replace(applied, executors=catalog.executors), catalog)
        for text, error in (
            (canonical([{"phone_device_id": "host-a", "maximum_thermal_status": 1}]), UnifiedTraceError),
            (canonical([{"phone_device_id": "absent", "maximum_thermal_status": 1}]), UnifiedTraceError),
            (canonical({"phone_device_id": HELPER, "maximum_thermal_status": 1}), UnifiedTraceError),
            (canonical([{"phone_device_id": HELPER, "maximum_thermal_status": 1}] * 2), UnifiedTraceError),
            (canonical([{"phone_device_id": HELPER, "maximum_thermal_status": 7}]), SchedulerConfigurationError),
        ):
            with self.subTest(text=text), self.assertRaises(error):
                runner._apply_phone_thermal_status_limits(catalog, text)


class ProbeTests(unittest.TestCase):
    HTTP = {"android_thermal_status": 1, "battery_level_pct": 24, "captured_epoch_s": 123,
            "charging": True, "mem_available_kib": 11_919_632, "mem_total_kib": 15_475_004,
            "schema": "s42-op15-live-snapshot-v1", "task_server_alive": True,
            "temperature_max_millic": 72_000}

    def http_probe(self, status, **options):
        with mock.patch(PROBES + "http_json", return_value={**self.HTTP, "android_thermal_status": status}), \
                mock.patch(PROBES + "time.time", return_value=124):
            return probe_phone_runtime("http://192.0.2.1:18383", **options)

    def test_http_probe_carries_the_raw_status_and_keeps_the_platform_rule(self):
        light = self.http_probe(1)
        self.assertEqual((light.thermal_status, light.thermal_qualified), (1, False))
        self.assertIs(light.thermal_qualified_under(0), False)
        self.assertIs(light.thermal_qualified_under(1), True)
        cool = self.http_probe(0)
        self.assertEqual((cool.thermal_status, cool.thermal_qualified), (0, True))
        # an unknown status (-1) is still a probe failure (MISSING), never a qualification
        self.assertIsNone(self.http_probe(-1))
        self.assertEqual(self.http_probe(-1, diagnostic=True).to_json()["validity"], "MISSING")

    @staticmethod
    def dumpsys(status: int, sensor_statuses: tuple[int, ...]):
        sensors = "".join(
            "\tTemperature{mValue=" + str(30 + index) + ".0, mType=0, mName=CPU" + str(index)
            + ", mStatus=" + str(sensor_status) + "}\n"
            for index, sensor_status in enumerate(sensor_statuses)
        )
        return SimpleNamespace(returncode=0, stderr="", stdout=(
            "MemTotal:       15475004 kB\nMemAvailable:   11919632 kB\n"
            "Thermal Status: " + str(status) + "\n"
            "Current temperatures from HAL:\n" + sensors
            + "Current cooling devices from HAL:\n"
            "S42_BATTERY_LEVEL=24\nS42_BATTERY_CHARGING=1\nS42_TASK_SERVER=1234\n"))

    def adb_probe(self, status, sensor_statuses):
        with mock.patch(PROBES + "subprocess.run", return_value=self.dumpsys(status, sensor_statuses)):
            return probe_android_phone_runtime("SYNTHETIC123", 5037)

    def test_adb_probe_reports_the_coarsest_of_platform_and_sensor_statuses(self):
        cool = self.adb_probe(0, (0, 0))
        self.assertEqual((cool.thermal_status, cool.thermal_qualified), (0, True))
        light = self.adb_probe(1, (0, 1))
        self.assertEqual((light.thermal_status, light.thermal_qualified), (1, False))
        self.assertIs(light.thermal_qualified_under(1), True)
        # a sensor throttled harder than the platform status follows the same limit
        sensor = self.adb_probe(0, (0, 2))
        self.assertEqual((sensor.thermal_status, sensor.thermal_qualified), (2, False))
        self.assertIs(sensor.thermal_qualified_under(1), False)
        self.assertIs(sensor.thermal_qualified_under(2), True)

    def test_probe_contract_validates_the_raw_status(self):
        probe = PhoneRuntimeProbe(2_000_000_000, 1_000_000_000, 35_000, 900_000, False, True)
        self.assertIsNone(probe.thermal_status)
        self.assertIs(probe.thermal_qualified_under(6), False)
        for value in (-1, "1", 1.0, True):
            with self.subTest(value=value), self.assertRaises(PhysicalAdapterError):
                replace(probe, thermal_status=value)


class RuntimeTests(unittest.TestCase):
    setUp = automated.AutomatedRuntimeTests.setUp
    tearDown = automated.AutomatedRuntimeTests.tearDown
    scheduler_and_manifest = automated.AutomatedRuntimeTests.scheduler_and_manifest

    def helper_rows(self, scheduler, manifest, snapshot, at_us: int, index: int):
        candidates = scheduler.generate_automated_candidates(
            automated.request(f"thermal-{index}", arrival_us=at_us), manifest.model_id, snapshot,
            observed_at_us=at_us)
        rows = [row for row in candidates.candidates if HELPER in row.device_ids]
        self.assertTrue(rows)
        return rows

    def assert_helper_limited(self, rows, limited: bool):
        self.assertEqual({"THERMAL_LIMIT" in row.rejection_reasons for row in rows}, {limited})

    def test_default_policy_excludes_status_one_with_the_legacy_deferral_row(self):
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = automated.runtime_snapshot(manifest)
        # today's probe reports Android status 1 as thermal_qualified False; the raw status is
        # withheld from the state under the default policy
        light = phone_at(snapshot, qualified=False)
        self.assert_helper_limited(self.helper_rows(scheduler, manifest, light, 2_000, 0), True)
        self.assert_helper_limited(
            self.helper_rows(scheduler, manifest, phone_at(snapshot, qualified=True), 3_000, 1), False)
        events = [dict(row) for row in scheduler.thermal_deferral_events()]
        self.assertEqual(events[0], {
            "at_us": 2_000, "device_id": HELPER, "executor_id": HELPER_EXECUTOR,
            "kind": "THERMAL_DEFERRAL", "maximum_temperature_millic": 90_000,
            "observed_temperature_millic": 40_000, "thermal_qualified": False})
        self.assertEqual([row["kind"] for row in events], ["THERMAL_DEFERRAL", "THERMAL_DEFERRAL_CLEARED"])
        self.assertTrue(all("maximum_thermal_status" not in row for row in events))

    def test_limit_one_qualifies_status_one_and_excludes_status_two(self):
        scheduler, manifest = self.scheduler_and_manifest(catalog_with_limit(1))
        snapshot = automated.runtime_snapshot(manifest)
        light = phone_at(snapshot, status=1, qualified=False)
        moderate = phone_at(snapshot, status=2, qualified=False)
        self.assert_helper_limited(self.helper_rows(scheduler, manifest, light, 1_000, 0), False)
        self.assert_helper_limited(self.helper_rows(scheduler, manifest, moderate, 2_000, 1), True)
        self.assert_helper_limited(self.helper_rows(scheduler, manifest, light, 3_000, 2), False)
        events = [dict(row) for row in scheduler.thermal_deferral_events()]
        self.assertEqual(events[0], {
            "at_us": 2_000, "device_id": HELPER, "executor_id": HELPER_EXECUTOR,
            "kind": "THERMAL_DEFERRAL", "maximum_temperature_millic": 90_000,
            "maximum_thermal_status": 1, "observed_temperature_millic": 40_000,
            "observed_thermal_status": 2, "thermal_qualified": False})
        self.assertEqual((events[1]["kind"], events[1]["onset_at_us"], events[1]["observed_thermal_status"]),
                         ("THERMAL_DEFERRAL_CLEARED", 2_000, 1))

    def test_limit_without_a_raw_status_keeps_the_temperature_rule(self):
        scheduler, manifest = self.scheduler_and_manifest(catalog_with_limit(1))
        snapshot = automated.runtime_snapshot(manifest)
        self.assert_helper_limited(
            self.helper_rows(scheduler, manifest, phone_at(snapshot, temperature_millic=60_000), 1_000, 0), False)
        self.assert_helper_limited(
            self.helper_rows(scheduler, manifest, phone_at(snapshot, temperature_millic=95_000), 2_000, 1), True)
        self.assert_helper_limited(
            self.helper_rows(scheduler, manifest, phone_at(snapshot, qualified=False), 3_000, 2), True)

    def build(self, catalog, manifest, telemetry: DeviceRuntimeTelemetry):
        memory = automated.runtime_snapshot(manifest).memory
        return RuntimeSnapshotBuilder(catalog, manifest).build(
            snapshot_id="thermal", captured_at_us=0, valid_until_us=10_000_000, memory=memory,
            endpoint_samples={row.endpoint: EndpointRuntimeSample("healthy", "live", 1)
                              for row in catalog.executors},
            device_telemetry={HELPER: telemetry},
        )

    def test_snapshot_builder_withholds_the_raw_status_unless_the_device_opts_in(self):
        _, manifest = self.scheduler_and_manifest()
        light = DeviceRuntimeTelemetry(temperature_millic=40_000, battery_ppm=900_000,
                                       thermal_qualified=False, thermal_status=1)
        legacy = replace(light, thermal_status=None)
        default = automated.catalog()
        with_status = self.build(default, manifest, light).to_json()
        self.assertEqual(canonical(with_status), canonical(self.build(default, manifest, legacy).to_json()))
        self.assertTrue(all("thermal_status" not in row for row in with_status["executors"]))
        limited = self.build(catalog_with_limit(1), manifest, light)
        self.assertEqual(limited.executors[HELPER_EXECUTOR].thermal_status, 1)
        self.assertIs(limited.executors[HELPER_EXECUTOR].thermal_qualified_under(1), True)
        for value in (-1, "1", 1.0, True):
            with self.subTest(value=value), self.assertRaises(PhysicalAdapterError):
                replace(light, thermal_status=value)

    def test_preflight_thermal_check_reports_the_status_only_under_a_policy(self):
        _, manifest = self.scheduler_and_manifest()
        snapshot = automated.runtime_snapshot(manifest)

        def thermal_check(catalog, observed):
            checks = []
            _check_model_phone_state(checks, catalog, SimpleNamespace(manifest=manifest, snapshot=observed))
            (check,) = [row for row in checks if row.check_id.startswith("phone-thermal")]
            return check.status, check.detail

        self.assertEqual(thermal_check(automated.catalog(), phone_at(snapshot, qualified=False)),
                         ("PASS", "phone temperature millic is 40000"))
        limited = catalog_with_limit(1)
        self.assertEqual(thermal_check(limited, phone_at(snapshot, status=1, qualified=False)),
                         ("PASS", "phone temperature millic is 40000, thermal status 1 (limit 1)"))
        self.assertEqual(thermal_check(limited, phone_at(snapshot, status=2, qualified=False)),
                         ("WARN", "phone temperature millic is 40000, thermal status 2 (limit 1)"))


class CanonicalIdentityTests(unittest.TestCase):
    """Catalog, snapshot and route identities hash dataclass fields generically; the two opt-in
    fields stay out of every identity while at their default (the arrival decision-log golden and
    every prematerialized catalog digest are unchanged) and enter it once a policy is set."""

    def test_default_fields_are_absent_from_canonical_identities(self):
        capability = automated.capability(HELPER, "phone")
        state = automated.executor_state(HELPER_EXECUTOR)
        for value, key in ((capability, "maximum_thermal_status"), (state, "thermal_status")):
            with self.subTest(key=key):
                self.assertNotIn(key, json.loads(canonical_json(value)))
        self.assertNotIn("maximum_thermal_status", _static_executor_identity(capability))
        catalog = automated.catalog()
        self.assertEqual(canonical_sha256(catalog), canonical_sha256(replace(catalog)))

    def test_a_set_policy_changes_the_capability_identity(self):
        capability = automated.capability(HELPER, "phone")
        limited = replace(capability, maximum_thermal_status=1)
        self.assertEqual(json.loads(canonical_json(limited))["maximum_thermal_status"], 1)
        self.assertEqual(_static_executor_identity(limited)["maximum_thermal_status"], 1)
        self.assertNotEqual(canonical_sha256(capability), canonical_sha256(limited))
        self.assertNotEqual(canonical_sha256(automated.catalog()), canonical_sha256(catalog_with_limit(1)))
        state = replace(automated.executor_state(HELPER_EXECUTOR), thermal_status=1)
        self.assertEqual(json.loads(canonical_json(state))["thermal_status"], 1)


if __name__ == "__main__":
    unittest.main()
