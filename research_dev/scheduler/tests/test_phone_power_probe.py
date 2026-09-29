#!/usr/bin/env python3

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from research_dev.scheduler.adapters import (
    HostEnergySampler,
    HostMetricCallbacks,
    PhysicalAdapterError,
    PhoneRuntimeProbe,
    PhoneActivityIntervalTracker,
    PolledPhonePowerSampler,
    RaplNvmlPhoneEnergyMeter,
    probe_android_phone_power,
    probe_android_phone_runtime,
    probe_nvidia_process_memory_bytes,
    probe_phone_power,
    probe_phone_runtime,
    probe_phone_power_with_adb_fallback,
    probe_phone_runtime_with_adb_fallback,
)
from research_dev.scheduler import RuntimePhonePowerProfile


class PhonePowerProbeTests(unittest.TestCase):
    def test_session_temperature_probe_uses_shell_reads_for_all_zones(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "adapters/native/direct_phone_ffn_session.sh"
        ).read_text(encoding="ascii")
        probe = script.split("            temperature_max_millic=$(\n", 1)[1]
        probe = probe.split("            )\n", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            thermal = root / "thermal"
            thermal.mkdir()
            for index, (name, temperature) in enumerate((
                ("cpu", "45000"),
                ("dsp", "61000"),
                ("battery", "29000"),
                ("cpu-hw-trip-critical", "120000"),
                ("offline", "-273000"),
            )):
                zone = thermal / ("thermal_zone" + str(index))
                zone.mkdir()
                (zone / "type").write_text(name + "\n", encoding="ascii")
                (zone / "temp").write_text(
                    temperature + "\n", encoding="ascii"
                )
            commands = root / "bin"
            commands.mkdir()
            cat = commands / "cat"
            cat.write_text("#!/bin/sh\nexit 99\n", encoding="ascii")
            cat.chmod(0o755)
            completed = subprocess.run(
                ["/bin/sh", "-c", probe.replace(
                    "/sys/class/thermal/", str(thermal) + "/"
                )],
                env={**os.environ, "PATH": str(commands) + os.pathsep
                     + os.defpath},
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
        self.assertEqual(completed.stdout.strip(), "61000")

    def test_assumed_phone_power_counts_active_and_idle_time(self) -> None:
        tracker = PhoneActivityIntervalTracker()
        tracker.record("load", "endpoint_preparation", 2_000, 4_000)
        estimate = tracker.estimate(
            1_000,
            11_000,
            RuntimePhonePowerProfile.assumed_4p5w(
                device_id="phone-a", domain_id="phone-system"
            ),
            charging_state="charging",
        )

        self.assertEqual(estimate.active_time_ns, 2_000)
        self.assertEqual(estimate.idle_time_ns, 8_000)
        self.assertEqual(estimate.energy_uj, 16)
        self.assertEqual(estimate.evidence_kind, "ASSUMED_4P5W")

    def test_overlapping_htp_sessions_are_counted_once(self) -> None:
        tracker = PhoneActivityIntervalTracker()
        tracker.record("htp0", "htp_execution", 1_000, 5_000)
        tracker.record("htp1", "htp_execution", 2_000, 6_000)
        tracker.record("htp2", "htp_execution", 3_000, 7_000)

        estimate = tracker.estimate(
            0,
            8_000,
            RuntimePhonePowerProfile.assumed_4p5w(
                device_id="phone-a", domain_id="phone-system"
            ),
            charging_state="charging",
        )

        self.assertEqual(estimate.active_time_ns, 6_000)
        self.assertEqual(estimate.idle_time_ns, 2_000)
        self.assertEqual(estimate.energy_uj, 29)

    def test_charging_uses_assumed_power_without_usb_double_count(self) -> None:
        start_ns = 1_000_000_000
        end_ns = 3_000_000_000
        server_samples = tuple({
            "gpu": {"sample_t_ns": sample_ns},
            "rapl_package": {"sample_t_ns": sample_ns},
        } for sample_ns in (0, 4_000_000_000))
        phone = PolledPhonePowerSampler(lambda: None)
        phone.rows.extend((
            {
                "battery_discharge_power_mw": 1000,
                "charging": 1,
                "host_sample_t_ns": 0,
                "phone_uptime_ns": 1,
                "usb_input_power_mw": 2000,
            },
            {
                "battery_discharge_power_mw": 1000,
                "charging": 1,
                "host_sample_t_ns": 4_000_000_000,
                "phone_uptime_ns": 4_000_000_001,
                "usb_input_power_mw": 2000,
            },
        ))
        tracker = PhoneActivityIntervalTracker()
        tracker.record("htp0", "htp_execution", start_ns, end_ns)
        meter = RaplNvmlPhoneEnergyMeter(
            lambda: server_samples,
            lambda _rows, _start, _end: {
                "cpu_package_energy_j": 1.0,
                "gpu_board_energy_j": 2.0,
            },
            phone,
            energy_boundary_id="synthetic-fleet",
            phone_power_profile=RuntimePhonePowerProfile.assumed_4p5w(
                device_id="phone-a", domain_id="phone-system"
            ),
            phone_activity=tracker,
        )

        measured = meter.measure(start_ns, end_ns)

        self.assertEqual(
            measured.fleet_energy_uj_by_domain["phone-system"], 9_000_000
        )
        self.assertEqual(measured.attribution_kind, "diagnostic")
        self.assertIn("ASSUMED_4P5W", measured.measurement_evidence_ids)
        self.assertEqual(
            measured.estimation_metadata["phone_charging_state"],
            "charging",
        )
        self.assertEqual(
            measured.estimation_metadata["phone_active_time_ns"],
            2_000_000_000,
        )

    def test_host_sampler_recovers_after_transient_gpu_probe_error(self) -> None:
        calls = 0

        def gpu():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("transient NVML failure")
            return {"power_mw": 1000}

        sampler = HostEnergySampler(
            HostMetricCallbacks(
                gpu_snapshot=gpu,
                rapl_package_snapshot=lambda: {
                    "energy_uj": calls,
                    "max_energy_range_uj": 1000,
                    "sample_t_ns": time.monotonic_ns(),
                },
                system_memory=lambda: {"available_bytes": 1},
            ),
            interval_s=0.01,
        )
        sampler.start()
        deadline = time.monotonic() + 1
        while len(sampler.rows()) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        sampler.stop()

        diagnostics = sampler.diagnostics()
        self.assertGreaterEqual(len(sampler.rows()), 2)
        self.assertIsNone(diagnostics["fatal_error"])
        self.assertIsNone(diagnostics["last_probe_error"])
        self.assertEqual(
            diagnostics["events"][0]["source"], "gpu"
        )
        self.assertEqual(diagnostics["process_exitcode"], 0)

    def test_host_sampler_isolated_from_parent_python_work(self) -> None:
        sampler = HostEnergySampler(
            HostMetricCallbacks(
                gpu_snapshot=lambda: {"power_mw": 1000},
                rapl_package_snapshot=lambda: {
                    "energy_uj": 1,
                    "max_energy_range_uj": 1000,
                    "sample_t_ns": time.monotonic_ns(),
                },
                system_memory=lambda: {"available_bytes": 1},
            ),
            interval_s=0.01,
        )
        previous_interval = sys.getswitchinterval()
        try:
            sys.setswitchinterval(1)
            sampler.start()
            deadline = time.monotonic() + 1
            while len(sampler.rows()) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            initial = len(sampler.rows())
            busy_until = time.monotonic() + 0.2
            while time.monotonic() < busy_until:
                pass
        finally:
            sys.setswitchinterval(previous_interval)
            sampler.stop()

        self.assertGreaterEqual(len(sampler.rows()), initial + 5)

    def test_server_energy_rejects_unmeasured_interior_gap(self) -> None:
        start_ns = 1_000_000_000
        end_ns = 9_000_000_000
        server_samples = tuple({
            "gpu": {"sample_t_ns": sample_ns},
            "rapl_package": {"sample_t_ns": sample_ns},
        } for sample_ns in (0, 2_000_000_000, 10_000_000_000))
        phone = PolledPhonePowerSampler(lambda: None)
        phone.rows.extend((
            {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": 0,
                "phone_uptime_ns": 1,
                "usb_input_power_mw": 2000,
            },
            {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": 10_000_000_000,
                "phone_uptime_ns": 10_000_000_001,
                "usb_input_power_mw": 2000,
            },
        ))
        meter = RaplNvmlPhoneEnergyMeter(
            lambda: server_samples,
            lambda _rows, _start, _end: {
                "cpu_package_energy_j": 1.0,
                "gpu_board_energy_j": 2.0,
            },
            phone,
            energy_boundary_id="synthetic-fleet",
            maximum_gap_s=5,
        )

        with self.assertRaisesRegex(
            PhysicalAdapterError, "coverage contains a gap"
        ):
            meter.measure(start_ns, end_ns)

    def test_server_energy_uses_bounded_window_provider(self) -> None:
        start_ns = 1_000_000_000
        end_ns = 2_000_000_000
        server_samples = (
            {
                "gpu": {"sample_t_ns": 0},
                "rapl_package": {"sample_t_ns": 0},
            },
            {
                "gpu": {"sample_t_ns": 3_000_000_000},
                "rapl_package": {"sample_t_ns": 3_000_000_000},
            },
        )
        phone = PolledPhonePowerSampler(lambda: None)
        phone.rows.extend((
            {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": 0,
                "phone_uptime_ns": 1,
                "usb_input_power_mw": 2000,
            },
            {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": 3_000_000_000,
                "phone_uptime_ns": 3_000_000_001,
                "usb_input_power_mw": 2000,
            },
        ))
        calls = []

        def window(left_ns, right_ns):
            calls.append((left_ns, right_ns))
            return server_samples

        meter = RaplNvmlPhoneEnergyMeter(
            lambda: self.fail("full history must not be requested"),
            lambda _rows, _start, _end: {
                "cpu_package_energy_j": 1.0,
                "gpu_board_energy_j": 2.0,
            },
            phone,
            energy_boundary_id="synthetic-fleet",
            server_window_rows=window,
        )

        measured = meter.measure(start_ns, end_ns)

        self.assertEqual(calls, [(start_ns, end_ns)])
        self.assertEqual(
            sum(measured.fleet_energy_uj_by_domain.values()), 6_000_000
        )

    def test_server_energy_waits_through_transient_probe_stall(self) -> None:
        clock = [0.0]
        start_ns = 1_000_000_000
        end_ns = 2_000_000_000
        server_samples = (
            {
                "gpu": {"sample_t_ns": 0},
                "rapl_package": {"sample_t_ns": 0},
            },
            {
                "gpu": {"sample_t_ns": 3_000_000_000},
                "rapl_package": {"sample_t_ns": 3_000_000_000},
            },
        )
        phone = PolledPhonePowerSampler(lambda: None)
        phone.rows.extend((
            {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": 0,
                "phone_uptime_ns": 1,
                "usb_input_power_mw": 2000,
            },
            {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": 3_000_000_000,
                "phone_uptime_ns": 3_000_000_001,
                "usb_input_power_mw": 2000,
            },
        ))

        def rows():
            return server_samples if clock[0] >= 4.0 else ()

        meter = RaplNvmlPhoneEnergyMeter(
            rows,
            lambda _rows, _start, _end: {
                "cpu_package_energy_j": 1.0,
                "gpu_board_energy_j": 2.0,
            },
            phone,
            energy_boundary_id="synthetic-fleet",
        )
        with mock.patch(
                "research_dev.scheduler.adapters.energy.time.monotonic",
                side_effect=lambda: clock[0]), mock.patch(
                "research_dev.scheduler.adapters.energy.time.sleep",
                side_effect=lambda seconds: clock.__setitem__(
                    0, clock[0] + seconds
                )):
            measured = meter.measure(start_ns, end_ns)

        self.assertGreaterEqual(clock[0], 4.0)
        self.assertEqual(
            measured.fleet_energy_uj_by_domain["gpu-board"], 2_000_000
        )

    def test_adb_runtime_probe_reads_current_thermal_state(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout=(
                "MemTotal:       15475004 kB\n"
                "MemAvailable:   11919632 kB\n"
                "Thermal Status: 0\n"
                "Cached temperatures:\n"
                "\tTemperature{mValue=98.0, mType=0, mName=stale, mStatus=0}\n"
                "Current temperatures from HAL:\n"
                "\tTemperature{mValue=72.0, mType=8, mName=socd, mStatus=0}\n"
                "\tTemperature{mValue=26.7, mType=0, mName=CPU0, mStatus=0}\n"
                "Current cooling devices from HAL:\n"
                "S42_BATTERY_LEVEL=24\n"
                "S42_BATTERY_CHARGING=1\n"
                "S42_TASK_SERVER=1234\n"
            ),
            stderr="",
        )
        with mock.patch(
                "research_dev.scheduler.adapters.probes.subprocess.run",
                return_value=completed):
            observed = probe_android_phone_runtime("SYNTHETIC123", 5037)

        self.assertEqual(observed.capacity_bytes, 15475004 * 1024)
        self.assertEqual(observed.available_bytes, 11919632 * 1024)
        self.assertEqual(observed.temperature_millic, 26_700)
        self.assertEqual(observed.battery_ppm, 240_000)
        self.assertTrue(observed.thermal_qualified)
        self.assertTrue(observed.task_server_alive)
        self.assertTrue(observed.charging)

    def test_http_runtime_probe_reads_live_battery_level(self) -> None:
        value = {
            "android_thermal_status": 0,
            "battery_level_pct": 24,
            "captured_epoch_s": 123,
            "charging": True,
            "mem_available_kib": 11_919_632,
            "mem_total_kib": 15_475_004,
            "schema": "s42-op15-live-snapshot-v1",
            "task_server_alive": True,
            "temperature_max_millic": 72_000,
        }
        with mock.patch(
                "research_dev.scheduler.adapters.probes.http_json",
                return_value=value), mock.patch(
                "research_dev.scheduler.adapters.probes.time.time",
                return_value=124):
            observed = probe_phone_runtime("http://192.0.2.1:18383")

        self.assertEqual(
            observed,
            PhoneRuntimeProbe(
                capacity_bytes=15_475_004 * 1024,
                available_bytes=11_919_632 * 1024,
                temperature_millic=72_000,
                battery_ppm=240_000,
                thermal_qualified=True,
                task_server_alive=True,
                charging=True,
                thermal_status=0,
            ),
        )

    def test_nvidia_process_memory_probe_is_pid_scoped(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout="123, 1024\n456, 2048\n123, 512\n",
            stderr="",
        )
        with mock.patch(
                "research_dev.scheduler.adapters.probes.subprocess.run",
                return_value=completed) as run:
            observed = probe_nvidia_process_memory_bytes(123)

        self.assertEqual(observed, 1536 * 1024 * 1024)
        self.assertEqual(
            run.call_args.args[0],
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
        )

    def test_adb_probe_returns_timestamped_power_observation(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout="123.25\n400000\n5000000\n250\n4000000\n4100000\n",
            stderr="",
        )
        with mock.patch(
                "research_dev.scheduler.adapters.probes.subprocess.run",
                return_value=completed) as run, mock.patch(
                "research_dev.scheduler.adapters.probes.time.monotonic_ns",
                side_effect=(1_000_000_000, 1_020_000_000)):
            row = probe_android_phone_power("SYNTHETIC123", 5037)

        self.assertEqual(row["host_sample_t_ns"], 1_010_000_000)
        self.assertEqual(row["phone_uptime_ns"], 123_250_000_000)
        self.assertEqual(row["usb_input_power_mw"], 2000)
        self.assertEqual(row["battery_discharge_power_mw"], 1000)
        self.assertEqual(
            run.call_args.args[0][:6],
            ["adb", "-P", "5037", "-s", "SYNTHETIC123", "shell"],
        )

    def test_http_loss_falls_back_to_restored_adb(self) -> None:
        expected = {"host_sample_t_ns": 123, "phone_uptime_ns": 456}
        with mock.patch(
                "research_dev.scheduler.adapters.probes.probe_phone_power",
                return_value=None) as http_probe, mock.patch(
                "research_dev.scheduler.adapters.probes.probe_android_phone_power",
                return_value=expected) as adb_probe:
            observed = probe_phone_power_with_adb_fallback(
                "http://192.0.2.1:18383", "SYNTHETIC123", 5037
            )

        self.assertIs(observed, expected)
        http_probe.assert_called_once_with("http://192.0.2.1:18383")
        adb_probe.assert_called_once_with("SYNTHETIC123", 5037)

    def test_stale_http_power_observation_is_rejected(self) -> None:
        value = {
            "battery_charge_counter_uah": 4_100_000,
            "battery_current_ma": 250,
            "battery_voltage_uv": 4_000_000,
            "captured_epoch_s": 100,
            "schema": "s42-op15-live-power-v1",
            "uptime_s": 123.25,
            "usb_current_ua": 400_000,
            "usb_voltage_uv": 5_000_000,
        }
        with mock.patch(
                "research_dev.scheduler.adapters.probes.http_json",
                return_value=value), mock.patch(
                "research_dev.scheduler.adapters.probes.time.time",
                return_value=200):
            self.assertIsNone(
                probe_phone_power("http://192.0.2.1:18383")
            )

    def test_cached_http_recovery_does_not_retimestamp_older_adb_uptime(self):
        value = {
            "schema": "s42-op15-live-power-v1", "captured_epoch_s": 0,
            "uptime_s": 78_286.15, "battery_charge_counter_uah": 7_094_000,
            "battery_current_ma": 51, "battery_voltage_uv": 4_454_000,
            "usb_current_ua": 497_000, "usb_voltage_uv": 5_062_000,
        }

        def http(endpoint, path, timeout, *, response_headers):
            response_headers["Date"] = "Thu, 01 Jan 1970 00:00:00 GMT"
            return value

        with mock.patch("research_dev.scheduler.adapters.probes.http_json", side_effect=http), \
                mock.patch("research_dev.scheduler.adapters.probes.time.monotonic_ns",
                           side_effect=(9_869_764_090_154, 9_869_784_090_154)):
            cached = probe_phone_power("http://192.0.2.1:18383")
        previous = {**cached, "host_sample_t_ns": 9_869_491_160_274,
                    "phone_uptime_ns": 78_286_220_000_000}
        following = {**cached, "host_sample_t_ns": 9_871_000_000_000,
                     "phone_uptime_ns": 78_287_750_000_000}
        samples = iter((previous, cached, following))

        def probe():
            row = next(samples, None)
            if row is None:
                sampler.stop_event.set()
            return row

        sampler = PolledPhonePowerSampler(probe)
        with mock.patch.object(sampler.stop_event, "wait"):
            sampler._run()
        self.assertIsNone(sampler.error)
        self.assertLess(cached["host_sample_t_ns"], previous["host_sample_t_ns"])
        self.assertEqual(sampler.rows, [previous, following])
        self.assertEqual(sampler.reboot_boundaries_ns, [])
        self.assertEqual(sampler.charging_state_between(
            previous["host_sample_t_ns"], following["host_sample_t_ns"]), "charging")
        self.assertGreater(sampler.energy_between(
            previous["host_sample_t_ns"], following["host_sample_t_ns"])["usb_input_uj"], 0)

    def test_phone_sampler_recovers_after_transient_probe_error(self) -> None:
        calls = 0

        def probe():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("transport handoff")
            now = time.monotonic_ns()
            return {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": now,
                "phone_uptime_ns": now,
                "usb_input_power_mw": 2000,
            }

        sampler = PolledPhonePowerSampler(
            probe, interval_s=0.01, coverage_timeout_s=1,
            maximum_gap_s=1,
        )
        sampler.start()
        ready_ns = sampler.wait_until_ready()
        deadline = time.monotonic() + 1
        while len(sampler.rows) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        sampler.stop()

        self.assertGreaterEqual(len(sampler.rows), 2)
        self.assertGreaterEqual(sampler.rows[-1]["host_sample_t_ns"], ready_ns)
        self.assertIsNone(sampler.error)

    def test_phone_sampler_accepts_fresh_samples_after_reboot(self) -> None:
        uptimes = iter((100_000_000_000, 1_000_000_000, 2_000_000_000))

        def probe():
            try:
                uptime_ns = next(uptimes)
            except StopIteration:
                return None
            return {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": time.monotonic_ns(),
                "phone_uptime_ns": uptime_ns,
                "usb_input_power_mw": 2000,
            }

        sampler = PolledPhonePowerSampler(
            probe, interval_s=0.01, coverage_timeout_s=1,
            maximum_gap_s=1,
        )
        sampler.start()
        deadline = time.monotonic() + 1
        while len(sampler.rows) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        sampler.stop()

        self.assertEqual(
            [row["phone_uptime_ns"] for row in sampler.rows],
            [100_000_000_000, 1_000_000_000, 2_000_000_000],
        )
        self.assertGreater(
            sampler.rows[-1]["host_sample_t_ns"],
            sampler.rows[0]["host_sample_t_ns"],
        )
        self.assertEqual(len(sampler.reboot_boundaries_ns), 1)

    def test_phone_local_history_fills_transport_handoff_gap(self) -> None:
        history = tuple({
            "battery_discharge_power_mw": 1000,
            "phone_uptime_ns": uptime_ns,
            "usb_input_power_mw": 2000,
        } for uptime_ns in range(
            9_000_000_000, 13_000_000_001, 500_000_000
        ))
        sampler = PolledPhonePowerSampler(
            lambda: None,
            history_probe=lambda: history,
            maximum_gap_s=1,
        )
        sampler.rows.extend((
            {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": 10_000_000_000,
                "phone_uptime_ns": 9_000_000_000,
                "usb_input_power_mw": 2000,
            },
            {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": 14_000_000_000,
                "phone_uptime_ns": 13_000_000_000,
                "usb_input_power_mw": 2000,
            },
        ))

        energy = sampler.energy_between(
            10_100_000_000, 13_900_000_000
        )

        self.assertEqual(energy["battery_discharge_uj"], 3_800_000)
        self.assertEqual(energy["usb_input_uj"], 7_600_000)

    def test_phone_energy_rejects_reboot_boundary(self) -> None:
        sampler = PolledPhonePowerSampler(
            lambda: None, maximum_gap_s=2
        )
        sampler.rows.extend({
            "battery_discharge_power_mw": 1000,
            "host_sample_t_ns": sample_ns,
            "phone_uptime_ns": sample_ns,
            "usb_input_power_mw": 2000,
        } for sample_ns in (
            10_000_000_000, 11_000_000_000, 12_000_000_000
        ))
        sampler.reboot_boundaries_ns.append(11_000_000_000)

        with self.assertRaisesRegex(
            PhysicalAdapterError, "crosses a reboot"
        ):
            sampler.energy_between(10_100_000_000, 11_900_000_000)
        with self.assertRaisesRegex(PhysicalAdapterError, "crosses a reboot"):
            sampler.charging_state_between(10_100_000_000, 11_900_000_000)

    def test_phone_energy_bridges_continuous_usb_restore_gap(self) -> None:
        sampler = PolledPhonePowerSampler(
            lambda: None, maximum_gap_s=5,
            maximum_continuity_gap_s=10,
        )
        sampler.rows.extend((
            {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": 10_000_000_000,
                "phone_uptime_ns": 100_000_000_000,
                "usb_input_power_mw": 2000,
            },
            {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": 15_500_000_000,
                "phone_uptime_ns": 105_400_000_000,
                "usb_input_power_mw": 2000,
            },
        ))

        energy = sampler.energy_between(
            10_100_000_000, 15_400_000_000
        )

        self.assertEqual(energy["battery_discharge_uj"], 5_300_000)
        self.assertEqual(energy["usb_input_uj"], 10_600_000)
        self.assertEqual(
            [row["event"] for row in sampler.events],
            ["PHONE_POWER_CONTINUITY_BRIDGE"],
        )

    def test_phone_energy_bridges_long_uptime_verified_gap(self) -> None:
        sampler = PolledPhonePowerSampler(
            lambda: None, maximum_gap_s=5
        )
        sampler.rows.extend((
            {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": 10_000_000_000,
                "phone_uptime_ns": 100_000_000_000,
                "usb_input_power_mw": 2000,
            },
            {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": 32_668_000_000,
                "phone_uptime_ns": 122_600_000_000,
                "usb_input_power_mw": 2000,
            },
        ))

        energy = sampler.energy_between(
            10_100_000_000, 32_500_000_000
        )

        self.assertEqual(energy["battery_discharge_uj"], 22_400_000)
        self.assertEqual(energy["usb_input_uj"], 44_800_000)
        self.assertEqual(
            [row["event"] for row in sampler.events],
            ["PHONE_POWER_CONTINUITY_BRIDGE"],
        )

    def test_phone_energy_drops_isolated_clock_alignment_outlier(self) -> None:
        sampler = PolledPhonePowerSampler(
            lambda: None, maximum_gap_s=5
        )
        sampler.rows.extend((
            {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": 10_000_000_000,
                "phone_uptime_ns": 100_000_000_000,
                "usb_input_power_mw": 2000,
            },
            {
                "battery_discharge_power_mw": 4000,
                "host_sample_t_ns": 22_000_000_000,
                "phone_uptime_ns": 101_500_000_000,
                "usb_input_power_mw": 8000,
            },
            {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": 33_000_000_000,
                "phone_uptime_ns": 123_000_000_000,
                "usb_input_power_mw": 2000,
            },
        ))

        energy = sampler.energy_between(
            10_100_000_000, 32_900_000_000
        )

        self.assertEqual(energy["battery_discharge_uj"], 22_800_000)
        self.assertEqual(energy["usb_input_uj"], 45_600_000)
        self.assertEqual(
            [row["event"] for row in sampler.events],
            [
                "PHONE_POWER_CLOCK_OUTLIER_DROPPED",
                "PHONE_POWER_CONTINUITY_BRIDGE",
            ],
        )

    def test_phone_energy_rejects_unexplained_restore_gap(self) -> None:
        sampler = PolledPhonePowerSampler(
            lambda: None, maximum_gap_s=5,
            maximum_continuity_gap_s=10,
        )
        sampler.rows.extend((
            {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": 10_000_000_000,
                "phone_uptime_ns": 100_000_000_000,
                "usb_input_power_mw": 2000,
            },
            {
                "battery_discharge_power_mw": 1000,
                "host_sample_t_ns": 15_500_000_000,
                "phone_uptime_ns": 120_000_000_000,
                "usb_input_power_mw": 2000,
            },
        ))

        with self.assertRaisesRegex(
            PhysicalAdapterError, "coverage contains a gap"
        ):
            sampler.energy_between(10_100_000_000, 15_400_000_000)

    def test_runtime_http_loss_falls_back_to_restored_adb(self) -> None:
        expected = SimpleNamespace(capacity_bytes=1, available_bytes=1)
        with mock.patch(
                "research_dev.scheduler.adapters.probes.probe_phone_runtime",
                return_value=None), mock.patch(
                "research_dev.scheduler.adapters.probes.probe_android_phone_runtime",
                return_value=expected) as adb_probe:
            observed = probe_phone_runtime_with_adb_fallback(
                "http://192.0.2.1:18383", "SYNTHETIC123", 5037
            )

        self.assertIs(observed, expected)
        adb_probe.assert_called_once_with("SYNTHETIC123", 5037)

    def test_live_http_source_does_not_probe_adb(self) -> None:
        expected = {"host_sample_t_ns": 123, "phone_uptime_ns": 456}
        with mock.patch(
                "research_dev.scheduler.adapters.probes.probe_phone_power",
                return_value=expected), mock.patch(
                "research_dev.scheduler.adapters.probes.probe_android_phone_power",
                side_effect=AssertionError("unexpected ADB probe")):
            observed = probe_phone_power_with_adb_fallback(
                "http://192.0.2.1:18383", "SYNTHETIC123", 5037
            )

        self.assertIs(observed, expected)


class EnergyReadinessTests(unittest.TestCase):
    def setUp(self):
        self.clock = 0.0
        self.start_ns = 1_000_000_000
        self.end_ns = 3_000_000_000
        self.server_rows = [{
            "gpu": {"sample_t_ns": timestamp},
            "rapl_package": {"sample_t_ns": timestamp},
        } for timestamp in (0, 4_000_000_000)]
        self.phone = PolledPhonePowerSampler(
            lambda: None, coverage_timeout_s=0.04
        )
        self.tracker = PhoneActivityIntervalTracker()
        self.tracker.record(
            "execution", "htp_execution", self.start_ns, 2_000_000_000
        )
        self.profile = RuntimePhonePowerProfile.assumed_4p5w(
            device_id="phone-a", domain_id="phone-system"
        )
        for name, kwargs in (
            ("monotonic_ns", {"return_value": self.start_ns}),
            ("monotonic", {"side_effect": lambda: self.clock}),
            ("sleep", {"side_effect": self.advance}),
        ):
            patcher = mock.patch(
                "research_dev.scheduler.adapters.energy.time." + name,
                **kwargs,
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def advance(self, seconds):
        self.clock += seconds

    def meter(self, *, assumed=True, maximum_gap_s=5):
        return RaplNvmlPhoneEnergyMeter(
            lambda: self.server_rows,
            lambda _rows, _start, _end: {
                "cpu_package_energy_j": 1.0,
                "gpu_board_energy_j": 2.0,
            },
            self.phone,
            energy_boundary_id="synthetic-fleet",
            coverage_timeout_s=0.04,
            maximum_gap_s=maximum_gap_s,
            phone_power_profile=self.profile if assumed else None,
            phone_activity=self.tracker if assumed else None,
        )

    def phone_row(self, timestamp):
        return {
            "host_sample_t_ns": timestamp,
            "phone_uptime_ns": timestamp + 1,
            "battery_discharge_power_mw": 1000,
            "usb_input_power_mw": 2000,
        }

    def test_assumed_prepare_and_measure_without_phone_samples(self):
        meter = self.meter()
        with mock.patch.object(
                self.phone, "wait_until_ready",
                wraps=self.phone.wait_until_ready) as ready, mock.patch.object(
                self.phone, "energy_between",
                wraps=self.phone.energy_between) as energy:
            meter.prepare()
            measured = meter.measure(self.start_ns, self.end_ns)
        ready.assert_not_called()
        energy.assert_not_called()
        self.assertEqual(self.clock, 0.0)
        self.assertEqual(measured.fleet_energy_uj_by_domain, {
            "cpu-package": 1_000_000,
            "gpu-board": 2_000_000,
            "phone-system": 5_375_000,
        })
        self.assertEqual(measured.attribution_kind, "diagnostic")
        self.assertIn("ASSUMED_4P5W", measured.measurement_evidence_ids)
        self.assertEqual(
            measured.estimation_metadata["phone_charging_state"], "unknown"
        )
        self.assertEqual(
            measured.estimation_metadata["phone_active_time_ns"],
            1_000_000_000,
        )
        self.assertEqual(
            measured.estimation_metadata["phone_idle_time_ns"],
            1_000_000_000,
        )

    def test_assumed_prepare_does_not_wait_for_stale_phone_samples(self):
        self.phone.rows.append(self.phone_row(0))
        self.meter().prepare()
        self.assertEqual(self.clock, 0.0)

    def test_prepare_requires_fresh_gpu_and_cpu_samples_in_both_modes(self):
        for assumed in (False, True):
            for missing in ("gpu", "rapl_package", "both"):
                with self.subTest(assumed=assumed, missing=missing):
                    self.server_rows = [] if missing == "both" else [{
                        name: {"sample_t_ns": 0 if name == missing else self.start_ns}
                        for name in ("gpu", "rapl_package")
                    }]
                    with mock.patch.object(self.phone, "wait_until_ready") as ready:
                        with self.assertRaisesRegex(
                                PhysicalAdapterError, "server energy sampler is not ready"):
                            self.meter(assumed=assumed).prepare()
                    ready.assert_not_called()

    def test_measured_prepare_and_measure_still_require_phone_samples(self):
        for timestamps in ((), (0,)):
            with self.subTest(timestamps=timestamps):
                self.phone.rows[:] = [self.phone_row(t) for t in timestamps]
                meter = self.meter(assumed=False)
                with self.assertRaisesRegex(
                        PhysicalAdapterError, "phone power sampler is not ready"):
                    meter.prepare()
                with self.assertRaisesRegex(
                        PhysicalAdapterError, "phone power sample coverage is incomplete"):
                    meter.measure(self.start_ns, self.end_ns)

    def test_measured_prepare_and_measure_use_fresh_phone_samples(self):
        self.phone.rows.extend(self.phone_row(t) for t in (0, 4_000_000_000))
        meter = self.meter(assumed=False)
        with mock.patch.object(
                self.phone, "wait_until_ready",
                wraps=self.phone.wait_until_ready) as ready:
            meter.prepare()
        ready.assert_called_once_with(self.start_ns)
        measured = meter.measure(self.start_ns, self.end_ns)
        self.assertEqual(
            measured.fleet_energy_uj_by_domain["phone-system"], 6_000_000
        )
        self.assertIn(
            "physical:phone-usb-plus-battery-power", measured.measurement_evidence_ids
        )
        self.assertNotIn("ASSUMED_4P5W", measured.measurement_evidence_ids)

    def test_assumed_measure_preserves_server_coverage_checks(self):
        for rows, reason in (
            ([], "server energy sample coverage is incomplete"),
            (self.server_rows, "server energy sample coverage contains a gap"),
        ):
            with self.subTest(reason=reason):
                self.server_rows = rows
                with self.assertRaisesRegex(PhysicalAdapterError, reason):
                    self.meter(maximum_gap_s=1).measure(self.start_ns, self.end_ns)


if __name__ == "__main__":
    unittest.main()
