"""Whole-model control must survive removal of USB ADB without relaxing identity."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research_dev.scheduler import RuntimeLinkState
from research_dev.scheduler.adapters.android_llama_server import (
    AndroidLlamaServerProcessConfiguration, AndroidLlamaServerProcessLauncher,
    ncm_control_script_sha256,
)
from research_dev.scheduler.adapters.catalog_materialization import (
    _whole_model_request_link,
)
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError
from research_dev.scheduler.adapters.heterogeneous_rig import HeterogeneousPhysicalRig
from research_dev.scheduler.adapters.snapshot import EndpointRuntimeSample
from research_dev.scheduler.adapters.probes import (
    parse_android_process_identity, PhoneRuntimeObservation, PhoneRuntimeProbe,
    probe_phone_runtime_with_adb_fallback,
)

try:
    from .test_phone_allocation_snapshot import allocation_output, identity_output
except ImportError:
    from test_phone_allocation_snapshot import allocation_output, identity_output


class AndroidNcmControlTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.configuration = AndroidLlamaServerProcessConfiguration(
            Path(sys.executable), "exact-phone", 5037, "/data/local/tmp/llama-server",
            "/data/local/tmp/lib", {"sha256:" + "a" * 64: "/data/local/tmp/model.gguf"},
            "/data/local/tmp/owned-state", "GPUOpenCL", Path(self.directory.name),
            control_transport="adb-ncm", ncm_adb_endpoint="192.168.42.1:5555",
        )
        self.launcher = AndroidLlamaServerProcessLauncher(self.configuration)
        self.boot_id = "12345678-1234-1234-1234-123456789abc"

    def test_ncm_configuration_is_explicit_and_fail_closed(self):
        for changes in ({"control_transport": "auto"}, {"ncm_adb_endpoint": None},
                        {"ncm_adb_endpoint": "127.0.0.1:5555"}, {"ncm_adb_endpoint": "8.8.8.8:5555"},
                        {"ncm_adb_endpoint": "192.168.42.1:0"}, {"control_transport": "adb-usb"}):
            with self.subTest(changes=changes), self.assertRaises(PhysicalAdapterError):
                replace(self.configuration, **changes)

    def test_usb_mode_remains_default_and_does_not_bootstrap(self):
        launcher = AndroidLlamaServerProcessLauncher(replace(
            self.configuration, control_transport="adb-usb", ncm_adb_endpoint=None))
        with patch("subprocess.run") as run:
            launcher.prepare_ncm_control(None)
            launcher.connect_ncm_control()
            launcher.close_control()
            run.assert_not_called()
        self.assertEqual(launcher.control_serial, "exact-phone")

    def test_ncm_allocation_uses_same_process_fence_without_usb_fallback(self):
        expected = parse_android_process_identity(identity_output(), 123)
        with patch("subprocess.run", return_value=SimpleNamespace(stdout=allocation_output())) as run:
            value = self.launcher.probe_process_allocation(expected, "/state/server.pid")
        self.assertEqual(value["allocated_bytes"], 9766 * 1024)
        self.assertEqual(run.call_args.args[0][3:5], ["-s", "192.168.42.1:5555"])
        self.assertEqual(run.call_args.kwargs["timeout"], 3)
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("adb-ncm", 3)) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                self.launcher.probe_process_allocation(expected, "/state/server.pid")
            self.assertEqual(run.call_count, 1)

    def test_connect_requires_bootstrap_and_exact_device_and_boot(self):
        with patch("subprocess.run") as run, self.assertRaises(PhysicalAdapterError):
            self.launcher.connect_ncm_control()
        run.assert_not_called()
        self.launcher._control_boot_id = self.boot_id
        for identity in ("wrong-phone\n" + self.boot_id, "exact-phone\nwrong-boot"):
            with patch("subprocess.run", side_effect=[
                SimpleNamespace(stdout="connected to 192.168.42.1:5555\n"),
                SimpleNamespace(stdout=identity),
            ]), self.assertRaisesRegex(PhysicalAdapterError, "identity differs"):
                self.launcher.connect_ncm_control()

    def test_unbootstrapped_ncm_does_not_advertise_a_loadable_endpoint(self):
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.RLock()
        rig._live_executors = {}
        rig._active_large = {}
        rig._android_phone_launcher = self.launcher
        rig._direct_phone_session = SimpleNamespace(active=False)
        endpoints = (
            SimpleNamespace(executor_id="phone", adapter_parameters={
                "android_control_transport": "adb-ncm",
                "request_transport_generation": "adb-ncm-token-http-v1",
            }),
            SimpleNamespace(executor_id="cpu", adapter_parameters={}),
        )
        rig.configuration = SimpleNamespace(catalog=SimpleNamespace(
            executors=endpoints, composite_executors=(), transitions=endpoints,
            placement_profile=SimpleNamespace(links=(
                SimpleNamespace(link_id="ncm", transport_generation="adb-ncm-token-http-v1"),
                SimpleNamespace(link_id="htp", transport_generation="functionfs-v3"),
            )),
        ))
        rig._runtime_monitor = SimpleNamespace(snapshot=lambda _: SimpleNamespace(
            stale=False, error=None, value=EndpointRuntimeSample("unavailable", "unavailable", 0),
        ))
        for armed, active in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(armed=armed, active=active), patch("subprocess.run") as run:
                self.launcher._control_boot_id = self.boot_id if armed else None
                self.launcher._control_root = "/owned/control" if armed else None
                rig._direct_phone_session.active = active
                samples = rig._executor_samples()
                self.assertEqual(samples["phone"].transition_available, armed and active)
                self.assertTrue(samples["cpu"].transition_available)
                links = rig._control_link_states({
                    key: RuntimeLinkState(key, True, 100, 0) for key in ("ncm", "htp")
                })
                self.assertEqual(links["ncm"].ready, armed and active)
                self.assertTrue(links["htp"].ready)
                run.assert_not_called()

    def test_connection_refusal_is_not_admission_and_does_not_fall_back(self):
        self.launcher._control_boot_id = self.boot_id
        with patch("subprocess.run", return_value=SimpleNamespace(stdout="failed to connect: Connection refused")) as run:
            with self.assertRaisesRegex(PhysicalAdapterError, "connection failed"):
                self.launcher.connect_ncm_control()
            self.assertEqual(run.call_count, 1)
            self.assertFalse(self.launcher._control_connected_here)

    def test_existing_connection_is_not_disconnected_on_cleanup(self):
        self.launcher._control_boot_id = self.boot_id
        with patch("subprocess.run", side_effect=[
            SimpleNamespace(stdout="already connected to 192.168.42.1:5555\n"),
            SimpleNamespace(stdout="exact-phone\n" + self.boot_id + "\n"),
        ]):
            self.launcher.connect_ncm_control()
        with patch("subprocess.run") as run:
            self.launcher.close_control()
            run.assert_not_called()

    def test_launch_rejects_mode_endpoint_and_script_mismatch_before_mutation(self):
        parameters = {"android_control_transport": "adb-ncm", "android_control_endpoint": self.launcher.control_serial,
                      "android_control_script_sha256": ncm_control_script_sha256(),
                      "request_transport_generation": "adb-ncm-token-http-v1"}
        for key in parameters:
            with self.subTest(key=key), patch("subprocess.run") as run:
                with self.assertRaisesRegex(PhysicalAdapterError, "transport differs"):
                    self.launcher.launch(SimpleNamespace(adapter_parameters={**parameters, key: "different"}),
                                         None, label="test", control_check=lambda: None)
                run.assert_not_called()

    def test_ncm_link_does_not_inherit_usb_measurement_qualification(self):
        link = _whole_model_request_link(
            link_id="ncm", source_device="cpu", target_device="phone", fixed_latency_us=1,
            bandwidth_bytes_per_s=100, maximum_payload_bytes=100, evidence=("sha256:" + "a" * 64,),
            transport_profile_id="ncm", qualification_identity_sha256="sha256:" + "b" * 64,
            transport_generation="adb-ncm-token-http-v1",
        )
        self.assertEqual(link.status, "estimated")
        self.assertIsNone(link.qualification_identity_sha256)

    def test_transport_cannot_be_removed_under_a_live_ncm_endpoint(self):
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.RLock()
        rig._current_direct_phone = Mock()
        rig._live_executors = {"whole": SimpleNamespace(parameters={"android_control_transport": "adb-ncm"})}
        rig._finish_direct_phone = Mock()
        with self.assertRaisesRegex(PhysicalAdapterError, "still leased"):
            rig._stop_phone_session()
        rig._finish_direct_phone.assert_not_called()
        rig._stop_phone_session(terminate_phone_session=False)
        rig._finish_direct_phone.assert_not_called()

    def test_cleanup_stops_ncm_consumers_before_functionfs(self):
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.RLock()
        rig._live_executors = {"desktop": SimpleNamespace(parameters={}),
                               "whole": SimpleNamespace(parameters={"android_control_transport": "adb-ncm"})}
        rig._phone_executor_id = None
        rig._stop_executor = Mock()
        rig._stop_dynamic_executors()
        self.assertEqual([call.args[0] for call in rig._stop_executor.call_args_list], ["whole", "desktop"])

    def test_health_fallback_uses_verified_control_without_changing_observation(self):
        missing = PhoneRuntimeObservation("http", None, 100, "MISSING", "absent", None)
        fresh = PhoneRuntimeObservation("adb:192.168.42.1:5555", 100, 101, "VALID", None,
                                       PhoneRuntimeProbe(1000, 500, 30000, 900000, True, True))
        provider = Mock(return_value="192.168.42.1:5555")
        with patch("research_dev.scheduler.adapters.probes.probe_phone_runtime", return_value=missing), \
             patch("research_dev.scheduler.adapters.probes.probe_android_phone_runtime", return_value=fresh) as probe:
            result = probe_phone_runtime_with_adb_fallback("http", "usb", 5037, diagnostic=True,
                                                          fallback_serial_provider=provider)
        probe.assert_called_once_with("192.168.42.1:5555", 5037, diagnostic=True)
        self.assertEqual(result.value, fresh.value)
        self.assertEqual(result.captured_at_ns, 100)
        self.assertEqual(len(result.attempts), 2)

    def test_unverified_control_keeps_health_unknown_and_never_uses_usb(self):
        missing = PhoneRuntimeObservation("http", None, 100, "MISSING", "absent", None)
        with patch("research_dev.scheduler.adapters.probes.probe_phone_runtime", return_value=missing), \
             patch("research_dev.scheduler.adapters.probes.probe_android_phone_runtime") as probe:
            result = probe_phone_runtime_with_adb_fallback("http", "usb", 5037, diagnostic=True,
                fallback_serial_provider=Mock(side_effect=PhysicalAdapterError("boot identity differs")))
        probe.assert_not_called()
        self.assertIsNone(result.value)
        self.assertEqual(result.validity, "UNAVAILABLE")
        self.assertEqual(result.failure_reason, "boot identity differs")

    def test_remote_stop_fences_process_and_requires_transport_success(self):
        expected = parse_android_process_identity(identity_output(), 123)
        with patch.object(self.launcher, "_su", side_effect=[
            SimpleNamespace(stdout=identity_output()), SimpleNamespace(stdout=""), SimpleNamespace(stdout="S42_EXITED"),
        ]) as su:
            self.launcher.stop_remote(123, "/state/server.pid", expected=expected)
        self.assertEqual(su.call_args_list[1].args[0], "kill -TERM 123")
        for value in (identity_output(start=901), identity_output(executable="/different/server")):
            with patch.object(self.launcher, "_su", return_value=SimpleNamespace(stdout=value)) as su:
                with self.assertRaisesRegex(PhysicalAdapterError, "identity differs"):
                    self.launcher.stop_remote(123, "/state/server.pid", expected=expected)
                self.assertEqual(su.call_count, 1)
        with patch.object(self.launcher, "_su", side_effect=subprocess.CalledProcessError(1, "adb")) as su:
            with self.assertRaises(subprocess.CalledProcessError):
                self.launcher.stop_remote(123, "/state/server.pid", expected=expected)
            self.assertEqual(su.call_count, 1)

    def test_disconnected_owned_control_cleanup_is_idempotent(self):
        self.launcher._control_connected_here = True
        with patch("subprocess.run", return_value=SimpleNamespace(stdout="List of devices attached\n")) as run:
            self.launcher.close_control()
            self.launcher.close_control()
        self.assertEqual(run.call_count, 1)

    def test_failed_ncm_stop_retains_the_authoritative_live_endpoint(self):
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.RLock()
        rig._phone_executor_id = None
        state = SimpleNamespace(parameters={"android_control_transport": "adb-ncm"}, owns_phone_session=False,
                                server=SimpleNamespace(stop=Mock(side_effect=PhysicalAdapterError("transport lost"))))
        rig._live_executors = {"whole": state}
        with self.assertRaisesRegex(PhysicalAdapterError, "transport lost"):
            rig._stop_executor("whole")
        self.assertIs(rig._live_executors["whole"], state)

    def test_failed_shutdown_does_not_disconnect_a_retained_endpoint(self):
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._stop_current = Mock(side_effect=PhysicalAdapterError("stop failed"))
        rig._android_phone_launcher = Mock()
        rig._live_executors = {"whole": SimpleNamespace(parameters={"android_control_transport": "adb-ncm"})}
        rig._resident_server = None
        rig._runtime_monitor_started = rig._phone_sampler_started = rig._server_sampler_started = False
        with self.assertRaisesRegex(PhysicalAdapterError, "still leased"):
            rig.close()
        rig._android_phone_launcher.close_control.assert_not_called()


if __name__ == "__main__":
    unittest.main()
