"""Android allocation probes reach admission through the physical snapshot path."""

from dataclasses import replace
import subprocess
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research_dev.scheduler.adapters.activity import RuntimeActivityTracker
from research_dev.scheduler.adapters.android_llama_server import (
    AndroidLlamaServerProcessLauncher, ManagedAndroidLlamaServer,
)
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError
from research_dev.scheduler.adapters.energy import PhoneActivityIntervalTracker
from research_dev.scheduler._internal.runtime_capabilities import RuntimeCapabilityError, RuntimePhonePowerProfile
from research_dev.scheduler.adapters.heterogeneous_rig import HeterogeneousPhysicalRig, _LiveExecutorResidency
from research_dev.scheduler.adapters.probes import (
    PhoneRuntimeProbe, parse_android_process_allocation, parse_android_process_identity,
    parse_android_process_memory_peak,
)
from research_dev.scheduler.adapters.snapshot import EndpointRuntimeSample, UnifiedRuntimeSnapshotBuilder
from research_dev.scheduler.adapters.ticket import (
    PhysicalExecutionCommand, PhysicalParticipantCommand, PhysicalTransitionCommand,
)

try:
    from . import test_phone_memory_cap as fixtures
except ImportError:
    import test_phone_memory_cap as fixtures


def identity_output(*, start=900, process_id=123, executable="/data/local/tmp/llama-server"):
    fields = ["S", *(["0"] * 18), str(start), *(["0"] * 5)]
    return ("12345678-1234-1234-1234-123456789abc\n"
            + str(process_id) + " (llama-server) " + " ".join(fields) + "\n"
            + executable + "\n" + str(process_id) + "\n")


def allocation_output(*, kilobytes=9766, after=None, memory=None):
    identity = identity_output()
    return "\nS42_PROCESS_MEMORY_V1\n".join((
        identity,
        ("** MEMINFO in pid 123 [llama-server] **\nApp Summary\n"
         + "TOTAL PSS: " + str(kilobytes) + " TOTAL RSS: 12000 TOTAL SWAP PSS: 0\n")
        if memory is None else memory,
        identity if after is None else after,
    ))


class AndroidAllocationProbeTests(unittest.TestCase):
    def peak_output(self, *, after=None, rss=100, peak=120, swap=0, gpu="200\n300\n0\n0\n0"):
        identity = identity_output()
        return "\nS42_PROCESS_PEAK_V1\n".join((
            identity, f"VmRSS:\t{rss} kB\nVmHWM:\t{peak} kB\nVmSwap:\t{swap} kB",
            gpu, identity if after is None else after,
        ))

    def test_peak_uses_lifetime_counters_without_claiming_pss_or_deduplication(self):
        value = parse_android_process_memory_peak(
            self.peak_output(), parse_android_process_identity(identity_output(), 123),
            captured_at_ns=100, finished_at_ns=200,
        )
        self.assertEqual(value["accounted_peak_bytes"], 120 * 1024 + 300)
        self.assertFalse(value["has_unbounded_accounting"])
        self.assertNotIn("allocated_bytes", value)
        for output in (self.peak_output(swap=1), self.peak_output(gpu="200\n300\n0\n0\n1")):
            self.assertTrue(parse_android_process_memory_peak(
                output, parse_android_process_identity(identity_output(), 123),
                captured_at_ns=100, finished_at_ns=200,
            )["has_unbounded_accounting"])

    def test_peak_missing_counters_and_lifetime_changes_fail_closed(self):
        for output in (self.peak_output(after=identity_output(start=901)), self.peak_output(peak=99),
                       self.peak_output(gpu="301\n300\n0\n0\n0"), self.peak_output(gpu="0\n0"),
                       self.peak_output().replace("VmHWM:", "Missing:")):
            with self.subTest(output=output), self.assertRaises(PhysicalAdapterError):
                parse_android_process_memory_peak(
                    output, parse_android_process_identity(identity_output(), 123),
                    captured_at_ns=100, finished_at_ns=200,
                )

    def test_process_lifetime_and_measured_pss_are_preserved(self):
        identity = parse_android_process_identity(identity_output(), 123)
        output = allocation_output()
        value = parse_android_process_allocation(output, identity, captured_at_ns=100, finished_at_ns=200)
        self.assertEqual(value["allocated_bytes"], 9766 * 1024)
        self.assertEqual(value["process_identity"]["start_ticks"], 900)
        self.assertEqual(value["captured_at_ns"], 100)
        self.assertTrue(value["raw_sha256"].startswith("sha256:"))

    def test_lifetime_change_missing_total_and_malformed_observations_fail_closed(self):
        identity = parse_android_process_identity(identity_output(), 123)
        for output in (
            allocation_output(after=identity_output(start=901)),
            allocation_output(after=identity_output().replace("123456789abc", "123456789abd")),
            allocation_output(after=identity_output(process_id=124)),
            allocation_output(after=identity_output(executable="/different/server")),
            allocation_output(memory="** MEMINFO in pid 124 [llama-server] **\nTOTAL PSS: 10000"),
            allocation_output(memory="** MEMINFO in pid 123 [llama-server] **\nTOTAL RSS: 10000"),
            allocation_output(kilobytes=0), "", allocation_output() + "\nS42_PROCESS_MEMORY_V1\n",
        ):
            with self.subTest(output=output):
                with self.assertRaises(PhysicalAdapterError):
                    parse_android_process_allocation(output, identity, captured_at_ns=100, finished_at_ns=200)

    def test_native_table_total_is_pss_not_rss(self):
        value = parse_android_process_allocation(
            allocation_output(memory="** MEMINFO in pid 123 [llama-server] **\n"
                              "Pss Private Rss\n TOTAL 9766 9000 12000\n"),
            parse_android_process_identity(identity_output(), 123), captured_at_ns=100, finished_at_ns=200,
        )
        self.assertEqual(value["allocated_bytes"], 9766 * 1024)


class PhoneAllocationSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WholePhoneMemoryReuseTests("runTest")
        self.fixture.setUp()
        self.model, self.catalog = self.fixture.model, self.fixture.catalog
        self.plan = self.fixture.whole_candidate().plan.to_json()
        self.launcher = object.__new__(AndroidLlamaServerProcessLauncher)
        self.launcher._su = Mock(return_value=SimpleNamespace(stdout=allocation_output()))
        managed = SimpleNamespace(process=SimpleNamespace(poll=lambda: None))
        self.server = ManagedAndroidLlamaServer(
            managed, self.launcher, 18382, "/state/whole.pid", 123,
            process_identity=parse_android_process_identity(identity_output(), 123),
            artifact_sha256=self.model.artifact_sha256, launch_parameters=self.plan["adapter_parameters"],
            endpoint="http://127.0.0.1:18382",
        )
        self.state = _LiveExecutorResidency(
            "executor:helper-c", self.server.endpoint, self.server, self.model,
            self.plan["adapter_parameters"], self.plan, 5, ("helper-c",), (), (),
        )
        self.rig = object.__new__(HeterogeneousPhysicalRig)
        rig = self.rig
        rig._lock = threading.RLock()
        rig._live_executors = {self.state.executor_id: self.state}
        rig._active_large, rig._request_shapes, rig._link_bandwidth_samples = {}, {}, {}
        rig._active_large_since_ns = {}
        rig._phone_residency = None
        rig._transition_active = False
        rig.epoch_ns = time.monotonic_ns() - 1_000_000
        rig.configuration = SimpleNamespace(
            catalog=self.catalog, manifests={self.model.model_id: self.model},
            phone_device_id="helper-c", phone_memory_resource_id="phone-memory",
            host_memory_resource_id="host-memory", gpu_memory_resource_id="gpu-memory",
            large_phase_id_by_model={self.model.model_id: 1}, active_device_cost_features={},
            preloaded_model_by_executor={}, transition_phase_id=10,
        )
        rig._host_probe = SimpleNamespace(sample=lambda: SimpleNamespace(
            memory_total_bytes=4_000_000_000, memory_available_bytes=3_000_000_000,
            cpu_utilization_pct=0, memory_stall_avg10_basis_points=0,
        ))
        rig._sampler = SimpleNamespace(latest_gpu=lambda: {
            "memory_total_bytes": 2_000_000_000, "memory_free_bytes": 1_500_000_000,
            "memory_used_bytes": 500_000_000, "power_mw": 1000,
        })
        rig._activity = RuntimeActivityTracker(self.catalog)
        rig._snapshot_builder = UnifiedRuntimeSnapshotBuilder(self.catalog)
        rig._executor_samples = lambda: {row.executor_id: EndpointRuntimeSample("healthy", "live", 1)
                                        for row in self.catalog.executors}
        rig.phone_runtime_observation = lambda: (
            PhoneRuntimeProbe(9766 * 1024 + 8192, 8192, 40_000, 900_000, True, True),
            {"validity": "VALID", "valid": True, "source": "test-phone-runtime",
             "sample_timestamp_ns": time.monotonic_ns(), "age_us": 0, "maximum_age_us": 5_000_000},
        )
        self.raw = rig._probe_phone_allocation(self.state.executor_id)
        self.sample = SimpleNamespace(value=self.raw, error=None, stale=False)
        rig._runtime_monitor = SimpleNamespace(snapshot=lambda name: self.sample, request_refresh=Mock())

    def snapshot(self):
        return self.rig.snapshot(fixtures.request("whole-memory-request"), self.model.model_id, 0)

    def test_snapshot_refreshes_when_capture_outlives_observation(self):
        fresh = self.snapshot()
        expired = replace(fresh, valid_until_us=fresh.captured_at_us + 1)
        now_ns = self.rig.epoch_ns + (fresh.captured_at_us + 2) * 1000
        with patch.object(self.rig._snapshot_builder, "build", side_effect=(expired, fresh)) as build, \
                patch("research_dev.scheduler.adapters.heterogeneous_rig.time.monotonic_ns", return_value=now_ns):
            result = self.snapshot()
        self.assertEqual(build.call_count, 2)
        self.assertEqual(result.valid_until_us, fresh.valid_until_us)
        self.assertEqual(result.residency, fresh.residency)
        result.validate_at(fresh.captured_at_us + 2)
        detail = result.telemetry_observations[self.rig.configuration.phone_device_id]
        self.assertEqual(detail["snapshot_expired_captures"][0]["valid_until_us"], expired.valid_until_us)

    def test_snapshot_expiry_retries_are_bounded_without_extending_validity(self):
        fresh = self.snapshot()
        expired = replace(fresh, valid_until_us=fresh.captured_at_us + 1)
        now_ns = self.rig.epoch_ns + (fresh.captured_at_us + 2) * 1000
        with patch.object(self.rig._snapshot_builder, "build", return_value=expired) as build, \
                patch("research_dev.scheduler.adapters.heterogeneous_rig.time.monotonic_ns", return_value=now_ns):
            with self.assertRaisesRegex(PhysicalAdapterError, "snapshot capture retries exhausted"):
                self.snapshot()
        self.assertEqual(build.call_count, 3)
        self.assertEqual(expired.valid_until_us, fresh.captured_at_us + 1)

    def test_request_snapshot_excludes_only_owned_protected_work(self):
        owner = SimpleNamespace(
            ticket_id="owned-ticket", request_id="whole-memory-request", model_id=self.model.model_id,
            participants=(SimpleNamespace(device_id="host-a"),),
        )
        self.rig._active_large = {owner.ticket_id: owner}
        self.rig._execution_markers = {owner.ticket_id: object()}
        own_snapshot = self.snapshot()
        self.assertIsNone(own_snapshot.protected_work)
        own_scope = own_snapshot.telemetry_observations[
            self.rig.configuration.phone_device_id]["protected_work_scope"]
        self.assertEqual(own_scope["protected_work_own_ticket_count"], 1)
        self.assertFalse(any(key.startswith("protected_work_")
                             for key in own_snapshot.cost_features))
        self.assertEqual(own_snapshot.residency[0].generation, 5)
        peer = SimpleNamespace(ticket_id="peer-ticket", request_id="peer", model_id=self.model.model_id,
                               participants=owner.participants)
        self.rig._active_large[peer.ticket_id] = peer
        self.rig._execution_markers[peer.ticket_id] = object()
        self.rig._scheduler = SimpleNamespace(runtime_protected_work_end_us=lambda _: 10_000_000)
        peer_snapshot = self.snapshot()
        self.assertIsNotNone(peer_snapshot.protected_work)
        self.assertFalse(peer_snapshot.protected_work.measured)
        peer_scope = peer_snapshot.telemetry_observations[
            self.rig.configuration.phone_device_id]["protected_work_scope"]
        self.assertEqual(peer_scope["protected_work_peer_count"], 1)
        self.assertFalse(any(key.startswith("protected_work_")
                             for key in peer_snapshot.cost_features))
        self.assertEqual(peer_snapshot.residency, own_snapshot.residency)

    def test_adapter_to_snapshot_to_admission_credits_only_measured_allocation(self):
        snapshot = self.snapshot()
        observed, = snapshot.residency
        self.assertEqual(observed.reclaimable_bytes, 9766 * 1024)
        self.assertEqual(observed.generation, 5)
        self.assertNotEqual(observed.reclaimable_bytes, self.fixture.peak)
        detail, = snapshot.telemetry_observations["helper-c"]["resident_allocations"]
        self.assertEqual(detail["validity"], "VALID")
        self.assertEqual(detail["generation"], observed.generation)
        self.assertEqual(detail["raw_sha256"], self.raw["raw_sha256"])
        candidate = self.fixture.whole_candidate(snapshot)
        self.assertTrue(candidate.admitted, candidate.rejection_reasons)
        self.assertEqual(sum(row.additional_bytes for row in candidate.plan.memory_demands), 0)
        self.assertEqual(self.launcher._su.call_count, 1)
        self.assertIn("dumpsys -t 1 meminfo -s 123", self.launcher._su.call_args.args[0])
        self.assertEqual(self.launcher._su.call_args.kwargs["timeout_s"], 3)

    def test_failed_and_stale_probes_do_not_turn_peak_configuration_into_residency(self):
        for sample in (
            SimpleNamespace(value=None, error="TimeoutExpired: phone probe", stale=False),
            SimpleNamespace(value=None, error="allocation unavailable", stale=True),
            SimpleNamespace(value=self.raw, error=None, stale=True),
            SimpleNamespace(value={**self.raw, "captured_at_ns": time.monotonic_ns() - 6_000_000_000},
                            error=None, stale=False),
            SimpleNamespace(value={**self.raw, "generation": 4}, error=None, stale=False),
            SimpleNamespace(value={**self.raw, "allocated_bytes": 99_000_000}, error=None, stale=False),
            SimpleNamespace(value={**self.raw, "raw_sha256": "missing"}, error=None, stale=False),
            SimpleNamespace(value={**self.raw, "source": "configured-peak"}, error=None, stale=False),
        ):
            with self.subTest(sample=sample):
                self.sample = sample
                snapshot = self.snapshot()
                self.assertIsNone(snapshot.residency[0].reclaimable_bytes)
                candidate = self.fixture.whole_candidate(snapshot)
                self.assertFalse(candidate.admitted)
                self.assertIn("MEMORY_CAPACITY", candidate.rejection_reasons)
                detail, = snapshot.telemetry_observations["helper-c"]["resident_allocations"]
                self.assertIsNotNone(detail["failure_reason"])
                self.assertFalse(detail["valid"])
                if sample.error == "TimeoutExpired: phone probe":
                    self.assertEqual(detail["validity"], "TIMED_OUT")
                if sample.stale and sample.value is not None:
                    self.assertEqual(detail["validity"], "STALE")
        self.assertEqual(self.launcher._su.call_count, 1)
        self.assertEqual(self.rig._live_executors[self.state.executor_id], self.state)

    def test_snapshot_cannot_outlive_its_allocation_evidence(self):
        started = time.monotonic_ns() - 4_000_000_000
        self.sample = SimpleNamespace(
            value={**self.raw, "captured_at_ns": started}, error=None, stale=False,
        )
        snapshot = self.snapshot()
        expiry_us = (started + 5_000_000_000 - self.rig.epoch_ns) // 1000
        self.assertEqual(snapshot.valid_until_us, expiry_us)
        self.assertEqual(snapshot.memory.valid_until_us, expiry_us)
        snapshot.validate_at(snapshot.captured_at_us)
        with self.assertRaises(RuntimeCapabilityError):
            snapshot.validate_at(expiry_us + 1)

    def test_partial_measured_footprint_leaves_unobserved_peak_reserved(self):
        self.launcher._su.return_value = SimpleNamespace(stdout=allocation_output(kilobytes=9760))
        self.sample = SimpleNamespace(
            value=self.rig._probe_phone_allocation(self.state.executor_id), error=None, stale=False,
        )
        snapshot = self.snapshot()
        self.assertEqual(snapshot.residency[0].reclaimable_bytes, 9760 * 1024)
        candidate = self.fixture.whole_candidate(snapshot)
        self.assertEqual(sum(row.additional_bytes for row in candidate.plan.memory_demands),
                         self.fixture.peak - 9760 * 1024)
        self.assertFalse(candidate.admitted)
        self.assertIn("MEMORY_CAPACITY", candidate.rejection_reasons)

    def test_changed_launch_and_generation_receive_no_credit_from_previous_probe(self):
        for changes in (
            {"generation": 6}, {"endpoint": "http://127.0.0.1:18383"},
            {"parameters": {**self.state.parameters, "context_size": 256}},
            {"manifest": replace(self.model, artifact_sha256="sha256:" + "f" * 64)},
        ):
            with self.subTest(changes=changes):
                state = replace(self.state, **changes)
                self.rig._live_executors[state.executor_id] = state
                snapshot = self.snapshot()
                self.assertIsNone(snapshot.residency[0].reclaimable_bytes)
                detail, = snapshot.telemetry_observations["helper-c"]["resident_allocations"]
                self.assertFalse(detail["valid"])
                self.assertIsNotNone(detail["failure_reason"])
        self.assertEqual(self.launcher._su.call_count, 1)

    def test_generation_change_during_remote_probe_discards_the_sample(self):
        def result(*args, **kwargs):
            self.rig._live_executors[self.state.executor_id] = replace(self.state, generation=6)
            return SimpleNamespace(stdout=allocation_output())
        self.launcher._su.side_effect = result
        with self.assertRaisesRegex(PhysicalAdapterError, "generation changed during probe"):
            self.rig._probe_phone_allocation(self.state.executor_id)

    def test_remote_pid_reuse_and_timeout_are_failures_not_estimates(self):
        self.launcher._su.return_value = SimpleNamespace(stdout=allocation_output(after=identity_output(start=901)))
        with self.assertRaisesRegex(PhysicalAdapterError, "lifetime changed"):
            self.rig._probe_phone_allocation(self.state.executor_id)
        self.launcher._su.side_effect = subprocess.TimeoutExpired("probe", 3)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.rig._probe_phone_allocation(self.state.executor_id)


class WholePhonePhysicalIntegrationTests(unittest.TestCase):
    def setUp(self):
        source = PhoneAllocationSnapshotTests("runTest")
        source.setUp()
        self.source = source
        self.fixture, self.rig, self.server = source.fixture, source.rig, source.server
        rig = self.rig
        rig.configuration.resident_executor_id = "other-resident"
        rig._execution_markers, rig._execution_proofs = {}, {}
        rig._phone_execution_activity_ids = set()
        rig._phone_activity = PhoneActivityIntervalTracker()
        rig._direct_phone_session = Mock(active=True)
        rig._direct_phone_receipts = []
        self.participant = PhysicalParticipantCommand(
            source.state.executor_id, "helper-c", source.server.endpoint,
            source.catalog.executor_by_id[source.state.executor_id].backend,
            source.catalog.executor_by_id[source.state.executor_id].execution_resource_ids,
        )
        self.profile = RuntimePhonePowerProfile.assumed_4p5w(
            device_id="helper-c", domain_id="energy:helper-c",
        )

    def cold_transition(self):
        snapshot = replace(self.fixture.snapshot, residency=tuple(
            row for row in self.fixture.snapshot.residency if row.device_id != "helper-c"
        ))
        plan = self.fixture.whole_candidate(snapshot, variant=None).plan
        self.rig._live_executors = {}
        command = PhysicalTransitionCommand(
            "cold-ticket", "whole-memory-request", self.fixture.model.artifact_sha256,
            plan.route_id, plan.plan_sha256, self.participant, plan.transitions[0],
            plan.execution_contract, plan.adapter_parameters, operator_plan=plan.to_json(),
        )
        state = self.rig._begin_transition_execution(command, self.fixture.model, 1_000_000_000)
        self.assertIsNone(state.phone)
        self.assertIsNone(state.direct_transport)
        return command, state

    def execution(self):
        plan = self.fixture.whole_candidate().plan
        command = PhysicalExecutionCommand(
            "whole-ticket", "whole-memory-request", self.fixture.model.model_id,
            self.fixture.model.artifact_sha256, plan.route_id, self.participant.executor_id,
            self.participant.endpoint, "synthetic-proof", plan.plan_sha256, 0, 1000, 1000,
            plan.to_json(), (self.participant,), (), (), (), plan.execution_contract,
            plan.adapter_parameters,
        )
        self.rig._request_shapes[command.request_id] = (12, 4)
        self.marker = SimpleNamespace(phone_contract=None)
        self.server._managed.begin_execution = Mock(return_value=self.marker)
        return command

    def test_cold_whole_model_rechecks_live_capacity_and_health(self):
        command, state = self.cold_transition()
        healthy = PhoneRuntimeProbe(2_000_000_000, 1_000_000_000, 35_000, 900_000, True, True)
        for sample, reason in (
            (replace(healthy, available_bytes=0), "live memory capacity is insufficient"),
            (replace(healthy, thermal_qualified=False), "live safety check failed"),
            (replace(healthy, battery_ppm=0), "live safety check failed"),
        ):
            with self.subTest(reason=reason):
                self.rig.phone_runtime_observation = Mock(return_value=(sample, {"valid": True}))
                with self.assertRaisesRegex(PhysicalAdapterError, reason):
                    self.rig._validate_phone_transition_observation(command, state, lambda: None)
                self.rig.phone_runtime_observation.assert_called_once()
                self.assertFalse(state.mutation_started)
                self.assertEqual(self.rig._direct_phone_receipts, [])
        self.rig.phone_runtime_observation = Mock(return_value=(healthy, {"valid": True}))
        self.rig._validate_phone_transition_observation(command, state, lambda: None)
        self.assertEqual(self.rig._direct_phone_receipts[0]["required_bytes"], self.fixture.peak)
        self.rig._prepare_transition_phone(command, state, lambda: None)
        self.rig._direct_phone_session.supports.assert_not_called()
        self.rig._direct_phone_session.start.assert_not_called()

    def test_whole_model_missing_telemetry_defers_without_mutating_sessions(self):
        command, state = self.cold_transition()
        healthy = PhoneRuntimeProbe(2_000_000_000, 1_000_000_000, 35_000, 900_000, True, True)
        self.rig.phone_runtime_observation = Mock(side_effect=[
            (None, {"failure_reason": "STALE"}), (healthy, {"valid": True}),
        ])
        check = Mock()
        with patch("research_dev.scheduler.adapters.heterogeneous_rig.time.sleep"):
            self.rig._validate_phone_transition_observation(command, state, check)
        check.assert_called_once_with()
        self.assertEqual(self.rig.phone_runtime_observation.call_count, 2)
        self.assertEqual(len(self.rig._direct_phone_receipts), 1)
        self.assertFalse(state.mutation_started)
        self.rig._direct_phone_session.stop.assert_not_called()

    def test_whole_model_physical_target_mismatch_fails_before_load(self):
        command, state = self.cold_transition()
        for changed in (
            replace(command, adapter_parameters={**command.adapter_parameters, "gpu_device_id": "host-a"}),
            replace(command, participant=replace(command.participant, device_id="host-a")),
            replace(command, participant=replace(command.participant, endpoint="http://127.0.0.1:18384")),
        ):
            with self.subTest(command=changed):
                with self.assertRaisesRegex(PhysicalAdapterError, "whole-phone physical device differs"):
                    self.rig._validate_phone_transition_observation(changed, state, lambda: None)
        self.assertFalse(state.mutation_started)
        self.assertEqual(self.rig._direct_phone_receipts, [])

    def test_cold_whole_model_preparation_counts_phone_activity(self):
        _, state = self.cold_transition()
        with patch("research_dev.scheduler.adapters.heterogeneous_rig.time.monotonic_ns",
                   return_value=3_000_000_000):
            self.rig._finish_transition_execution(state)
        value = self.rig._phone_activity.estimate(1_000_000_000, 3_000_000_000,
                                                self.profile, charging_state="unknown")
        self.assertEqual(value.active_time_ns, 2_000_000_000)
        self.assertEqual(value.energy_uj, 9_000_000)

    def test_whole_model_terminal_callback_preserves_all_proof_arguments(self):
        command = self.execution()
        self.rig._execution_start(command)
        proof = SimpleNamespace(phone_calls_by_session={}, to_json=lambda: {"phone_call_count": 0})
        self.server._managed.finish_execution = Mock(return_value=proof)
        ack = {"synthetic-ack": True}
        result = self.rig._execution_success(command, {"static_ffn_control_ack": ack})
        self.assertEqual(result, proof.to_json())
        self.server._managed.finish_execution.assert_called_once_with(
            self.marker, command, self.fixture.model, output_tokens=4,
            adaptive_observation=None, static_control_ack=ack, helper_envelopes=(),
        )
        self.rig._direct_phone_session.bind_ticket_generation.assert_not_called()
        self.rig._direct_phone_session.record_execution_proof.assert_not_called()

    def test_whole_model_terminal_proof_failure_is_not_swallowed(self):
        command = self.execution()
        self.rig._execution_start(command)
        self.server._managed.finish_execution = Mock(side_effect=PhysicalAdapterError("invalid terminal proof"))
        with self.assertRaisesRegex(PhysicalAdapterError, "invalid terminal proof"):
            self.rig._execution_success(command, {})
        self.assertEqual(self.rig._execution_proofs, {})
        self.rig._execution_finish(command)

    def test_whole_model_execution_counts_activity_without_htp_calls(self):
        command = self.execution()
        with patch("research_dev.scheduler.adapters.heterogeneous_rig.time.monotonic_ns",
                   return_value=1_000_000_000):
            self.rig._execution_start(command)
        self.rig._execution_proofs[command.ticket_id] = {"phone_call_count": 0}
        with patch("research_dev.scheduler.adapters.heterogeneous_rig.time.monotonic_ns",
                   return_value=3_000_000_000):
            self.rig._execution_finish(command)
        value = self.rig._phone_activity.estimate(1_000_000_000, 3_000_000_000,
                                                self.profile, charging_state="unknown")
        self.assertEqual(value.active_time_ns, 2_000_000_000)
        self.assertEqual(value.energy_uj, 9_000_000)
        self.assertEqual(self.rig._activity.snapshot().active_by_device_id, {})
        self.rig._direct_phone_session.bind_ticket_generation.assert_not_called()
