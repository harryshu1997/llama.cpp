"""Permanently unsupported request shapes are rejected, never fatal.

Physical run sparse24-v14 aborted at arrival 49: a 2,375-token request on a
2,048-token preallocated context made every route infeasible, the mandatory
desktop control could not be generated and the campaign died. The scheduler
now judges the shape statically at submission and raises a typed rejection
with an exact reason; the arrival coordinator records it and keeps serving
every other request.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from research_dev.scheduler import (
    RequestShapeUnsupportedError,
    UnifiedScheduler,
)
from research_dev.scheduler.adapters.coordinator import (
    CanonicalArrivalCoordinator,
    CanonicalRuntimeSubmission,
)
from research_dev.scheduler._internal.policy import Request

try:
    from .test_arrival_coordinator import FakeMeasuredBackend
    from .test_automated_runtime import catalog, request, runtime_snapshot
    from .test_gguf_cost import write_synthetic_gguf
    from .test_work_conserving_start import WorkConservingStartTests
except ImportError:  # pragma: no cover - direct invocation
    from test_arrival_coordinator import FakeMeasuredBackend
    from test_automated_runtime import catalog, request, runtime_snapshot
    from test_gguf_cost import write_synthetic_gguf
    from test_work_conserving_start import WorkConservingStartTests


class RequestShapeAdmissionTests(unittest.TestCase):
    def _preallocated_scheduler(self):
        fixture = WorkConservingStartTests("startup_fixture")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(
            fixture.materialize(route_shape_profiles=(fixture.cpu_profile(),))
        )
        scheduler.register_model_manifest(fixture.manifest)
        return scheduler, fixture

    def test_shape_support_reads_preallocated_context_capacity(self) -> None:
        scheduler, fixture = self._preallocated_scheduler()
        model_id = fixture.manifest.model_id
        fits = scheduler.request_shape_support(
            request("fits", input_tokens=60, output_tokens=4), model_id
        )
        self.assertTrue(fits.supported)
        self.assertIsNone(fits.reason)
        # The fixture preallocates a 64-token context (quantum 1) on every
        # coordinator of the model, so capacity is 64 tokens each.
        self.assertTrue(fits.capacity_tokens_by_executor)
        self.assertEqual(set(fits.capacity_tokens_by_executor.values()), {64})
        too_big = scheduler.request_shape_support(
            request("too-big", input_tokens=60, output_tokens=5), model_id
        )
        self.assertFalse(too_big.supported)
        self.assertEqual(too_big.reason, "REQUEST_EXCEEDS_CONTEXT_CAPACITY")
        self.assertEqual(too_big.tokens, 65)
        self.assertEqual(too_big.details()["maximum_capacity_tokens"], 64)
        self.assertEqual(too_big.to_json()["supported"], False)

    def test_desktop_control_capacity_is_its_own_reason(self) -> None:
        """A shape that some coordinator could hold but the mandatory desktop
        control cannot is still permanently unsupported, with its own reason."""
        from types import SimpleNamespace
        from research_dev.scheduler._unified.automated_requests_ops import admission

        def coordinator(executor_id, context_size, quantum, resource):
            return SimpleNamespace(
                executor_id=executor_id, artifact_sha256="sha256:" + "a" * 64,
                adapter_parameters={
                    "request_memory_mode": "preallocated", "context_size": context_size,
                    "context_token_quantum": quantum, "context_resource_id": resource,
                },
            )

        catalog_ = SimpleNamespace(
            composite_executors=(
                coordinator("physical:cold:desktop", 2048, 512, "context:cold:desktop"),
                coordinator("physical:cold:cpu", 4096, 512, "context:cold:cpu"),
                SimpleNamespace(  # another model's coordinator is ignored
                    executor_id="physical:hot:desktop", artifact_sha256="sha256:" + "b" * 64,
                    adapter_parameters={"request_memory_mode": "preallocated", "context_size": 2048,
                                        "context_token_quantum": 1024, "context_resource_id": "context:hot:desktop"},
                ),
            ),
            resources={
                "context:cold:desktop": SimpleNamespace(capacity=4),
                "context:cold:cpu": SimpleNamespace(capacity=8),
                "context:hot:desktop": SimpleNamespace(capacity=2),
            },
            desktop_control_by_artifact={
                "sha256:" + "a" * 64: SimpleNamespace(executor_id="physical:cold:desktop"),
            },
        )
        controller = SimpleNamespace(
            _runtime_capabilities=catalog_,
            runtime_model_manifest=lambda model_id: SimpleNamespace(artifact_sha256="sha256:" + "a" * 64),
        )
        # Request 49 of sparse_locality24: 1,884 + 491 tokens.
        verdict = admission.request_shape_support(
            controller, request("index-49", input_tokens=1884, output_tokens=491), "gemma"
        )
        self.assertEqual(verdict.capacity_tokens_by_executor,
                         {"physical:cold:desktop": 2048, "physical:cold:cpu": 4096})
        self.assertEqual(verdict.desktop_control_capacity_tokens, 2048)
        self.assertFalse(verdict.supported)
        self.assertEqual(verdict.reason, admission.REQUEST_EXCEEDS_DESKTOP_CONTROL_CONTEXT)
        # Beyond every coordinator: the stronger reason wins.
        huge = admission.request_shape_support(
            controller, request("huge", input_tokens=4000, output_tokens=500), "gemma"
        )
        self.assertEqual(huge.reason, admission.REQUEST_EXCEEDS_CONTEXT_CAPACITY)
        # Exactly at the desktop control's capacity fits (5 slots of 512 = 2560
        # after the 2560-token qualification; 4 slots = 2048 here).
        fits = admission.request_shape_support(
            controller, request("edge", input_tokens=2000, output_tokens=48), "gemma"
        )
        self.assertTrue(fits.supported)
        with self.assertRaises(RequestShapeUnsupportedError) as caught:
            admission.require_supported_request_shape(
                controller, request("index-49", input_tokens=1884, output_tokens=491), "gemma"
            )
        self.assertEqual(caught.exception.details["desktop_control_capacity_tokens"], 2048)
        self.assertEqual(caught.exception.details["maximum_capacity_tokens"], 4096)

    def test_submission_rejects_with_exact_reason_and_no_journal_record(self) -> None:
        scheduler, fixture = self._preallocated_scheduler()
        model_id = fixture.manifest.model_id
        snapshot = runtime_snapshot(fixture.manifest, include_phone=False)
        oversize = request("oversize", input_tokens=60, output_tokens=5)
        with self.assertRaises(RequestShapeUnsupportedError) as caught:
            scheduler.submit_automated_request(
                oversize, model_id, snapshot, observed_at_us=oversize.arrival_us
            )
        self.assertEqual(caught.exception.reason, "REQUEST_EXCEEDS_CONTEXT_CAPACITY")
        self.assertEqual(caught.exception.details["tokens"], 65)
        self.assertEqual(caught.exception.status, "REJECTED")
        self.assertEqual(scheduler.runtime_decision_log()["records"], [])
        self.assertTrue(str(caught.exception).startswith("REQUEST_EXCEEDS_CONTEXT_CAPACITY"))

    def test_coordinator_records_rejection_and_keeps_serving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "arrival.gguf"
            write_synthetic_gguf(model_path)
            scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
            scheduler.register_runtime_capabilities(catalog())
            manifest = scheduler.register_gguf_model("synthetic-arrival-model", model_path)
            snapshot = runtime_snapshot(manifest)
            real_submit = scheduler.submit_automated_request

            def submit(request_, *args, **kwargs):
                if request_.request_id == "too-big":
                    raise RequestShapeUnsupportedError(
                        "REQUEST_EXCEEDS_CONTEXT_CAPACITY",
                        {"tokens": 2375, "maximum_capacity_tokens": 2048},
                    )
                return real_submit(request_, *args, **kwargs)

            coordinator = CanonicalArrivalCoordinator(
                scheduler, FakeMeasuredBackend(5_000), epoch_ns=time.monotonic_ns(),
                snapshot_provider=lambda _ticket, _at_us: snapshot, max_workers=2,
            )
            try:
                with mock.patch.object(scheduler, "submit_automated_request", side_effect=submit):
                    big = Request(request_id="too-big", workload_id="synthetic-arrival-work",
                                  arrival_us=0, deadline_us=1_000_000, input_tokens=12,
                                  output_tokens=4, quality_requirement="exact")
                    rejected = coordinator.submit(
                        CanonicalRuntimeSubmission(big, manifest.model_id, snapshot, {}),
                        observed_at_us=1,
                    )
                    self.assertIsNone(rejected)
                    ok = Request(request_id="fits", workload_id="synthetic-arrival-work",
                                 arrival_us=0, deadline_us=1_000_000, input_tokens=12,
                                 output_tokens=4, quality_requirement="exact")
                    ticket = coordinator.submit(
                        CanonicalRuntimeSubmission(ok, manifest.model_id, snapshot, {}),
                        observed_at_us=2,
                    )
                    self.assertIsNotNone(ticket)
                    # A duplicate of a rejected arrival is still a duplicate.
                    with self.assertRaisesRegex(Exception, "duplicated"):
                        coordinator.submit(
                            CanonicalRuntimeSubmission(big, manifest.model_id, snapshot, {}),
                            observed_at_us=3,
                        )
                result = coordinator.drain(timeout_s=30)
            finally:
                coordinator.close()
            rejections = coordinator.rejections()
            self.assertEqual([row.request_id for row in rejections], ["too-big"])
            self.assertEqual(rejections[0].reason, "REQUEST_EXCEEDS_CONTEXT_CAPACITY")
            self.assertEqual(rejections[0].to_json()["details"]["tokens"], 2375)
            self.assertEqual(result.request_ids, ("fits",))
            self.assertEqual(
                {row["request_ids"][0] for row in scheduler.runtime_decision_log()["records"]},
                {"fits"},
            )


if __name__ == "__main__":
    unittest.main()
