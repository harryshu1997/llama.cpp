#!/usr/bin/env python3

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from research_dev.scheduler import (
    DeviceMemoryCapacity,
    RuntimeCapabilityCatalog,
    RuntimePlacementSnapshot,
    RuntimeRouteShapeProfile,
    UnifiedScheduleError,
    UnifiedScheduler,
)
from research_dev.scheduler.adapters import (
    EndpointRuntimeSample,
    materialize_preflight_executor_samples,
    PhysicalPreflightModel,
    run_physical_preflight,
    validate_decision_candidate_coverage,
)
from research_dev.scheduler.adapters.preflight import _check_desktop_baseline
from research_dev.scheduler._internal.desktop_parent import (
    select_desktop_parent_for_live_vram,
)
import test_desktop_parent_capacity as parent_fixture

try:
    from .test_automated_runtime import (
        catalog,
        catalog_with_gpu_desktop_control,
        request,
        runtime_snapshot,
    )
    from .test_gguf_cost import write_synthetic_gguf
except ImportError:
    from test_automated_runtime import (
        catalog,
        catalog_with_gpu_desktop_control,
        request,
        runtime_snapshot,
    )
    from test_gguf_cost import write_synthetic_gguf


def http_catalog(model_path: Path) -> RuntimeCapabilityCatalog:
    source = catalog()
    executors = tuple(
        replace(
            row,
            endpoint="http://127.0.0.1:" + str(19001 + index),
            adapter_parameters={
                **dict(row.adapter_parameters),
                "model_alias": "synthetic-model",
            },
        )
        for index, row in enumerate(source.executors)
    )
    artifact_sha256 = "sha256:" + hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()
    route_profiles = tuple(
        RuntimeRouteShapeProfile(
            selector_id="measured-preflight-" + device_id,
            artifact_sha256=artifact_sha256,
            route_family="whole_model",
            device_ids=(device_id,),
            assisted_operator_kind=None,
            split_axis="none",
            split_fraction_ppm=0,
            residency_variant="hot",
            minimum_input_tokens=1,
            maximum_input_tokens=128,
            minimum_output_tokens=1,
            maximum_output_tokens=128,
            service_fixed_us=100,
            service_input_token_us=10,
            service_output_token_us=20,
            service_upper_add_us=50,
            energy_fixed_uj=1_000,
            energy_input_token_uj=10,
            energy_output_token_uj=20,
            energy_lower_error_ppm=100_000,
            energy_upper_error_ppm=100_000,
            sample_count=4,
            maturity="QUALIFIED",
            evidence_ids=("held-out-preflight-" + device_id,),
            executor_id="executor:" + device_id,
        )
        for device_id in ("host-a", "accelerator-b")
    ) + (RuntimeRouteShapeProfile(
        selector_id="measured-preflight-cpu-gpu",
        artifact_sha256=artifact_sha256,
        route_family="layer_placement",
        device_ids=("accelerator-b", "host-a"),
        assisted_operator_kind=None,
        split_axis="none",
        split_fraction_ppm=0,
        residency_variant="hot",
        minimum_input_tokens=1,
        maximum_input_tokens=128,
        minimum_output_tokens=1,
        maximum_output_tokens=128,
        service_fixed_us=100,
        service_input_token_us=10,
        service_output_token_us=20,
        service_upper_add_us=50,
        energy_fixed_uj=1_000,
        energy_input_token_uj=10,
        energy_output_token_uj=20,
        energy_lower_error_ppm=100_000,
        energy_upper_error_ppm=100_000,
        sample_count=4,
        maturity="QUALIFIED",
        evidence_ids=("held-out-preflight-cpu-gpu",),
    ),)
    return RuntimeCapabilityCatalog.from_json(replace(
        source,
        executors=executors,
        route_shape_profiles=route_profiles,
    ).to_json())


class PhysicalPreflightTests(unittest.TestCase):
    def test_extra_vram_does_not_replace_a_frozen_qualified_parent(self) -> None:
        parent_fixture.DesktopParentCapacityTests.setUpClass()
        fixture = parent_fixture.DesktopParentCapacityTests()
        source = fixture._source()
        capacity = DeviceMemoryCapacity("cuda0-vram", 100_000_000_000, 0, 0)
        parent = select_desktop_parent_for_live_vram(
            fixture.manifest, source, capacity, memory_snapshot_id="frozen"
        ).selected
        source = replace(source, adapter_parameters={
            **source.adapter_parameters,
            "capacity_parent_maximum_gpu_layers": fixture.manifest.block_count,
        })
        exploratory = select_desktop_parent_for_live_vram(
            fixture.manifest, source, capacity, memory_snapshot_id="more-vram"
        ).selected
        self.assertGreater(exploratory.gpu_layers, parent.gpu_layers)
        control = SimpleNamespace(
            executor_id=source.executor_id, placement_sha256=parent.placement_sha256,
            maturity="QUALIFIED", evidence_ids=("physical-parent",),
        )
        catalog = SimpleNamespace(
            desktop_control_by_artifact={fixture.manifest.artifact_sha256: control},
            composite_executor_by_id={source.executor_id: source},
            placement_profile=SimpleNamespace(devices={
                "desktop-cuda": SimpleNamespace(kind="gpu", memory_pool_id="cuda0-vram"),
            }),
        )
        scheduler = SimpleNamespace(
            runtime_model_manifest=lambda _: fixture.manifest,
            _runtime_capabilities=catalog,
        )
        scheduler.select_live_vram_desktop_parent = lambda *args, **kwargs: (
            UnifiedScheduler.select_live_vram_desktop_parent.__wrapped__(
                scheduler, *args, **kwargs
            )
        )
        baseline = SimpleNamespace(
            device_ids=("desktop-cpu", "desktop-cuda"), admitted=True,
            maturity="QUALIFIED", rejection_reasons=(), candidate_id="frozen-parent",
            binding=SimpleNamespace(executor_id=source.executor_id,
                                    endpoint=source.endpoint, operator_plan_protocol="v1"),
            plan=SimpleNamespace(transitions=(), route_profile_id=None,
                                 desktop_placement_sha256=parent.placement_sha256),
            cost=SimpleNamespace(latency_evidence="ASSUMED", energy_evidence="ASSUMED",
                                 fleet_energy_uj=100),
        )
        for capacity_bytes, fits in ((100_000_000_000, True), (10_000_000_000, False)):
            with self.subTest(capacity_bytes=capacity_bytes):
                row = SimpleNamespace(
                    manifest=fixture.manifest, expected_model_alias=None,
                    snapshot=SimpleNamespace(memory=RuntimePlacementSnapshot(
                        snapshot_id="live", captured_at_us=0, valid_until_us=1_000_000,
                        capacities={capacity.resource_id: replace(
                            capacity, capacity_bytes=capacity_bytes
                        )},
                    )),
                )
                checks = []
                args = (checks, scheduler, catalog, row, baseline,
                        {"desktop-cpu": "cpu", "desktop-cuda": "gpu"}, {}, {})
                if fits:
                    self.assertTrue(_check_desktop_baseline(*args))
                    self.assertEqual(checks[0].status, "PASS")
                    self.assertIn("selected_gpu_layers=24;", checks[0].detail)
                else:
                    with self.assertRaisesRegex(UnifiedScheduleError, "does not fit"):
                        _check_desktop_baseline(*args)
        self.assertEqual(source.adapter_parameters["gpu_layers"], parent.gpu_layers)

    def test_scheduler_adapter_materializes_preflight_readiness(self) -> None:
        source = catalog()
        source = replace(
            source,
            transitions=tuple(
                replace(
                    row,
                    executor_id="executor:" + row.device_id,
                )
                for row in source.transitions
            ),
        )
        raw = {
            row.endpoint: EndpointRuntimeSample(
                "unavailable", "unavailable", 0
            )
            for row in source.executors
        }
        accelerator = source.executor_by_id["executor:accelerator-b"]
        raw[accelerator.endpoint] = EndpointRuntimeSample(
            "healthy", "live", 1
        )

        samples = materialize_preflight_executor_samples(
            source,
            raw,
            bootstrap_executor_ids=("executor:host-a",),
        )

        self.assertTrue(samples["executor:host-a"].ready)
        self.assertTrue(samples["executor:accelerator-b"].ready)
        self.assertFalse(samples["executor:helper-c"].ready)
        self.assertTrue(
            samples["executor:helper-c"].transition_available
        )

    def test_desktop_pass_reports_unavailable_phone_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.gguf"
            executable = root / "llama-server"
            write_synthetic_gguf(model_path)
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
            executable.chmod(0o700)

            scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
            capability_catalog = http_catalog(model_path)
            scheduler.register_runtime_capabilities(capability_catalog)
            manifest = scheduler.register_gguf_model(
                "synthetic-preflight-model", model_path
            )
            runtime = runtime_snapshot(manifest, include_phone=False)
            report = run_physical_preflight(
                scheduler,
                capability_catalog,
                (
                    PhysicalPreflightModel(
                        manifest,
                        model_path,
                        request("synthetic-preflight-request"),
                        runtime,
                        "synthetic-model",
                    ),
                ),
                executable_paths={"server": executable},
                required_paths={"model-index": model_path},
                library_directories={"server-libraries": root},
            )

        self.assertEqual(report.status, "PASS")
        self.assertTrue(report.desktop_baseline_ready)
        self.assertFalse(report.phone_assistance_ready)
        self.assertTrue(any(
            row.check_id.startswith("candidate-families:")
            and row.status == "PASS"
            for row in report.checks
        ))
        unavailable = [
            row for row in report.phone_routes if not row.admitted
        ]
        self.assertTrue(unavailable)
        self.assertTrue(all(
            row.maturity == "QUALIFIED"
            and "fresh endpoint health and free-slot observation"
                in row.required_evidence
            for row in unavailable
        ))
        self.assertTrue(all(
            row["maturity"] != "QUALIFIED" or not row["admitted"]
            for row in report.to_json()[
                "shadow_or_unavailable_phone_routes"
            ]
        ))

    def test_unsupported_trace_request_shape_blocks_preflight(self) -> None:
        """sparse24-v14 aborted on a 2,375-token request against a 2,048-token
        context. Preflight now screens every trace request's shape and blocks
        before a paid run when any can never be preallocated."""
        from research_dev.scheduler._unified.automated_requests_ops.admission import (
            RequestShapeSupport,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.gguf"
            executable = root / "llama-server"
            write_synthetic_gguf(model_path)
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
            executable.chmod(0o700)
            scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
            capability_catalog = http_catalog(model_path)
            scheduler.register_runtime_capabilities(capability_catalog)
            manifest = scheduler.register_gguf_model(
                "synthetic-preflight-model", model_path
            )
            runtime = runtime_snapshot(manifest, include_phone=False)
            model_row = PhysicalPreflightModel(
                manifest, model_path, request("synthetic-preflight-request"),
                runtime, "synthetic-model",
            )
            real_support = scheduler.request_shape_support

            def support(request_, model_id):
                verdict = real_support(request_, model_id)
                if request_.request_id == "index-49":
                    return RequestShapeSupport(
                        request_id=request_.request_id, model_id=model_id,
                        tokens=2375, supported=False,
                        reason="REQUEST_EXCEEDS_CONTEXT_CAPACITY",
                        desktop_control_executor_id="executor:accelerator-b",
                        desktop_control_capacity_tokens=2048,
                        capacity_tokens_by_executor={"executor:accelerator-b": 2048},
                    )
                return verdict

            trace = (
                (request("index-48", input_tokens=12, output_tokens=4), manifest.model_id),
                (request("index-49", input_tokens=12, output_tokens=4), manifest.model_id),
            )
            with mock.patch.object(scheduler, "request_shape_support", side_effect=support):
                blocked = run_physical_preflight(
                    scheduler, capability_catalog, (model_row,),
                    executable_paths={"server": executable},
                    required_paths={"model-index": model_path},
                    library_directories={"server-libraries": root},
                    trace_requests=trace,
                )
            clean = run_physical_preflight(
                scheduler, capability_catalog, (model_row,),
                executable_paths={"server": executable},
                required_paths={"model-index": model_path},
                library_directories={"server-libraries": root},
                trace_requests=trace[:1],
            )
        self.assertEqual(blocked.status, "BLOCKED")
        shape_check = next(row for row in blocked.checks if row.check_id == "request-shapes")
        self.assertEqual(shape_check.status, "BLOCKED")
        self.assertIn("index-49=REQUEST_EXCEEDS_CONTEXT_CAPACITY(2375 tokens)", shape_check.detail)
        report = blocked.to_json()
        self.assertEqual(report["request_shapes"]["checked"], 2)
        self.assertEqual(
            [row["request_id"] for row in report["request_shapes"]["unsupported"]],
            ["index-49"],
        )
        self.assertEqual(clean.status, "PASS")
        self.assertEqual(
            next(row for row in clean.checks if row.check_id == "request-shapes").status,
            "PASS",
        )
        self.assertEqual(clean.to_json()["request_shapes"], {"checked": 1, "unsupported": []})

    def test_missing_executable_fails_closed_without_promoting_phone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.gguf"
            write_synthetic_gguf(model_path)
            scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
            capability_catalog = http_catalog(model_path)
            scheduler.register_runtime_capabilities(capability_catalog)
            manifest = scheduler.register_gguf_model(
                "synthetic-blocked-model", model_path
            )
            report = run_physical_preflight(
                scheduler,
                capability_catalog,
                (
                    PhysicalPreflightModel(
                        manifest,
                        model_path,
                        request("synthetic-blocked-request"),
                        runtime_snapshot(manifest),
                    ),
                ),
                executable_paths={"server": root / "missing-server"},
            )

        self.assertEqual(report.status, "BLOCKED")
        self.assertFalse(report.desktop_baseline_ready)
        self.assertTrue(any(
            row.check_id == "executable:server"
            and row.status == "BLOCKED"
            for row in report.checks
        ))
        self.assertTrue(all(
            row.maturity == "QUALIFIED"
            for row in report.phone_routes if row.admitted
        ))

    def test_shared_endpoint_variants_use_live_memory_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.gguf"
            executable = root / "llama-server"
            write_synthetic_gguf(model_path)
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
            executable.chmod(0o700)

            scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
            capability_catalog = http_catalog(model_path)
            executors = list(capability_catalog.executors)
            executors[1] = replace(
                executors[1], endpoint=executors[0].endpoint
            )
            capability_catalog = RuntimeCapabilityCatalog.from_json(replace(
                capability_catalog, executors=tuple(executors)
            ).to_json())
            scheduler.register_runtime_capabilities(capability_catalog)
            manifest = scheduler.register_gguf_model(
                "synthetic-shared-endpoint-model", model_path
            )
            runtime = runtime_snapshot(manifest, include_phone=False)
            capacities = dict(runtime.memory.capacities)
            host = capacities["host-memory"]
            capacities["host-memory"] = replace(
                host, capacity_bytes=host.capacity_bytes + 1024
            )
            runtime = replace(
                runtime,
                memory=replace(runtime.memory, capacities=capacities),
            )
            report = run_physical_preflight(
                scheduler,
                capability_catalog,
                (
                    PhysicalPreflightModel(
                        manifest,
                        model_path,
                        request("synthetic-shared-endpoint-request"),
                        runtime,
                        "synthetic-model",
                    ),
                ),
                executable_paths={"server": executable},
            )

        self.assertEqual(report.status, "PASS")
        checks = {row.check_id: row for row in report.checks}
        self.assertEqual(checks["endpoint-contracts"].status, "PASS")
        self.assertEqual(
            checks["memory:" + manifest.model_id].status, "PASS"
        )

    def test_qualified_control_allows_shadow_cost_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.gguf"
            executable = root / "llama-server"
            write_synthetic_gguf(model_path)
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
            executable.chmod(0o700)

            source = http_catalog(model_path)
            bootstrap = UnifiedScheduler.for_runtime_discovery("enforce")
            bootstrap.register_runtime_capabilities(source)
            manifest = bootstrap.register_gguf_model(
                "synthetic-controlled-model", model_path
            )
            controlled = catalog_with_gpu_desktop_control(manifest, source)
            controlled = RuntimeCapabilityCatalog.from_json(replace(
                controlled,
                route_shape_profiles=tuple(
                    replace(profile, maturity="SHADOW")
                    for profile in controlled.route_shape_profiles
                ),
            ).to_json())
            scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
            scheduler.register_runtime_capabilities(controlled)
            scheduler.register_model_manifest(manifest)
            report = run_physical_preflight(
                scheduler,
                controlled,
                (
                    PhysicalPreflightModel(
                        manifest,
                        model_path,
                        request("synthetic-controlled-request"),
                        runtime_snapshot(manifest, include_phone=False),
                        "synthetic-model",
                    ),
                ),
                executable_paths={"server": executable},
            )

        checks = {row.check_id: row for row in report.checks}
        self.assertEqual(report.status, "PASS")
        self.assertIn("control_qualified=True", checks[
            "desktop-baseline:" + manifest.model_id
        ].detail)
        self.assertEqual(checks[
            "desktop-cost-evidence:" + manifest.model_id
        ].status, "WARN")

    def test_shadow_cost_without_control_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.gguf"
            executable = root / "llama-server"
            write_synthetic_gguf(model_path)
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
            executable.chmod(0o700)

            source = http_catalog(model_path)
            source = RuntimeCapabilityCatalog.from_json(replace(
                source,
                route_shape_profiles=tuple(
                    replace(profile, maturity="SHADOW")
                    for profile in source.route_shape_profiles
                ),
            ).to_json())
            scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
            scheduler.register_runtime_capabilities(source)
            manifest = scheduler.register_gguf_model(
                "synthetic-uncontrolled-model", model_path
            )
            report = run_physical_preflight(
                scheduler,
                source,
                (
                    PhysicalPreflightModel(
                        manifest,
                        model_path,
                        request("synthetic-uncontrolled-request"),
                        runtime_snapshot(manifest, include_phone=False),
                        "synthetic-model",
                    ),
                ),
                executable_paths={"server": executable},
            )

        checks = {row.check_id: row for row in report.checks}
        self.assertEqual(report.status, "BLOCKED")
        self.assertIn("control_qualified=False", checks[
            "desktop-baseline:" + manifest.model_id
        ].detail)
        self.assertEqual(checks[
            "desktop-cost-evidence:" + manifest.model_id
        ].status, "WARN")

    def test_per_request_coverage_rejects_a_silently_missing_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            write_synthetic_gguf(model_path)
            scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
            capability_catalog = http_catalog(model_path)
            scheduler.register_runtime_capabilities(capability_catalog)
            manifest = scheduler.register_gguf_model(
                "synthetic-coverage-model", model_path
            )
            current = request("synthetic-coverage-request")
            scheduler.submit_automated_request(
                current,
                manifest.model_id,
                runtime_snapshot(manifest, include_phone=False),
            )
            records = copy.deepcopy(
                scheduler.runtime_decision_log()["records"]
            )
        decision = next(
            row for row in records if row["event_kind"] == "DECISION"
        )
        decision["candidates"] = [
            row for row in decision["candidates"]
            if row["details"]["device_ids"] != ["helper-c"]
        ]
        with self.assertRaisesRegex(Exception, "families are absent"):
            validate_decision_candidate_coverage(
                capability_catalog,
                records,
                {current.request_id: manifest},
            )


if __name__ == "__main__":
    unittest.main()
