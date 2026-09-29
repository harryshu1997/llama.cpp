#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from research_dev.scheduler import (
    GGUFModelManifestLoader,
    MemoryPoolProfile,
    RuntimeCapabilityCatalog,
    RuntimeCompositeOperatorPlacement,
    RuntimeDesktopControlProfile,
    RuntimeLinkState,
    RuntimePhoneSessionCapability,
    UnifiedScheduler,
)
from research_dev.scheduler.adapters import (
    RuntimeModelEndpointCapability,
    RuntimePhysicalTopology,
    base_executor_capabilities,
    derived_desktop_gpu_first_layer,
    desktop_model_composite_capability,
    measured_desktop_control_profile,
    measured_ffn_assist_profile,
    materialize_whole_model_endpoint,
    merge_runtime_capability_catalogs,
    model_composite_capabilities,
    model_transition_capabilities,
    resource_profiles_for_catalog,
)
from research_dev.scheduler.campaigns.burstgpt.catalog import (
    measured_desktop_plans,
    validate_desktop_control_bindings,
    validate_final_catalog,
)

try:
    from .test_gguf_cost import write_synthetic_gguf
except ImportError:
    from test_gguf_cost import write_synthetic_gguf


EVIDENCE = "sha256:" + "1" * 64


def topology(prefix: str = "new") -> RuntimePhysicalTopology:
    return RuntimePhysicalTopology(
        cpu_device_id=prefix + "-cpu",
        gpu_device_id=prefix + "-gpu",
        phone_device_id=prefix + "-phone",
        cpu_resource_id=prefix + "-cpu-compute",
        gpu_resource_id=prefix + "-gpu-compute",
        host_memory_resource_id=prefix + "-host-memory",
        gpu_memory_resource_id=prefix + "-gpu-memory",
        phone_memory_resource_id=prefix + "-phone-memory",
        functionfs_resource_id=prefix + "-functionfs",
        phone_transport_resource_ids=(
            prefix + "-functionfs",
            prefix + "-usb",
        ),
        phone_compute_resource_ids=(prefix + "-phone-compute",),
        gpu_exclusive_residency_resource_id=prefix + "-gpu-residency",
        resource_capacities={
            prefix + "-cpu-compute": 4,
            prefix + "-gpu-compute": 1,
            prefix + "-gpu-residency": 1,
            prefix + "-functionfs": 1,
            prefix + "-usb": 1,
            prefix + "-phone-compute": 1,
        },
        resource_identities={prefix + "-usb": prefix + "-usb-root"},
    )


class CatalogMaterializationTests(unittest.TestCase):
    def manifest(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "unseen.gguf"
        write_synthetic_gguf(path, block_count=2)
        return GGUFModelManifestLoader.load("unseen-model", path)

    def test_measured_desktop_plan_preserves_runtime_parameters(
        self,
    ) -> None:
        manifest = self.manifest()
        control = RuntimeDesktopControlProfile(
            profile_id="synthetic-desktop-control",
            artifact_sha256=manifest.artifact_sha256,
            executor_id="synthetic-desktop-executor",
            operator_placements=tuple(
                RuntimeCompositeOperatorPlacement(
                    operator_id=operator.operator_id,
                    primary_device_id="new-gpu",
                    helper_device_id=None,
                    split_axis="none",
                    split_fraction_ppm=0,
                )
                for operator in manifest.operators
            ),
            maturity="QUALIFIED",
            evidence_ids=(EVIDENCE,),
        )
        value = {
            "plans": [{
                "adapter_parameters": {
                    "context_size": 4096,
                    "kv_cache_swa_padding_tokens": 512,
                    "parallel": 2,
                },
                "artifact_sha256": manifest.artifact_sha256,
                "desktop_control_profile_id": control.profile_id,
                "desktop_executor_id": control.executor_id,
                "desktop_placement_sha256": control.placement_sha256,
                "evidence_ids": [EVIDENCE],
                "gpu_first_layer": 1,
                "identity_evidence_id": EVIDENCE,
                "maturity": "QUALIFIED",
            }],
            "schema": "s42-measured-desktop-baseline-plans-v1",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plans.json"
            path.write_text(
                json.dumps(value, ensure_ascii=True), encoding="ascii"
            )

            plans = measured_desktop_plans(
                path, {"model": manifest}
            )

        self.assertEqual(
            plans[manifest.artifact_sha256]["adapter_parameters"],
            value["plans"][0]["adapter_parameters"],
        )
        validate_desktop_control_bindings(plans, (control,))
        with self.assertRaisesRegex(
            ValueError, "physical profile binding"
        ):
            validate_desktop_control_bindings(
                plans,
                (replace(control, executor_id="different-executor"),),
            )

    def test_phone_residency_scope_is_declared_by_topology(self) -> None:
        profile = replace(
            topology(),
            phone_exclusive_residency_resource_id="new-phone-compute",
        )

        phone = next(
            row for row in base_executor_capabilities(
                profile, evidence_id=EVIDENCE
            )
            if row.device_id == "new-phone"
        )

        self.assertEqual(
            phone.exclusive_residency_resource_id,
            "new-phone-compute",
        )

    def test_current_link_measurement_wins_during_overlay_merge(self) -> None:
        try:
            from .test_automated_runtime import catalog
        except ImportError:
            from test_automated_runtime import catalog

        current = catalog()
        link_id = next(
            resource_id for resource_id in current.resources
            if resource_id.startswith("link:")
        )
        overlay_resources = dict(current.resources)
        overlay_resources[link_id] = replace(
            overlay_resources[link_id],
            capacity=overlay_resources[link_id].capacity + 1,
        )
        overlay = replace(
            current,
            resources=overlay_resources,
            composite_executors=(),
            desktop_control_profiles=(),
            transitions=(),
            route_shape_profiles=(),
            system_cost_profiles=(),
        )

        merged = merge_runtime_capability_catalogs(current, overlay)

        self.assertEqual(merged.resources[link_id], current.resources[link_id])

    def test_overlay_merge_preserves_discovered_session_pool(self) -> None:
        try:
            from .test_automated_runtime import catalog
        except ImportError:
            from test_automated_runtime import catalog

        overlay = catalog()
        phone = next(
            row for row in overlay.executors if row.device_id == "helper-c"
        )
        transport = next(
            resource_id for resource_id in overlay.resources
            if resource_id.startswith("link:")
        )
        session = RuntimePhoneSessionCapability(
            session_id="session-a",
            device_id=phone.device_id,
            endpoint="session://helper-c/session-a",
            worker_identity_sha256="sha256:" + "a" * 64,
            memory_resource_id="phone-memory:session-a",
            resident_memory_limit_bytes=1_000_000,
            shared_compute_resource_id="compute:helper-c",
            shared_transport_resource_ids=(transport,),
            supported_layer_mask=3,
            maximum_columns=32,
            column_quantum=16,
            supported_data_types=("F16",),
            batch_plans=("coalesced-batch",),
            ready=True,
            residency_state="cold",
        )
        pools = dict(overlay.placement_profile.memory_pools)
        pools[session.memory_resource_id] = MemoryPoolProfile(
            pool_id=session.memory_resource_id,
            capacity_bytes=session.resident_memory_limit_bytes,
            reserved_bytes=0,
        )
        base = replace(
            overlay,
            placement_profile=replace(
                overlay.placement_profile,
                memory_pools=pools,
            ),
            executors=tuple(
                replace(
                    row,
                    execution_resource_ids=tuple(sorted({
                        *row.execution_resource_ids,
                        transport,
                    })),
                    phone_sessions=(session,),
                )
                if row.device_id == phone.device_id else row
                for row in overlay.executors
            ),
        )

        merged = merge_runtime_capability_catalogs(base, replace(
            overlay,
            composite_executors=(),
            desktop_control_profiles=(),
            transitions=(),
            route_shape_profiles=(),
            system_cost_profiles=(),
        ))

        self.assertEqual(
            merged.placement_profile.memory_pools[
                session.memory_resource_id
            ].capacity_bytes,
            session.resident_memory_limit_bytes,
        )
        self.assertEqual(
            merged.executor_by_device[phone.device_id].phone_sessions,
            (session,),
        )

    def test_measured_phone_work_normalizes_without_model_policy(self) -> None:
        manifest = self.manifest()
        split = measured_ffn_assist_profile(manifest, ({
            "execution_mode": "parallel_split",
            "n_embd": manifest.embedding_length,
            "phone_columns": manifest.feed_forward_length // 2,
        },))
        offload = measured_ffn_assist_profile(manifest, ({
            "execution_mode": "full_replacement",
            "n_embd": manifest.embedding_length,
            "phone_columns": manifest.feed_forward_length,
        },))

        self.assertEqual(split.route_family, "operator_split")
        self.assertEqual(split.split_fraction_ppm, 500_000)
        self.assertEqual(offload.route_family, "operator_offload")
        self.assertEqual(offload.split_fraction_ppm, 0)

    def test_unseen_model_and_device_ids_generate_physical_routes(self) -> None:
        manifest = self.manifest()
        hardware = topology("discovered")
        model = RuntimeModelEndpointCapability(
            manifest=manifest,
            desktop_executor_id="endpoint:desktop",
            desktop_endpoint="http://desktop.invalid:1",
            desktop_backend="llama-server",
            phone_executor_prefix="endpoint:phone-assisted",
            phone_endpoint="http://phone.invalid:1",
            phone_backend="llama-server-phone",
            desktop_gpu_first_layer=1,
            adapter_parameters={"parallel": 2},
            phone_adapter_parameters={"ffn_activation": "geglu"},
            phone_runtime_control_protocol="decode-boundary-v1",
            desktop_evidence_ids=(EVIDENCE,),
            phone_evidence_ids=(EVIDENCE,),
            phone_preloaded=False,
            phone_resident_limit_bytes=1_000_000,
            ffn_column_quantum=32,
            cpu_executor_id="endpoint:cpu",
            cpu_endpoint="http://cpu.invalid:1",
            cpu_backend="llama-server-cpu",
            cpu_phone_executor_prefix="endpoint:cpu-phone-assisted",
            cpu_phone_endpoint="http://cpu-phone.invalid:1",
            cpu_phone_backend="llama-server-cpu-phone",
        )
        routes = model_composite_capabilities(model, hardware)
        desktop = next(
            row for row in routes
            if row.executor_id == model.desktop_executor_id
        )
        control = measured_desktop_control_profile(
            manifest,
            executor_id=desktop.executor_id,
            operator_placements=desktop.operator_placements,
            evidence_ids=model.desktop_evidence_ids,
        )
        transitions = model_transition_capabilities(
            model,
            hardware,
            routes,
            latency_us=10,
            energy_uj=10,
        )
        executors = base_executor_capabilities(
            hardware, evidence_id=EVIDENCE
        )
        resources = resource_profiles_for_catalog(
            hardware, routes, ("synthetic-link",)
        )

        self.assertEqual(
            {row.route_family for row in routes},
            {"layer_placement", "operator_offload", "operator_split"},
        )
        self.assertEqual(control.executor_id, desktop.executor_id)
        self.assertIn(
            hardware.gpu_device_id,
            {
                row.primary_device_id
                for row in control.operator_placements
            },
        )
        self.assertEqual(
            RuntimeDesktopControlProfile.from_json(control.to_json()),
            control,
        )
        self.assertEqual(
            {row.device_id for row in executors},
            {"discovered-cpu", "discovered-gpu", "discovered-phone"},
        )
        gpu = next(row for row in executors if row.backend == "gpu")
        self.assertEqual(
            gpu.exclusive_residency_resource_id,
            "discovered-gpu-residency",
        )
        split = next(
            row for row in routes
            if row.route_family == "operator_split"
            and hardware.gpu_device_id in row.participant_device_ids
        )
        self.assertEqual(
            split.split_fractions_ppm,
            (250_000, 500_000, 750_000),
        )
        self.assertEqual(
            set(split.participant_device_ids),
            {"discovered-cpu", "discovered-gpu", "discovered-phone"},
        )
        self.assertIn("link:synthetic-link", resources)
        self.assertIn("discovered-functionfs", resources)
        cpu_phone = next(
            row for row in routes
            if row.executor_id
                == "endpoint:cpu-phone-assisted:operator_split"
        )
        self.assertEqual(
            set(cpu_phone.participant_device_ids),
            {"discovered-cpu", "discovered-phone"},
        )
        self.assertNotIn(
            "discovered-gpu-residency", cpu_phone.resource_ids
        )
        for assisted in (
            row for row in routes
            if row.route_family in {
                "operator_offload", "operator_split"
            }
        ):
            parent = next(
                row for row in routes
                if row.executor_id == assisted.baseline_executor_id
            )
            self.assertEqual(parent.maturity, "QUALIFIED")
            self.assertEqual(
                parent.artifact_sha256, assisted.artifact_sha256
            )
        cpu_transition = next(
            row for row in transitions
            if row.executor_id == cpu_phone.executor_id
            and row.source_state == "cold"
        )
        self.assertEqual(cpu_transition.device_id, "discovered-cpu")
        self.assertEqual(
            set(cpu_transition.prepares_device_ids),
            {"discovered-cpu", "discovered-phone"},
        )
        self.assertNotIn(
            "discovered-gpu-residency", cpu_transition.resource_ids
        )
        self.assertEqual(cpu_transition.maturity, "QUALIFIED")
        self.assertEqual(
            cpu_phone.adapter_parameters[
                "requires_measured_route_profile"
            ],
            1,
        )
        self.assertEqual(
            cpu_phone.adapter_parameters[
                "ffn_runtime_control_protocol"
            ],
            "decode-boundary-v1",
        )

    def test_batch_plans_become_distinct_physical_capabilities(self) -> None:
        manifest = self.manifest()
        hardware = topology("batch")
        model = RuntimeModelEndpointCapability(
            manifest=manifest,
            desktop_executor_id="endpoint:batch-desktop",
            desktop_endpoint="http://desktop.invalid:1",
            desktop_backend="llama-server",
            phone_executor_prefix="endpoint:batch-phone",
            phone_endpoint="http://phone.invalid:1",
            phone_backend="llama-server-phone",
            desktop_gpu_first_layer=1,
            adapter_parameters={"parallel": 4},
            phone_adapter_parameters={"ffn_activation": "geglu"},
            desktop_evidence_ids=(EVIDENCE,),
            phone_evidence_ids=(EVIDENCE,),
            phone_preloaded=True,
            phone_resident_limit_bytes=1_000_000,
            phone_batch_plans=("split-row", "coalesced-batch"),
            qualified_phone_batch_plans=("split-row",),
        )
        for batch_plans, qualified, override in (
            (("split-row", "coalesced-batch"), ("split-row",), None),
            (("split-row", "coalesced-batch"), ("split-row",), "coalesced-batch"),
            (("coalesced-batch",), ("coalesced-batch",), None),
            (("split-row",), (), None),
        ):
            with self.subTest(batch_plans=batch_plans, qualified=qualified, override=override):
                configured = replace(
                    model,
                    phone_batch_plans=batch_plans,
                    qualified_phone_batch_plans=qualified,
                    phone_adapter_parameters={
                        **model.phone_adapter_parameters,
                        **({"usb_batch_plan": override} if override else {}),
                    },
                    cpu_executor_id="endpoint:batch-cpu",
                    cpu_endpoint="http://cpu.invalid:1",
                    cpu_backend="llama-server-cpu",
                    cpu_phone_executor_prefix="endpoint:batch-cpu-phone",
                    cpu_phone_endpoint="http://cpu-phone.invalid:1",
                    cpu_phone_backend="llama-server-cpu-phone",
                )
                routes = model_composite_capabilities(configured, hardware)
                resources = resource_profiles_for_catalog(hardware, routes, ())
                transitions = model_transition_capabilities(
                    configured, hardware, routes, latency_us=1, energy_uj=1,
                )
                for route in routes:
                    self.assertLessEqual(set(route.resource_ids), set(resources))
                    self.assertTrue(any(row.executor_id == route.executor_id for row in transitions))
                    for resource_id in route.resource_ids:
                        if resource_id.startswith("coordinator:"):
                            self.assertEqual(resources[resource_id].capacity, 4)
                for prefix in ("endpoint:batch-phone", "endpoint:batch-cpu-phone"):
                    for family in ("operator_split", "operator_offload"):
                        base_id = prefix + ":" + family
                        helpers = [row for row in routes if row.executor_id.startswith(base_id)]
                        self.assertEqual(len(helpers), len(batch_plans))
                        for batch_plan in batch_plans:
                            matches = [row for row in helpers
                                       if row.adapter_parameters["usb_batch_plan"] == batch_plan]
                            self.assertEqual(len(matches), 1)
                            helper = matches[0]
                            self.assertEqual(helper.executor_id, base_id + (
                                "" if batch_plan == "split-row" else ":" + batch_plan))
                            self.assertEqual(helper.maturity,
                                             "QUALIFIED" if batch_plan in qualified else "SHADOW")

    def test_unseen_model_gets_capacity_derived_desktop_binding(self) -> None:
        manifest = self.manifest()
        hardware = topology("portable")
        first_layer = derived_desktop_gpu_first_layer(
            manifest,
            gpu_capacity_bytes=manifest.tensor_bytes * 4,
            gpu_reserve_bytes=manifest.tensor_bytes,
        )
        route = desktop_model_composite_capability(
            manifest,
            hardware,
            executor_id="endpoint:portable-desktop",
            endpoint="http://portable.invalid:1",
            backend="llama-server-cuda-cpu",
            gpu_first_layer=first_layer,
            adapter_parameters={
                "batch_size": 32,
                "context_size": 128,
                "model_alias": "portable-model",
                "parallel": 2,
                "ubatch_size": 16,
            },
            evidence_ids=(EVIDENCE,),
        )

        self.assertEqual(first_layer, 0)
        self.assertEqual(route.artifact_sha256, manifest.artifact_sha256)
        self.assertEqual(route.adapter_parameters["gpu_layers"], 2)
        self.assertEqual(
            route.adapter_parameters["request_memory_mode"],
            "preallocated",
        )
        context_resource_id = route.adapter_parameters[
            "context_resource_id"
        ]
        self.assertEqual(
            route.adapter_parameters["context_token_quantum"], 16
        )
        self.assertIn(context_resource_id, route.resource_ids)
        self.assertIn(
            context_resource_id,
            route.participant_resource_ids["portable-cpu"],
        )
        resources = resource_profiles_for_catalog(
            hardware, (route,), ()
        )
        self.assertEqual(resources[context_resource_id].capacity, 8)
        self.assertEqual(
            resources[context_resource_id].kind, "memory-pool"
        )
        self.assertEqual(
            set(route.participant_device_ids),
            {"portable-cpu", "portable-gpu"},
        )
        self.assertEqual(
            {
                row.primary_device_id
                for row in route.operator_placements
                if row.operator_id.startswith("layer:")
            },
            {"portable-gpu"},
        )

    def test_whole_model_endpoint_adds_token_transport_and_transitions(
        self,
    ) -> None:
        try:
            from .test_automated_runtime import (
                catalog,
                catalog_with_gpu_desktop_control,
                request,
                runtime_snapshot,
            )
        except ImportError:
            from test_automated_runtime import (
                catalog,
                catalog_with_gpu_desktop_control,
                request,
                runtime_snapshot,
            )

        manifest = self.manifest()
        source_value = catalog().to_json()
        for kernel in source_value["placement_profile"]["kernels"]:
            if kernel["device_id"] == "helper-c":
                kernel["status"] = "estimated"
        source = RuntimeCapabilityCatalog.from_json(source_value)
        updated = materialize_whole_model_endpoint(
            source,
            manifest,
            executor_id="executor:helper-c",
            endpoint="http://127.0.0.1:18382",
            backend="android-llama-server-opencl",
            adapter_parameters={
                "batch_size": 32,
                "context_size": 128,
                "cpu_device_id": "host-a",
                "executable_device": "GPUOpenCL",
                "execution_adapter": "android-llama-server-v1",
                "forward_port": 18382,
                "gpu_device_id": "helper-c",
                "model_alias": manifest.model_id,
                "parallel": 1,
                "persistent_residency": 1,
                "remote_library_directory": "/data/local/tmp/lib",
                "remote_model_path": "/data/local/tmp/model.gguf",
                "remote_port": 18382,
                "remote_server_path": "/data/local/tmp/llama-server",
                "remote_server_sha256": "sha256:" + "2" * 64,
                "request_io_protocol": "token-ids-v1",
                "token_id_bytes": 4,
                "ubatch_size": 16,
                "whole_model_power_prior_mw": 5_000,
            },
            evidence_ids=(EVIDENCE,),
            transition_latency_us=1_000,
            transition_energy_uj=5_000,
            transition_energy_maturity="QUALIFIED",
            request_transport_identity_sha256="sha256:" + "3" * 64,
        )
        restored = RuntimeCapabilityCatalog.from_json(updated.to_json())
        validated = catalog_with_gpu_desktop_control(manifest, restored)
        validate_final_catalog(
            validated,
            (),
            manifest,
            require_whole_phone=True,
        )
        phone = restored.executor_by_id["executor:helper-c"]
        token_links = tuple(
            row for row in restored.placement_profile.links
            if row.transport_generation == "adb-token-http-v1"
        )

        self.assertEqual(restored, updated)
        self.assertEqual(phone.endpoint, "http://127.0.0.1:18382")
        self.assertEqual(
            phone.adapter_parameters["request_io_protocol"],
            "token-ids-v1",
        )
        self.assertEqual(len(token_links), 2)
        self.assertEqual(
            {(row.source_device, row.target_device) for row in token_links},
            {("host-a", "helper-c"), ("helper-c", "host-a")},
        )
        self.assertTrue(all(
            "link:" + row.link_id in restored.resources
            for row in token_links
        ))
        self.assertEqual(
            {
                row.source_state for row in restored.transitions
                if row.executor_id == "executor:helper-c"
                and row.artifact_sha256 == manifest.artifact_sha256
            },
            {"cold", "hot", "warm"},
        )
        self.assertTrue(all(
            row.kernel.active_power_mw >= 5_000
            for row in restored.placement_profile.kernels.values()
            if row.device_id == "helper-c" and row.status == "estimated"
        ))

        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(restored)
        scheduler.register_model_manifest(manifest)
        snapshot = runtime_snapshot(manifest)
        without_phone_residency = replace(
            snapshot,
            residency=tuple(
                row for row in snapshot.residency
                if row.device_id != "helper-c"
            ),
        )
        shadow_catalog = replace(
            restored,
            transitions=tuple(
                replace(row, energy_maturity="SHADOW")
                if row.executor_id == "executor:helper-c"
                and row.artifact_sha256 == manifest.artifact_sha256
                else row
                for row in restored.transitions
            ),
        )
        shadow_scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        shadow_scheduler.register_runtime_capabilities(shadow_catalog)
        shadow_scheduler.register_model_manifest(manifest)
        self.assertEqual(
            dict(shadow_scheduler
                ._persistent_phone_service_reserve_by_artifact(
                    "helper-c", without_phone_residency
                )),
            {},
        )
        self.assertEqual(
            dict(scheduler._persistent_phone_service_reserve_by_artifact(
                "helper-c", without_phone_residency
            )),
            {manifest.artifact_sha256: manifest.tensor_bytes},
        )
        self.assertEqual(
            dict(scheduler._persistent_phone_service_reserve_by_artifact(
                "helper-c", snapshot
            )),
            {},
        )
        live_links = dict(snapshot.links)
        for link in token_links:
            live_links[link.link_id] = RuntimeLinkState(
                link_id=link.link_id,
                ready=True,
                measured_bandwidth_bytes_per_s=(
                    link.bandwidth_bytes_per_s
                ),
                busy_until_us=0,
            )
        candidates = scheduler.generate_automated_candidates(
            request("whole-phone-token-boundary"),
            manifest.model_id,
            replace(snapshot, links=live_links),
        )
        route = next(
            row for row in candidates.candidates
            if row.route_family == "whole_model"
            and row.device_ids == ("helper-c",)
            and row.residency_variant == "hot"
        )
        self.assertEqual(
            route.plan.adapter_parameters["request_transport"],
            "adb-token-http-v1",
        )
        self.assertEqual(
            {
                resource_id for resource_id in route.plan.resource_ids
                if "token-rpc" in resource_id
            },
            {"link:" + row.link_id for row in token_links},
        )
        self.assertTrue(
            set(source.executor_by_id[
                "executor:helper-c"
            ].execution_resource_ids).issubset(route.plan.resource_ids)
        )
        self.assertLess(route.cost.transfer_us, 10_000)
        self.assertEqual(route.binding.endpoint, phone.endpoint)


if __name__ == "__main__":
    unittest.main()
