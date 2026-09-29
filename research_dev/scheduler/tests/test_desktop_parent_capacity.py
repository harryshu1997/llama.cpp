#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import unittest

from research_dev.scheduler import (
    DeviceMemoryCapacity,
    ModelManifest,
    select_desktop_parent_for_live_vram,
)
from research_dev.scheduler.adapters import (
    RuntimeModelEndpointCapability,
    RuntimePhysicalTopology,
    desktop_model_composite_capability,
    model_composite_capabilities,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = "sha256:" + "1" * 64


def _topology() -> RuntimePhysicalTopology:
    return RuntimePhysicalTopology(
        cpu_device_id="desktop-cpu",
        gpu_device_id="desktop-cuda",
        phone_device_id="op15-phone",
        cpu_resource_id="desktop-cpu",
        gpu_resource_id="cuda0",
        host_memory_resource_id="host-ram",
        gpu_memory_resource_id="cuda0-vram",
        phone_memory_resource_id="op15-ram",
        functionfs_resource_id="op15-functionfs",
        phone_transport_resource_ids=("op15-functionfs",),
        phone_compute_resource_ids=("op15-htp",),
        gpu_exclusive_residency_resource_id="cuda0",
        resource_capacities={
            "desktop-cpu": 4,
            "cuda0": 1,
            "op15-functionfs": 1,
            "op15-htp": 1,
        },
        resource_identities={},
    )


class DesktopParentCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = ModelManifest.from_json(json.loads(
            (ROOT / "campaigns/burstgpt/data/GEMMA_MANIFEST.json")
            .read_text(encoding="ascii")
        ))
        cls.parameters = {
            "batch_size": 4096,
            "context_size": 32768,
            "kv_cache_swa_padding_tokens": 512,
            "memory_model_weight_allocation_ppm:desktop-cpu": 1050000,
            "memory_model_weight_allocation_ppm:desktop-cuda": 1150000,
            "memory_workspace_minimum_bytes:desktop-cpu": 268435456,
            "memory_workspace_minimum_bytes:desktop-cuda": 268435456,
            "model_alias": "gemma-4-12b-it-q4_0",
            "parallel": 2,
            "ubatch_size": 512,
        }

    def _source(self):
        return desktop_model_composite_capability(
            self.manifest,
            _topology(),
            executor_id="physical:cold:desktop",
            endpoint="http://127.0.0.1:18573",
            backend="llama-server-cuda-cpu",
            gpu_first_layer=24,
            adapter_parameters=self.parameters,
            evidence_ids=(EVIDENCE,),
        )

    def test_selects_largest_live_vram_feasible_suffix(self) -> None:
        capacity = DeviceMemoryCapacity(
            "cuda0-vram",
            16380 * 1024**2,
            (16380 - 12770) * 1024**2,
            512 * 1024**2,
        )

        selection = select_desktop_parent_for_live_vram(
            self.manifest,
            self._source(),
            capacity,
            memory_snapshot_id="nvml-test",
        )

        self.assertEqual(selection.maximum_gpu_layers, 24)
        self.assertEqual(selection.selected.gpu_layers, 22)
        self.assertEqual(selection.selected.gpu_first_layer, 26)
        self.assertEqual(
            selection.selected.required_with_reserve_bytes,
            13_232_818_996,
        )
        self.assertLessEqual(
            selection.selected.required_with_reserve_bytes,
            selection.selected.live_free_vram_bytes,
        )
        twenty_three = next(
            row for row in selection.candidates if row.gpu_layers == 23
        )
        self.assertFalse(twenty_three.feasible)

    def test_missing_gpu_weight_factor_uses_conservative_default(self) -> None:
        parameters = dict(self.parameters)
        parameters.pop(
            "memory_model_weight_allocation_ppm:desktop-cuda"
        )
        source = desktop_model_composite_capability(
            self.manifest,
            _topology(),
            executor_id="physical:cold:desktop",
            endpoint="http://127.0.0.1:18573",
            backend="llama-server-cuda-cpu",
            gpu_first_layer=24,
            adapter_parameters=parameters,
            evidence_ids=(EVIDENCE,),
        )
        capacity = DeviceMemoryCapacity(
            "cuda0-vram",
            16380 * 1024**2,
            (16380 - 12770) * 1024**2,
            512 * 1024**2,
        )

        selected = select_desktop_parent_for_live_vram(
            self.manifest,
            source,
            capacity,
            memory_snapshot_id="nvml-test",
        ).selected

        self.assertEqual(selected.gpu_weight_allocation_ppm, 1_150_000)
        self.assertEqual(
            selected.adapter_parameters[
                "memory_model_weight_allocation_ppm:desktop-cuda"
            ],
            1_150_000,
        )

    def test_assisted_routes_preserve_selected_desktop_parent(self) -> None:
        capacity = DeviceMemoryCapacity(
            "cuda0-vram",
            16380 * 1024**2,
            (16380 - 12770) * 1024**2,
            512 * 1024**2,
        )
        selected = select_desktop_parent_for_live_vram(
            self.manifest,
            self._source(),
            capacity,
            memory_snapshot_id="nvml-test",
        ).selected
        model = RuntimeModelEndpointCapability(
            manifest=self.manifest,
            desktop_executor_id="physical:cold:desktop",
            desktop_endpoint="http://127.0.0.1:18573",
            desktop_backend="llama-server-cuda-cpu",
            phone_executor_prefix="physical:cold:phone-assisted",
            phone_endpoint="http://127.0.0.1:18574",
            phone_backend="llama-server-cuda-cpu-op15",
            desktop_gpu_first_layer=selected.gpu_first_layer,
            adapter_parameters=self.parameters,
            desktop_evidence_ids=(EVIDENCE,),
            phone_evidence_ids=(EVIDENCE,),
            phone_preloaded=False,
            phone_resident_limit_bytes=3_208_646_656,
            phone_adapter_parameters={"ffn_activation": "geglu"},
        )
        routes = model_composite_capabilities(model, _topology())
        desktop = next(
            row for row in routes if row.executor_id == model.desktop_executor_id
        )
        desktop_by_id = {
            row.operator_id: row for row in desktop.operator_placements
        }

        for assisted in (
            row for row in routes if row.helper_device_id is not None
        ):
            self.assertEqual(
                assisted.baseline_executor_id, desktop.executor_id
            )
            self.assertEqual(
                assisted.adapter_parameters["gpu_layers"], 22
            )
            for placement in assisted.operator_placements:
                self.assertEqual(
                    placement.primary_device_id,
                    desktop_by_id[placement.operator_id].primary_device_id,
                )


if __name__ == "__main__":
    unittest.main()
