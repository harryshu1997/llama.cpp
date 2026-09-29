"""Synthetic OP15 + Pixel rig for the two-phone gap tests: catalog built by the materialization code.

Eight-layer model, 128 FFN columns. Desktop parent: layers 0-5 on the CPU, 6-7 on the GPU. The
primary phone (``op15-phone``, FunctionFS) may assist layers 0-3 (its resident limit fits four
layers); the co-helper (``pixel-phone``, adb-tcp) statically owns CPU layers 4-5. Quanta mirror the
real rig: primary 16 columns (1/8 like OP15's 2176 of 17408), co-helper 32 (1/4 like the Pixel's
4352), so a union policy must use widths on the 32-column grid.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile

from research_dev.scheduler import (
    DeviceMemoryCapacity,
    GGUFModelManifestLoader,
    HeterogeneousRuntimeSnapshot,
    ModelResidencyObservation,
    PlacementHardwareProfile,
    Request,
    RuntimeCapabilityCatalog,
    RuntimeExecutorState,
    RuntimeLinkState,
    RuntimePlacementSnapshot,
)
from research_dev.scheduler.adapters import (
    RuntimeModelEndpointCapability,
    RuntimePhysicalTopology,
    base_executor_capabilities,
    measured_desktop_control_profile,
    model_composite_capabilities,
    model_transition_capabilities,
    resource_profiles_for_catalog,
)

try:
    from .test_gguf_cost import write_synthetic_gguf
except ImportError:
    from test_gguf_cost import write_synthetic_gguf


EVIDENCE = "sha256:" + "1" * 64
SHARD_SHA = "sha256:" + "d" * 64
CPU, GPU, OP15, PIXEL = "desk-cpu", "desk-gpu", "op15-phone", "pixel-phone"
OP15_SERIAL, PIXEL_SERIAL = "3C15AU002CL00000", "5A040DLCH004ES"
PIXEL_LAYERS = (4, 5)
PIXEL_MASK = sum(1 << value for value in PIXEL_LAYERS)
FUNCTIONFS_PARAMETERS = {
    "ffn_transport": "functionfs-usb", "usb_allocator": "devmem", "usb_batch_plan": "split-row",
    "usb_full_duplex": 1, "usb_max_payload_bytes": 40960, "usb_product_id": 0x2D00,
    "usb_queue_depth": 4, "usb_concurrent_streams": 4, "usb_slot_safety_bytes": 65536,
    "usb_split_h2d": 0, "usb_transport_generation": "functionfs-dmabuf-async-ring-v2",
    "usb_transport_profile_id": "op15-profile", "usb_vendor_id": 0x18D1, "usbfs_available_bytes": 1 << 30,
}
PIXEL_TRANSPORT = {
    "adb_port": 5037, "adb_serial": PIXEL_SERIAL, "ffn_transport": "adb-tcp",
    "ffn_worker_host": "127.0.0.1", "ffn_worker_port": 26991, "phone_worker_port": 26990,
}
_KINDS = ("attention", "attention_projection", "embedding", "ffn", "kv_cache", "lm_head")


def manifest(directory: str, block_count: int = 8):
    path = Path(directory) / "two-phone.gguf"
    write_synthetic_gguf(path, block_count=block_count)
    return GGUFModelManifestLoader.load("two-phone-model", path)


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def phone_helpers_json(primary_mask: int, pixel_mask: int = PIXEL_MASK, forward_port: int = 26991) -> str:
    """The ``phone_helpers`` launch binding, spelled out (the format of ``PhoneHelperBinding``)."""
    return _canonical([
        {"device_id": OP15, "label": "op15", "layer_mask": primary_mask, "serial": OP15_SERIAL,
         "transport_parameters": {}},
        {"device_id": PIXEL, "label": "pixel", "layer_mask": pixel_mask, "serial": PIXEL_SERIAL,
         "transport_parameters": {**PIXEL_TRANSPORT, "ffn_worker_port": forward_port}},
    ])


def co_helpers_json() -> str:
    """The ``phone_co_helpers_v1`` declaration of :func:`co_helpers`, spelled out."""
    return _canonical({
        "helpers": [{
            "column_quantum": 32, "device_id": PIXEL, "label": "pixel", "layer_mask": PIXEL_MASK,
            "max_tokens": 4, "resident_bytes": 1 << 20, "serial": PIXEL_SERIAL, "session_id": "PIXEL0",
            "shard_sha256": SHARD_SHA, "transport_parameters": PIXEL_TRANSPORT,
        }],
        "primary_label": "op15",
        "primary_serial": OP15_SERIAL,
        "schema": "research-scheduler-phone-co-helpers-v1",
    })


def co_helpers(
    *,
    column_quantum: int = 32,
    max_tokens: int = 4,
    layer_mask: int = PIXEL_MASK,
):
    from research_dev.scheduler._internal.plan_contracts.co_helpers import (
        RuntimeCoHelperDeclaration,
        RuntimeCoHelperPhone,
    )

    return RuntimeCoHelperDeclaration(
        primary_label="op15",
        primary_serial=OP15_SERIAL,
        helpers=(RuntimeCoHelperPhone(
            device_id=PIXEL, serial=PIXEL_SERIAL, label="pixel", session_id="PIXEL0",
            layer_mask=layer_mask, column_quantum=column_quantum, max_tokens=max_tokens,
            shard_sha256=SHARD_SHA, resident_bytes=1 << 20,
            transport_parameters=PIXEL_TRANSPORT,
        ),),
    )


def topology(*, with_pixel: bool = True) -> RuntimePhysicalTopology:
    helper_phones = {}
    if with_pixel:
        from research_dev.scheduler.adapters import RuntimeHelperPhoneTopology

        helper_phones["helper_phones"] = (RuntimeHelperPhoneTopology(
            device_id=PIXEL,
            memory_resource_id="pixel-ram",
            transport_resource_ids=("usb-root", "pixel-adb"),
            compute_resource_ids=("pixel-gpu",),
        ),)
    return RuntimePhysicalTopology(
        cpu_device_id=CPU,
        gpu_device_id=GPU,
        phone_device_id=OP15,
        cpu_resource_id="desk-cpu",
        gpu_resource_id="cuda0",
        host_memory_resource_id="host-ram",
        gpu_memory_resource_id="cuda-vram",
        phone_memory_resource_id="op15-ram",
        functionfs_resource_id="op15-functionfs",
        phone_transport_resource_ids=("usb-root", "op15-functionfs"),
        phone_compute_resource_ids=("op15-htp",),
        gpu_exclusive_residency_resource_id="cuda0",
        resource_capacities={
            "desk-cpu": 4, "cuda0": 8, "usb-root": 1, "op15-functionfs": 1,
            "op15-htp": 1, "pixel-adb": 1, "pixel-gpu": 1,
        },
        resource_identities={"usb-root": "usb-root"},
        phone_exclusive_residency_resource_id="op15-htp",
        **helper_phones,
    )


def placement_profile(*, with_pixel: bool = True) -> PlacementHardwareProfile:
    devices = [
        (CPU, "cpu", "host-ram", 30_000, 1_000_000_000, _KINDS),
        (GPU, "gpu", "cuda-vram", 55_000, 8_000_000_000, _KINDS),
        (OP15, "phone", "op15-ram", 4_000, 10_000_000_000, _KINDS),
    ]
    links = [
        (CPU, GPU, "pcie-out", 8_000_000_000), (GPU, CPU, "pcie-in", 8_000_000_000),
        (CPU, OP15, "usb-out", 8_000_000_000), (OP15, CPU, "usb-in", 8_000_000_000),
    ]
    if with_pixel:
        devices.append((PIXEL, "phone", "pixel-ram", 4_000, 10_000_000_000, ("ffn",)))
        links += [(CPU, PIXEL, "usb2-out", 8_000_000_000), (PIXEL, CPU, "usb2-in", 8_000_000_000)]
    return PlacementHardwareProfile.from_json({
        "devices": [
            {"allocation_limit_bytes": 2_000_000_000, "device_id": device_id, "kind": kind,
             "memory_pool_id": pool, "ready": True}
            for device_id, kind, pool, _, _, _ in devices
        ],
        "domains": [
            {"domain_id": "energy:" + device_id, "evidence_ids": ["domain:" + device_id],
             "idle_power_mw": 500, "status": "measured"}
            for device_id, *_ in devices
        ],
        "energy_boundary_id": "synthetic-two-phone-fleet",
        "idle_charge_domains": sorted("energy:" + row[0] for row in devices),
        "kernels": [
            {"active_power_mw": power, "device_id": device_id, "domain_id": "energy:" + device_id,
             "effective_bytes_per_s": rate, "effective_ops_per_s": rate,
             "evidence_ids": [f"kernel:{device_id}:{kind}"], "kernel_id": f"prior:{device_id}:{kind}",
             "launch_us": 2, "profile_id": f"prior:{device_id}:{kind}", "status": "measured"}
            for device_id, _, _, power, rate, kinds in devices
            for kind in kinds
        ],
        "links": [
            {"bandwidth_bytes_per_s": bandwidth, "domain_active_power_mw": {}, "dynamic_pj_per_byte": 50,
             "evidence_ids": ["link:" + name], "fixed_dynamic_uj": 10, "fixed_latency_us": 10,
             "link_id": name, "ready": True, "source_device": source, "status": "measured",
             "target_device": target}
            for source, target, name, bandwidth in links
        ],
        "memory_pools": [
            {"capacity_bytes": 2_000_000_000, "pool_id": pool, "reserved_bytes": 0}
            for _, _, pool, _, _, _ in devices
        ],
        "profile_id": "synthetic-two-phone-profile",
        "schema": "s42-placement-hardware-profile-v1",
    })


def layer_ffn_bytes(model, layer: int) -> int:
    return sum(
        model.tensor_by_id[tensor_id].nbytes
        for operator in model.operators
        if operator.kind == "ffn" and operator.layer_id == f"layer:{layer}"
        for tensor_id in operator.tensor_ids
    )


def model_endpoint(model, declaration) -> RuntimeModelEndpointCapability:
    return RuntimeModelEndpointCapability(
        manifest=model,
        desktop_executor_id="physical:two:desktop",
        desktop_endpoint="http://127.0.0.1:18571",
        desktop_backend="llama-server-cuda-cpu",
        phone_executor_prefix="physical:two:phone-assisted",
        phone_endpoint="http://127.0.0.1:18572",
        phone_backend="llama-server-cuda-cpu-op15",
        desktop_gpu_first_layer=6,
        adapter_parameters={
            "batch_size": 4, "context_size": 128, "model_alias": "two-phone", "parallel": 4,
            "ubatch_size": 4,
        },
        phone_adapter_parameters={
            **FUNCTIONFS_PARAMETERS,
            "ffn_activation": "swiglu", "ffn_host_share_release": 1, "ffn_timeout_ms": 1000,
            "ffn_weight_buffer_layout": "selected-width",
        },
        phone_runtime_control_protocol="decode-boundary-v1",
        desktop_evidence_ids=(EVIDENCE,),
        phone_evidence_ids=(EVIDENCE,),
        phone_preloaded=False,
        # the primary phone fits exactly layers 0-3 (the co-helper's 4-5 are never its candidates)
        phone_resident_limit_bytes=sum(layer_ffn_bytes(model, index) for index in range(4)),
        ffn_column_quantum=16,
        **({} if declaration is None else {"co_helpers": declaration}),
    )


def runtime_catalog(
    model,
    *,
    declaration=None,
    qualified_pixel: bool = True,
) -> RuntimeCapabilityCatalog:
    with_pixel = declaration is not None
    hardware = topology(with_pixel=with_pixel)
    profile = placement_profile(with_pixel=with_pixel)
    endpoint = model_endpoint(model, declaration)
    composites = model_composite_capabilities(endpoint, hardware)
    desktop = next(row for row in composites if row.executor_id == endpoint.desktop_executor_id)
    return RuntimeCapabilityCatalog(
        catalog_id="synthetic-two-phone",
        placement_profile=profile,
        resources=resource_profiles_for_catalog(
            hardware, composites, tuple(row.link_id for row in profile.links)
        ),
        executors=base_executor_capabilities(
            hardware, evidence_id=EVIDENCE,
            **({"qualified_helper_phone_ids": (PIXEL,) if qualified_pixel else ()}
               if with_pixel else {}),
        ),
        composite_executors=composites,
        desktop_control_profiles=(measured_desktop_control_profile(
            model, executor_id=desktop.executor_id,
            operator_placements=desktop.operator_placements, evidence_ids=(EVIDENCE,),
        ),),
        transitions=model_transition_capabilities(
            endpoint, hardware, composites, latency_us=100, energy_uj=100
        ),
        minimum_energy_saving_ppm=10_000,
        maximum_latency_ppm=10_000_000,
    )


def snapshot(
    model,
    catalog: RuntimeCapabilityCatalog,
    *,
    pixel_hot: bool = True,
    desktop_hot: bool = False,
) -> HeterogeneousRuntimeSnapshot:
    """A cold desktop by default: its launch then carries the dormant FFN runtime."""
    devices = sorted(
        device_id for device_id in catalog.placement_profile.devices
        if desktop_hot or device_id not in {CPU, GPU}
    )
    return HeterogeneousRuntimeSnapshot(
        snapshot_id="synthetic-two-phone-runtime",
        captured_at_us=0,
        valid_until_us=10_000_000,
        memory=RuntimePlacementSnapshot(
            snapshot_id="synthetic-two-phone-memory",
            captured_at_us=0,
            valid_until_us=10_000_000,
            capacities={
                pool: DeviceMemoryCapacity(pool, 2_000_000_000, 1_500_000_000, 10_000_000)
                for pool in catalog.placement_profile.memory_pools
            },
        ),
        executors={
            executor_id: RuntimeExecutorState(
                executor_id=executor_id, healthy=True, ready=True, temperature_millic=40_000,
                battery_ppm=900_000, free_slots=4, busy_until_us=0,
            )
            for executor_id in (
                *(row.executor_id for row in catalog.executors),
                *(row.executor_id for row in catalog.composite_executors),
            )
        },
        links={
            row.link_id: RuntimeLinkState(
                link_id=row.link_id, ready=True,
                measured_bandwidth_bytes_per_s=row.bandwidth_bytes_per_s, busy_until_us=0,
            )
            for row in catalog.placement_profile.links
        },
        residency=tuple(
            ModelResidencyObservation(
                model_id=model.model_id, artifact_sha256=model.artifact_sha256, device_id=device_id,
                state="hot", resident_tensor_ids=tuple(row.tensor_id for row in model.tensors),
                resident_bytes=model.tensor_bytes, generation=1,
            )
            for device_id in devices
            if device_id != PIXEL or pixel_hot
        ),
    )


def request(request_id: str, *, input_tokens: int = 8, output_tokens: int = 30) -> Request:
    return Request(
        request_id=request_id, workload_id="two-phone", arrival_us=1_000,
        deadline_us=1_000 + 50_000_000, input_tokens=input_tokens, output_tokens=output_tokens,
        quality_requirement="exact",
    )


class TemporaryModel:
    def __enter__(self):
        self.directory = tempfile.TemporaryDirectory()
        return manifest(self.directory.name)

    def __exit__(self, *_):
        self.directory.cleanup()


def with_residency_executor(value: HeterogeneousRuntimeSnapshot, executor_id: str) -> HeterogeneousRuntimeSnapshot:
    return replace(value, residency=tuple(
        replace(row, executor_id=executor_id) if row.device_id != PIXEL else row
        for row in value.residency
    ))
