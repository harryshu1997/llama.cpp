"""Replay catalog identity checks from archived inputs without using the rig."""

from dataclasses import replace
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

from research_dev.scheduler import ModelManifest, RuntimeCapabilityCatalog
from research_dev.scheduler.adapters import (
    model_composite_capabilities,
    resource_profiles_for_catalog,
)
from research_dev.scheduler.campaigns.burstgpt.catalog import _model_endpoint, _physical_topology
from research_dev.scheduler.configuration.models import CampaignModelConfiguration
from research_dev.scheduler.configuration.rig import RigManifest


ROOT = Path(__file__).resolve().parent
REPORT = ROOT.parent.parent
CHECK = runpy.run_path(str(REPORT / "CHECK_TASK1_ADMISSION.py"))["validate_helper_identities"]


def read(path):
    return json.loads(path.read_text())


def replay(arm):
    inputs = REPORT / "physical" / arm
    catalog = RuntimeCapabilityCatalog.from_json(read(inputs / "run-treatment-1/UNIFIED_RUNTIME_CATALOG.json"))
    groups = {}
    for row in catalog.composite_executors:
        if row.executor_id.startswith("physical:hot:") and row.route_family == "operator_split":
            groups.setdefault(row.baseline_executor_id, []).append(row)
    duplicates = []
    for parent, rows in sorted(groups.items()):
        assert len(rows) == 2
        parameters = [row.to_json() for row in rows]
        for value in parameters:
            value.pop("executor_id")
        assert parameters[0] == parameters[1]
        duplicates.append({"desktop_parent": parent,
                           "executor_ids": [row.executor_id for row in rows],
                           "usb_batch_plans": [row.adapter_parameters["usb_batch_plan"] for row in rows],
                           "all_other_capability_fields_equal": True})
    try:
        CHECK(catalog)
    except RuntimeError as exc:
        rejected = str(exc)
    else:
        raise AssertionError("old catalog was not rejected")
    rig = RigManifest.from_json(read(inputs / "rig.json"), inputs)
    raw_model = next(row for row in read(inputs / "models.json")["models"] if row["model_key"] == "hot")
    manifest = ModelManifest.from_json(read(REPORT.parent.parent / "data/QWEN_MANIFEST.json"))
    model = CampaignModelConfiguration.from_json(raw_model, inputs)
    parent = catalog.composite_executor_by_id["physical:hot:desktop"]
    phone = catalog.composite_executor_by_id["physical:hot:phone-assisted:operator_split:coalesced-batch"]
    hardware = _physical_topology(rig, rig.topology,
                                 catalog.executor_by_device[rig.topology.phone_device_id].phone_sessions)
    plans = {manifest.artifact_sha256: {
        "gpu_first_layer": manifest.block_count - parent.adapter_parameters["gpu_layers"],
        "adapter_parameters": dict(parent.adapter_parameters), "evidence_ids": parent.evidence_ids,
    }}
    endpoint = _model_endpoint(
        "hot", manifest, model, rig.endpoints, plans,
        {"desktop": parent.evidence_ids, "phone": phone.evidence_ids},
        SimpleNamespace(desktop_calibration_arm="desktop", assisted_calibration_arm="phone"),
    )
    endpoint = replace(
        endpoint, phone_batch_plans=("coalesced-batch",),
        qualified_phone_batch_plans=("coalesced-batch",),
        phone_adapter_parameters={k: v for k, v in endpoint.phone_adapter_parameters.items()
                                  if k != "usb_batch_plan"},
    )
    composites = model_composite_capabilities(endpoint, hardware)
    assert all(row == catalog.composite_executor_by_id[row.executor_id] for row in composites)
    removed = {row.executor_id for row in catalog.composite_executors
               if row.executor_id.startswith("physical:hot:")} - {row.executor_id for row in composites}
    resources = resource_profiles_for_catalog(hardware, composites, ())
    assert all(set(row.resource_ids) <= set(resources) for row in composites)
    updated = replace(
        catalog,
        composite_executors=tuple(row for row in catalog.composite_executors
                                  if not row.executor_id.startswith("physical:hot:")) + composites,
        transitions=tuple(row for row in catalog.transitions if row.executor_id not in removed),
        route_shape_profiles=tuple(row for row in catalog.route_shape_profiles if row.executor_id not in removed),
    )
    identities = CHECK(updated)
    assert RuntimeCapabilityCatalog.from_json(updated.to_json()) == updated
    return {"arm": arm, "status": "PASS", "original_duplicate_identities": duplicates,
            "original_catalog_guard": {"status": "FAIL", "expected": True, "reason": rejected},
            "corrected_catalog_guard": {"status": "PASS", "helper_identities": identities},
            "remaining_capabilities_exactly_match_archive": True,
            "all_resource_references_present": True, "catalog_roundtrip": "PASS",
            "physical_execution": False}


if __name__ == "__main__":
    results = [replay(arm) for arm in ("m4a6-r2", "m4a7")]
    (ROOT / "REPLAY.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, sort_keys=True))
