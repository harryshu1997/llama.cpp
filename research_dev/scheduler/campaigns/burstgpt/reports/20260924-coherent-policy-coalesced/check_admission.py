#!/usr/bin/env python3
"""Check a preflight catalog: one qualified helper per Qwen parent, coalesced for Qwen, split-row for Gemma,
and coalesced transport capacity 40,960 bytes at queue depth 4 (4 rows x 5,120 x 2 bytes)."""
import json
from pathlib import Path
import sys

from research_dev.scheduler import RuntimeCapabilityCatalog
from research_dev.scheduler._internal.route_generation import AutomatedRouteCompiler


def helper_identities(catalog, prefix, plan):
    by_parent = {}
    for row in catalog.composite_executors:
        if row.executor_id.startswith(prefix) and row.route_family == "operator_split":
            by_parent.setdefault((row.artifact_sha256, row.baseline_executor_id), []).append(row)
    if not by_parent:
        raise RuntimeError(prefix + " helper capabilities are absent")
    identities = []
    for (artifact, parent), helpers in sorted(by_parent.items()):
        qualified = [row for row in helpers if row.maturity == "QUALIFIED"]
        if len(qualified) != 1 or qualified[0].adapter_parameters.get("usb_batch_plan") != plan:
            raise RuntimeError(f"{prefix} parent {parent}: qualified helpers "
                               f"{[(r.executor_id, r.adapter_parameters.get('usb_batch_plan')) for r in qualified]}"
                               f" != one {plan}")
        shadows = [row for row in helpers if row.maturity != "QUALIFIED"]
        identities.append({"artifact_sha256": artifact, "desktop_parent": parent,
                           "qualified_executor_id": qualified[0].executor_id, "usb_batch_plan": plan,
                           "shadow_executor_ids": [row.executor_id for row in shadows],
                           "declared_plans": sorted({row.adapter_parameters.get("usb_batch_plan") for row in helpers})})
    return identities


def main(inputs, preflight, qwen_plan):
    catalog = RuntimeCapabilityCatalog.from_json(json.loads((preflight / "UNIFIED_RUNTIME_CATALOG.json").read_text()))
    result = {"qwen": helper_identities(catalog, "physical:hot:", qwen_plan),
              "gemma": helper_identities(catalog, "physical:cold:", "split-row")}
    profile = catalog.placement_profile
    links = tuple(row.link_id for row in profile.links
                  if row.maximum_payload_bytes == 10240
                  and profile.devices[row.source_device].kind.startswith("phone")
                      != profile.devices[row.target_device].kind.startswith("phone"))
    if len(links) != 2:
        raise RuntimeError("expected one measured cost link in each direction")
    if qwen_plan == "coalesced-batch":
        parameters = dict(AutomatedRouteCompiler._transport_adapter_parameters(
            profile, links, required_maximum_payload_bytes=40960, batch_plan="coalesced-batch", maximum_tokens=4))
        if parameters["usb_max_payload_bytes"] != 40960 or parameters["usb_queue_depth"] != 4:
            raise RuntimeError("coalesced capacity differs: " + json.dumps(parameters, sort_keys=True))
        result["coalesced_transport"] = parameters
    identity = json.loads((inputs / "TRANSPORT_QUALIFICATION_IDENTITY.json").read_text())
    rig = json.loads((inputs / "rig.json").read_text())
    if rig["phone"]["boot_image_sha256"] != identity["hardware_identity"]["phone_boot_image_sha256"]:
        raise RuntimeError("rig boot identity differs from the transport identity")
    result["status"] = "PASS"
    out = inputs / ("TRANSPORT_ADMISSION-" + preflight.name + ".json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
