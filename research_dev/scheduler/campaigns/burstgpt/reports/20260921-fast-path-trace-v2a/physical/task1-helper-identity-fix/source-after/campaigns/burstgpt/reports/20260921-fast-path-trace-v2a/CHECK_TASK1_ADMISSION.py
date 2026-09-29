"""Check coalesced Qwen capacity against a freshly materialized catalog."""

import json
from pathlib import Path
import sys

from research_dev.scheduler import RuntimeCapabilityCatalog
from research_dev.scheduler._internal.route_generation import AutomatedRouteCompiler


def validate_helper_identities(catalog):
    by_parent = {}
    for row in catalog.composite_executors:
        if (row.executor_id.startswith("physical:hot:")
                and row.route_family == "operator_split"):
            key = (row.artifact_sha256, row.baseline_executor_id)
            by_parent.setdefault(key, []).append(row)
    if not by_parent:
        raise RuntimeError("Qwen helper capabilities are absent")
    identities = []
    for (artifact, parent), helpers in sorted(by_parent.items()):
        if len(helpers) != 1:
            raise RuntimeError("Qwen parent has multiple helper identities: " + parent)
        helper = helpers[0]
        if (helper.adapter_parameters.get("usb_batch_plan") != "coalesced-batch"
                or helper.maturity != "QUALIFIED"):
            raise RuntimeError("Qwen helper is not qualified for coalesced batching")
        identities.append({"artifact_sha256": artifact, "desktop_parent": parent,
                           "helper_executor_id": helper.executor_id,
                           "usb_batch_plan": "coalesced-batch"})
    return identities


def main(inputs):
    catalog = RuntimeCapabilityCatalog.from_json(json.loads(
        (inputs / "preflight-1/UNIFIED_RUNTIME_CATALOG.json").read_text()))
    helper_identities = validate_helper_identities(catalog)
    profile = catalog.placement_profile
    links = tuple(row.link_id for row in profile.links
                  if row.maximum_payload_bytes == 10240
                  and profile.devices[row.source_device].kind.startswith("phone")
                      != profile.devices[row.target_device].kind.startswith("phone"))
    if len(links) != 2:
        raise RuntimeError("expected one measured cost link in each direction")
    parameters = dict(AutomatedRouteCompiler._transport_adapter_parameters(
        profile, links, required_maximum_payload_bytes=40960,
        batch_plan="coalesced-batch", maximum_tokens=4))
    if parameters["usb_max_payload_bytes"] != 40960 or parameters["usb_queue_depth"] != 4:
        raise RuntimeError("coalesced capacity differs")
    identity = json.loads((inputs / "TRANSPORT_QUALIFICATION_IDENTITY.json").read_text())
    rig = json.loads((inputs / "rig.json").read_text())
    if rig["phone"]["boot_image_sha256"] != identity["hardware_identity"]["phone_boot_image_sha256"]:
        raise RuntimeError("rig boot identity differs")
    result = {"status": "PASS", "rig_identity_matches": True, "parameters": parameters,
              "helper_identities": helper_identities}
    with (inputs / "TRANSPORT_ADMISSION.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
