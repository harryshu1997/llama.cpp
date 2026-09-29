#!/usr/bin/env python3
"""Admission check for coalesced multi-row phone FFN helpers of BOTH models against a resolved catalog.

For each model: exactly one QUALIFIED operator_split helper per desktop parent, its plan coalesced-batch,
and the route compiler's transport parameters for the model's decode payload (parallel x n_embd x 2 bytes,
depth 4) resolve to a measured capacity profile of the given payload bound to the given identity. Also
records the parallel-5 Gemma fallback (38,400 bytes) and, when --negative-catalog is given, that the same
61,440-byte request is refused against the old (40,960-byte) identity.

usage: check_admission_both.py <inputs-dir> <resolve-output-dir> [--negative-catalog PATH]
"""
import json
from pathlib import Path
import sys

from research_dev.scheduler import RuntimeCapabilityCatalog
from research_dev.scheduler._internal.route_generation import AutomatedRouteCompiler
from research_dev.scheduler._internal.route_generation.feasibility import RouteGenerationError

MODELS = {
    # model_key: (executor prefix, n_embd, parallel, ubatch)
    "hot": ("physical:hot:", 5120, 4, 512),
    "cold": ("physical:cold:", 3840, 8, 512),
}
SLOT_MULTIPLIER = 4  # models.json phone_adapter_parameters.ffn_transport_slot_payload_multiplier


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
                           "declared_plans": sorted({row.adapter_parameters.get("usb_batch_plan") for row in helpers}),
                           "qualified_adapter_usb_parameters": {
                               k: v for k, v in sorted(qualified[0].adapter_parameters.items())
                               if k.startswith("usb_") or k in ("ffn_max_tokens", "parallel", "ffn_n_embd")}})
    return identities


def cost_links(profile):
    links = tuple(row.link_id for row in profile.links
                  if row.maximum_payload_bytes == 10240
                  and profile.devices[row.source_device].kind.startswith("phone")
                      != profile.devices[row.target_device].kind.startswith("phone"))
    if len(links) != 2:
        raise RuntimeError("expected one measured 10,240-byte cost link in each direction, got " + repr(links))
    return links


def transport(profile, links, payload, tokens):
    return dict(AutomatedRouteCompiler._transport_adapter_parameters(
        profile, links, required_maximum_payload_bytes=payload, slot_payload_multiplier=SLOT_MULTIPLIER,
        batch_plan="coalesced-batch", maximum_tokens=tokens))


def main(argv):
    inputs, resolve = Path(argv[0]), Path(argv[1])
    negative = Path(argv[argv.index("--negative-catalog") + 1]) if "--negative-catalog" in argv else None
    catalog = RuntimeCapabilityCatalog.from_json(json.loads((resolve / "UNIFIED_RUNTIME_CATALOG.json").read_text()))
    identity = json.loads((inputs / "TRANSPORT_QUALIFICATION_IDENTITY.json").read_text())
    rig = json.loads((inputs / "rig.json").read_text())
    models = json.loads((inputs / "models.json").read_text())
    profile = catalog.placement_profile
    links = cost_links(profile)
    result = {"status": "PASS", "identity_id": identity["identity_id"],
              "identity_receipts": len(identity["receipt_sha256s"])}
    # FunctionFS USB links only: the catalog also carries the whole-phone NCM overlay link, which is bound
    # to the whole-server identity, not to a transport qualification identity.
    phone_links = [row for row in profile.links
                   if profile.devices[row.source_device].kind.startswith("phone")
                   != profile.devices[row.target_device].kind.startswith("phone")
                   and row.transport_generation == "functionfs-dmabuf-async-ring-v2"]
    result["other_phone_links"] = sorted(
        {(row.transport_generation, row.qualification_identity_sha256) for row in profile.links
         if profile.devices[row.source_device].kind.startswith("phone")
         != profile.devices[row.target_device].kind.startswith("phone")
         and row.transport_generation != "functionfs-dmabuf-async-ring-v2"})
    result["measured_phone_links"] = sorted(
        {(row.maximum_payload_bytes, row.queue_depth, row.transport_profile_id.rsplit(":", 1)[1],
          int(row.bandwidth_bytes_per_s / 1_000_000)) for row in phone_links})
    identity_shas = {row.qualification_identity_sha256 for row in phone_links}
    if len(identity_shas) != 1:
        raise RuntimeError("FunctionFS links bound to several identities: " + repr(identity_shas))
    result["catalog_identity_sha256"] = next(iter(identity_shas))
    if rig["phone"]["boot_image_sha256"] != identity["hardware_identity"]["phone_boot_image_sha256"]:
        raise RuntimeError("rig boot identity differs from the transport identity")
    for model in models["models"]:
        key = model.get("model_key")
        if key not in MODELS:
            continue
        prefix, n_embd, parallel, ubatch = MODELS[key]
        runtime = model["runtime_parameters"]
        if runtime["parallel"] != parallel or runtime["ubatch_size"] != ubatch:
            raise RuntimeError(f"{key}: models.json parallel/ubatch {runtime} != expected {parallel}/{ubatch}")
        if model.get("qualified_phone_batch_plans") != ["coalesced-batch"]:
            raise RuntimeError(f"{key}: qualified plans {model.get('qualified_phone_batch_plans')}")
        tokens = min(ubatch, parallel)
        payload = n_embd * tokens * 2
        entry = {"n_embd": n_embd, "parallel": parallel, "decode_rows": tokens, "required_payload_bytes": payload,
                 "helpers": helper_identities(catalog, prefix, "coalesced-batch")}
        parameters = transport(profile, links, payload, tokens)
        expected_suffix = f":payload-{payload}:"
        if (parameters["usb_max_payload_bytes"] != payload or parameters["usb_queue_depth"] != 4
                or expected_suffix not in parameters["usb_capacity_h2d_transport_profile_id"]
                or expected_suffix not in parameters["usb_capacity_d2h_transport_profile_id"]
                or parameters["usb_transport_qualification_identity_sha256"] != result["catalog_identity_sha256"]):
            raise RuntimeError(f"{key}: coalesced capacity differs: " + json.dumps(parameters, sort_keys=True))
        entry["coalesced_transport"] = parameters
        result[key] = entry
    fallback_payload = 3840 * 5 * 2
    parameters = transport(profile, links, fallback_payload, 5)
    result["cold_parallel5_fallback"] = {"required_payload_bytes": fallback_payload,
                                        "usb_max_payload_bytes": parameters["usb_max_payload_bytes"],
                                        "capacity_h2d": parameters["usb_capacity_h2d_transport_profile_id"],
                                        "capacity_d2h": parameters["usb_capacity_d2h_transport_profile_id"]}
    # The route compiler sizes a decode call with the helper's OWN adapter parameter `parallel` (the desktop
    # parent's measured plan, catalog rows), not models.json runtime_parameters.parallel. Record that case too.
    catalog_parallel = {}
    for row in catalog.composite_executors:
        if row.executor_id.startswith("physical:cold:") and row.route_family == "operator_split" \
                and row.adapter_parameters.get("usb_batch_plan") == "coalesced-batch":
            catalog_parallel[row.executor_id] = (row.adapter_parameters.get("parallel"),
                                                 row.adapter_parameters.get("ubatch_size"))
    rows = {min(ubatch, parallel) for parallel, ubatch in catalog_parallel.values()}
    if len(rows) != 1:
        raise RuntimeError("cold helpers disagree on parallel: " + repr(catalog_parallel))
    catalog_rows = rows.pop()
    catalog_payload = 3840 * catalog_rows * 2
    parameters = transport(profile, links, catalog_payload, catalog_rows)
    result["cold_catalog_parallel"] = {"helper_parallel": catalog_parallel, "decode_rows": catalog_rows,
                                      "required_payload_bytes": catalog_payload,
                                      "usb_max_payload_bytes": parameters["usb_max_payload_bytes"],
                                      "capacity_h2d": parameters["usb_capacity_h2d_transport_profile_id"],
                                      "capacity_d2h": parameters["usb_capacity_d2h_transport_profile_id"]}
    if negative is not None:
        old = RuntimeCapabilityCatalog.from_json(json.loads(negative.read_text()))
        old_profile = old.placement_profile
        old_links = cost_links(old_profile)
        try:
            transport(old_profile, old_links, 61440, 8)
        except RouteGenerationError as error:
            result["negative_control_61440_against_old_identity"] = {"refused": True, "error": str(error)}
        else:
            raise RuntimeError("negative control: 61,440 bytes admitted against the old identity")
        parameters = transport(old_profile, old_links, catalog_payload, catalog_rows)
        result["old_identity_cold_catalog_parallel"] = {
            "required_payload_bytes": catalog_payload, "usb_max_payload_bytes": parameters["usb_max_payload_bytes"],
            "capacity_h2d": parameters["usb_capacity_h2d_transport_profile_id"],
            "identity_sha256": parameters["usb_transport_qualification_identity_sha256"]}
    out = inputs / ("TRANSPORT_ADMISSION-" + resolve.name + ".json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(result, sort_keys=True, default=str, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
