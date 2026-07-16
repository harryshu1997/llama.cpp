#!/usr/bin/env python3
"""Derive the S9-V0-R2/v4 record schemas from the frozen v3 bundle.

v4 is a STATIC bundle-coherence repair. Most of its teeth are in the VALIDATOR's new
cross_record_v4 checks and in the v4 digest family (s9lib), so 13 record schemas + the
bundle envelope are structurally identical to v3 and only bump schema_version /
bundle_version const 3->4 (and the $id / title). TWO schemas gain the fields the review
requires the dispatch to commit to:

  dispatch_decision  += decision_ts_us            (repair 2: a decision timestamp)
                     += device_status_ref{device_id,boot_epoch,status_seq}
                                                   (repair 1: pin exactly one DeviceInventory snapshot)
  transport_frame    += device_id                 (repairs 7,8: bind BULK/EXECUTE frames to a device)

sim_config / sim_run_manifest are NOT emitted: v4 does not touch the simulator. Deterministic;
ASCII only. Re-running fully regenerates schemas/v4/ (idempotent).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(HERE, "schemas", "v3")
V4 = os.path.join(HERE, "schemas", "v4")
SKIP = {"sim_config.schema.json", "sim_run_manifest.schema.json"}


def bump(text):
    text = text.replace('"const": 3 }', '"const": 4 }')
    text = text.replace('/schemas/v3/', '/schemas/v4/')
    text = text.replace('(v3)"', '(v4)"')
    return text


def patch_dispatch(obj):
    obj["properties"]["decision_ts_us"] = {
        "type": "integer", "minimum": 0, "maximum": 9007199254740991}
    obj["properties"]["device_status_ref"] = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "device_id": {"type": "string", "minLength": 1, "maxLength": 256},
            "boot_epoch": {"type": "integer", "minimum": 0, "maximum": 9007199254740991},
            "status_seq": {"type": "integer", "minimum": 0, "maximum": 9007199254740991},
        },
        "required": ["device_id", "boot_epoch", "status_seq"]}
    for k in ("decision_ts_us", "device_status_ref"):
        if k not in obj["required"]:
            obj["required"].append(k)
    return obj


def patch_transport(obj):
    obj["properties"]["device_id"] = {"type": "string", "minLength": 1, "maxLength": 256}
    if "device_id" not in obj["required"]:
        obj["required"].append("device_id")
    return obj


PATCHERS = {"dispatch_decision.schema.json": patch_dispatch,
            "transport_frame.schema.json": patch_transport}


def main():
    os.makedirs(V4, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(V3)):
        if not fn.endswith(".schema.json") or fn in SKIP:
            continue
        out = bump(open(os.path.join(V3, fn)).read())
        obj = json.loads(out)                     # must stay valid JSON
        if fn in PATCHERS:
            obj = PATCHERS[fn](obj)
            out = json.dumps(obj, indent=2) + "\n"
        open(os.path.join(V4, fn), "w").write(out)
        n += 1
    print(f"generated {n} v4 record schemas under schemas/v4/ (2 structurally extended)")


if __name__ == "__main__":
    main()
