#!/usr/bin/env python3
"""Derive the R1/v3 record schemas from the frozen v2 bundle.

The R1 contract's teeth are in the VALIDATOR's coherent-chain cross-record checks and
in the simulator, not in new record FIELDS -- the v3 records are structurally identical
to v2 (so v2 stays frozen and is not silently rewritten). This bumps schema_version and
bundle_version const 2->3 and the $id/title, for the 14 bundle-record schemas plus the
bundle envelope. sim_config.v3 and sim_run_manifest.v3 are hand-written (they DO add
fields) and are not touched here. Deterministic; ASCII only.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "schemas", "v2")
V3 = os.path.join(HERE, "schemas", "v3")
SKIP = {"sim_config.schema.json", "sim_run_manifest.schema.json"}


def bump(text):
    text = text.replace('"const": 2 }', '"const": 3 }')
    text = text.replace('/schemas/v2/', '/schemas/v3/')
    text = text.replace('(v2)"', '(v3)"')
    return text


def main():
    os.makedirs(V3, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(V2)):
        if not fn.endswith(".schema.json") or fn in SKIP:
            continue
        src = open(os.path.join(V2, fn)).read()
        out = bump(src)
        json.loads(out)                       # must stay valid JSON
        open(os.path.join(V3, fn), "w").write(out)
        n += 1
    print(f"generated {n} v3 record schemas under schemas/v3/")


if __name__ == "__main__":
    main()
