#!/usr/bin/env python3
"""Build the fixed B32 physical-launch input manifest."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
S15 = HERE.parent / "s15_runtime_dispatch"
sys.path.insert(0, str(S15))

from input_manifest import canonical, load_manifest  # noqa: E402


OUTPUT = HERE / "fixtures" / "op15_b32_fixed_input.manifest.json"
PROMPT = b"Explain batching."
REQUEST_IDS = tuple(f"live-b32-{index:03d}" for index in range(32))


def sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build() -> bytes:
    prompt_b64 = base64.b64encode(PROMPT).decode("ascii")
    prompt_sha256 = sha256(PROMPT)
    value = {
        "schema": "s15-input-manifest-v1",
        "workload_id": "op15-b32-fixed-input-smoke-v1",
        "source_kind": "fixed_certified_smoke_input_not_trace",
        "repeat_constraint": "all_input_bytes_identical",
        "route_identity": {
            "route_id": "op15-gemma-head-0-8",
            "profile_id": "sha256:947f1fd95b3f1a7c881b71d793bf3177fc9612a46d959fcef987da0026531cae",
            "device_id": "op15:3C15AU002CL00000",
            "model_id": "gemma-4-12b-it-f16",
            "model_sha256": "sha256:bac4e29337273d1cb41aa36f4209ebcf63baa1162e6ab4dc55f6034b3d69b35a",
            "shard_sha256": "sha256:a74845294bc1a3cba6ac26c747615d64d9127c904aae04c43c7ea2ebc1ea25c8",
            "layer_start": 0,
            "layer_end": 8,
        },
        "requests": [
            {
                "request_id": request_id,
                "input_b64": prompt_b64,
                "input_sha256": prompt_sha256,
            }
            for request_id in REQUEST_IDS
        ],
    }
    return canonical(value)


def main() -> int:
    payload = build()
    load_manifest(payload)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(payload)
    print(json.dumps({"path": str(OUTPUT), "sha256": sha256(payload)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
