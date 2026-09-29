#!/usr/bin/env python3
"""Build the hash-bound three-session OP15 residency plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    PhoneResidencyPlan,
    PhoneSessionPlan,
    ResidentPhoneSlice,
)


MIB = 1024 * 1024
GEMMA_SHA256 = (
    "sha256:ed76f2183d2d1d65091986033023e6c78d27f6276c1b0c5826cc92acf73538cf"
)
QWEN_SHA256 = (
    "sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718"
)


def build_plan() -> PhoneResidencyPlan:
    evidence = (
        "op15-resident-triple-full-v3",
        "qwen-full-ffn-energy-screen-r1-r2",
        "qwen-full-ffn-m1-m4-energy-screen-abba-v2",
    )
    return PhoneResidencyPlan(
        plan_id="op15-gemma-qwen-three-session-v1",
        phone_serial="3C15AU002CL00000",
        memory_resource_id="op15-dram",
        shared_compute_resource_id="op15-htp",
        transport_resource_ids=(
            "op15-functionfs",
            "desktop-usb-root",
        ),
        memory_capacity_bytes=15_846_404_096,
        minimum_available_bytes=2 * 1024 * MIB,
        reset_generation=0,
        sessions=(
            PhoneSessionPlan(
                session_id="htp0",
                compute_backend="HTP0",
                mapping_limit_bytes=3_328 * MIB,
                slices=(ResidentPhoneSlice(
                    slice_id="gemma-ffn-layers-0-22-suffix-6144",
                    model_id="gemma-4-12b-f16-proxy",
                    model_hash=GEMMA_SHA256,
                    operator_family="dense_ffn_geglu_suffix",
                    weight_hash=(
                        "sha256:e11e4c136f37fbd92eb12a8aefe21d801b27cc0c4db343bb5df3ffc0b27d4524"
                    ),
                    resident_bytes=3_255_877_632,
                    physical_m_min=1,
                    physical_m_max=16,
                    evidence_ids=evidence,
                ),),
            ),
            PhoneSessionPlan(
                session_id="htp1",
                compute_backend="HTP1",
                mapping_limit_bytes=3_328 * MIB,
                slices=(ResidentPhoneSlice(
                    slice_id="qwen-ffn-layers-0-5-full",
                    model_id="qwen3-14b-f16-proxy",
                    model_hash=QWEN_SHA256,
                    operator_family="dense_ffn_swiglu_full_replacement",
                    weight_hash=(
                        "sha256:712fa1e4417542fc4ff1725185fd8da4bc7a2d78167944c9f24464b93706ca3f"
                    ),
                    resident_bytes=3_208_646_656,
                    physical_m_min=1,
                    physical_m_max=4,
                    evidence_ids=evidence,
                ),),
            ),
            PhoneSessionPlan(
                session_id="htp2",
                compute_backend="HTP2",
                mapping_limit_bytes=3_328 * MIB,
                slices=(ResidentPhoneSlice(
                    slice_id="qwen-ffn-layers-6-11-full",
                    model_id="qwen3-14b-f16-proxy",
                    model_hash=QWEN_SHA256,
                    operator_family="dense_ffn_swiglu_full_replacement",
                    weight_hash=(
                        "sha256:0dbe02d29cf35e17da2868088fec6a70d433f01f340ab0c5fd7b3893a50fc33b"
                    ),
                    resident_bytes=3_208_646_656,
                    physical_m_min=1,
                    physical_m_max=4,
                    evidence_ids=evidence,
                ),),
            ),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an unused absolute path")
    plan = build_plan()
    value = plan.to_json()
    args.output.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "output": str(args.output),
        "resident_bytes": plan.resident_bytes,
        "sessions": len(plan.sessions),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
