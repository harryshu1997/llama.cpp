#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from research_dev.scheduler import RuntimeCompositeExecutorCapability
from research_dev.scheduler.adapters import (
    CanonicalStaticSplitPrewarmer,
    LlamaCppHttpClient,
    compile_static_column_split_contract,
    static_split_launch_contract,
    validate_static_split_ready_log,
)


ARTIFACT_SHA256 = "sha256:" + "a" * 64


def manifest() -> dict[str, object]:
    return {
        "geometry": {
            "n_embd": 16,
            "n_ff": 8,
            "resident_layer_ids": [0],
        },
        "split_contract": {
            "layer_mask": 1,
            "max_columns": 4,
        },
    }


def policy() -> dict[str, object]:
    return {
        "compiled_buckets": [{
            "max_tokens": 4,
            "phone_columns": 4,
        }],
        "policy_text": "tokens<=4:phone_columns=4",
    }


def capability() -> RuntimeCompositeExecutorCapability:
    contract = compile_static_column_split_contract(
        manifest(), policy(), protocol_namespace="synthetic-static-v1"
    )
    return RuntimeCompositeExecutorCapability(
        executor_id="coordinator:host-helper",
        endpoint="http://127.0.0.1:19001",
        backend="synthetic-static-split",
        coordinator_device_id="host-a",
        participant_device_ids=("host-a", "helper-b"),
        participant_resource_ids={
            "host-a": ("compute:host-a",),
            "helper-b": ("compute:helper-b", "link:host-helper"),
        },
        route_family="operator_split",
        assisted_operator_kind="ffn",
        split_axis="column",
        split_fractions_ppm=(contract.fraction_ppm,),
        layer_fractions_ppm=(),
        residency_states=("hot",),
        resource_ids=(
            "compute:host-a",
            "compute:helper-b",
            "link:host-helper",
        ),
        operator_plan_protocol=contract.protocol,
        maturity="QUALIFIED",
        evidence_ids=("synthetic-physical-split",),
        artifact_sha256=ARTIFACT_SHA256,
        operator_ids=contract.operator_ids,
    )


class FakeLlamaCppHttpClient(LlamaCppHttpClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def complete(
        self,
        endpoint,
        payload,
        control_check,
        *,
        scheduler_headers=None,
    ):
        control_check()
        self.calls.append((endpoint, payload, scheduler_headers))
        return {"stream_sha256": "b" * 64}


class StaticSplitAdapterTests(unittest.TestCase):
    def test_launch_contract_owns_environment_and_readiness_validation(
        self,
    ) -> None:
        contract = static_split_launch_contract(
            manifest(), policy(), phone_host="192.0.2.10", phone_port=9000
        )
        self.assertEqual(
            contract.environment["S41_SERVER_FFN_COLUMNS"], "4"
        )
        self.assertEqual(
            contract.environment["S41_SERVER_FFN_HOST"], "192.0.2.10"
        )
        validate_static_split_ready_log(
            ["S41SERVERFFN ready policy=" + contract.policy_text],
            contract,
        )

    def test_prewarmer_validates_capability_and_executes_exact_endpoint(
        self,
    ) -> None:
        client = FakeLlamaCppHttpClient()
        prewarmer = CanonicalStaticSplitPrewarmer(client)
        with tempfile.TemporaryDirectory() as directory:
            receipt = prewarmer.prewarm(
                capability(),
                artifact_sha256=ARTIFACT_SHA256,
                manifest=manifest(),
                policy=policy(),
                rows=({
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "overlay_request_index": 0,
                    "prompt_tokens": [1, 2],
                },),
                event_id="synthetic-prewarm",
                request_index=99,
                expected_model_alias="synthetic-model",
                stream_path=Path(directory) / "prewarm.raw",
                timeout_s=5,
            )
        self.assertEqual(len(client.calls), 1)
        endpoint, payload, headers = client.calls[0]
        self.assertEqual(endpoint, capability().endpoint)
        self.assertEqual(payload.input_tokens, 4)
        self.assertEqual(payload.prompt_tokens, (1, 2, 1, 2))
        self.assertIsNone(headers)
        self.assertEqual(receipt.executor_id, capability().executor_id)
        self.assertEqual(receipt.phone_columns, 4)
        self.assertEqual(receipt.stream_sha256, "b" * 64)


if __name__ == "__main__":
    unittest.main()
