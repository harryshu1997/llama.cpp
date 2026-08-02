#!/usr/bin/env python3

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
HISTORY = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "prephase_20260726T0915Z"
    / "token-history.json"
)
HISTORY_SHA256 = "3755944451dac26ee046f6fc89b1ebd5ea060150d0447d739707138bdcb93314"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phone = load_module("v24_phone_history_consumer", "phone_route_v1.py")
cuda = load_module("v24_cuda_history_consumer", "cuda_route_v1.py")
monolithic = load_module("v24_monolithic_history_consumer", "cuda_monolithic_v1.py")


class HistoryConsumerTests(unittest.TestCase):
    def test_persisted_history_passes_all_three_consumers(self):
        raw = HISTORY.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), HISTORY_SHA256)
        value = json.loads(raw)
        phone_value, _ = phone.load_histories(
            HISTORY,
            HISTORY_SHA256,
            phone.MODEL_SHA256,
        )
        cuda_value, _ = cuda.load_histories(HISTORY)
        prompts = [
            base64.b64decode(request["prompt_utf8_base64"], validate=True)
            for request in value["requests"]
        ]
        prompt_sha256s = [
            hashlib.sha256(prompt).hexdigest()
            for prompt in prompts
        ]
        histories, plan, reopened = monolithic.load_histories(
            HISTORY,
            phone.MODEL_SHA256,
            value["corpus_sha256"],
            prompt_sha256s,
            prompts,
        )
        self.assertEqual(reopened, raw)
        self.assertEqual(phone_value, cuda_value)
        self.assertEqual(len(histories), 8)
        self.assertEqual(plan, value["quality_groups"][0])


if __name__ == "__main__":
    unittest.main()
