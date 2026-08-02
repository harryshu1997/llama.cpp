#!/usr/bin/env python3

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mixed_generation import (  # noqa: E402
    DynamicOutcome,
    DynamicRequest,
    GenerationResponse,
    GenerationWork,
    LlamaServerTokenizer,
    MixedGenerationAdapter,
    MixedGenerationError,
    RouteEvidence,
    choose_route,
)


MODEL = "a" * 64


def route(
    route_id: str = "op15-c8",
    cut: int = 8,
    model: str = MODEL,
    correctness: str = "PASS",
    max_prompt: int = 4096,
    predicted: int = 100_000,
) -> RouteEvidence:
    return RouteEvidence(
        route_id=route_id,
        device="op15",
        cut=cut,
        model_sha256=model,
        file_type=7,
        n_ctx_seq=4096,
        max_streams=8,
        max_rows=64,
        min_prompt_tokens=1,
        max_prompt_tokens=max_prompt,
        min_output_tokens=1,
        max_output_tokens=256,
        predicted_p95_us=predicted,
        safety_margin_us=10_000,
        correctness=correctness,
        placement="PASS",
        latency="PASS",
        artifacts=("sha256:" + "1" * 64,),
    )


def work(mode: str = "slo", slo_us: int | None = 500_000) -> GenerationWork:
    return GenerationWork(1, 2, 4, 1, 20_000, mode, slo_us)


class FakeTokenizer:
    def __init__(self, prompt: tuple[int, ...] = (1, 2, 3)) -> None:
        self.prompt = prompt
        self.template_options = None

    def encode_messages(self, messages, template_options=None):
        if not messages:
            raise AssertionError("messages missing")
        self.template_options = template_options
        return self.prompt

    def decode_tokens(self, tokens):
        return ",".join(str(token) for token in tokens)


class FakeRunner:
    def __init__(self) -> None:
        self.requests: list[DynamicRequest] = []
        hello = SimpleNamespace(
            file_type=7,
            n_ctx_seq=4096,
            max_streams=8,
            n_batch=64,
            n_ubatch=64,
        )
        self._routes = {
            "op15-c8": SimpleNamespace(
                cut=8,
                head=SimpleNamespace(name="op15", hello=hello),
                tail=SimpleNamespace(name="tail", hello=hello),
            ),
        }

    @property
    def routes(self):
        return dict(self._routes)

    def run(self, request, timeout_s, scheduled_arrival_ns=None):
        self.requests.append(request)
        now = time.monotonic_ns()
        return DynamicOutcome(
            request_id=request.request_id,
            route_epoch=request.route_epoch,
            route_id=request.route_id,
            device="op15",
            cut=8,
            priority=request.priority,
            prompt_length=len(request.prompt_tokens),
            output_tokens=(7, 8, 9, 10),
            scheduled_arrival_ns=scheduled_arrival_ns or now,
            call_ns=now,
            lease_ready_ns=now,
            first_token_ns=now,
            completed_ns=now,
            slo_us=request.slo_us,
        )


class MixedGenerationTests(unittest.TestCase):
    def test_deepest_feasible_route_is_selected(self) -> None:
        admission = choose_route(
            (route("op15-c4", 4, predicted=50_000), route("op15-c8", 8, predicted=100_000)),
            work(),
            100,
            MODEL,
            7,
        )
        self.assertEqual(admission.action, "stage")
        self.assertEqual(admission.route_id, "op15-c8")

    def test_every_fail_open_gate_falls_back(self) -> None:
        cases = (
            (route(model="b" * 64), 100, work(), "MODEL_IDENTITY_MISMATCH"),
            (
                route(),
                100,
                GenerationWork(1, 2, 4, 1, 20_000, "slo", 500_000, 1),
                "REASONING_BUDGET_UNSUPPORTED",
            ),
            (route(correctness="FAIL"), 100, work(), "EVIDENCE_NOT_PASS"),
            (route(max_prompt=4), 100, work(), "PROMPT_OUTSIDE_MEASURED_ENVELOPE"),
            (route(max_prompt=4096), 4094, work(), "CONTEXT_OVERFLOW"),
            (route(predicted=600_000), 100, work(), "SLO_INFEASIBLE"),
        )
        for evidence, prompt_tokens, request, reason in cases:
            with self.subTest(reason=reason):
                admission = choose_route((evidence,), request, prompt_tokens, MODEL, 7)
                self.assertEqual(admission.action, "server")
                self.assertEqual(admission.rejected_routes, ((evidence.route_id, reason),))

    def test_static_mode_does_not_invent_an_slo(self) -> None:
        request = work("static", None)
        admission = choose_route((route(predicted=10_000_000),), request, 100, MODEL, 7)
        self.assertEqual(admission.action, "stage")

    def test_adapter_submits_s36_dynamic_request(self) -> None:
        runner = FakeRunner()
        tokenizer = FakeTokenizer()
        adapter = MixedGenerationAdapter(runner, tokenizer, (route(),), MODEL, 7)
        result = adapter.run(
            ({"role": "user", "content": "question"},),
            work(),
            lambda: GenerationResponse("fallback", "", {}, "stop"),
            1.0,
        )
        self.assertEqual(result.admission.route_id, "op15-c8")
        self.assertEqual(result.response.content, "7,8,9,10")
        self.assertEqual(result.response.finish_reason, "length")
        self.assertEqual(len(runner.requests), 1)
        self.assertEqual(runner.requests[0].prompt_tokens, (1, 2, 3))
        self.assertEqual(runner.requests[0].prefill_quantum, 64)
        self.assertEqual(tokenizer.template_options, {"thinking_budget_tokens": 0})

    def test_rejected_route_calls_server_once(self) -> None:
        runner = FakeRunner()
        adapter = MixedGenerationAdapter(
            runner,
            FakeTokenizer(tuple(range(200))),
            (route(max_prompt=4),),
            MODEL,
            7,
        )
        calls = []

        def fallback():
            calls.append(True)
            return GenerationResponse("server", "reason", {"prompt_tokens": 200}, "stop")

        result = adapter.run(
            ({"role": "user", "content": "question"},),
            work(),
            fallback,
            1.0,
        )
        self.assertEqual(result.admission.action, "server")
        self.assertEqual(result.response.content, "server")
        self.assertEqual(calls, [True])
        self.assertEqual(runner.requests, [])

    def test_runtime_failure_never_silently_retries(self) -> None:
        class BrokenRunner:
            def __init__(self):
                self.routes = FakeRunner().routes

            def run(self, request, timeout_s, scheduled_arrival_ns=None):
                raise RuntimeError("worker failed")

        fallback_calls = []
        adapter = MixedGenerationAdapter(BrokenRunner(), FakeTokenizer(), (route(),), MODEL, 7)
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            adapter.run(
                ({"role": "user", "content": "question"},),
                work(),
                lambda: fallback_calls.append(True),
                1.0,
            )
        self.assertEqual(fallback_calls, [])

    def test_live_route_alias_mismatch_fails_before_compute(self) -> None:
        runner = FakeRunner()
        runner._routes["op15-c8"].cut = 4
        adapter = MixedGenerationAdapter(runner, FakeTokenizer(), (route(),), MODEL, 7)
        result = adapter.run(
            ({"role": "user", "content": "question"},),
            work(),
            lambda: GenerationResponse("fallback", "", {}, "stop"),
            1.0,
        )
        self.assertEqual(result.admission.action, "server")
        self.assertEqual(result.admission.reason, "LIVE_ROUTE_MISMATCH")
        self.assertEqual(runner.requests, [])

    def test_llama_server_tokenizer_uses_template_and_special_tokens(self) -> None:
        calls = []

        def request_json(url, body, timeout):
            calls.append((url, body, timeout))
            if url.endswith("/apply-template"):
                return {"prompt": "formatted"}
            if url.endswith("/tokenize"):
                return {"tokens": [2, 3]}
            return {"content": "answer"}

        tokenizer = LlamaServerTokenizer("http://localhost:8080/", request_json)
        tokens = tokenizer.encode_messages(
            ({"role": "user", "content": "q"},),
            {"thinking_budget_tokens": 32},
        )
        self.assertEqual(tokens, (2, 3))
        self.assertEqual(tokenizer.decode_tokens((4, 5)), "answer")
        self.assertTrue(calls[0][1]["add_generation_prompt"])
        self.assertEqual(calls[0][1]["thinking_budget_tokens"], 32)
        self.assertTrue(calls[1][1]["add_special"])

    def test_strict_types(self) -> None:
        with self.assertRaises(ValueError):
            GenerationWork(True, 1, 1, 0, 0, "static", None)
        with self.assertRaises(ValueError):
            GenerationWork(1, 1, 1, 0, 0, "static", None, True)
        with self.assertRaises(ValueError):
            work("slo", None)
        with self.assertRaises(ValueError):
            choose_route((route(), route()), work(), 10, MODEL, 7)
        with self.assertRaises(ValueError):
            RouteEvidence(
                **{
                    **route().__dict__,
                    "artifacts": ("not-a-digest",),
                }
            )
        with self.assertRaises(MixedGenerationError):
            LlamaServerTokenizer("http://x", lambda *_: {}).encode_messages(
                ({"role": "user", "content": "q"},)
            )
        with self.assertRaises(ValueError):
            LlamaServerTokenizer("http://x", lambda *_: {}).encode_messages(
                ({"role": "user", "content": "q"},),
                {"unknown": 1},
            )


if __name__ == "__main__":
    unittest.main()
