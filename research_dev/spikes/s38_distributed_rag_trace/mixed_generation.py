#!/usr/bin/env python3
"""Bind S38 RAG generation to the measured StageNet mixed-phase runtime."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


HERE = Path(__file__).resolve().parent
S36 = HERE.parent / "s36_dynamic_cut_scheduler"
if str(S36) not in sys.path:
    sys.path.insert(0, str(S36))

from dynamic_route_runtime import DynamicOutcome, DynamicRequest  # noqa: E402


class MixedGenerationError(RuntimeError):
    pass


class Tokenizer(Protocol):
    def encode_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        template_options: Mapping[str, Any] | None = None,
    ) -> tuple[int, ...]: ...

    def decode_tokens(self, tokens: Sequence[int]) -> str: ...


class RouteRunner(Protocol):
    @property
    def routes(self) -> Mapping[str, object]: ...

    def run(
        self,
        request: DynamicRequest,
        timeout_s: float,
        scheduled_arrival_ns: int | None = None,
    ) -> DynamicOutcome: ...


@dataclass(frozen=True)
class GenerationResponse:
    content: str
    reasoning_content: str
    usage: Mapping[str, int]
    finish_reason: str | None

    def __post_init__(self) -> None:
        if (
            type(self.content) is not str
            or type(self.reasoning_content) is not str
            or self.finish_reason is not None
            and type(self.finish_reason) is not str
            or type(self.usage) is not dict
            or any(type(key) is not str for key in self.usage)
            or any(type(value) is not int or value < 0 for value in self.usage.values())
        ):
            raise ValueError("invalid generation response")


@dataclass(frozen=True)
class GenerationWork:
    request_id: int
    route_epoch: int
    max_output_tokens: int
    priority: int
    batch_wait_us: int
    policy_mode: str
    slo_us: int | None
    reasoning_budget_tokens: int = 0
    prefill_quantum: int = 64
    stop_tokens: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.request_id) is not int
            or self.request_id <= 0
            or type(self.route_epoch) is not int
            or self.route_epoch <= 0
            or type(self.max_output_tokens) is not int
            or self.max_output_tokens <= 0
            or type(self.priority) is not int
            or self.priority < 0
            or type(self.batch_wait_us) is not int
            or self.batch_wait_us < 0
            or self.policy_mode not in ("static", "slo")
            or self.slo_us is not None
            and (type(self.slo_us) is not int or self.slo_us <= 0)
            or self.policy_mode == "slo"
            and self.slo_us is None
            or type(self.reasoning_budget_tokens) is not int
            or self.reasoning_budget_tokens < 0
            or type(self.prefill_quantum) is not int
            or not 1 <= self.prefill_quantum <= 64
            or type(self.stop_tokens) is not tuple
            or any(type(token) is not int or token < 0 for token in self.stop_tokens)
            or len(set(self.stop_tokens)) != len(self.stop_tokens)
        ):
            raise ValueError("invalid generation work")


@dataclass(frozen=True)
class RouteEvidence:
    route_id: str
    device: str
    cut: int
    model_sha256: str
    file_type: int
    n_ctx_seq: int
    max_streams: int
    max_rows: int
    min_prompt_tokens: int
    max_prompt_tokens: int
    min_output_tokens: int
    max_output_tokens: int
    predicted_p95_us: int
    safety_margin_us: int
    correctness: str
    placement: str
    latency: str
    artifacts: tuple[str, ...]

    def __post_init__(self) -> None:
        integer_fields = (
            self.cut,
            self.file_type,
            self.n_ctx_seq,
            self.max_streams,
            self.max_rows,
            self.min_prompt_tokens,
            self.max_prompt_tokens,
            self.min_output_tokens,
            self.max_output_tokens,
            self.predicted_p95_us,
            self.safety_margin_us,
        )
        if (
            not self.route_id
            or not self.device
            or len(self.model_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.model_sha256)
            or any(type(value) is not int for value in integer_fields)
            or self.cut <= 0
            or self.file_type < 0
            or min(self.n_ctx_seq, self.max_streams, self.max_rows) <= 0
            or self.min_prompt_tokens <= 0
            or self.min_prompt_tokens > self.max_prompt_tokens
            or self.min_output_tokens <= 0
            or self.min_output_tokens > self.max_output_tokens
            or self.predicted_p95_us <= 0
            or self.safety_margin_us < 0
            or self.correctness not in ("PASS", "FAIL", "UNKNOWN")
            or self.placement not in ("PASS", "FAIL", "UNKNOWN")
            or self.latency not in ("PASS", "FAIL", "UNKNOWN")
            or type(self.artifacts) is not tuple
            or any(
                type(value) is not str
                or not value.startswith("sha256:")
                or len(value) != 71
                or any(char not in "0123456789abcdef" for char in value[7:])
                for value in self.artifacts
            )
        ):
            raise ValueError("invalid route evidence")

    @property
    def evidence_pass(self) -> bool:
        return (
            self.correctness == "PASS"
            and self.placement == "PASS"
            and self.latency == "PASS"
            and bool(self.artifacts)
        )

    @property
    def predicted_bound_us(self) -> int:
        return self.predicted_p95_us + self.safety_margin_us


@dataclass(frozen=True)
class Admission:
    action: str
    route_id: str | None
    reason: str
    rejected_routes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RoutedGeneration:
    response: GenerationResponse
    admission: Admission
    prompt_tokens: int
    output_tokens: tuple[int, ...] | None
    outcome: DynamicOutcome | None


def evaluate_route(
    route: RouteEvidence,
    work: GenerationWork,
    prompt_tokens: int,
    expected_model_sha256: str,
    expected_file_type: int,
) -> str | None:
    if route.model_sha256 != expected_model_sha256 or route.file_type != expected_file_type:
        return "MODEL_IDENTITY_MISMATCH"
    if work.reasoning_budget_tokens > 0:
        return "REASONING_BUDGET_UNSUPPORTED"
    if not route.evidence_pass:
        return "EVIDENCE_NOT_PASS"
    if not route.min_prompt_tokens <= prompt_tokens <= route.max_prompt_tokens:
        return "PROMPT_OUTSIDE_MEASURED_ENVELOPE"
    if not route.min_output_tokens <= work.max_output_tokens <= route.max_output_tokens:
        return "OUTPUT_OUTSIDE_MEASURED_ENVELOPE"
    if prompt_tokens + work.max_output_tokens > route.n_ctx_seq:
        return "CONTEXT_OVERFLOW"
    if work.policy_mode == "slo" and route.predicted_bound_us > work.slo_us:
        return "SLO_INFEASIBLE"
    return None


def choose_route(
    routes: Sequence[RouteEvidence],
    work: GenerationWork,
    prompt_tokens: int,
    expected_model_sha256: str,
    expected_file_type: int,
) -> Admission:
    if (
        type(prompt_tokens) is not int
        or prompt_tokens <= 0
        or len(expected_model_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_model_sha256)
        or type(expected_file_type) is not int
        or expected_file_type < 0
    ):
        raise ValueError("invalid admission inputs")
    if not routes:
        return Admission("server", None, "NO_PHONE_ROUTE", ())

    accepted: list[RouteEvidence] = []
    rejected: list[tuple[str, str]] = []
    route_ids: set[str] = set()
    for route in routes:
        if route.route_id in route_ids:
            raise ValueError("duplicate route evidence")
        route_ids.add(route.route_id)
        reason = evaluate_route(
            route,
            work,
            prompt_tokens,
            expected_model_sha256,
            expected_file_type,
        )
        if reason is None:
            accepted.append(route)
        else:
            rejected.append((route.route_id, reason))

    if not accepted:
        return Admission(
            "server",
            None,
            "NO_ELIGIBLE_PHONE_ROUTE",
            tuple(sorted(rejected)),
        )
    accepted.sort(key=lambda route: (-route.cut, route.predicted_bound_us, route.route_id))
    selected = accepted[0]
    return Admission(
        "stage",
        selected.route_id,
        "DEEPEST_FEASIBLE_CUT",
        tuple(sorted(rejected)),
    )


class LlamaServerTokenizer:
    def __init__(
        self,
        base_url: str,
        request_json: Callable[[str, dict[str, Any], float], Any],
    ) -> None:
        if not base_url or not callable(request_json):
            raise ValueError("tokenizer endpoint is required")
        self._base_url = base_url.rstrip("/")
        self._request_json = request_json

    def encode_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        template_options: Mapping[str, Any] | None = None,
    ) -> tuple[int, ...]:
        if not messages:
            raise ValueError("messages cannot be empty")
        normalized = []
        for message in messages:
            if (
                type(message) is not dict
                or set(message) != {"role", "content"}
                or type(message["role"]) is not str
                or type(message["content"]) is not str
            ):
                raise ValueError("invalid chat message")
            normalized.append(dict(message))
        if template_options is None:
            template_options = {}
        if (
            type(template_options) is not dict
            or set(template_options) - {"thinking_budget_tokens"}
            or "thinking_budget_tokens" in template_options
            and (
                type(template_options["thinking_budget_tokens"]) is not int
                or template_options["thinking_budget_tokens"] < 0
            )
        ):
            raise ValueError("invalid template options")
        template_request = {
            "messages": normalized,
            "add_generation_prompt": True,
            **template_options,
        }
        templated = self._request_json(
            self._base_url + "/apply-template",
            template_request,
            30.0,
        )
        prompt = templated.get("prompt") if isinstance(templated, dict) else None
        if not isinstance(prompt, str) or not prompt:
            raise MixedGenerationError("chat template returned no prompt")
        tokenized = self._request_json(
            self._base_url + "/tokenize",
            {"content": prompt, "add_special": True, "parse_special": True},
            30.0,
        )
        tokens = tokenized.get("tokens") if isinstance(tokenized, dict) else None
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise MixedGenerationError("tokenizer returned invalid tokens")
        return tuple(tokens)

    def decode_tokens(self, tokens: Sequence[int]) -> str:
        values = list(tokens)
        if not values or any(type(token) is not int or token < 0 for token in values):
            raise ValueError("invalid output tokens")
        decoded = self._request_json(
            self._base_url + "/detokenize",
            {"tokens": values},
            30.0,
        )
        content = decoded.get("content") if isinstance(decoded, dict) else None
        if not isinstance(content, str):
            raise MixedGenerationError("detokenizer returned invalid content")
        return content


class MixedGenerationAdapter:
    def __init__(
        self,
        runner: RouteRunner,
        tokenizer: Tokenizer,
        routes: Sequence[RouteEvidence],
        expected_model_sha256: str,
        expected_file_type: int,
    ) -> None:
        if runner is None or tokenizer is None:
            raise ValueError("runner and tokenizer are required")
        self._runner = runner
        self._tokenizer = tokenizer
        self._routes = tuple(routes)
        self._expected_model_sha256 = expected_model_sha256
        self._expected_file_type = expected_file_type

    def _live_route_matches(self, evidence: RouteEvidence) -> bool:
        live_routes = self._runner.routes
        if not isinstance(live_routes, Mapping):
            return False
        live = live_routes.get(evidence.route_id)
        head = getattr(live, "head", None)
        tail = getattr(live, "tail", None)
        head_hello = getattr(head, "hello", None)
        tail_hello = getattr(tail, "hello", None)
        live_capacity = min(
            getattr(head_hello, "n_batch", 0),
            getattr(head_hello, "n_ubatch", 0),
            getattr(tail_hello, "n_batch", 0),
            getattr(tail_hello, "n_ubatch", 0),
            64,
        )
        return not (
            live is None
            or getattr(live, "cut", None) != evidence.cut
            or getattr(head, "name", None) != evidence.device
            or getattr(head_hello, "file_type", None) != evidence.file_type
            or getattr(tail_hello, "file_type", None) != evidence.file_type
            or getattr(head_hello, "n_ctx_seq", None) != evidence.n_ctx_seq
            or getattr(tail_hello, "n_ctx_seq", None) != evidence.n_ctx_seq
            or min(
                getattr(head_hello, "max_streams", 0),
                getattr(tail_hello, "max_streams", 0),
            )
            != evidence.max_streams
            or live_capacity != evidence.max_rows
        )

    def run(
        self,
        messages: Sequence[Mapping[str, str]],
        work: GenerationWork,
        fallback: Callable[[], GenerationResponse],
        timeout_s: float,
        scheduled_arrival_ns: int | None = None,
    ) -> RoutedGeneration:
        if (
            not callable(fallback)
            or type(timeout_s) not in (int, float)
            or isinstance(timeout_s, bool)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("invalid generation execution inputs")
        prompt = self._tokenizer.encode_messages(
            messages,
            {"thinking_budget_tokens": work.reasoning_budget_tokens},
        )
        admission = choose_route(
            self._routes,
            work,
            len(prompt),
            self._expected_model_sha256,
            self._expected_file_type,
        )
        if admission.action == "server":
            response = fallback()
            if not isinstance(response, GenerationResponse):
                raise MixedGenerationError("fallback returned an invalid response")
            return RoutedGeneration(response, admission, len(prompt), None, None)

        if admission.route_id is None:
            raise MixedGenerationError("stage admission omitted its route")
        evidence_by_id = {route.route_id: route for route in self._routes}
        selected_evidence = evidence_by_id.get(admission.route_id)
        if selected_evidence is None:
            raise MixedGenerationError("stage admission selected unknown evidence")
        if not self._live_route_matches(selected_evidence):
            fallback_admission = Admission(
                "server",
                None,
                "LIVE_ROUTE_MISMATCH",
                tuple(sorted(
                    admission.rejected_routes
                    + ((selected_evidence.route_id, "LIVE_ROUTE_MISMATCH"),)
                )),
            )
            response = fallback()
            if not isinstance(response, GenerationResponse):
                raise MixedGenerationError("fallback returned an invalid response")
            return RoutedGeneration(
                response,
                fallback_admission,
                len(prompt),
                None,
                None,
            )
        request = DynamicRequest(
            request_id=work.request_id,
            route_epoch=work.route_epoch,
            route_id=admission.route_id,
            prompt_tokens=prompt,
            output_steps=work.max_output_tokens,
            priority=work.priority,
            slo_us=work.slo_us if work.slo_us is not None else (1 << 62),
            batch_wait_us=work.batch_wait_us,
            prefill_quantum=work.prefill_quantum,
            stop_tokens=work.stop_tokens,
        )
        outcome = self._runner.run(request, timeout_s, scheduled_arrival_ns)
        if (
            not isinstance(outcome, DynamicOutcome)
            or outcome.request_id != work.request_id
            or outcome.route_epoch != work.route_epoch
            or outcome.route_id != admission.route_id
            or not 1 <= len(outcome.output_tokens) <= work.max_output_tokens
        ):
            raise MixedGenerationError("stage runner returned an invalid outcome")
        output_tokens = tuple(outcome.output_tokens)
        response = GenerationResponse(
            content=self._tokenizer.decode_tokens(output_tokens),
            reasoning_content="",
            usage={
                "prompt_tokens": len(prompt),
                "completion_tokens": len(output_tokens),
                "total_tokens": len(prompt) + len(output_tokens),
            },
            finish_reason=outcome.finish_reason,
        )
        return RoutedGeneration(response, admission, len(prompt), output_tokens, outcome)
