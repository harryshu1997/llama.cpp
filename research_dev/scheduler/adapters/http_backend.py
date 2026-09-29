"""HTTP execution backend for desktop, phone, and composite llama.cpp routes."""

from __future__ import annotations

from dataclasses import dataclass, replace
import errno
import hashlib
import http.client
import json
from pathlib import Path
import threading
import time
from typing import Callable, Mapping, Sequence
from urllib.parse import urlsplit

from .contracts import (
    CompletionStreamError,
    dormant_phone_ffn_parameters,
    PhysicalAdapterError,
    PhysicalBackendFailure,
    PhysicalFailureClassification,
    RawExecutionObservation,
    RawTransitionObservation,
    StalePhysicalSlotError,
)
from .ticket import (
    PhysicalExecutionCommand,
    PhysicalTransitionCommand,
    bind_ready_helper_to_physical_command,
    static_decode_policy,
    validate_physical_execution_command,
)
from .decode_cohort import (
    DecodeCohortExecutionTracker,
    DecodeCohortPolicyCoordinator,
    DecodeCohortPolicyView,
)
from .speculative_rows import (
    SpeculativeRequestContract,
    SpeculativeRowLedger,
    speculative_completion_statistics,
    speculative_request_contract,
)
from .output_quality import (
    accounting_output_assessment,
    assess_semantic_output,
)
from .._internal.adaptive_decode_contracts import (
    AdaptiveDecodeControl,
    AdaptiveDecodePolicy,
    AdaptiveDecodePolicyAck,
    AdaptiveDecodeRawWindowObservation,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _stream_error_message(value: object) -> str | None:
    """The ``error.message`` (or string ``error``) of one SSE error chunk."""
    if type(value) is not dict:
        return None
    error = value.get("error")
    if type(error) is str:
        return error
    if type(error) is dict and type(error.get("message")) is str:
        return error["message"]
    return None


def _server_control_failure(
    error: PhysicalAdapterError, server_message: object = None
) -> PhysicalAdapterError:
    """Mark one failed server control call (FFN policy, window stats, cohort).

    The server answered the call with an error status or a failed
    acknowledgement (a released slot included). The type and message stay
    as they were; ``server_control_message`` (the server's own text, empty
    for a bare status) lets a drop-recovery rig read the failure as
    server-side and name a helper that text names.
    """
    error.server_control_message = (
        "" if server_message is None else str(server_message)
    )
    return error


def parse_http_endpoint(endpoint: str) -> tuple[str, int]:
    value = urlsplit(endpoint)
    if not (
        value.scheme == "http"
        and value.hostname is not None
        and value.port is not None
        and not value.path
        and not value.query
        and not value.fragment
    ):
        raise PhysicalAdapterError("physical command endpoint is invalid")
    return value.hostname, value.port


@dataclass(frozen=True)
class LlamaCppCompletionPayload:
    request_id: str
    expected_model_alias: str
    input_tokens: int
    output_tokens: int
    prompt_tokens: tuple[int, ...]
    seed: int
    stream_path: Path
    on_first_token: Callable[[int], None]
    on_decode_progress: Callable[[int, int, int, bool], None] | None = None
    on_active_batch: Callable[[int], None] | None = None
    quality_mode: str = "semantic"
    timeout_s: float = 3600
    diagnostic_top_logprobs: int = 0
    cohort_submission_order: tuple[int, ...] = ()
    # campaign ``speculative_rows``: bound by the backend from the endpoint ledger; None keeps
    # the request body and the completion result byte-identical
    speculative: SpeculativeRequestContract | None = None

    def __post_init__(self) -> None:
        if self.speculative is not None and not isinstance(
            self.speculative, SpeculativeRequestContract
        ):
            raise PhysicalAdapterError("completion speculative contract is invalid")
        for name in ("request_id", "expected_model_alias"):
            value = getattr(self, name)
            if type(value) is not str or not value or not value.isascii():
                raise PhysicalAdapterError(
                    f"completion {name} must be non-empty ASCII text"
                )
        if (
            type(self.input_tokens) is not int
            or self.input_tokens <= 0
            or type(self.output_tokens) is not int
            or self.output_tokens <= 0
            or type(self.seed) is not int
        ):
            raise PhysicalAdapterError("completion shape is invalid")
        if (
            type(self.diagnostic_top_logprobs) is not int
            or not 0 <= self.diagnostic_top_logprobs <= 256
        ):
            raise PhysicalAdapterError("completion diagnostic logprob count is invalid")
        order = tuple(self.cohort_submission_order)
        if len(order) > 8 or any(type(index) is not int for index in order) or sorted(order) != list(range(len(order))):
            raise PhysicalAdapterError("completion cohort submission order is invalid")
        object.__setattr__(self, "cohort_submission_order", order)
        tokens = tuple(self.prompt_tokens)
        if (
            len(tokens) != self.input_tokens
            or any(type(value) is not int for value in tokens)
        ):
            raise PhysicalAdapterError("completion prompt is invalid")
        if not isinstance(self.stream_path, Path) or self.stream_path.exists():
            raise PhysicalAdapterError("completion stream path is not new")
        if (
            not callable(self.on_first_token)
            or (
                self.on_decode_progress is not None
                and not callable(self.on_decode_progress)
            )
            or (
                self.on_active_batch is not None
                and not callable(self.on_active_batch)
            )
            or self.timeout_s <= 0
            or self.quality_mode not in {"accounting-only", "semantic"}
        ):
            raise PhysicalAdapterError("completion callback is invalid")
        object.__setattr__(self, "prompt_tokens", tokens)


class HttpEndpointFailure(OSError):
    def __init__(self, phase: str, error: OSError, retry_safe: bool) -> None:
        super().__init__(error.errno, f"{phase}: {error.strerror or error}")
        self.phase = phase
        self.retry_safe = retry_safe


class LlamaCppHttpClient:
    """Execute and validate one deterministic llama.cpp completion."""

    def __init__(
        self,
        slots_probe: Callable[[str, int, float], list[dict[str, object]]]
            | None = None,
    ) -> None:
        if slots_probe is not None and not callable(slots_probe):
            raise PhysicalAdapterError("slots probe is invalid")
        self._slots_probe = self.slots if slots_probe is None else slots_probe

    @staticmethod
    def _progress_slot(
        reported_slot: object, active_task_by_slot: Mapping[int, int]
    ) -> int:
        if type(reported_slot) is int and reported_slot >= 0:
            return reported_slot
        active_slots = tuple(sorted(active_task_by_slot))
        if reported_slot == -1 and len(active_slots) == 1:
            return active_slots[0]
        raise PhysicalAdapterError(
            "completion progress slot cannot be resolved"
        )

    @staticmethod
    def slots(host: str, port: int, timeout_s: float) -> list[dict[str, object]]:
        connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
        try:
            connection.request("GET", "/slots")
            response = connection.getresponse()
            content = response.read()
            if response.status != 200:
                raise PhysicalAdapterError("slots endpoint status is invalid")
            value = json.loads(content)
            if type(value) is not list or any(
                type(row) is not dict for row in value
            ):
                raise PhysicalAdapterError("slots endpoint JSON is invalid")
            return value
        finally:
            connection.close()

    def complete(
        self,
        endpoint: str,
        payload: LlamaCppCompletionPayload,
        control_check: Callable[[], None],
        *,
        scheduler_headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        if not isinstance(payload, LlamaCppCompletionPayload):
            raise PhysicalAdapterError("completion payload is invalid")
        if not callable(control_check):
            raise PhysicalAdapterError("completion control check is invalid")
        extra_headers = (
            {} if scheduler_headers is None else dict(scheduler_headers)
        )
        if any(
            type(name) is not str
            or not name.startswith("X-Scheduler-")
            or not name.isascii()
            or type(value) is not str
            or not value
            or not value.isascii()
            for name, value in extra_headers.items()
        ):
            raise PhysicalAdapterError("scheduler HTTP headers are invalid")
        host, port = parse_http_endpoint(endpoint)
        request_body = {
            "cache_prompt": False,
            "ignore_eos": True,
            "n_predict": payload.output_tokens,
            "prompt": list(payload.prompt_tokens),
            "return_tokens": True,
            "seed": payload.seed,
            "stream": True,
            "temperature": 0.0,
        }
        if payload.diagnostic_top_logprobs:
            request_body.update({
                "n_probs": payload.diagnostic_top_logprobs,
                "post_sampling_probs": False,
            })
        if payload.speculative is not None:
            request_body.update(payload.speculative.body_fields())
        body = _canonical(request_body)
        connection = http.client.HTTPConnection(
            host, port, timeout=payload.timeout_s
        )
        final = None
        tokens: list[int] = []
        text_parts: list[str] = []
        task_by_slot: dict[int, int] = {}
        slot_probe_errors: list[str] = []
        first_reported = False
        last_progress = -1
        terminal_progress_reported = False
        try:
            control_check()
            try:
                connection.connect()
            except OSError as error:
                raise HttpEndpointFailure(
                    "connect",
                    error,
                    error.errno in {
                        errno.ECONNREFUSED,
                        errno.EHOSTUNREACH,
                        errno.ENETUNREACH,
                    },
                ) from error
            try:
                control_check()
                connection.request(
                    "POST",
                    "/completion",
                    body=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Scheduler-Request-ID": payload.request_id,
                        **extra_headers,
                    },
                )
            except OSError as error:
                raise HttpEndpointFailure("request", error, False) from error
            response = connection.getresponse()
            if response.status != 200:
                raise PhysicalAdapterError(
                    f"completion status is {response.status}"
                )
            with payload.stream_path.open("xb") as raw_stream:
                while raw_line := response.readline():
                    control_check()
                    raw_stream.write(raw_line)
                    payload_line = raw_line.decode("utf-8").strip()
                    if not payload_line.startswith("data:"):
                        continue
                    encoded = payload_line[5:].strip()
                    if not encoded or encoded == "[DONE]":
                        continue
                    value = json.loads(encoded)
                    if type(value) is not dict or "error" in value:
                        # Keep the server's own error text (for example
                        # "Compute aborted." or "helper <label>: ...") so a
                        # drop-recovery rig can classify the failure.
                        raise CompletionStreamError(
                            "completion stream chunk is invalid",
                            server_error_message=_stream_error_message(
                                value
                            ),
                        )
                    chunk = value.get("tokens", [])
                    if type(chunk) is not list or any(
                        type(token) is not int for token in chunk
                    ):
                        raise PhysicalAdapterError(
                            "completion stream tokens are invalid"
                        )
                    content = value.get("content", "")
                    if type(content) is not str:
                        raise PhysicalAdapterError(
                            "completion stream text is invalid"
                        )
                    text_parts.append(content)
                    predicted = value.get("tokens_predicted")
                    if (
                        not first_reported
                        and type(predicted) is int
                        and predicted > 0
                        and not value.get("stop", False)
                    ):
                        payload.on_first_token(time.monotonic_ns())
                        first_reported = True
                        try:
                            slots = self._slots_probe(host, port, 1)
                        except BaseException as error:
                            slot_probe_errors.append(
                                f"active:{type(error).__name__}:{error}"
                            )
                        else:
                            processing = {
                                row["id"]: row["id_task"]
                                for row in slots
                                if type(row.get("id")) is int
                                and type(row.get("id_task")) is int
                                and row.get("is_processing") is True
                            }
                            task_by_slot.update(processing)
                            if payload.on_active_batch is not None:
                                payload.on_active_batch(max(1, len(processing)))
                    tokens.extend(chunk)
                    progress_slot = value.get("id_slot")
                    terminal_progress = (
                        bool(value.get("stop", False))
                        or (
                            type(predicted) is int
                            and predicted >= payload.output_tokens
                        )
                    )
                    if (
                        payload.on_decode_progress is not None
                        and type(progress_slot) is int
                        and type(predicted) is int
                        and predicted >= 0
                        and (
                            predicted > last_progress
                            or (
                                terminal_progress
                                and not terminal_progress_reported
                            )
                        )
                    ):
                        progress_slot = self._progress_slot(
                            progress_slot, task_by_slot
                        )
                        payload.on_decode_progress(
                            progress_slot,
                            predicted,
                            time.monotonic_ns(),
                            terminal_progress,
                        )
                        last_progress = max(last_progress, predicted)
                        terminal_progress_reported = (
                            terminal_progress_reported or terminal_progress
                        )
                    if value.get("stop", False):
                        final = value
            control_check()
        finally:
            connection.close()
        if final is None:
            raise PhysicalAdapterError("completion final chunk is absent")
        timings = final.get("timings")
        final_slot = final.get("id_slot")
        if type(final_slot) is int and final_slot not in task_by_slot:
            for _ in range(8):
                control_check()
                try:
                    slots = self._slots_probe(host, port, 1)
                except BaseException as error:
                    slot_probe_errors.append(
                        f"complete:{type(error).__name__}:{error}"
                    )
                else:
                    task_by_slot.update({
                        row["id"]: row["id_task"]
                        for row in slots
                        if type(row.get("id")) is int
                        and type(row.get("id_task")) is int
                        and row["id_task"] >= 0
                    })
                    if final_slot in task_by_slot:
                        break
                time.sleep(0.1)
        if not (
            type(timings) is dict
            and type(final_slot) is int
            and final_slot in task_by_slot
            and final.get("model") == payload.expected_model_alias
            and timings.get("prompt_n") == payload.input_tokens
            and timings.get("predicted_n") == payload.output_tokens
            and len(tokens) == payload.output_tokens
        ):
            raise PhysicalAdapterError("completion accounting differs")
        digest = hashlib.sha256(payload.stream_path.read_bytes()).hexdigest()
        output_text = "".join(text_parts)
        quality = (
            assess_semantic_output(output_text, tuple(tokens))
            if payload.quality_mode == "semantic"
            else accounting_output_assessment(output_text, tuple(tokens))
        )
        if not quality.accepted:
            raise PhysicalAdapterError(
                "completion semantic quality failed: "
                + ",".join(quality.reasons)
            )
        result = {
            "endpoint_slot_id": final_slot,
            "endpoint_task_id": task_by_slot[final_slot],
            "endpoint_model_alias": final["model"],
            "endpoint_slot_probe_errors": slot_probe_errors,
            "output_quality": quality.to_json(),
            "output_text": output_text,
            "predicted_ms": timings["predicted_ms"],
            "prompt_ms": timings["prompt_ms"],
            "runtime_prompt_tokens": timings["prompt_n"],
            "stream_sha256": digest,
            "tokens": tokens,
        }
        if payload.speculative is not None:
            result["speculative"] = speculative_completion_statistics(
                timings, payload.output_tokens, payload.speculative
            )
        return result

    @staticmethod
    def _runtime_stats(value: object) -> dict[str, int]:
        if type(value) is not dict:
            raise PhysicalAdapterError("FFN runtime stats are invalid")
        required_names = {
            "calls",
            "desktop_compute_us",
            "download_bytes",
            "exposed_tail_us",
            "phone_compute_us",
            "rpc_us",
            "upload_bytes",
            "usb_transfer_us",
            "useful_overlap_us",
        }
        optional_names = {
            "batched_calls",
            "configured_queue_depth",
            # decode-only relocation: the server's release state rides in every acknowledgement
            "dormant_host_columns",
            "dormant_host_share",
            "dormant_layer_mask",
            "dormant_release_elapsed_us",
            "dormant_release_generation",
            "dormant_released_bytes",
            "input_rows",
            "maximum_active_slots",
            "maximum_outstanding_transfers",
            "maximum_tokens",
            "transfer_subrequests",
            "usb_d2h_us",
            "usb_h2d_us",
        }
        if (
            not required_names.issubset(value)
            or not set(value).issubset(required_names | optional_names)
            or any(
            type(value[name]) is not int or value[name] < 0
            for name in value
            )
        ):
            raise PhysicalAdapterError("FFN runtime stats are invalid")
        return {
            name: value.get(name, 0)
            for name in sorted(required_names | optional_names)
        }

    @staticmethod
    def apply_ffn_control(
        endpoint: str,
        control: AdaptiveDecodeControl,
        *,
        timeout_s: float = 5,
    ) -> tuple[dict[str, object], int]:
        if not isinstance(control, AdaptiveDecodeControl) or timeout_s <= 0:
            raise PhysicalAdapterError("FFN runtime control is invalid")
        host, port = parse_http_endpoint(endpoint)
        connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions/control",
                body=_canonical(control.to_server_json()),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            content = response.read()
            if response.status != 200:
                raise _server_control_failure(PhysicalAdapterError(
                    f"FFN control status is {response.status}"
                ))
            value = json.loads(content)
            if (
                type(value) is not dict
                or value.get("success") is not True
                or type(value.get("slot_id")) is not int
                or type(value.get("plan_generation")) is not int
                or type(value.get("applied_token_index")) is not int
                or type(value.get("policy_hash")) is not str
            ):
                message = (
                    value.get("message", "invalid acknowledgement")
                    if type(value) is dict else "invalid acknowledgement"
                )
                if str(message) in {
                    "request and active slot differ",
                    "FFN control request and active slot differ",
                }:
                    raise _server_control_failure(StalePhysicalSlotError(
                        "FFN control failed: " + str(message)
                    ), message)
                raise _server_control_failure(PhysicalAdapterError(
                    "FFN control failed: " + str(message)
                ), message)
            value["runtime_stats"] = LlamaCppHttpClient._runtime_stats(
                value.get("runtime_stats")
            )
            return value, time.monotonic_ns()
        finally:
            connection.close()

    @staticmethod
    def read_ffn_stats(
        endpoint: str,
        request_id: str,
        slot_id: int,
        *,
        timeout_s: float = 5,
    ) -> tuple[dict[str, object], int]:
        if (
            type(request_id) is not str
            or not request_id
            or not request_id.isascii()
            or type(slot_id) is not int
            or slot_id < 0
            or timeout_s <= 0
        ):
            raise PhysicalAdapterError("FFN stats request is invalid")
        host, port = parse_http_endpoint(endpoint)
        connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions/control",
                body=_canonical({
                    "action": "ffn_split_stats",
                    "request_id": request_id,
                    "slot_id": slot_id,
                }),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            content = response.read()
            if response.status != 200:
                raise _server_control_failure(PhysicalAdapterError(
                    f"FFN stats status is {response.status}"
                ))
            value = json.loads(content)
            if (
                type(value) is not dict
                or value.get("success") is not True
                or type(value.get("slot_id")) is not int
                or type(value.get("plan_generation")) is not int
                or type(value.get("applied_token_index")) is not int
                or type(value.get("policy_hash")) is not str
            ):
                message = (
                    value.get("message", "invalid FFN stats")
                    if type(value) is dict else "invalid FFN stats"
                )
                if str(message) in {
                    "request and active slot differ",
                    "FFN stats request and active slot differ",
                }:
                    raise _server_control_failure(StalePhysicalSlotError(
                        "FFN stats failed: " + str(message)
                    ), message)
                raise _server_control_failure(PhysicalAdapterError(
                    "FFN stats failed: " + str(message)
                ), message)
            value["runtime_stats"] = LlamaCppHttpClient._runtime_stats(
                value.get("runtime_stats")
            )
            return value, time.monotonic_ns()
        finally:
            connection.close()

    @staticmethod
    def _cohort_members(
        members: Sequence[tuple[str, int]],
    ) -> tuple[tuple[str, int], ...]:
        rows = tuple(members)
        if (
            not 2 <= len(rows) <= 8
            or any(
                type(row) is not tuple
                or len(row) != 2
                or type(row[0]) is not str
                or not row[0]
                or not row[0].isascii()
                or type(row[1]) is not int
                or row[1] < 0
                for row in rows
            )
        ):
            raise PhysicalAdapterError("FFN cohort members are invalid")
        if (
            len({row[0] for row in rows}) != len(rows)
            or len({row[1] for row in rows}) != len(rows)
        ):
            raise PhysicalAdapterError("FFN cohort members are invalid")
        return rows

    @staticmethod
    def _parse_cohort_response(
        value: object,
        members: tuple[tuple[str, int], ...],
        *,
        policy_hash: str | None,
        plan_generation: int | None,
    ) -> dict[str, object]:
        if type(value) is not dict:
            raise PhysicalAdapterError("FFN cohort response is invalid")
        rows = value.get("cohort_members")
        if (
            value.get("success") is not True
            or type(rows) is not list
            or len(rows) != len(members)
            or type(value.get("slot_id")) is not int
            or type(value.get("plan_generation")) is not int
            or type(value.get("applied_token_index")) is not int
            or type(value.get("policy_hash")) is not str
        ):
            message = value.get("message", "invalid acknowledgement")
            raise _server_control_failure(PhysicalAdapterError(
                "FFN cohort control failed: " + str(message)
            ), message)
        observed = []
        for expected, row in zip(members, rows):
            if (
                type(row) is not dict
                or row.get("request_id") != expected[0]
                or row.get("slot_id") != expected[1]
                or type(row.get("applied_token_index")) is not int
                or row["applied_token_index"] < 0
                or type(row.get("plan_generation")) is not int
                or row["plan_generation"] < 0
            ):
                raise PhysicalAdapterError(
                    "FFN cohort acknowledgement members differ"
                )
            if (
                plan_generation is not None
                and row["plan_generation"] != plan_generation
            ):
                raise PhysicalAdapterError(
                    "FFN cohort acknowledgement generation differs"
                )
            observed.append(dict(row))
        if (
            value["slot_id"] != members[0][1]
            or value["applied_token_index"] != min(
                row["applied_token_index"] for row in observed
            )
            or (
                policy_hash is not None
                and value["policy_hash"] != policy_hash
            )
            or (
                plan_generation is not None
                and value["plan_generation"] != plan_generation
            )
        ):
            raise PhysicalAdapterError(
                "FFN cohort acknowledgement identity differs"
            )
        result = dict(value)
        result["cohort_members"] = observed
        result["runtime_stats"] = LlamaCppHttpClient._runtime_stats(
            value.get("runtime_stats")
        )
        return result

    @staticmethod
    def apply_ffn_cohort_control(
        endpoint: str,
        control: AdaptiveDecodeControl,
        members: Sequence[tuple[str, int]],
        *,
        timeout_s: float = 5,
    ) -> tuple[dict[str, object], int]:
        if not isinstance(control, AdaptiveDecodeControl) or timeout_s <= 0:
            raise PhysicalAdapterError("FFN cohort control is invalid")
        rows = LlamaCppHttpClient._cohort_members(members)
        body = control.to_server_json()
        body["action"] = "ffn_split_cohort"
        body["members"] = [
            {"request_id": request_id, "slot_id": slot_id}
            for request_id, slot_id in rows
        ]
        host, port = parse_http_endpoint(endpoint)
        connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions/control",
                body=_canonical(body),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            content = response.read()
            if response.status != 200:
                raise _server_control_failure(PhysicalAdapterError(
                    f"FFN cohort control status is {response.status}"
                ))
            value = LlamaCppHttpClient._parse_cohort_response(
                json.loads(content),
                rows,
                policy_hash=control.policy.policy_hash,
                plan_generation=control.plan_generation,
            )
            return value, time.monotonic_ns()
        finally:
            connection.close()

    @staticmethod
    def read_ffn_cohort_stats(
        endpoint: str,
        members: Sequence[tuple[str, int]],
        *,
        timeout_s: float = 5,
    ) -> tuple[dict[str, object], int]:
        rows = LlamaCppHttpClient._cohort_members(members)
        if timeout_s <= 0:
            raise PhysicalAdapterError("FFN cohort stats request is invalid")
        host, port = parse_http_endpoint(endpoint)
        connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions/control",
                body=_canonical({
                    "action": "ffn_split_cohort_stats",
                    "members": [
                        {"request_id": request_id, "slot_id": slot_id}
                        for request_id, slot_id in rows
                    ],
                }),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            content = response.read()
            if response.status != 200:
                raise _server_control_failure(PhysicalAdapterError(
                    f"FFN cohort stats status is {response.status}"
                ))
            value = LlamaCppHttpClient._parse_cohort_response(
                json.loads(content),
                rows,
                policy_hash=None,
                plan_generation=None,
            )
            return value, time.monotonic_ns()
        finally:
            connection.close()


class _AdaptivePayloadController:
    _GAUGE_NAMES = (
        "configured_queue_depth",
        "maximum_active_slots",
        "maximum_outstanding_transfers",
        "maximum_tokens",
        # decode-only relocation state: resets to zero when the share is populated again;
        # only dormant_release_generation is a monotonic counter
        "dormant_host_share",
        "dormant_layer_mask",
        "dormant_host_columns",
        "dormant_released_bytes",
        "dormant_release_elapsed_us",
    )
    _CUMULATIVE_NAMES = ("batched_calls", "transfer_subrequests")

    def __init__(
        self,
        backend,
        command: PhysicalExecutionCommand,
        payload: LlamaCppCompletionPayload,
        cohort: DecodeCohortPolicyView | None,
        fence_sink: Callable[[Mapping[str, object]], None] | None,
    ) -> None:
        self.backend = backend
        self.command = command
        self.payload = payload
        self.cohort = cohort
        self.fence_sink = fence_sink
        self.scheduler = backend._adaptive_scheduler
        if self.scheduler is None:
            raise PhysicalAdapterError(
                "adaptive HTTP execution lacks a scheduler"
            )
        self.original_progress = payload.on_decode_progress
        self.started = False
        self.closed = False
        self.pending_stale_boundary = None
        self.cohort_contract = command.decode_cohort
        original_members = (
            () if self.cohort_contract is None
            else tuple(self.cohort_contract["member_request_ids"])
        )
        if (cohort is None) != (self.cohort_contract is None):
            raise PhysicalAdapterError(
                "adaptive decode cohort binding differs"
            )
        self.live_active_batch = (
            command.adapter_parameters.get(
                "active_request_batch_size", 1
            )
            if cohort is None else len(original_members)
        )
        self.window_active_batch = self.live_active_batch
        self.live_members: tuple[tuple[int, int], ...] | None = None
        self.live_context_available = True
        if (
            type(self.live_active_batch) is not int
            or self.live_active_batch <= 0
        ):
            raise PhysicalAdapterError("active request batch is invalid")
        names = (
            "calls",
            "desktop_compute_us",
            "download_bytes",
            "exposed_tail_us",
            "input_rows",
            "phone_compute_us",
            "rpc_us",
            "upload_bytes",
            "usb_d2h_us",
            "usb_h2d_us",
            "usb_transfer_us",
            "useful_overlap_us",
            *self._GAUGE_NAMES,
            *self._CUMULATIVE_NAMES,
        )
        self.last_stats = {name: 0 for name in names}
        self.last_cohort_tokens: dict[str, int] | None = None
        self.accounting_members: tuple[str, ...] = ()
        self.last_window_end_token: int | None = None
        self.last_window_finished_at_us: int | None = None
        self.request_queue_delay_us = 0
        self.protected_interference_us = 0
        self._load_ticket_cost_context()

    def _load_ticket_cost_context(self) -> None:
        runtime_ticket = getattr(self.scheduler, "runtime_ticket", None)
        if not callable(runtime_ticket):
            return
        ticket = runtime_ticket(self.command.request_id)
        self.request_queue_delay_us = max(
            0, ticket.decision.start_us - ticket.request.arrival_us
        )
        selected_cost = next(
            row for row in ticket.cost_estimates.estimates
            if row.route_id == ticket.decision.route_id
        )
        marginal = selected_cost.details.get("marginal_system_cost")
        self.protected_interference_us = (
            int(marginal.get("interference_upper_us", 0))
            if isinstance(marginal, Mapping) else 0
        )

    def _relative_us(self, value_ns: int) -> int:
        return self.backend._relative_us(value_ns)

    def _record_final_d2h_fence(
        self,
        raw: Mapping[str, object],
        observed_ns: int,
        request_id: str,
        slot_id: int,
        source: str,
        members: Sequence[tuple[str, int]] = (),
    ) -> None:
        if raw.get("slot_id") != slot_id:
            raise PhysicalAdapterError(
                "FFN final D2H fence slot differs"
            )
        runtime_stats = raw.get("runtime_stats")
        if not isinstance(runtime_stats, Mapping):
            raise PhysicalAdapterError(
                "FFN final D2H fence statistics are absent"
            )
        if self.fence_sink is None:
            return
        self.fence_sink({
            "applied_token_index": int(raw["applied_token_index"]),
            "cohort_members": [
                {"request_id": member_id, "slot_id": member_slot}
                for member_id, member_slot in members
            ],
            "completed_phone_calls": int(runtime_stats["calls"]),
            "download_bytes": int(runtime_stats["download_bytes"]),
            "final_d2h_completed": True,
            "observed_at_us": self._relative_us(observed_ns),
            "plan_generation": int(raw["plan_generation"]),
            "policy_hash": str(raw["policy_hash"]),
            "request_id": request_id,
            "runtime_stats_sha256": (
                "sha256:"
                + hashlib.sha256(_canonical(runtime_stats)).hexdigest()
            ),
            "schema": "scheduler-request-slot-d2h-fence-v1",
            "slot_id": slot_id,
            "source": source,
            "usb_d2h_us": int(runtime_stats["usb_d2h_us"]),
        })

    def _stats_delta(
        self, current: Mapping[str, int]
    ) -> dict[str, int]:
        moved_backward = tuple(
            name for name in sorted(self.last_stats)
            if name not in self._GAUGE_NAMES
            and current.get(name, 0) < self.last_stats[name]
        )
        if moved_backward:
            detail = ",".join(
                name + ":" + str(self.last_stats[name])
                + "->" + str(current[name])
                for name in moved_backward
            )
            raise PhysicalAdapterError(
                "FFN runtime counters moved backward: " + detail
            )
        if (
            self.last_stats["configured_queue_depth"] > 0
            and current["configured_queue_depth"] == 0
        ):
            raise PhysicalAdapterError(
                "FFN runtime transport became unavailable"
            )
        result = {
            name: current.get(name, 0) - self.last_stats[name]
            for name in self.last_stats
            if name not in self._GAUGE_NAMES
        }
        # gauges (including the dormant release state) may be absent from older servers' statistics
        result.update({
            name: current.get(name, 0) for name in self._GAUGE_NAMES
        })
        return result

    def _record_phone_window(
        self,
        work_id: str,
        started_at_us: int,
        finished_at_us: int,
        delta: Mapping[str, int],
        policy=None,
    ) -> None:
        record = getattr(
            self.backend._energy_meter,
            "record_phone_activity_duration",
            None,
        )
        if not callable(record):
            return
        active_us = min(
            max(0, finished_at_us - started_at_us),
            max(
                delta["rpc_us"],
                delta["usb_transfer_us"] + delta["phone_compute_us"],
            ),
        )
        owners = getattr(policy, "device_layer_masks", ()) if policy is not None else ()
        from .phone_helpers import phone_helper_bindings_from_parameters
        try:
            bindings = (() if not owners else
                        phone_helper_bindings_from_parameters(self.command.adapter_parameters) or ())
        except PhysicalAdapterError:
            bindings = ()
        # a device set without the primary phone (co-helpers only) leaves the primary idle
        if not bindings or bindings[0].device_id in {device for device, _mask in owners}:
            record(
                work_id,
                "adaptive_phone_window",
                self.backend._epoch_ns + started_at_us * 1_000,
                self.backend._epoch_ns + finished_at_us * 1_000,
                active_us,
            )
        helper_record = getattr(self.backend._energy_meter, "record_helper_phone_window", None)
        if callable(helper_record) and active_us:
            if not owners:
                bindings = phone_helper_bindings_from_parameters(self.command.adapter_parameters) or ()
                owners = tuple((row.device_id, row.layer_mask) for row in
                               bindings[1:])
            helper_record(work_id, self.backend._epoch_ns + started_at_us * 1000,
                          self.backend._epoch_ns + finished_at_us * 1000,
                          tuple(device for device, mask in owners if mask))

    def _cohort_rows(self) -> tuple[tuple[str, int], ...]:
        return () if self.cohort is None else self.cohort.member_slots()

    def _cohort_stats_member(
        self, members: tuple[tuple[str, int], ...]
    ) -> tuple[str, int]:
        if not members:
            raise PhysicalAdapterError(
                "FFN cohort has no live statistics member"
            )
        return next(
            (
                row for row in members
                if row[0] == self.command.request_id
            ),
            members[0],
        )

    def _cohort_positions(
        self,
        raw: Mapping[str, object],
        members: tuple[tuple[str, int], ...],
    ) -> dict[str, int]:
        if self.cohort is None:
            return {}
        if len(members) == 1:
            if raw.get("slot_id") != members[0][1]:
                raise PhysicalAdapterError(
                    "FFN singleton handoff slot differs"
                )
            return {
                members[0][0]: int(raw["applied_token_index"])
            }
        rows = raw.get("cohort_members")
        if type(rows) is not list:
            raise PhysicalAdapterError(
                "FFN cohort token positions are absent"
            )
        result = {
            str(row["request_id"]): int(row["applied_token_index"])
            for row in rows
        }
        if tuple(result) != tuple(row[0] for row in members):
            raise PhysicalAdapterError(
                "FFN cohort token positions differ"
            )
        return result

    def _accounting_tokens(
        self, current: Mapping[str, int], fallback: int
    ) -> int | None:
        if self.cohort is None:
            return None
        members = tuple(current)
        if (
            self.last_cohort_tokens is None
            or members != self.accounting_members
        ):
            self.accounting_members = members
            self.last_cohort_tokens = dict(current)
            return max(fallback, fallback * len(members))
        count = sum(
            current[request_id]
                - self.last_cohort_tokens[request_id]
            for request_id in members
        )
        if count < fallback:
            raise PhysicalAdapterError(
                "FFN cohort accounting moved backward"
            )
        self.last_cohort_tokens = dict(current)
        return count

    def _raw_observation(
        self,
        energy,
        delta: Mapping[str, int],
        *,
        evidence_id: str,
        token_count: int,
        positions: Mapping[str, int],
        next_active_batch: int | None = None,
        membership_changed: bool = False,
        execution_context_available: bool = True,
        failure_reason: str | None = None,
    ) -> AdaptiveDecodeRawWindowObservation:
        members = tuple(positions)
        cohort_accounting = (
            self.cohort is not None and len(members) >= 2
        )
        return AdaptiveDecodeRawWindowObservation(
            fleet_energy_uj_by_domain=energy.fleet_energy_uj_by_domain,
            energy_boundary_id=energy.energy_boundary_id,
            energy_attribution_kind=energy.attribution_kind,
            phone_compute_us=delta["phone_compute_us"],
            usb_transfer_us=delta["usb_transfer_us"],
            rpc_us=delta["rpc_us"],
            exposed_tail_us=delta["exposed_tail_us"],
            output_valid=True,
            evidence_ids=tuple(sorted(set((
                *energy.measurement_evidence_ids, evidence_id,
            )))),
            usb_upload_bytes=delta["upload_bytes"],
            usb_download_bytes=delta["download_bytes"],
            desktop_compute_us=delta["desktop_compute_us"],
            useful_overlap_us=delta["useful_overlap_us"],
            request_queue_delay_us=self.request_queue_delay_us,
            protected_interference_us=self.protected_interference_us,
            active_batch=(
                len(members)
                if self.cohort is not None
                else self.window_active_batch
            ),
            next_active_batch=next_active_batch,
            membership_changed=membership_changed,
            execution_context_available=execution_context_available,
            usb_h2d_us=delta["usb_h2d_us"],
            usb_d2h_us=delta["usb_d2h_us"],
            accounting_token_count=self._accounting_tokens(
                positions, token_count
            ),
            cohort_id=(
                None if not cohort_accounting
                else str(self.cohort_contract["cohort_id"])
            ),
            cohort_member_request_ids=(
                members if cohort_accounting else ()
            ),
            energy_owner_request_id=(
                members[0] if cohort_accounting else None
            ),
            configured_queue_depth=delta["configured_queue_depth"],
            maximum_active_slots=delta["maximum_active_slots"],
            maximum_outstanding_transfers=delta[
                "maximum_outstanding_transfers"
            ],
            batched_calls=delta["batched_calls"],
            transfer_subrequests=delta["transfer_subrequests"],
            maximum_tokens=delta["maximum_tokens"],
            completed_phone_calls=delta["calls"],
            completed_phone_input_rows=delta["input_rows"],
            failure_reason=failure_reason,
        )

    def _control_members(self) -> tuple[tuple[str, int], ...]:
        if self.cohort is None:
            return ()
        return (
            self.cohort.surviving_member_slots()
            if self.cohort.membership_changed
            else self._cohort_rows()
        )

    @staticmethod
    def _server_control(
        control: AdaptiveDecodeControl,
        members: tuple[tuple[str, int], ...],
    ) -> AdaptiveDecodeControl:
        if not members:
            return control
        return replace(
            control,
            request_id=members[0][0],
            slot_id=members[0][1],
        )

    def _control_timeout_s(self) -> float:
        # Control tasks share the server decode thread with in-flight prefill.
        return min(
            self.payload.timeout_s,
            max(5.0, self.backend._cohort_service_timeout_s(self.command)),
        )

    def _send_control(
        self,
        control: AdaptiveDecodeControl,
        members: tuple[tuple[str, int], ...],
    ):
        server_control = self._server_control(control, members)
        self.backend._guard_control(self.command, server_control)
        self.backend._record_adaptive_timing({
            "kind": "CONTROL_ISSUED",
            "request_id": self.command.request_id,
            "ticket_id": self.command.ticket_id,
            "observed_at_us": self._relative_us(time.monotonic_ns()),
            "control": server_control.to_json(),
            "response_timeout_s": self._control_timeout_s(),
        })
        if self.cohort is None or len(members) == 1:
            raw, acknowledged_ns = self.backend._client.apply_ffn_control(
                self.command.endpoint, server_control,
                timeout_s=self._control_timeout_s(),
            )
        else:
            raw, acknowledged_ns = (
                self.backend._client.apply_ffn_cohort_control(
                    self.command.endpoint, server_control, members,
                    timeout_s=self._control_timeout_s(),
                )
            )
        return raw, acknowledged_ns, server_control

    def _apply_control(self, control: AdaptiveDecodeControl) -> bool:
        members = self._control_members()
        try:
            raw, acknowledged_ns, server_control = self._send_control(
                control, members
            )
        except StalePhysicalSlotError:
            discard = getattr(
                self.scheduler,
                "discard_stale_adaptive_decode_control",
                None,
            )
            if not callable(discard):
                raise
            discard(
                self.command.request_id,
                control,
                "stale_slot_control_discarded",
                at_us=self._relative_us(time.monotonic_ns()),
            )
            return True
        except BaseException as error:
            directive = self.scheduler.fail_adaptive_decode_control(
                self.command.request_id,
                control,
                f"{type(error).__name__}:{error}",
                at_us=self._relative_us(time.monotonic_ns()),
            )
            if directive.control is None:
                raise
            control = directive.control
            members = self._control_members()
            raw, acknowledged_ns, server_control = self._send_control(
                control, members
            )
        acknowledgement = AdaptiveDecodePolicyAck(
            request_id=self.command.request_id,
            slot_id=control.slot_id,
            plan_generation=int(raw["plan_generation"]),
            applied_token_index=int(raw["applied_token_index"]),
            applied_at_us=self._relative_us(acknowledged_ns),
            policy_hash=str(raw["policy_hash"]),
        )
        self.backend.notify_control_ack(self.command.endpoint, self.command.request_id, raw)
        current_stats = raw["runtime_stats"]
        self._record_final_d2h_fence(
            raw,
            acknowledged_ns,
            server_control.request_id,
            server_control.slot_id,
            "control_ack",
            members,
        )
        delta = self._stats_delta(current_stats)
        positions = self._cohort_positions(raw, members)
        transition_observation = None
        if self.last_window_end_token is not None:
            if (
                self.last_window_finished_at_us is None
                or acknowledgement.applied_token_index
                    < self.last_window_end_token
            ):
                raise PhysicalAdapterError(
                    "FFN control acknowledgement moved backward"
                )
            if (
                acknowledgement.applied_token_index
                > self.last_window_end_token
            ):
                self._record_phone_window(
                    self.command.ticket_id
                        + ":control:"
                        + str(self.last_window_end_token),
                    self.last_window_finished_at_us,
                    acknowledgement.applied_at_us,
                    delta,
                )
                energy = self.backend._energy_meter.measure(
                    self.backend._epoch_ns
                        + self.last_window_finished_at_us * 1000,
                    self.backend._epoch_ns
                        + acknowledgement.applied_at_us * 1000,
                )
                transition_observation = self._raw_observation(
                    energy,
                    delta,
                    evidence_id="physical:control-transition-ack",
                    token_count=(
                        acknowledgement.applied_token_index
                        - self.last_window_end_token
                    ),
                    positions=positions,
                )
        if self.cohort is not None and transition_observation is None:
            self.last_cohort_tokens = dict(positions)
        directive = self.scheduler.acknowledge_adaptive_decode_control(
            self.command.request_id,
            acknowledgement,
            transition_observation=transition_observation,
        )
        self.last_stats = dict(current_stats)
        self.last_window_end_token = acknowledgement.applied_token_index
        self.last_window_finished_at_us = acknowledgement.applied_at_us
        if directive is not None and directive.reason in {
            "TAIL_SEALED", "COHORT_MEMBERSHIP_CHANGED"
        }:
            return True
        if directive is not None and directive.control is not None:
            return self._apply_control(directive.control)
        if (
            self.cohort is None
            and directive is not None
            and directive.target_token_index is not None
            and acknowledgement.applied_token_index >= self._tail_seal_at()
        ):
            self.scheduler.seal_adaptive_decode_tail(
                self.command.request_id,
                slot_id=acknowledgement.slot_id,
                token_index=acknowledgement.applied_token_index,
                reason="server_release_guard",
            )
            return True
        return False

    def _read_boundary_stats(self, boundary):
        if self.cohort is None:
            members = ()
            stats_members = ()
            request_id = self.command.request_id
            slot_id = boundary.slot_id
        else:
            members = self._cohort_rows()
            handoff = self.cohort.membership_changed
            stats_members = (
                self.cohort.surviving_member_slots()
                if handoff else members
            )
            request_id, slot_id = self._cohort_stats_member(
                stats_members
            )
        raw, observed_ns = self.backend._client.read_ffn_stats(
            self.command.endpoint, request_id, slot_id,
            timeout_s=self._control_timeout_s(),
        )
        self.backend.notify_control_ack(self.command.endpoint, self.command.request_id, raw)
        return (
            raw,
            observed_ns,
            members,
            stats_members,
            request_id,
            slot_id,
            self.cohort is not None
                and self.cohort.membership_changed,
        )

    def _discard_stale_boundary(
        self, boundary, *, terminal_token_index=None, terminal_at_us=None,
    ):
        discard = getattr(
            self.scheduler,
            "discard_stale_adaptive_decode_window",
            None,
        )
        if not callable(discard):
            raise
        released = terminal_token_index is not None
        end_us = terminal_at_us if released else boundary.finished_at_us
        reason = (
            ("released_slot_baseline_tail" if boundary.policy.baseline
             else "released_slot_phone_tail")
            if released else "stale_slot_stats_discarded"
        )
        energy = self.backend._energy_meter.measure(
            self.backend._epoch_ns + boundary.started_at_us * 1000,
            self.backend._epoch_ns + end_us * 1000,
        )
        empty_delta = {
            name: (
                self.last_stats[name]
                if name in self._GAUGE_NAMES else 0
            )
            for name in self.last_stats
        }
        positions = (
            {}
            if self.cohort is None
            else self.cohort.token_positions()
        )
        directive = discard(
            self.command.request_id,
            boundary,
            self._raw_observation(
                energy,
                empty_delta,
                evidence_id=("physical:terminal-release-confirmed" if released
                             else "physical:stale-slot-stats-discarded"),
                token_count=(terminal_token_index - boundary.token_start
                             if released else boundary.token_count),
                positions=positions,
                failure_reason=reason,
            ),
            reason,
            **({"terminal_token_index": terminal_token_index,
                "terminal_at_us": terminal_at_us} if released else {}),
        )
        if released and not boundary.policy.baseline:
            self.backend._record_adaptive_timing({
                "kind": "SLOT_STATS_UNMEASURED_TAIL",
                "request_id": self.command.request_id,
                "ticket_id": self.command.ticket_id,
                "observed_at_us": end_us,
                "slot_id": boundary.slot_id,
                "measured_through_token": boundary.token_start,
                "terminal_token_index": terminal_token_index,
                "policy_hash": boundary.policy.policy_hash,
                "plan_generation": (boundary.applied_ack.plan_generation
                                    if boundary.applied_ack is not None else None),
                "reason": reason,
            })
        self.last_window_end_token = terminal_token_index if released else boundary.token_end
        self.last_window_finished_at_us = end_us
        return directive

    def _refresh_active_members(self) -> bool:
        if self.cohort is not None:
            return False
        probe = getattr(self.backend._client, "_slots_probe", None)
        if probe is None:
            return self.live_active_batch != self.window_active_batch
        try:
            host, port = parse_http_endpoint(self.command.endpoint)
            slots = probe(host, port, 1)
            members = tuple(sorted(
                (row["id"], row["id_task"]) for row in slots
                if row.get("is_processing") is True
                and type(row.get("id")) is int and type(row.get("id_task")) is int
            ))
            if not members:
                raise PhysicalAdapterError("live decode membership is empty")
        except (OSError, ValueError, PhysicalAdapterError) as error:
            self.live_context_available = False
            self.backend._record_adaptive_timing({
                "kind": "DECODE_CONTEXT_UNAVAILABLE", "request_id": self.command.request_id,
                "observed_at_us": self._relative_us(time.monotonic_ns()),
                "reason": f"{type(error).__name__}:{error}",
                "previous_active_batch": self.window_active_batch,
                "previous_members": self.live_members,
            })
            return False
        recovered = not self.live_context_available
        self.live_context_available = True
        changed = ((self.live_members is not None and members != self.live_members)
                   or len(members) != self.window_active_batch)
        previous_members = self.live_members
        self.live_members = members
        self.live_active_batch = len(members)
        if changed or recovered:
            self.backend._record_adaptive_timing({
                "kind": "DECODE_CONTEXT_CHANGED" if changed else "DECODE_CONTEXT_RECOVERED",
                "request_id": self.command.request_id,
                "observed_at_us": self._relative_us(time.monotonic_ns()),
                "previous_active_batch": self.window_active_batch,
                "previous_members": previous_members,
                "active_batch": len(members), "members": [list(row) for row in members],
            })
        return changed

    def _process_boundary(self, boundary):
        try:
            (
                raw,
                observed_ns,
                _members,
                stats_members,
                stats_request_id,
                stats_slot_id,
                membership_handoff,
            ) = self._read_boundary_stats(boundary)
        except StalePhysicalSlotError:
            baseline_tail = boundary.policy.baseline and (
                boundary.token_end < self._tail_seal_at()
                or boundary.applied_ack is not None
                or self.last_stats.get("calls", 0) > 0
            )
            phone_tail = (
                not boundary.policy.baseline
                and boundary.token_end >= self._tail_seal_at()
                and boundary.token_start == self.last_window_end_token
                and self.last_stats.get("calls", 0) > 0
            )
            if self.cohort is None and (baseline_tail or phone_tail):
                self.pending_stale_boundary = boundary
                self.backend._record_adaptive_timing({
                    "kind": "SLOT_STATS_DEFERRED_UNTIL_TERMINAL",
                    "request_id": self.command.request_id,
                    "ticket_id": self.command.ticket_id,
                    "observed_at_us": self._relative_us(time.monotonic_ns()),
                    "slot_id": boundary.slot_id,
                    "token_index": boundary.token_end,
                    "policy_hash": boundary.policy.policy_hash,
                })
                return None
            return self._discard_stale_boundary(boundary)
        if (
            raw["slot_id"] != stats_slot_id
            or raw["applied_token_index"] < boundary.token_end
            or (
                boundary.applied_ack is not None
                and raw["policy_hash"] != boundary.policy.policy_hash
            )
        ):
            raise PhysicalAdapterError(
                "FFN window stats differ from the applied policy"
            )
        self._record_final_d2h_fence(
            raw,
            observed_ns,
            stats_request_id,
            stats_slot_id,
            "window_stats",
            stats_members,
        )
        current_stats = raw["runtime_stats"]
        delta = self._stats_delta(current_stats)
        self.last_stats = dict(current_stats)
        # Capture request counters before another query can outlive the slot.
        live_membership_changed = self._refresh_active_members()
        self._record_phone_window(
            self.command.ticket_id
                + ":window:"
                + str(boundary.token_start)
                + ":"
                + str(boundary.token_end),
            boundary.started_at_us,
            boundary.finished_at_us,
            delta,
            boundary.policy,
        )
        energy = self.backend._energy_meter.measure(
            self.backend._epoch_ns + boundary.started_at_us * 1000,
            self.backend._epoch_ns + boundary.finished_at_us * 1000,
        )
        positions = (
            self.cohort.token_positions()
            if self.cohort is not None
            else self._cohort_positions(raw, stats_members)
        )
        directive = self.scheduler.record_adaptive_decode_window(
            self.command.request_id,
            boundary,
            self._raw_observation(
                energy,
                delta,
                evidence_id="physical:token-boundary-ack",
                token_count=boundary.token_count,
                positions=positions,
                next_active_batch=(
                    len(stats_members)
                    if (
                        self.cohort is not None
                        and membership_handoff
                    )
                    else self.live_active_batch if live_membership_changed else None
                ),
                membership_changed=live_membership_changed,
                execution_context_available=self.live_context_available,
            ),
        )
        self.window_active_batch = self.live_active_batch
        self.last_window_end_token = boundary.token_end
        self.last_window_finished_at_us = boundary.finished_at_us
        return directive

    def _process_directive(self, directive) -> None:
        while directive is not None:
            if directive.boundary is not None:
                directive = self._process_boundary(directive.boundary)
                continue
            if directive.control is not None:
                if self._apply_control(directive.control):
                    self.closed = True
            return

    def _start_decode(
        self,
        slot_id: int,
        token_index: int,
        at_ns: int,
        terminal: bool,
    ) -> bool:
        if terminal or token_index >= self.payload.output_tokens:
            self.closed = True
            return False
        self._refresh_active_members()
        self.window_active_batch = self.live_active_batch
        if self.cohort is not None:
            members = self._cohort_rows()
            stats_member = self._cohort_stats_member(members)
            initial, observed_ns = self.backend._client.read_ffn_stats(
                self.command.endpoint,
                stats_member[0],
                stats_member[1],
                timeout_s=self._control_timeout_s(),
            )
            self.backend.notify_control_ack(self.command.endpoint, self.command.request_id, initial)
            self.last_stats = dict(initial["runtime_stats"])
            self.last_cohort_tokens = dict(
                self.cohort.token_positions()
            )
            self.accounting_members = tuple(self.last_cohort_tokens)
            token_index = min(self.last_cohort_tokens.values())
            at_ns = max(at_ns, observed_ns)
        self.last_window_end_token = token_index
        self.last_window_finished_at_us = self._relative_us(at_ns)
        directive = self.scheduler.start_adaptive_decode(
            self.command.request_id,
            slot_id=slot_id,
            first_token_index=token_index,
            at_us=self._relative_us(at_ns),
            context_length=self.payload.input_tokens,
            active_batch=self.live_active_batch,
            **({"execution_context_available": False} if not self.live_context_available else {}),
        )
        self.started = True
        self._process_directive(directive)
        return True

    def _tail_seal_at(self) -> int:
        return self.payload.output_tokens - min(2, self.payload.output_tokens - 1)

    def _released_slot_progress(self, slot_id, token_index, at_us, terminal):
        boundary = self.pending_stale_boundary
        if slot_id != boundary.slot_id or not (
            boundary.token_end <= token_index <= self.payload.output_tokens
        ):
            raise PhysicalAdapterError("released slot terminal identity differs")
        if terminal:
            if token_index != self.payload.output_tokens:
                raise PhysicalAdapterError("released slot terminal token count differs")
            self._discard_stale_boundary(
                boundary, terminal_token_index=token_index, terminal_at_us=at_us,
            )
            self.pending_stale_boundary = None
            self.closed = True

    def progress(
        self,
        slot_id: int,
        token_index: int,
        at_ns: int,
        terminal: bool,
    ) -> bool:
        if self.original_progress is not None:
            self.original_progress(slot_id, token_index, at_ns, terminal)
        if self.closed:
            return False
        self.backend._record_adaptive_timing({
            "kind": "DECODE_BOUNDARY_OBSERVED",
            "request_id": self.command.request_id,
            "ticket_id": self.command.ticket_id,
            "observed_at_us": self._relative_us(time.monotonic_ns()),
            "token_observed_at_us": self._relative_us(at_ns),
            "slot_id": slot_id,
            "token_index": token_index,
            "terminal": terminal,
        })
        if not self.started:
            return self._start_decode(
                slot_id, token_index, at_ns, terminal
            )
        progress_at_us = self._relative_us(at_ns)
        if self.pending_stale_boundary is not None:
            self._released_slot_progress(slot_id, token_index, progress_at_us, terminal)
            return False
        if (
            self.last_window_end_token is not None
            and (
                token_index <= self.last_window_end_token
                or (
                    self.last_window_finished_at_us is not None
                    and progress_at_us
                        <= self.last_window_finished_at_us
                )
            )
        ):
            return False
        seal_tail = token_index >= self._tail_seal_at()
        membership_changed = (
            self.cohort is not None
            and self.cohort.membership_changed
        )
        if seal_tail:
            self.scheduler.seal_adaptive_decode_tail(
                self.command.request_id,
                slot_id=slot_id,
                token_index=token_index,
                reason="server_release_guard",
            )
        directive = self.scheduler.adaptive_decode_boundary(
            self.command.request_id,
            slot_id=slot_id,
            token_index=token_index,
            at_us=progress_at_us,
            terminal=(
                seal_tail
                or membership_changed
                or terminal
                or token_index >= self.payload.output_tokens
            ),
        )
        self._process_directive(directive)
        if self.pending_stale_boundary is not None and terminal:
            self._released_slot_progress(slot_id, token_index, progress_at_us, terminal)
        if self.pending_stale_boundary is None and (
            seal_tail
            or (terminal and not membership_changed)
            or token_index >= self.payload.output_tokens
        ):
            self.closed = True
        return True

    def update_active_batch(self, value: int) -> None:
        if type(value) is int and value > 0:
            self.live_active_batch = value
            if not self.started:
                self.window_active_batch = value

    def bind(self) -> LlamaCppCompletionPayload:
        return replace(
            self.payload,
            on_decode_progress=self.progress,
            on_active_batch=self.update_active_batch,
        )


@dataclass(frozen=True)
class _HttpExecutionResult:
    value: Mapping[str, object]
    inference_finished_ns: int
    proof_drain_started_ns: int
    proof_drain_finished_ns: int


class CanonicalHttpExecutionBackend:
    """Execute exact desktop, phone, or composite commands over HTTP."""

    def __init__(
        self,
        client: LlamaCppHttpClient,
        energy_meter: object,
        *,
        epoch_ns: int,
        prepare_transition: Callable[
            [PhysicalTransitionCommand, object, Callable[[], None]],
            bool | None,
        ] | None = None,
        rollback_transition: Callable[
            [PhysicalTransitionCommand], Mapping[str, object]
        ] | None = None,
        on_execution_start: Callable[[PhysicalExecutionCommand], None]
            | None = None,
        on_execution_success: Callable[
            [PhysicalExecutionCommand, Mapping[str, object]],
            Mapping[str, object] | None,
        ] | None = None,
        on_execution_finish: Callable[[PhysicalExecutionCommand], None]
            | None = None,
        on_scheduler_bound: Callable[[object], None] | None = None,
        on_control_ack: Callable[[str, str, Mapping[str, object]], None] | None = None,
        before_prompt: Callable[[str, str], None] | None = None,
        # called as classifier(command, error, started_ns=attempt start)
        failure_classifier: Callable[
            ...,
            PhysicalFailureClassification | None,
        ] | None = None,
        # called as guard(command, control) before any FFN control reaches a server; raising
        # refuses the control (elastic phones S2a: a masked-out helper is never re-owned)
        control_guard: Callable[
            [PhysicalExecutionCommand, AdaptiveDecodeControl], None
        ] | None = None,
    ) -> None:
        if not isinstance(client, LlamaCppHttpClient):
            raise PhysicalAdapterError("HTTP client is invalid")
        if on_control_ack is not None and not callable(on_control_ack):
            raise PhysicalAdapterError("control acknowledgement callback is invalid")
        if before_prompt is not None and not callable(before_prompt):
            raise PhysicalAdapterError("prompt admission callback is invalid")
        if failure_classifier is not None and not callable(failure_classifier):
            raise PhysicalAdapterError("execution failure classifier is invalid")
        if control_guard is not None and not callable(control_guard):
            raise PhysicalAdapterError("FFN control guard is invalid")
        if not callable(getattr(energy_meter, "measure", None)):
            raise PhysicalAdapterError("whole-fleet energy meter is invalid")
        if type(epoch_ns) is not int or epoch_ns < 0:
            raise PhysicalAdapterError("HTTP execution epoch is invalid")
        if prepare_transition is not None and not callable(
            prepare_transition
        ):
            raise PhysicalAdapterError("transition preparer is invalid")
        if rollback_transition is not None and not callable(
            rollback_transition
        ):
            raise PhysicalAdapterError("transition rollback is invalid")
        if on_execution_start is not None and not callable(on_execution_start):
            raise PhysicalAdapterError("execution start callback is invalid")
        if on_execution_success is not None and not callable(
            on_execution_success
        ):
            raise PhysicalAdapterError("execution success callback is invalid")
        if on_execution_finish is not None and not callable(on_execution_finish):
            raise PhysicalAdapterError("execution finish callback is invalid")
        self._client = client
        self._energy_meter = energy_meter
        self._epoch_ns = epoch_ns
        self._prepare_transition = prepare_transition
        self._rollback_transition = rollback_transition
        self._on_start = on_execution_start
        self._on_success = on_execution_success
        self._on_finish = on_execution_finish
        self._on_scheduler_bound = on_scheduler_bound
        # decode-only relocation: every control acknowledgement carries the server's release state, and a
        # prompt may only be sent once the server's released share can be resident again
        self._on_control_ack = on_control_ack
        self._before_prompt = before_prompt
        # elastic_phones.drop_recovery: the rig reads a failed execution
        # (server stderr since its marker, SSE error, process exit) into a
        # helper_lost / server_exited failure; None keeps today's failures.
        self._failure_classifier = failure_classifier
        self._control_guard = control_guard
        self._measurement_started_ns: dict[str, int] = {}
        self._measurement_lock = threading.Lock()
        self._static_control_lock = threading.Lock()
        self._cohort_execution = DecodeCohortExecutionTracker(energy_meter)
        self._cohort_policy = DecodeCohortPolicyCoordinator()
        self._adaptive_scheduler = None
        self._adaptive_timing_events: list[dict[str, object]] = []
        # campaign ``speculative_rows``: rows reserved per endpoint by live speculating requests
        self._speculative_ledger = SpeculativeRowLedger()

    def _speculative_payload(
        self,
        command: PhysicalExecutionCommand,
        payload: LlamaCppCompletionPayload,
    ) -> LlamaCppCompletionPayload:
        """Bind the request's speculative contract from its adapter parameters.

        Without ``speculative_rows`` parameters the payload is returned unchanged. Under
        per-request control the request's draft bound is granted and its rows reserved
        atomically against the endpoint ledger, so concurrent arrivals never exceed the budget."""
        if payload.speculative is not None:
            raise PhysicalAdapterError("completion speculative contract is bound by the backend")
        contracts: list[SpeculativeRequestContract | None] = []

        def grant(reserved: tuple[int, ...]) -> int | None:
            contract = speculative_request_contract(command.adapter_parameters, reserved)
            contracts.append(contract)
            if contract is None or not contract.per_request_control:
                return None
            return contract.reserved_rows

        self._speculative_ledger.admit(command.endpoint, command.request_id, grant)
        contract = contracts[0]
        return payload if contract is None else replace(payload, speculative=contract)

    def _release_speculative_rows(
        self,
        command: PhysicalExecutionCommand,
        payload: LlamaCppCompletionPayload,
    ) -> None:
        if payload.speculative is not None and payload.speculative.per_request_control:
            self._speculative_ledger.release(command.endpoint, command.request_id)

    def _guard_control(
        self, command: PhysicalExecutionCommand, control: AdaptiveDecodeControl
    ) -> None:
        """Refuse an FFN control the rig's guard rejects (before it reaches the server)."""
        if self._control_guard is not None:
            self._control_guard(command, control)

    def notify_control_ack(self, endpoint: str, request_id: str, raw: Mapping[str, object]) -> None:
        """Hand a control acknowledgement's runtime stats to the rig (release credit)."""
        if self._on_control_ack is None or not isinstance(raw, Mapping):
            return
        stats = raw.get("runtime_stats")
        if isinstance(stats, Mapping):
            self._on_control_ack(endpoint, request_id, stats)

    def _record_adaptive_timing(self, event: Mapping[str, object]) -> None:
        with self._measurement_lock:
            self._adaptive_timing_events.append(dict(event))

    @property
    def adaptive_timing_events(self) -> tuple[dict[str, object], ...]:
        with self._measurement_lock:
            return tuple(dict(row) for row in self._adaptive_timing_events)

    def bind_scheduler(self, scheduler: object) -> None:
        required = (
            "start_adaptive_decode",
            "adaptive_decode_boundary",
            "seal_adaptive_decode_tail",
            "record_adaptive_decode_window",
            "acknowledge_adaptive_decode_control",
            "fail_adaptive_decode_control",
            "preview_adaptive_decode_completion",
            "complete_adaptive_decode",
        )
        if any(not callable(getattr(scheduler, name, None)) for name in required):
            raise PhysicalAdapterError("adaptive scheduler interface is invalid")
        progress = getattr(scheduler, "record_runtime_decode_progress", None)
        if progress is not None and not callable(progress):
            raise PhysicalAdapterError(
                "runtime decode progress interface is invalid"
            )
        if self._adaptive_scheduler not in {None, scheduler}:
            raise PhysicalAdapterError("HTTP backend scheduler is already bound")
        self._adaptive_scheduler = scheduler
        bind_meter = getattr(self._energy_meter, "bind_scheduler", None)
        if callable(bind_meter):
            bind_meter(scheduler, self._epoch_ns)
        if self._on_scheduler_bound is not None:
            self._on_scheduler_bound(scheduler)

    def _measure_receipt(self, command, start_ns: int, end_ns: int):
        measure = getattr(self._energy_meter, "measure_receipt", None)
        if callable(measure):
            return measure(command, start_ns, end_ns)
        return self._energy_meter.measure(start_ns, end_ns)

    @staticmethod
    def _adaptive_enabled(command: PhysicalExecutionCommand) -> bool:
        helper = command.helper_envelope
        if helper is not None:
            return (
                helper.helper_plan.execution_contract.execution_mode
                    == "adaptive-split"
                and helper.helper_plan.to_json().get(
                    "assisted_operator_kind"
                ) == "ffn"
                and type(helper.helper_plan.adapter_parameters.get(
                    "phone_device_id"
                )) is str
            )
        if (
            command.execution_contract.execution_mode == "adaptive-split"
            and command.operator_plan.get("assisted_operator_kind") == "ffn"
            and type(command.adapter_parameters.get("phone_device_id")) is str
        ):
            return True
        dormant = dormant_phone_ffn_parameters(
            command.adapter_parameters
        )
        return bool(
            dormant is not None
            and dormant.get("ffn_assistance_phase") == "decode"
            and dormant.get("ffn_runtime_control_protocol")
                == "decode-boundary-v1"
            and type(dormant.get("phone_device_id")) is str
        )

    @staticmethod
    def _static_decode_policy(
        command: PhysicalExecutionCommand,
    ) -> AdaptiveDecodePolicy | None:
        return static_decode_policy(command)

    def _static_decode_payload(
        self,
        command: PhysicalExecutionCommand,
        payload: LlamaCppCompletionPayload,
        on_control_ack: Callable[[Mapping[str, object]], None] | None = None,
    ) -> LlamaCppCompletionPayload:
        policy = self._static_decode_policy(command)
        if policy is None:
            return payload
        original_progress = payload.on_decode_progress
        applied = False

        def progress(
            slot_id: int,
            token_index: int,
            at_ns: int,
            terminal: bool,
        ) -> None:
            nonlocal applied
            if original_progress is not None:
                original_progress(slot_id, token_index, at_ns, terminal)
            if applied or terminal or token_index >= payload.output_tokens:
                return
            control = AdaptiveDecodeControl(
                request_id=command.request_id,
                slot_id=slot_id,
                plan_generation=1,
                policy=policy,
            )
            self._guard_control(command, control)
            with self._static_control_lock:
                raw, _acknowledged_ns = self._client.apply_ffn_control(
                    command.endpoint, control
                )
            if (
                raw.get("slot_id") != slot_id
                or raw.get("plan_generation") != 1
                or raw.get("policy_hash") != policy.policy_hash
                or type(raw.get("applied_token_index")) is not int
                or raw["applied_token_index"] < token_index
                or raw["applied_token_index"] > payload.output_tokens
            ):
                raise PhysicalAdapterError(
                    "static FFN runtime acknowledgement differs"
                )
            self.notify_control_ack(command.endpoint, command.request_id, raw)
            if on_control_ack is not None:
                on_control_ack({
                    "applied_token_index": raw["applied_token_index"],
                    "plan_generation": 1,
                    "policy_hash": policy.policy_hash,
                    "request_id": command.request_id,
                    "slot_id": slot_id,
                })
            applied = True

        return replace(payload, on_decode_progress=progress)

    def _adaptive_payload(
        self,
        command: PhysicalExecutionCommand,
        payload: LlamaCppCompletionPayload,
        execution_started_ns: int,
        cohort: DecodeCohortPolicyView | None = None,
        fence_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> LlamaCppCompletionPayload:
        if not self._adaptive_enabled(command):
            return payload
        if fence_sink is not None and not callable(fence_sink):
            raise PhysicalAdapterError(
                "adaptive D2H fence sink is invalid"
            )
        return _AdaptivePayloadController(
            self,
            command,
            payload,
            cohort,
            fence_sink,
        ).bind()

    def _static_decode_cohort_payload(
        self,
        command: PhysicalExecutionCommand,
        payload: LlamaCppCompletionPayload,
        execution_started_ns: int,
    ) -> LlamaCppCompletionPayload:
        def factory(
            leader_command: PhysicalExecutionCommand,
            leader_payload: object,
            _leader_started_ns: int,
            cohort: DecodeCohortPolicyView,
        ) -> tuple[
            Callable[[int, int, int, bool], bool],
            Callable[[int], None],
        ]:
            if not isinstance(leader_payload, LlamaCppCompletionPayload):
                raise PhysicalAdapterError(
                    "static decode cohort leader payload is invalid"
                )
            policy = static_decode_policy(leader_command)
            if policy is None:
                raise PhysicalAdapterError(
                    "static decode cohort policy is absent"
                )
            applied = False

            def progress(
                slot_id: int,
                token_index: int,
                _at_ns: int,
                terminal: bool,
            ) -> bool:
                nonlocal applied
                if applied:
                    return True
                if terminal:
                    raise PhysicalAdapterError(
                        "static decode cohort ended before control"
                    )
                members = cohort.member_slots()
                positions = cohort.token_positions()
                control = AdaptiveDecodeControl(
                    request_id=leader_command.request_id,
                    slot_id=slot_id,
                    plan_generation=1,
                    policy=policy,
                )
                self._guard_control(leader_command, control)
                raw, _acknowledged_ns = (
                    self._client.apply_ffn_cohort_control(
                        leader_command.endpoint,
                        control,
                        members,
                    )
                )
                rows = raw.get("cohort_members")
                by_request = {
                    row.get("request_id"): row
                    for row in rows
                    if isinstance(row, Mapping)
                } if type(rows) is list else {}
                if (
                    raw.get("policy_hash") != policy.policy_hash
                    or raw.get("plan_generation") != 1
                    or set(by_request) != set(positions)
                    or any(
                        type(by_request[request_id].get(
                            "applied_token_index"
                        )) is not int
                        or by_request[request_id][
                            "applied_token_index"
                        ] < positions[request_id]
                        for request_id in positions
                    )
                ):
                    raise PhysicalAdapterError(
                        "static decode cohort acknowledgement differs"
                    )
                cohort.publish_control_ack(raw)
                applied = True
                return True

            return progress, lambda _value: None

        progress, active_batch = self._cohort_policy.register(
            command,
            payload,
            execution_started_ns,
            factory,
            timeout_s=self._cohort_service_timeout_s(command),
        )
        original_progress = payload.on_decode_progress
        original_active_batch = payload.on_active_batch

        def combined_progress(
            slot_id: int,
            token_index: int,
            at_ns: int,
            terminal: bool,
        ) -> None:
            if original_progress is not None:
                original_progress(slot_id, token_index, at_ns, terminal)
            progress(slot_id, token_index, at_ns, terminal)

        def combined_active_batch(value: int) -> None:
            if original_active_batch is not None:
                original_active_batch(value)
            active_batch(value)

        return replace(
            payload,
            on_decode_progress=combined_progress,
            on_active_batch=combined_active_batch,
        )

    def _adaptive_cohort_payload(
        self,
        command: PhysicalExecutionCommand,
        payload: LlamaCppCompletionPayload,
        execution_started_ns: int,
        fence_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> LlamaCppCompletionPayload:
        def factory(
            leader_command: PhysicalExecutionCommand,
            leader_payload: object,
            leader_started_ns: int,
            cohort: DecodeCohortPolicyView,
        ) -> tuple[
            Callable[[int, int, int, bool], bool],
            Callable[[int], None],
        ]:
            if not isinstance(leader_payload, LlamaCppCompletionPayload):
                raise PhysicalAdapterError(
                    "decode cohort leader payload is invalid"
                )
            controlled = self._adaptive_payload(
                leader_command,
                replace(leader_payload, on_decode_progress=None),
                leader_started_ns,
                cohort=cohort,
                fence_sink=fence_sink,
            )
            if (
                controlled.on_decode_progress is None
                or controlled.on_active_batch is None
            ):
                raise PhysicalAdapterError(
                    "decode cohort policy callbacks are absent"
                )
            return (
                controlled.on_decode_progress,
                controlled.on_active_batch,
            )

        progress, active_batch = self._cohort_policy.register(
            command,
            payload,
            execution_started_ns,
            factory,
            timeout_s=self._cohort_service_timeout_s(command),
        )
        original_progress = payload.on_decode_progress
        original_active_batch = payload.on_active_batch

        def combined_progress(
            slot_id: int,
            token_index: int,
            at_ns: int,
            terminal: bool,
        ) -> None:
            if original_progress is not None:
                original_progress(slot_id, token_index, at_ns, terminal)
            progress(slot_id, token_index, at_ns, terminal)

        def combined_active_batch(value: int) -> None:
            if original_active_batch is not None:
                original_active_batch(value)
            active_batch(value)

        return replace(
            payload,
            on_decode_progress=combined_progress,
            on_active_batch=combined_active_batch,
        )

    def _start_measurement(self, ticket_id: str, started_ns: int) -> None:
        with self._measurement_lock:
            self._measurement_started_ns.setdefault(ticket_id, started_ns)

    def _begin_measurement(self, ticket_id: str) -> int:
        with self._measurement_lock:
            already_started = ticket_id in self._measurement_started_ns
        if not already_started:
            prepare = getattr(self._energy_meter, "prepare", None)
            if callable(prepare):
                prepare()
        started_ns = time.monotonic_ns()
        self._start_measurement(ticket_id, started_ns)
        return started_ns

    def _finish_measurement(self, ticket_id: str, fallback_ns: int) -> int:
        with self._measurement_lock:
            return self._measurement_started_ns.pop(ticket_id, fallback_ns)

    def _discard_measurement(self, ticket_id: str) -> None:
        with self._measurement_lock:
            self._measurement_started_ns.pop(ticket_id, None)

    def _relative_us(self, value_ns: int) -> int:
        return max(0, (value_ns - self._epoch_ns) // 1000)

    def _runtime_progress_payload(
        self,
        command: PhysicalExecutionCommand,
        payload: LlamaCppCompletionPayload,
    ) -> LlamaCppCompletionPayload:
        scheduler = self._adaptive_scheduler
        if scheduler is None:
            return payload
        record = getattr(scheduler, "record_runtime_decode_progress", None)
        if record is None:
            return payload
        original_progress = payload.on_decode_progress

        def progress(
            slot_id: int,
            token_index: int,
            at_ns: int,
            terminal: bool,
        ) -> None:
            record(
                command.request_id,
                token_index=token_index,
                at_us=self._relative_us(at_ns),
            )
            if original_progress is not None:
                original_progress(slot_id, token_index, at_ns, terminal)

        return replace(payload, on_decode_progress=progress)

    @staticmethod
    def _cohort_service_timeout_s(
        command: PhysicalExecutionCommand,
    ) -> float:
        service_upper_us = max(
            1,
            command.planned_finish_upper_us - command.planned_start_us,
        )
        return max(1.0, service_upper_us / 1_000_000)

    @staticmethod
    def _scheduler_headers(
        command: PhysicalExecutionCommand | PhysicalTransitionCommand,
    ) -> dict[str, str]:
        if isinstance(command, PhysicalTransitionCommand):
            route_id = command.route_id
            executor_id = command.participant.executor_id
        else:
            route_id = command.route_id
            executor_id = command.executor_id
        return {
            "X-Scheduler-Executor-ID": executor_id,
            "X-Scheduler-Operator-Plan-SHA256": (
                command.operator_plan_sha256
            ),
            "X-Scheduler-Route-ID": route_id,
            "X-Scheduler-Ticket-ID": command.ticket_id,
        }

    @staticmethod
    def _failure(
        error: HttpEndpointFailure,
        started_ns: int,
        finished_ns: int,
        epoch_ns: int,
        *,
        actual_execution: bool,
    ) -> PhysicalBackendFailure:
        return PhysicalBackendFailure(
            str(error),
            phase=error.phase,
            retry_safe=error.retry_safe,
            execution_started=(
                actual_execution and error.phase != "connect"
            ),
            started_us=max(0, (started_ns - epoch_ns) // 1000),
            finished_us=max(0, (finished_ns - epoch_ns) // 1000),
        )

    def rollback_transition(
        self, command: PhysicalTransitionCommand
    ) -> Mapping[str, object]:
        """Physically undo one completed transition after a failed commit."""

        if not isinstance(command, PhysicalTransitionCommand):
            raise PhysicalAdapterError("transition command is invalid")
        if self._rollback_transition is None:
            raise PhysicalAdapterError(
                "transition rollback executor is absent"
            )
        return self._rollback_transition(command)

    def apply_transition(
        self,
        command: PhysicalTransitionCommand,
        payload: object,
        control_check: Callable[[], None],
    ) -> RawTransitionObservation:
        if not isinstance(command, PhysicalTransitionCommand):
            raise PhysicalAdapterError("transition command is invalid")
        if not isinstance(payload, LlamaCppCompletionPayload):
            raise PhysicalAdapterError("transition payload is invalid")
        started_ns = self._begin_measurement(command.ticket_id)
        transition_path = payload.stream_path.with_name(
            payload.stream_path.stem
            + "-transition-"
            + command.transition.transition_id.replace("/", "_")
            + payload.stream_path.suffix
        )
        try:
            prepared = False
            if self._prepare_transition is not None:
                prepared = self._prepare_transition(
                    command, payload, control_check
                ) is True
            if command.transition.evictions and not prepared:
                raise PhysicalAdapterError(
                    "eviction transition executor is absent"
                )
            if not prepared:
                self._client.complete(
                    command.participant.endpoint,
                    replace(
                        payload,
                        stream_path=transition_path,
                        on_first_token=lambda _: None,
                    ),
                    control_check,
                    scheduler_headers=self._scheduler_headers(command),
                )
        except PhysicalBackendFailure:
            self._discard_measurement(command.ticket_id)
            raise
        except HttpEndpointFailure as error:
            finished_ns = time.monotonic_ns()
            self._discard_measurement(command.ticket_id)
            raise self._failure(
                error,
                started_ns,
                finished_ns,
                self._epoch_ns,
                actual_execution=False,
            ) from error
        except BaseException as error:
            finished_ns = time.monotonic_ns()
            self._discard_measurement(command.ticket_id)
            raise PhysicalBackendFailure(
                str(error),
                phase="transition_prepare",
                retry_safe=True,
                execution_started=False,
                started_us=self._relative_us(started_ns),
                finished_us=self._relative_us(finished_ns),
            ) from error
        finished_ns = time.monotonic_ns()
        measurement_started_ns = self._finish_measurement(
            command.ticket_id, started_ns
        )
        try:
            energy = self._measure_receipt(
                command, measurement_started_ns, finished_ns
            )
        except BaseException as error:
            raise PhysicalBackendFailure(
                str(error),
                phase="transition_energy_measurement",
                retry_safe=True,
                execution_started=False,
                started_us=self._relative_us(started_ns),
                finished_us=self._relative_us(finished_ns),
            ) from error
        return RawTransitionObservation(
            started_us=self._relative_us(started_ns),
            finished_us=self._relative_us(finished_ns),
            status="COMPLETED",
            evicted_artifact_sha256s=tuple(sorted({
                row.artifact_sha256
                for row in command.transition.evictions
            })),
            energy=energy,
        )

    def _http_execution_payload(
        self,
        command: PhysicalExecutionCommand,
        payload: LlamaCppCompletionPayload,
        started_ns: int,
        adaptive_cohort: bool,
        static_cohort: bool,
    ) -> tuple[
        LlamaCppCompletionPayload,
        list[dict[str, object]],
        list[dict[str, object]],
    ]:
        adaptive_fences: list[dict[str, object]] = []
        static_ack: list[dict[str, object]] = []
        if adaptive_cohort:
            request_payload = self._adaptive_cohort_payload(
                command,
                payload,
                started_ns,
                fence_sink=adaptive_fences.append,
            )
        elif static_cohort:
            request_payload = self._static_decode_cohort_payload(
                command, payload, started_ns
            )
        else:
            request_payload = self._adaptive_payload(
                command,
                payload,
                started_ns,
                fence_sink=adaptive_fences.append,
            )
        if not static_cohort:
            def record_static_control_ack(value: Mapping[str, object]) -> None:
                if static_ack:
                    raise PhysicalAdapterError(
                        "static FFN runtime acknowledgement is duplicated"
                    )
                static_ack.append(dict(value))

            request_payload = self._static_decode_payload(
                command, request_payload, record_static_control_ack
            )
        return request_payload, static_ack, adaptive_fences

    def _complete_http_execution(
        self,
        command: PhysicalExecutionCommand,
        payload: LlamaCppCompletionPayload,
        control_check: Callable[[], None],
        capacity_release: Callable[[int], None] | None,
        static_cohort: bool,
        static_ack: list[dict[str, object]],
        adaptive_fences: list[dict[str, object]],
    ) -> tuple[dict[str, object], int, int]:
        if self._before_prompt is not None:
            # the server must have its released FFN share resident before it processes a prompt
            self._before_prompt(command.endpoint, command.request_id)
        value = self._client.complete(
            command.endpoint,
            payload,
            control_check,
            scheduler_headers=self._scheduler_headers(command),
        )
        inference_finished_ns = time.monotonic_ns()
        inference_finished_us = self._relative_us(inference_finished_ns)
        if capacity_release is not None:
            capacity_release(inference_finished_us)
        if static_cohort:
            static_ack.append(dict(self._cohort_policy.control_ack(command)))
        result = dict(value)
        if static_ack:
            result["static_ffn_control_ack"] = static_ack[0]
        if adaptive_fences:
            result["request_slot_final_d2h_fences"] = [
                dict(row) for row in adaptive_fences
            ]
        return result, inference_finished_ns, inference_finished_us

    def _adaptive_execution_preview(
        self,
        command: PhysicalExecutionCommand,
        value: dict[str, object],
        cohort_policy_registered: bool,
    ) -> tuple[dict[str, object], PhysicalExecutionCommand, object | None, str | None]:
        owner_request_id = command.request_id
        if cohort_policy_registered:
            owner_request_id = self._cohort_policy.complete_http(command)
        preview = None
        proof_command = command
        if not self._adaptive_enabled(command) or owner_request_id is None:
            return value, proof_command, preview, owner_request_id
        preview = self._adaptive_scheduler.preview_adaptive_decode_completion(
            owner_request_id
        )
        value["adaptive_decode_observation"] = preview.to_json()
        helper_required = any(
            not row.policy.baseline
            for row in getattr(preview, "windows", ())
        ) or (
            getattr(preview, "unmeasured_tail_tokens", 0) > 0
            and not getattr(preview, "final_policy", None).baseline
        )
        if command.helper_envelope is None and helper_required:
            attached_helper = getattr(
                self._adaptive_scheduler,
                "runtime_attached_helper_envelope",
                None,
            )
            if not callable(attached_helper):
                raise PhysicalAdapterError(
                    "adaptive scheduler lacks helper proof"
                )
            helper = attached_helper(
                owner_request_id, expected_ticket_id=command.ticket_id
            )
            if helper is None:
                raise PhysicalAdapterError(
                    "adaptive phone work lacks its attached helper"
                )
            proof_command = bind_ready_helper_to_physical_command(
                command, helper
            )
        if helper_required:
            history_reader = getattr(
                self._adaptive_scheduler,
                "runtime_attached_helper_envelopes",
                None,
            )
            helper_history = (
                (proof_command.helper_envelope,)
                if not callable(history_reader)
                else history_reader(
                    owner_request_id,
                    expected_ticket_id=command.ticket_id,
                )
            )
            if not helper_history or any(row is None for row in helper_history):
                raise PhysicalAdapterError(
                    "adaptive phone work lacks helper history"
                )
            value["_runtime_helper_envelopes"] = tuple(helper_history)
        return value, proof_command, preview, owner_request_id

    def _collect_http_execution_proof(
        self,
        command: PhysicalExecutionCommand,
        proof_command: PhysicalExecutionCommand,
        value: dict[str, object],
        adaptive_preview: object | None,
        adaptive_owner_request_id: str | None,
        started_ns: int,
        inference_finished_ns: int,
        inference_finished_us: int,
        adaptive_fences: Sequence[Mapping[str, object]],
    ) -> _HttpExecutionResult:
        proof_drain_started_ns = time.monotonic_ns()
        if self._on_success is not None:
            proof = self._on_success(proof_command, value)
            if proof is not None:
                proof = dict(proof)
                if not proof:
                    raise PhysicalAdapterError(
                        "physical execution proof is empty"
                    )
                value["physical_execution_proof"] = proof
        value.pop("_runtime_helper_envelopes", None)
        proof_drain_finished_ns = time.monotonic_ns()
        if adaptive_preview is not None:
            completed = self._adaptive_scheduler.complete_adaptive_decode(
                adaptive_owner_request_id
            )
            if (
                completed.grouped_observation_sha256
                != adaptive_preview.grouped_observation_sha256
            ):
                raise PhysicalAdapterError(
                    "adaptive physical proof and terminal group differ"
                )
        value["physical_execution_timing"] = {
            "inference_finished_us": inference_finished_us,
            "proof_drain_duration_us": max(
                0,
                (proof_drain_finished_ns - proof_drain_started_ns) // 1000,
            ),
            "proof_drain_finished_us": self._relative_us(
                proof_drain_finished_ns
            ),
            "proof_drain_started_us": self._relative_us(
                proof_drain_started_ns
            ),
            "reported_inference_duration_us": max(
                0, (inference_finished_ns - started_ns) // 1000
            ),
            "request_slot_final_d2h_fence_count": len(adaptive_fences),
            "schema": "scheduler-http-execution-timing-v1",
        }
        return _HttpExecutionResult(
            value=value,
            inference_finished_ns=inference_finished_ns,
            proof_drain_started_ns=proof_drain_started_ns,
            proof_drain_finished_ns=proof_drain_finished_ns,
        )

    def _run_http_execution(
        self,
        command: PhysicalExecutionCommand,
        payload: LlamaCppCompletionPayload,
        control_check: Callable[[], None],
        capacity_release: Callable[[int], None] | None,
        started_ns: int,
        adaptive_cohort: bool,
        static_cohort: bool,
        cohort_policy_registered: bool,
    ) -> _HttpExecutionResult:
        request_payload, static_ack, fences = self._http_execution_payload(
            command, payload, started_ns, adaptive_cohort, static_cohort
        )
        value, inference_ns, inference_us = self._complete_http_execution(
            command,
            request_payload,
            control_check,
            capacity_release,
            static_cohort,
            static_ack,
            fences,
        )
        value, proof_command, preview, owner = self._adaptive_execution_preview(
            command, value, cohort_policy_registered
        )
        return self._collect_http_execution_proof(
            command,
            proof_command,
            value,
            preview,
            owner,
            started_ns,
            inference_ns,
            inference_us,
            fences,
        )

    def _classified_execution_failure(
        self,
        command: PhysicalExecutionCommand,
        error: BaseException,
        *,
        execution_started: bool,
        started_ns: int,
    ) -> PhysicalBackendFailure | None:
        """Structured helper_lost / server_exited failure, or None.

        Runs before the rig's finish callback drops the execution marker, so
        the classifier can read the server stderr since that marker. A
        classifier error keeps today's failure (no recovery is claimed).
        """
        if (
            self._failure_classifier is None
            or not isinstance(error, Exception)
            or isinstance(error, PhysicalBackendFailure)
        ):
            return None
        try:
            classification = self._failure_classifier(
                command, error, started_ns=started_ns
            )
            if classification is None:
                return None
            if not isinstance(
                classification, PhysicalFailureClassification
            ):
                raise PhysicalAdapterError(
                    "execution failure classification is invalid"
                )
            helper_lost = bool(classification.failed_device_ids)
            finished_ns = max(started_ns, time.monotonic_ns())
            return PhysicalBackendFailure(
                classification.phase
                + ": "
                + (classification.evidence or str(error)),
                phase=classification.phase,
                retry_safe=True,
                # A lost helper always interrupted a started request; an
                # exited server interrupted it only after the start.
                execution_started=helper_lost or execution_started,
                started_us=self._relative_us(started_ns),
                finished_us=self._relative_us(finished_ns),
                failed_device_ids=classification.failed_device_ids,
                executor_id=classification.executor_id,
                returncode=classification.returncode,
                stale_device_ids=classification.stale_device_ids,
                masked_executor_id=classification.masked_executor_id,
            )
        except Exception as classification_error:
            error.add_note(
                "execution failure classification failed: "
                + type(classification_error).__name__
                + ": "
                + str(classification_error)
            )
            return None

    def _fail_http_execution(
        self,
        command: PhysicalExecutionCommand,
        error: BaseException,
        cohort_execution: bool,
        cohort_policy_registered: bool,
    ) -> None:
        if cohort_execution:
            self._cohort_execution.fail(command, error)
            self._cohort_execution.abort(command)
            if cohort_policy_registered:
                self._cohort_policy.fail(command, error)
        else:
            self._discard_measurement(command.ticket_id)

    def _finish_http_execution_callbacks(
        self,
        command: PhysicalExecutionCommand,
        finish_required: bool,
        cohort_execution: bool,
        cohort_policy_registered: bool,
    ) -> None:
        if self._on_finish is not None and finish_required:
            try:
                self._on_finish(command)
            except BaseException:
                if not cohort_execution:
                    self._discard_measurement(command.ticket_id)
                else:
                    cleanup_error = PhysicalAdapterError(
                        "decode cohort execution cleanup failed"
                    )
                    self._cohort_execution.fail(command, cleanup_error)
                    self._cohort_execution.abort(command)
                    if cohort_policy_registered:
                        self._cohort_policy.fail(command, cleanup_error)
                raise
        if cohort_policy_registered:
            self._cohort_policy.release(command)

    def _cohort_execution_energy(
        self,
        command: PhysicalExecutionCommand,
        finished_ns: int,
    ) -> object:
        try:
            result = self._cohort_execution.finish(command, finished_ns)
        except BaseException:
            self._cohort_execution.abort(command)
            raise
        if result is None:
            raise PhysicalAdapterError(
                "physical decode cohort result is absent"
            )
        energy = result.energy
        if energy is None:
            return None
        cohort = command.decode_cohort
        record_cohort = getattr(
            self._adaptive_scheduler, "record_decode_cohort_measurement", None
        )
        if not callable(record_cohort):
            raise PhysicalAdapterError(
                "decode cohort execution lacks a receipt sink"
            )
        record_cohort(
            str(cohort["cohort_id"]),
            started_at_us=self._relative_us(result.started_ns),
            finished_at_us=self._relative_us(result.finished_ns),
            fleet_energy_uj_by_domain=energy.fleet_energy_uj_by_domain,
            transfer_energy_uj_by_link=energy.transfer_energy_uj_by_link,
            measurement_evidence_ids=energy.measurement_evidence_ids,
            attribution_kind=energy.attribution_kind,
            energy_boundary_id=energy.energy_boundary_id,
            total_input_tokens=result.total_input_tokens,
            total_output_tokens=result.total_output_tokens,
            energy_estimation_metadata=energy.estimation_metadata,
        )
        return energy

    def _http_execution_energy(
        self,
        command: PhysicalExecutionCommand,
        started_ns: int,
        finished_ns: int,
        cohort_execution: bool,
        value: dict[str, object],
    ) -> object:
        if cohort_execution:
            return self._cohort_execution_energy(command, finished_ns)
        measurement_started_ns = self._finish_measurement(
            command.ticket_id, started_ns
        )
        try:
            return self._measure_receipt(
                command, measurement_started_ns, finished_ns
            )
        except PhysicalAdapterError as error:
            value["energy_measurement_error"] = (
                type(error).__name__ + ":" + str(error)
            )
            return None

    def _execute(
        self,
        command: PhysicalExecutionCommand,
        payload: object,
        control_check: Callable[[], None],
        capacity_release: Callable[[int], None] | None,
    ) -> RawExecutionObservation:
        if not isinstance(command, PhysicalExecutionCommand):
            raise PhysicalAdapterError("execution command is invalid")
        validate_physical_execution_command(command)
        if not isinstance(payload, LlamaCppCompletionPayload):
            raise PhysicalAdapterError("execution payload is invalid")
        payload = self._runtime_progress_payload(command, payload)
        payload = self._speculative_payload(command, payload)
        cohort_execution = command.decode_cohort is not None
        adaptive_cohort = cohort_execution and self._adaptive_enabled(command)
        static_cohort = (
            cohort_execution
            and not adaptive_cohort
            and static_decode_policy(command) is not None
        )
        cohort_policy_registered = False
        if cohort_execution:
            self._cohort_execution.begin(
                command,
                input_tokens=payload.input_tokens,
                output_tokens=payload.output_tokens,
                timeout_s=self._cohort_service_timeout_s(command),
            )
            started_ns = time.monotonic_ns()
        else:
            started_ns = self._begin_measurement(command.ticket_id)
        finish_required = self._on_start is None
        try:
            if self._on_start is not None:
                self._on_start(command)
                finish_required = True
            cohort_policy_registered = adaptive_cohort or static_cohort
            result = self._run_http_execution(
                command,
                payload,
                control_check,
                capacity_release,
                started_ns,
                adaptive_cohort,
                static_cohort,
                cohort_policy_registered,
            )
        except HttpEndpointFailure as error:
            finished_ns = time.monotonic_ns()
            classified = self._classified_execution_failure(
                command,
                error,
                execution_started=(
                    finish_required and error.phase != "connect"
                ),
                started_ns=started_ns,
            )
            self._fail_http_execution(
                command, error, cohort_execution, cohort_policy_registered
            )
            if classified is not None:
                raise classified from error
            raise self._failure(
                error,
                started_ns,
                finished_ns,
                self._epoch_ns,
                actual_execution=True,
            ) from error
        except BaseException as error:
            classified = self._classified_execution_failure(
                command,
                error,
                execution_started=finish_required,
                started_ns=started_ns,
            )
            self._fail_http_execution(
                command, error, cohort_execution, cohort_policy_registered
            )
            if classified is not None:
                raise classified from error
            raise
        finally:
            self._release_speculative_rows(command, payload)
            self._finish_http_execution_callbacks(
                command,
                finish_required,
                cohort_execution,
                cohort_policy_registered,
            )
        finished_ns = result.inference_finished_ns
        finished_us = self._relative_us(finished_ns)
        value = dict(result.value)
        energy = self._http_execution_energy(
            command,
            started_ns,
            finished_ns,
            cohort_execution,
            value,
        )
        return RawExecutionObservation(
            started_us=self._relative_us(started_ns),
            finished_us=finished_us,
            output_sha256=value["stream_sha256"],
            payload=value,
            energy=energy,
            energy_scope=(
                "warm_execution" if command.transitions else "route_total"
            ),
        )

    def execute(
        self,
        command: PhysicalExecutionCommand,
        payload: object,
        control_check: Callable[[], None],
    ) -> RawExecutionObservation:
        return self._execute(
            command, payload, control_check, None
        )

    def execute_with_capacity_release(
        self,
        command: PhysicalExecutionCommand,
        payload: object,
        control_check: Callable[[], None],
        capacity_release: Callable[[int], None],
    ) -> RawExecutionObservation:
        if not callable(capacity_release):
            raise PhysicalAdapterError(
                "physical capacity release callback is invalid"
            )
        return self._execute(
            command, payload, control_check, capacity_release
        )
