"""llama-server adapter contracts: launch contracts, phone FFN contracts, call parsing and execution proof records."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
from statistics import median
from types import MappingProxyType
from typing import Mapping, Sequence
from .._internal.model_manifest import ModelManifest
from .._internal.runtime_capabilities import RuntimeExecutorCapability
from .._internal.types import canonical_sha256
from .contracts import PhysicalAdapterError, dormant_phone_ffn_parameters
from .phone_helpers import phone_helper_bindings_from_parameters, phone_helper_launch_environment
from .phone_transport import phone_transport_contract
from .speculative_rows import LlamaServerSpeculativeContract, speculative_launch_contract
from .ticket import PhysicalExecutionCommand, PhysicalTransitionCommand


def llama_server_runtime_timing(lines: Sequence[str]) -> dict[str, object]:
    """Absolute native timings; host gaps exclude the preceding USB RPC."""
    def summary(values):
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "median": median(ordered) if ordered else None,
            "p95": ordered[min(len(ordered) - 1, len(ordered) * 95 // 100)]
                if ordered else None,
            "maximum": ordered[-1] if ordered else None,
        }

    calls = []
    token_intervals = []
    previous_token = {}
    captures = []
    for index, line in enumerate(lines):
        if "CUDA graph warmup complete" in line:
            captures.append(index)
        if line.startswith("S41SERVERFFNUSB "):
            fields = dict(item.split("=", 1) for item in line.split()[1:])
            calls.append({key: int(value) for key, value in fields.items()})
        token = re.search(
            r"^(\d+)\.(\d+)\.(\d+)\.(\d+) .*id\s+(\d+) \| task (\d+)"
            r" \| n_decoded = (\d+),", line,
        )
        if token is None:
            continue
        minute, second, millis, micros, slot, task, decoded = map(int, token.groups())
        at_us = ((minute * 60 + second) * 1000 + millis) * 1000 + micros
        previous = previous_token.get((slot, task))
        if previous is not None and decoded == previous[0] + 1:
            token_intervals.append({"slot": slot, "task": task, "token": decoded,
                                    "latency_us": at_us - previous[1]})
        previous_token[(slot, task)] = (decoded, at_us)
    gaps = {}
    for left, right in zip(calls, calls[1:]):
        if right["request"] != left["request"] + 1 or (
            left["tokens"], left["columns"]
        ) != (right["tokens"], right["columns"]):
            continue
        key = json.dumps({"from_layer": left["layer"], "to_layer": right["layer"],
                          "tokens": right["tokens"], "columns": right["columns"]}, sort_keys=True)
        gaps.setdefault(key, []).append(right["started_ns"] - left["d2h_completed_ns"])
    return {
        "schema": "s42-server-absolute-runtime-timing-v1",
        "cuda_capture_log_count": len(captures),
        "cuda_capture_log_line_indices": captures,
        "capture_note": "Warmup-complete markers include initial captures and recaptures; not capture duration.",
        "rpc_latency_ns": summary([row["d2h_completed_ns"] - row["started_ns"] for row in calls]),
        "host_submission_gap_ns_by_call_class": {
            key: summary(values) for key, values in sorted(gaps.items())
        },
        "decode_token_latency_us": summary([row["latency_us"] for row in token_intervals]),
        "decode_token_intervals": token_intervals,
    }


@dataclass(frozen=True)
class _PhoneFfnCommandView:
    artifact_sha256: str
    route_id: str
    operator_plan_sha256: str
    operator_plan: Mapping[str, object]
    adapter_parameters: Mapping[str, int | str]
    execution_contract: object


def _phone_ffn_command(
    command: PhysicalExecutionCommand | PhysicalTransitionCommand,
) -> PhysicalExecutionCommand | PhysicalTransitionCommand | _PhoneFfnCommandView:
    helper = getattr(command, "helper_envelope", None)
    if helper is None:
        return command
    return _PhoneFfnCommandView(
        artifact_sha256=helper.artifact_sha256,
        route_id=helper.route_id,
        operator_plan_sha256=helper.operator_plan_sha256,
        operator_plan=MappingProxyType(helper.helper_plan.to_json()),
        adapter_parameters=helper.helper_plan.adapter_parameters,
        execution_contract=helper.helper_plan.execution_contract,
    )


def _adaptive_phone_ffn(
    command: PhysicalExecutionCommand | PhysicalTransitionCommand,
) -> bool:
    source = _phone_ffn_command(command)
    if (
        source.execution_contract.execution_mode == "adaptive-split"
        and source.operator_plan.get("assisted_operator_kind") == "ffn"
        and type(source.adapter_parameters.get("phone_device_id")) is str
    ):
        return True
    dormant = dormant_phone_ffn_parameters(command.adapter_parameters)
    return bool(
        source is command
        and source.execution_contract.execution_mode == "desktop"
        and dormant is not None
        and dormant.get("ffn_assistance_phase") == "decode"
        and dormant.get("ffn_runtime_control_protocol")
            == "decode-boundary-v1"
        and type(dormant.get("phone_device_id")) is str
    )


@dataclass(frozen=True)
class LlamaServerLaunchContract:
    model_alias: str
    context_size: int
    parallel: int
    batch_size: int
    ubatch_size: int
    gpu_layers: int
    cpu_device_id: str
    gpu_device_id: str
    phone_device_id: str | None
    ffn_environment: Mapping[str, str]
    threads: int = 0
    threads_batch: int = 0
    cpu_affinity: str | None = None
    cuda_graph_mode: str = "default"
    desktop_launch_mode: str = "canonical"
    kv_cpu_layers: tuple[int, ...] = ()
    kv_device_cells: tuple[tuple[int, int], ...] = ()
    ffn_host_share_drop_cache: int = 1
    ffn_host_share_populate: int = 1
    scheduler_trace_path: str | None = None
    logits_trace_path: str | None = None
    ffn_row_diagnostic_steps: int = 0
    # campaign ``speculative_rows``: the draft this server loads; None launches byte-identically
    speculative: LlamaServerSpeculativeContract | None = None

    def __post_init__(self) -> None:
        if self.speculative is not None and not isinstance(
            self.speculative, LlamaServerSpeculativeContract
        ):
            raise PhysicalAdapterError("llama-server speculative contract is invalid")
        for name in (
            "model_alias",
            "cpu_device_id",
            "gpu_device_id",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value or not value.isascii():
                raise PhysicalAdapterError(
                    "llama-server launch identity is invalid"
                )
        if self.phone_device_id is not None and (
            type(self.phone_device_id) is not str
            or not self.phone_device_id
            or not self.phone_device_id.isascii()
        ):
            raise PhysicalAdapterError(
                "llama-server phone identity is invalid"
            )
        for name in (
            "context_size",
            "parallel",
            "batch_size",
            "ubatch_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise PhysicalAdapterError(
                    "llama-server launch shape is invalid"
                )
        if type(self.gpu_layers) is not int or self.gpu_layers < 0:
            raise PhysicalAdapterError(
                "llama-server GPU layer count is invalid"
            )
        if (
            type(self.threads) is not int
            or self.threads < 0
            or type(self.threads_batch) is not int
            or self.threads_batch < 0
            or bool(self.threads) != bool(self.threads_batch)
            or (
                self.cpu_affinity is not None
                and (
                    type(self.cpu_affinity) is not str
                    or not self.cpu_affinity
                    or not self.cpu_affinity.isascii()
                )
            )
        ):
            raise PhysicalAdapterError(
                "llama-server CPU launch contract is invalid"
            )
        if self.cuda_graph_mode not in ("default", "disabled"):
            raise PhysicalAdapterError("llama-server CUDA graph mode is invalid")
        if self.scheduler_trace_path is not None and (
            type(self.scheduler_trace_path) is not str
            or not self.scheduler_trace_path.startswith("/")
            or any(c in self.scheduler_trace_path for c in ("\0", "\n", "\r"))
        ):
            raise PhysicalAdapterError("llama-server scheduler trace path must be an absolute path")
        if self.logits_trace_path is not None and (
            type(self.logits_trace_path) is not str
            or not self.logits_trace_path.startswith("/")
            or any(c in self.logits_trace_path for c in ("\0", "\n", "\r"))
            or self.desktop_launch_mode != "canonical"
        ):
            raise PhysicalAdapterError("llama-server logits trace requires a canonical launch and absolute path")
        if self.desktop_launch_mode not in ("canonical", "runtime-defaults"):
            raise PhysicalAdapterError("llama-server desktop launch mode is invalid")
        layers = tuple(self.kv_cpu_layers)
        if any(type(layer) is not int or layer < 0 for layer in layers) or len(set(layers)) != len(layers):
            raise PhysicalAdapterError("llama-server CPU KV layers are invalid")
        object.__setattr__(self, "kv_cpu_layers", tuple(sorted(layers)))
        prefixes = tuple(self.kv_device_cells)
        if any(not isinstance(row, tuple) or len(row) != 2 or type(row[0]) is not int or row[0] < 0
               or type(row[1]) is not int or row[1] < 0 or row[1] > self.context_size or row[1] % 256
               or (row[0] in layers and row[1] != 0) for row in prefixes) or len({row[0] for row in prefixes}) != len(prefixes):
            raise PhysicalAdapterError("llama-server split KV prefixes are invalid")
        if any(cells > 0 for _, cells in prefixes) and (self.gpu_layers == 0 or self.desktop_launch_mode != "canonical"):
            raise PhysicalAdapterError("split KV requires a canonical GPU parent")
        object.__setattr__(self, "kv_device_cells", tuple(sorted(prefixes)))
        environment = dict(self.ffn_environment)
        diagnostic_key = "S41_SERVER_FFN_ROW_DIAGNOSTIC_STEPS"
        if type(self.ffn_row_diagnostic_steps) is not int or self.ffn_row_diagnostic_steps not in (0, 5, 64):
            raise PhysicalAdapterError("FFN row diagnostic steps must be zero, five or 64")
        if diagnostic_key in environment and environment[diagnostic_key] != str(self.ffn_row_diagnostic_steps):
            raise PhysicalAdapterError("FFN row diagnostic conflicts with its typed contract")
        if self.ffn_row_diagnostic_steps:
            if (environment.get("S41_SERVER_FFN_DORMANT_HOST_SHARE") != "1"
                    or environment.get("S41_SERVER_FFN_RUNTIME_CONTROL") != "1"
                    or environment.get("S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK", "0") != "0"):
                raise PhysicalAdapterError("FFN row diagnostic requires dormant assisted-copy runtime control")
            environment[diagnostic_key] = str(self.ffn_row_diagnostic_steps)
        for field_name, suffix in (("ffn_host_share_drop_cache", "DROP_CACHE"), ("ffn_host_share_populate", "POPULATE")):
            value = getattr(self, field_name)
            key = "S41_SERVER_FFN_DORMANT_" + suffix
            if type(value) is not int or value not in (0, 1):
                raise PhysicalAdapterError("llama-server dormant host share policy flag is invalid")
            if (value != 1 or key in environment) and environment.get("S41_SERVER_FFN_DORMANT_HOST_SHARE") != "1":
                raise PhysicalAdapterError("llama-server dormant host share policy requires release")
            if key in environment and environment[key] != str(value):
                raise PhysicalAdapterError("llama-server dormant host share policy conflicts with its environment")
            if value != 1:
                environment[key] = str(value)
        if any(
            type(key) is not str
            or not key.startswith("S41_SERVER_FFN_")
            or not key.isascii()
            or type(value) is not str
            or not value
            or not value.isascii()
            for key, value in environment.items()
        ):
            raise PhysicalAdapterError(
                "llama-server FFN environment is invalid"
            )
        if environment.get("S41_SERVER_FFN_USB_BATCH_PLAN") == "coalesced-batch":
            shape = tuple(environment.get("S41_SERVER_FFN_" + name, "") for name in (
                "MAX_TOKENS", "N_EMBD", "USB_MAX_PAYLOAD_BYTES"))
            if any(not value.isdecimal() for value in shape):
                raise PhysicalAdapterError("coalesced FFN shape is invalid")
            max_tokens, n_embd, max_payload = map(int, shape)
            if (environment.get("S41_SERVER_FFN_RUNTIME_CONTROL") != "1"
                    or environment.get("S41_SERVER_FFN_TRANSPORT") != "functionfs-usb"
                    or not self.parallel <= max_tokens <= min(self.ubatch_size, 512)
                    or self.parallel > 8 or n_embd <= 0 or max_payload < max_tokens * n_embd * 2):
                raise PhysicalAdapterError("coalesced FFN transport does not cover the decode slots")
        object.__setattr__(
            self,
            "ffn_environment",
            MappingProxyType(dict(sorted(environment.items()))),
        )


def _kv_device_cells(parameters, manifest):
    value = parameters.get("kv_device_cells", "")
    if type(value) is not str:
        raise PhysicalAdapterError("split KV prefixes must be a string")
    if not value:
        return ()
    parts = [part.split(":") for part in value.split(",")]
    if any(len(part) != 2 or any(not item.isascii() or not item.isdecimal() for item in part) for part in parts):
        raise PhysicalAdapterError("invalid split KV prefix syntax")
    result = tuple((int(il), int(cells)) for il, cells in parts)
    if len({il for il, _ in result}) != len(result) or any(il >= manifest.block_count for il, _ in result):
        raise PhysicalAdapterError("split KV layers exceed the model or contain duplicates")
    return tuple(sorted(result))


def _kv_cpu_layers(parameters, manifest):
    value = parameters.get("kv_cpu_layers", "")
    if type(value) is not str:
        raise PhysicalAdapterError("llama-server CPU KV layers must be a comma-separated string")
    if not value:
        return ()
    parts = value.split(",")
    if any(not part.isascii() or not part.isdecimal() for part in parts):
        raise PhysicalAdapterError("llama-server CPU KV layers are invalid")
    layers = tuple(int(part) for part in parts)
    if len(set(layers)) != len(layers) or any(layer >= manifest.block_count for layer in layers):
        raise PhysicalAdapterError("llama-server CPU KV layers exceed the model or contain duplicates")
    return tuple(sorted(layers))


def _launch_contract_supports_execution(
    launched: LlamaServerLaunchContract,
    requested: LlamaServerLaunchContract,
) -> bool:
    if launched == requested:
        return True
    environment = launched.ffn_environment
    requested_environment = requested.ffn_environment
    if (
        launched.phone_device_id == requested.phone_device_id
        and launched.phone_device_id is not None
        and environment.get("S41_SERVER_FFN_RUNTIME_CONTROL") == "1"
        and requested_environment.get(
            "S41_SERVER_FFN_RUNTIME_CONTROL"
        ) == "1"
        and "S41_SERVER_FFN_POLICY" not in environment
        and "S41_SERVER_FFN_POLICY" not in requested_environment
        and "S41_SERVER_FFN_SHARDS" not in environment
        and "S41_SERVER_FFN_SHARDS" not in requested_environment
        and replace(launched, ffn_environment={}, ffn_host_share_drop_cache=1, ffn_host_share_populate=1,
                    speculative=None)
            == replace(requested, ffn_environment={}, ffn_host_share_drop_cache=1, ffn_host_share_populate=1,
                       speculative=None)
    ):
        launched_values = dict(environment)
        requested_values = dict(requested_environment)
        launched_mask_text = launched_values.pop(
            "S41_SERVER_FFN_LAYER_MASK", None
        )
        requested_mask_text = requested_values.pop(
            "S41_SERVER_FFN_LAYER_MASK", None
        )
        try:
            launched_mask = int(launched_mask_text or "")
            requested_mask = int(requested_mask_text or "")
        except ValueError:
            return False
        return (
            launched_mask > 0
            and requested_mask > 0
            and str(launched_mask) == launched_mask_text
            and str(requested_mask) == requested_mask_text
            and requested_mask & ~launched_mask == 0
            and launched_values == requested_values
        )
    if (
        requested.phone_device_id is not None
        or requested.ffn_environment
        or launched.phone_device_id is None
        or environment.get("S41_SERVER_FFN_RUNTIME_CONTROL") != "1"
        or "S41_SERVER_FFN_POLICY" in environment
        or "S41_SERVER_FFN_SHARDS" in environment
    ):
        return False
    return replace(
        launched,
        phone_device_id=None,
        ffn_environment={},
        ffn_host_share_drop_cache=1,
        ffn_host_share_populate=1,
        speculative=None,
    ) == replace(requested, speculative=None)


def _speculative_phone_max_tokens(environment: Mapping[str, str]) -> int | None:
    """The phone contract rows of a launch environment, None when it declares none."""
    text = environment.get("S41_SERVER_FFN_MAX_TOKENS")
    return int(text) if type(text) is str and text.isdecimal() else None


@dataclass(frozen=True)
class PhoneFfnExecutionContract:
    """Exact phone FFN shape carried by a scheduler execution plan."""

    device_id: str
    n_embd: int
    layer_indices: tuple[int, ...]
    layer_mask: int
    columns: int
    max_tokens: int
    activation: str

    @property
    def layers(self) -> str:
        spans: list[str] = []
        start = self.layer_indices[0]
        end = start
        for index in self.layer_indices[1:]:
            if index == end + 1:
                end = index
                continue
            spans.append(str(start) if start == end else f"{start}-{end}")
            start = end = index
        spans.append(str(start) if start == end else f"{start}-{end}")
        return ",".join(spans)


@dataclass(frozen=True)
class LlamaServerFfnCallContext:
    scheduler_request_id: str
    server_slot_id: int
    rows: int
    plan_generation: int


@dataclass(frozen=True)
class LlamaServerFfnCall:
    request_id: int
    layer: int
    tokens: int
    columns: int
    payload_bytes: int
    contexts: tuple[LlamaServerFfnCallContext, ...] = ()

    def rows_for(self, scheduler_request_id: str) -> int:
        return sum(
            row.rows for row in self.contexts
            if row.scheduler_request_id == scheduler_request_id
        )


@dataclass(frozen=True)
class LlamaServerPhoneSessionProof:
    session_id: str
    endpoint: str
    artifact_sha256: str
    resident_geometry_sha256: str
    operator_plan_sha256: str
    session_generation: int
    layer_mask: int
    calls: int
    rows: int
    payload_bytes: int

    def to_json(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "calls": self.calls,
            "endpoint": self.endpoint,
            "layer_mask": self.layer_mask,
            "operator_plan_sha256": self.operator_plan_sha256,
            "payload_bytes": self.payload_bytes,
            "resident_geometry_sha256": (
                self.resident_geometry_sha256
            ),
            "rows": self.rows,
            "session_generation": self.session_generation,
            "session_id": self.session_id,
        }


@dataclass(frozen=True)
class LlamaServerExecutionMarker:
    ticket_id: str
    artifact_sha256: str
    operator_plan_sha256: str
    executor_id: str
    stderr_index: int
    phone_contract: PhoneFfnExecutionContract | None
    remote_resident_identity_sha256: str | None = None


@dataclass(frozen=True)
class LlamaServerExecutionProof:
    ticket_id: str
    artifact_sha256: str
    operator_plan_sha256: str
    executor_id: str
    phone_call_count: int
    phone_calls_by_layer: tuple[tuple[int, int], ...]
    phone_calls_sha256: str
    phone_first_request_id: int | None
    phone_last_request_id: int | None
    phone_payload_bytes: int
    stderr_start_index: int
    stderr_end_index: int
    phone_calls_by_session: tuple[
        LlamaServerPhoneSessionProof, ...
    ] = ()
    adaptive_grouped_observation_sha256: str | None = None
    adaptive_group_owner_request_id: str | None = None
    adaptive_window_count: int = 0
    static_policy_applied_token_index: int | None = None
    static_policy_hash: str | None = None
    static_policy_plan_generation: int | None = None
    static_policy_slot_id: int | None = None

    def _json_without_hash(self) -> dict[str, object]:
        result = {
            "artifact_sha256": self.artifact_sha256,
            "executor_id": self.executor_id,
            "operator_plan_sha256": self.operator_plan_sha256,
            "phone_call_count": self.phone_call_count,
            "phone_calls_by_layer": [
                {"calls": calls, "layer": layer}
                for layer, calls in self.phone_calls_by_layer
            ],
            "phone_calls_sha256": self.phone_calls_sha256,
            "phone_first_request_id": self.phone_first_request_id,
            "phone_last_request_id": self.phone_last_request_id,
            "phone_payload_bytes": self.phone_payload_bytes,
            "stderr_end_index": self.stderr_end_index,
            "stderr_start_index": self.stderr_start_index,
            "ticket_id": self.ticket_id,
        }
        if self.phone_calls_by_session:
            result["phone_calls_by_session"] = [
                row.to_json() for row in self.phone_calls_by_session
            ]
        if self.adaptive_grouped_observation_sha256 is not None:
            result["adaptive_grouped_observation_sha256"] = (
                self.adaptive_grouped_observation_sha256
            )
            result["adaptive_group_owner_request_id"] = (
                self.adaptive_group_owner_request_id
            )
            result["adaptive_window_count"] = self.adaptive_window_count
        if self.static_policy_applied_token_index is not None:
            result["static_policy_applied_token_index"] = (
                self.static_policy_applied_token_index
            )
            result["static_policy_hash"] = self.static_policy_hash
            result["static_policy_plan_generation"] = (
                self.static_policy_plan_generation
            )
            result["static_policy_slot_id"] = self.static_policy_slot_id
        return result

    def to_json(self) -> dict[str, object]:
        result = self._json_without_hash()
        result["proof_sha256"] = canonical_sha256(result)
        return result


_FFN_CALL_PATTERN = re.compile(
    r"^S41SERVERFFNCALL request=([0-9]+) layer=([0-9]+) "
    r"tokens=([0-9]+) columns=([0-9]+) payload_bytes=([0-9]+)$"
)


_SCOPED_FFN_CALL_PATTERN = re.compile(
    r"^S41SERVERFFNCALL context="
    r"([0-9a-f]+:[0-9]+:[0-9]+:[0-9]+"
    r"(?:,[0-9a-f]+:[0-9]+:[0-9]+:[0-9]+)*) "
    r"request=([0-9]+) layer=([0-9]+) tokens=([0-9]+) "
    r"columns=([0-9]+) payload_bytes=([0-9]+)$"
)


_LLAMA_LOGGED_FFN_CALL_PREFIX = re.compile(
    r"^[0-9]+\.[0-9]{2}\.[0-9]{3}\.[0-9]{3} (?:[A-Z] )?"
    r"(?=S41SERVERFFNCALL )"
)


_TAIL_RELEASE_REASONS = frozenset({
    "server_release_guard",
    "stale_slot_control_discarded",
})


def parse_llama_server_ffn_call(line: str) -> LlamaServerFfnCall | None:
    if type(line) is not str:
        raise PhysicalAdapterError("llama-server FFN call line is invalid")
    prefix = _LLAMA_LOGGED_FFN_CALL_PREFIX.match(line)
    if prefix is not None:
        line = line[prefix.end():]
    scoped = _SCOPED_FFN_CALL_PATTERN.fullmatch(line)
    match = _FFN_CALL_PATTERN.fullmatch(line) if scoped is None else None
    if scoped is None and match is None:
        return None
    raw_values = match.groups() if scoped is None else scoped.groups()[1:]
    values = tuple(int(value) for value in raw_values)
    if (
        values[0] <= 0
        or values[1] < 0
        or any(value <= 0 for value in values[2:])
    ):
        raise PhysicalAdapterError("llama-server FFN call is invalid")
    contexts = ()
    if scoped is not None:
        parsed_contexts = []
        for encoded in scoped.group(1).split(","):
            request_hex, slot, rows, generation = encoded.split(":")
            try:
                request_id = bytes.fromhex(request_hex).decode("ascii")
            except (ValueError, UnicodeDecodeError) as exc:
                raise PhysicalAdapterError(
                    "llama-server FFN call context is invalid"
                ) from exc
            parsed_contexts.append(LlamaServerFfnCallContext(
                scheduler_request_id=request_id,
                server_slot_id=int(slot),
                rows=int(rows),
                plan_generation=int(generation),
            ))
        contexts = tuple(parsed_contexts)
        if (
            not contexts
            or len({row.scheduler_request_id for row in contexts})
                != len(contexts)
            or len({row.server_slot_id for row in contexts})
                != len(contexts)
            or any(
                not row.scheduler_request_id
                or row.server_slot_id < 0
                or row.rows <= 0
                or row.plan_generation < 0
                for row in contexts
            )
            or sum(row.rows for row in contexts) != values[2]
        ):
            raise PhysicalAdapterError(
                "llama-server FFN call context is invalid"
            )
    return LlamaServerFfnCall(*values, contexts=contexts)


def _integer(parameters: Mapping[str, int | str], name: str) -> int:
    value = parameters.get(name)
    if type(value) is not int or value < 0:
        raise PhysicalAdapterError(
            "llama-server adapter parameter is invalid: " + name
        )
    return value


def _text(parameters: Mapping[str, int | str], name: str) -> str:
    value = parameters.get(name)
    if type(value) is not str or not value or not value.isascii():
        raise PhysicalAdapterError(
            "llama-server adapter parameter is invalid: " + name
        )
    return value


def _layer_index(layer_id: object) -> int | None:
    if type(layer_id) is not str or not layer_id.startswith("layer:"):
        return None
    try:
        value = int(layer_id[6:].split(":", 1)[0])
    except ValueError as error:
        raise PhysicalAdapterError(
            "llama-server operator layer is invalid"
        ) from error
    if value < 0:
        raise PhysicalAdapterError(
            "llama-server operator layer is invalid"
        )
    return value


def _operator_rows(
    command: PhysicalExecutionCommand | PhysicalTransitionCommand
        | _PhoneFfnCommandView,
) -> tuple[dict[str, object], ...]:
    plan = command.operator_plan
    if not plan:
        return ()
    if (
        plan.get("plan_sha256") != command.operator_plan_sha256
        or plan.get("route_id") != command.route_id
    ):
        raise PhysicalAdapterError(
            "llama-server operator plan identity differs from the ticket"
        )
    operators = plan.get("operators")
    if type(operators) is not list or not operators or any(
        type(row) is not dict for row in operators
    ):
        raise PhysicalAdapterError("llama-server operator plan is invalid")
    return tuple(operators)


def _gpu_layers_from_plan(
    operators: tuple[dict[str, object], ...],
    manifest: ModelManifest,
    cpu_device_id: str,
    gpu_device_id: str,
    phone_device_id: str | None,
    co_helper_device_ids: tuple[str, ...] = (),
) -> int:
    if not operators:
        raise PhysicalAdapterError(
            "llama-server launch requires an executable operator plan"
        )
    devices_by_layer: dict[int, set[str]] = {}
    for row in operators:
        index = _layer_index(row.get("operator_id"))
        if index is None:
            continue
        raw_devices = row.get("device_ids")
        if type(raw_devices) is not list or any(
            type(value) is not str for value in raw_devices
        ):
            raise PhysicalAdapterError(
                "llama-server operator devices are invalid"
            )
        local = {
            value for value in raw_devices
            if value != phone_device_id and value not in co_helper_device_ids
        }
        if local:
            devices_by_layer.setdefault(index, set()).update(local)
    if set(devices_by_layer) != set(range(manifest.block_count)):
        raise PhysicalAdapterError(
            "llama-server plan does not place every model layer"
        )
    gpu_indices = []
    for index in range(manifest.block_count):
        devices = devices_by_layer[index]
        if devices == {gpu_device_id}:
            gpu_indices.append(index)
        elif devices != {cpu_device_id}:
            raise PhysicalAdapterError(
                "llama-server plan requires unsupported intra-layer placement"
            )
    if gpu_indices:
        first = gpu_indices[0]
        if gpu_indices != list(range(first, manifest.block_count)):
            raise PhysicalAdapterError(
                "llama-server GPU placement is not a contiguous suffix"
            )
    return len(gpu_indices)


def _phone_ffn_from_plan(
    command: PhysicalExecutionCommand | PhysicalTransitionCommand
        | _PhoneFfnCommandView,
    operators: tuple[dict[str, object], ...],
    manifest: ModelManifest,
    phone_device_id: str,
    max_tokens: int,
) -> tuple[int, int, str]:
    assisted = tuple(
        row for row in operators
        if row.get("operator_kind") == "ffn"
        and phone_device_id in row.get("device_ids", [])
    )
    indices = sorted({
        index for row in assisted
        for index in (_layer_index(row.get("operator_id")),)
        if index is not None
    })
    if not assisted or len(indices) != len(assisted) or max(indices) >= 64:
        raise PhysicalAdapterError(
            "llama-server phone FFN placement is invalid"
        )
    fractions = {row.get("split_fraction_ppm") for row in assisted}
    axes = {row.get("split_axis") for row in assisted}
    if len(fractions) != 1 or len(axes) != 1:
        raise PhysicalAdapterError(
            "llama-server phone FFN splits are inconsistent"
        )
    fraction = fractions.pop()
    axis = axes.pop()
    if axis == "none" and fraction == 0:
        columns = manifest.feed_forward_length
    elif axis == "column" and type(fraction) is int and 0 < fraction < 1_000_000:
        scaled = manifest.feed_forward_length * fraction
        if scaled % 1_000_000:
            raise PhysicalAdapterError(
                "llama-server phone FFN fraction is not column exact"
            )
        columns = scaled // 1_000_000
    else:
        raise PhysicalAdapterError(
            "llama-server phone FFN split is not physically supported"
        )
    if columns <= 0 or columns % 32:
        raise PhysicalAdapterError(
            "llama-server phone FFN columns are invalid"
        )
    layer_mask = sum(1 << index for index in indices)
    return layer_mask, columns, f"{max_tokens}:{columns}"


def phone_ffn_execution_contract(
    command: PhysicalExecutionCommand | PhysicalTransitionCommand,
    manifest: ModelManifest,
) -> PhoneFfnExecutionContract:
    """Validate and expose the phone FFN placement selected in a ticket."""
    if not isinstance(
        command, (PhysicalExecutionCommand, PhysicalTransitionCommand)
    ) or not isinstance(manifest, ModelManifest):
        raise PhysicalAdapterError("phone FFN execution input is invalid")
    if command.artifact_sha256 != manifest.artifact_sha256:
        raise PhysicalAdapterError(
            "phone FFN execution artifact differs from the ticket"
        )
    source = _phone_ffn_command(command)
    parameters = source.adapter_parameters
    device_id = _text(parameters, "phone_device_id")
    remote = source.execution_contract.remote_resident_ffn
    if remote is not None:
        _remote_resident_launch_environment(command, manifest, parameters, remote)
        return PhoneFfnExecutionContract(
            device_id=device_id,
            n_embd=manifest.embedding_length,
            layer_indices=remote.layer_indices,
            layer_mask=remote.layer_mask,
            columns=manifest.feed_forward_length,
            max_tokens=_integer(parameters, "ffn_max_tokens"),
            activation=_text(parameters, "ffn_activation"),
        )
    operators = _operator_rows(source)
    if not operators:
        raise PhysicalAdapterError(
            "phone FFN execution requires an operator plan"
        )
    max_tokens = _integer(
        parameters,
        "ffn_max_tokens"
        if "ffn_max_tokens" in parameters else "ubatch_size",
    )
    ubatch_size = _integer(parameters, "ubatch_size")
    assistance_phase = parameters.get("ffn_assistance_phase", "all")
    runtime_protocol = parameters.get("ffn_runtime_control_protocol")
    if assistance_phase not in {"all", "decode"} or (
        assistance_phase == "decode"
        and runtime_protocol != "decode-boundary-v1"
    ):
        raise PhysicalAdapterError(
            "phone FFN assistance phase is unsupported"
        )
    n_embd = _integer(parameters, "ffn_n_embd")
    if (
        max_tokens > ubatch_size
        or (assistance_phase == "all" and max_tokens < ubatch_size)
    ):
        raise PhysicalAdapterError(
            "phone FFN policy does not cover the maximum batch"
        )
    if n_embd != manifest.embedding_length:
        raise PhysicalAdapterError(
            "phone FFN embedding shape differs from the model"
        )
    layer_mask, columns, _ = _phone_ffn_from_plan(
        source,
        operators,
        manifest,
        device_id,
        max_tokens,
    )
    selected_layer_mask = parameters.get("ffn_selected_layer_mask")
    selected_columns = parameters.get("ffn_selected_columns")
    if (
        selected_layer_mask is not None
        and selected_layer_mask != layer_mask
    ) or (
        selected_columns is not None
        and selected_columns != columns
    ):
        raise PhysicalAdapterError(
            "phone FFN selected geometry differs from the operator plan"
        )
    indices = tuple(
        index for index in range(manifest.block_count)
        if layer_mask & (1 << index)
    )
    return PhoneFfnExecutionContract(
        device_id=device_id,
        n_embd=n_embd,
        layer_indices=indices,
        layer_mask=layer_mask,
        columns=columns,
        max_tokens=max_tokens,
        activation=_text(parameters, "ffn_activation"),
    )


def phone_ffn_resident_contract(
    command: PhysicalExecutionCommand | PhysicalTransitionCommand,
    manifest: ModelManifest,
) -> PhoneFfnExecutionContract:
    """Return the immutable helper slice that may serve runtime policies."""
    selected = phone_ffn_execution_contract(command, manifest)
    parameters = _phone_ffn_command(command).adapter_parameters
    if parameters.get("ffn_weight_buffer_layout") != "resident-superset":
        return selected
    layer_mask = _integer(parameters, "ffn_resident_layer_mask")
    columns = _integer(parameters, "ffn_resident_columns")
    resident_bytes = _integer(parameters, "ffn_resident_weight_bytes")
    identity = _text(parameters, "resident_model_identity_sha256")
    if (
        resident_bytes <= 0
        or not identity.startswith("sha256:")
        or len(identity) != 71
        or any(value not in "0123456789abcdef" for value in identity[7:])
        or layer_mask >> manifest.block_count
        or selected.layer_mask & ~layer_mask
        or selected.columns > columns
        or columns != manifest.feed_forward_length
        or columns % 32
    ):
        raise PhysicalAdapterError(
            "phone FFN execution exceeds the resident superset"
        )
    indices = tuple(
        index for index in range(manifest.block_count)
        if layer_mask & (1 << index)
    )
    if not indices:
        raise PhysicalAdapterError("phone FFN resident layer set is empty")
    return PhoneFfnExecutionContract(
        device_id=selected.device_id,
        n_embd=selected.n_embd,
        layer_indices=indices,
        layer_mask=layer_mask,
        columns=columns,
        max_tokens=selected.max_tokens,
        activation=selected.activation,
    )


def primary_phone_ffn_contract(
    command: PhysicalExecutionCommand | PhysicalTransitionCommand,
    contract: PhoneFfnExecutionContract,
) -> PhoneFfnExecutionContract:
    """Narrow a server helper slice to the layers of the ticket's own phone.

    With ``phone_helpers`` the server serves the union of every helper's layers; the direct phone
    session, its shards and its stored residency hold only the first row's layers.
    """
    helpers = phone_helper_bindings_from_parameters(
        getattr(_phone_ffn_command(command), "adapter_parameters", {})
    )
    if helpers is None:
        return contract
    union = 0
    for row in helpers:
        union |= row.layer_mask
    layer_mask = contract.layer_mask & helpers[0].layer_mask
    if (
        helpers[0].device_id != contract.device_id
        or contract.layer_mask & ~union
        or not layer_mask
    ):
        raise PhysicalAdapterError(
            "phone helpers differ from the phone FFN slice"
        )
    if layer_mask == contract.layer_mask:
        return contract
    return replace(
        contract,
        layer_mask=layer_mask,
        layer_indices=tuple(
            index for index in contract.layer_indices
            if layer_mask >> index & 1
        ),
    )


def _remote_resident_launch_environment(
    command: PhysicalExecutionCommand | PhysicalTransitionCommand,
    manifest: ModelManifest,
    parameters: Mapping[str, int | str],
    remote_resident,
) -> tuple[str, dict[str, str]]:
    """Environment of a desktop parent whose FFN weights live on phone sessions.

    The server connects to the owning sessions before the model loads, omits the weights,
    validates ownership in its warm-up and refuses controls that target the remote layers.
    Prefill runs on the phone as well, so the phone must accept whole ubatches.
    """
    if getattr(command, "helper_envelope", None) is not None:
        raise PhysicalAdapterError(
            "remote-resident launch cannot carry a helper envelope"
        )
    if remote_resident.parent_artifact_sha256 != manifest.artifact_sha256:
        raise PhysicalAdapterError(
            "remote-resident group belongs to another artifact"
        )
    if not remote_resident.bound:
        raise PhysicalAdapterError(
            "remote-resident launch requires bound session generations"
        )
    phone_device_id = _text(parameters, "phone_device_id")
    layer_mask = _integer(parameters, "ffn_resident_layer_mask")
    columns = _integer(parameters, "ffn_resident_columns")
    max_tokens = _integer(parameters, "ffn_max_tokens")
    ubatch_size = _integer(parameters, "ubatch_size")
    n_embd = _integer(parameters, "ffn_n_embd")
    if (
        parameters.get("ffn_assistance_phase") != "decode"
        or parameters.get("ffn_runtime_control_protocol") != "decode-boundary-v1"
    ):
        raise PhysicalAdapterError(
            "remote-resident launch requires the decode-boundary runtime control"
        )
    if (
        remote_resident.layer_mask & ~layer_mask
        or layer_mask >> manifest.block_count
        or columns != manifest.feed_forward_length
        or max_tokens != ubatch_size
        or n_embd != manifest.embedding_length
    ):
        raise PhysicalAdapterError(
            "remote-resident launch shape differs from the model"
        )
    environment = {
        "S41_SERVER_FFN_ACTIVATION": _text(parameters, "ffn_activation"),
        "S41_SERVER_FFN_ARTIFACT_SHA256": command.artifact_sha256,
        "S41_SERVER_FFN_COLUMNS": str(columns),
        "S41_SERVER_FFN_F16_IO": "1",
        "S41_SERVER_FFN_LAYER_MASK": str(layer_mask),
        "S41_SERVER_FFN_MAX_TOKENS": str(max_tokens),
        "S41_SERVER_FFN_N_EMBD": str(n_embd),
        "S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK": str(remote_resident.layer_mask),
        "S41_SERVER_FFN_RUNTIME_CONTROL": "1",
        "S41_SERVER_FFN_SHARDS": ";".join(
            row.session_id + "@" + row.endpoint + ":" + str(row.layer_mask)
            for row in remote_resident.sessions
        ),
        "S41_SERVER_FFN_TIMEOUT_MS": str(_integer(parameters, "ffn_timeout_ms")),
    }
    environment.update(phone_transport_contract(parameters).server_environment())
    return phone_device_id, environment


def llama_server_launch_contract(
    command: PhysicalExecutionCommand | PhysicalTransitionCommand,
    manifest: ModelManifest,
) -> LlamaServerLaunchContract:
    """Return the exact launch contract already selected by the scheduler."""
    if not isinstance(
        command, (PhysicalExecutionCommand, PhysicalTransitionCommand)
    ) or not isinstance(manifest, ModelManifest):
        raise PhysicalAdapterError("llama-server launch input is invalid")
    if command.artifact_sha256 != manifest.artifact_sha256:
        raise PhysicalAdapterError(
            "llama-server launch artifact differs from the ticket"
        )
    parameters = command.adapter_parameters
    cpu_device_id = _text(parameters, "cpu_device_id")
    gpu_device_id = _text(parameters, "gpu_device_id")
    phone_source = _phone_ffn_command(command)
    dormant_phone_parameters = dormant_phone_ffn_parameters(parameters)
    phone_parameters = (
        phone_source.adapter_parameters
        if dormant_phone_parameters is None
        else dormant_phone_parameters
    )
    host_share_drop_cache = phone_parameters.get("ffn_host_share_drop_cache", 1)
    host_share_populate = phone_parameters.get("ffn_host_share_populate", 1)
    phone_device_id = phone_parameters.get("phone_device_id")
    if phone_device_id is not None and (
        type(phone_device_id) is not str
        or not phone_device_id
        or not phone_device_id.isascii()
    ):
        raise PhysicalAdapterError(
            "llama-server phone adapter parameter is invalid"
        )
    operators = _operator_rows(command)
    helpers = phone_helper_bindings_from_parameters(phone_parameters)
    gpu_layers = (
        _gpu_layers_from_plan(
            operators,
            manifest,
            cpu_device_id,
            gpu_device_id,
            None if phone_source is not command else phone_device_id,
            () if phone_source is not command or helpers is None
            else tuple(row.device_id for row in helpers[1:]),
        )
        if operators else _integer(parameters, "gpu_layers")
    )

    ffn_environment: dict[str, str] = {}
    remote_resident = getattr(
        command.execution_contract, "remote_resident_ffn", None
    )
    if remote_resident is not None:
        phone_device_id, ffn_environment = _remote_resident_launch_environment(
            command, manifest, parameters, remote_resident
        )
    elif phone_device_id is not None:
        if dormant_phone_parameters is not None:
            runtime_control = (
                phone_parameters.get("ffn_assistance_phase") == "decode"
                and phone_parameters.get("ffn_runtime_control_protocol")
                    == "decode-boundary-v1"
            )
            if not runtime_control:
                raise PhysicalAdapterError(
                    "dormant phone FFN runtime control is invalid"
                )
            phone_operators = ()
            layer_mask = _integer(
                phone_parameters, "ffn_resident_layer_mask"
            )
            columns = _integer(
                phone_parameters, "ffn_resident_columns"
            )
            max_tokens = _integer(phone_parameters, "ffn_max_tokens")
            policy = None
        else:
            runtime_control = (
                phone_parameters.get("ffn_assistance_phase") == "decode"
                and phone_source.execution_contract.execution_mode in {
                    "adaptive-split", "static-split"
                }
            )
            phone_operators = _operator_rows(phone_source)
            if phone_operators:
                selected_phone_contract = phone_ffn_execution_contract(
                    command, manifest
                )
                phone_contract = phone_ffn_resident_contract(
                    command, manifest
                )
                layer_mask = phone_contract.layer_mask
                columns = phone_contract.columns
                max_tokens = phone_contract.max_tokens
                policy = (
                    None if runtime_control else
                    f"{selected_phone_contract.max_tokens}:"
                    f"{selected_phone_contract.columns}"
                )
            else:
                layer_mask = _integer(parameters, "ffn_layer_mask")
                columns = _integer(parameters, "ffn_columns")
                max_tokens = _integer(
                    phone_parameters,
                    "ffn_max_tokens"
                    if "ffn_max_tokens" in phone_parameters
                    else "ubatch_size",
                )
                policy = _text(parameters, "ffn_policy")
        n_embd = _integer(phone_parameters, "ffn_n_embd")
        if (
            n_embd != manifest.embedding_length
            or layer_mask <= 0
            or columns <= 0
            or max_tokens <= 0
        ):
            raise PhysicalAdapterError(
                "llama-server FFN shape differs from the model"
            )
        ffn_environment = {
            "S41_SERVER_FFN_ACTIVATION": _text(
                phone_parameters, "ffn_activation"
            ),
            "S41_SERVER_FFN_ARTIFACT_SHA256": (
                command.artifact_sha256
            ),
            "S41_SERVER_FFN_COLUMNS": str(columns),
            "S41_SERVER_FFN_F16_IO": "1",
            "S41_SERVER_FFN_LAYER_MASK": str(layer_mask),
            "S41_SERVER_FFN_MAX_TOKENS": str(max_tokens),
            "S41_SERVER_FFN_N_EMBD": str(n_embd),
            "S41_SERVER_FFN_TIMEOUT_MS": str(
                _integer(phone_parameters, "ffn_timeout_ms")
            ),
        }
        if policy is not None:
            ffn_environment["S41_SERVER_FFN_POLICY"] = policy
        if runtime_control:
            ffn_environment["S41_SERVER_FFN_RUNTIME_CONTROL"] = "1"
        host_share_release = phone_parameters.get("ffn_host_share_release", 0)
        if type(host_share_release) is not int or host_share_release not in (0, 1):
            raise PhysicalAdapterError(
                "llama-server FFN host share release flag is invalid"
            )
        if host_share_release:
            # decode-only relocation: the server releases the phone-executed column suffix while
            # every slot decodes; it needs the decode-boundary runtime control to know the phase
            if not runtime_control:
                raise PhysicalAdapterError(
                    "llama-server FFN host share release requires decode-phase runtime control"
                )
            ffn_environment["S41_SERVER_FFN_DORMANT_HOST_SHARE"] = "1"
        if (
            dormant_phone_parameters is None
            and phone_source.execution_contract.phone_shards
        ):
            ffn_environment["S41_SERVER_FFN_SHARDS"] = ";".join(
                shard.session_id
                + "@"
                + shard.endpoint
                + ":"
                + str(shard.layer_mask)
                for shard in phone_source.execution_contract.phone_shards
            )
        ffn_environment.update(
            phone_transport_contract(
                phone_parameters
            ).server_environment()
        )
        if helpers is not None:
            # several helper phones: one union layer mask, one client per owning phone
            ffn_environment = dict(phone_helper_launch_environment(
                helpers, ffn_environment, phone_parameters
            ))
    if phone_device_id is None or not ffn_environment:
        # A server launched without an FFN runtime (plain desktop route, or a cold desktop parent whose
        # phone helper is not attached yet) has no cache policy to apply and cannot confirm one; keep the
        # defaults so its contract and digest are those of a plain server. A phone route whose FFN
        # environment lacks the release still fails closed in server_environment().
        host_share_drop_cache = 1
        host_share_populate = 1
    speculative = speculative_launch_contract(
        parameters,
        phone_attached=phone_device_id is not None and bool(ffn_environment),
        phone_max_tokens=_speculative_phone_max_tokens(ffn_environment),
    )
    return LlamaServerLaunchContract(
        model_alias=_text(parameters, "model_alias"),
        context_size=_integer(parameters, "context_size"),
        parallel=_integer(parameters, "parallel"),
        batch_size=_integer(parameters, "batch_size"),
        ubatch_size=_integer(parameters, "ubatch_size"),
        gpu_layers=gpu_layers,
        cpu_device_id=cpu_device_id,
        gpu_device_id=gpu_device_id,
        phone_device_id=phone_device_id,
        ffn_environment=ffn_environment,
        speculative=speculative,
        ffn_host_share_drop_cache=host_share_drop_cache,
        ffn_host_share_populate=host_share_populate,
        kv_cpu_layers=_kv_cpu_layers(parameters, manifest),
        kv_device_cells=_kv_device_cells(parameters, manifest),
        cuda_graph_mode=parameters.get("cuda_graph_mode", "default"),
        scheduler_trace_path=parameters.get("scheduler_trace_path"),
        logits_trace_path=parameters.get("logits_trace_path"),
        desktop_launch_mode=parameters.get("desktop_launch_mode", "canonical"),
        threads=_integer(parameters, "threads")
            if "threads" in parameters else 0,
        threads_batch=_integer(parameters, "threads_batch")
            if "threads_batch" in parameters else 0,
        cpu_affinity=(
            _text(parameters, "cpu_affinity")
            if "cpu_affinity" in parameters else None
        ),
    )


def llama_server_capability_contract(
    capability: RuntimeExecutorCapability,
    manifest: ModelManifest,
) -> LlamaServerLaunchContract:
    """Validate launch data registered for one resident base executor."""
    if (
        not isinstance(capability, RuntimeExecutorCapability)
        or not isinstance(manifest, ModelManifest)
    ):
        raise PhysicalAdapterError(
            "llama-server executor capability is invalid"
        )
    parameters = capability.adapter_parameters
    if not parameters:
        raise PhysicalAdapterError(
            "llama-server executor launch parameters are absent"
        )
    phone_device_id = parameters.get("phone_device_id")
    if phone_device_id is not None:
        raise PhysicalAdapterError(
            "base llama-server launch cannot configure a phone split"
        )
    return LlamaServerLaunchContract(
        model_alias=_text(parameters, "model_alias"),
        context_size=_integer(parameters, "context_size"),
        parallel=_integer(parameters, "parallel"),
        batch_size=_integer(parameters, "batch_size"),
        ubatch_size=_integer(parameters, "ubatch_size"),
        gpu_layers=_integer(parameters, "gpu_layers"),
        cpu_device_id=_text(parameters, "cpu_device_id"),
        gpu_device_id=_text(parameters, "gpu_device_id"),
        phone_device_id=None,
        ffn_environment={},
        speculative=speculative_launch_contract(
            parameters, phone_attached=False, phone_max_tokens=None
        ),
        ffn_host_share_drop_cache=parameters.get("ffn_host_share_drop_cache", 1),
        ffn_host_share_populate=parameters.get("ffn_host_share_populate", 1),
        kv_cpu_layers=_kv_cpu_layers(parameters, manifest),
        kv_device_cells=_kv_device_cells(parameters, manifest),
        cuda_graph_mode=parameters.get("cuda_graph_mode", "default"),
        scheduler_trace_path=parameters.get("scheduler_trace_path"),
        logits_trace_path=parameters.get("logits_trace_path"),
        desktop_launch_mode=parameters.get("desktop_launch_mode", "canonical"),
        threads=_integer(parameters, "threads")
            if "threads" in parameters else 0,
        threads_batch=_integer(parameters, "threads_batch")
            if "threads_batch" in parameters else 0,
        cpu_affinity=(
            _text(parameters, "cpu_affinity")
            if "cpu_affinity" in parameters else None
        ),
    )
