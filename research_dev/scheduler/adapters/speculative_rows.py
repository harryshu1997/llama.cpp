"""Speculative rows: draft-model speculation sized to the phone FFN row budget.

Decode is bandwidth-bound on every device of the rig: a desktop step costs the same at batch 1
or 4 and a phone FFN call 9.6 ms for one row or 12.7 ms for four, so joules per token divide by
the rows a step carries. Draft-model speculation turns one request into ``1 + drafted``
verification rows per step without changing the greedy output (temperature 0, exact
verification against the target). The rows of one phone FFN call are the sum over the batched
slots of ``1 + drafted`` and must never exceed the phone contract (``S41_SERVER_FFN_MAX_TOKENS``),
which the phone refuses. Everything here is inert unless the campaign carries
``speculative_rows``; without the key no adapter parameter, launch argument, request field or
RESULT key changes.

Server facts the policy rests on (``tools/server/server-context.cpp``, ``common/speculative.cpp``):
``--spec-type draft-simple`` must accompany ``--spec-draft-model``; the launch
``--spec-draft-n-max`` bounds every slot; the per-request ``speculative.n_max`` field is
compiled out (``server-schema.cpp`` ``#if 0``); a phone-split server exports
``LLAMA_FFN_SPLIT_*`` to the process before loading, which a draft model loaded in the same
process would inherit. Phone-assisted launches therefore carry a draft only when the campaign
pins a patched server build (``patched_server_sha256``, see ``server_speculative.diff``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Mapping, Sequence

from .._internal.gguf_metadata import GGUFMetadataError, read_gguf_metadata
from .._internal.model_manifest import ModelManifest, _sha256_file
from .contracts import PhysicalAdapterError

DRAFT_MODEL_PATH_PARAMETER = "speculative_draft_model_path"
DRAFT_SHA256_PARAMETER = "speculative_draft_sha256"
DRAFT_BYTES_PARAMETER = "speculative_draft_bytes"
DRAFT_MAX_PARAMETER = "speculative_draft_max"
DRAFT_MIN_PARAMETER = "speculative_draft_min"
DRAFT_GPU_LAYERS_PARAMETER = "speculative_draft_gpu_layers"
ROW_BUDGET_PARAMETER = "speculative_row_budget"
QUALIFIED_ROWS_PARAMETER = "speculative_qualified_rows"
PATCHED_SERVER_PARAMETER = "speculative_patched_server_sha256"

REQUIRED_SPECULATIVE_PARAMETERS = frozenset({
    DRAFT_MODEL_PATH_PARAMETER,
    DRAFT_SHA256_PARAMETER,
    DRAFT_BYTES_PARAMETER,
    DRAFT_MAX_PARAMETER,
    DRAFT_MIN_PARAMETER,
    DRAFT_GPU_LAYERS_PARAMETER,
})
OPTIONAL_SPECULATIVE_PARAMETERS = frozenset({
    ROW_BUDGET_PARAMETER,
    QUALIFIED_ROWS_PARAMETER,
    PATCHED_SERVER_PARAMETER,
})
SPECULATIVE_PARAMETERS = REQUIRED_SPECULATIVE_PARAMETERS | OPTIONAL_SPECULATIVE_PARAMETERS

# the fork's draft loop accepts at most this many draft tokens per step for a 4-row phone budget
DRAFT_MAX_LIMIT = 3
TOKEN_EMBEDDING_TENSOR = "token_embd.weight"


def _sha256_text(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(char not in "0123456789abcdef" for char in value[7:])
    ):
        raise PhysicalAdapterError(name + " must be a sha256: digest")
    return value


def _count(value: object, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise PhysicalAdapterError(name + " is invalid")
    return value


def draft_vocabulary_size(path: Path) -> int:
    """Vocabulary size of a GGUF from its token embedding (GGUF ``ne[1]`` of ``token_embd``)."""
    try:
        reader = read_gguf_metadata(path)
    except (OSError, GGUFMetadataError) as error:
        raise PhysicalAdapterError("speculative draft GGUF is unreadable: " + str(error)) from error
    for tensor in reader.tensors:
        if tensor.name == TOKEN_EMBEDDING_TENSOR and len(tensor.shape) == 2:
            return tensor.shape[1]
    raise PhysicalAdapterError("speculative draft GGUF has no token embedding")


def target_vocabulary_size(manifest: ModelManifest) -> int:
    """Vocabulary size of the target from its manifest's token embedding."""
    for tensor in manifest.tensors:
        if tensor.tensor_id == TOKEN_EMBEDDING_TENSOR and len(tensor.shape) == 2:
            return tensor.shape[1]
    raise PhysicalAdapterError("speculative target manifest has no token embedding")


def speculative_adapter_parameters(
    configuration: object, manifest: ModelManifest
) -> dict[str, int | str]:
    """Adapter parameters of one model's draft: the checked draft file bound by digest and size.

    ``configuration`` is the campaign's ``SpeculativeRowsModelConfiguration``; the draft must
    exist and share the target's vocabulary (the server refuses a mismatch only after loading).
    """
    path = Path(configuration.draft_model_path)
    if not path.is_absolute() or not path.is_file():
        raise PhysicalAdapterError("speculative draft model path is not a file: " + str(path))
    draft_vocabulary = draft_vocabulary_size(path)
    target_vocabulary = target_vocabulary_size(manifest)
    if draft_vocabulary != target_vocabulary:
        raise PhysicalAdapterError(
            f"speculative draft vocabulary {draft_vocabulary} differs from the target {target_vocabulary}"
        )
    parameters: dict[str, int | str] = {
        DRAFT_MODEL_PATH_PARAMETER: str(path),
        DRAFT_SHA256_PARAMETER: _sha256_file(path),
        DRAFT_BYTES_PARAMETER: path.stat().st_size,
        DRAFT_MAX_PARAMETER: _count(configuration.draft_max, "speculative draft_max", minimum=1, maximum=DRAFT_MAX_LIMIT),
        DRAFT_MIN_PARAMETER: _count(configuration.draft_min, "speculative draft_min", maximum=configuration.draft_max),
        DRAFT_GPU_LAYERS_PARAMETER: _count(configuration.draft_gpu_layers, "speculative draft_gpu_layers"),
    }
    if configuration.row_budget is not None:
        parameters[ROW_BUDGET_PARAMETER] = _count(configuration.row_budget, "speculative row_budget", minimum=1)
    if configuration.qualified_rows is not None:
        rows = tuple(configuration.qualified_rows)
        if not rows or any(type(row) is not int or row < 1 for row in rows) or len(set(rows)) != len(rows):
            raise PhysicalAdapterError("speculative qualified_rows are invalid")
        parameters[QUALIFIED_ROWS_PARAMETER] = ",".join(str(row) for row in sorted(rows))
    if configuration.patched_server_sha256 is not None:
        parameters[PATCHED_SERVER_PARAMETER] = _sha256_text(
            configuration.patched_server_sha256, "speculative patched_server_sha256")
    return parameters


def reachable_call_rows(parallel: int, draft_max: int, draft_min: int) -> frozenset[int]:
    """Every row count one phone FFN call can carry with up to ``parallel`` speculating slots.

    A slot contributes one row plus its accepted draft: the draft loop returns either nothing
    (``n_min`` not met, context or budget exhausted) or between ``draft_min`` and ``draft_max``
    tokens, so a slot carries ``1`` or ``1 + draft_min .. 1 + draft_max`` rows."""
    parallel = _count(parallel, "speculative parallel", minimum=1)
    draft_max = _count(draft_max, "speculative draft_max", maximum=DRAFT_MAX_LIMIT)
    draft_min = _count(draft_min, "speculative draft_min", maximum=draft_max)
    per_slot = {1} | {1 + drafted for drafted in range(max(1, draft_min), draft_max + 1)}
    sums = {0}
    reachable = set()
    for _ in range(parallel):
        sums = {total + rows for total in sums for rows in per_slot}
        reachable |= sums
    return frozenset(reachable)


def static_draft_max(
    *,
    draft_max: int,
    draft_min: int,
    row_budget: int,
    parallel: int,
    qualified_rows: Sequence[int] | None = None,
) -> tuple[int, str | None]:
    """The largest launch-wide draft bound whose every reachable call fits the phone.

    The server applies one ``--spec-draft-n-max`` to every slot and the fork exposes no
    per-request field, so with ``parallel`` slots the bound must keep
    ``parallel x (1 + K)`` within ``row_budget`` and every reachable row count inside
    ``qualified_rows``. Returns ``(K, None)`` or ``(0, reason)`` when no draft row fits."""
    row_budget = _count(row_budget, "speculative row_budget", minimum=1)
    qualified = frozenset(range(1, row_budget + 1)) if qualified_rows is None else frozenset(qualified_rows)
    if not qualified or any(type(row) is not int or row < 1 for row in qualified):
        raise PhysicalAdapterError("speculative qualified rows are invalid")
    for candidate in range(_count(draft_max, "speculative draft_max", maximum=DRAFT_MAX_LIMIT), 0, -1):
        reachable = reachable_call_rows(parallel, candidate, min(draft_min, candidate))
        if max(reachable) <= row_budget and reachable <= qualified:
            return candidate, None
    return 0, (
        f"{parallel} decode slots fill the {row_budget}-row phone budget (qualified rows "
        + ",".join(str(row) for row in sorted(qualified)) + "); no draft row fits without per-request control"
    )


def per_request_draft_max(
    *,
    draft_max: int,
    row_budget: int,
    parallel: int,
    reserved_rows: Sequence[int],
) -> int:
    """Draft bound of one arriving request under per-request control (patched server only).

    The request keeps its bound for its lifetime, so it may only take what the budget leaves
    after every active request's reservation and one plain row for every slot that can still
    fill: ``row_budget - sum(reserved) - (parallel - 1 - len(reserved)) - 1``. Unknown or
    over-full state yields 0 (fail closed)."""
    draft_max = _count(draft_max, "speculative draft_max", maximum=DRAFT_MAX_LIMIT)
    row_budget = _count(row_budget, "speculative row_budget", minimum=1)
    parallel = _count(parallel, "speculative parallel", minimum=1)
    reserved = tuple(reserved_rows)
    if any(type(rows) is not int or rows < 1 for rows in reserved):
        raise PhysicalAdapterError("speculative reserved rows are invalid")
    if len(reserved) >= parallel:
        return 0
    free_slots = parallel - 1 - len(reserved)
    return max(0, min(draft_max, row_budget - sum(reserved) - free_slots - 1))


class SpeculativeRowLedger:
    """Rows reserved by the live requests of every server endpoint (per-request control)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[str, dict[str, int]] = {}

    def reserved_rows(self, endpoint: str) -> tuple[int, ...]:
        """Reservations of the other live requests on ``endpoint``, in request-id order."""
        with self._lock:
            return tuple(rows for _, rows in sorted(self._rows.get(endpoint, {}).items()))

    def reserve(self, endpoint: str, request_id: str, rows: int) -> None:
        with self._lock:
            self._reserve_locked(endpoint, request_id, rows)

    def _reserve_locked(self, endpoint: str, request_id: str, rows: int) -> None:
        if type(rows) is not int or rows < 1:
            raise PhysicalAdapterError("speculative reservation is invalid")
        live = self._rows.setdefault(endpoint, {})
        if request_id in live:
            raise PhysicalAdapterError("speculative reservation is duplicated")
        live[request_id] = rows

    def admit(self, endpoint: str, request_id: str, grant) -> int | None:
        """Grant one arriving request under the lock: ``grant(reserved_rows)`` returns the rows
        to reserve (None reserves nothing); concurrent arrivals see each other's reservation."""
        with self._lock:
            rows = grant(tuple(rows for _, rows in sorted(self._rows.get(endpoint, {}).items())))
            if rows is not None:
                self._reserve_locked(endpoint, request_id, rows)
            return rows

    def release(self, endpoint: str, request_id: str) -> None:
        with self._lock:
            live = self._rows.get(endpoint, {})
            live.pop(request_id, None)
            if not live:
                self._rows.pop(endpoint, None)


@dataclass(frozen=True)
class LlamaServerSpeculativeContract:
    """Draft-model speculation of one launched llama-server.

    ``draft_max`` is the launch ``--spec-draft-n-max``: the static bound every slot obeys, or
    under ``per_request_control`` the cap the per-request ``speculative.n_max`` stays below.
    ``row_budget`` is the phone contract the rows of one FFN call stay within (0: no phone)."""

    draft_model_path: str
    draft_sha256: str
    draft_bytes: int
    draft_max: int
    draft_min: int
    draft_gpu_layers: int
    row_budget: int
    per_request_control: bool
    patched_server_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.draft_model_path) is not str or not self.draft_model_path.startswith("/"):
            raise PhysicalAdapterError("speculative draft model path must be absolute")
        _sha256_text(self.draft_sha256, "speculative draft digest")
        _count(self.draft_bytes, "speculative draft bytes", minimum=1)
        _count(self.draft_max, "speculative draft_max", minimum=1, maximum=DRAFT_MAX_LIMIT)
        _count(self.draft_min, "speculative draft_min", maximum=self.draft_max)
        _count(self.draft_gpu_layers, "speculative draft_gpu_layers")
        _count(self.row_budget, "speculative row_budget")
        if type(self.per_request_control) is not bool:
            raise PhysicalAdapterError("speculative per-request control flag is invalid")
        if self.per_request_control and self.patched_server_sha256 is None:
            raise PhysicalAdapterError("speculative per-request control requires the patched server pin")
        if self.patched_server_sha256 is not None:
            _sha256_text(self.patched_server_sha256, "speculative patched server digest")

    def to_json(self) -> dict[str, object]:
        return {
            "draft_bytes": self.draft_bytes,
            "draft_gpu_layers": self.draft_gpu_layers,
            "draft_max": self.draft_max,
            "draft_min": self.draft_min,
            "draft_model_path": self.draft_model_path,
            "draft_sha256": self.draft_sha256,
            "patched_server_sha256": self.patched_server_sha256,
            "per_request_control": self.per_request_control,
            "row_budget": self.row_budget,
        }


def _qualified_rows(parameters: Mapping[str, int | str]) -> tuple[int, ...] | None:
    text = parameters.get(QUALIFIED_ROWS_PARAMETER)
    if text is None:
        return None
    if type(text) is not str or not text:
        raise PhysicalAdapterError("speculative qualified rows parameter is invalid")
    rows = []
    for item in text.split(","):
        if not item.isdecimal() or int(item) < 1:
            raise PhysicalAdapterError("speculative qualified rows parameter is invalid")
        rows.append(int(item))
    return tuple(rows)


def speculative_launch_contract(
    parameters: Mapping[str, int | str],
    *,
    phone_attached: bool,
    phone_max_tokens: int | None,
) -> LlamaServerSpeculativeContract | None:
    """The speculation a launch carries, from its adapter parameters and phone contract.

    Absent parameters give None (byte-identical launch). A desktop-only launch drafts up to
    ``draft_max``: its rows never reach a phone. A phone-assisted launch drafts only under the
    patched server pin (the stock server leaks its FFN split into the draft model and ignores
    per-request bounds); the launch then carries the cap and every request its own
    ``speculative.n_max`` from ``per_request_draft_max``. When even a lone request gets no
    draft row (``parallel`` plain rows fill the budget) the launch carries no draft.
    """
    present = SPECULATIVE_PARAMETERS & set(parameters)
    if not present:
        return None
    if not REQUIRED_SPECULATIVE_PARAMETERS <= present:
        raise PhysicalAdapterError("speculative adapter parameters are incomplete")
    path = parameters[DRAFT_MODEL_PATH_PARAMETER]
    if type(path) is not str or not path.startswith("/"):
        raise PhysicalAdapterError("speculative draft model path must be absolute")
    draft_max = _count(parameters[DRAFT_MAX_PARAMETER], "speculative draft_max", minimum=1, maximum=DRAFT_MAX_LIMIT)
    draft_min = _count(parameters[DRAFT_MIN_PARAMETER], "speculative draft_min", maximum=draft_max)
    parallel = _count(parameters.get("parallel"), "speculative launch parallel", minimum=1)
    pinned = parameters.get(PATCHED_SERVER_PARAMETER)
    if pinned is not None:
        _sha256_text(pinned, "speculative patched server digest")
    if type(phone_attached) is not bool:
        raise PhysicalAdapterError("speculative phone attachment flag is invalid")
    row_budget = 0
    if phone_attached:
        if pinned is None or type(phone_max_tokens) is not int or phone_max_tokens < 1:
            return None
        declared = parameters.get(ROW_BUDGET_PARAMETER)
        row_budget = phone_max_tokens if declared is None else min(
            _count(declared, "speculative row_budget", minimum=1), phone_max_tokens)
        # the patched server drafts per request; a lone request must still get a draft row
        if per_request_draft_max(draft_max=draft_max, row_budget=row_budget, parallel=parallel,
                                 reserved_rows=()) == 0:
            return None
    return LlamaServerSpeculativeContract(
        draft_model_path=path,
        draft_sha256=_sha256_text(parameters[DRAFT_SHA256_PARAMETER], "speculative draft digest"),
        draft_bytes=_count(parameters[DRAFT_BYTES_PARAMETER], "speculative draft bytes", minimum=1),
        draft_max=draft_max,
        draft_min=min(draft_min, draft_max),
        draft_gpu_layers=_count(parameters[DRAFT_GPU_LAYERS_PARAMETER], "speculative draft_gpu_layers"),
        row_budget=row_budget,
        per_request_control=pinned is not None,
        patched_server_sha256=pinned,
    )


@dataclass(frozen=True)
class SpeculativeRequestContract:
    """What one completion request tells the server and what its RESULT row records.

    ``n_max`` is the per-request draft bound sent as ``speculative.n_max`` (only under
    per-request control, None otherwise: the launch bound applies and the body is unchanged);
    ``reserved_rows`` is the ledger reservation ``1 + n_max`` released when the request ends."""

    draft_max: int
    draft_min: int
    row_budget: int
    parallel: int
    per_request_control: bool
    n_max: int | None = None

    def __post_init__(self) -> None:
        _count(self.draft_max, "speculative draft_max", minimum=1, maximum=DRAFT_MAX_LIMIT)
        _count(self.draft_min, "speculative draft_min", maximum=self.draft_max)
        _count(self.row_budget, "speculative row_budget")
        _count(self.parallel, "speculative parallel", minimum=1)
        if type(self.per_request_control) is not bool:
            raise PhysicalAdapterError("speculative per-request control flag is invalid")
        if self.per_request_control != (self.n_max is not None):
            raise PhysicalAdapterError("speculative per-request bound requires per-request control")
        if self.n_max is not None:
            _count(self.n_max, "speculative n_max", maximum=self.draft_max)

    @property
    def reserved_rows(self) -> int:
        return 1 + (0 if self.n_max is None else self.n_max)

    def body_fields(self) -> dict[str, object]:
        """The ``speculative`` object of the completion body; empty without per-request control."""
        if self.n_max is None:
            return {}
        return {"speculative": {"n_max": self.n_max, "n_min": min(self.draft_min, self.n_max)}}


def speculative_request_contract(
    parameters: Mapping[str, int | str],
    reserved_rows: Sequence[int],
) -> SpeculativeRequestContract | None:
    """The request-side contract from a command's adapter parameters and the endpoint ledger."""
    present = SPECULATIVE_PARAMETERS & set(parameters)
    if not present:
        return None
    if not REQUIRED_SPECULATIVE_PARAMETERS <= present:
        raise PhysicalAdapterError("speculative adapter parameters are incomplete")
    draft_max = _count(parameters[DRAFT_MAX_PARAMETER], "speculative draft_max", minimum=1, maximum=DRAFT_MAX_LIMIT)
    draft_min = _count(parameters[DRAFT_MIN_PARAMETER], "speculative draft_min", maximum=draft_max)
    parallel = _count(parameters.get("parallel"), "speculative request parallel", minimum=1)
    phone_attached = parameters.get("phone_device_id") is not None
    pinned = parameters.get(PATCHED_SERVER_PARAMETER)
    if pinned is not None:
        _sha256_text(pinned, "speculative patched server digest")
    budget = parameters.get(ROW_BUDGET_PARAMETER)
    max_tokens = parameters.get("ffn_max_tokens", min(_count(parameters.get("ubatch_size"), "speculative ubatch", minimum=1), parallel))
    row_budget = 0
    if phone_attached:
        row_budget = _count(max_tokens, "speculative phone max tokens", minimum=1) if budget is None else min(
            _count(budget, "speculative row_budget", minimum=1), _count(max_tokens, "speculative phone max tokens", minimum=1))
    n_max = None
    if pinned is not None:
        n_max = draft_max if not phone_attached else per_request_draft_max(
            draft_max=draft_max, row_budget=row_budget, parallel=parallel, reserved_rows=reserved_rows)
    return SpeculativeRequestContract(
        draft_max=draft_max,
        draft_min=draft_min,
        row_budget=row_budget,
        parallel=parallel,
        per_request_control=pinned is not None,
        n_max=n_max,
    )


def speculative_completion_statistics(
    timings: Mapping[str, object],
    output_tokens: int,
    request: SpeculativeRequestContract,
) -> dict[str, object]:
    """Per-request draft statistics from the completion ``timings``.

    The server reports ``draft_n`` and ``draft_n_accepted`` only when it drafted; every
    verification step yields one sampled token plus the accepted draft, so the step count is
    ``output_tokens - draft_n_accepted`` and tokens per step follow."""
    output_tokens = _count(output_tokens, "speculative output tokens", minimum=1)
    draft_n = _count(timings.get("draft_n", 0), "speculative draft_n")
    accepted = _count(timings.get("draft_n_accepted", 0), "speculative draft_n_accepted", maximum=draft_n)
    if accepted > output_tokens - 1:
        raise PhysicalAdapterError("speculative accepted draft exceeds the output")
    steps = output_tokens - accepted
    return {
        "acceptance_rate_ppm": 0 if draft_n == 0 else accepted * 1_000_000 // draft_n,
        "draft_accepted": accepted,
        "draft_n": draft_n,
        "n_max": request.n_max,
        "per_request_control": request.per_request_control,
        "row_budget": request.row_budget,
        "schema": "s42-speculative-rows-request-v1",
        "tokens_per_step_ppm": output_tokens * 1_000_000 // steps,
        "verif_steps": steps,
    }


__all__ = [
    "DRAFT_BYTES_PARAMETER",
    "DRAFT_GPU_LAYERS_PARAMETER",
    "DRAFT_MAX_LIMIT",
    "DRAFT_MAX_PARAMETER",
    "DRAFT_MIN_PARAMETER",
    "DRAFT_MODEL_PATH_PARAMETER",
    "DRAFT_SHA256_PARAMETER",
    "LlamaServerSpeculativeContract",
    "PATCHED_SERVER_PARAMETER",
    "QUALIFIED_ROWS_PARAMETER",
    "ROW_BUDGET_PARAMETER",
    "SPECULATIVE_PARAMETERS",
    "SpeculativeRequestContract",
    "SpeculativeRowLedger",
    "draft_vocabulary_size",
    "per_request_draft_max",
    "reachable_call_rows",
    "speculative_adapter_parameters",
    "speculative_completion_statistics",
    "speculative_launch_contract",
    "speculative_request_contract",
    "static_draft_max",
    "target_vocabulary_size",
]
