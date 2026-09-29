"""Measured decode-split selection and decode-phase memory accounting for the FFN relocation.

Selection: choose the phone share of the decode-only FFN relocation from a digest-pinned atlas of
measured points (fraction -> decode ms/token, decode-phase host power, bytes released at the first
token). A point applies to a request only when the execution environment matches exactly (artifact,
desktop placement, KV plan, context cells, batch shape, column quantum, phone sessions, runtime
bundle) and the request's prompt length lies inside the point's validated range with its expected
total context not beyond what the measurement covered. Objectives: ``latency``, ``energy`` (host plus
assumed-phone energy per generated token), ``memory`` (largest release under optional bounds).
Nothing is interpolated beyond the validated ranges: unmeasured requests fail closed.

Accounting: the bytes the server releases at the first generated token are host memory only while
that server decodes. ``DecodeReleaseAccountant`` books the phone share of one physical allocation
(a server endpoint) as its own ledger owner: reserved before the prompt (the pages are resident
during prefill), released against that server's ``phase=decode`` proof, bound to the reserved
layers/columns and consumed once per release event, and re-reserved before the next prompt in one
transaction that leaves the ledger untouched on failure. If the room was consumed in between (KV
growth, another tenant), the prompt is held. Decode-phase credit therefore never funds prefill,
which is what ``reserve_layer_kv`` already refuses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping, Sequence

from .adaptive_decode_contracts import AdaptiveDecodePolicy
from .runtime_cost import RuntimeMemoryDemand
from .runtime_resources import RuntimeHostShareReleaseProof, RuntimeMemoryLedger, RuntimeResourceError

ATLAS_SCHEMA = "scheduler-decode-split-atlas-v2"
OBJECTIVES = ("latency", "energy", "memory")
SHARE_KIND = "dormant-host-share"
SHORTFALL_KIND = "dormant-host-share-shortfall"
GROWTH_KIND = "decode-kv-growth"
ENVIRONMENT_FIELDS = ("artifact_sha256", "gpu_layers", "context_cells", "parallel", "batch", "ubatch", "kv_plan_sha256",
                      "column_quantum", "session_masks", "runtime_sha256")


class DecodeSplitSelectionError(ValueError):
    pass


def _int(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DecodeSplitSelectionError(f"{name} must be an integer >= {minimum}")
    return value


def _number(name: str, value: object) -> float:
    if type(value) not in (int, float) or value != value or value < 0:
        raise DecodeSplitSelectionError(f"{name} must be a non-negative number")
    return float(value)


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise DecodeSplitSelectionError(f"{name} must be non-empty ASCII text")
    return value


def _sha(name: str, value: object) -> str:
    text = _text(name, value)
    if not text.startswith("sha256:") or len(text) != 71:
        raise DecodeSplitSelectionError(f"{name} must be a sha256: digest")
    return text


@dataclass(frozen=True)
class DecodeSplitEnvironment:
    """What must match exactly between a measured point and a request before a prediction is used."""
    artifact_sha256: str
    gpu_layers: int
    context_cells: int
    parallel: int
    batch: int
    ubatch: int
    kv_plan_sha256: str
    column_quantum: int
    session_masks: tuple[tuple[str, int], ...]
    runtime_sha256: str

    def __post_init__(self) -> None:
        _sha("artifact", self.artifact_sha256)
        _int("gpu layers", self.gpu_layers)
        _int("context cells", self.context_cells, 1)
        _int("parallel", self.parallel, 1)
        _int("batch", self.batch, 1)
        _int("ubatch", self.ubatch, 1)
        _sha("kv plan", self.kv_plan_sha256)
        _int("column quantum", self.column_quantum, 1)
        masks = tuple(self.session_masks)
        if not masks or any(type(row) is not tuple or len(row) != 2 for row in masks):
            raise DecodeSplitSelectionError("session masks must be (name, mask) pairs")
        for name, mask in masks:
            _text("session name", name)
            _int("session mask", mask, 1)
        if list(masks) != sorted(masks):
            raise DecodeSplitSelectionError("session masks must be sorted by name")
        _sha("runtime bundle", self.runtime_sha256)

    @classmethod
    def from_json(cls, value: object) -> "DecodeSplitEnvironment":
        if not isinstance(value, Mapping) or set(value) != set(ENVIRONMENT_FIELDS):
            raise DecodeSplitSelectionError("environment has unexpected fields")
        masks = value.get("session_masks")
        if not isinstance(masks, (list, tuple, Mapping)):
            raise DecodeSplitSelectionError("session masks are invalid")
        pairs = tuple(sorted((str(k), int(v)) for k, v in (masks.items() if isinstance(masks, Mapping) else masks)))
        row = {k: value[k] for k in ENVIRONMENT_FIELDS if k != "session_masks"}
        return cls(session_masks=pairs, **row)

    def to_json(self) -> dict[str, object]:
        return {"artifact_sha256": self.artifact_sha256, "gpu_layers": self.gpu_layers, "context_cells": self.context_cells,
                "parallel": self.parallel, "batch": self.batch, "ubatch": self.ubatch, "kv_plan_sha256": self.kv_plan_sha256,
                "column_quantum": self.column_quantum, "session_masks": [list(row) for row in self.session_masks],
                "runtime_sha256": self.runtime_sha256}


@dataclass(frozen=True)
class DecodeSplitPoint:
    run: str
    request_id: str
    result_sha256: str
    environment: DecodeSplitEnvironment
    prompt_tokens: int
    output_tokens: int
    validated_prompt_tokens_min: int
    validated_prompt_tokens_max: int
    validated_total_tokens_max: int
    memory_budget: str | None
    phone_attached: bool
    split_fraction_ppm: int
    host_columns: int
    dormant_release: bool
    decode_ms_per_token: float
    prefill_s: float | None
    decode_host_power_w: float | None
    released_bytes: int
    release_elapsed_us: int | None
    kv_cpu_bytes_per_token: int
    kv_gpu_bytes_per_token: int

    def __post_init__(self) -> None:
        for name in ("run", "request_id"):
            _text(name, getattr(self, name))
        _sha("result", self.result_sha256)
        if not isinstance(self.environment, DecodeSplitEnvironment):
            raise DecodeSplitSelectionError("atlas row needs an environment")
        _int("prompt tokens", self.prompt_tokens, 1)
        _int("output tokens", self.output_tokens, 1)
        lo, hi = _int("validated prompt min", self.validated_prompt_tokens_min, 1), _int("validated prompt max", self.validated_prompt_tokens_max, 1)
        if not lo <= self.prompt_tokens <= hi:
            raise DecodeSplitSelectionError("validated prompt range must contain the measured prompt")
        if _int("validated total max", self.validated_total_tokens_max, 1) != self.prompt_tokens + self.output_tokens:
            raise DecodeSplitSelectionError("validated total must equal the measured prompt plus output")
        if self.validated_total_tokens_max > self.environment.context_cells:
            raise DecodeSplitSelectionError("measured total exceeds the configured context")
        if self.memory_budget is not None:
            _text("memory budget", self.memory_budget)
        if type(self.phone_attached) is not bool or type(self.dormant_release) is not bool:
            raise DecodeSplitSelectionError("atlas flags must be booleans")
        if not 0 <= _int("split fraction", self.split_fraction_ppm) <= 1_000_000:
            raise DecodeSplitSelectionError("split fraction is out of range")
        _int("host columns", self.host_columns)
        if self.host_columns % self.environment.column_quantum:
            raise DecodeSplitSelectionError("host columns are not a multiple of the quantum")
        if _number("decode ms per token", self.decode_ms_per_token) <= 0:
            raise DecodeSplitSelectionError("decode ms per token must be positive")
        if self.prefill_s is not None:
            _number("prefill seconds", self.prefill_s)
        if self.decode_host_power_w is not None and _number("decode host power", self.decode_host_power_w) <= 0:
            raise DecodeSplitSelectionError("decode host power must be positive")
        _int("released bytes", self.released_bytes)
        if self.release_elapsed_us is not None:
            _int("release elapsed us", self.release_elapsed_us)
        _int("kv cpu bytes per token", self.kv_cpu_bytes_per_token, 1)
        _int("kv gpu bytes per token", self.kv_gpu_bytes_per_token)
        if self.split_fraction_ppm == 0 and self.released_bytes:
            raise DecodeSplitSelectionError("a baseline point cannot release bytes")
        if self.split_fraction_ppm and self.dormant_release and self.released_bytes <= 0:
            raise DecodeSplitSelectionError("a dormant split point must carry its released bytes")

    @property
    def split(self) -> bool:
        return self.split_fraction_ppm > 0

    def covers(self, prompt_tokens: int, output_tokens: int) -> bool:
        return (self.validated_prompt_tokens_min <= prompt_tokens <= self.validated_prompt_tokens_max
                and prompt_tokens + output_tokens <= self.validated_total_tokens_max)

    def energy_per_token_j(self, assumed_phone_power_w: float) -> float | None:
        if self.decode_host_power_w is None:
            return None
        phone = assumed_phone_power_w if self.split else 0.0
        return (self.decode_host_power_w + phone) * self.decode_ms_per_token / 1000.0

    @classmethod
    def from_json(cls, value: object) -> "DecodeSplitPoint":
        fields_ = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields_:
            raise DecodeSplitSelectionError("decode split atlas row has unexpected fields")
        row = dict(value)
        row["environment"] = DecodeSplitEnvironment.from_json(row["environment"])
        return cls(**row)


@dataclass(frozen=True)
class DecodeSplitAtlas:
    model_id: str
    rig: str
    rows: tuple[DecodeSplitPoint, ...]

    @classmethod
    def from_json(cls, value: object) -> "DecodeSplitAtlas":
        if not isinstance(value, Mapping) or value.get("schema") != ATLAS_SCHEMA:
            raise DecodeSplitSelectionError("decode split atlas schema is unknown")
        rows = value.get("rows")
        if not isinstance(rows, list) or not rows:
            raise DecodeSplitSelectionError("decode split atlas has no rows")
        return cls(_text("atlas model id", value.get("model_id")), _text("atlas rig", value.get("rig")),
                   tuple(DecodeSplitPoint.from_json(row) for row in rows))

    @classmethod
    def load(cls, path: Path) -> "DecodeSplitAtlas":
        return cls.from_json(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class DecodeSplitSelection:
    objective: str
    split_fraction_ppm: int
    host_columns: int
    phone_columns: int
    column_quantum: int
    dormant_release: bool
    expected_release_bytes: int
    decode_ms_per_token: float
    decode_host_power_w: float | None
    energy_per_token_j: float | None
    source_run: str
    source_request_id: str
    source_result_sha256: str
    validated_prompt_tokens: tuple[int, int]
    validated_total_tokens_max: int
    candidates_ppm: tuple[int, ...]
    reason: str

    def to_json(self) -> dict[str, object]:
        return {
            "objective": self.objective, "split_fraction_ppm": self.split_fraction_ppm,
            "host_columns": self.host_columns, "phone_columns": self.phone_columns, "column_quantum": self.column_quantum,
            "dormant_release": self.dormant_release, "expected_release_bytes": self.expected_release_bytes,
            "decode_ms_per_token": self.decode_ms_per_token, "decode_host_power_w": self.decode_host_power_w,
            "energy_per_token_j": self.energy_per_token_j,
            "source": {"run": self.source_run, "request_id": self.source_request_id, "result_sha256": self.source_result_sha256},
            "validated_prompt_tokens": list(self.validated_prompt_tokens), "validated_total_tokens_max": self.validated_total_tokens_max,
            "candidates_ppm": list(self.candidates_ppm), "reason": self.reason,
        }


def select_decode_split(
    atlas: DecodeSplitAtlas,
    *,
    environment: DecodeSplitEnvironment,
    prompt_tokens: int,
    output_tokens: int,
    feed_forward_length: int,
    objective: str,
    required_release_bytes: int = 0,
    max_decode_ms_per_token: float | None = None,
    assumed_phone_power_w: float = 4.5,
    phone_available: bool = True,
) -> DecodeSplitSelection:
    """Pick the measured split point that meets the objective for this exact environment and request
    shape; fail closed on anything unmeasured."""
    if objective not in OBJECTIVES:
        raise DecodeSplitSelectionError(f"objective must be one of {OBJECTIVES}")
    if not isinstance(environment, DecodeSplitEnvironment):
        raise DecodeSplitSelectionError("selection needs the request's environment")
    prompt = _int("prompt tokens", prompt_tokens, 1)
    output = _int("output tokens", output_tokens, 1)
    n_ff = _int("feed forward length", feed_forward_length, 1)
    required = _int("required release bytes", required_release_bytes)
    if max_decode_ms_per_token is not None and _number("latency bound", max_decode_ms_per_token) <= 0:
        raise DecodeSplitSelectionError("latency bound must be positive")
    phone_power = _number("assumed phone power", assumed_phone_power_w)
    if prompt + output > environment.context_cells:
        raise DecodeSplitSelectionError("request does not fit the configured context")
    same_environment = [row for row in atlas.rows if row.environment == environment and row.memory_budget is None]
    if not same_environment:
        raise DecodeSplitSelectionError("no uncapped atlas rows for this execution environment")
    rows = [row for row in same_environment if row.covers(prompt, output)]
    if not rows:
        ranges = sorted({(row.validated_prompt_tokens_min, row.validated_prompt_tokens_max, row.validated_total_tokens_max)
                         for row in same_environment})
        raise DecodeSplitSelectionError(
            f"request shape prompt={prompt} output={output} is outside every validated range {ranges}")
    for row in rows:
        if row.host_columns > n_ff or n_ff % row.environment.column_quantum:
            raise DecodeSplitSelectionError("atlas row does not fit the model's feed-forward length")
    candidates = list(rows)
    if not phone_available:
        candidates = [row for row in candidates if not row.split]
    candidates = [row for row in candidates if (not row.split) or row.dormant_release]
    if required:
        candidates = [row for row in candidates if row.released_bytes >= required]
        if not candidates:
            raise DecodeSplitSelectionError(f"no measured split releases {required} bytes for this request shape")
    if max_decode_ms_per_token is not None:
        candidates = [row for row in candidates if row.decode_ms_per_token <= max_decode_ms_per_token]
        if not candidates:
            raise DecodeSplitSelectionError("no measured split meets the decode latency bound")
    if not candidates:
        raise DecodeSplitSelectionError("no admissible split point")
    if objective == "latency":
        key = lambda row: (row.decode_ms_per_token, -row.released_bytes)  # noqa: E731
        reason = "fastest measured decode"
    elif objective == "energy":
        if any(row.energy_per_token_j(phone_power) is None for row in candidates):
            raise DecodeSplitSelectionError("energy objective needs decode host power for every candidate")
        key = lambda row: (row.energy_per_token_j(phone_power), -row.released_bytes)  # noqa: E731
        reason = f"lowest host + assumed phone ({phone_power:g} W) energy per generated token"
    else:
        key = lambda row: (-row.released_bytes, row.decode_ms_per_token)  # noqa: E731
        reason = "largest measured decode-phase release" + (" under the latency bound" if max_decode_ms_per_token else "")
    best = min(candidates, key=key)
    return DecodeSplitSelection(
        objective=objective, split_fraction_ppm=best.split_fraction_ppm, host_columns=best.host_columns if best.split else n_ff,
        phone_columns=(n_ff - best.host_columns) if best.split else 0, column_quantum=best.environment.column_quantum,
        dormant_release=best.split and best.dormant_release, expected_release_bytes=best.released_bytes if best.split else 0,
        decode_ms_per_token=best.decode_ms_per_token, decode_host_power_w=best.decode_host_power_w,
        energy_per_token_j=best.energy_per_token_j(phone_power), source_run=best.run, source_request_id=best.request_id,
        source_result_sha256=best.result_sha256,
        validated_prompt_tokens=(best.validated_prompt_tokens_min, best.validated_prompt_tokens_max),
        validated_total_tokens_max=best.validated_total_tokens_max,
        candidates_ppm=tuple(sorted({row.split_fraction_ppm for row in candidates})), reason=reason,
    )


def adaptive_policy_for_selection(
    selection: DecodeSplitSelection,
    *,
    route_id: str,
    executor_id: str,
    operator_plan_sha256: str,
    desktop_parent_route_id: str,
    desktop_placement_sha256: str,
    layer_indices: Sequence[int],
    layer_mask: int,
    resource_ids: Sequence[str],
) -> AdaptiveDecodePolicy:
    """The runtime-control policy for a selected split (the static phone plan stays full width)."""
    if not selection.split_fraction_ppm:
        raise DecodeSplitSelectionError("a baseline selection needs no runtime control")
    return AdaptiveDecodePolicy(
        route_id=route_id, executor_id=executor_id, operator_plan_sha256=operator_plan_sha256,
        desktop_parent_route_id=desktop_parent_route_id, desktop_placement_sha256=desktop_placement_sha256,
        layer_indices=tuple(layer_indices), layer_mask=layer_mask, columns=selection.phone_columns,
        split_fraction_ppm=selection.split_fraction_ppm, resource_ids=tuple(resource_ids),
    )


@dataclass(frozen=True)
class ShareBinding:
    """The physical allocation a share reservation belongs to and a proof must come from.

    ``host_columns`` is the smallest desktop prefix the server may keep, i.e. the largest share it may
    release; ``expected_release_bytes`` is that largest release, reserved as resident before the prompt.
    A proof may release any quantum-aligned suffix no larger than that (a runtime controller picks the
    fraction per request); the unreleased remainder stays charged as the shortfall."""
    endpoint: str
    artifact_sha256: str
    layer_mask: int
    host_columns: int
    expected_release_bytes: int
    column_quantum: int = 1

    def __post_init__(self) -> None:
        _text("endpoint", self.endpoint)
        _sha("artifact", self.artifact_sha256)
        _int("layer mask", self.layer_mask, 1)
        _int("host columns", self.host_columns)
        _int("expected release bytes", self.expected_release_bytes, 1)
        _int("column quantum", self.column_quantum, 1)
        if self.host_columns % self.column_quantum:
            raise DecodeSplitSelectionError("binding host columns are not a multiple of the quantum")

    def covers(self, proof: RuntimeHostShareReleaseProof) -> bool:
        """A proof may release a subset of the booked layers (a helper attached to fewer layers) and any
        quantum-aligned suffix no larger than the booked one; never layers or columns outside the booking."""
        return (proof.layer_mask != 0 and proof.layer_mask & ~self.layer_mask == 0
                and proof.host_columns >= self.host_columns and proof.host_columns % self.column_quantum == 0)


@dataclass
class _ShareState:
    binding: ShareBinding
    state: str
    consumed_generations: set[int] = field(default_factory=set)


class DecodeReleaseAccountant:
    """Phase-conditional accounting of the released host share on the runtime memory ledger.

    Owners are physical allocations (one per server endpoint). Proofs are bound to the owner's
    endpoint, layers and columns and consumed once per release generation; restoration is
    transactional (a failure leaves every reservation as it was)."""

    def __init__(self, ledger: RuntimeMemoryLedger, *, host_pool: str) -> None:
        if not isinstance(ledger, RuntimeMemoryLedger):
            raise DecodeSplitSelectionError("accountant needs a runtime memory ledger")
        self._ledger = ledger
        self._pool = _text("host pool", host_pool)
        self._shares: dict[str, _ShareState] = {}
        self._consumed_events: set[tuple[str, int]] = set()

    @staticmethod
    def share_owner_id(owner_id: str) -> str:
        return _text("owner", owner_id) + ":dormant-host-share"

    @staticmethod
    def growth_owner_id(owner_id: str) -> str:
        return _text("owner", owner_id) + ":decode-kv-growth"

    def state(self, owner_id: str) -> str | None:
        share = self._shares.get(owner_id)
        return None if share is None else share.state

    def binding(self, owner_id: str) -> ShareBinding | None:
        share = self._shares.get(owner_id)
        return None if share is None else share.binding

    def _share_demand(self, owner_id: str, amount: int, kind: str = SHARE_KIND) -> RuntimeMemoryDemand:
        return RuntimeMemoryDemand(demand_id=f"{owner_id}:{kind}", resource_id=self._pool, kind=kind,
                                   required_bytes=amount, resident_bytes=0, lifetime="request")

    def _share_reserved_bytes(self, owner_id: str) -> int:
        share_owner = self.share_owner_id(owner_id)
        return sum(row["reserved_bytes"] for row in self._ledger.snapshot()["reservations"] if row["owner_id"] == share_owner)

    def reserve_share(self, owner_id: str, binding: ShareBinding, snapshot, *, start_us: int = 0,
                      reserved_until_us: int | None = None):
        """Before the first prompt: the phone share is resident on the host until decode starts."""
        if not isinstance(binding, ShareBinding):
            raise DecodeSplitSelectionError("share reservation needs its physical binding")
        if owner_id in self._shares:
            raise DecodeSplitSelectionError("a share is already booked for this owner")
        for other in self._shares.values():
            if other.binding.endpoint == binding.endpoint:
                raise DecodeSplitSelectionError("this endpoint's share is already booked under another owner")
        reservations = self._ledger.reserve(self.share_owner_id(owner_id),
                                            (self._share_demand(owner_id, binding.expected_release_bytes),), snapshot,
                                            start_us=start_us, reserved_until_us=reserved_until_us)
        self._shares[owner_id] = _ShareState(binding=binding, state="prefill-resident")
        return reservations

    def enter_decode(self, owner_id: str, proof: RuntimeHostShareReleaseProof, snapshot, *, endpoint: str,
                     release_generation: int) -> int:
        """Credit the server's release proof once: it must come from the owner's endpoint, cover the reserved
        layers and columns, and carry a release generation not consumed before. The share reservation is
        dropped for the proven bytes; a shortfall against the reservation stays charged. Returns the credit."""
        share = self._shares.get(owner_id)
        if share is None:
            raise DecodeSplitSelectionError("no share is booked for this owner")
        generation = _int("release generation", release_generation, 1)
        event = (_text("proof endpoint", endpoint), generation)
        if event in self._consumed_events or generation in share.consumed_generations:
            raise DecodeSplitSelectionError("release event was already credited")
        if share.state not in ("prefill-resident", "decode-released"):
            raise DecodeSplitSelectionError("no resident share is reserved for this owner")
        if not isinstance(proof, RuntimeHostShareReleaseProof) or proof.phase != "decode":
            raise DecodeSplitSelectionError("decode credit needs a decode-phase release proof")
        binding = share.binding
        if endpoint != binding.endpoint:
            raise DecodeSplitSelectionError("release proof comes from another endpoint")
        if not binding.covers(proof):
            raise DecodeSplitSelectionError("release proof does not match the reserved layers and columns")
        expected = binding.expected_release_bytes
        credited = min(_int("proof released bytes", proof.released_bytes), expected)
        shortfall = expected - credited
        # a new generation while already released is a re-release (the controller changed the fraction or the
        # layer subset mid-request): the retained shortfall is re-sized to the new proof; growing it back needs
        # capacity, so a consumed room keeps the previous accounting and the event is reported as unaccounted
        checkpoint = self._ledger.checkpoint()
        try:
            self._ledger.release_owner(self.share_owner_id(owner_id))
            if shortfall:
                self._ledger.reserve(self.share_owner_id(owner_id), (self._share_demand(owner_id, shortfall, SHORTFALL_KIND),), snapshot)
        except RuntimeResourceError:
            self._ledger.restore(checkpoint)
            raise
        self._consumed_events.add(event)
        share.consumed_generations.add(generation)
        share.state = "decode-released"
        return credited

    def reserve_decode_growth(self, owner_id: str, growth_bytes: int, snapshot):
        """Charge decode-phase consumption (KV growth of a live request, another tenant) against live capacity."""
        amount = _int("growth bytes", growth_bytes, 1)
        return self._ledger.reserve(self.growth_owner_id(owner_id),
                                    (RuntimeMemoryDemand(demand_id=f"{owner_id}:{GROWTH_KIND}", resource_id=self._pool,
                                                         kind=GROWTH_KIND, required_bytes=amount, resident_bytes=0,
                                                         lifetime="request"),), snapshot)

    def release_growth(self, owner_id: str) -> None:
        self._ledger.release_owner(self.growth_owner_id(owner_id))

    def preview_prompt_admission(self, owner_id: str, snapshot) -> bool:
        """Can the released share be re-reserved now? False means the next prompt on this server must wait."""
        share = self._shares.get(owner_id)
        if share is None:
            raise DecodeSplitSelectionError("no share is known for this owner")
        if share.state == "prefill-resident":
            return True
        missing = share.binding.expected_release_bytes - self._share_reserved_bytes(owner_id)
        if missing <= 0:
            return True
        try:
            self._ledger.preview((self._share_demand(owner_id, missing),), snapshot)
        except RuntimeResourceError:
            return False
        return True

    def restore_before_prompt(self, owner_id: str, snapshot):
        """Re-reserve the share before the next prompt (the server populates the pages then). Transactional:
        on insufficient capacity every existing reservation, including a retained shortfall, is kept and
        RuntimeResourceError is raised so the caller holds the prompt."""
        share = self._shares.get(owner_id)
        if share is None:
            raise DecodeSplitSelectionError("no share is known for this owner")
        if share.state == "prefill-resident":
            return ()
        expected = share.binding.expected_release_bytes
        checkpoint = self._ledger.checkpoint()
        try:
            self._ledger.release_owner(self.share_owner_id(owner_id))
            reservations = self._ledger.reserve(self.share_owner_id(owner_id), (self._share_demand(owner_id, expected),), snapshot)
        except RuntimeResourceError:
            self._ledger.restore(checkpoint)
            share.state = "restore-blocked"
            raise
        share.state = "prefill-resident"
        return reservations

    def forget(self, owner_id: str) -> None:
        """The allocation is gone: drop every reservation the accountant made for it. A later server on the
        same endpoint is a new allocation whose release generations start again, so its consumed events go too."""
        self._ledger.release_owner(self.share_owner_id(owner_id))
        self._ledger.release_owner(self.growth_owner_id(owner_id))
        share = self._shares.pop(owner_id, None)
        if share is not None:
            self._consumed_events = {event for event in self._consumed_events if event[0] != share.binding.endpoint}

    def decode_phase_headroom_bytes(self, snapshot) -> int:
        """Live host bytes not reserved by anyone: the room decoding requests may grow into."""
        capacity = snapshot.capacities.get(self._pool)
        if capacity is None:
            raise DecodeSplitSelectionError("host pool is absent from the snapshot")
        reserved = self._ledger.snapshot()["by_resource_bytes"].get(self._pool, 0)
        return max(0, capacity.available_bytes - reserved)

    def kv_growth_tokens(self, snapshot, kv_bytes_per_token: int) -> int:
        return self.decode_phase_headroom_bytes(snapshot) // _int("kv bytes per token", kv_bytes_per_token, 1)

    def to_json(self) -> dict[str, object]:
        return {"host_pool": self._pool,
                "shares": {owner: {"state": share.state, "binding": {"endpoint": share.binding.endpoint,
                                   "layer_mask": share.binding.layer_mask, "host_columns": share.binding.host_columns,
                                   "expected_release_bytes": share.binding.expected_release_bytes},
                                   "consumed_generations": sorted(share.consumed_generations)}
                           for owner, share in sorted(self._shares.items())},
                "ledger": self._ledger.snapshot()}


__all__ = [
    "ATLAS_SCHEMA", "OBJECTIVES", "ENVIRONMENT_FIELDS", "DecodeSplitSelectionError", "DecodeSplitEnvironment",
    "DecodeSplitPoint", "DecodeSplitAtlas", "DecodeSplitSelection", "ShareBinding", "DecodeReleaseAccountant",
    "select_decode_split", "adaptive_policy_for_selection",
]
