"""Certified operator-split geometry and work accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


__all__ = [
    "OPERATOR_SPLIT_SCHEMA",
    "BalancedSplitEstimate",
    "ParallelSplitBalance",
    "ParallelSplitShapeMeasurement",
    "OperatorSplitError",
    "OperatorSplitInvocation",
    "OperatorSplitPolicy",
    "OperatorWorkSummary",
    "ShapeBucket",
    "balance_parallel_split",
    "parse_split_table",
    "select_split_columns",
    "split_table_text",
]


OPERATOR_SPLIT_SCHEMA = "research-scheduler-operator-split-v1"


class OperatorSplitError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise OperatorSplitError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise OperatorSplitError(f"{name} must be ASCII") from exc
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise OperatorSplitError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class ShapeBucket:
    max_tokens: int
    phone_columns: int

    def __post_init__(self) -> None:
        _integer("shape max_tokens", self.max_tokens, 1)
        _integer("shape phone_columns", self.phone_columns)


@dataclass(frozen=True)
class ParallelSplitShapeMeasurement:
    tokens: int
    calls: int
    total_columns: int
    phone_columns: int
    phone_rpc_us: int
    phone_compute_us: int
    host_us: int
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "tokens",
            "calls",
            "total_columns",
            "phone_columns",
            "phone_rpc_us",
            "phone_compute_us",
            "host_us",
        ):
            _integer(f"parallel split measurement {name}", getattr(self, name), 1)
        if self.phone_columns >= self.total_columns:
            raise OperatorSplitError(
                "parallel split phone columns must leave host work"
            )
        if self.phone_compute_us > self.phone_rpc_us:
            raise OperatorSplitError(
                "parallel split phone compute exceeds RPC time"
            )
        evidence = tuple(self.evidence_ids)
        if (
            not evidence
            or len(evidence) != len(set(evidence))
            or any(type(value) is not str or not value for value in evidence)
        ):
            raise OperatorSplitError(
                "parallel split measurement evidence must be non-empty and unique"
            )
        for value in evidence:
            _text("parallel split measurement evidence", value)
        object.__setattr__(self, "evidence_ids", evidence)


@dataclass(frozen=True)
class BalancedSplitEstimate:
    tokens: int
    calls: int
    baseline_columns: int
    selected_columns: int
    baseline_service_us: int
    selected_host_us: int
    selected_phone_us: int
    selected_service_us: int

    def __post_init__(self) -> None:
        for name in (
            "tokens",
            "calls",
            "baseline_columns",
            "selected_columns",
            "baseline_service_us",
            "selected_host_us",
            "selected_phone_us",
            "selected_service_us",
        ):
            _integer(f"balanced split estimate {name}", getattr(self, name), 1)

    @property
    def saving_ppm(self) -> int:
        return max(
            0,
            (self.baseline_service_us - self.selected_service_us)
            * 1_000_000
            // self.baseline_service_us,
        )

    def to_json(self) -> dict[str, int]:
        return {
            "baseline_columns": self.baseline_columns,
            "baseline_service_us": self.baseline_service_us,
            "calls": self.calls,
            "saving_ppm": self.saving_ppm,
            "selected_columns": self.selected_columns,
            "selected_host_us": self.selected_host_us,
            "selected_phone_us": self.selected_phone_us,
            "selected_service_us": self.selected_service_us,
            "tokens": self.tokens,
        }


@dataclass(frozen=True)
class ParallelSplitBalance:
    buckets: tuple[ShapeBucket, ...]
    estimates: tuple[BalancedSplitEstimate, ...]
    baseline_weighted_service_us: int
    selected_weighted_service_us: int
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        buckets = tuple(self.buckets)
        estimates = tuple(self.estimates)
        evidence = tuple(self.evidence_ids)
        if not buckets or not estimates or not evidence:
            raise OperatorSplitError("parallel split balance must not be empty")
        _integer(
            "parallel split baseline weighted service",
            self.baseline_weighted_service_us,
            1,
        )
        _integer(
            "parallel split selected weighted service",
            self.selected_weighted_service_us,
            1,
        )
        if self.selected_weighted_service_us > self.baseline_weighted_service_us:
            raise OperatorSplitError("parallel split balance regresses service")
        object.__setattr__(self, "buckets", buckets)
        object.__setattr__(self, "estimates", estimates)
        object.__setattr__(self, "evidence_ids", evidence)

    @property
    def table(self) -> str:
        return split_table_text(self.buckets)

    @property
    def predicted_saving_ppm(self) -> int:
        return (
            (self.baseline_weighted_service_us - self.selected_weighted_service_us)
            * 1_000_000
            // self.baseline_weighted_service_us
        )

    def to_json(self) -> dict[str, object]:
        return {
            "baseline_weighted_service_us": self.baseline_weighted_service_us,
            "estimates": [row.to_json() for row in self.estimates],
            "evidence_ids": list(self.evidence_ids),
            "predicted_saving_ppm": self.predicted_saving_ppm,
            "selected_weighted_service_us": self.selected_weighted_service_us,
            "table": self.table,
        }


def _ceil_ratio(value: int, numerator: int, denominator: int) -> int:
    return (value * numerator + denominator - 1) // denominator


def balance_parallel_split(
    measurements: Sequence[ParallelSplitShapeMeasurement],
    *,
    candidate_columns: Sequence[int],
    alternate_columns: Sequence[int] = (),
    physical_max_tokens: int,
    policy_max_tokens: int,
    resident_phone_columns: int,
    column_quantum: int,
    minimum_saving_ppm: int = 0,
) -> ParallelSplitBalance:
    """Fit measured shapes while retaining the qualified width elsewhere."""

    rows = tuple(measurements)
    candidates = tuple(sorted(set(candidate_columns)))
    alternates = tuple(sorted(set(alternate_columns)))
    for name, value in (
        ("physical_max_tokens", physical_max_tokens),
        ("policy_max_tokens", policy_max_tokens),
        ("resident_phone_columns", resident_phone_columns),
        ("column_quantum", column_quantum),
    ):
        _integer(f"parallel split {name}", value, 1)
    _integer("parallel split minimum_saving_ppm", minimum_saving_ppm)
    if minimum_saving_ppm > 1_000_000:
        raise OperatorSplitError("parallel split minimum saving exceeds one")
    if physical_max_tokens >= policy_max_tokens:
        raise OperatorSplitError(
            "parallel split policy maximum must exceed physical maximum"
        )
    if (
        not rows
        or len({row.tokens for row in rows}) != len(rows)
        or any(not isinstance(row, ParallelSplitShapeMeasurement) for row in rows)
    ):
        raise OperatorSplitError(
            "parallel split measurements must be typed and shape-unique"
        )
    if not candidates or resident_phone_columns not in candidates:
        raise OperatorSplitError(
            "parallel split candidates must include resident width"
        )
    if (
        len(alternates) != len(tuple(alternate_columns))
        or any(value <= 0 or value >= resident_phone_columns for value in alternates)
    ):
        raise OperatorSplitError("parallel split alternate width is invalid")
    if any(
        value <= 0
        or value > resident_phone_columns
        or (
            value != resident_phone_columns
            and value not in alternates
            and value % column_quantum != 0
        )
        for value in candidates
    ):
        raise OperatorSplitError("parallel split candidate width is invalid")
    if any(
        row.tokens > physical_max_tokens
        or row.phone_columns != resident_phone_columns
        for row in rows
    ):
        raise OperatorSplitError("parallel split measurement identity differs")

    selected_by_tokens = {
        tokens: resident_phone_columns
        for tokens in range(1, physical_max_tokens + 1)
    }
    estimates: list[BalancedSplitEstimate] = []
    baseline_weighted = 0
    selected_weighted = 0
    for row in sorted(rows, key=lambda value: value.tokens):
        host_columns = row.total_columns - row.phone_columns
        phone_fixed_us = row.phone_rpc_us - row.phone_compute_us
        baseline_service_us = max(row.phone_rpc_us, row.host_us)
        choices = []
        for columns in candidates:
            phone_us = phone_fixed_us + _ceil_ratio(
                row.phone_compute_us, columns, row.phone_columns
            )
            host_us = _ceil_ratio(
                row.host_us,
                row.total_columns - columns,
                host_columns,
            )
            service_us = max(phone_us, host_us)
            choices.append((service_us, -columns, columns, host_us, phone_us))
        _, _, columns, host_us, phone_us = min(choices)
        saving_ppm = max(
            0,
            (baseline_service_us - max(host_us, phone_us))
            * 1_000_000
            // baseline_service_us,
        )
        if saving_ppm < minimum_saving_ppm:
            columns = row.phone_columns
            host_us = row.host_us
            phone_us = row.phone_rpc_us
        estimate = BalancedSplitEstimate(
            tokens=row.tokens,
            calls=row.calls,
            baseline_columns=row.phone_columns,
            selected_columns=columns,
            baseline_service_us=baseline_service_us,
            selected_host_us=host_us,
            selected_phone_us=phone_us,
            selected_service_us=max(host_us, phone_us),
        )
        estimates.append(estimate)
        selected_by_tokens[row.tokens] = columns
        baseline_weighted += row.calls * baseline_service_us
        selected_weighted += row.calls * estimate.selected_service_us

    buckets: list[ShapeBucket] = []
    current_columns = selected_by_tokens[1]
    for tokens in range(2, physical_max_tokens + 1):
        columns = selected_by_tokens[tokens]
        if columns != current_columns:
            buckets.append(ShapeBucket(tokens - 1, current_columns))
            current_columns = columns
    buckets.append(ShapeBucket(physical_max_tokens, current_columns))
    buckets.append(ShapeBucket(policy_max_tokens, 0))
    evidence = tuple(sorted({
        evidence_id
        for row in rows
        for evidence_id in row.evidence_ids
    }))
    return ParallelSplitBalance(
        buckets=tuple(buckets),
        estimates=tuple(estimates),
        baseline_weighted_service_us=baseline_weighted,
        selected_weighted_service_us=selected_weighted,
        evidence_ids=evidence,
    )


def parse_split_table(text: str) -> tuple[ShapeBucket, ...]:
    """Parse an ordered inclusive token-limit table."""

    _text("split table", text)
    result: list[ShapeBucket] = []
    for raw in text.split(","):
        limit_text, separator, columns_text = raw.partition(":")
        if separator != ":":
            raise OperatorSplitError("split table syntax")
        try:
            bucket = ShapeBucket(int(limit_text), int(columns_text))
        except ValueError as exc:
            raise OperatorSplitError("split table integer") from exc
        if result and bucket.max_tokens <= result[-1].max_tokens:
            raise OperatorSplitError("split table limits must increase")
        result.append(bucket)
    if not result:
        raise OperatorSplitError("split table must not be empty")
    return tuple(result)


def split_table_text(buckets: Sequence[ShapeBucket]) -> str:
    rows = tuple(buckets)
    if not rows:
        raise OperatorSplitError("split table must not be empty")
    return ",".join(
        f"{bucket.max_tokens}:{bucket.phone_columns}" for bucket in rows
    )


def select_split_columns(
    buckets: Sequence[ShapeBucket], tokens: int
) -> int:
    rows = tuple(buckets)
    _integer("split tokens", tokens, 1)
    if not rows:
        raise OperatorSplitError("split table must not be empty")
    for bucket in rows:
        if tokens <= bucket.max_tokens:
            return bucket.phone_columns
    raise OperatorSplitError("split policy does not cover tokens")


@dataclass(frozen=True)
class OperatorSplitInvocation:
    layer: int
    tokens: int
    phone_columns: int
    host_columns: int
    activation_bytes: int
    phone_macs: int

    def __post_init__(self) -> None:
        _integer("invocation layer", self.layer)
        _integer("invocation tokens", self.tokens, 1)
        _integer("invocation phone_columns", self.phone_columns)
        _integer("invocation host_columns", self.host_columns)
        _integer("invocation activation_bytes", self.activation_bytes, 1)
        _integer("invocation phone_macs", self.phone_macs)


@dataclass(frozen=True)
class OperatorWorkSummary:
    calls: int
    token_rows: int
    upload_bytes: int
    download_bytes: int
    phone_macs: int
    weighted_phone_columns: int

    def __post_init__(self) -> None:
        for name in (
            "calls",
            "token_rows",
            "upload_bytes",
            "download_bytes",
            "phone_macs",
            "weighted_phone_columns",
        ):
            _integer(f"operator work {name}", getattr(self, name))


@dataclass(frozen=True)
class OperatorSplitPolicy:
    policy_id: str
    operator_family: str
    layer_ids: tuple[int, ...]
    n_embd: int
    eligible_columns: int
    max_tokens: int
    column_quantum: int
    alternate_columns: tuple[int, ...]
    io_type: str
    weight_layout: str
    buckets: tuple[ShapeBucket, ...]
    activation: str = "geglu"

    def __post_init__(self) -> None:
        _text("split policy id", self.policy_id)
        _text("split operator family", self.operator_family)
        _integer("split n_embd", self.n_embd, 1)
        _integer("split eligible_columns", self.eligible_columns, 1)
        _integer("split max_tokens", self.max_tokens, 1)
        _integer("split column_quantum", self.column_quantum, 1)
        if self.io_type not in {"f16", "f32"}:
            raise OperatorSplitError("split io_type must be f16 or f32")
        if self.activation not in {"geglu", "swiglu"}:
            raise OperatorSplitError("split activation must be geglu or swiglu")
        _text("split weight_layout", self.weight_layout)

        layers = tuple(self.layer_ids)
        if (
            not layers
            or any(type(layer) is not int or layer < 0 for layer in layers)
            or tuple(sorted(layers)) != layers
            or len(set(layers)) != len(layers)
        ):
            raise OperatorSplitError(
                "split layer_ids must be sorted, unique, and non-empty"
            )
        object.__setattr__(self, "layer_ids", layers)

        alternates = tuple(self.alternate_columns)
        if (
            len(alternates) != len(set(alternates))
            or any(
                type(columns) is not int
                or columns <= 0
                or columns >= self.eligible_columns
                for columns in alternates
            )
        ):
            raise OperatorSplitError("split alternate_columns are invalid")
        object.__setattr__(self, "alternate_columns", tuple(sorted(alternates)))

        buckets = tuple(self.buckets)
        if not buckets:
            raise OperatorSplitError("split buckets must not be empty")
        if tuple(sorted(bucket.max_tokens for bucket in buckets)) != tuple(
            bucket.max_tokens for bucket in buckets
        ) or len({bucket.max_tokens for bucket in buckets}) != len(buckets):
            raise OperatorSplitError("split bucket limits must increase")
        if buckets[-1].max_tokens < self.max_tokens:
            raise OperatorSplitError("split buckets do not cover max_tokens")
        for bucket in buckets:
            columns = bucket.phone_columns
            if columns > self.eligible_columns:
                raise OperatorSplitError("split columns exceed eligible columns")
            if (
                columns not in {0, self.eligible_columns, *alternates}
                and columns % self.column_quantum != 0
            ):
                raise OperatorSplitError("split columns violate column quantum")
        object.__setattr__(self, "buckets", buckets)

    @classmethod
    def from_table(
        cls,
        *,
        policy_id: str,
        operator_family: str,
        layer_ids: Sequence[int],
        n_embd: int,
        eligible_columns: int,
        max_tokens: int,
        column_quantum: int,
        alternate_columns: Sequence[int],
        io_type: str,
        weight_layout: str,
        table: str,
        activation: str = "geglu",
    ) -> "OperatorSplitPolicy":
        return cls(
            policy_id=policy_id,
            operator_family=operator_family,
            layer_ids=tuple(layer_ids),
            n_embd=n_embd,
            eligible_columns=eligible_columns,
            max_tokens=max_tokens,
            column_quantum=column_quantum,
            alternate_columns=tuple(alternate_columns),
            io_type=io_type,
            weight_layout=weight_layout,
            buckets=parse_split_table(table),
            activation=activation,
        )

    @classmethod
    def from_json(cls, value: object) -> "OperatorSplitPolicy":
        if type(value) is not dict:
            raise OperatorSplitError("operator split must be an object")
        row: Mapping[str, Any] = value
        if row.get("schema") != OPERATOR_SPLIT_SCHEMA:
            raise OperatorSplitError("operator split schema mismatch")
        raw_layers = row.get("layer_ids")
        raw_alternates = row.get("alternate_columns", [])
        if type(raw_layers) is not list or type(raw_alternates) is not list:
            raise OperatorSplitError("operator split list field")
        return cls.from_table(
            policy_id=row.get("policy_id"),
            operator_family=row.get("operator_family"),
            layer_ids=raw_layers,
            n_embd=row.get("n_embd"),
            eligible_columns=row.get("eligible_columns"),
            max_tokens=row.get("max_tokens"),
            column_quantum=row.get("column_quantum"),
            alternate_columns=raw_alternates,
            io_type=row.get("io_type"),
            weight_layout=row.get("weight_layout"),
            table=row.get("table"),
            activation=row.get("activation", "geglu"),
        )

    @property
    def element_bytes(self) -> int:
        return 2 if self.io_type == "f16" else 4

    @property
    def table(self) -> str:
        return split_table_text(self.buckets)

    def columns_for(self, tokens: int) -> int:
        _integer("split tokens", tokens, 1)
        if tokens > self.max_tokens:
            raise OperatorSplitError("split tokens exceed max_tokens")
        return select_split_columns(self.buckets, tokens)

    def invocation(self, layer: int, tokens: int) -> OperatorSplitInvocation:
        _integer("split layer", layer)
        if layer not in self.layer_ids:
            raise OperatorSplitError("split layer is not eligible")
        columns = self.columns_for(tokens)
        activation_bytes = self.n_embd * tokens * self.element_bytes
        return OperatorSplitInvocation(
            layer=layer,
            tokens=tokens,
            phone_columns=columns,
            host_columns=self.eligible_columns - columns,
            activation_bytes=activation_bytes,
            phone_macs=3 * self.n_embd * columns * tokens,
        )

    def summarize(
        self, shape_calls: Sequence[tuple[int, int]]
    ) -> OperatorWorkSummary:
        calls = 0
        token_rows = 0
        weighted_columns = 0
        for tokens, count in shape_calls:
            _integer("shape tokens", tokens, 1)
            _integer("shape calls", count)
            columns = self.columns_for(tokens)
            calls += count
            token_rows += count * tokens
            weighted_columns += count * tokens * columns
        transfer_bytes = token_rows * self.n_embd * self.element_bytes
        return OperatorWorkSummary(
            calls=calls,
            token_rows=token_rows,
            upload_bytes=transfer_bytes,
            download_bytes=transfer_bytes,
            phone_macs=3 * self.n_embd * weighted_columns,
            weighted_phone_columns=weighted_columns,
        )

    def to_json(self) -> dict[str, object]:
        value: dict[str, object] = {
            "alternate_columns": list(self.alternate_columns),
            "column_quantum": self.column_quantum,
            "eligible_columns": self.eligible_columns,
            "io_type": self.io_type,
            "layer_ids": list(self.layer_ids),
            "max_tokens": self.max_tokens,
            "n_embd": self.n_embd,
            "operator_family": self.operator_family,
            "policy_id": self.policy_id,
            "schema": OPERATOR_SPLIT_SCHEMA,
            "table": self.table,
            "weight_layout": self.weight_layout,
        }
        if self.activation != "geglu":
            value["activation"] = self.activation
        return value
