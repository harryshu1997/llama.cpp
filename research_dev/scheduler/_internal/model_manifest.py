"""Device-independent GGUF model manifests and request work estimates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
from math import prod
from pathlib import Path
import re
import sys
from types import MappingProxyType
from typing import Mapping, Sequence

from .gguf_metadata import GGUFMetadataError, read_gguf_metadata


MODEL_MANIFEST_SCHEMA = "research-scheduler-model-manifest-v1"


class ModelManifestError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise ModelManifestError(f"{name} must be non-empty ASCII text")
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ModelManifestError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class ModelTensorManifest:
    tensor_id: str
    shape: tuple[int, ...]
    quantization: str
    quantization_block_size: int
    quantization_type_size: int
    nbytes: int
    role: str
    layer_index: int | None

    def __post_init__(self) -> None:
        _text("tensor id", self.tensor_id)
        shape = tuple(self.shape)
        if not shape or any(type(value) is not int or value <= 0 for value in shape):
            raise ModelManifestError("tensor shape must contain positive integers")
        _text("tensor quantization", self.quantization)
        _integer(
            "tensor quantization block size",
            self.quantization_block_size,
            1,
        )
        _integer(
            "tensor quantization type size",
            self.quantization_type_size,
            1,
        )
        _integer("tensor bytes", self.nbytes, 1)
        _text("tensor role", self.role)
        if self.layer_index is not None:
            _integer("tensor layer index", self.layer_index)
        object.__setattr__(self, "shape", shape)

    @property
    def elements(self) -> int:
        return prod(self.shape)

    def to_json(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "nbytes": self.nbytes,
            "quantization": self.quantization,
            "quantization_block_size": self.quantization_block_size,
            "quantization_type_size": self.quantization_type_size,
            "role": self.role,
            "shape": list(self.shape),
            "tensor_id": self.tensor_id,
        }

    @classmethod
    def from_json(cls, value: object) -> "ModelTensorManifest":
        if type(value) is not dict:
            raise ModelManifestError("tensor manifest must be an object")
        shape = value.get("shape")
        if type(shape) is not list:
            raise ModelManifestError("tensor manifest shape must be a list")
        return cls(
            tensor_id=value.get("tensor_id"),
            shape=tuple(shape),
            quantization=value.get("quantization"),
            quantization_block_size=value.get("quantization_block_size"),
            quantization_type_size=value.get("quantization_type_size"),
            nbytes=value.get("nbytes"),
            role=value.get("role"),
            layer_index=value.get("layer_index"),
        )


@dataclass(frozen=True)
class ModelOperatorManifest:
    operator_id: str
    layer_id: str
    kind: str
    dependencies: tuple[str, ...]
    tensor_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("operator_id", "layer_id", "kind"):
            _text(f"operator {name}", getattr(self, name))
        dependencies = tuple(self.dependencies)
        tensors = tuple(self.tensor_ids)
        if len(dependencies) != len(set(dependencies)):
            raise ModelManifestError("operator dependencies must be unique")
        if len(tensors) != len(set(tensors)):
            raise ModelManifestError("operator tensor ids must be unique")
        for value in dependencies + tensors:
            _text("operator reference", value)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "tensor_ids", tensors)

    def to_json(self) -> dict[str, object]:
        return {
            "dependencies": list(self.dependencies),
            "kind": self.kind,
            "layer_id": self.layer_id,
            "operator_id": self.operator_id,
            "tensor_ids": list(self.tensor_ids),
        }

    @classmethod
    def from_json(cls, value: object) -> "ModelOperatorManifest":
        if type(value) is not dict:
            raise ModelManifestError("operator manifest must be an object")
        dependencies = value.get("dependencies")
        tensor_ids = value.get("tensor_ids")
        if type(dependencies) is not list or type(tensor_ids) is not list:
            raise ModelManifestError(
                "operator manifest references must be lists"
            )
        return cls(
            operator_id=value.get("operator_id"),
            layer_id=value.get("layer_id"),
            kind=value.get("kind"),
            dependencies=tuple(dependencies),
            tensor_ids=tuple(tensor_ids),
        )


@dataclass(frozen=True)
class OperatorWorkEstimate:
    operator_id: str
    kind: str
    compute_ops: int
    memory_bytes: int
    weight_bytes: int
    activation_bytes: int
    output_bytes: int
    workspace_bytes: int
    kv_cache_bytes: int

    def __post_init__(self) -> None:
        _text("operator work id", self.operator_id)
        _text("operator work kind", self.kind)
        for name in (
            "compute_ops",
            "memory_bytes",
            "weight_bytes",
            "activation_bytes",
            "output_bytes",
            "workspace_bytes",
            "kv_cache_bytes",
        ):
            _integer(f"operator work {name}", getattr(self, name))

    def to_json(self) -> dict[str, int | str]:
        return {
            "activation_bytes": self.activation_bytes,
            "compute_ops": self.compute_ops,
            "kind": self.kind,
            "kv_cache_bytes": self.kv_cache_bytes,
            "memory_bytes": self.memory_bytes,
            "operator_id": self.operator_id,
            "output_bytes": self.output_bytes,
            "weight_bytes": self.weight_bytes,
            "workspace_bytes": self.workspace_bytes,
        }


@dataclass(frozen=True)
class ModelPhaseWork:
    phase: str
    tokens: int
    invocations: int
    operators: tuple[OperatorWorkEstimate, ...]

    def __post_init__(self) -> None:
        if self.phase not in {"decode", "prefill"}:
            raise ModelManifestError("request work phase is invalid")
        _integer("request phase tokens", self.tokens, 1)
        _integer("request phase invocations", self.invocations, 1)
        rows = tuple(self.operators)
        if (
            not rows
            or any(not isinstance(row, OperatorWorkEstimate) for row in rows)
            or len({row.operator_id for row in rows}) != len(rows)
        ):
            raise ModelManifestError("request phase operators are invalid")
        object.__setattr__(self, "operators", rows)

    @property
    def by_operator_id(self) -> Mapping[str, OperatorWorkEstimate]:
        return MappingProxyType({row.operator_id: row for row in self.operators})


@dataclass(frozen=True)
class ModelRequestWork:
    input_tokens: int
    output_tokens: int
    operators: tuple[OperatorWorkEstimate, ...]
    phases: tuple[ModelPhaseWork, ...] = ()

    def __post_init__(self) -> None:
        _integer("request work input tokens", self.input_tokens, 1)
        _integer("request work output tokens", self.output_tokens, 1)
        rows = tuple(self.operators)
        if not rows or any(not isinstance(row, OperatorWorkEstimate) for row in rows):
            raise ModelManifestError("request work operators are invalid")
        if len({row.operator_id for row in rows}) != len(rows):
            raise ModelManifestError("request work operator ids are duplicated")
        phases = tuple(self.phases)
        if phases and (
            tuple(row.phase for row in phases) != ("prefill", "decode")
            or any(
                tuple(item.operator_id for item in phase.operators)
                != tuple(item.operator_id for item in rows)
                for phase in phases
            )
        ):
            raise ModelManifestError("request work phases are invalid")
        object.__setattr__(self, "operators", rows)
        object.__setattr__(self, "phases", phases)

    @property
    def by_operator_id(self) -> Mapping[str, OperatorWorkEstimate]:
        return MappingProxyType({row.operator_id: row for row in self.operators})

    def to_json(self) -> dict[str, object]:
        return {
            "input_tokens": self.input_tokens,
            "operators": [row.to_json() for row in self.operators],
            "output_tokens": self.output_tokens,
        }


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    artifact_sha256: str
    artifact_bytes: int
    architecture: str
    context_length: int
    sliding_window: int
    embedding_length: int
    feed_forward_length: int
    block_count: int
    head_count: int
    head_count_kv: int
    key_length: int
    value_length: int
    tensors: tuple[ModelTensorManifest, ...]
    operators: tuple[ModelOperatorManifest, ...]
    head_count_kv_by_layer: tuple[int, ...] = ()
    sliding_window_pattern: tuple[bool, ...] = ()
    key_length_swa: int | None = None
    value_length_swa: int | None = None

    def __post_init__(self) -> None:
        _text("model id", self.model_id)
        digest = _text("model artifact hash", self.artifact_sha256)
        if (
            not digest.startswith("sha256:")
            or len(digest) != 71
            or any(value not in "0123456789abcdef" for value in digest[7:])
        ):
            raise ModelManifestError("model artifact hash must be SHA-256")
        _integer("model artifact bytes", self.artifact_bytes, 1)
        _text("model architecture", self.architecture)
        for name in (
            "context_length",
            "sliding_window",
            "embedding_length",
            "feed_forward_length",
            "block_count",
            "head_count",
            "head_count_kv",
            "key_length",
            "value_length",
        ):
            _integer(f"model {name}", getattr(self, name), 1)
        tensors = tuple(self.tensors)
        operators = tuple(self.operators)
        head_count_kv_by_layer = tuple(self.head_count_kv_by_layer)
        sliding_window_pattern = tuple(self.sliding_window_pattern)
        if head_count_kv_by_layer and (
            len(head_count_kv_by_layer) != self.block_count
            or any(
                type(value) is not int or value <= 0
                for value in head_count_kv_by_layer
            )
        ):
            raise ModelManifestError(
                "model per-layer KV head counts are invalid"
            )
        if sliding_window_pattern and (
            len(sliding_window_pattern) != self.block_count
            or any(type(value) is not bool for value in sliding_window_pattern)
        ):
            raise ModelManifestError(
                "model sliding-window pattern is invalid"
            )
        for name in ("key_length_swa", "value_length_swa"):
            value = getattr(self, name)
            if value is not None:
                _integer(f"model {name}", value, 1)
        if not tensors or not operators:
            raise ModelManifestError("model manifest cannot be empty")
        tensor_ids = {row.tensor_id for row in tensors}
        operator_ids = {row.operator_id for row in operators}
        if len(tensor_ids) != len(tensors) or len(operator_ids) != len(operators):
            raise ModelManifestError("model manifest ids must be unique")
        seen: set[str] = set()
        for operator in operators:
            if set(operator.dependencies) - seen:
                raise ModelManifestError("model operator DAG is not topological")
            if set(operator.tensor_ids) - tensor_ids:
                raise ModelManifestError("model operator references an unknown tensor")
            seen.add(operator.operator_id)
        object.__setattr__(self, "tensors", tensors)
        object.__setattr__(self, "operators", operators)
        object.__setattr__(
            self, "head_count_kv_by_layer", head_count_kv_by_layer
        )
        object.__setattr__(
            self, "sliding_window_pattern", sliding_window_pattern
        )

    @property
    def tensor_bytes(self) -> int:
        return sum(row.nbytes for row in self.tensors)

    @cached_property
    def tensor_by_id(self) -> Mapping[str, ModelTensorManifest]:
        return MappingProxyType({row.tensor_id: row for row in self.tensors})

    @property
    def operator_by_id(self) -> Mapping[str, ModelOperatorManifest]:
        return MappingProxyType({row.operator_id: row for row in self.operators})

    @staticmethod
    def _operator_layer_index(operator: ModelOperatorManifest) -> int | None:
        prefix = "layer:"
        if not operator.layer_id.startswith(prefix):
            return None
        value = operator.layer_id[len(prefix):]
        return int(value) if value.isdigit() else None

    def _layer_is_sliding(self, layer_index: int | None) -> bool:
        if layer_index is None:
            return self.sliding_window < self.context_length
        if self.sliding_window_pattern:
            return self.sliding_window_pattern[layer_index]
        return self.sliding_window < self.context_length

    def _layer_head_count_kv(self, layer_index: int | None) -> int:
        if layer_index is None or not self.head_count_kv_by_layer:
            return self.head_count_kv
        return self.head_count_kv_by_layer[layer_index]

    def _layer_kv_lengths(self, layer_index: int | None) -> tuple[int, int]:
        if self._layer_is_sliding(layer_index):
            return (
                self.key_length if self.key_length_swa is None
                else self.key_length_swa,
                self.value_length if self.value_length_swa is None
                else self.value_length_swa,
            )
        return self.key_length, self.value_length

    @staticmethod
    def _dense_flops(
        tensors: Sequence[ModelTensorManifest], tokens: int
    ) -> int:
        return sum(
            2 * tensor.elements * tokens
            for tensor in tensors
            if len(tensor.shape) >= 2
        )

    @staticmethod
    def _capped_context_sum(
        initial_context: int, token_count: int, window: int
    ) -> int:
        uncapped = min(token_count, max(0, window - initial_context))
        return (
            uncapped * (2 * initial_context + uncapped - 1) // 2
            + (token_count - uncapped) * window
        )

    def _attention_flops(
        self,
        input_tokens: int,
        output_tokens: int,
        layer_index: int | None = None,
    ) -> int:
        window = (
            self.sliding_window
            if self._layer_is_sliding(layer_index)
            else self.context_length
        )
        key_length, _ = self._layer_kv_lengths(layer_index)
        capped = min(input_tokens, window)
        prefill_context_sum = capped * (capped + 1) // 2
        if input_tokens > window:
            prefill_context_sum += (input_tokens - window) * window
        decode_context_sum = self._capped_context_sum(
            input_tokens, output_tokens, window
        )
        return (
            4
            * self.head_count
            * key_length
            * (prefill_context_sum + decode_context_sum)
        )

    def _kv_bytes(
        self,
        input_tokens: int,
        output_tokens: int,
        layer_index: int | None = None,
    ) -> int:
        key_length, value_length = self._layer_kv_lengths(layer_index)
        return (
            (input_tokens + output_tokens)
            * self._layer_head_count_kv(layer_index)
            * (key_length + value_length)
            * 2
        )

    def preallocated_kv_cache_bytes(
        self,
        operator_id: str,
        *,
        context_size: int,
        parallel: int,
        sliding_window_padding_tokens: int,
    ) -> int:
        _integer("preallocated context size", context_size, 1)
        _integer("preallocated parallelism", parallel, 1)
        _integer(
            "preallocated sliding-window padding",
            sliding_window_padding_tokens,
        )
        operator = self.operator_by_id.get(operator_id)
        if operator is None or operator.kind != "kv_cache":
            raise ModelManifestError(
                "preallocated KV operator is invalid"
            )
        layer_index = self._operator_layer_index(operator)
        cells = context_size
        if self._layer_is_sliding(layer_index):
            cells = min(
                context_size,
                self.sliding_window * parallel
                + sliding_window_padding_tokens,
            )
        key_length, value_length = self._layer_kv_lengths(layer_index)
        return (
            cells
            * self._layer_head_count_kv(layer_index)
            * (key_length + value_length)
            * 2
        )

    def _phase_work(
        self,
        input_tokens: int,
        output_tokens: int,
        phase: str,
        tensor_by_id: Mapping[str, ModelTensorManifest] | None = None,
    ) -> ModelPhaseWork:
        if phase == "prefill":
            tokens = input_tokens
            invocations = 1
        elif phase == "decode":
            tokens = output_tokens
            invocations = output_tokens
        else:
            raise ModelManifestError("request work phase is invalid")
        activation_bytes = tokens * self.embedding_length * 2
        if tensor_by_id is None:
            tensor_by_id = self.tensor_by_id
        attention_context_by_window: dict[int, int] = {}
        rows = []
        for operator in self.operators:
            layer_index = self._operator_layer_index(operator)
            tensors = tuple(
                tensor_by_id[value] for value in operator.tensor_ids
            )
            weight_bytes = sum(row.nbytes for row in tensors)
            weight_traffic = weight_bytes * invocations
            compute_ops = self._dense_flops(tensors, tokens)
            kv_cache_bytes = 0
            memory_bytes = weight_traffic + 2 * activation_bytes
            if operator.kind == "embedding":
                compute_ops += tokens * self.embedding_length
            elif operator.kind == "attention":
                window = (
                    self.sliding_window
                    if self._layer_is_sliding(layer_index)
                    else self.context_length
                )
                attention_context_sum = attention_context_by_window.get(
                    window
                )
                if attention_context_sum is None:
                    if phase == "prefill":
                        attention_context_sum = (
                            min(input_tokens, window)
                            * (min(input_tokens, window) + 1)
                            // 2
                        )
                        if input_tokens > window:
                            attention_context_sum += (
                                input_tokens - window
                            ) * window
                    else:
                        attention_context_sum = self._capped_context_sum(
                            input_tokens, output_tokens, window
                        )
                    attention_context_by_window[window] = (
                        attention_context_sum
                    )
                key_length, value_length = self._layer_kv_lengths(
                    layer_index
                )
                compute_ops += (
                    4
                    * self.head_count
                    * key_length
                    * attention_context_sum
                )
                memory_bytes += (
                    attention_context_sum
                    * self._layer_head_count_kv(layer_index)
                    * (key_length + value_length)
                    * 2
                )
            elif operator.kind == "kv_cache":
                kv_cache_bytes = (
                    tokens
                    * self._layer_head_count_kv(layer_index)
                    * sum(self._layer_kv_lengths(layer_index))
                    * 2
                )
                memory_bytes += kv_cache_bytes
            rows.append(OperatorWorkEstimate(
                operator_id=operator.operator_id,
                kind=operator.kind,
                compute_ops=compute_ops,
                memory_bytes=memory_bytes,
                weight_bytes=weight_bytes,
                activation_bytes=activation_bytes,
                output_bytes=activation_bytes,
                workspace_bytes=activation_bytes,
                kv_cache_bytes=kv_cache_bytes,
            ))
        return ModelPhaseWork(phase, tokens, invocations, tuple(rows))

    def request_work(self, input_tokens: int, output_tokens: int) -> ModelRequestWork:
        _integer("request input tokens", input_tokens, 1)
        _integer("request output tokens", output_tokens, 1)
        total_tokens = input_tokens + output_tokens
        activation_bytes = total_tokens * self.embedding_length * 2
        tensor_by_id = self.tensor_by_id
        attention_context_by_window: dict[int, int] = {}
        rows = []
        for operator in self.operators:
            layer_index = self._operator_layer_index(operator)
            tensors = tuple(tensor_by_id[value] for value in operator.tensor_ids)
            weight_bytes = sum(row.nbytes for row in tensors)
            compute_ops = self._dense_flops(tensors, total_tokens)
            kv_cache_bytes = 0
            memory_bytes = weight_bytes + 2 * activation_bytes
            if operator.kind == "embedding":
                compute_ops += total_tokens * self.embedding_length
            elif operator.kind == "attention":
                window = (
                    self.sliding_window
                    if self._layer_is_sliding(layer_index)
                    else self.context_length
                )
                context_sum = attention_context_by_window.get(window)
                if context_sum is None:
                    capped = min(input_tokens, window)
                    context_sum = capped * (capped + 1) // 2
                    if input_tokens > window:
                        context_sum += (input_tokens - window) * window
                    context_sum += self._capped_context_sum(
                        input_tokens, output_tokens, window
                    )
                    attention_context_by_window[window] = context_sum
                key_length, value_length = self._layer_kv_lengths(
                    layer_index
                )
                attention_flops = (
                    4 * self.head_count * key_length * context_sum
                )
                compute_ops += attention_flops
                context_reads = context_sum
                memory_bytes += (
                    context_reads
                    * self._layer_head_count_kv(layer_index)
                    * (key_length + value_length)
                    * 2
                )
            elif operator.kind == "kv_cache":
                kv_cache_bytes = self._kv_bytes(
                    input_tokens, output_tokens, layer_index
                )
                memory_bytes += kv_cache_bytes
            rows.append(OperatorWorkEstimate(
                operator_id=operator.operator_id,
                kind=operator.kind,
                compute_ops=compute_ops,
                memory_bytes=memory_bytes,
                weight_bytes=weight_bytes,
                activation_bytes=activation_bytes,
                output_bytes=activation_bytes,
                workspace_bytes=activation_bytes,
                kv_cache_bytes=kv_cache_bytes,
            ))
        phases = (
            self._phase_work(
                input_tokens, output_tokens, "prefill", tensor_by_id
            ),
            self._phase_work(
                input_tokens, output_tokens, "decode", tensor_by_id
            ),
        )
        return ModelRequestWork(
            input_tokens, output_tokens, tuple(rows), phases
        )

    def to_json(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "artifact_bytes": self.artifact_bytes,
            "artifact_sha256": self.artifact_sha256,
            "block_count": self.block_count,
            "context_length": self.context_length,
            "sliding_window": self.sliding_window,
            "embedding_length": self.embedding_length,
            "feed_forward_length": self.feed_forward_length,
            "head_count": self.head_count,
            "head_count_kv": self.head_count_kv,
            "head_count_kv_by_layer": list(self.head_count_kv_by_layer),
            "key_length": self.key_length,
            "key_length_swa": self.key_length_swa,
            "model_id": self.model_id,
            "operators": [row.to_json() for row in self.operators],
            "schema": MODEL_MANIFEST_SCHEMA,
            "tensor_bytes": self.tensor_bytes,
            "tensors": [row.to_json() for row in self.tensors],
            "value_length": self.value_length,
            "value_length_swa": self.value_length_swa,
            "sliding_window_pattern": list(self.sliding_window_pattern),
        }

    @classmethod
    def from_json(cls, value: object) -> "ModelManifest":
        if type(value) is not dict:
            raise ModelManifestError("model manifest must be an object")
        if value.get("schema") != MODEL_MANIFEST_SCHEMA:
            raise ModelManifestError("model manifest schema is invalid")
        tensors = value.get("tensors")
        operators = value.get("operators")
        if type(tensors) is not list or type(operators) is not list:
            raise ModelManifestError("model manifest rows must be lists")
        result = cls(
            model_id=value.get("model_id"),
            artifact_sha256=value.get("artifact_sha256"),
            artifact_bytes=value.get("artifact_bytes"),
            architecture=value.get("architecture"),
            context_length=value.get("context_length"),
            sliding_window=value.get("sliding_window"),
            embedding_length=value.get("embedding_length"),
            feed_forward_length=value.get("feed_forward_length"),
            block_count=value.get("block_count"),
            head_count=value.get("head_count"),
            head_count_kv=value.get("head_count_kv"),
            key_length=value.get("key_length"),
            value_length=value.get("value_length"),
            tensors=tuple(ModelTensorManifest.from_json(row) for row in tensors),
            operators=tuple(
                ModelOperatorManifest.from_json(row) for row in operators
            ),
            head_count_kv_by_layer=tuple(
                value.get("head_count_kv_by_layer", ())
            ),
            sliding_window_pattern=tuple(
                value.get("sliding_window_pattern", ())
            ),
            key_length_swa=value.get("key_length_swa"),
            value_length_swa=value.get("value_length_swa"),
        )
        tensor_bytes = value.get("tensor_bytes")
        if tensor_bytes is not None and tensor_bytes != result.tensor_bytes:
            raise ModelManifestError("model manifest tensor bytes differ")
        return result


_LAYER_PATTERN = re.compile(r"(?:^|\.)blk\.(\d+)\.")


def _tensor_role(name: str) -> str:
    if "token_embd" in name:
        return "embedding"
    if name == "output.weight" or name.startswith("output."):
        return "lm_head"
    if ".attn_" in name or "attn_norm" in name:
        return "attention_projection"
    if ".ffn_" in name or "ffn_norm" in name:
        return "ffn"
    return "lm_head" if "output_norm" in name else "other"


def _field_value(reader: object, key: str) -> object:
    fields = getattr(reader, "fields")
    if key not in fields:
        raise ModelManifestError(f"GGUF metadata is missing {key}")
    field = fields[key]
    values = []
    for index in field.data:
        value = field.parts[index]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        values.append(value)
    return values[0] if len(values) == 1 else values


def _field_int(reader: object, key: str) -> int:
    value = _field_value(reader, key)
    if isinstance(value, (list, tuple)):
        if not value:
            raise ModelManifestError(f"GGUF metadata is empty: {key}")
        value = max(value)
    if type(value) is not int:
        raise ModelManifestError(f"GGUF metadata is not integer: {key}")
    return value


def _field_optional_int(reader: object, key: str, default: int) -> int:
    if key not in getattr(reader, "fields"):
        return default
    return _field_int(reader, key)


def _field_optional_int_tuple(
    reader: object, key: str, count: int
) -> tuple[int, ...]:
    if key not in getattr(reader, "fields"):
        return ()
    value = _field_value(reader, key)
    if not isinstance(value, (list, tuple)):
        return ()
    result = tuple(value)
    if (
        len(result) != count
        or any(type(item) is not int or item <= 0 for item in result)
    ):
        raise ModelManifestError(
            f"GGUF per-layer integer metadata is invalid: {key}"
        )
    return result


def _field_optional_bool_tuple(
    reader: object, key: str, count: int
) -> tuple[bool, ...]:
    if key not in getattr(reader, "fields"):
        return ()
    value = _field_value(reader, key)
    if not isinstance(value, (list, tuple)):
        return ()
    result = tuple(value)
    if (
        len(result) != count
        or any(type(item) is not bool for item in result)
    ):
        raise ModelManifestError(
            f"GGUF per-layer boolean metadata is invalid: {key}"
        )
    return result


def _architecture(reader: object) -> str:
    value = _field_value(reader, "general.architecture")
    if isinstance(value, list) and all(type(item) is int for item in value):
        value = bytes(value).decode("utf-8")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return _text("GGUF architecture", value)


def _reader_class():
    try:
        from gguf import GGUFReader
    except ModuleNotFoundError:
        repository = Path(__file__).resolve().parents[3]
        package_path = repository / "gguf-py"
        if not package_path.is_dir():
            raise ModuleNotFoundError("local gguf-py package is unavailable")
        sys.path.insert(0, str(package_path))
        from gguf import GGUFReader
    return GGUFReader


class GGUFModelManifestLoader:
    """Extract a generic operator DAG without architecture-name branches."""

    @classmethod
    def load(cls, model_id: str, path: str | Path) -> ModelManifest:
        model_id = _text("model id", model_id)
        source = Path(path).resolve()
        if not source.is_file():
            raise ModelManifestError("GGUF artifact does not exist")
        try:
            reader_class = _reader_class()
            from gguf.constants import GGML_QUANT_SIZES
            reader = reader_class(source, "r")
        except (ImportError, ModuleNotFoundError):
            try:
                reader = read_gguf_metadata(source)
                GGML_QUANT_SIZES = None
            except GGUFMetadataError as exc:
                raise ModelManifestError(
                    f"cannot read GGUF artifact: {exc}"
                ) from exc
        except Exception as exc:
            raise ModelManifestError(f"cannot read GGUF artifact: {exc}") from exc
        architecture = _architecture(reader)
        prefix = architecture + "."
        context_length = _field_int(reader, prefix + "context_length")
        embedding_length = _field_int(
            reader, prefix + "embedding_length"
        )
        head_count = _field_int(reader, prefix + "attention.head_count")
        if embedding_length % head_count:
            raise ModelManifestError(
                "GGUF embedding length is not divisible by attention heads"
            )
        default_head_length = embedding_length // head_count
        block_count = _field_int(reader, prefix + "block_count")
        metadata = {
            "context_length": context_length,
            "sliding_window": _field_optional_int(
                reader,
                prefix + "attention.sliding_window",
                context_length,
            ),
            "embedding_length": embedding_length,
            "feed_forward_length": _field_int(
                reader, prefix + "feed_forward_length"
            ),
            "block_count": block_count,
            "head_count": head_count,
            "head_count_kv": _field_optional_int(
                reader, prefix + "attention.head_count_kv", head_count
            ),
            "head_count_kv_by_layer": _field_optional_int_tuple(
                reader,
                prefix + "attention.head_count_kv",
                block_count,
            ),
            "key_length": _field_optional_int(
                reader, prefix + "attention.key_length", default_head_length
            ),
            "key_length_swa": (
                None
                if prefix + "attention.key_length_swa"
                    not in getattr(reader, "fields")
                else _field_int(
                    reader, prefix + "attention.key_length_swa"
                )
            ),
            "value_length": _field_optional_int(
                reader, prefix + "attention.value_length", default_head_length
            ),
            "value_length_swa": (
                None
                if prefix + "attention.value_length_swa"
                    not in getattr(reader, "fields")
                else _field_int(
                    reader, prefix + "attention.value_length_swa"
                )
            ),
            "sliding_window_pattern": _field_optional_bool_tuple(
                reader,
                prefix + "attention.sliding_window_pattern",
                block_count,
            ),
        }
        tensors = []
        for tensor in reader.tensors:
            name = _text("GGUF tensor name", tensor.name)
            match = _LAYER_PATTERN.search(name)
            if GGML_QUANT_SIZES is None:
                block_size = tensor.quantization_block_size
                type_size = tensor.quantization_type_size
            else:
                block_size, type_size = GGML_QUANT_SIZES[
                    tensor.tensor_type
                ]
            tensors.append(ModelTensorManifest(
                tensor_id=name,
                shape=tuple(int(value) for value in tensor.shape),
                quantization=_text(
                    "GGUF tensor quantization", tensor.tensor_type.name
                ),
                quantization_block_size=int(block_size),
                quantization_type_size=int(type_size),
                nbytes=int(tensor.n_bytes),
                role=_tensor_role(name),
                layer_index=(None if match is None else int(match.group(1))),
            ))
        by_role: dict[tuple[int | None, str], list[str]] = {}
        for tensor in tensors:
            by_role.setdefault((tensor.layer_index, tensor.role), []).append(
                tensor.tensor_id
            )

        operators = []
        previous: str | None = None

        def append(operator_id: str, layer_id: str, kind: str, tensor_ids=()):
            nonlocal previous
            operators.append(ModelOperatorManifest(
                operator_id=operator_id,
                layer_id=layer_id,
                kind=kind,
                dependencies=(() if previous is None else (previous,)),
                tensor_ids=tuple(sorted(tensor_ids)),
            ))
            previous = operator_id

        append(
            "embedding",
            "embedding",
            "embedding",
            by_role.get((None, "embedding"), ()),
        )
        for layer_index in range(metadata["block_count"]):
            layer_id = f"layer:{layer_index}"
            append(
                layer_id + ":attention_projection",
                layer_id,
                "attention_projection",
                by_role.get((layer_index, "attention_projection"), ()),
            )
            append(layer_id + ":attention", layer_id, "attention")
            append(layer_id + ":kv_cache", layer_id, "kv_cache")
            append(
                layer_id + ":ffn",
                layer_id,
                "ffn",
                by_role.get((layer_index, "ffn"), ()),
            )
        head_tensors = list(by_role.get((None, "lm_head"), ()))
        head_tensors.extend(by_role.get((None, "other"), ()))
        append("lm_head", "lm_head", "lm_head", head_tensors)
        return ModelManifest(
            model_id=model_id,
            artifact_sha256=_sha256_file(source),
            artifact_bytes=source.stat().st_size,
            architecture=architecture,
            tensors=tuple(tensors),
            operators=tuple(operators),
            **metadata,
        )
