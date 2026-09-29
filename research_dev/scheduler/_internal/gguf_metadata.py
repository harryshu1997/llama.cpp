"""Small dependency-free GGUF metadata and tensor-directory reader."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path
import struct
from types import SimpleNamespace


class GGUFMetadataError(ValueError):
    pass


_SCALAR_FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}

_QUANTIZATION = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18),
}

_METADATA_SUFFIXES = (
    ".attention.head_count",
    ".attention.head_count_kv",
    ".attention.key_length",
    ".attention.key_length_swa",
    ".attention.sliding_window",
    ".attention.sliding_window_pattern",
    ".attention.value_length",
    ".attention.value_length_swa",
    ".block_count",
    ".context_length",
    ".embedding_length",
    ".feed_forward_length",
)


@dataclass(frozen=True)
class PortableGGUFTensor:
    name: str
    shape: tuple[int, ...]
    tensor_type: object
    n_bytes: int
    quantization_block_size: int
    quantization_type_size: int


class _Field:
    def __init__(self, value: object) -> None:
        self.parts = (value,)
        self.data = (0,)


class PortableGGUFReader:
    def __init__(self, fields, tensors) -> None:
        self.fields = fields
        self.tensors = tensors


class _Stream:
    def __init__(self, path: Path) -> None:
        self.source = path.open("rb")
        self.size = path.stat().st_size

    def close(self) -> None:
        self.source.close()

    def read(self, amount: int) -> bytes:
        if amount < 0 or self.source.tell() + amount > self.size:
            raise GGUFMetadataError("GGUF record exceeds the artifact")
        value = self.source.read(amount)
        if len(value) != amount:
            raise GGUFMetadataError("GGUF artifact is truncated")
        return value

    def unpack(self, format_code: str):
        return struct.unpack("<" + format_code, self.read(
            struct.calcsize("<" + format_code)
        ))[0]

    def skip(self, amount: int) -> None:
        if amount < 0 or self.source.tell() + amount > self.size:
            raise GGUFMetadataError("GGUF record exceeds the artifact")
        self.source.seek(amount, 1)

    def string(self, *, decode: bool) -> str | None:
        length = self.unpack("Q")
        if length > self.size:
            raise GGUFMetadataError("GGUF string length is invalid")
        if not decode:
            self.skip(length)
            return None
        try:
            return self.read(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFMetadataError("GGUF string is not UTF-8") from exc


def _keep_metadata(key: str) -> bool:
    return key in {"general.alignment", "general.architecture"} or key.endswith(
        _METADATA_SUFFIXES
    )


def _value(stream: _Stream, value_type: int, keep: bool):
    scalar = _SCALAR_FORMATS.get(value_type)
    if scalar is not None:
        if not keep:
            stream.skip(struct.calcsize("<" + scalar))
            return None
        return stream.unpack(scalar)
    if value_type == 8:
        return stream.string(decode=keep)
    if value_type != 9:
        raise GGUFMetadataError("GGUF metadata type is unsupported")
    element_type = stream.unpack("I")
    count = stream.unpack("Q")
    if count > stream.size:
        raise GGUFMetadataError("GGUF array length is invalid")
    scalar = _SCALAR_FORMATS.get(element_type)
    if not keep and scalar is not None:
        stream.skip(count * struct.calcsize("<" + scalar))
        return None
    if element_type == 9:
        raise GGUFMetadataError("nested GGUF arrays are unsupported")
    return tuple(_value(stream, element_type, keep) for _ in range(count))


def read_gguf_metadata(path: str | Path) -> PortableGGUFReader:
    source = Path(path)
    stream = _Stream(source)
    try:
        if stream.read(4) != b"GGUF":
            raise GGUFMetadataError("GGUF magic is invalid")
        version = stream.unpack("I")
        if version not in {2, 3}:
            raise GGUFMetadataError("GGUF version is unsupported")
        tensor_count = stream.unpack("Q")
        metadata_count = stream.unpack("Q")
        if tensor_count > stream.size or metadata_count > stream.size:
            raise GGUFMetadataError("GGUF header counts are invalid")

        fields = {}
        for _ in range(metadata_count):
            key = stream.string(decode=True)
            assert key is not None
            value = _value(stream, stream.unpack("I"), _keep_metadata(key))
            if value is not None:
                fields[key] = _Field(value)

        tensors = []
        names = set()
        for _ in range(tensor_count):
            name = stream.string(decode=True)
            assert name is not None
            if name in names:
                raise GGUFMetadataError("GGUF tensor names are duplicated")
            names.add(name)
            dimension_count = stream.unpack("I")
            if not 1 <= dimension_count <= 8:
                raise GGUFMetadataError("GGUF tensor rank is invalid")
            shape = tuple(stream.unpack("Q") for _ in range(dimension_count))
            if any(value < 1 for value in shape):
                raise GGUFMetadataError("GGUF tensor shape is invalid")
            quantization_id = stream.unpack("I")
            stream.unpack("Q")
            quantization = _QUANTIZATION.get(quantization_id)
            if quantization is None:
                raise GGUFMetadataError("GGUF tensor quantization is unsupported")
            name_value, block_size, type_size = quantization
            nbytes = prod(shape) * type_size // block_size
            if nbytes < 1:
                raise GGUFMetadataError("GGUF tensor byte size is invalid")
            tensors.append(PortableGGUFTensor(
                name=name,
                shape=shape,
                tensor_type=SimpleNamespace(name=name_value),
                n_bytes=nbytes,
                quantization_block_size=block_size,
                quantization_type_size=type_size,
            ))
        if not fields or not tensors:
            raise GGUFMetadataError("GGUF metadata or tensors are empty")
        return PortableGGUFReader(fields, tuple(tensors))
    finally:
        stream.close()
