"""Phone FFN configuration, physical events and receipts: common."""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Mapping

from ..contracts import PhysicalAdapterError


_TERMINAL_PATTERN = re.compile(
    r"^\[ffn-worker\] DMA-BUF complete transport=(direct|staged) "
    r"requests=([0-9]+)(?: queue_depth=([0-9]+) "
    r"maximum_pending_outputs=([0-9]+))? "
    r"(?:phone_payload_copies=([0-9]+) )?"
    r"(?:d2h_completions=([0-9]+) d2h_queue_us=([0-9]+) "
    r"d2h_queue_max_us=([0-9]+) )?"
    r"recoveries=([0-9]+) status=([0-9]+)$"
)


_MULTI_TERMINAL_PREFIX = "MULTIPHONEFFN "


_RESIDENCY_PHASE_PREFIX = "RESIDENTPHASE "


_RESIDENCY_CALL_PREFIX = "RESIDENTCALL "


def _android_path(name: str, value: object) -> str:
    if (
        type(value) is not str
        or not value.startswith("/")
        or not value.isascii()
        or any(character in value for character in ("\n", "\r", "\x00"))
    ):
        raise PhysicalAdapterError(name + " is not an absolute ASCII path")
    return value


def _artifact_mapping(
    name: str,
    values: Mapping[str, str],
) -> Mapping[str, str]:
    result = {}
    for artifact, path in values.items():
        if (
            type(artifact) is not str
            or not artifact.startswith("sha256:")
            or len(artifact) != 71
            or any(value not in "0123456789abcdef" for value in artifact[7:])
        ):
            raise PhysicalAdapterError(name + " artifact is invalid")
        result[artifact] = _android_path(name + " path", path)
    if not result:
        raise PhysicalAdapterError(name + " mapping is empty")
    return MappingProxyType(dict(sorted(result.items())))
