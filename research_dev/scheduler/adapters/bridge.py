"""Lifecycle handling for a scheduler-selected FunctionFS bridge."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
from types import MappingProxyType
import time
from typing import Callable, Mapping, Sequence

from .contracts import PhysicalAdapterError


_PROTOCOL_MAGIC = 0x46534631
_PROTOCOL_VERSION = 5
_HELLO_REQUEST = 1
_HELLO_RESPONSE = 2
_EXECUTE_REQUEST = 3
_EXECUTE_RESPONSE = 4
_F16_IO_FLAG = 1
_SWIGLU_FLAG = 2


def _receive_exact(stream: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            raise PhysicalAdapterError(
                "FunctionFS bridge closed during qualification"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _hash_bytes(value: bytes) -> int:
    result = 2166136261
    for item in value:
        result ^= item
        result = (result * 16777619) & 0xFFFFFFFF
    return result


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


@dataclass(frozen=True)
class FunctionFsBridgeTerminalReceipt:
    status: str
    calls: int
    allocator: str
    reset_recoveries: int
    values: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.status != "ok":
            raise PhysicalAdapterError("FunctionFS bridge status is not ok")
        if type(self.calls) is not int or self.calls < 0:
            raise PhysicalAdapterError("FunctionFS bridge calls are invalid")
        if (
            type(self.allocator) is not str
            or not self.allocator
            or not self.allocator.isascii()
        ):
            raise PhysicalAdapterError(
                "FunctionFS bridge allocator is invalid"
            )
        if (
            type(self.reset_recoveries) is not int
            or self.reset_recoveries < 0
        ):
            raise PhysicalAdapterError(
                "FunctionFS bridge recovery count is invalid"
            )
        object.__setattr__(
            self,
            "values",
            MappingProxyType(dict(sorted(self.values.items()))),
        )

    def to_json(self) -> dict[str, object]:
        return dict(self.values)


@dataclass(frozen=True)
class FunctionFsUsbObservation:
    sysfs_device: str
    vendor_id: str
    product_id: str
    negotiated_speed_mbps: int
    kernel_release: str

    def __post_init__(self) -> None:
        if (
            type(self.sysfs_device) is not str
            or not self.sysfs_device
            or not self.sysfs_device.isascii()
            or type(self.vendor_id) is not str
            or len(self.vendor_id) != 4
            or type(self.product_id) is not str
            or len(self.product_id) != 4
            or any(
                character not in "0123456789abcdef"
                for character in self.vendor_id + self.product_id
            )
            or type(self.negotiated_speed_mbps) is not int
            or self.negotiated_speed_mbps <= 0
            or type(self.kernel_release) is not str
            or not self.kernel_release
            or not self.kernel_release.isascii()
        ):
            raise PhysicalAdapterError(
                "FunctionFS USB observation is invalid"
            )

    def to_json(self) -> dict[str, object]:
        return {
            "kernel_release": self.kernel_release,
            "negotiated_speed_mbps": self.negotiated_speed_mbps,
            "product_id": self.product_id,
            "sysfs_device": self.sysfs_device,
            "vendor_id": self.vendor_id,
        }


@dataclass(frozen=True)
class AndroidUsbRestorationReceipt:
    serial: str
    adb_port: int
    sysfs_device: str
    vendor_id: str
    product_id: str
    negotiated_speed_mbps: int

    def __post_init__(self) -> None:
        if (
            type(self.serial) is not str
            or not self.serial
            or not self.serial.isascii()
            or type(self.adb_port) is not int
            or not 0 < self.adb_port <= 65535
            or type(self.sysfs_device) is not str
            or not self.sysfs_device
            or not self.sysfs_device.isascii()
            or type(self.vendor_id) is not str
            or len(self.vendor_id) != 4
            or type(self.product_id) is not str
            or len(self.product_id) != 4
            or any(
                character not in "0123456789abcdef"
                for character in self.vendor_id + self.product_id
            )
            or type(self.negotiated_speed_mbps) is not int
            or self.negotiated_speed_mbps <= 0
        ):
            raise PhysicalAdapterError(
                "Android USB restoration receipt is invalid"
            )

    def to_json(self) -> dict[str, object]:
        return {
            "adb_port": self.adb_port,
            "adb_state": "device",
            "negotiated_speed_mbps": self.negotiated_speed_mbps,
            "product_id": self.product_id,
            "serial": self.serial,
            "status": "RESTORED",
            "sysfs_device": self.sysfs_device,
            "vendor_id": self.vendor_id,
        }


@dataclass(frozen=True)
class FunctionFsTransportQualification:
    host: str
    port: int
    layer: int
    n_embd: int
    columns: int
    max_tokens: int
    activation: str
    queue_depth: int
    samples: tuple[Mapping[str, int], ...]

    def __post_init__(self) -> None:
        if (
            type(self.host) is not str
            or not self.host
            or not self.host.isascii()
            or type(self.port) is not int
            or not 0 < self.port <= 65535
            or type(self.layer) is not int
            or not 0 <= self.layer < 64
            or type(self.n_embd) is not int
            or self.n_embd <= 0
            or type(self.columns) is not int
            or self.columns <= 0
            or type(self.max_tokens) is not int
            or self.max_tokens <= 0
            or self.activation not in {"geglu", "swiglu"}
            or self.queue_depth != 1
        ):
            raise PhysicalAdapterError(
                "FunctionFS transport qualification is invalid"
            )
        rows = tuple(MappingProxyType(dict(row)) for row in self.samples)
        if not rows or any(
            set(row) != {
                "compute_us",
                "latency_us",
                "request_payload_bytes",
                "response_payload_bytes",
                "tokens",
            }
            or any(type(value) is not int or value < 0 for value in row.values())
            or row["tokens"] <= 0
            or row["request_payload_bytes"] <= 0
            or row["response_payload_bytes"] <= 0
            for row in rows
        ):
            raise PhysicalAdapterError(
                "FunctionFS transport qualification samples are invalid"
            )
        object.__setattr__(self, "samples", rows)

    def to_json(self) -> dict[str, object]:
        latencies = [row["latency_us"] for row in self.samples]
        decode = [
            row["latency_us"] for row in self.samples
            if row["tokens"] == 1
        ]
        prefill = [
            row["latency_us"] for row in self.samples
            if row["tokens"] > 1
        ]
        return {
            "activation": self.activation,
            "calls": len(self.samples),
            "columns": self.columns,
            "decode_latency_us": {
                "p50": _percentile(decode, 0.50),
                "p90": _percentile(decode, 0.90),
            },
            "host": self.host,
            "layer": self.layer,
            "latency_us": {
                "p50": _percentile(latencies, 0.50),
                "p90": _percentile(latencies, 0.90),
            },
            "max_tokens": self.max_tokens,
            "n_embd": self.n_embd,
            "port": self.port,
            "prefill_latency_us": {
                "p50": _percentile(prefill, 0.50),
                "p90": _percentile(prefill, 0.90),
            },
            "queue_depth": self.queue_depth,
            "request_payload_bytes": sum(
                row["request_payload_bytes"] for row in self.samples
            ),
            "response_payload_bytes": sum(
                row["response_payload_bytes"] for row in self.samples
            ),
            "samples": [dict(row) for row in self.samples],
            "schema": "research-scheduler-functionfs-qualification-v1",
            "status": "PASS",
        }


def parse_functionfs_bridge_terminal(
    lines: Sequence[str],
) -> FunctionFsBridgeTerminalReceipt:
    matches = []
    for line in lines:
        if not line.startswith("FFNDMABUF "):
            continue
        try:
            value = json.loads(line.removeprefix("FFNDMABUF "))
        except json.JSONDecodeError as error:
            raise PhysicalAdapterError(
                "FunctionFS bridge terminal receipt is invalid"
            ) from error
        if type(value) is not dict:
            raise PhysicalAdapterError(
                "FunctionFS bridge terminal receipt is invalid"
            )
        matches.append(value)
    if len(matches) != 1:
        raise PhysicalAdapterError(
            "FunctionFS bridge terminal receipt is not unique"
        )
    value = matches[0]
    return FunctionFsBridgeTerminalReceipt(
        status=value.get("status"),
        calls=value.get("calls"),
        allocator=value.get("allocator"),
        reset_recoveries=value.get("reset_recoveries"),
        values=value,
    )


def parse_functionfs_bridge_qualification(
    lines: Sequence[str],
) -> Mapping[str, object]:
    matches = []
    for line in lines:
        if not line.startswith("FFNDMABUFQUAL "):
            continue
        try:
            value = json.loads(line.removeprefix("FFNDMABUFQUAL "))
        except json.JSONDecodeError as error:
            raise PhysicalAdapterError(
                "FunctionFS qualification receipt is invalid"
            ) from error
        if type(value) is not dict:
            raise PhysicalAdapterError(
                "FunctionFS qualification receipt is invalid"
            )
        matches.append(value)
    if len(matches) != 1:
        raise PhysicalAdapterError(
            "FunctionFS qualification receipt is not unique"
        )
    value = matches[0]
    required_integer = (
        "calls",
        "queue_depth",
        "reset_recoveries",
        "upload_bytes",
        "download_bytes",
    )
    required_rate = (
        "h2d_payload_bytes_per_s",
        "d2h_conservative_payload_bytes_per_s",
        "d2h_exposed_payload_bytes_per_s",
        "full_duplex_payload_bytes_per_s",
    )
    if (
        value.get("status") != "ok"
        or value.get("queue_depth") != 1
        or any(
            type(value.get(name)) is not int or value[name] < 0
            for name in required_integer
        )
        or value.get("calls", 0) <= 0
        or value.get("upload_bytes", 0) <= 0
        or value.get("download_bytes", 0) <= 0
        or any(
            type(value.get(name)) not in {int, float}
            or value[name] < 0
            for name in required_rate
        )
        or value.get("h2d_payload_bytes_per_s", 0) <= 0
        or value.get("d2h_conservative_payload_bytes_per_s", 0) <= 0
        or value.get("full_duplex_measured") is not False
        or type(value.get("allocator")) is not str
        or not value["allocator"].isascii()
    ):
        raise PhysicalAdapterError(
            "FunctionFS qualification receipt failed validation"
        )
    return MappingProxyType(dict(sorted(value.items())))


def probe_functionfs_usb_device(
    *,
    vendor_id: str,
    product_id: str,
    sysfs_root: Path = Path("/sys/bus/usb/devices"),
) -> FunctionFsUsbObservation:
    """Resolve one exact FunctionFS USB function and negotiated link speed."""

    vendor = vendor_id.lower()
    product = product_id.lower()
    if (
        len(vendor) != 4
        or len(product) != 4
        or any(
            character not in "0123456789abcdef"
            for character in vendor + product
        )
        or not isinstance(sysfs_root, Path)
        or not sysfs_root.is_dir()
    ):
        raise PhysicalAdapterError(
            "FunctionFS USB probe contract is invalid"
        )
    matches = []
    for path in sorted(sysfs_root.iterdir(), key=lambda item: item.name):
        try:
            observed_vendor = (path / "idVendor").read_text(
                encoding="ascii"
            ).strip().lower()
            observed_product = (path / "idProduct").read_text(
                encoding="ascii"
            ).strip().lower()
        except (FileNotFoundError, NotADirectoryError, OSError):
            continue
        if (observed_vendor, observed_product) != (vendor, product):
            continue
        try:
            speed_text = (path / "speed").read_text(
                encoding="ascii"
            ).strip()
            speed_mbps = int(float(speed_text))
        except (FileNotFoundError, OSError, ValueError) as error:
            raise PhysicalAdapterError(
                "FunctionFS USB speed observation is invalid"
            ) from error
        matches.append(FunctionFsUsbObservation(
            sysfs_device=path.name,
            vendor_id=vendor,
            product_id=product,
            negotiated_speed_mbps=speed_mbps,
            kernel_release=os.uname().release,
        ))
    if len(matches) != 1:
        raise PhysicalAdapterError(
            "FunctionFS USB device observation is not unique"
        )
    return matches[0]


def verify_android_usb_restored(
    *,
    serial: str,
    adb_port: int,
    minimum_speed_mbps: int,
    timeout_s: float = 60.0,
    sysfs_root: Path = Path("/sys/bus/usb/devices"),
    adb_probe: Callable[[], tuple[int, str]] | None = None,
) -> AndroidUsbRestorationReceipt:
    """Wait for ADB and normal USB together; zero timeout probes once."""

    if (
        type(serial) is not str
        or not serial
        or not serial.isascii()
        or type(adb_port) is not int
        or not 0 < adb_port <= 65535
        or type(minimum_speed_mbps) is not int
        or minimum_speed_mbps <= 0
        or type(timeout_s) not in {int, float}
        or timeout_s < 0
        or not isinstance(sysfs_root, Path)
        or not sysfs_root.is_dir()
        or adb_probe is not None
        and not callable(adb_probe)
    ):
        raise PhysicalAdapterError(
            "Android USB restoration contract is invalid"
        )

    def default_adb_probe() -> tuple[int, str]:
        completed = subprocess.run(
            ["adb", "-P", str(adb_port), "-s", serial, "get-state"],
            check=False,
            capture_output=True,
            text=True,
            encoding="ascii",
            errors="backslashreplace",
            timeout=5,
        )
        return completed.returncode, completed.stdout.strip()

    probe = default_adb_probe if adb_probe is None else adb_probe
    deadline = time.monotonic() + float(timeout_s)
    last_detail = "ADB is unavailable"
    while True:
        try:
            returncode, state = probe()
        except (OSError, subprocess.SubprocessError) as error:
            returncode, state = 1, ""
            last_detail = str(error)
        if type(returncode) is not int or type(state) is not str:
            raise PhysicalAdapterError(
                "Android USB ADB probe result is invalid"
            )
        if returncode == 0 and state == "device":
            matches = []
            for path in sorted(
                sysfs_root.iterdir(), key=lambda item: item.name
            ):
                try:
                    observed_serial = (path / "serial").read_text(
                        encoding="ascii"
                    ).strip()
                    vendor = (path / "idVendor").read_text(
                        encoding="ascii"
                    ).strip().lower()
                    product = (path / "idProduct").read_text(
                        encoding="ascii"
                    ).strip().lower()
                    speed = int(float((path / "speed").read_text(
                        encoding="ascii"
                    ).strip()))
                except (
                    FileNotFoundError,
                    NotADirectoryError,
                    OSError,
                    ValueError,
                ):
                    continue
                if observed_serial == serial:
                    matches.append((path.name, vendor, product, speed))
            if len(matches) == 1 and matches[0][3] >= minimum_speed_mbps:
                name, vendor, product, speed = matches[0]
                return AndroidUsbRestorationReceipt(
                    serial=serial,
                    adb_port=adb_port,
                    sysfs_device=name,
                    vendor_id=vendor,
                    product_id=product,
                    negotiated_speed_mbps=speed,
                )
            last_detail = (
                "serial-matched normal USB gadget is absent, ambiguous, "
                "or below the required speed"
            )
        else:
            last_detail = "ADB state is " + state
        if time.monotonic() >= deadline:
            raise PhysicalAdapterError(
                "Android USB restoration failed: " + last_detail
            )
        time.sleep(0.25)


def qualify_functionfs_bridge(
    *,
    host: str,
    port: int,
    layer_mask: int,
    n_embd: int,
    columns: int,
    max_tokens: int,
    activation: str,
    token_shapes: Sequence[int] = (1, 32),
    repeats: int = 4,
    timeout_s: float = 30.0,
) -> FunctionFsTransportQualification:
    """Exercise the deployed bridge before its inference client connects."""

    shapes = tuple(token_shapes)
    if (
        type(layer_mask) is not int
        or layer_mask <= 0
        or any(type(tokens) is not int or tokens <= 0 for tokens in shapes)
        or not shapes
        or max(shapes) > max_tokens
        or type(repeats) is not int
        or repeats <= 0
    ):
        raise PhysicalAdapterError(
            "FunctionFS qualification shape is invalid"
        )
    layer = (layer_mask & -layer_mask).bit_length() - 1
    flags = _F16_IO_FLAG | (
        _SWIGLU_FLAG if activation == "swiglu" else 0
    )
    hello = struct.pack(
        "<IHHQIIHH4x",
        _PROTOCOL_MAGIC,
        _PROTOCOL_VERSION,
        _HELLO_REQUEST,
        layer_mask,
        n_embd,
        columns,
        flags,
        max_tokens,
    )
    samples = []
    with socket.create_connection((host, port), timeout=timeout_s) as stream:
        stream.settimeout(timeout_s)
        stream.sendall(hello)
        response = _receive_exact(stream, 64)
        magic, version, message, status = struct.unpack_from(
            "<IHHH", response
        )
        response_n_embd = struct.unpack_from("<I", response, 12)[0]
        response_columns = struct.unpack_from("<I", response, 24)[0]
        response_max_tokens = struct.unpack_from("<H", response, 60)[0]
        if (
            magic != _PROTOCOL_MAGIC
            or version != _PROTOCOL_VERSION
            or message != _HELLO_RESPONSE
            or status != 0
            or response_n_embd != n_embd
            or response_columns != columns
            or response_max_tokens != max_tokens
        ):
            raise PhysicalAdapterError(
                "FunctionFS qualification HELLO differs"
            )
        request_id = 1
        for tokens in shapes:
            payload = bytes(n_embd * tokens * 2)
            payload_hash = _hash_bytes(payload)
            for _ in range(repeats):
                request = struct.pack(
                    "<IHHIiIIIII",
                    _PROTOCOL_MAGIC,
                    _PROTOCOL_VERSION,
                    _EXECUTE_REQUEST,
                    request_id,
                    layer,
                    n_embd * tokens,
                    len(payload),
                    payload_hash,
                    columns,
                    tokens,
                )
                started_ns = time.monotonic_ns()
                stream.sendall(request)
                stream.sendall(payload)
                raw_response = _receive_exact(stream, 48)
                output = _receive_exact(stream, len(payload))
                finished_ns = time.monotonic_ns()
                fields = struct.unpack("<IHHHHIiIIIIIQ", raw_response)
                (
                    result_magic,
                    result_version,
                    result_message,
                    result_status,
                    _reserved,
                    result_id,
                    result_layer,
                    result_elements,
                    result_payload_bytes,
                    result_payload_hash,
                    result_columns,
                    result_tokens,
                    compute_us,
                ) = fields
                if (
                    result_magic != _PROTOCOL_MAGIC
                    or result_version != _PROTOCOL_VERSION
                    or result_message != _EXECUTE_RESPONSE
                    or result_status != 0
                    or result_id != request_id
                    or result_layer != layer
                    or result_elements != n_embd * tokens
                    or result_payload_bytes != len(payload)
                    or result_payload_hash != _hash_bytes(output)
                    or result_columns != columns
                    or result_tokens != tokens
                ):
                    raise PhysicalAdapterError(
                        "FunctionFS qualification response differs"
                    )
                samples.append({
                    "compute_us": compute_us,
                    "latency_us": (finished_ns - started_ns) // 1000,
                    "request_payload_bytes": len(payload),
                    "response_payload_bytes": len(output),
                    "tokens": tokens,
                })
                request_id += 1
    return FunctionFsTransportQualification(
        host=host,
        port=port,
        layer=layer,
        n_embd=n_embd,
        columns=columns,
        max_tokens=max_tokens,
        activation=activation,
        queue_depth=1,
        samples=tuple(samples),
    )


def close_functionfs_bridge(
    *,
    poll: Callable[[], int | None],
    stderr_lines: Callable[[], Sequence[str]],
    request_shutdown: Callable[[], int],
    finalize: Callable[[], None],
    wait_for_exit: Callable[[], int] | None = None,
) -> FunctionFsBridgeTerminalReceipt:
    """Close a bridge once and accept an existing clean terminal receipt."""

    for name, callback in (
        ("poll", poll),
        ("stderr", stderr_lines),
        ("shutdown", request_shutdown),
        ("finalize", finalize),
    ):
        if not callable(callback):
            raise PhysicalAdapterError(
                "FunctionFS bridge " + name + " callback is invalid"
            )
    if wait_for_exit is not None and not callable(wait_for_exit):
        raise PhysicalAdapterError(
            "FunctionFS bridge wait callback is invalid"
        )

    returncode = poll()
    shutdown_returncode = None
    try:
        if returncode is None:
            shutdown_returncode = request_shutdown()
            if type(shutdown_returncode) is not int:
                raise PhysicalAdapterError(
                    "FunctionFS bridge shutdown result is invalid"
                )
            returncode = poll()
            if returncode is None and wait_for_exit is not None:
                returncode = wait_for_exit()
            if returncode is None:
                raise PhysicalAdapterError(
                    "FunctionFS bridge did not exit after shutdown"
                )
    finally:
        finalize()

    receipt = parse_functionfs_bridge_terminal(stderr_lines())
    if returncode != 0:
        raise PhysicalAdapterError(
            "FunctionFS bridge exited unsuccessfully"
        )
    if shutdown_returncode not in {None, 0}:
        # A close helper can lose a race to a bridge that already exited.
        # The zero exit status and terminal receipt are authoritative.
        return receipt
    return receipt
