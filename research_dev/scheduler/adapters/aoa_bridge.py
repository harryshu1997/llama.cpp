#!/usr/bin/env python3
"""Host side of the Pixel AOA bridge (WS10): llama-server's TCP endpoint -> Android Open Accessory bulk link.

llama-server keeps dialing ``127.0.0.1:<port>`` exactly as it dials an ``adb forward``; this process owns that
port instead of adb and carries every TCP connection over the Pixel's accessory bulk endpoints (S43A frames,
``native/aoa_bridge/s43a_protocol.h``) to the phone relay (``native/aoa_bridge/s43_aoa_relay.c``), which
connects to the unchanged FFN worker on the phone's loopback. Neither the server nor the worker changes.

    aoa_bridge.py inspect --sysfs-device 2-9.2 --serial 5A040DLCH004ES
    aoa_bridge.py switch  --sysfs-device 2-9.2 --serial 5A040DLCH004ES      # normal -> accessory+adb (18d1:2d01)
    aoa_bridge.py restore --sysfs-device 2-9.2 --serial 5A040DLCH004ES      # USB reset -> Android leaves accessory
    aoa_bridge.py serve   --sysfs-device 2-9.2 --serial 5A040DLCH004ES --listen-port 26991 \\
        [--echo-port P] [--keepalive-ms 2 --keepalive-window-ms 1500] [--ready-file F] [--status-file F] \\
        [--trace-file F] [--shutdown-relay-on-exit]
    aoa_bridge.py ping    --sysfs-device 2-9.2 --serial 5A040DLCH004ES --count 300 --request-bytes 10280 \\
        --response-bytes 10288 --gap-ms 6 [--token-gap-ms 300:500 --calls-per-token 6] --output F

Device selection is by the sysfs port path AND the serial, never by VID:PID: the OP15 on the same xHCI enumerates
as FunctionFS 18d1:2d00 (serial SCHEDFFN0001) and must never be opened. Any listed forbidden serial (default: the
OP15's ADB and gadget serials) is refused wherever it appears. The Pixel must enumerate as accessory+adb
(18d1:2d01); accessory-only 18d1:2d00 would collide with the OP15 gadget's first-match opens (llama-server's USB
client, the FunctionFS probe) and is refused (``switch`` resets it straight back).

Standard library only (ctypes + libusb-1.0), so the desktop runs it as a script with its system Python.
Exit codes of ``serve``: 0 stopped (SIGTERM/SIGINT), 2 selection/config, 3 USB link lost, 4 relay unresponsive or
handshake failed, 5 protocol error, 6 listen failed.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import ctypes.util
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import select
import selectors
import signal
import socket
import struct
import sys
import threading
import time
from typing import Callable, Iterable, Sequence


# --- S43A protocol (mirror of native/aoa_bridge/s43a_protocol.h) ---------------------------------

MAGIC = 0x41333453
VERSION = 1
HEADER = struct.Struct("<IHHIIIIQQ")
HEADER_BYTES = 40
MAGIC_BYTES = struct.pack("<I", MAGIC)
MAX_PAYLOAD = 1 << 20
PAD_QUANTUM = 512
HELLO, HELLO_ACK, OPEN, OPEN_ACK, DATA, CLOSE, NOP, PING, PONG, SHUTDOWN, ERROR = range(1, 12)
TYPE_NAMES = {HELLO: "HELLO", HELLO_ACK: "HELLO_ACK", OPEN: "OPEN", OPEN_ACK: "OPEN_ACK", DATA: "DATA",
              CLOSE: "CLOSE", NOP: "NOP", PING: "PING", PONG: "PONG", SHUTDOWN: "SHUTDOWN", ERROR: "ERROR"}
CLOSE_EOF, CLOSE_CONNECT_FAILED, CLOSE_IO, CLOSE_PROTOCOL, CLOSE_SESSION_RESET, CLOSE_SHUTDOWN = range(6)
OPEN_LOCAL_ECHO = 1
HELLO_PAYLOAD = struct.Struct("<II")
HELLO_ACK_PAYLOAD = struct.Struct("<IIII")
PING_PAYLOAD = struct.Struct("<QII")
PONG_PAYLOAD = struct.Struct("<QQQ")
U32 = struct.Struct("<I")
I32 = struct.Struct("<i")

GOOGLE_VENDOR = "18d1"
ACCESSORY_ADB_PRODUCT = "2d01"
ACCESSORY_PRODUCTS = frozenset({"2d00", "2d01", "2d02", "2d03", "2d04", "2d05"})
DEFAULT_FORBIDDEN_SERIALS = ("3C15AU002CL00000", "SCHEDFFN0001")
AOA_STRINGS = ("S43", "PixelFFNRelay", "WS10 FFN relay (no app)", "1", "https://example.invalid", "S43AOA1")
IN_TRANSFER_BYTES = 128 * 1024
TCP_CHUNK = 64 * 1024
USB_SYSFS_ROOT = Path("/sys/bus/usb/devices")
STATUS_SCHEMA = "ws10-aoa-bridge-status-v1"
EXIT_OK, EXIT_CONFIG, EXIT_LINK_LOST, EXIT_RELAY_UNRESPONSIVE, EXIT_PROTOCOL, EXIT_LISTEN = 0, 2, 3, 4, 5, 6


class BridgeError(Exception):
    exit_code = EXIT_CONFIG


class SelectionError(BridgeError):
    exit_code = EXIT_CONFIG


class LinkLost(BridgeError):
    exit_code = EXIT_LINK_LOST


class RelayUnresponsive(BridgeError):
    exit_code = EXIT_RELAY_UNRESPONSIVE


class ProtocolError(BridgeError):
    exit_code = EXIT_PROTOCOL


class AccessoryOnlyError(SelectionError):
    """The Pixel enumerated as accessory without adb (18d1:2d00 family): the OP15 gadget collision hazard."""


def pad_for(length: int) -> int:
    """1 exactly when header + payload would end on a 512-byte (hence max-packet) boundary."""
    return 1 if (HEADER_BYTES + length) % PAD_QUANTUM == 0 else 0


def pack_frame(kind: int, session: int, stream: int, payload: bytes = b"", *, aux: int = 0,
               stamp_ns: int | None = None) -> bytes:
    if not HELLO <= kind <= ERROR or len(payload) > MAX_PAYLOAD:
        raise ProtocolError("frame is invalid")
    pad = pad_for(len(payload))
    header = HEADER.pack(MAGIC, VERSION, kind, session & 0xFFFFFFFF, stream & 0xFFFFFFFF, len(payload), pad,
                         time.monotonic_ns() if stamp_ns is None else stamp_ns, aux & 0xFFFFFFFFFFFFFFFF)
    return header + bytes(payload) + b"\0" * pad


@dataclass(frozen=True)
class Frame:
    kind: int
    session: int
    stream: int
    payload: bytes
    stamp_ns: int
    aux: int

    @property
    def name(self) -> str:
        return TYPE_NAMES.get(self.kind, str(self.kind))


def _header_valid(fields: tuple) -> bool:
    magic, version, kind, _session, _stream, length, pad, _stamp, _aux = fields
    return (magic == MAGIC and version == VERSION and HELLO <= kind <= ERROR and length <= MAX_PAYLOAD
            and pad == pad_for(length))


class FrameParser:
    """Incremental S43A parser for the phone -> host direction.

    Until :meth:`expect` is satisfied the parser scans byte by byte for a valid header of the expected kind
    and session (stale bytes of an earlier session are discarded); afterwards an invalid header is a
    protocol error. Frames of another session are dropped and counted."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.session: int | None = None
        self._sync: tuple[int, int] | None = None
        self.discarded_bytes = 0
        self.stale_frames = 0

    def expect(self, kind: int, session: int) -> None:
        """Resynchronise: drop everything until a frame of ``kind`` in ``session``."""
        self._sync = (kind, session)
        self.session = session

    def feed(self, data: bytes) -> list[Frame]:
        self.buffer.extend(data)
        frames: list[Frame] = []
        offset = 0
        buffer = self.buffer
        while len(buffer) - offset >= HEADER_BYTES:
            if self._sync is not None:
                found = buffer.find(MAGIC_BYTES, offset)
                if found < 0:
                    keep = max(offset, len(buffer) - (len(MAGIC_BYTES) - 1))
                    self.discarded_bytes += keep - offset
                    offset = keep
                    break
                self.discarded_bytes += found - offset
                offset = found
                if len(buffer) - offset < HEADER_BYTES:
                    break
            fields = HEADER.unpack_from(buffer, offset)
            valid = _header_valid(fields)
            if self._sync is not None and not (valid and (fields[2], fields[3]) == self._sync):
                offset += 1
                self.discarded_bytes += 1
                continue
            if not valid:
                self.buffer = bytearray(buffer[offset:])
                raise ProtocolError("invalid S43A header from the relay: " + repr(fields[:7]))
            length, pad = fields[5], fields[6]
            total = HEADER_BYTES + length + pad
            if len(buffer) - offset < total:
                break
            self._sync = None
            if fields[3] != self.session:
                self.stale_frames += 1
            else:
                frames.append(Frame(fields[2], fields[3], fields[4],
                                    bytes(buffer[offset + HEADER_BYTES: offset + HEADER_BYTES + length]),
                                    fields[7], fields[8]))
            offset += total
        if offset:
            del buffer[:offset]
        return frames


# --- USB device selection (sysfs, read-only) -----------------------------------------------------

@dataclass(frozen=True)
class UsbDeviceState:
    sysfs_device: str
    serial: str
    vendor: str
    product: str
    busnum: int
    devnum: int
    speed_mbps: int

    @property
    def vendor_product(self) -> str:
        return self.vendor + ":" + self.product

    @property
    def port_numbers(self) -> tuple[int, ...]:
        return tuple(int(part) for part in self.sysfs_device.split("-", 1)[1].split("."))

    @property
    def accessory(self) -> bool:
        return self.vendor == GOOGLE_VENDOR and self.product in ACCESSORY_PRODUCTS

    def to_json(self) -> dict[str, object]:
        return {"busnum": self.busnum, "devnum": self.devnum, "serial": self.serial, "speed_mbps": self.speed_mbps,
                "sysfs_device": self.sysfs_device, "vendor_product": self.vendor_product}


def _valid_sysfs_device(name: str) -> bool:
    bus, separator, path = name.partition("-")
    return bool(separator) and bus.isdigit() and all(part.isdigit() for part in path.split(".")) and bool(path)


def read_usb_device(sysfs_device: str, *, sysfs_root: Path = USB_SYSFS_ROOT) -> UsbDeviceState:
    if not _valid_sysfs_device(sysfs_device):
        raise SelectionError("USB sysfs device must look like 2-9.2: " + repr(sysfs_device))
    device = sysfs_root / sysfs_device
    try:
        read = lambda name: (device / name).read_text(encoding="ascii").strip()  # noqa: E731
        state = UsbDeviceState(sysfs_device, read("serial"), read("idVendor").lower(), read("idProduct").lower(),
                               int(read("busnum")), int(read("devnum")), int(float(read("speed"))))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        raise SelectionError("USB device " + sysfs_device + " is absent or unreadable: " + str(error)) from error
    if str(state.busnum) != sysfs_device.split("-", 1)[0]:
        raise SelectionError("USB device " + sysfs_device + " reports another bus")
    return state


def check_pixel(state: UsbDeviceState, *, serial: str, forbidden_serials: Iterable[str],
                require_accessory: bool | None) -> None:
    """The device at the pinned port is the pinned Pixel, never a forbidden phone, and in the required mode.

    ``require_accessory``: True = accessory+adb (18d1:2d01) only, False = a normal (non-accessory) mode only,
    None = either. Accessory-only (2d00) and the audio accessory PIDs are always refused."""
    forbidden = set(forbidden_serials)
    if state.serial in forbidden:
        raise SelectionError(f"USB device {state.sysfs_device} is a forbidden phone ({state.serial})")
    if state.serial != serial or state.vendor != GOOGLE_VENDOR:
        raise SelectionError(f"USB device {state.sysfs_device} is {state.vendor_product} serial {state.serial}, "
                             f"not the pinned Pixel {serial}")
    if state.accessory and state.product != ACCESSORY_ADB_PRODUCT:
        raise AccessoryOnlyError(f"Pixel enumerates as {state.vendor_product}: accessory without adb collides with "
                                 "the OP15 FunctionFS gadget (18d1:2d00); restore it")
    if require_accessory is True and not state.accessory:
        raise SelectionError(f"Pixel is {state.vendor_product}, not in accessory+adb mode (18d1:2d01)")
    if require_accessory is False and state.accessory:
        raise SelectionError(f"Pixel is already in accessory mode ({state.vendor_product})")


@dataclass(frozen=True)
class AccessoryInterface:
    interface: int
    out_endpoint: int
    in_endpoint: int
    max_packet: int
    has_adb_interface: bool

    def to_json(self) -> dict[str, object]:
        return {"has_adb_interface": self.has_adb_interface, "in_endpoint": self.in_endpoint,
                "interface": self.interface, "max_packet": self.max_packet, "out_endpoint": self.out_endpoint}


def accessory_interface(descriptors: bytes) -> AccessoryInterface:
    """The AOA accessory interface (class/subclass 0xff/0xff, protocol 0) and its two bulk endpoints, from the
    raw descriptors sysfs exposes; the ADB interface (0xff/0x42/0x01) is only recorded, never used."""
    offset, current, found, adb = 0, None, {}, False
    while offset + 2 <= len(descriptors):
        length, kind = descriptors[offset], descriptors[offset + 1]
        if length < 2 or offset + length > len(descriptors):
            raise SelectionError("USB descriptors are malformed")
        part = descriptors[offset:offset + length]
        if kind == 4 and length >= 9:
            number, alternate, klass, subclass, protocol = part[2], part[3], part[5], part[6], part[7]
            adb = adb or (klass, subclass, protocol) == (0xFF, 0x42, 0x01)
            current = number if (alternate == 0 and (klass, subclass, protocol) == (0xFF, 0xFF, 0x00)) else None
            if current is not None:
                if current in found:
                    raise SelectionError("USB descriptors list the accessory interface twice")
                found[current] = []
        elif kind == 5 and length >= 7 and current is not None and part[3] & 3 == 2:
            found[current].append((part[2], part[4] | part[5] << 8))
        offset += length
    if len(found) != 1:
        raise SelectionError(f"expected one AOA accessory interface, found {len(found)}")
    (interface, endpoints), = found.items()
    incoming = [row for row in endpoints if row[0] & 0x80]
    outgoing = [row for row in endpoints if not row[0] & 0x80]
    if len(incoming) != 1 or len(outgoing) != 1:
        raise SelectionError("the accessory interface does not have exactly one bulk IN and one bulk OUT endpoint")
    return AccessoryInterface(interface, outgoing[0][0], incoming[0][0], max(incoming[0][1], outgoing[0][1]), adb)


def lpm_state(sysfs_device: str, *, sysfs_root: Path = USB_SYSFS_ROOT) -> dict[str, str]:
    """USB3 link power management as the host sees it (read-only; absent attributes are omitted)."""
    device = sysfs_root / sysfs_device
    result = {}
    for name, path in (("usb3_hardware_lpm_u1", device / "power" / "usb3_hardware_lpm_u1"),
                       ("usb3_hardware_lpm_u2", device / "power" / "usb3_hardware_lpm_u2"),
                       ("usb3_lpm_permit", device / "port" / "usb3_lpm_permit"),
                       ("power_control", device / "power" / "control"),
                       ("runtime_status", device / "power" / "runtime_status")):
        try:
            result[name] = path.read_text(encoding="ascii").strip()
        except OSError:
            continue
    return result


# --- links -----------------------------------------------------------------------------------------

class UsbLink:
    """One bulk OUT / bulk IN pair. ``read`` returns b"" on timeout."""

    description: dict

    def write(self, data: bytes, timeout_ms: int) -> None:
        raise NotImplementedError

    def read(self, timeout_ms: int) -> bytes:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class SocketLink(UsbLink):
    """A SOCK_SEQPACKET peer standing in for the accessory endpoints (host loopback tests); each OUT message is
    at most ``chunk`` bytes, like one 16 KiB f_accessory read request."""

    def __init__(self, sock: socket.socket, *, chunk: int = 16384) -> None:
        self.sock, self.chunk = sock, chunk
        self.description = {"kind": "socket", "chunk": chunk}
        self._closed = False

    def write(self, data: bytes, timeout_ms: int) -> None:
        view = memoryview(data)
        try:
            self.sock.settimeout(timeout_ms / 1000)
            for start in range(0, len(view), self.chunk):
                self.sock.send(view[start:start + self.chunk])
        except socket.timeout as error:
            raise RelayUnresponsive("relay stopped reading (OUT timeout)") from error
        except OSError as error:
            raise LinkLost("loopback link lost: " + str(error)) from error

    def read(self, timeout_ms: int) -> bytes:
        try:
            ready, _, _ = select.select([self.sock], [], [], timeout_ms / 1000)
            if not ready:
                return b""
            data = self.sock.recv(IN_TRANSFER_BYTES)
        except OSError as error:
            if self._closed:
                return b""
            raise LinkLost("loopback link lost: " + str(error)) from error
        if not data:
            raise LinkLost("loopback link closed")
        return data

    def close(self) -> None:
        self._closed = True
        try:
            self.sock.close()
        except OSError:
            pass


class _DeviceDescriptor(ctypes.Structure):
    _fields_ = [("bLength", ctypes.c_uint8), ("bDescriptorType", ctypes.c_uint8), ("bcdUSB", ctypes.c_uint16),
                ("bDeviceClass", ctypes.c_uint8), ("bDeviceSubClass", ctypes.c_uint8),
                ("bDeviceProtocol", ctypes.c_uint8), ("bMaxPacketSize0", ctypes.c_uint8),
                ("idVendor", ctypes.c_uint16), ("idProduct", ctypes.c_uint16), ("bcdDevice", ctypes.c_uint16),
                ("iManufacturer", ctypes.c_uint8), ("iProduct", ctypes.c_uint8), ("iSerialNumber", ctypes.c_uint8),
                ("bNumConfigurations", ctypes.c_uint8)]


LIBUSB_ERROR_TIMEOUT, LIBUSB_ERROR_NO_DEVICE, LIBUSB_ERROR_NOT_FOUND = -7, -4, -5


def load_libusb(path: str | None = None):
    name = path or ctypes.util.find_library("usb-1.0")
    if not name:
        raise SelectionError("libusb-1.0 is not installed")
    lib = ctypes.CDLL(name)
    vp, c_int, c_ubyte = ctypes.c_void_p, ctypes.c_int, ctypes.c_ubyte
    signatures = {
        "libusb_init": ([ctypes.POINTER(vp)], c_int), "libusb_exit": ([vp], None),
        "libusb_get_device_list": ([vp, ctypes.POINTER(ctypes.POINTER(vp))], ctypes.c_ssize_t),
        "libusb_free_device_list": ([ctypes.POINTER(vp), c_int], None),
        "libusb_get_bus_number": ([vp], ctypes.c_uint8), "libusb_get_device_address": ([vp], ctypes.c_uint8),
        "libusb_get_port_numbers": ([vp, ctypes.POINTER(ctypes.c_uint8), c_int], c_int),
        "libusb_get_device_descriptor": ([vp, ctypes.POINTER(_DeviceDescriptor)], c_int),
        "libusb_open": ([vp, ctypes.POINTER(vp)], c_int), "libusb_close": ([vp], None),
        "libusb_get_string_descriptor_ascii": ([vp, ctypes.c_uint8, ctypes.POINTER(c_ubyte), c_int], c_int),
        "libusb_kernel_driver_active": ([vp, c_int], c_int), "libusb_detach_kernel_driver": ([vp, c_int], c_int),
        "libusb_claim_interface": ([vp, c_int], c_int), "libusb_release_interface": ([vp, c_int], c_int),
        "libusb_bulk_transfer": ([vp, c_ubyte, ctypes.POINTER(c_ubyte), c_int, ctypes.POINTER(c_int),
                                  ctypes.c_uint], c_int),
        "libusb_control_transfer": ([vp, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint16,
                                     ctypes.POINTER(c_ubyte), ctypes.c_uint16, ctypes.c_uint], c_int),
        "libusb_reset_device": ([vp], c_int), "libusb_error_name": ([c_int], ctypes.c_char_p),
    }
    for function, (arguments, result) in signatures.items():
        getattr(lib, function).argtypes = arguments
        getattr(lib, function).restype = result
    return lib


class _LibusbDevice:
    """libusb handle of exactly the pinned device: bus + address + port path from sysfs, then VID:PID and the
    serial string read from the opened device, then sysfs again (the address must not have changed)."""

    def __init__(self, state: UsbDeviceState, *, library=None, sysfs_root: Path = USB_SYSFS_ROOT) -> None:
        self.lib = library or load_libusb()
        self.state = state
        self.context = ctypes.c_void_p()
        if self.lib.libusb_init(ctypes.byref(self.context)) != 0:
            raise SelectionError("libusb_init failed")
        self.handle = ctypes.c_void_p()
        try:
            self._open(sysfs_root)
        except BaseException:
            self.close()
            raise

    def error(self, code: int) -> str:
        name = self.lib.libusb_error_name(code)
        return (name or b"?").decode("ascii", "replace") + f" ({code})"

    def _open(self, sysfs_root: Path) -> None:
        devices = ctypes.POINTER(ctypes.c_void_p)()
        count = self.lib.libusb_get_device_list(self.context, ctypes.byref(devices))
        if count < 0:
            raise SelectionError("libusb_get_device_list failed")
        try:
            matches = []
            for index in range(count):
                device = devices[index]
                if (self.lib.libusb_get_bus_number(device) != self.state.busnum
                        or self.lib.libusb_get_device_address(device) != self.state.devnum):
                    continue
                ports = (ctypes.c_uint8 * 8)()
                depth = self.lib.libusb_get_port_numbers(device, ports, 8)
                if tuple(ports[:max(depth, 0)]) != self.state.port_numbers:
                    raise SelectionError("libusb port path differs from sysfs " + self.state.sysfs_device)
                matches.append(device)
            if len(matches) != 1:
                raise SelectionError(f"no unique libusb device at {self.state.sysfs_device}")
            descriptor = _DeviceDescriptor()
            if self.lib.libusb_get_device_descriptor(matches[0], ctypes.byref(descriptor)) != 0 or \
                    f"{descriptor.idVendor:04x}:{descriptor.idProduct:04x}" != self.state.vendor_product:
                raise SelectionError("libusb VID:PID differs from sysfs " + self.state.sysfs_device)
            code = self.lib.libusb_open(matches[0], ctypes.byref(self.handle))
            if code != 0:
                raise SelectionError("cannot open the Pixel: " + self.error(code))
        finally:
            self.lib.libusb_free_device_list(devices, 1)
        text = (ctypes.c_ubyte * 128)()
        size = self.lib.libusb_get_string_descriptor_ascii(self.handle, descriptor.iSerialNumber, text, 128)
        if size < 0 or bytes(text[:size]).decode("ascii", "replace") != self.state.serial:
            raise SelectionError("the opened device's serial string differs from the pinned Pixel")
        after = read_usb_device(self.state.sysfs_device, sysfs_root=sysfs_root)
        if after != self.state:
            raise SelectionError("the Pixel re-enumerated while it was being opened")

    def close(self) -> None:
        if self.handle:
            self.lib.libusb_close(self.handle)
            self.handle = ctypes.c_void_p()
        if self.context:
            self.lib.libusb_exit(self.context)
            self.context = ctypes.c_void_p()


class LibusbLink(UsbLink):
    """The accessory interface's bulk endpoints of the pinned Pixel (accessory+adb mode only)."""

    def __init__(self, sysfs_device: str, serial: str, *, forbidden_serials: Iterable[str] = DEFAULT_FORBIDDEN_SERIALS,
                 sysfs_root: Path = USB_SYSFS_ROOT, library=None) -> None:
        state = read_usb_device(sysfs_device, sysfs_root=sysfs_root)
        check_pixel(state, serial=serial, forbidden_serials=forbidden_serials, require_accessory=True)
        interface = accessory_interface((sysfs_root / sysfs_device / "descriptors").read_bytes())
        self.device = _LibusbDevice(state, library=library, sysfs_root=sysfs_root)
        self.lib, self.handle, self.interface = self.device.lib, self.device.handle, interface
        try:
            if self.lib.libusb_kernel_driver_active(self.handle, interface.interface) == 1:
                self.lib.libusb_detach_kernel_driver(self.handle, interface.interface)
            code = self.lib.libusb_claim_interface(self.handle, interface.interface)
            if code != 0:
                raise SelectionError("cannot claim the accessory interface: " + self.device.error(code))
        except BaseException:
            self.device.close()
            raise
        self._in_buffer = (ctypes.c_ubyte * IN_TRANSFER_BYTES)()
        self.description = {"kind": "libusb", "device": state.to_json(), "interface": interface.to_json(),
                            "lpm": lpm_state(sysfs_device, sysfs_root=sysfs_root)}

    def write(self, data: bytes, timeout_ms: int) -> None:
        buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        done = ctypes.c_int(0)
        code = self.lib.libusb_bulk_transfer(self.handle, self.interface.out_endpoint, buffer, len(data),
                                             ctypes.byref(done), timeout_ms)
        if code == LIBUSB_ERROR_TIMEOUT:
            raise RelayUnresponsive(f"bulk OUT timed out after {done.value}/{len(data)} bytes")
        if code != 0 or done.value != len(data):
            raise LinkLost(f"bulk OUT failed: {self.device.error(code)} {done.value}/{len(data)}")

    def read(self, timeout_ms: int) -> bytes:
        done = ctypes.c_int(0)
        code = self.lib.libusb_bulk_transfer(self.handle, self.interface.in_endpoint, self._in_buffer,
                                             IN_TRANSFER_BYTES, ctypes.byref(done), timeout_ms)
        if code not in (0, LIBUSB_ERROR_TIMEOUT):
            raise LinkLost("bulk IN failed: " + self.device.error(code))
        return ctypes.string_at(self._in_buffer, done.value) if done.value else b""

    def close(self) -> None:
        if self.device.handle:
            self.lib.libusb_release_interface(self.handle, self.interface.interface)
        self.device.close()


# --- accessory mode switching ----------------------------------------------------------------------

def wait_for_mode(sysfs_device: str, serial: str, *, accessory: bool, forbidden_serials: Iterable[str],
                  timeout_s: float, sysfs_root: Path = USB_SYSFS_ROOT, sleep: Callable[[float], None] = time.sleep,
                  clock: Callable[[], float] = time.monotonic) -> UsbDeviceState:
    """Poll the pinned port until the pinned Pixel is back in the wanted mode (absent while it re-enumerates)."""
    deadline = clock() + timeout_s
    forbidden = set(forbidden_serials)
    last = "absent"
    while True:
        try:
            state = read_usb_device(sysfs_device, sysfs_root=sysfs_root)
        except SelectionError:
            state = None
        if state is not None:
            last = state.vendor_product + " " + state.serial
            if state.serial in forbidden:
                raise SelectionError(f"a forbidden phone ({state.serial}) appeared at {sysfs_device}")
            if state.serial == serial and state.accessory and state.product != ACCESSORY_ADB_PRODUCT:
                raise AccessoryOnlyError(f"Pixel enumerated as {state.vendor_product} (accessory without adb)")
            if state.serial == serial and state.vendor == GOOGLE_VENDOR and state.accessory == accessory:
                return state
        if clock() > deadline:
            raise SelectionError(f"Pixel did not reach {'accessory' if accessory else 'normal'} mode at "
                                 f"{sysfs_device} (last {last})")
        sleep(0.25)


def switch_to_accessory(sysfs_device: str, serial: str, *, forbidden_serials: Iterable[str] = DEFAULT_FORBIDDEN_SERIALS,
                        sysfs_root: Path = USB_SYSFS_ROOT, library=None, timeout_s: float = 40.0,
                        strings: Sequence[str] = AOA_STRINGS) -> dict[str, object]:
    """Send the AOA start sequence to the pinned Pixel only; wait until it is accessory+adb at the same port."""
    before = read_usb_device(sysfs_device, sysfs_root=sysfs_root)
    check_pixel(before, serial=serial, forbidden_serials=forbidden_serials, require_accessory=None)
    if before.product == ACCESSORY_ADB_PRODUCT:
        return {"already_accessory": True, "before": before.to_json(), "after": before.to_json()}
    device = _LibusbDevice(before, library=library, sysfs_root=sysfs_root)
    try:
        version = (ctypes.c_ubyte * 2)()
        code = device.lib.libusb_control_transfer(device.handle, 0xC0, 51, 0, 0, version, 2, 2000)
        if code != 2 or version[0] | version[1] << 8 < 1:
            raise SelectionError("AOA GET_PROTOCOL failed: " + str(code))
        for index, text in enumerate(strings):
            data = text.encode("ascii") + b"\0"
            buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
            code = device.lib.libusb_control_transfer(device.handle, 0x40, 52, 0, index, buffer, len(data), 2000)
            if code != len(data):
                raise SelectionError(f"AOA SEND_STRING {index} failed: {code}")
        code = device.lib.libusb_control_transfer(device.handle, 0x40, 53, 0, 0, None, 0, 2000)
        if code not in (0, -1, LIBUSB_ERROR_NO_DEVICE):  # the device may drop off while acknowledging START
            raise SelectionError("AOA START failed: " + str(code))
    finally:
        device.close()
    try:
        after = wait_for_mode(sysfs_device, serial, accessory=True, forbidden_serials=forbidden_serials,
                              timeout_s=timeout_s, sysfs_root=sysfs_root)
    except AccessoryOnlyError:
        # never leave an accessory-only Pixel next to the OP15 gadget: reset it straight back
        restore_normal(sysfs_device, serial, forbidden_serials=forbidden_serials, sysfs_root=sysfs_root,
                       library=library)
        raise
    return {"already_accessory": False, "aoa_protocol": version[0] | version[1] << 8, "before": before.to_json(),
            "after": after.to_json()}


def restore_normal(sysfs_device: str, serial: str, *, forbidden_serials: Iterable[str] = DEFAULT_FORBIDDEN_SERIALS,
                   sysfs_root: Path = USB_SYSFS_ROOT, library=None, timeout_s: float = 40.0,
                   attempts: int = 2) -> dict[str, object]:
    """USB-reset the pinned Pixel while it is in any accessory mode; Android leaves accessory mode on the
    disconnect and re-enumerates it normally (the prototype saw libusb return -5 while it re-enumerated).

    On the rig (2026-10-01) one reset left the Pixel in accessory+adb mode for the whole wait and a second reset
    restored it within a second, so up to ``attempts`` resets are issued, each after re-checking that the port
    still holds the pinned Pixel in accessory mode; ``timeout_s`` is split evenly between them."""
    before = read_usb_device(sysfs_device, sysfs_root=sysfs_root)
    if before.serial in set(forbidden_serials) or before.serial != serial or before.vendor != GOOGLE_VENDOR:
        raise SelectionError(f"USB device {sysfs_device} is not the pinned Pixel")
    if not before.accessory:
        return {"already_normal": True, "before": before.to_json(), "after": before.to_json()}
    attempts = max(1, int(attempts))
    per_attempt = timeout_s / attempts
    codes: list[int] = []
    state = before
    for attempt in range(attempts):
        if attempt:
            state = read_usb_device(sysfs_device, sysfs_root=sysfs_root)
            if state.serial in set(forbidden_serials) or state.serial != serial or state.vendor != GOOGLE_VENDOR:
                raise SelectionError(f"USB device {sysfs_device} is not the pinned Pixel")
            if not state.accessory:
                return {"already_normal": False, "reset_return_code": codes[-1], "reset_return_codes": codes,
                        "attempts": attempt, "before": before.to_json(), "after": state.to_json()}
        device = _LibusbDevice(state, library=library, sysfs_root=sysfs_root)
        try:
            code = device.lib.libusb_reset_device(device.handle)
        finally:
            device.close()
        codes.append(code)
        if code not in (0, -1, LIBUSB_ERROR_NO_DEVICE, LIBUSB_ERROR_NOT_FOUND):
            raise SelectionError("USB reset of the Pixel failed: " + str(code))
        try:
            after = wait_for_mode(sysfs_device, serial, accessory=False, forbidden_serials=forbidden_serials,
                                  timeout_s=per_attempt, sysfs_root=sysfs_root)
        except AccessoryOnlyError:
            raise
        except SelectionError as error:
            if "did not reach" not in str(error) or attempt + 1 >= attempts:
                raise
            continue
        result = {"already_normal": False, "reset_return_code": code, "before": before.to_json(),
                  "after": after.to_json()}
        if attempt:
            result.update({"reset_return_codes": codes, "attempts": attempt + 1})
        return result
    raise SelectionError(f"Pixel did not reach normal mode at {sysfs_device}")


# --- bridge --------------------------------------------------------------------------------------

@dataclass
class BridgeOptions:
    listen_host: str = "127.0.0.1"
    listen_port: int = 0
    echo_port: int = 0                 # connections here become relay-local echo streams (hop benchmark)
    keepalive_ms: float = 0.0          # >0: NOP every keepalive_ms while traffic is recent (USB link out of U2)
    keepalive_window_ms: float = 1500.0
    ping_interval_ms: float = 1000.0   # liveness PING after this much inbound silence
    ping_timeout_ms: float = 5000.0
    handshake_timeout_ms: float = 5000.0
    handshake_attempts: int = 3
    write_timeout_ms: int = 10000
    status_path: Path | None = None
    status_interval_ms: float = 2000.0
    ready_path: Path | None = None
    trace_path: Path | None = None
    trace_limit: int = 200000
    shutdown_relay_on_exit: bool = False


class _Stream:
    def __init__(self, stream_id: int, sock: socket.socket, echo: bool) -> None:
        self.id, self.sock, self.echo = stream_id, sock, echo
        self.closed = False
        self.bytes_up = 0
        self.bytes_down = 0


class Bridge:
    """Carries TCP connections accepted on ``options.listen_port`` over one S43A session."""

    def __init__(self, link: UsbLink, options: BridgeOptions, *, session: int | None = None,
                 clock: Callable[[], int] = time.monotonic_ns) -> None:
        self.link, self.options, self.clock = link, options, clock
        self.session = session if session is not None else random.SystemRandom().randrange(1, 1 << 32)
        self.parser = FrameParser()
        self._out_lock = threading.Lock()
        self._link_closed = False
        self._streams: dict[int, _Stream] = {}
        self._streams_lock = threading.Lock()
        self._next_stream = 1
        self._stop = threading.Event()
        self._exit_code = EXIT_OK
        self._exit_reason = "running"
        self._threads: list[threading.Thread] = []
        self._listeners: list[tuple[socket.socket, bool]] = []
        self._ports = (0, 0)
        self._last_inbound_ns = 0
        self._last_activity_ns = 0
        self._ping_outstanding: tuple[int, int] | None = None
        self._ping_id = 0
        self.hello: dict[str, object] = {}
        self.pong_rtt_us: collections.deque = collections.deque(maxlen=512)
        self.trace: collections.deque = collections.deque(maxlen=max(1, options.trace_limit))
        self.counters = collections.Counter()
        self.started_ns = clock()

    # -- link I/O
    def _send(self, kind: int, stream: int = 0, payload: bytes = b"", aux: int = 0) -> None:
        frame = pack_frame(kind, self.session, stream, payload, aux=aux)
        with self._out_lock:
            if self._link_closed:
                raise LinkLost("the bridge closed the link")
            self.link.write(frame, self.options.write_timeout_ms)
        self.counters["frames_out"] += 1
        self.counters["bytes_out"] += len(frame)
        if kind == DATA:
            self.counters["data_out"] += len(payload)
            self._last_activity_ns = self.clock()
        if self.options.trace_path is not None:
            self.trace.append((self.clock(), "out", kind, stream, len(payload), 0, aux))

    def handshake(self) -> dict[str, object]:
        """Drain stale IN bytes, then HELLO until the relay answers HELLO_ACK in this session."""
        for _ in range(256):
            if not self.link.read(20):
                break
            self.counters["stale_in_transfers"] += 1
        for attempt in range(1, self.options.handshake_attempts + 1):
            self.parser.expect(HELLO_ACK, self.session)
            sent_ns = self.clock()
            self._send(HELLO, 0, HELLO_PAYLOAD.pack(os.getpid() & 0xFFFFFFFF, 0))
            deadline = sent_ns + int(self.options.handshake_timeout_ms * 1e6)
            while self.clock() < deadline:
                data = self.link.read(max(1, int((deadline - self.clock()) / 1e6)))
                if not data:
                    continue
                for frame in self.parser.feed(data):
                    if frame.kind == HELLO_ACK and len(frame.payload) >= HELLO_ACK_PAYLOAD.size:
                        version, pid, streams, options = HELLO_ACK_PAYLOAD.unpack_from(frame.payload)
                        now = self.clock()
                        self._last_inbound_ns = now
                        self.hello = {"attempt": attempt, "session": self.session, "relay_version": version,
                                      "relay_pid": pid, "relay_max_streams": streams, "relay_options": options,
                                      "rtt_us": (now - sent_ns) / 1000,
                                      "discarded_bytes": self.parser.discarded_bytes}
                        return self.hello
            self.counters["handshake_timeouts"] += 1
        raise RelayUnresponsive("the relay did not answer HELLO")

    # -- TCP side
    def listen(self) -> None:
        for port, echo in ((self.options.listen_port, False), (self.options.echo_port, True)):
            if not port and echo:
                continue
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                listener.bind((self.options.listen_host, port))
                listener.listen(8)
            except OSError as error:
                listener.close()
                exception = BridgeError(f"cannot listen on {self.options.listen_host}:{port}: {error}")
                exception.exit_code = EXIT_LISTEN
                raise exception from error
            self._listeners.append((listener, echo))
        self._ports = (self._listeners[0][0].getsockname()[1],
                       next((sock.getsockname()[1] for sock, echo in self._listeners if echo), 0))

    @property
    def listen_port(self) -> int:
        return self._ports[0]

    @property
    def echo_listen_port(self) -> int:
        return self._ports[1]

    def _accept_loop(self) -> None:
        selector = selectors.DefaultSelector()
        for listener, echo in self._listeners:
            selector.register(listener, selectors.EVENT_READ, echo)
        try:
            while not self._stop.is_set():
                for key, _ in selector.select(timeout=0.2):
                    try:
                        client, _ = key.fileobj.accept()
                    except OSError:
                        continue
                    self._open_stream(client, key.data)
        finally:
            selector.close()

    def _open_stream(self, client: socket.socket, echo: bool) -> None:
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with self._streams_lock:
            stream_id = self._next_stream
            self._next_stream = self._next_stream % 0x7FFFFFFF + 1
            stream = _Stream(stream_id, client, echo)
            self._streams[stream_id] = stream
        self.counters["streams_opened"] += 1
        try:
            self._send(OPEN, stream_id, U32.pack(OPEN_LOCAL_ECHO if echo else 0))
        except BridgeError as error:
            self._fail(error)
            return
        thread = threading.Thread(target=self._upstream, args=(stream,), name=f"s43a-up-{stream_id}", daemon=True)
        thread.start()

    def _upstream(self, stream: _Stream) -> None:
        reason = CLOSE_EOF
        try:
            while not self._stop.is_set():
                data = stream.sock.recv(TCP_CHUNK)
                if not data:
                    break
                stream.bytes_up += len(data)
                self._send(DATA, stream.id, data)
        except BridgeError as error:
            self._fail(error)
            return
        except OSError:
            reason = CLOSE_IO
        if self._retire(stream) and not self._stop.is_set():
            try:
                self._send(CLOSE, stream.id, U32.pack(reason))
            except BridgeError as error:
                self._fail(error)

    def _retire(self, stream: _Stream) -> bool:
        """Close the TCP side once; True for the caller that closed it."""
        with self._streams_lock:
            if stream.closed:
                return False
            stream.closed = True
            self._streams.pop(stream.id, None)
        self.counters["streams_closed"] += 1
        for action in (lambda: stream.sock.shutdown(socket.SHUT_RDWR), stream.sock.close):
            try:
                action()
            except OSError:
                pass
        return True

    # -- USB IN side
    def _downstream(self) -> None:
        while not self._stop.is_set():
            try:
                data = self.link.read(200)
                if not data:
                    continue
                now = self.clock()
                self._last_inbound_ns = now
                for frame in self.parser.feed(data):
                    self._handle(frame, now)
            except BridgeError as error:
                self._fail(error)
                return

    def _handle(self, frame: Frame, now: int) -> None:
        self.counters["frames_in"] += 1
        self.counters["bytes_in"] += HEADER_BYTES + len(frame.payload) + pad_for(len(frame.payload))
        if self.options.trace_path is not None:
            self.trace.append((now, "in", frame.kind, frame.stream, len(frame.payload), frame.stamp_ns, frame.aux))
        if frame.kind == DATA:
            self.counters["data_in"] += len(frame.payload)
            self._last_activity_ns = now
            with self._streams_lock:
                stream = self._streams.get(frame.stream)
            if stream is None:
                self.counters["dropped_data_frames"] += 1
                return
            try:
                stream.sock.sendall(frame.payload)
                stream.bytes_down += len(frame.payload)
            except OSError:
                if self._retire(stream):
                    self._send(CLOSE, stream.id, U32.pack(CLOSE_IO))
        elif frame.kind == CLOSE:
            with self._streams_lock:
                stream = self._streams.get(frame.stream)
            if stream is not None:
                reason = U32.unpack_from(frame.payload)[0] if len(frame.payload) >= 4 else -1
                self.counters[f"relay_close_reason_{reason}"] += 1
                self._retire(stream)
        elif frame.kind == PONG and len(frame.payload) >= PONG_PAYLOAD.size:
            ping_id, relay_rx, relay_tx = PONG_PAYLOAD.unpack_from(frame.payload)
            outstanding = self._ping_outstanding
            if outstanding is not None and outstanding[0] == ping_id:
                self._ping_outstanding = None
                self.pong_rtt_us.append((now - outstanding[1]) / 1000)
            self.counters["pongs"] += 1
        elif frame.kind == OPEN_ACK:
            status = I32.unpack_from(frame.payload)[0] if len(frame.payload) >= 4 else -1
            self.counters["open_ack_ok" if status == 0 else "open_ack_failed"] += 1
        elif frame.kind == ERROR:
            self.counters["relay_errors"] += 1
            self.last_relay_error = frame.payload[4:].decode("ascii", "replace")
        else:
            self.counters["unexpected_frames"] += 1

    # -- timers: keep-alive, liveness, status
    def _timers(self) -> None:
        options = self.options
        next_status = self.clock()
        keepalive_s = options.keepalive_ms / 1000
        window_ns = options.keepalive_window_ms * 1e6
        idle_s = max(0.01, min(0.1, options.ping_interval_ms / 4000))  # liveness/status granularity when idle
        wait_s = idle_s
        while not self._stop.wait(wait_s):
            now = self.clock()
            wait_s = idle_s
            try:
                if keepalive_s > 0 and now - self._last_activity_ns <= window_ns:
                    self._send(NOP)
                    self.counters["nops"] += 1
                    wait_s = keepalive_s  # tight ticks only inside the keep-alive window
                outstanding = self._ping_outstanding
                if outstanding is not None:
                    if now - outstanding[1] > options.ping_timeout_ms * 1e6:
                        raise RelayUnresponsive(f"no PONG within {options.ping_timeout_ms:.0f} ms")
                elif now - self._last_inbound_ns > options.ping_interval_ms * 1e6:
                    self._ping_id += 1
                    self._ping_outstanding = (self._ping_id, self.clock())
                    self._send(PING, 0, PING_PAYLOAD.pack(self._ping_id, PONG_PAYLOAD.size, 0), aux=self._ping_id)
                    self.counters["pings"] += 1
            except BridgeError as error:
                self._fail(error)
                return
            if options.status_path is not None and now >= next_status:
                self.write_status("up")
                next_status = now + int(options.status_interval_ms * 1e6)

    def status(self, state: str) -> dict[str, object]:
        rtts = sorted(self.pong_rtt_us)
        return {
            "schema": STATUS_SCHEMA, "state": state, "pid": os.getpid(), "session": self.session,
            "listen_port": self.listen_port, "echo_port": self.echo_listen_port,
            "link": getattr(self.link, "description", {}), "hello": self.hello,
            "counters": dict(sorted(self.counters.items())),
            "open_streams": len(self._streams),
            "pong_rtt_us_p50": rtts[len(rtts) // 2] if rtts else None,
            "discarded_bytes": self.parser.discarded_bytes, "stale_frames": self.parser.stale_frames,
            "last_relay_error": getattr(self, "last_relay_error", None),
            "exit_code": self._exit_code if state != "up" else None, "exit_reason": self._exit_reason,
            "options": {"keepalive_ms": self.options.keepalive_ms,
                        "keepalive_window_ms": self.options.keepalive_window_ms,
                        "ping_interval_ms": self.options.ping_interval_ms,
                        "ping_timeout_ms": self.options.ping_timeout_ms},
            "updated_monotonic_ns": self.clock(), "updated_epoch_s": time.time(),
            "process_cpu_s": time.process_time(),  # host CPU cost of the bridge (keep-alive included)
        }

    def write_status(self, state: str) -> None:
        path = self.options.status_path
        if path is None:
            return
        temporary = path.with_name(path.name + ".tmp")
        try:
            temporary.write_text(json.dumps(self.status(state), sort_keys=True) + "\n")
            os.replace(temporary, path)
        except OSError:
            pass

    # -- lifecycle
    def _fail(self, error: BridgeError) -> None:
        if not self._stop.is_set():
            self._exit_code = getattr(error, "exit_code", EXIT_PROTOCOL)
            self._exit_reason = type(error).__name__ + ": " + str(error)
            self._stop.set()

    def request_stop(self, reason: str = "stop requested") -> None:
        if not self._stop.is_set():
            self._exit_reason = reason
            self._stop.set()

    def start(self) -> None:
        """Handshake done by the caller; start the threads and publish readiness."""
        if not self._listeners:
            self.listen()
        self._last_inbound_ns = self._last_activity_ns = self.clock()
        for target, name in ((self._downstream, "s43a-in"), (self._accept_loop, "s43a-accept"),
                             (self._timers, "s43a-timers")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        self.write_status("up")
        if self.options.ready_path is not None:
            self.options.ready_path.write_text(json.dumps({
                "listen_port": self.listen_port, "echo_port": self.echo_listen_port, "pid": os.getpid(),
                "session": self.session, "hello": self.hello, "link": getattr(self.link, "description", {}),
            }, sort_keys=True) + "\n")

    def wait(self, timeout_s: float | None = None) -> bool:
        return self._stop.wait(timeout_s)

    def finish(self) -> int:
        """Stop the threads, close every TCP client (CLOSE to the relay while the link lives) and the link."""
        self._stop.set()
        for listener, _ in self._listeners:
            try:
                listener.close()
            except OSError:
                pass
        with self._streams_lock:
            streams = list(self._streams.values())
        link_alive = self._exit_code in (EXIT_OK,)
        for stream in streams:
            if self._retire(stream) and link_alive:
                try:
                    self._send(CLOSE, stream.id, U32.pack(CLOSE_SHUTDOWN))
                except BridgeError:
                    link_alive = False
        if link_alive and self.options.shutdown_relay_on_exit:
            try:
                self._send(SHUTDOWN)
            except BridgeError:
                pass
        joined = True
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=5.0)
                joined = joined and not thread.is_alive()
        with self._out_lock:
            self._link_closed = True
        if joined:  # never free a libusb handle another thread may still be using
            self.link.close()
        if self._exit_reason == "running":
            self._exit_reason = "stopped"
        self.write_status("down")
        if self.options.trace_path is not None:
            try:
                with self.options.trace_path.open("w") as out:
                    for row in self.trace:
                        out.write(json.dumps(dict(zip(("host_ns", "dir", "kind", "stream", "bytes", "peer_stamp_ns",
                                                       "aux"), row))) + "\n")
            except OSError:
                pass
        return self._exit_code


# --- ping benchmark (transport only: bridge <-> relay, no TCP hop) --------------------------------

def ping_benchmark(link: UsbLink, *, count: int, request_bytes: int, response_bytes: int, gap_ms: float,
                   token_gap_ms: tuple[float, float] | None, calls_per_token: int, warmup: int, seed: int = 1,
                   keepalive_ms: float = 0.0, clock: Callable[[], int] = time.monotonic_ns,
                   sleep: Callable[[float], None] = time.sleep) -> dict[str, object]:
    """PING/PONG at the serving cadence; per call t1 (host send), relay rx/tx (phone clock), t4 (host receive).
    link-side time = rtt - (relay_tx - relay_rx) needs no clock synchronisation."""
    bridge = Bridge(link, BridgeOptions(), clock=clock)
    hello = bridge.handshake()
    rng = random.Random(seed)
    filler = b"\xa5" * max(0, request_bytes - PING_PAYLOAD.size)
    rows = []
    stop_keepalive = threading.Event()

    def keepalive() -> None:
        while not stop_keepalive.wait(keepalive_ms / 1000):
            bridge._send(NOP)

    helper = None
    if keepalive_ms > 0:
        helper = threading.Thread(target=keepalive, daemon=True)
        helper.start()
    try:
        for index in range(1, count + warmup + 1):
            t1 = clock()
            bridge._send(PING, 0, PING_PAYLOAD.pack(index, max(response_bytes, PONG_PAYLOAD.size), 0) + filler,
                         aux=index)
            pong = None
            while pong is None:
                data = link.read(5000)
                if not data:
                    raise RelayUnresponsive(f"no PONG for ping {index}")
                t4 = clock()
                for frame in bridge.parser.feed(data):
                    if frame.kind == PONG and PONG_PAYLOAD.unpack_from(frame.payload)[0] == index:
                        pong = frame
            _, relay_rx, relay_tx = PONG_PAYLOAD.unpack_from(pong.payload)
            rtt_us = (t4 - t1) / 1000
            rows.append({"index": index, "warmup": index <= warmup, "position": (index - 1) % max(1, calls_per_token),
                         "rtt_us": rtt_us, "relay_us": (relay_tx - relay_rx) / 1000,
                         "link_us": rtt_us - (relay_tx - relay_rx) / 1000, "t1_ns": t1, "t4_ns": t4,
                         "relay_rx_ns": relay_rx, "relay_tx_ns": relay_tx, "response_bytes": len(pong.payload)})
            last_of_token = token_gap_ms is not None and index % calls_per_token == 0
            pause = rng.uniform(*token_gap_ms) if last_of_token else gap_ms
            if pause > 0:
                sleep(pause / 1000)
    finally:
        stop_keepalive.set()
        if helper is not None:
            helper.join(timeout=1)
    measured = sorted(row["rtt_us"] for row in rows if not row["warmup"])
    links = sorted(row["link_us"] for row in rows if not row["warmup"])

    def quantile(values, q):
        return values[min(len(values) - 1, int(len(values) * q))] if values else None
    return {"hello": hello, "calls": rows,
            "summary": {"count": len(measured), "rtt_us_p50": quantile(measured, 0.5),
                        "rtt_us_p90": quantile(measured, 0.9), "link_us_p50": quantile(links, 0.5),
                        "link_us_p90": quantile(links, 0.9)}}


# --- CLI -----------------------------------------------------------------------------------------

def _token_gap(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    low, _, high = text.partition(":")
    return float(low), float(high or low)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("inspect", "switch", "restore", "serve", "ping", "lpm"))
    parser.add_argument("--sysfs-device", required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--forbid-serial", action="append", default=None,
                        help="refuse this serial anywhere (default: the OP15 ADB + gadget serials)")
    parser.add_argument("--sysfs-root", type=Path, default=USB_SYSFS_ROOT)
    parser.add_argument("--libusb", default=None)
    parser.add_argument("--link-fd", type=int, default=None, help="tests: SOCK_SEQPACKET fd instead of libusb")
    parser.add_argument("--link-unix", default=None, help="tests: connect a UNIX SOCK_SEQPACKET path instead of libusb")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=0)
    parser.add_argument("--echo-port", type=int, default=0)
    parser.add_argument("--keepalive-ms", type=float, default=0.0)
    parser.add_argument("--keepalive-window-ms", type=float, default=1500.0)
    parser.add_argument("--ping-interval-ms", type=float, default=1000.0)
    parser.add_argument("--ping-timeout-ms", type=float, default=5000.0)
    parser.add_argument("--handshake-timeout-ms", type=float, default=5000.0)
    parser.add_argument("--status-file", type=Path, default=None)
    parser.add_argument("--status-interval-ms", type=float, default=2000.0)
    parser.add_argument("--ready-file", type=Path, default=None)
    parser.add_argument("--trace-file", type=Path, default=None)
    parser.add_argument("--shutdown-relay-on-exit", action="store_true")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--request-bytes", type=int, default=10280)
    parser.add_argument("--response-bytes", type=int, default=10288)
    parser.add_argument("--gap-ms", type=float, default=0.0)
    parser.add_argument("--token-gap-ms", default=None, help="LOW:HIGH ms after every --calls-per-token calls")
    parser.add_argument("--calls-per-token", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _open_link(args) -> UsbLink:
    if args.link_fd is not None:
        return SocketLink(socket.socket(fileno=args.link_fd))
    if args.link_unix is not None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        sock.connect(args.link_unix)
        return SocketLink(sock)
    return LibusbLink(args.sysfs_device, args.serial, forbidden_serials=args.forbid_serial,
                      sysfs_root=args.sysfs_root, library=load_libusb(args.libusb) if args.libusb else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.forbid_serial is None:
        args.forbid_serial = list(DEFAULT_FORBIDDEN_SERIALS)
    if args.serial in args.forbid_serial:
        print("the pinned serial is also forbidden", file=sys.stderr)
        return EXIT_CONFIG
    try:
        if args.command in ("inspect", "lpm"):
            state = read_usb_device(args.sysfs_device, sysfs_root=args.sysfs_root)
            check_pixel(state, serial=args.serial, forbidden_serials=args.forbid_serial, require_accessory=None)
            result: dict[str, object] = {"device": state.to_json(),
                                         "lpm": lpm_state(args.sysfs_device, sysfs_root=args.sysfs_root)}
            if args.command == "inspect" and state.accessory:
                result["interface"] = accessory_interface(
                    (args.sysfs_root / args.sysfs_device / "descriptors").read_bytes()).to_json()
            print(json.dumps(result, indent=2, sort_keys=True))
            return EXIT_OK
        if args.command == "switch":
            print(json.dumps(switch_to_accessory(args.sysfs_device, args.serial, forbidden_serials=args.forbid_serial,
                                                 sysfs_root=args.sysfs_root), indent=2, sort_keys=True))
            return EXIT_OK
        if args.command == "restore":
            print(json.dumps(restore_normal(args.sysfs_device, args.serial, forbidden_serials=args.forbid_serial,
                                            sysfs_root=args.sysfs_root), indent=2, sort_keys=True))
            return EXIT_OK
        if args.command == "ping":
            link = _open_link(args)
            try:
                result = ping_benchmark(link, count=args.count, request_bytes=args.request_bytes,
                                        response_bytes=args.response_bytes, gap_ms=args.gap_ms,
                                        token_gap_ms=_token_gap(args.token_gap_ms),
                                        calls_per_token=args.calls_per_token, warmup=args.warmup, seed=args.seed,
                                        keepalive_ms=args.keepalive_ms)
                result["link"] = link.description
            finally:
                link.close()
            text = json.dumps(result, sort_keys=True)
            if args.output is not None:
                args.output.write_text(text + "\n")
            print(json.dumps(result["summary"], sort_keys=True))
            return EXIT_OK
        # serve
        options = BridgeOptions(
            listen_host=args.listen_host, listen_port=args.listen_port, echo_port=args.echo_port,
            keepalive_ms=args.keepalive_ms, keepalive_window_ms=args.keepalive_window_ms,
            ping_interval_ms=args.ping_interval_ms, ping_timeout_ms=args.ping_timeout_ms,
            handshake_timeout_ms=args.handshake_timeout_ms, status_path=args.status_file,
            status_interval_ms=args.status_interval_ms, ready_path=args.ready_file, trace_path=args.trace_file,
            shutdown_relay_on_exit=args.shutdown_relay_on_exit)
        link = _open_link(args)
        bridge = Bridge(link, options)
        try:
            bridge.listen()
            bridge.handshake()
        except BridgeError as error:
            bridge._fail(error)
            code = bridge.finish()
            print("aoa_bridge: " + bridge._exit_reason, file=sys.stderr)
            return code
        signal.signal(signal.SIGTERM, lambda *_: bridge.request_stop("SIGTERM"))
        signal.signal(signal.SIGINT, lambda *_: bridge.request_stop("SIGINT"))
        bridge.start()
        print(f"[s43-aoa-bridge] ready listen_port={bridge.listen_port} echo_port={bridge.echo_listen_port} "
              f"session={bridge.session} relay_pid={bridge.hello.get('relay_pid')}", file=sys.stderr, flush=True)
        while not bridge.wait(0.5):
            pass
        code = bridge.finish()
        print(f"[s43-aoa-bridge] exit code={code} reason={bridge._exit_reason}", file=sys.stderr, flush=True)
        return code
    except BridgeError as error:
        print("aoa_bridge: " + str(error), file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    sys.exit(main())
