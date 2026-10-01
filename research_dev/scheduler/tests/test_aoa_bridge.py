"""WS10 Pixel AOA bridge: S43A framing, strict device selection, accessory switching and the host bridge
against the real phone relay built for this host on a SOCK_SEQPACKET loopback (one message = one USB transfer).

No phone, adb or USB device is touched: sysfs is a temporary tree and libusb a Python fake.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import time
import unittest

from research_dev.scheduler.adapters import aoa_bridge as ab

SCHEDULER = Path(__file__).resolve().parents[1]
RELAY_SOURCE = SCHEDULER / "native" / "aoa_bridge" / "s43_aoa_relay.c"
PROTOCOL_HEADER = SCHEDULER / "native" / "aoa_bridge" / "s43a_protocol.h"
PIXEL_SERIAL = "5A040DLCH004ES"
OP15_SERIAL = "3C15AU002CL00000"
OP15_GADGET_SERIAL = "SCHEDFFN0001"


def build_host_relay(directory: Path) -> Path:
    compiler = shutil.which(os.environ.get("CC", "cc")) or shutil.which("gcc")
    if compiler is None:
        raise unittest.SkipTest("no C compiler for the host relay")
    binary = directory / "s43-aoa-relay-host"
    subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread", "-o", str(binary),
                    str(RELAY_SOURCE)], check=True, capture_output=True, text=True, timeout=120)
    return binary


# --- fake sysfs / libusb ------------------------------------------------------------------------

def descriptors(*, accessory: bool = True, adb: bool = True, accessory_endpoints: int = 2) -> bytes:
    device = bytes([18, 1, 0x20, 0x03, 0, 0, 0, 9, 0xd1, 0x18, 0x01, 0x2d, 0x40, 0x04, 1, 2, 3, 1])
    body = b""
    number = 0
    if accessory:
        body += bytes([9, 4, number, 0, accessory_endpoints, 0xFF, 0xFF, 0x00, 0])
        for address in (0x81, 0x01)[:accessory_endpoints]:
            body += bytes([7, 5, address, 2, 0x00, 0x04, 0]) + bytes([6, 0x30, 0, 0, 0, 0])
        number += 1
    if adb:
        body += bytes([9, 4, number, 0, 2, 0xFF, 0x42, 0x01, 0])
        for address in (0x82, 0x02):
            body += bytes([7, 5, address, 2, 0x00, 0x04, 0]) + bytes([6, 0x30, 0, 0, 0, 0])
    config = bytes([9, 2]) + struct.pack("<H", 9 + len(body)) + bytes([number, 1, 0, 0x80, 0x32])
    return device + config + body


class FakeSysfs:
    """/sys/bus/usb/devices with the desktop layout: OP15 at 2-2, the Pixel behind the hub at 2-9.2."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.devices: dict[str, dict] = {}

    def set(self, name: str, serial: str, vendor: str, product: str, devnum: int, *, speed: str = "5000",
            descriptor_bytes: bytes | None = None, lpm: dict | None = None) -> None:
        device = self.root / name
        device.mkdir(parents=True, exist_ok=True)
        for field, value in (("serial", serial), ("idVendor", vendor), ("idProduct", product),
                             ("busnum", name.split("-")[0]), ("devnum", str(devnum)), ("speed", speed)):
            (device / field).write_text(value + "\n")
        (device / "descriptors").write_bytes(descriptor_bytes if descriptor_bytes is not None
                                             else descriptors(accessory=product.startswith("2d")))
        (device / "power").mkdir(exist_ok=True)
        for key, value in (lpm or {}).items():
            (device / "power" / key).write_text(value + "\n")
        self.devices[name] = {"serial": serial, "vendor": vendor, "product": product, "devnum": devnum}

    def remove(self, name: str) -> None:
        shutil.rmtree(self.root / name, ignore_errors=True)
        self.devices.pop(name, None)


class FakeLibusb:
    """Enough of libusb-1.0 for _LibusbDevice: a device list from FakeSysfs, control transfers and reset.

    ``on_start`` / ``on_reset`` emulate the re-enumeration the phone performs."""

    def __init__(self, sysfs: FakeSysfs, *, on_start=None, on_reset=None) -> None:
        self.sysfs, self.on_start, self.on_reset = sysfs, on_start, on_reset
        self.opened: list[str] = []
        self.controls: list[tuple] = []
        self.resets: list[str] = []
        self._arrays: list = []
        self._names: list[str] = []

    @staticmethod
    def _set_pointer(reference, value: int) -> None:
        ctypes.c_void_p.from_buffer(reference._obj).value = value

    def libusb_init(self, reference):
        self._set_pointer(reference, 0x1000)
        return 0

    def libusb_exit(self, context):
        return None

    def libusb_get_device_list(self, context, reference):
        self._names = sorted(self.sysfs.devices)
        array = (ctypes.c_void_p * (len(self._names) + 1))(*range(1, len(self._names) + 1), None)
        self._arrays.append(array)
        self._set_pointer(reference, ctypes.addressof(array))
        return len(self._names)

    def libusb_free_device_list(self, devices, unref):
        return None

    def _name(self, device) -> str:
        return self._names[int(device) - 1]

    def libusb_get_bus_number(self, device):
        return int(self._name(device).split("-")[0])

    def libusb_get_device_address(self, device):
        return self.sysfs.devices[self._name(device)]["devnum"]

    def libusb_get_port_numbers(self, device, ports, size):
        numbers = [int(part) for part in self._name(device).split("-")[1].split(".")]
        for index, value in enumerate(numbers):
            ports[index] = value
        return len(numbers)

    def libusb_get_device_descriptor(self, device, reference):
        row = self.sysfs.devices[self._name(device)]
        descriptor = reference._obj
        descriptor.idVendor, descriptor.idProduct = int(row["vendor"], 16), int(row["product"], 16)
        descriptor.iSerialNumber = 3
        return 0

    def libusb_open(self, device, reference):
        name = self._name(device)
        self.opened.append(name)
        self._set_pointer(reference, 0x5000 + int(device))
        self._open_name = name
        return 0

    def libusb_close(self, handle):
        return None

    def libusb_get_string_descriptor_ascii(self, handle, index, buffer, size):
        serial = self.sysfs.devices[self._open_name]["serial"].encode()
        for position, byte in enumerate(serial):
            buffer[position] = byte
        return len(serial)

    def libusb_control_transfer(self, handle, request_type, request, value, index, data, length, timeout):
        self.controls.append((self._open_name, request_type, request, index))
        if request == 51:
            data[0], data[1] = 2, 0
            return 2
        if request == 52:
            return length
        if request == 53 and self.on_start is not None:
            self.on_start(self._open_name)
        return 0

    def libusb_reset_device(self, handle):
        self.resets.append(self._open_name)
        if self.on_reset is not None:
            self.on_reset(self._open_name)
        return -5

    def libusb_error_name(self, code):
        return b"FAKE"


# --- protocol -----------------------------------------------------------------------------------

class ProtocolTests(unittest.TestCase):
    def test_python_mirror_matches_the_c_header(self) -> None:
        text = PROTOCOL_HEADER.read_text()
        self.assertEqual(int(re.search(r"S43A_MAGIC UINT32_C\((0x[0-9a-fA-F]+)\)", text).group(1), 16), ab.MAGIC)
        self.assertEqual(int(re.search(r"S43A_HEADER_BYTES (\d+)", text).group(1)), ab.HEADER_BYTES)
        self.assertEqual(int(re.search(r"S43A_PAD_QUANTUM (\d+)", text).group(1)), ab.PAD_QUANTUM)
        self.assertEqual(ab.HEADER.size, ab.HEADER_BYTES)
        for name, value in re.findall(r"S43A_([A-Z_]+) = (\d+),\s+//", text):
            if hasattr(ab, name):
                self.assertEqual(getattr(ab, name), int(value), name)
        self.assertEqual(struct.pack("<I", ab.MAGIC), b"S43A")

    def test_no_frame_ends_on_a_max_packet_boundary(self) -> None:
        for length in range(0, 70000, 7):
            total = len(ab.pack_frame(ab.DATA, 1, 1, b"x" * length))
            self.assertNotEqual(total % 512, 0, length)
            self.assertEqual(total, ab.HEADER_BYTES + length + ab.pad_for(length))
        self.assertEqual(ab.pad_for(512 - 40), 1)
        self.assertEqual(ab.pad_for(1024 - 40), 1)
        self.assertEqual(ab.pad_for(10280), 0)

    def test_parser_reassembles_split_and_joined_frames(self) -> None:
        parser = ab.FrameParser()
        parser.session = 7
        frames = [ab.pack_frame(ab.DATA, 7, 3, bytes([index]) * (1000 * index + 1), aux=index)
                  for index in range(1, 5)]
        stream = b"".join(frames)
        seen = []
        for start in range(0, len(stream), 333):
            seen.extend(parser.feed(stream[start:start + 333]))
        self.assertEqual([(row.kind, row.stream, len(row.payload), row.aux) for row in seen],
                         [(ab.DATA, 3, 1000 * index + 1, index) for index in range(1, 5)])
        self.assertEqual(parser.buffer, bytearray())

    def test_resync_discards_stale_bytes_and_other_sessions(self) -> None:
        parser = ab.FrameParser()
        parser.expect(ab.HELLO_ACK, 42)
        garbage = b"\x00junk" * 50 + ab.pack_frame(ab.DATA, 41, 1, b"old")[:30]
        stale = ab.pack_frame(ab.HELLO_ACK, 41, 0, b"\0" * 16)
        ack = ab.pack_frame(ab.HELLO_ACK, 42, 0, ab.HELLO_ACK_PAYLOAD.pack(1, 99, 8, 0))
        frames = parser.feed(garbage + stale + ack + ab.pack_frame(ab.DATA, 41, 2, b"late"))
        self.assertEqual([row.kind for row in frames], [ab.HELLO_ACK])
        self.assertGreater(parser.discarded_bytes, 250)
        self.assertEqual(parser.stale_frames, 1)
        with self.assertRaises(ab.ProtocolError):
            parser.feed(b"\xff" * 64)


# --- selection ----------------------------------------------------------------------------------

class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.sysfs = FakeSysfs(Path(self.directory.name))
        self.sysfs.set("2-2", OP15_GADGET_SERIAL, "18d1", "2d00", 45)   # OP15 FunctionFS gadget
        self.sysfs.set("2-9.2", PIXEL_SERIAL, "18d1", "4ee7", 47,
                       lpm={"usb3_hardware_lpm_u1": "enabled", "usb3_hardware_lpm_u2": "enabled",
                            "control": "on"})

    def state(self, name: str = "2-9.2") -> ab.UsbDeviceState:
        return ab.read_usb_device(name, sysfs_root=self.sysfs.root)

    def test_the_pinned_port_and_serial_select_the_pixel_only(self) -> None:
        state = self.state()
        self.assertEqual((state.vendor_product, state.port_numbers, state.busnum), ("18d1:4ee7", (9, 2), 2))
        ab.check_pixel(state, serial=PIXEL_SERIAL, forbidden_serials=ab.DEFAULT_FORBIDDEN_SERIALS,
                       require_accessory=False)
        with self.assertRaisesRegex(ab.SelectionError, "not in accessory"):
            ab.check_pixel(state, serial=PIXEL_SERIAL, forbidden_serials=(), require_accessory=True)
        with self.assertRaisesRegex(ab.SelectionError, "forbidden phone"):
            ab.check_pixel(self.state("2-2"), serial=PIXEL_SERIAL, forbidden_serials=ab.DEFAULT_FORBIDDEN_SERIALS,
                           require_accessory=None)
        with self.assertRaisesRegex(ab.SelectionError, "not the pinned Pixel"):
            ab.check_pixel(self.state("2-2"), serial=PIXEL_SERIAL, forbidden_serials=(), require_accessory=None)
        self.assertEqual(ab.lpm_state("2-9.2", sysfs_root=self.sysfs.root),
                         {"usb3_hardware_lpm_u1": "enabled", "usb3_hardware_lpm_u2": "enabled",
                          "power_control": "on"})

    def test_accessory_without_adb_is_refused_as_the_op15_collision_hazard(self) -> None:
        self.sysfs.set("2-9.2", PIXEL_SERIAL, "18d1", "2d00", 48)
        with self.assertRaises(ab.AccessoryOnlyError):
            ab.check_pixel(self.state(), serial=PIXEL_SERIAL, forbidden_serials=(), require_accessory=None)
        self.sysfs.set("2-9.2", PIXEL_SERIAL, "18d1", "2d01", 49)
        ab.check_pixel(self.state(), serial=PIXEL_SERIAL, forbidden_serials=(), require_accessory=True)

    def test_bad_names_and_missing_devices_are_refused(self) -> None:
        for name in ("2-", "x-1", "2-9..2", "../2-9.2", "2-9.3"):
            with self.subTest(name=name), self.assertRaises(ab.SelectionError):
                ab.read_usb_device(name, sysfs_root=self.sysfs.root)

    def test_accessory_interface_comes_from_the_descriptors(self) -> None:
        interface = ab.accessory_interface(descriptors())
        self.assertEqual((interface.interface, interface.out_endpoint, interface.in_endpoint, interface.max_packet,
                          interface.has_adb_interface), (0, 0x01, 0x81, 1024, True))
        for bad in (descriptors(accessory=False), descriptors(accessory_endpoints=1), descriptors()[:40] + b"\x00"):
            with self.assertRaises(ab.SelectionError):
                ab.accessory_interface(bad)

    def test_switch_talks_to_the_pinned_pixel_only_and_waits_for_accessory_adb(self) -> None:
        def start(name):
            self.sysfs.set(name, PIXEL_SERIAL, "18d1", "2d01", 50)
        library = FakeLibusb(self.sysfs, on_start=start)
        result = ab.switch_to_accessory("2-9.2", PIXEL_SERIAL, sysfs_root=self.sysfs.root, library=library,
                                        timeout_s=5)
        self.assertEqual(result["after"]["vendor_product"], "18d1:2d01")
        self.assertEqual(set(library.opened), {"2-9.2"})
        self.assertEqual([row[2] for row in library.controls], [51, 52, 52, 52, 52, 52, 52, 53])
        again = ab.switch_to_accessory("2-9.2", PIXEL_SERIAL, sysfs_root=self.sysfs.root, library=library)
        self.assertTrue(again["already_accessory"])

    def test_an_accessory_only_result_is_reset_straight_back(self) -> None:
        def start(name):
            self.sysfs.set(name, PIXEL_SERIAL, "18d1", "2d00", 50)

        def reset(name):
            self.sysfs.set(name, PIXEL_SERIAL, "18d1", "4ee7", 51)
        library = FakeLibusb(self.sysfs, on_start=start, on_reset=reset)
        with self.assertRaises(ab.AccessoryOnlyError):
            ab.switch_to_accessory("2-9.2", PIXEL_SERIAL, sysfs_root=self.sysfs.root, library=library, timeout_s=5)
        self.assertEqual(library.resets, ["2-9.2"])
        self.assertEqual(self.state().product, "4ee7")
        self.assertNotIn("2-2", library.opened)

    def test_restore_resets_only_an_accessory_pixel(self) -> None:
        def reset(name):
            self.sysfs.set(name, PIXEL_SERIAL, "18d1", "4ee7", 52)
        library = FakeLibusb(self.sysfs, on_reset=reset)
        self.assertTrue(ab.restore_normal("2-9.2", PIXEL_SERIAL, sysfs_root=self.sysfs.root,
                                          library=library)["already_normal"])
        self.sysfs.set("2-9.2", PIXEL_SERIAL, "18d1", "2d01", 51)
        result = ab.restore_normal("2-9.2", PIXEL_SERIAL, sysfs_root=self.sysfs.root, library=library, timeout_s=5)
        self.assertEqual((result["reset_return_code"], result["after"]["vendor_product"]), (-5, "18d1:4ee7"))
        with self.assertRaisesRegex(ab.SelectionError, "not the pinned Pixel"):
            ab.restore_normal("2-2", PIXEL_SERIAL, sysfs_root=self.sysfs.root, library=library)
        self.assertEqual(library.resets, ["2-9.2"])

    def test_restore_retries_a_reset_that_left_the_pixel_in_accessory_mode(self) -> None:
        resets = []

        def reset(name):
            resets.append(name)
            if len(resets) == 2:        # the first reset is ignored by the phone, the second one works
                self.sysfs.set(name, PIXEL_SERIAL, "18d1", "4ee7", 61)
        library = FakeLibusb(self.sysfs, on_reset=reset)
        self.sysfs.set("2-9.2", PIXEL_SERIAL, "18d1", "2d01", 60)
        result = ab.restore_normal("2-9.2", PIXEL_SERIAL, sysfs_root=self.sysfs.root, library=library, timeout_s=2)
        self.assertEqual((result["attempts"], result["after"]["vendor_product"]), (2, "18d1:4ee7"))
        self.assertEqual(library.resets, ["2-9.2", "2-9.2"])
        self.sysfs.set("2-9.2", PIXEL_SERIAL, "18d1", "2d01", 62)
        with self.assertRaisesRegex(ab.SelectionError, "did not reach normal mode"):
            ab.restore_normal("2-9.2", PIXEL_SERIAL, sysfs_root=self.sysfs.root,
                              library=FakeLibusb(self.sysfs, on_reset=lambda name: None), timeout_s=1, attempts=2)

    def test_a_device_that_re_enumerated_while_opening_is_refused(self) -> None:
        self.sysfs.set("2-9.2", PIXEL_SERIAL, "18d1", "2d01", 53)
        state = self.state()

        class Moving(FakeLibusb):
            def libusb_get_string_descriptor_ascii(inner, handle, index, buffer, size):
                self.sysfs.set("2-9.2", PIXEL_SERIAL, "18d1", "2d01", 54)
                return super().libusb_get_string_descriptor_ascii(handle, index, buffer, size)
        with self.assertRaisesRegex(ab.SelectionError, "re-enumerated"):
            ab._LibusbDevice(state, library=Moving(self.sysfs), sysfs_root=self.sysfs.root)

    def test_wait_for_mode_stops_on_forbidden_serials_and_times_out(self) -> None:
        self.sysfs.set("2-9.2", OP15_SERIAL, "22d9", "2772", 60)
        with self.assertRaisesRegex(ab.SelectionError, "forbidden phone"):
            ab.wait_for_mode("2-9.2", PIXEL_SERIAL, accessory=True, forbidden_serials=ab.DEFAULT_FORBIDDEN_SERIALS,
                             timeout_s=1, sysfs_root=self.sysfs.root, sleep=lambda _: None)
        self.sysfs.remove("2-9.2")
        clock = iter(range(100)).__next__
        with self.assertRaisesRegex(ab.SelectionError, "did not reach accessory"):
            ab.wait_for_mode("2-9.2", PIXEL_SERIAL, accessory=True, forbidden_serials=(), timeout_s=3,
                             sysfs_root=self.sysfs.root, sleep=lambda _: None, clock=clock)


# --- bridge <-> relay loopback ------------------------------------------------------------------

class FakeWorker:
    """Phone-side stand-in for the FFN worker: one client at a time; request N bytes -> transformed reply."""

    def __init__(self, request_bytes: int = 10280, reply_extra: bytes = b"12345678") -> None:
        self.request_bytes, self.reply_extra = request_bytes, reply_extra
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(4)
        self.port = self.listener.getsockname()[1]
        self.connections = 0
        self.disconnects = 0
        threading.Thread(target=self._serve, daemon=True).start()

    def reply(self, data: bytes) -> bytes:
        return data[::-1] + self.reply_extra

    def _serve(self) -> None:
        while True:
            try:
                client, _ = self.listener.accept()
            except OSError:
                return
            self.connections += 1
            with client:
                while True:
                    data = b""
                    while len(data) < self.request_bytes:
                        part = client.recv(self.request_bytes - len(data))
                        if not part:
                            break
                        data += part
                    if len(data) < self.request_bytes:
                        break
                    client.sendall(self.reply(data))
            self.disconnects += 1

    def close(self) -> None:
        self.listener.close()


def exchange(sock: socket.socket, payload: bytes, response_bytes: int) -> bytes:
    sock.sendall(payload)
    data = b""
    while len(data) < response_bytes:
        part = sock.recv(1 << 16)
        if not part:
            break
        data += part
    return data


class BridgeLoopbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.relay = build_host_relay(Path(cls.directory.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def start_relay(self, worker_port: int, *extra: str) -> tuple[subprocess.Popen, socket.socket]:
        host, phone = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        process = subprocess.Popen([str(self.relay), "--accessory-fd", str(phone.fileno()), "--worker-port",
                                    str(worker_port), *extra], pass_fds=[phone.fileno()], stderr=subprocess.PIPE,
                                   text=True)
        phone.close()
        self.addCleanup(self._reap, process)
        return process, host

    @staticmethod
    def _reap(process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        if process.stderr is not None:
            process.stderr.close()

    def bridge(self, host: socket.socket, **options) -> ab.Bridge:
        bridge = ab.Bridge(ab.SocketLink(host), ab.BridgeOptions(**options))
        bridge.listen()
        bridge.handshake()
        bridge.start()
        self.addCleanup(lambda: bridge._stop.is_set() or bridge.finish())
        return bridge

    def test_streams_carry_bytes_exactly_across_reconnects_and_batch_sizes(self) -> None:
        for rows in (1, 2, 4):
            with self.subTest(rows=rows):
                worker = FakeWorker(request_bytes=10280 * rows)
                self.addCleanup(worker.close)
                relay, host = self.start_relay(worker.port)
                bridge = self.bridge(host)
                for connection in range(3):  # the server re-arms by reconnecting
                    with socket.create_connection(("127.0.0.1", bridge.listen_port)) as client:
                        for _ in range(20):
                            payload = os.urandom(10280 * rows)
                            self.assertEqual(exchange(client, payload, 10280 * rows + 8), worker.reply(payload))
                deadline = time.monotonic() + 10
                while worker.disconnects < 3 and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertEqual((worker.connections, worker.disconnects), (3, 3))
                bridge.request_stop("test")
                self.assertEqual(bridge.finish(), ab.EXIT_OK)
                relay.wait(timeout=10)
                self.assertIn('"streams_opened":3', relay.stderr.read())

    def test_concurrent_streams_are_multiplexed(self) -> None:
        worker_a, worker_b = FakeWorker(request_bytes=5000), FakeWorker(request_bytes=5000)
        for worker in (worker_a, worker_b):
            self.addCleanup(worker.close)
        relay, host = self.start_relay(worker_a.port)
        bridge = self.bridge(host)
        clients = [socket.create_connection(("127.0.0.1", bridge.listen_port)) for _ in range(3)]
        errors = []

        def run(client):
            try:
                for _ in range(30):
                    payload = os.urandom(5000)
                    if exchange(client, payload, 5008) != worker_a.reply(payload):
                        errors.append("mismatch")
            except OSError as error:
                errors.append(repr(error))
        # the fake worker serves one client at a time; the others queue exactly as behind adb forward
        threads = [threading.Thread(target=run, args=(client,)) for client in clients]
        for thread in threads:
            thread.start()
        for client, thread in zip(clients, threads):
            thread.join(timeout=30)
            client.close()
        self.assertEqual(errors, [])

    def test_local_echo_stream_and_ping_measure_the_relay_without_the_phone_hop(self) -> None:
        worker = FakeWorker()
        self.addCleanup(worker.close)
        relay, host = self.start_relay(worker.port)
        bridge = self.bridge(host, echo_port=0)
        self.assertEqual(bridge.echo_listen_port, 0)
        bridge.request_stop("test")
        bridge.finish()
        relay.wait(timeout=10)

        relay, host = self.start_relay(worker.port)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            echo_port = probe.getsockname()[1]
        bridge = self.bridge(host, echo_port=echo_port)
        with socket.create_connection(("127.0.0.1", echo_port)) as client:
            payload = os.urandom(41000)
            self.assertEqual(exchange(client, payload, 41000), payload)
        self.assertEqual(worker.connections, 0)
        bridge.request_stop("test")
        bridge.finish()

        relay, host = self.start_relay(worker.port)
        result = ab.ping_benchmark(ab.SocketLink(host), count=12, request_bytes=10280, response_bytes=10288,
                                   gap_ms=1, token_gap_ms=(5, 10), calls_per_token=6, warmup=2)
        self.assertEqual(len(result["calls"]), 14)
        self.assertEqual({row["response_bytes"] for row in result["calls"]}, {10288})
        self.assertEqual([row["position"] for row in result["calls"][:7]], [0, 1, 2, 3, 4, 5, 0])
        self.assertTrue(all(row["link_us"] <= row["rtt_us"] for row in result["calls"]))
        self.assertIsNotNone(result["summary"]["rtt_us_p50"])

    def test_a_new_bridge_session_resynchronises_a_running_relay(self) -> None:
        worker = FakeWorker()
        self.addCleanup(worker.close)
        relay, host = self.start_relay(worker.port)
        first = ab.Bridge(ab.SocketLink(host), ab.BridgeOptions())
        first.listen()
        first.handshake()
        first.start()
        client = socket.create_connection(("127.0.0.1", first.listen_port))
        payload = os.urandom(10280)
        self.assertEqual(exchange(client, payload, 10288), worker.reply(payload))
        # a crashed bridge: its threads stop, the link stays open with a half-sent frame on it
        first._stop.set()
        time.sleep(0.4)
        host.send(ab.pack_frame(ab.DATA, first.session, 1, b"z" * 100)[:60])
        second = ab.Bridge(ab.SocketLink(host), ab.BridgeOptions())
        second.listen()
        hello = second.handshake()
        self.assertNotEqual(second.session, first.session)
        self.assertEqual(hello["relay_pid"], relay.pid)
        second.start()
        with socket.create_connection(("127.0.0.1", second.listen_port)) as fresh:
            payload = os.urandom(10280)
            self.assertEqual(exchange(fresh, payload, 10288), worker.reply(payload))
        client.close()
        second.request_stop("test")
        self.assertEqual(second.finish(), ab.EXIT_OK)
        relay.wait(timeout=10)
        stats = json.loads(relay.stderr.read().split("S43AOARELAYSTATS ", 1)[1])
        self.assertEqual(stats["hellos"], 2)
        self.assertGreaterEqual(stats["resyncs"], 1)

    def test_worker_exit_closes_the_client_and_client_exit_closes_the_worker(self) -> None:
        worker = FakeWorker()
        self.addCleanup(worker.close)
        relay, host = self.start_relay(worker.port)
        bridge = self.bridge(host)
        with socket.create_connection(("127.0.0.1", bridge.listen_port)) as client:
            client.sendall(b"short")
        deadline = time.monotonic() + 10
        while worker.disconnects < 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(worker.disconnects, 1)
        with socket.socket() as unused:  # bound, never listening: the relay's connect is refused
            unused.bind(("127.0.0.1", 0))
            dead_port = unused.getsockname()[1]
        relay2, host2 = self.start_relay(dead_port)
        bridge2 = self.bridge(host2)
        with socket.create_connection(("127.0.0.1", bridge2.listen_port)) as client:
            client.settimeout(10)
            self.assertEqual(client.recv(10), b"")
        self.assertEqual(bridge2.counters["relay_close_reason_1"], 1)

    def test_relay_loss_and_silence_end_the_bridge_with_distinct_codes(self) -> None:
        worker = FakeWorker()
        self.addCleanup(worker.close)
        relay, host = self.start_relay(worker.port)
        bridge = self.bridge(host, ping_interval_ms=100, ping_timeout_ms=600)
        client = socket.create_connection(("127.0.0.1", bridge.listen_port))
        os.kill(relay.pid, signal.SIGSTOP)
        self.addCleanup(lambda: relay.poll() is None and os.kill(relay.pid, signal.SIGCONT))
        self.assertTrue(bridge.wait(10))
        self.assertEqual(bridge.finish(), ab.EXIT_RELAY_UNRESPONSIVE)
        client.settimeout(5)
        self.assertEqual(client.recv(10), b"")  # the server sees its helper connection end
        os.kill(relay.pid, signal.SIGCONT)
        relay.terminate()
        relay.wait(timeout=10)

        relay, host = self.start_relay(worker.port)
        bridge = self.bridge(host)
        relay.terminate()
        relay.wait(timeout=10)
        self.assertTrue(bridge.wait(10))
        self.assertEqual(bridge.finish(), ab.EXIT_LINK_LOST)

    def test_keepalive_sends_nops_only_inside_the_activity_window(self) -> None:
        worker = FakeWorker()
        self.addCleanup(worker.close)
        relay, host = self.start_relay(worker.port)
        bridge = self.bridge(host, keepalive_ms=2, keepalive_window_ms=150)
        with socket.create_connection(("127.0.0.1", bridge.listen_port)) as client:
            payload = os.urandom(10280)
            exchange(client, payload, 10288)
        time.sleep(0.6)
        nops = bridge.counters["nops"]
        self.assertGreater(nops, 10)
        time.sleep(0.4)
        self.assertEqual(bridge.counters["nops"], nops)  # the window closed: no more keep-alive traffic
        bridge.request_stop("test")
        bridge.finish()
        relay.terminate()
        relay.wait(timeout=10)
        stats = json.loads(relay.stderr.read().split("S43AOARELAYSTATS ", 1)[1])
        self.assertEqual(stats["nops"], nops)

    def test_qos_window_is_armed_by_stream_traffic_only_and_released_after_it(self) -> None:
        worker = FakeWorker()
        self.addCleanup(worker.close)
        device = Path(self.directory.name) / "cpu_dma_latency"
        device.write_bytes(b"")
        relay, host = self.start_relay(worker.port, "--qos-latency-us", "0", "--qos-window-ms", "300",
                                       "--qos-device", str(device))
        bridge = self.bridge(host, ping_interval_ms=100)
        time.sleep(0.8)  # liveness PINGs only: an idle helper never holds the QoS request
        self.assertGreater(bridge.counters["pongs"], 2)
        with socket.create_connection(("127.0.0.1", bridge.listen_port)) as client:
            payload = os.urandom(10280)
            exchange(client, payload, 10288)
        time.sleep(1.0)  # the window (300 ms) ends after the last DATA frame, while PINGs continue
        bridge.request_stop("test")
        bridge.finish()
        relay.terminate()
        relay.wait(timeout=10)
        stats = json.loads(relay.stderr.read().split("S43AOARELAYSTATS ", 1)[1])
        self.assertEqual((stats["qos_acquires"], stats["qos_releases"]), (1, 1))
        self.assertEqual(device.read_bytes(), struct.pack("<i", 0))  # the requested latency, as a binary s32
        self.assertGreater(stats["cpu_s"], 0)

    def test_relay_options_parse_and_bad_ones_are_usage_errors(self) -> None:
        worker = FakeWorker()
        self.addCleanup(worker.close)
        relay, host = self.start_relay(worker.port, "--qos-latency-us", "0", "--qos-window-ms", "500",
                                       "--uclamp-min", "0")
        bridge = self.bridge(host)
        self.assertEqual(bridge.hello["relay_options"] & 1, 1)
        bridge.request_stop("test")
        bridge.finish()
        relay.terminate()
        relay.wait(timeout=10)
        for arguments in (["--worker-port", "0"], ["--worker-port", "1", "--cpus", "0"], ["--bogus", "1"],
                          ["--worker-port", "1", "--qos-latency-us", "-3"]):
            with self.subTest(arguments=arguments):
                completed = subprocess.run([str(self.relay), *arguments], capture_output=True, timeout=10)
                self.assertEqual(completed.returncode, 2)

    def test_command_line_serve_writes_ready_and_status_and_stops_cleanly(self) -> None:
        worker = FakeWorker()
        self.addCleanup(worker.close)
        relay, host = self.start_relay(worker.port)
        root = Path(self.directory.name)
        ready, status = root / "ready.json", root / "status.json"
        for path in (ready, status):
            path.unlink(missing_ok=True)
        process = subprocess.Popen(
            [os.sys.executable, str(Path(ab.__file__)), "serve", "--sysfs-device", "2-9.2", "--serial", PIXEL_SERIAL,
             "--link-fd", str(host.fileno()), "--ready-file", str(ready), "--status-file", str(status),
             "--status-interval-ms", "100", "--shutdown-relay-on-exit"],
            pass_fds=[host.fileno()], stderr=subprocess.PIPE, text=True)
        host.close()
        deadline = time.monotonic() + 20
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        port = json.loads(ready.read_text())["listen_port"]
        with socket.create_connection(("127.0.0.1", port)) as client:
            payload = os.urandom(10280)
            self.assertEqual(exchange(client, payload, 10288), worker.reply(payload))
        time.sleep(0.3)
        self.assertEqual(json.loads(status.read_text())["state"], "up")
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=20), ab.EXIT_OK)
        self.assertEqual(relay.wait(timeout=10), 0)  # SHUTDOWN frame: the relay exits by itself
        final = json.loads(status.read_text())
        self.assertEqual((final["state"], final["exit_code"], final["exit_reason"]), ("down", 0, "SIGTERM"))
        self.assertEqual(final["counters"]["streams_opened"], 1)


if __name__ == "__main__":
    unittest.main()
