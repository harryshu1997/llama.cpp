#!/usr/bin/env python3
"""Switch a phone into AOA accessory mode and benchmark the bulk endpoints.

Compares the Android Open Accessory transport against the adb-forward path for
the same activation-sized exchange. AOA removes the adb server, adbd, and both
TCP loopback hops: the host talks straight to bulk endpoints and the phone-side
daemon reads /dev/usb_accessory directly.

Uses libusb through ctypes so no -dev headers are required. Host udev already
grants plugdev access to both 22d9 (OnePlus) and 18d1 (accessory mode).

  python3 aoa_bench.py --serial <SER> switch     # enter accessory mode
  python3 aoa_bench.py bench --req N --rsp M     # measure the bulk round trip
  python3 aoa_bench.py reset                     # leave accessory mode
"""

import argparse
import ctypes
import ctypes.util
import statistics as st
import sys
import time

AOA_VID = 0x18D1
ACCESSORY_PIDS = (0x2D00, 0x2D01, 0x2D04, 0x2D05)
REQ_GET_PROTOCOL = 51
REQ_SEND_STRING = 52
REQ_START = 53
STRINGS = [
    "Anthropic",            # manufacturer
    "TPSliceProbe",         # model
    "operator-split latency probe",
    "1.0",
    "https://example.invalid",
    "0000000000000001",
]


class Ctx(ctypes.Structure):
    pass


CTXP = ctypes.POINTER(Ctx)


def load():
    lu = ctypes.CDLL(ctypes.util.find_library("usb-1.0"))
    lu.libusb_init.argtypes = [ctypes.POINTER(CTXP)]
    lu.libusb_open_device_with_vid_pid.restype = ctypes.c_void_p
    lu.libusb_open_device_with_vid_pid.argtypes = [CTXP, ctypes.c_uint16, ctypes.c_uint16]
    lu.libusb_control_transfer.argtypes = [
        ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint16,
        ctypes.c_uint16, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint16, ctypes.c_uint]
    lu.libusb_bulk_transfer.argtypes = [
        ctypes.c_void_p, ctypes.c_ubyte, ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_uint]
    lu.libusb_close.argtypes = [ctypes.c_void_p]
    lu.libusb_claim_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lu.libusb_release_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lu.libusb_reset_device.argtypes = [ctypes.c_void_p]
    lu.libusb_get_device_list.argtypes = [CTXP, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))]
    lu.libusb_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    lu.libusb_get_bus_number.argtypes = [ctypes.c_void_p]
    lu.libusb_get_bus_number.restype = ctypes.c_ubyte
    lu.libusb_get_device_address.argtypes = [ctypes.c_void_p]
    lu.libusb_get_device_address.restype = ctypes.c_ubyte
    lu.libusb_open.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lu.libusb_detach_kernel_driver.argtypes = [ctypes.c_void_p, ctypes.c_int]
    ctx = CTXP()
    if lu.libusb_init(ctypes.byref(ctx)) != 0:
        sys.exit("libusb_init failed")
    return lu, ctx


def open_accessory(lu, ctx):
    for pid in ACCESSORY_PIDS:
        h = lu.libusb_open_device_with_vid_pid(ctx, AOA_VID, pid)
        if h:
            return h, pid
    return None, None


def open_by_bus_addr(lu, ctx, bus, addr):
    """Both phones can share a VID:PID, so select by USB bus/address."""
    lst = ctypes.POINTER(ctypes.c_void_p)()
    n = lu.libusb_get_device_list(ctx, ctypes.byref(lst))
    handle = None
    for i in range(n):
        dev = lst[i]
        if lu.libusb_get_bus_number(dev) == bus and lu.libusb_get_device_address(dev) == addr:
            h = ctypes.c_void_p()
            if lu.libusb_open(dev, ctypes.byref(h)) == 0:
                handle = h.value
            break
    lu.libusb_free_device_list(lst, 1)
    return handle


def cmd_switch(lu, ctx, args):
    if args.bus and args.addr:
        h = open_by_bus_addr(lu, ctx, args.bus, args.addr)
        if not h:
            sys.exit(f"cannot open bus {args.bus} addr {args.addr}")
        print(f"targeting bus {args.bus} addr {args.addr}")
    else:
        h = lu.libusb_open_device_with_vid_pid(ctx, args.vid, args.pid)
    if not h:
        sys.exit(f"cannot open {args.vid:04x}:{args.pid:04x}")
    buf = (ctypes.c_ubyte * 2)()
    rc = lu.libusb_control_transfer(h, 0xC0, REQ_GET_PROTOCOL, 0, 0, buf, 2, 2000)
    if rc != 2:
        sys.exit(f"GET_PROTOCOL failed: {rc}")
    print(f"AOA protocol version {buf[0] | (buf[1] << 8)}")
    for index, text in enumerate(STRINGS):
        payload = text.encode() + b"\x00"
        arr = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        rc = lu.libusb_control_transfer(h, 0x40, REQ_SEND_STRING, 0, index,
                                        arr, len(payload), 2000)
        if rc < 0:
            sys.exit(f"SEND_STRING[{index}] failed: {rc}")
    rc = lu.libusb_control_transfer(h, 0x40, REQ_START, 0, 0, None, 0, 2000)
    print("ACCESSORY_START ->", rc)
    lu.libusb_close(ctypes.c_void_p(h))


def cmd_bench(lu, ctx, args):
    h, pid = open_accessory(lu, ctx)
    if not h:
        sys.exit("no accessory-mode device found (run `switch` first)")
    print(f"accessory device 18d1:{pid:04x}")
    lu.libusb_detach_kernel_driver(ctypes.c_void_p(h), 0)
    rc = lu.libusb_claim_interface(ctypes.c_void_p(h), 0)
    if rc != 0:
        sys.exit(f"claim_interface failed: {rc}")

    out_ep, in_ep = args.out_ep, args.in_ep
    tx = (ctypes.c_ubyte * args.req).from_buffer_copy(b"x" * args.req)
    rx = (ctypes.c_ubyte * args.rsp)()
    n = ctypes.c_int(0)

    def once():
        rc = lu.libusb_bulk_transfer(ctypes.c_void_p(h), out_ep, tx, args.req,
                                     ctypes.byref(n), 2000)
        if rc != 0:
            raise RuntimeError(f"bulk OUT {rc}")
        got = 0
        while got < args.rsp:
            rc = lu.libusb_bulk_transfer(ctypes.c_void_p(h), in_ep,
                                         ctypes.cast(ctypes.byref(rx, got),
                                                     ctypes.POINTER(ctypes.c_ubyte)),
                                         args.rsp - got, ctypes.byref(n), 2000)
            if rc != 0:
                raise RuntimeError(f"bulk IN {rc}")
            got += n.value

    for _ in range(30):
        once()
    samples = []
    for _ in range(args.iters):
        t0 = time.perf_counter_ns()
        once()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    q = sorted(samples)
    print(f"AOA bulk {args.req}B -> {args.rsp}B, {args.iters} samples:")
    print(f"  min {q[0]:.3f}  median {st.median(samples):.3f}  "
          f"p90 {q[int(.9 * len(q))]:.3f}  p99 {q[int(.99 * len(q))]:.3f} ms")
    lu.libusb_release_interface(ctypes.c_void_p(h), 0)
    lu.libusb_close(ctypes.c_void_p(h))


def cmd_reset(lu, ctx, args):
    h, pid = open_accessory(lu, ctx)
    if not h:
        print("no accessory device present; nothing to reset")
        return
    print(f"resetting 18d1:{pid:04x}")
    lu.libusb_reset_device(ctypes.c_void_p(h))
    lu.libusb_close(ctypes.c_void_p(h))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["switch", "bench", "reset"])
    p.add_argument("--vid", type=lambda x: int(x, 16), default=0x22D9)
    p.add_argument("--pid", type=lambda x: int(x, 16), default=0x2772)
    p.add_argument("--req", type=int, default=4080)
    p.add_argument("--rsp", type=int, default=1024)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--bus", type=int, default=0)
    p.add_argument("--addr", type=int, default=0)
    p.add_argument("--out-ep", type=lambda x: int(x, 16), default=0x01)
    p.add_argument("--in-ep", type=lambda x: int(x, 16), default=0x81)
    args = p.parse_args()
    lu, ctx = load()
    {"switch": cmd_switch, "bench": cmd_bench, "reset": cmd_reset}[args.command](lu, ctx, args)
    lu.libusb_exit(ctx)


if __name__ == "__main__":
    main()
