#!/usr/bin/env python3
"""Materialize one exact-stack FunctionFS transport qualification identity."""

from __future__ import annotations

import argparse
from pathlib import Path

from .._internal.types import canonical_json
from .transport_profiles import build_transport_qualification_identity


def _dependency(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if (
        separator != "="
        or not name
        or not name.isascii()
        or ":" in name
        or not raw_path
    ):
        raise argparse.ArgumentTypeError(
            "host dependency must be NAME=PATH"
        )
    return name, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-id", required=True)
    parser.add_argument("--transport-generation", required=True)
    parser.add_argument("--functionfs-identity", required=True)
    parser.add_argument("--phone-boot-image-sha256", required=True)
    parser.add_argument("--phone-kernel-release", required=True)
    parser.add_argument("--phone-usb-controller", required=True)
    parser.add_argument("--phone-usb-serial", required=True)
    parser.add_argument("--phone-usb-sysfs-device", required=True)
    parser.add_argument("--phone-session-sha256", required=True)
    parser.add_argument("--phone-worker-sha256", required=True)
    parser.add_argument("--phone-resident-workers", type=Path)
    parser.add_argument("--phone-resident-router", type=Path)
    parser.add_argument(
        "--qualification-phone-session-sha256", required=True
    )
    parser.add_argument(
        "--qualification-phone-worker-sha256", required=True
    )
    parser.add_argument("--host-binary", type=Path, required=True)
    parser.add_argument(
        "--host-dependency", type=_dependency, action="append", default=[]
    )
    parser.add_argument(
        "--qualification-binary", type=Path, required=True
    )
    parser.add_argument(
        "--transport-client-source", type=Path, required=True
    )
    parser.add_argument(
        "--qualified-allocator", action="append", required=True
    )
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument(
        "--minimum-usb-speed-mbps", type=int, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be a new absolute path")
    dependencies = dict(args.host_dependency)
    if len(dependencies) != len(args.host_dependency):
        parser.error("host dependency names must be unique")
    identity = build_transport_qualification_identity(
        identity_id=args.identity_id,
        transport_generation=args.transport_generation,
        hardware_identity={
            "functionfs_identity": args.functionfs_identity,
            "phone_boot_image_sha256": args.phone_boot_image_sha256,
            "phone_kernel_release": args.phone_kernel_release,
            "phone_usb_controller": args.phone_usb_controller,
            "phone_usb_serial": args.phone_usb_serial,
            "phone_usb_sysfs_device": args.phone_usb_sysfs_device,
        },
        phone_session_sha256=args.phone_session_sha256,
        phone_worker_sha256=args.phone_worker_sha256,
        qualification_phone_session_sha256=(
            args.qualification_phone_session_sha256
        ),
        qualification_phone_worker_sha256=(
            args.qualification_phone_worker_sha256
        ),
        host_binary_path=args.host_binary,
        qualification_binary_path=args.qualification_binary,
        transport_client_source_path=args.transport_client_source,
        qualified_allocators=tuple(args.qualified_allocator),
        receipt_paths=tuple(args.receipt),
        minimum_usb_speed_mbps=args.minimum_usb_speed_mbps,
        host_dependency_paths=dependencies,
        phone_resident_workers_path=args.phone_resident_workers,
        phone_resident_router_path=args.phone_resident_router,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        parser.error("temporary output already exists")
    try:
        temporary.write_text(
            canonical_json(identity.to_json()) + "\n", encoding="ascii"
        )
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(identity.identity_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
