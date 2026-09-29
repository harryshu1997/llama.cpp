#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from research_dev.scheduler.adapters import (
    TransportProfileError,
    TransportQualificationIdentity,
    build_transport_qualification_identity,
    materialize_measured_usb_links,
)


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def receipt(request_bytes: int, response_bytes: int) -> dict[str, object]:
    return {
        "configured_queue_depth": 1,
        "d2h_payload_MBps": 410.0,
        "device_mode": "dmabuf",
        "h2d_payload_MBps": 420.0,
        "host_allocator": "devmem",
        "host_mode": "async",
        "queue_depth": 1,
        "request_bytes": request_bytes,
        "reset_recoveries": 0,
        "response_bytes": response_bytes,
        "schema": "s41_ffs_dmabuf_transport_v2",
        "slot_safety_bytes": 65_536,
        "transport_generation": "synthetic-functionfs-v1",
        "usbfs_available_bytes": 16_777_216,
    }


def identity(receipt_sha256s: tuple[str, ...]) -> (
    TransportQualificationIdentity
):
    digest = "sha256:" + "1" * 64
    return TransportQualificationIdentity(
        identity_id="synthetic-fixed-stack",
        transport_generation="synthetic-functionfs-v1",
        hardware_identity={
            "functionfs_identity": "synthetic-ffs",
            "phone_boot_image_sha256": digest,
            "phone_kernel_release": "synthetic-kernel",
            "phone_usb_controller": "synthetic-controller",
            "phone_usb_serial": "synthetic-phone",
            "phone_usb_sysfs_device": "1-1",
        },
        software_identity={
            "host_binary_sha256": digest,
            "phone_session_sha256": digest,
            "phone_worker_sha256": digest,
            "qualification_binary_sha256": digest,
            "qualification_phone_session_sha256": digest,
            "qualification_phone_worker_sha256": digest,
            "transport_client_source_sha256": digest,
        },
        qualified_allocators=("devmem",),
        receipt_sha256s=receipt_sha256s,
        minimum_usb_speed_mbps=5_000,
    )


class TransportProfileTests(unittest.TestCase):
    def write_receipts(self, directory: Path) -> tuple[str, ...]:
        values = (
            receipt(1_048_576, 1),
            receipt(1, 1_048_576),
            receipt(1_048_576, 1_048_576),
        )
        hashes = []
        for index, value in enumerate(values):
            path = directory / ("receipt-" + str(index) + ".json")
            data = canonical(value)
            path.write_bytes(data)
            hashes.append("sha256:" + hashlib.sha256(data).hexdigest())
        return tuple(hashes)

    def test_identity_bound_receipts_materialize_measured_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = identity(self.write_receipts(root))
            links = materialize_measured_usb_links(
                (root,),
                qualification,
                host_device_id="host-a",
                phone_device_id="helper-b",
            )

        self.assertEqual(len(links), 2)
        self.assertEqual(
            {row.source_device for row in links}, {"host-a", "helper-b"}
        )
        self.assertTrue(all(row.status == "measured" for row in links))
        self.assertTrue(all(row.ready for row in links))
        self.assertTrue(all(
            row.qualification_identity_sha256
                == qualification.identity_sha256
            for row in links
        ))
        self.assertTrue(all(
            qualification.identity_sha256 in row.evidence_ids
            for row in links
        ))

    def test_unbound_receipts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_receipts(root)
            qualification = identity(("sha256:" + "2" * 64,))
            with self.assertRaises(
                TransportProfileError,
                msg="unbound transport receipts must not become measured",
            ):
                materialize_measured_usb_links(
                    (root,),
                    qualification,
                    host_device_id="host-a",
                    phone_device_id="helper-b",
                )

    def test_stack_identity_hash_is_canonical(self) -> None:
        first = identity(("sha256:" + "2" * 64, "sha256:" + "3" * 64))
        second = identity(("sha256:" + "3" * 64, "sha256:" + "2" * 64))
        self.assertEqual(first.identity_sha256, second.identity_sha256)
        self.assertEqual(first.to_json(), second.to_json())

    def test_identity_builder_binds_binary_and_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_hashes = self.write_receipts(root)
            host_binary = root / "host-binary"
            host_binary.write_bytes(b"synthetic-host-binary")
            qualification_binary = root / "qualification-binary"
            qualification_binary.write_bytes(b"synthetic-qualification")
            transport_source = root / "transport-source.cpp"
            transport_source.write_bytes(b"synthetic-transport-source")
            host_dependency = root / "host-dependency.so"
            host_dependency.write_bytes(
                b"synthetic-host-dependency S41SERVERFFN ready"
            )
            resident_workers = root / "resident-workers"
            resident_workers.write_bytes(b"synthetic-resident-workers")
            resident_router = root / "resident-router"
            resident_router.write_bytes(b"synthetic-resident-router")
            digest = "sha256:" + "1" * 64
            qualification = build_transport_qualification_identity(
                identity_id="synthetic-fixed-stack",
                transport_generation="synthetic-functionfs-v1",
                hardware_identity={
                    "functionfs_identity": "1234:5678",
                    "phone_boot_image_sha256": digest,
                    "phone_kernel_release": "synthetic-kernel",
                    "phone_usb_controller": "synthetic-controller",
                    "phone_usb_serial": "synthetic-phone",
                    "phone_usb_sysfs_device": "1-1",
                },
                phone_session_sha256=digest,
                phone_worker_sha256=digest,
                qualification_phone_session_sha256=digest,
                qualification_phone_worker_sha256=digest,
                host_binary_path=host_binary,
                qualification_binary_path=qualification_binary,
                transport_client_source_path=transport_source,
                qualified_allocators=("devmem",),
                receipt_paths=tuple(
                    sorted(root.glob("receipt-*.json"))
                ),
                minimum_usb_speed_mbps=5_000,
                host_dependency_paths={
                    "server-impl": host_dependency
                },
                phone_resident_workers_path=resident_workers,
                phone_resident_router_path=resident_router,
            )

        self.assertEqual(
            qualification.receipt_sha256s,
            tuple(sorted(receipt_hashes)),
        )
        self.assertEqual(
            qualification.software_identity["host_binary_sha256"],
            "sha256:"
            + hashlib.sha256(b"synthetic-host-binary").hexdigest(),
        )
        self.assertEqual(
            qualification.software_identity[
                "host_dependency_sha256:server-impl"
            ],
            "sha256:"
            + hashlib.sha256(
                b"synthetic-host-dependency S41SERVERFFN ready"
            ).hexdigest(),
        )
        self.assertEqual(
            qualification.software_identity[
                "phone_resident_workers_sha256"
            ],
            "sha256:"
            + hashlib.sha256(b"synthetic-resident-workers").hexdigest(),
        )
        self.assertEqual(
            qualification.software_identity[
                "phone_resident_router_sha256"
            ],
            "sha256:"
            + hashlib.sha256(b"synthetic-resident-router").hexdigest(),
        )


    def test_identity_builder_rejects_server_without_ffn_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_receipts(root)
            host_binary = root / "host-binary"
            host_binary.write_bytes(b"synthetic-host-binary")
            qualification_binary = root / "qualification-binary"
            qualification_binary.write_bytes(b"synthetic-qualification")
            transport_source = root / "transport-source.cpp"
            transport_source.write_bytes(b"synthetic-transport-source")
            host_dependency = root / "host-dependency.so"
            host_dependency.write_bytes(b"synthetic-host-dependency")
            digest = "sha256:" + "1" * 64
            with self.assertRaisesRegex(
                TransportProfileError, "lacks the FFN split client"
            ):
                build_transport_qualification_identity(
                    identity_id="synthetic-fixed-stack",
                    transport_generation="synthetic-functionfs-v1",
                    hardware_identity={
                        "functionfs_identity": "1234:5678",
                        "phone_boot_image_sha256": digest,
                        "phone_kernel_release": "synthetic-kernel",
                        "phone_usb_controller": "synthetic-controller",
                        "phone_usb_serial": "synthetic-phone",
                        "phone_usb_sysfs_device": "1-1",
                    },
                    phone_session_sha256=digest,
                    phone_worker_sha256=digest,
                    qualification_phone_session_sha256=digest,
                    qualification_phone_worker_sha256=digest,
                    host_binary_path=host_binary,
                    qualification_binary_path=qualification_binary,
                    transport_client_source_path=transport_source,
                    qualified_allocators=("devmem",),
                    receipt_paths=tuple(
                        sorted(root.glob("receipt-*.json"))
                    ),
                    minimum_usb_speed_mbps=5_000,
                    host_dependency_paths={
                        "server-impl": host_dependency
                    },
                )


if __name__ == "__main__":
    unittest.main()
