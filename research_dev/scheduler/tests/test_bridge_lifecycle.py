#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from research_dev.scheduler.adapters import (
    AndroidUsbRestorationReceipt,
    PhysicalAdapterError,
    DirectPhoneFfnSession,
    close_functionfs_bridge,
    parse_functionfs_bridge_qualification,
    probe_functionfs_usb_device,
    parse_direct_phone_ffn_terminal,
    parse_phone_residency_call_events,
    parse_phone_residency_phase_events,
    verify_android_usb_restored,
)
from research_dev.scheduler.adapters.phone_transport import (
    PhoneTransportContract,
)


class BridgeLifecycleTests(unittest.TestCase):
    @staticmethod
    def terminal() -> str:
        return (
            'FFNDMABUF {"status":"ok","calls":12,'
            '"allocator":"malloc-split","reset_recoveries":0,'
            '"upload_bytes":4096,"download_bytes":4096}'
        )

    def test_exited_clean_bridge_is_not_closed_twice(self) -> None:
        shutdown_calls = []
        finalized = []
        receipt = close_functionfs_bridge(
            poll=lambda: 0,
            stderr_lines=lambda: (self.terminal(),),
            request_shutdown=lambda: shutdown_calls.append(True) or 0,
            finalize=lambda: finalized.append(True),
        )
        self.assertEqual(shutdown_calls, [])
        self.assertEqual(finalized, [True])
        self.assertEqual(receipt.status, "ok")
        self.assertEqual(receipt.calls, 12)
        self.assertEqual(receipt.reset_recoveries, 0)

    def test_live_bridge_uses_close_helper_once(self) -> None:
        state = {"returncode": None}
        shutdown_calls = []

        def shutdown() -> int:
            shutdown_calls.append(True)
            state["returncode"] = 0
            return 0

        receipt = close_functionfs_bridge(
            poll=lambda: state["returncode"],
            stderr_lines=lambda: (self.terminal(),),
            request_shutdown=shutdown,
            finalize=lambda: None,
        )
        self.assertEqual(shutdown_calls, [True])
        self.assertEqual(receipt.allocator, "malloc-split")

    def test_qualification_receipt_keeps_exact_runtime_path(self) -> None:
        receipt = parse_functionfs_bridge_qualification((
            'FFNDMABUFQUAL {"status":"ok","calls":8,'
            '"allocator":"malloc-split","queue_depth":1,'
            '"reset_recoveries":0,"upload_bytes":270336,'
            '"download_bytes":270336,'
            '"h2d_payload_bytes_per_s":123000000.0,'
            '"d2h_conservative_payload_bytes_per_s":81000000.0,'
            '"d2h_exposed_payload_bytes_per_s":102000000.0,'
            '"full_duplex_payload_bytes_per_s":0.0,'
            '"full_duplex_measured":false}',
        ))
        self.assertEqual(receipt["allocator"], "malloc-split")
        self.assertEqual(receipt["queue_depth"], 1)
        self.assertFalse(receipt["full_duplex_measured"])

    def test_direct_phone_terminal_keeps_d2h_completion_metrics(self) -> None:
        receipt = parse_direct_phone_ffn_terminal((
            "[ffn-worker] DMA-BUF complete transport=direct requests=8 "
            "queue_depth=4 maximum_pending_outputs=4 "
            "phone_payload_copies=16 "
            "d2h_completions=8 d2h_queue_us=1200 "
            "d2h_queue_max_us=250 recoveries=0 status=0",
        ))

        self.assertEqual(receipt.queue_depth, 4)
        self.assertEqual(receipt.maximum_pending_outputs, 4)
        self.assertEqual(receipt.phone_payload_copies, 16)
        self.assertEqual(receipt.d2h_completions, 8)
        self.assertEqual(receipt.d2h_queue_us, 1200)
        self.assertEqual(receipt.d2h_queue_max_us, 250)

    def test_phone_residency_phase_keeps_physical_clock_and_epoch(self) -> None:
        artifact = "sha256:" + "a" * 64
        receipt = parse_phone_residency_phase_events((
            "RESIDENTPHASE {\"artifact_sha256\":\"" + artifact
            + "\",\"component\":\"ffn-worker\",\"epoch_us\":200,"
            "\"monotonic_us\":100,\"phase\":\"HTP_INIT_READY\","
            "\"schema\":\"s42-phone-residency-phase-v1\","
            "\"session_generation\":2,\"session_id\":\"HTP0\"}",
        ))[0]

        self.assertEqual(receipt.session_id, "HTP0")
        self.assertEqual(receipt.session_generation, 2)
        self.assertEqual(receipt.monotonic_us, 100)
        self.assertEqual(receipt.epoch_us, 200)

    def test_phone_residency_call_keeps_session_epoch_and_counter(self) -> None:
        artifact = "sha256:" + "b" * 64
        receipt = parse_phone_residency_call_events((
            "RESIDENTCALL {\"artifact_sha256\":\"" + artifact
            + "\",\"calls\":16,\"epoch_us\":400,"
            "\"monotonic_us\":300,"
            "\"schema\":\"s42-phone-residency-call-v1\","
            "\"session_generation\":3,\"session_id\":\"HTP1\"}",
        ))[0]

        self.assertEqual(receipt.session_id, "HTP1")
        self.assertEqual(receipt.session_generation, 3)
        self.assertEqual(receipt.calls, 16)
        self.assertEqual(receipt.monotonic_us, 300)
        self.assertEqual(receipt.epoch_us, 400)

    def test_progressive_launch_exposes_all_discovered_htp_sessions(
        self,
    ) -> None:
        session = object.__new__(DirectPhoneFfnSession)
        session.configuration = SimpleNamespace(
            multi_session_device_count=3,
            android_gadget_path=None,
            diagnostic_port=0,
        )
        words = []

        session._append_start_environment(words)

        self.assertEqual(words, ["S42_PHONE_SESSION_COUNT=3"])

    def test_functionfs_usb_probe_records_unique_speed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = root / "2-2"
            device.mkdir()
            (device / "idVendor").write_text("18d1\n", encoding="ascii")
            (device / "idProduct").write_text("2d00\n", encoding="ascii")
            (device / "speed").write_text("5000\n", encoding="ascii")
            observed = probe_functionfs_usb_device(
                vendor_id="18D1",
                product_id="2D00",
                sysfs_root=root,
            )
        self.assertEqual(observed.sysfs_device, "2-2")
        self.assertEqual(observed.negotiated_speed_mbps, 5000)
        self.assertEqual(observed.vendor_id, "18d1")

    def test_functionfs_usb_probe_fails_on_ambiguous_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("2-2", "3-1"):
                device = root / name
                device.mkdir()
                (device / "idVendor").write_text(
                    "18d1\n", encoding="ascii"
                )
                (device / "idProduct").write_text(
                    "2d00\n", encoding="ascii"
                )
                (device / "speed").write_text(
                    "5000\n", encoding="ascii"
                )
            with self.assertRaises(PhysicalAdapterError):
                probe_functionfs_usb_device(
                    vendor_id="18d1",
                    product_id="2d00",
                    sysfs_root=root,
                )

    def test_android_usb_restore_binds_adb_to_normal_gadget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = root / "2-2"
            device.mkdir()
            (device / "serial").write_text(
                "SYNTHETIC123\n", encoding="ascii"
            )
            (device / "idVendor").write_text("18d1\n", encoding="ascii")
            (device / "idProduct").write_text("4ee7\n", encoding="ascii")
            (device / "speed").write_text("5000\n", encoding="ascii")
            receipt = verify_android_usb_restored(
                serial="SYNTHETIC123",
                adb_port=5037,
                minimum_speed_mbps=5000,
                timeout_s=1,
                sysfs_root=root,
                adb_probe=lambda: (0, "device"),
            )
        self.assertEqual(receipt.sysfs_device, "2-2")
        self.assertEqual(receipt.product_id, "4ee7")
        self.assertEqual(receipt.to_json()["status"], "RESTORED")

    def test_android_usb_zero_timeout_checks_once_without_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            probe = mock.Mock(return_value=(1, "offline"))
            with mock.patch("research_dev.scheduler.adapters.bridge.time.sleep") as sleep:
                with self.assertRaisesRegex(PhysicalAdapterError, "ADB state is offline"):
                    verify_android_usb_restored(
                        serial="SYNTHETIC123", adb_port=5037,
                        minimum_speed_mbps=5000, timeout_s=0,
                        sysfs_root=Path(directory), adb_probe=probe,
                    )
                probe.assert_called_once()
                sleep.assert_not_called()

    def test_direct_session_reports_worker_failure_before_usb_timeout(
        self,
    ) -> None:
        session = object.__new__(DirectPhoneFfnSession)
        with mock.patch.object(
            session,
            "_adb_root",
            return_value="TERMINAL 137\n",
        ):
            failure = session._remote_launch_failure("/data/local/tmp/run")
        self.assertEqual(
            failure,
            "phone session terminated before USB enumeration: status=137",
        )

    def test_direct_session_preflights_worker_dynamic_dependencies(
        self,
    ) -> None:
        session = object.__new__(DirectPhoneFfnSession)
        commands = []
        session._adb_root = commands.append

        session._validate_worker_loadability((
            "/data/local/tmp/runtime/llama-ffn-split-worker",
        ))

        self.assertEqual(len(commands), 1)
        self.assertIn(
            "LD_LIBRARY_PATH=/data/local/tmp/runtime", commands[0]
        )
        self.assertIn(
            "/data/local/tmp/runtime/llama-ffn-split-worker",
            commands[0],
        )
        self.assertIn('worker_status\" -ne 2', commands[0])

    def test_direct_session_rejects_unloadable_worker(self) -> None:
        session = object.__new__(DirectPhoneFfnSession)
        session._adb_root = mock.Mock(side_effect=PhysicalAdapterError(
            "CANNOT LINK EXECUTABLE: libggml.so not found"
        ))

        with self.assertRaisesRegex(
            PhysicalAdapterError,
            "direct phone worker is not loadable.*libggml.so not found",
        ):
            session._validate_worker_loadability((
                "/data/local/tmp/runtime/llama-ffn-split-worker",
            ))

    def test_direct_session_rejects_unqualified_phone_kernel(self) -> None:
        session = object.__new__(DirectPhoneFfnSession)
        session.configuration = SimpleNamespace(
            required_kernel_release="qualified-kernel"
        )
        with mock.patch.object(
            session, "_adb", return_value="stock-kernel\n"
        ):
            with self.assertRaisesRegex(
                PhysicalAdapterError,
                "phone kernel is not qualified",
            ):
                session._phone_kernel_release()

    def test_direct_session_uses_ticket_transport_queue_depth(self) -> None:
        transport = PhoneTransportContract(
            transport="functionfs-usb",
            allocator="malloc",
            queue_depth=4,
            concurrent_streams=4,
            max_payload_bytes=10240,
            full_duplex=True,
            split_h2d=False,
            generation="synthetic-v1",
            profile_id="synthetic-profile",
            usbfs_available_bytes=16 * 1024 * 1024,
            slot_safety_bytes=4096,
            vendor_id=0x18D1,
            product_id=0x2D00,
            control_host="direct-functionfs",
            control_port=0,
        )
        environment = DirectPhoneFfnSession._worker_environment(
            SimpleNamespace(max_tokens=4), transport, 1280
        )

        self.assertIn("S41_FFN_QUEUE_DEPTH=4", environment)
        self.assertIn("S41_FFN_COLUMN_QUANTUM=1280", environment)

    def test_direct_session_binds_only_matching_resident_shards(self) -> None:
        session = object.__new__(DirectPhoneFfnSession)
        transport = mock.Mock(spec=PhoneTransportContract)
        transport.shares_resident_session_with.return_value = True
        artifact = "sha256:" + "a" * 64
        execution = SimpleNamespace(layer_mask=0b1111, columns=8)
        shards = (
            SimpleNamespace(
                artifact_sha256=artifact,
                layer_mask=0b0011,
                maximum_columns=8,
            ),
            SimpleNamespace(
                artifact_sha256=artifact,
                layer_mask=0b1100,
                maximum_columns=8,
            ),
        )
        session._launch = SimpleNamespace(
            execution=execution,
            phone_shards=shards,
            remote_hashes={
                "model": artifact,
                "model:" + artifact: artifact,
            },
            transport=transport,
        )
        session._remote_root = "/data/local/tmp/resident"
        session._bound_ticket_ids = ["ticket-a"]
        manifest = SimpleNamespace(artifact_sha256=artifact)
        command = SimpleNamespace(
            artifact_sha256=artifact,
            execution_contract=SimpleNamespace(phone_shards=shards),
            ticket_id="ticket-b",
        )

        with mock.patch(
            "research_dev.scheduler.adapters.phone_session_ops.identity."
            "phone_ffn_resident_contract",
            return_value=execution,
        ):
            session.bind(command, manifest, transport)
            session.bind(command, manifest, transport)
            incompatible = SimpleNamespace(
                artifact_sha256=artifact,
                execution_contract=SimpleNamespace(
                    phone_shards=(object(),)
                ),
                ticket_id="ticket-c",
            )
            with self.assertRaisesRegex(
                PhysicalAdapterError,
                "residency differs from the ticket",
            ):
                session.bind(incompatible, manifest, transport)

        self.assertEqual(
            session._bound_ticket_ids,
            ["ticket-a", "ticket-b"],
        )

    def test_direct_session_connects_diagnostic_ncm(self) -> None:
        session = object.__new__(DirectPhoneFfnSession)
        session.configuration = SimpleNamespace(
            launch_timeout_s=1,
            network_manager_path=Path("/usr/bin/nmcli"),
        )
        with mock.patch.object(
            session, "_ncm_interfaces", return_value=("enxtest",)
        ), mock.patch.object(
            session, "_diagnostic_available", side_effect=(False, True)
        ), mock.patch(
            "research_dev.scheduler.adapters.phone_session.subprocess.run"
        ) as connect:
            interface = session._connect_diagnostic_ncm(
                SimpleNamespace(sysfs_device="2-2")
            )

        self.assertEqual(interface, "enxtest")
        connect.assert_called_once_with(
            ["/usr/bin/nmcli", "device", "connect", "enxtest"],
            check=False,
            stdout=mock.ANY,
            stderr=mock.ANY,
            timeout=15,
        )

    def test_direct_session_reuses_mixed_manifest_by_artifact(self) -> None:
        session = object.__new__(DirectPhoneFfnSession)
        first_artifact = "sha256:" + "a" * 64
        second_artifact = "sha256:" + "b" * 64
        transport = mock.Mock(spec=PhoneTransportContract)
        transport.shares_resident_session_with.return_value = True
        shards = (
            SimpleNamespace(
                artifact_sha256=first_artifact,
                layer_mask=0b0011,
                maximum_columns=8,
            ),
            SimpleNamespace(
                artifact_sha256=second_artifact,
                layer_mask=0b0011,
                maximum_columns=16,
            ),
        )
        session._launch = SimpleNamespace(
            execution=SimpleNamespace(layer_mask=0b0011, columns=8),
            phone_shards=shards,
            remote_hashes={
                "model:" + first_artifact: first_artifact,
                "model:" + second_artifact: second_artifact,
            },
            transport=transport,
        )
        session._remote_root = "/data/local/tmp/resident"
        session._bound_ticket_ids = ["ticket-a"]
        command = SimpleNamespace(
            artifact_sha256=second_artifact,
            execution_contract=SimpleNamespace(phone_shards=shards),
            ticket_id="ticket-b",
        )

        with mock.patch(
            "research_dev.scheduler.adapters.phone_session_ops.identity."
            "phone_ffn_resident_contract",
            return_value=SimpleNamespace(layer_mask=0b0011, columns=16),
        ):
            session.bind(
                command,
                SimpleNamespace(artifact_sha256=second_artifact),
                transport,
            )

        self.assertEqual(
            session._bound_ticket_ids, ["ticket-a", "ticket-b"]
        )
    def test_direct_session_abort_does_not_require_worker_log(self) -> None:
        session = object.__new__(DirectPhoneFfnSession)
        session.configuration = SimpleNamespace(
            adb_port=5037,
            minimum_usb_speed_mbps=5000,
            serial="SYNTHETIC123",
        )
        session._launch = object()
        session._remote_root = "/data/local/tmp/run"
        restored = AndroidUsbRestorationReceipt(
            serial="SYNTHETIC123",
            adb_port=5037,
            sysfs_device="2-2",
            vendor_id="22d9",
            product_id="2772",
            negotiated_speed_mbps=5000,
        )
        with mock.patch(
            "research_dev.scheduler.adapters.phone_session_ops.completion."
            "verify_android_usb_restored",
            return_value=restored,
        ), mock.patch.object(
            session,
            "_adb",
            side_effect=AssertionError("unexpected worker log read"),
        ):
            self.assertIs(session.abort(), restored)
        self.assertFalse(session.active)

    def test_direct_session_abort_recovers_after_phone_reboot(self) -> None:
        session = object.__new__(DirectPhoneFfnSession)
        session.configuration = SimpleNamespace(
            adb_port=5037,
            minimum_usb_speed_mbps=5000,
            serial="SYNTHETIC123",
        )
        session._launch = SimpleNamespace(execution=object(), transport=object())
        session._remote_root = "/data/local/tmp/run"
        restored = AndroidUsbRestorationReceipt(
            serial="SYNTHETIC123",
            adb_port=5037,
            sysfs_device="2-2",
            vendor_id="22d9",
            product_id="2772",
            negotiated_speed_mbps=5000,
        )
        with mock.patch(
            "research_dev.scheduler.adapters.phone_session_ops.completion."
            "verify_android_usb_restored",
            side_effect=(
                PhysicalAdapterError("ADB is restarting"),
                restored,
            ),
        ) as verify, mock.patch.object(
            session,
            "_close_direct_usb",
            side_effect=PhysicalAdapterError("FunctionFS is gone"),
        ) as close:
            self.assertIs(session.abort(), restored)

        self.assertFalse(session.active)
        self.assertEqual(verify.call_args_list[1].kwargs["timeout_s"], 180)
        close.assert_called_once()

    def test_direct_session_finish_waits_after_redundant_close(self) -> None:
        session = object.__new__(DirectPhoneFfnSession)
        session.configuration = SimpleNamespace(
            adb_port=5037,
            minimum_usb_speed_mbps=5000,
            serial="SYNTHETIC123",
        )
        session._launch = SimpleNamespace(
            phone_shards=(), ticket_id="ticket-a"
        )
        session._remote_root = "/data/local/tmp/run"
        session._bound_ticket_ids = ["ticket-a"]
        session._proof_shards = []
        session._executed_proof_shards = []
        session._execution_by_artifact = {}
        session._transport_by_artifact = {}
        session._load_count_by_session = {}
        session._max_tokens_by_session = {}
        restored = AndroidUsbRestorationReceipt(
            serial="SYNTHETIC123",
            adb_port=5037,
            sysfs_device="2-2",
            vendor_id="22d9",
            product_id="2772",
            negotiated_speed_mbps=5000,
        )
        terminal = (
            "[ffn-worker] DMA-BUF complete transport=direct requests=1 "
            "queue_depth=1 maximum_pending_outputs=0 "
            "phone_payload_copies=2 d2h_completions=1 "
            "d2h_queue_us=0 d2h_queue_max_us=0 "
            "recoveries=0 status=0\n"
        )
        with mock.patch(
            "research_dev.scheduler.adapters.phone_session_ops.completion."
            "verify_android_usb_restored",
            side_effect=(
                PhysicalAdapterError("ADB is restarting"),
                restored,
            ),
        ) as verify, mock.patch.object(
            session,
            "_close_current_usb",
            side_effect=PhysicalAdapterError("HELLO identity mismatch"),
        ) as close, mock.patch.object(
            session, "_adb", return_value=terminal
        ):
            receipt = session.finish(require_execution=False)

        self.assertIs(receipt.restoration, restored)
        self.assertFalse(session.active)
        self.assertEqual(verify.call_args_list[0].kwargs["timeout_s"], 0)
        self.assertEqual(verify.call_args_list[1].kwargs["timeout_s"], 90)
        close.assert_called_once()

    def test_direct_session_usb_close_timeout_is_adapter_failure(self) -> None:
        session = object.__new__(DirectPhoneFfnSession)
        session.configuration = SimpleNamespace(
            usb_close_path=Path("/synthetic/usb-close")
        )
        execution = SimpleNamespace(
            activation="geglu",
            columns=128,
            layer_mask=1,
            max_tokens=4,
            n_embd=64,
        )
        transport = SimpleNamespace(
            generation="synthetic-v1",
            product_id=0x2D00,
            usbfs_available_bytes=1024,
            vendor_id=0x18D1,
        )
        with mock.patch(
            "research_dev.scheduler.adapters.phone_session.subprocess.run",
            side_effect=subprocess.TimeoutExpired("usb-close", 30),
        ), self.assertRaisesRegex(
            PhysicalAdapterError, "USB close timed out"
        ):
            session._close_direct_usb(execution, transport)

    def test_failed_direct_abort_still_clears_session_state(self) -> None:
        session = object.__new__(DirectPhoneFfnSession)
        session.configuration = SimpleNamespace(
            adb_port=5037,
            minimum_usb_speed_mbps=5000,
            serial="SYNTHETIC123",
        )
        session._launch = SimpleNamespace(execution=object(), transport=object())
        session._remote_root = "/data/local/tmp/run"
        with mock.patch(
            "research_dev.scheduler.adapters.phone_session_ops.completion."
            "verify_android_usb_restored",
            side_effect=PhysicalAdapterError("ADB is absent"),
        ), mock.patch.object(
            session,
            "_close_direct_usb",
            side_effect=PhysicalAdapterError("FunctionFS is absent"),
        ):
            with self.assertRaisesRegex(
                PhysicalAdapterError, "FunctionFS is absent"
            ):
                session.abort()

        self.assertFalse(session.active)


if __name__ == "__main__":
    unittest.main()
