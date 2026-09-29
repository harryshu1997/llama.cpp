#!/usr/bin/env python3

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


HERE = Path(__file__).resolve().parent
WAVEFRONT_DIR = (
    HERE.parent
    / "full_fp16_burstgpt_v1"
    / "gpu_wavefront_v1"
)
sys.path.insert(0, str(WAVEFRONT_DIR))

from analyze_wavefront_run import (  # noqa: E402
    AnalysisError,
    bridge_result,
    expected_ffn_shapes,
    ffn_split_result,
    phone_residency_receipt,
    router_result,
)
from analyze_phone_arbiter_abba import saving_pct  # noqa: E402
from analyze_gpu_prefix_wavefront_run import (  # noqa: E402
    completion_boundary_skew_ms,
    require_phase_affinity,
    require_process_affinity_receipt,
    server_ffn_result,
    valid_prefill_execution,
)
from run_gemma_wavefront_driver import (  # noqa: E402
    COMMAND_SCHEMA_V2,
    COMMAND_SCHEMA_V3,
    DriverError,
    command,
    read_reply,
    switch_affinity_after_receipt,
)


class PhoneArbiterWavefrontTests(unittest.TestCase):
    @staticmethod
    def reply_process(value: dict[str, object]) -> object:
        class Process:
            stdout = io.StringIO(json.dumps(value) + "\n")

        return Process()

    def test_dynamic_ffn_command_binds_both_widths(self) -> None:
        value = command(
            7,
            [1, 2, 3],
            4,
            "DETACH",
            ffn_prefill_columns=0,
            ffn_decode_columns=6144,
        )
        self.assertEqual(value["schema"], COMMAND_SCHEMA_V3)
        self.assertEqual(value["ffn_prefill_columns"], 0)
        self.assertEqual(value["ffn_decode_columns"], 6144)

    def test_static_command_retains_v2_schema(self) -> None:
        value = command(1, [1], 1, "DETACH")
        self.assertEqual(value["schema"], COMMAND_SCHEMA_V2)
        self.assertNotIn("ffn_prefill_columns", value)

    def test_dynamic_ffn_command_rejects_an_incomplete_width_pair(self) -> None:
        with self.assertRaisesRegex(DriverError, "widths are incomplete"):
            command(
                1,
                [1],
                1,
                "DETACH",
                ffn_prefill_columns=0,
            )

    def test_streaming_reply_requires_chunk_receipt(self) -> None:
        process = self.reply_process({
            "batch_size": 1,
            "launch_id": 2,
            "outcome": "completed",
            "request_count": 1,
            "schema": "layersplit-persistent-result-v3",
        })
        with self.assertRaisesRegex(DriverError, "chunked prefill receipt"):
            read_reply(
                process,
                2,
                dynamic_ffn=True,
                prefill_mode="async_stream",
            )

    def test_streaming_reply_accepts_positive_chunk_receipt(self) -> None:
        process = self.reply_process({
            "batch_size": 1,
            "launch_id": 2,
            "outcome": "completed",
            "prefill_mode": "async_stream",
            "prefill_chunk_tokens": 8,
            "prefill_chunks": 34,
            "prefill_max_ready_depth": 12,
            "prefill_overlap_us": 100,
            "prefill_pipeline_wall_us": 2900,
            "prefill_stage_us": 1000,
            "prefill_tail_us": 2000,
            "prefill_us": 3000,
            "request_count": 1,
            "schema": "layersplit-persistent-result-v3",
        })
        value = read_reply(
            process,
            2,
            dynamic_ffn=True,
            prefill_mode="async_stream",
        )
        self.assertEqual(value["prefill_chunks"], 34)
        self.assertEqual(value["prefill_max_ready_depth"], 12)

    def test_streaming_reply_rejects_inconsistent_pipeline_times(self) -> None:
        process = self.reply_process({
            "batch_size": 1,
            "launch_id": 2,
            "outcome": "completed",
            "prefill_mode": "async_stream",
            "prefill_chunk_tokens": 8,
            "prefill_chunks": 34,
            "prefill_max_ready_depth": 12,
            "prefill_overlap_us": 100,
            "prefill_pipeline_wall_us": 2900,
            "prefill_stage_us": 1000,
            "prefill_tail_us": 2000,
            "prefill_us": 2999,
            "request_count": 1,
            "schema": "layersplit-persistent-result-v3",
        })
        with self.assertRaisesRegex(DriverError, "chunked prefill receipt"):
            read_reply(
                process,
                2,
                dynamic_ffn=True,
                prefill_mode="async_stream",
            )

    def test_sync_chunked_reply_binds_mode_and_depth(self) -> None:
        process = self.reply_process({
            "batch_size": 1,
            "launch_id": 2,
            "outcome": "completed",
            "prefill_chunk_tokens": 8,
            "prefill_chunks": 34,
            "prefill_max_ready_depth": 1,
            "prefill_mode": "sync_chunked",
            "prefill_overlap_us": 0,
            "prefill_pipeline_wall_us": 3000,
            "prefill_stage_us": 1000,
            "prefill_tail_us": 2000,
            "prefill_us": 3000,
            "request_count": 1,
            "schema": "layersplit-persistent-result-v3",
        })
        value = read_reply(
            process,
            2,
            dynamic_ffn=True,
            prefill_mode="sync_chunked",
        )
        self.assertEqual(value["prefill_mode"], "sync_chunked")
        self.assertEqual(value["prefill_max_ready_depth"], 1)

    def test_bridge_and_terminal_router_receipts_are_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bridge = root / "bridge.log"
            router = root / "router.log"
            bridge.write_text(
                'PHONEARBITER {"status":"MECHANICS_ONLY"}\n',
                encoding="ascii",
            )
            router.write_text(
                'RESIDENTROUTER {"status":"ok",'
                '"terminate_requested":true}\n',
                encoding="ascii",
            )
            self.assertEqual(
                bridge_result(bridge)["status"],
                "MECHANICS_ONLY",
            )
            self.assertTrue(
                router_result(router)["terminate_requested"]
            )
            bridge.write_text(
                bridge.read_text(encoding="ascii")
                + 'FFNDMABUF {"status":"ok"}\n',
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                AnalysisError,
                "bridge result is not unique",
            ):
                bridge_result(bridge)

    def test_ffn_summary_is_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "driver.stderr"
            path.write_text(
                'FFNSPLIT {"status":"FFN_OVERLAP_OK"}\n',
                encoding="ascii",
            )
            self.assertEqual(
                ffn_split_result(path)["status"],
                "FFN_OVERLAP_OK",
            )
            path.write_text(
                path.read_text(encoding="ascii")
                + 'FFNSPLIT {"status":"FFN_OVERLAP_OK"}\n',
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                AnalysisError,
                "FFN split result is not unique",
            ):
                ffn_split_result(path)

    def test_server_ffn_summary_ignores_ready_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "server.stderr"
            path.write_text(
                "S41SERVERFFN ready tail_fence=enabled\n"
                'S41SERVERFFN {"status":"ok","tail_fence_calls":3}\n',
                encoding="ascii",
            )
            self.assertEqual(server_ffn_result(path)["tail_fence_calls"], 3)
            path.write_text(
                path.read_text(encoding="ascii")
                + 'S41SERVERFFN {"status":"ok"}\n',
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                AnalysisError,
                "Qwen FFN summary is not unique",
            ):
                server_ffn_result(path)

    def test_completion_boundary_accepts_bounded_thread_race(self) -> None:
        self.assertEqual(completion_boundary_skew_ms(115_000_000, 100_000_000), 15.0)
        self.assertEqual(completion_boundary_skew_ms(85_000_000, 100_000_000), -15.0)
        with self.assertRaisesRegex(AnalysisError, "completion boundary skew"):
            completion_boundary_skew_ms(200_000_001, 100_000_000)

    def test_control_accepts_bounded_zero_overlap_pipeline(self) -> None:
        self.assertTrue(valid_prefill_execution(
            "control", "async_stream", 36_017_377, 40_523_774,
            76_714_140, 0, 33, 34,
        ))
        self.assertFalse(valid_prefill_execution(
            "mechanics", "async_stream", 36_017_377, 40_523_774,
            76_714_140, 0, 33, 34,
        ))
        self.assertFalse(valid_prefill_execution(
            "control", "async_stream", 36_000_000, 40_000_000,
            77_000_001, 0, 33, 34,
        ))

    def test_phase_affinity_receipt_binds_protected_and_target_cpus(self) -> None:
        record = {
            "at_paid_start": {"cpu_sets": {"2,3": 4}, "thread_count": 4},
            "at_ready": {"cpu_sets": {"2,3": 4}, "thread_count": 4},
            "post_protected_switch": {
                "after": {"cpu_sets": {"0,1": 4}, "thread_count": 4},
                "before": {"cpu_sets": {"2,3": 4}, "thread_count": 4},
                "protected_done_ns": 100,
                "status": "PASS",
                "switch_completed_ns": 130,
                "switch_started_ns": 110,
            },
            "protected_requested_cpus": [2, 3],
            "requested_cpus": [0, 1],
        }
        switch = require_phase_affinity(
            record, {2, 3}, {0, 1}, protected_done_ns=100, paid_end_ns=200
        )
        self.assertEqual(switch["switch_completed_ns"], 130)

    def test_phase_affinity_rejects_switch_before_protected_done(self) -> None:
        record = {
            "at_paid_start": {"cpu_sets": {"2": 1}, "thread_count": 1},
            "at_ready": {"cpu_sets": {"2": 1}, "thread_count": 1},
            "post_protected_switch": {
                "after": {"cpu_sets": {"0": 1}, "thread_count": 1},
                "before": {"cpu_sets": {"2": 1}, "thread_count": 1},
                "protected_done_ns": 100,
                "status": "PASS",
                "switch_completed_ns": 95,
                "switch_started_ns": 90,
            },
            "protected_requested_cpus": [2],
            "requested_cpus": [0],
        }
        with self.assertRaisesRegex(AnalysisError, "switch boundary"):
            require_phase_affinity(
                record, {2}, {0}, protected_done_ns=100, paid_end_ns=200
            )

    def test_physical_affinity_switch_moves_a_live_child(self) -> None:
        available = sorted(os.sched_getaffinity(0))
        if len(available) < 2:
            self.skipTest("two CPUs are required")
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )
        try:
            os.sched_setaffinity(process.pid, {available[0]})
            with tempfile.TemporaryDirectory() as temporary:
                receipt = Path(temporary) / "protected.done"
                receipt.write_text(f"{time.monotonic_ns()}\n", encoding="ascii")
                result: dict[str, object] = {}
                switch_affinity_after_receipt(
                    process,
                    receipt,
                    (available[0],),
                    (available[1],),
                    1.0,
                    result,
                )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(os.sched_getaffinity(process.pid), {available[1]})
        finally:
            process.terminate()
            process.wait()

    def test_process_affinity_receipt_binds_each_named_process(self) -> None:
        record = {
            "captured_ns": 100,
            "processes": {
                "gpu-stage": {
                    "cmdline_sha256": "1" * 64,
                    "cpu_sets": {"1": 3},
                    "expected_cpus": [1],
                    "pid": 10,
                    "start_ticks": 20,
                    "thread_count": 3,
                },
                "wavefront-gate": {
                    "cmdline_sha256": "2" * 64,
                    "cpu_sets": {"3": 1},
                    "expected_cpus": [3],
                    "pid": 11,
                    "start_ticks": 21,
                    "thread_count": 1,
                },
            },
            "schema": "s42-process-affinity-receipt-v1",
            "status": "PASS",
        }
        processes = require_process_affinity_receipt(
            record, {"gpu-stage": {1}, "wavefront-gate": {3}}
        )
        self.assertEqual(processes["gpu-stage"]["pid"], 10)
        record["processes"]["gpu-stage"]["cpu_sets"] = {"1,2": 3}
        with self.assertRaisesRegex(AnalysisError, "gpu-stage"):
            require_process_affinity_receipt(
                record, {"gpu-stage": {1}, "wavefront-gate": {3}}
            )

    def test_prefill_shapes_bind_each_physical_ubatch(self) -> None:
        rows = [{"input_tokens": 35, "output_tokens": 4}]
        self.assertEqual(
            expected_ffn_shapes(rows, 16, True),
            {1: 69, 3: 23, 16: 46},
        )
        self.assertEqual(
            expected_ffn_shapes(rows, 16, False),
            {1: 69},
        )

    def test_m1_prefill_merges_with_decode_shape(self) -> None:
        rows = [{"input_tokens": 3, "output_tokens": 4}]
        self.assertEqual(
            expected_ffn_shapes(rows, 1, True),
            {1: 138},
        )

    def test_phone_residency_binds_vmem_and_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / "session.log"
            workers = root / "workers.log"
            session.write_text(
                "[resident-session] HTP0+HTP1+HTP2 WARM "
                "mem_available_kib=2200000\n",
                encoding="ascii",
            )
            workers.write_text(
                "HTP0 op batching: n-bufs 16 vmem 3422552064\n"
                "HTP1 op batching: n-bufs 16 vmem 3422552064\n"
                "HTP2 op batching: n-bufs 16 vmem 3422552064\n"
                'RESIDENTWORKERS {"status":"WARM",'
                '"layout":"gemma46-qwen6-full-v1",'
                '"sessions":["HTP0","HTP1","HTP2"],'
                '"gemma_layers":"0-45","qwen_layers":"0-5",'
                '"qwen_columns":17408}\n',
                encoding="ascii",
            )
            receipt = phone_residency_receipt(
                session,
                workers,
                expected_vmem_mib=3264,
                minimum_available_kib=2097152,
                expected_layout="gemma46-qwen6-full-v1",
            )
            self.assertEqual(receipt["available_kib"], 2200000)
            self.assertEqual(
                receipt["workers"]["gemma_layers"], "0-45"
            )
            with self.assertRaisesRegex(
                AnalysisError, "resident layout receipt"
            ):
                phone_residency_receipt(
                    session,
                    workers,
                    expected_vmem_mib=3264,
                    minimum_available_kib=2097152,
                    expected_layout="gemma23-qwen12-full-v1",
                )

    def test_abba_saving_uses_control_as_the_denominator(self) -> None:
        self.assertAlmostEqual(saving_pct(100.0, 75.0), 25.0)


if __name__ == "__main__":
    unittest.main()
