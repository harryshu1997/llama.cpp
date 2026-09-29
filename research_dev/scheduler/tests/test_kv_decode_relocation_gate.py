"""Launch validation for the bounded decode-relocation measurement gate."""
import importlib.util
import json
import struct
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from research_dev.scheduler.adapters.contracts import PhysicalAdapterError


GATE = Path(__file__).resolve().parents[1] / "campaigns/burstgpt/reports/20260917-decode-relocation-kv-headroom/kv_decode_relocation_gate.py"
SPEC = importlib.util.spec_from_file_location("kv_decode_relocation_gate", GATE)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class GateLaunchTests(unittest.TestCase):
    def contract(self, **options):
        return gate.launch_contract(
            {"batch": 2048, "ubatch": 128, "gpu_layers": 16, **options},
            SimpleNamespace(context_size=32768, cpu_layers=tuple(range(32))),
            SimpleNamespace(model_id="qwen"), False, {})

    def test_default_and_invalid_cpu_settings(self):
        self.assertEqual(self.contract().threads, 0)
        self.assertIsNone(self.contract().cpu_affinity)
        for options in ({"threads": 8}, {"threads": "8", "threads_batch": 8},
                        {"threads": -1, "threads_batch": 8}, {"cpu_affinity": ""}):
            with self.subTest(options=options), self.assertRaises(PhysicalAdapterError):
                self.contract(**options)

    def record(self, output, contract, listener_pid=123, command=b"llama-server\0--port\0" b"4567\0"):
        managed = SimpleNamespace(pid=123, command=("taskset", "--cpu-list", "0,2", "llama-server", "--port", "4567"),
                                  launch_contract=contract)
        with patch.object(Path, "read_bytes", return_value=command), patch.object(
                gate.subprocess, "check_output", return_value=f'LISTEN users:(("llama-server",pid={listener_pid},fd=3))'):
            gate.record_server_identity(managed, 4567, output, {"server": "sha256:" + "a" * 64})

    def test_cpu_settings_enter_runtime_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            identities = []
            for threads in (8, 12):
                output = Path(directory) / str(threads)
                output.mkdir()
                self.record(output, self.contract(threads=threads, threads_batch=8, cpu_affinity="0,2"),
                            command=b"llama-server\0--port\0" + b"4567\0")
                identities.append(json.loads((output / "SERVER_IDENTITY.json").read_text()))
            self.assertNotEqual(identities[0]["runtime_launch_sha256"], identities[1]["runtime_launch_sha256"])
            self.assertEqual(identities[1]["launch_contract"]["threads"], 12)

    def test_unrelated_listener_or_command_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            for pid, command in ((456, b"llama-server\0--port\0" + b"4567\0"), (123, b"unrelated\0")):
                with self.subTest(pid=pid), self.assertRaisesRegex(RuntimeError, "answering server process"):
                    self.record(Path(directory), self.contract(), pid, command)
            self.assertFalse((Path(directory) / "SERVER_IDENTITY.json").exists())

    def test_changed_cpu_or_cache_policy_cannot_reuse_an_atlas_environment(self):
        config = {"gpu_layers": 16, "batch": 2048, "ubatch": 128, "column_quantum": 2176,
                  "phone": {"session_masks": {"HTP0": 1}}}
        plan = SimpleNamespace(context_size=32768, artifact_sha256="sha256:" + "a" * 64, to_json=lambda: {})
        runtime = {"server": "sha256:" + "b" * 64}
        original = gate.selection_environment(config, plan, runtime)
        for fields in ({"threads": 12, "threads_batch": 8}, {"cpu_affinity": "0,2"},
                       {"ffn_host_share_drop_cache": 0}, {"ffn_host_share_populate": 0}, {"ffn_max_tokens": 4},
                       {"usb_batch_plan": "coalesced-batch"}, {"ffn_row_diagnostic_steps": 5},
                       {"cohort_submission_order": [1, 0]}, {"logits_trace_path": "/mnt/storage/LOGITS.bin"}):
            with self.subTest(fields=fields):
                changed = gate.selection_environment({**config, **fields}, plan, runtime)
                self.assertNotEqual(changed.runtime_sha256, original.runtime_sha256)
                self.assertEqual(changed.kv_plan_sha256, original.kv_plan_sha256)
        defaults = gate.selection_environment({**config, "ffn_host_share_drop_cache": 1,
            "ffn_host_share_populate": 1, "ffn_max_tokens": 128}, plan, runtime)
        self.assertEqual(defaults, original)

    def test_coalesced_launch_rejects_a_worker_or_payload_smaller_than_slots(self):
        environment = {"S41_SERVER_FFN_USB_BATCH_PLAN": "coalesced-batch", "S41_SERVER_FFN_MAX_TOKENS": "8",
            "S41_SERVER_FFN_N_EMBD": "64", "S41_SERVER_FFN_USB_MAX_PAYLOAD_BYTES": "1024",
            "S41_SERVER_FFN_RUNTIME_CONTROL": "1", "S41_SERVER_FFN_TRANSPORT": "functionfs-usb"}
        def launch(env, parallel=8):
            return gate.launch_contract({"batch": 2048, "ubatch": 1024, "gpu_layers": 16, "parallel": parallel},
                SimpleNamespace(context_size=32768, cpu_layers=()), SimpleNamespace(model_id="qwen"), True, env)
        self.assertEqual(launch(environment).parallel, 8)
        for changed in ({"S41_SERVER_FFN_MAX_TOKENS": "4"}, {"S41_SERVER_FFN_MAX_TOKENS": "513"},
                        {"S41_SERVER_FFN_USB_MAX_PAYLOAD_BYTES": "1023"}, {"S41_SERVER_FFN_MAX_TOKENS": "True"},
                        {"S41_SERVER_FFN_RUNTIME_CONTROL": "0"}):
            with self.subTest(changed=changed), self.assertRaises(PhysicalAdapterError):
                launch({**environment, **changed})
        with self.assertRaises(PhysicalAdapterError):
            launch(environment, 9)

    def test_row_diagnostic_is_typed_and_requires_resident_host_weights(self):
        environment = {"S41_SERVER_FFN_DORMANT_HOST_SHARE": "1", "S41_SERVER_FFN_RUNTIME_CONTROL": "1"}
        def launch(steps, env):
            return gate.launch_contract({"batch": 2048, "ubatch": 1024, "gpu_layers": 16,
                "ffn_row_diagnostic_steps": steps}, SimpleNamespace(context_size=32768, cpu_layers=()),
                SimpleNamespace(model_id="qwen"), True, env)
        self.assertNotIn("S41_SERVER_FFN_ROW_DIAGNOSTIC_STEPS", launch(0, environment).ffn_environment)
        for steps in (5, 64):
            self.assertEqual(launch(steps, environment).ffn_environment["S41_SERVER_FFN_ROW_DIAGNOSTIC_STEPS"], str(steps))
        for steps, env in ((True, environment), (1, environment), (65, environment), ("5", environment), (5, {}), (64, {}),
                           (5, {**environment, "S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK": "1"}),
                           (0, {**environment, "S41_SERVER_FFN_ROW_DIAGNOSTIC_STEPS": "5"})):
            with self.subTest(steps=steps, env=env), self.assertRaises(PhysicalAdapterError):
                launch(steps, env)

    def test_raw_logits_trace_is_typed_and_off_by_default(self):
        self.assertIsNone(self.contract().logits_trace_path)
        self.assertNotIn('S41_SERVER_LOGITS_TRACE', self.contract().ffn_environment)
        self.assertEqual(self.contract(logits_trace_path='/mnt/storage/LOGITS.bin').logits_trace_path,
                         '/mnt/storage/LOGITS.bin')
        for path in (True, 1, '', 'relative.bin', '/tmp/bad\npath'):
            with self.subTest(path=path), self.assertRaises(PhysicalAdapterError):
                self.contract(logits_trace_path=path)

    def test_cohort_proofs_attribute_only_each_requests_rows(self):
        shard = SimpleNamespace(session_id="HTP0", endpoint="session://op15/HTP0", artifact_sha256="sha256:" + "a" * 64,
            resident_geometry_sha256="sha256:" + "b" * 64, operator_plan_sha256="sha256:" + "c" * 64,
            session_generation=1, layer_mask=1)
        manifest = SimpleNamespace(embedding_length=4, block_count=1)
        line = "S41SERVERFFNCALL context=61:0:1:1,62:1:1:1 request=1 layer=0 tokens=2 columns=32 payload_bytes=16"
        proofs, summary = gate.split_phone_proofs([line], (shard,), manifest, {"a": (32, 1), "b": (32, 1)})
        for rid in ("a", "b"):
            self.assertTrue(summary[rid]["exact"])
            self.assertEqual((proofs[rid][0].calls, proofs[rid][0].rows, proofs[rid][0].payload_bytes), (1, 1, 8))
        for lines, expected in (([line, line], {"a": (32, 1), "b": (32, 1)}),
                                ([line], {"a": (32, 1)}), ([line], {"a": (32, 1), "b": (64, 1)})):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                gate.split_phone_proofs(lines, (shard,), manifest, expected)

    def test_watchdog_tracks_each_slot_and_ignores_completed_slots(self):
        watch = gate.DecodeWatchdog(("a", "b"))
        self.assertEqual(watch.stalled(100_000_000_000), {})
        watch.observe("a", 1, 1, False)
        self.assertEqual(watch.stalled(60_000_000_001), {})
        self.assertEqual(set(watch.stalled(60_000_000_002)), {"a", "b"})
        watch.observe("a", 2, 2, True)
        watch.observe("b", 2, 60_000_000_000, False)
        self.assertEqual(watch.stalled(60_000_000_002), {})
        self.assertEqual(set(watch.stalled(120_000_000_001)), {"b"})

    def test_cohort_applies_one_control_and_measures_energy_once(self):
        class Client:
            def __init__(self):
                self.barrier = threading.Barrier(2)
                self.controls = []

            def apply_ffn_cohort_control(self, endpoint, control, members, **kwargs):
                self.controls.append(members)
                return {"cohort_members": [{"request_id": rid, "slot_id": slot, "applied_token_index": 1}
                        for rid, slot in members]}, time.monotonic_ns()

            def complete(self, endpoint, payload, check):
                slot = int(payload.request_id.split('-s')[1].split('-')[0])
                payload.on_first_token(time.monotonic_ns())
                for predicted in (1, 2, 3):
                    payload.on_decode_progress(slot, predicted, time.monotonic_ns(), predicted == 3)
                    self.barrier.wait(timeout=5)
                return {"tokens": [slot, 11, 12], "predicted_ms": 3}

        policy = gate.AdaptiveDecodePolicy(route_id="test", executor_id="test",
            operator_plan_sha256="sha256:" + "a"*64, desktop_parent_route_id="test-parent",
            desktop_placement_sha256="sha256:" + "b"*64, layer_indices=(0,), layer_mask=1, columns=32,
            split_fraction_ppm=1_000_000, resource_ids=("cpu", "phone"))
        client = Client()
        record = {"requests": []}
        with tempfile.TemporaryDirectory() as root, patch.object(gate, '_energy', return_value={"joules": 1}) as energy:
            expected = gate.run_decode_cohort(client, 'http://127.0.0.1:1', SimpleNamespace(model_id="tiny"),
                Path(root), [(1,), (2,)], 3, "combined", 0, policy, None, None, record)
            self.assertEqual(energy.call_count, 2)
        self.assertEqual(len(client.controls), 1)
        self.assertEqual(expected, {"kvd-combined-s0-h0": (32, 2), "kvd-combined-s1-h0": (32, 2)})
        self.assertTrue(record['cohort']['watchdog_passed'])
        self.assertEqual(len(record['requests']), 2)
        self.assertLessEqual(record['cohort']['phone_active_union_s'], record['cohort']['request_s'])

    def test_ordered_cohort_waits_for_allocation_before_first_token(self):
        class Client:
            def __init__(self):
                self.barrier = threading.Barrier(2)
                self.submitted = []

            def allocations(self):
                return [{"slot_id": index, "task_id": index, "line": "allocated", "observed_epoch_us": 0}
                        for index in range(len(self.submitted))]

            def complete(self, endpoint, payload, check):
                slot = len(self.submitted)
                self.submitted.append(payload.request_id)
                self.barrier.wait(timeout=5)
                payload.on_first_token(time.monotonic_ns())
                for predicted in (1, 2, 3):
                    payload.on_decode_progress(slot, predicted, time.monotonic_ns(), predicted == 3)
                    self.barrier.wait(timeout=5)
                return {"tokens": [1, 11, 12], "predicted_ms": 3}

        for order in ((0, 1), (1, 0)):
            client = Client()
            record = {"requests": []}
            with tempfile.TemporaryDirectory() as root, patch.object(gate, '_energy', return_value={"joules": 1}):
                gate.run_decode_cohort(client, 'http://127.0.0.1:1', SimpleNamespace(model_id="tiny"),
                    Path(root), [(1,), (2,)], 3, "control", 32, None, None, None, record, order, client.allocations)
                members = record['cohort']['members']
                self.assertEqual(client.submitted, [members[index] for index in order])
                self.assertEqual([row['slot_id'] for row in record['requests']], [order.index(i) for i in range(2)])
                self.assertEqual(record['cohort']['submission_order'], list(order))
                self.assertEqual(len(record['cohort']['slot_allocation_observations']), 1)
                self.assertEqual(len(record['cohort']['request_fixture_sha256']), 71)
        for order in ((0, 0), (True, 1), (1,), tuple(range(9))):
            with tempfile.TemporaryDirectory() as root, self.assertRaises((PhysicalAdapterError, ValueError)):
                gate.run_decode_cohort(Client(), 'http://127.0.0.1:1', SimpleNamespace(model_id="tiny"),
                    Path(root), [(1,), (2,)], 3, "control", 32, None, None, None, {"requests": []}, order)



class RowDiagnosticCheckTests(unittest.TestCase):
    def setUp(self):
        path = GATE.parent.parent / '20260920-fast-path-M2/ANALYZE.py'
        spec = importlib.util.spec_from_file_location('fast_path_row_check', path)
        self.checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.checker)

    @staticmethod
    def row(payload=0, input_hash='a', returned_hash='b', distance=0.001):
        metadata = {'call': 1, 'layer': 0, 'ubatch_row': payload, 'member': 1-payload,
                    'request_id': f'r{payload}', 'slot_id': payload, 'payload_row': payload,
                    'position': 260, 'decoded': 4, 'applied': 3, 'step': 2, 'wire_type': 'f16'}
        return {stage: {**metadata, 'stage': stage, 'wire_sha256': digest * 64,
                        'f32_sha256': digest * 64, 'local_rel_l2': distance if stage == 'returned' else -1,
                        'local_max_abs': 0.002 if stage == 'returned' else -1,
                        'local_l2': 3.0 if stage == 'returned' else -1}
                for stage, digest in (('input', input_hash), ('local', 'c'), ('returned', returned_hash))}

    def test_numeric_threshold_and_missing_shadow_fail_closed(self):
        self.assertEqual(self.checker.numeric_rows([self.row(distance=1e-2)], True)['status'], 'PASS')
        failure = self.checker.numeric_rows([self.row(distance=1.1e-2)], True)
        self.assertEqual(failure['status'], 'FAIL')
        self.assertEqual(failure['rows_above_threshold'][0]['ubatch_row'], 0)
        for distance in (-1, float('nan'), float('inf')):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                self.checker.numeric_rows([self.row(distance=distance)], True)
        row = self.row()
        del row['returned']['local_rel_l2']
        with self.assertRaises(ValueError):
            self.checker.numeric_rows([row], True)

    def test_different_returns_require_identical_complete_call_inputs(self):
        before = [self.row(), self.row(payload=1)]
        after = [self.row(returned_hash='d'), self.row(payload=1)]
        result = self.checker.compare_wire_rows(before, after)
        self.assertTrue(result['nondeterminism_observed'])
        self.assertEqual(result['differing_returns_with_identical_call_inputs'], 1)
        self.assertEqual(result['rows_with_identical_call_inputs'], 2)
        self.assertEqual(result['matched_call_return_agreement_fraction'], 0)
        after[1] = self.row(payload=1, input_hash='e')
        result = self.checker.compare_wire_rows(before, after)
        self.assertFalse(result['nondeterminism_observed'])
        self.assertEqual(result['rows_with_identical_row_inputs'], 1)
        self.assertEqual(result['rows_with_identical_call_inputs'], 0)
        self.assertIsNone(result['matched_call_return_agreement_fraction'])
        result = self.checker.compare_wire_rows(before, before)
        self.assertFalse(result['nondeterminism_observed'])
        self.assertEqual(result['rows_with_identical_call_inputs'], 2)
        self.assertEqual(result['matched_call_return_agreement_fraction'], 1)

    @staticmethod
    def logits(path, rows, slot=0):
        with path.open('wb') as stream:
            stream.write(b'S41LOG1\0')
            for step, values in enumerate(rows, 1):
                stream.write(struct.pack('<IIII', slot, 3, step, len(values)))
                stream.write(np.asarray(values, dtype='<f4').tobytes())

    def test_first_mismatch_decides_and_later_context_differences_are_recorded(self):
        entry = {'index': 0, 'slot_id': 0, 'prompt_tokens': [1, 2], 'tokens': [0, 0, 0]}
        phone = {**entry, 'tokens': [0, 1, 1]}
        with tempfile.TemporaryDirectory() as root:
            host_dir, phone_dir = Path(root)/'host', Path(root)/'phone'
            host_dir.mkdir()
            phone_dir.mkdir()
            self.logits(host_dir/'LOGITS.bin', [[10, 9], [10, 9.99], [10, 1]])
            self.logits(phone_dir/'LOGITS.bin', [[10, 9], [9.99, 10], [0, 10]])
            result = self.checker.compare_tokens([phone], [entry], phone_dir, host_dir)[0]
            self.assertTrue(result['accepted'])
            self.assertEqual(result['acceptance'], 'NEAR_TIE')
            self.assertEqual([row['step'] for row in result['mismatches']], [2, 3])
            self.assertEqual(result['mismatches'][1]['comparison'], 'after_context_divergence')
            self.assertGreater(result['mismatches'][1]['nmse'], 5e-4)
            self.assertFalse(result['mismatches'][1]['shared_token_prefix'])
            for values in ([[10, 9], [0, 10], [0, 10]], [[10, 9], [9.6, 10], [0, 10]]):
                self.logits(phone_dir/'LOGITS.bin', values)
                self.assertFalse(self.checker.compare_tokens([phone], [entry], phone_dir, host_dir)[0]['accepted'])
            self.logits(host_dir/'LOGITS.bin', [[10, 9], [10, 9.94], [10, 1]])
            self.logits(phone_dir/'LOGITS.bin', [[10, 9], [9.94, 10], [0, 10]])
            self.assertFalse(self.checker.compare_tokens([phone], [entry], phone_dir, host_dir)[0]['accepted'])

    def test_logit_capture_fails_closed_and_exact_tokens_remain_exact(self):
        entry = {'index': 0, 'slot_id': 0, 'prompt_tokens': [1], 'tokens': [0]}
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            result = self.checker.compare_tokens([entry], [entry], directory, directory)[0]
            self.assertEqual(result['acceptance'], 'EXACT')
            mismatch = {**entry, 'tokens': [1]}
            self.assertFalse(self.checker.compare_tokens([mismatch], [entry], directory, directory)[0]['accepted'])
            path = directory/'LOGITS.bin'
            self.logits(path, [[1, 2]], slot=1)
            with self.assertRaisesRegex(ValueError, 'slot/step'):
                self.checker.LogitsTrace(path, [entry])
            self.logits(path, [[1, 2]])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'truncated'):
                self.checker.LogitsTrace(path, [entry])
            self.logits(path, [[float('nan'), 2]])
            self.assertFalse(self.checker.compare_tokens([mismatch], [entry], directory, directory)[0]['accepted'])
