"""Scheduler trace accounting, launch isolation and native numerical identity."""
from dataclasses import replace
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_dev.scheduler._internal.matmul import MatmulScheduleError
from research_dev.scheduler._internal.profile_materializer import read_scheduler_inventory
from research_dev.scheduler.adapters import (
    LlamaServerProcessConfiguration, LlamaServerProcessLauncher, PhysicalAdapterError,
    llama_server_launch_contract,
)
from research_dev.scheduler.adapters.llama_server_contracts import _launch_contract_supports_execution
from research_dev.scheduler import GGUFModelManifestLoader
from test_gguf_cost import write_synthetic_gguf
from test_llama_server_adapter import execution_command
from test_kv_decode_relocation_gate import gate
from test_remote_resident_native import PROBE
from tiny_llama_gguf import write_tiny_llama_gguf


class SchedulerInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'trace.jsonl'
        base = {'schema': 'ggml-sched-trace-v1', 'scheduler': 's', 'graph': 1}
        weight = {'name': 'blk.2.ffn_up.weight', 'op': 'NONE', 'dtype': 'f16', 'shape': [8, 16, 1, 1]}
        node = {'name': 'ffn_up-2', 'op': 'MUL_MAT', 'dtype': 'f32', 'shape': [16, 128, 1, 1]}
        self.rows = [
            {**base, 'event': 'copy', 'split': 0, 'input': 0, 'source': 'CPU', 'destination': 'CUDA0',
             'weights': True, 'bytes': 256, 'ranges': [[0, 256]], 'host_us': 30, 'wait_us': 10, 'tensor': weight},
            {**base, 'event': 'split', 'split': 0, 'backend': 'CUDA0', 'node_count': 1, 'nodes': [node],
             'copied_bytes': 256, 'copy_host_us': 30, 'compute_host_us': 5, 'wall_us': 40},
            {**base, 'event': 'graph', 'tokens': 128, 'started_us': 100, 'finished_us': 145,
             'host_wall_us': 45, 'split_count': 1, 'completion': 'async_submission'},
            {**base, 'event': 'sync', 'first_graph': 1, 'started_us': 146, 'finished_us': 166, 'host_wait_us': 20},
        ]

    def read(self, rows=None, **bounds):
        self.path.write_text(''.join(json.dumps(r) + '\n' for r in (self.rows if rows is None else rows)))
        return read_scheduler_inventory(self.path, start_us=bounds.get('start', 90), end_us=bounds.get('end', 180))

    def test_counts_copy_and_completion_once_without_calling_enqueue_device_time(self):
        result = self.read()
        self.assertEqual(result['accounted_host_us'], 55)
        self.assertEqual(result['weight_copy_host_excluding_wait_us'], 20)
        self.assertEqual(result['weight_copy_bytes'], 256)
        self.assertEqual(result['components'][0]['layers'], [2])
        self.assertEqual(result['components'][0]['families'], ['ffn'])
        self.assertEqual(result['graphs'][0]['phase'], 'prefill')

    def test_incomplete_duplicate_or_inconsistent_evidence_fails_closed(self):
        cases = [self.rows[:-1], self.rows[1:], self.rows + [self.rows[0]],
                 [self.rows[0], self.rows[1], self.rows[3]]]
        for index, field, value in ((0, 'ranges', [[0, 512]]), (0, 'wait_us', 31),
                                    (1, 'node_count', 2), (1, 'wall_us', 34),
                                    (2, 'completion', 'device_complete'), (2, 'tokens', 0),
                                    (3, 'first_graph', 0)):
            rows = copy.deepcopy(self.rows)
            rows[index][field] = value
            cases.append(rows)
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(MatmulScheduleError):
                self.read(rows)

    def test_window_requires_complete_graph_and_completion(self):
        for bounds in ({'start': 120}, {'end': 150}, {'start': 200, 'end': 300}):
            with self.subTest(bounds=bounds), self.assertRaises(MatmulScheduleError):
                self.read(**bounds)

    def test_selected_expert_ranges_count_actual_bytes(self):
        self.rows[0]['ranges'] = [[0, 128], [512, 128]]
        self.assertEqual(self.read()['copy_bytes'], 256)

    def test_suppressed_prefill_outputs_can_have_zero_rows(self):
        self.rows[1]['nodes'].append({'name': 'result_output', 'op': 'MUL_MAT',
                                     'dtype': 'f32', 'shape': [151936, 0, 1, 1]})
        self.rows[1]['node_count'] = 2
        self.assertEqual(self.read()['accounted_host_us'], 55)
        self.rows[1]['nodes'][-1]['shape'][1] = -1
        with self.assertRaises(MatmulScheduleError):
            self.read()


class SchedulerTraceLaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        model = self.root / 'model.gguf'
        write_synthetic_gguf(model)
        self.manifest = GGUFModelManifestLoader.load('synthetic-model-id', model)
        command = execution_command(self.manifest.artifact_sha256)
        self.contract = replace(llama_server_launch_contract(command, self.manifest),
                                gpu_layers=0, phone_device_id=None, ffn_environment={})
        server = self.root / 'server'
        server.write_text('#!/bin/sh\n')
        server.chmod(0o755)
        self.launcher = LlamaServerProcessLauncher(LlamaServerProcessConfiguration(
            server_path=server, model_paths_by_artifact={self.manifest.artifact_sha256: model},
            library_paths_by_device={}, executable_device_names={}, output_directory=self.root))

    def test_default_validation_and_process_reuse(self):
        self.assertIsNone(self.contract.scheduler_trace_path)
        tracing = replace(self.contract, scheduler_trace_path='/tmp/trace.jsonl')
        self.assertFalse(_launch_contract_supports_execution(self.contract, tracing))
        for value in ('', 'relative.jsonl', '/tmp/trace\n', '/tmp/trace\0', True, 1):
            with self.subTest(value=value), self.assertRaises(PhysicalAdapterError):
                replace(self.contract, scheduler_trace_path=value)

    def test_trace_setting_enters_the_measurement_identity(self):
        config = {'gpu_layers': 16, 'batch': 2048, 'ubatch': 1024, 'column_quantum': 2176,
                  'phone': {'session_masks': {'HTP0': 1}}}
        plan = SimpleNamespace(context_size=32768, artifact_sha256='sha256:' + 'a'*64, to_json=lambda: {})
        runtime = {'server': 'sha256:' + 'b'*64}
        plain = gate.selection_environment(config, plan, runtime)
        traced = gate.selection_environment({**config, 'scheduler_trace_path': '/tmp/trace.jsonl'}, plan, runtime)
        self.assertNotEqual(plain.runtime_sha256, traced.runtime_sha256)
        self.assertEqual(plain, gate.selection_environment({**config, 'scheduler_trace_path': None}, plan, runtime))

    def test_inherited_environment_is_cleared_and_old_binaries_are_rejected(self):
        class FakeServer:
            confirm = False

            def __init__(self, command, environment, directory, label, contract):
                self.environment = environment
                self.process = SimpleNamespace(poll=lambda: None)
                self.stderr_lines = []
                if self.confirm and contract.scheduler_trace_path is not None:
                    self.stderr_lines = ['GGML_SCHED_TRACE schema=ggml-sched-trace-v1 path=' + contract.scheduler_trace_path]

            def start(self):
                pass

            def stop(self):
                pass

        trace_path = '/tmp/trace.jsonl'
        contract = replace(self.contract, scheduler_trace_path=trace_path)
        with patch.dict(os.environ, {'GGML_SCHED_TRACE': '/tmp/foreign.jsonl'}), patch(
            'research_dev.scheduler.adapters.llama_server.ManagedLlamaServer', FakeServer
        ), patch.object(self.launcher, '_healthy', return_value=True):
            with self.assertRaisesRegex(PhysicalAdapterError, 'tracing support'):
                self.launcher.launch_contract('http://127.0.0.1:19000', contract, self.manifest,
                                              label='trace', control_check=lambda: None)
            FakeServer.confirm = True
            process = self.launcher.launch_contract('http://127.0.0.1:19000', contract, self.manifest,
                                                   label='trace', control_check=lambda: None)
            self.assertEqual(process.environment['GGML_SCHED_TRACE'], trace_path)
            FakeServer.confirm = False
            process = self.launcher.launch_contract('http://127.0.0.1:19000', self.contract, self.manifest,
                                                   label='plain', control_check=lambda: None)
            self.assertNotIn('GGML_SCHED_TRACE', process.environment)


@unittest.skipUnless(PROBE.exists(), 'native probe is not built')
class SchedulerTraceNativeTests(unittest.TestCase):
    def test_tiny_prefill_decode_inventory_and_exact_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = write_tiny_llama_gguf(root / 'tiny.gguf', n_layer=4, head_dim=64)
            gpu_layers = os.environ.get('S42_SPLIT_KV_GPU_LAYERS', '0')
            for arm in ('off', 'on'):
                environment = {k: v for k, v in os.environ.items() if k != 'GGML_SCHED_TRACE'}
                if arm == 'on':
                    environment['GGML_SCHED_TRACE'] = str(root / 'trace.jsonl')
                result = subprocess.run([str(PROBE), '--model', str(model), '--out', str(root/f'{arm}.json'),
                    '--tokens', ','.join(map(str, range(1, 65))), '--batch-size', '64', '--ubatch-size', '64',
                    '--max-tokens', '64', '--gpu-layers', gpu_layers, '--threads', '2',
                    '--decode', '3', '--logits', str(root/f'{arm}.bin')],
                    env=environment, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            self.assertEqual((root/'off.bin').read_bytes(), (root/'on.bin').read_bytes())
            inventory = read_scheduler_inventory(root/'trace.jsonl', start_us=0, end_us=10**18)
            self.assertEqual([g['tokens'] for g in inventory['graphs']], [64, 1, 1, 1])
            self.assertGreater(inventory['accounted_host_us'], 0)
            if int(gpu_layers) > 0:
                self.assertGreater(inventory['weight_copy_bytes'], 0)
            for phase, count in (('prefill', 1), ('decode', 3)):
                filtered = read_scheduler_inventory(root/'trace.jsonl', start_us=0, end_us=10**18, phase=phase)
                self.assertEqual(filtered['graph_count'], count)
            environment['GGML_SCHED_TRACE'] = str(root/'missing/trace.jsonl')
            result = subprocess.run([str(PROBE), '--model', str(model), '--out', str(root/'bad.json')],
                                    env=environment, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
