from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeGroupedObservation,
    AdaptiveDecodePolicy,
    AdaptiveDecodeWindowReceipt,
)
from research_dev.scheduler.campaigns.burstgpt.compare_ab import (
    ComparisonError,
    _desktop_parent_identity,
    canonical,
    load_adaptive_windows,
    phone_preflight_identity,
    result_counts,
    transport_summary,
    validate_matched,
)


def seal(value, key):
    value.pop(key, None)
    value[key] = 'sha256:' + hashlib.sha256(canonical(value)[:-1]).hexdigest()


def observation_fixture():
    _, result, _ = fixture()
    row = result['request_results'][0]
    result['request_results'] = [row]
    artifact = result['model_artifacts'][row['model_id']]['artifact_sha256']
    parent = row['execution_command']['operator_plan']['desktop_placement_sha256']
    policy = AdaptiveDecodePolicy(
        route_id='desktop', executor_id='desktop', operator_plan_sha256='sha256:' + 'd' * 64,
        desktop_parent_route_id='desktop', desktop_placement_sha256=parent,
        layer_indices=(), layer_mask=0, columns=0, split_fraction_ppm=0,
        resource_ids=('cpu', 'gpu'), baseline=True,
    )
    window = AdaptiveDecodeWindowReceipt(
        request_id=row['request_id'], slot_id=0, window_index=0, token_start=1, token_end=5,
        context_length=16, active_batch=1, started_at_us=10, finished_at_us=50,
        policy=policy, applied_ack=None, fleet_energy_uj_by_domain={'cpu': 40},
        latency_per_token_us=10, phone_compute_us=0, usb_transfer_us=0, rpc_us=0,
        exposed_tail_us=0, output_valid=True, evidence_ids=('test:physical',),
        energy_boundary_id='fleet', energy_attribution_kind='isolated', failure_reason=None,
        previous_record_sha256='0' * 64,
    )
    group = AdaptiveDecodeGroupedObservation(
        request_id=row['request_id'], ticket_id='ticket-current', model_artifact_sha256=artifact,
        planning_profile_sha256='sha256:' + 'e' * 64, desktop_placement_sha256=parent,
        windows=(window,), final_policy=policy, terminal_status='COMPLETED',
        state_history=('BASELINE', 'COMPLETED'),
    )
    row['terminal_ticket']['ticket_id'] = group.ticket_id
    row['fraction_history'] = {
        'adaptive_grouped_observation_sha256': group.grouped_observation_sha256,
        'adaptive_observation_ticket_id': group.ticket_id, 'adaptive_observation_scope': 'request',
    }
    row['physical_execution_proof'] = {
        'ticket_id': group.ticket_id, 'artifact_sha256': artifact,
        'adaptive_grouped_observation_sha256': group.grouped_observation_sha256,
        'adaptive_group_owner_request_id': group.request_id,
    }
    seal(row['physical_execution_proof'], 'proof_sha256')
    result['adaptive_observation_store_path'] = 'observations.json'
    return result, group


def fixture():
    roles = {role: role + '-model' for role in ('gemma', 'llama', 'qwen')}
    model_names = ['qwen'] * 15 + ['gemma'] * 6 + ['llama'] * 3
    expected = {
        'schema': 'research-scheduler-burstgpt-replay-v1',
        'trace_name': 'synthetic-reduced-24',
        'arrivals': [{'combined_request_index': i, 'replay_arrival_us': i * 1000}
                     for i in range(24)],
    }
    schedule = [{**row, 'request_id': 'r' + str(i), 'source_arrival_us': i * 2000}
                for i, row in enumerate(expected['arrivals'])]
    params = {
        'capacity_parent_source_placement_sha256': 'sha256:' + '1' * 64,
        'capacity_parent_qualification_sha256': 'sha256:' + '2' * 64,
        'capacity_parent_calibration_live_free_bytes': 10000,
        'capacity_parent_peak_vram_bytes': 8000,
        'capacity_parent_required_with_reserve_bytes': 9000,
        'capacity_parent_maximum_gpu_layers': 4,
        'gpu_layers': 4, 'gpu_device_id': 'desktop-cuda',
        'cpu_device_id': 'desktop-cpu', 'cuda_graph_mode': 'disabled',
        'context_size': 1024, 'parallel': 1, 'batch_size': 128, 'ubatch_size': 64,
    }
    baseline = {
        'status': 'PASS', 'selection_mode': 'desktop-baseline',
        'counts': {'qwen': 15, 'gemma': 6, 'llama': 3, 'requests': 24, 'terminals': 24},
        'catalog_sha256': 'sha256:' + '3' * 64,
        'model_roles': roles,
        'model_artifacts': {model: {'artifact_sha256': 'sha256:' + str(i + 4) * 64}
                            for i, model in enumerate(roles.values())},
        'trace_identity': {'large_sha256': '7' * 64, 'overlay_sha256': '8' * 64},
        'execution_identity': {
            'source_manifest_sha256': 'sha256:' + '9' * 64,
            'binaries': {'server': 'sha256:' + 'a' * 64},
            'cuda_graph_mode_by_artifact': {'sha256:' + '4' * 64: 'disabled'},
        },
        'maximum_latency_ppm': 1250000,
        'replay_schedule': {
            'schema': expected['schema'], 'trace_name': expected['trace_name'],
            'schedule': schedule,
            'schedule_sha256': 'sha256:' + hashlib.sha256(canonical(schedule)).hexdigest(),
        },
        'trace_energy': {
            'energy_boundary_id': 'connected-fleet', 'attribution_kind': 'diagnostic',
            'measurement_evidence_ids': ['physical:rapl', 'physical:nvml', 'ASSUMED_4P5W'],
            'fleet_energy_uj_by_domain': {'cpu-package': 100, 'gpu-board': 100, 'phone-system': 10},
        },
        'request_results': [],
    }
    for i, role in enumerate(model_names):
        baseline['request_results'].append({
            'combined_request_index': i, 'request_id': 'r' + str(i),
            'replay_arrival_us': i * 1000, 'model_id': roles[role],
            'input_tokens': 16, 'output_tokens': 64, 'seed': i,
            'prompt_sha256': 'sha256:' + 'b' * 64,
            'terminal_ticket': {
                'dispatch_state': 'COMPLETED', 'selection_mode': 'desktop-baseline',
                'execution_plan': {'device_ids': ['desktop-cuda'], 'execution_contract': {}},
            },
            'execution_command': {'adapter_parameters': dict(params), 'operator_plan': {
                'desktop_placement_sha256': 'sha256:' + 'c' * 64,
            }},
            'output_quality': {'accepted': True}, 'recoveries': [],
        })
    adaptive = deepcopy(baseline)
    adaptive['selection_mode'] = 'energy-aware'
    for row in adaptive['request_results']:
        row['terminal_ticket']['selection_mode'] = 'energy-aware'
    return baseline, adaptive, expected


class MatchedComparisonTests(unittest.TestCase):
    def test_only_current_execution_group_is_counted(self):
        result, group = observation_fixture()
        historical = replace(group, ticket_id='previous-run-ticket')
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'observations.json').write_bytes(canonical({
                'groups': [historical.to_json(), group.to_json()],
            }))
            windows = load_adaptive_windows(path / 'RESULT.json', result)
            self.assertEqual(windows, (group.windows[0].to_json(),))
            self.assertEqual(windows, load_adaptive_windows(path / 'RESULT.json', result))

    def test_current_observation_identity_fails_closed(self):
        for mutation in ('missing', 'duplicate', 'hash', 'ticket', 'parent', 'artifact', 'owner'):
            result, group = observation_fixture()
            groups = [group.to_json()]
            row = result['request_results'][0]
            proof = row['physical_execution_proof']
            if mutation == 'missing':
                groups.clear()
            elif mutation == 'duplicate':
                groups *= 2
            elif mutation == 'hash':
                groups[0]['windows'][0]['finished_at_us'] += 1
            elif mutation == 'ticket':
                row['terminal_ticket']['ticket_id'] = 'old-ticket'
            elif mutation == 'parent':
                row['execution_command']['operator_plan']['desktop_placement_sha256'] = 'sha256:' + 'f' * 64
            elif mutation == 'artifact':
                proof['artifact_sha256'] = 'sha256:' + 'f' * 64
            else:
                proof['adaptive_group_owner_request_id'] = 'old-request'
            seal(proof, 'proof_sha256')
            with self.subTest(mutation=mutation), TemporaryDirectory() as directory:
                path = Path(directory)
                (path / 'observations.json').write_bytes(canonical({'groups': groups}))
                with self.assertRaises(ComparisonError):
                    load_adaptive_windows(path / 'RESULT.json', result)

    def test_nested_terminal_counts_and_generation_identity(self):
        result, _ = observation_fixture()
        proof = result['request_results'][0]['physical_execution_proof']
        session = {
            'session_id': 'HTP1', 'session_generation': 3,
            'artifact_sha256': proof['artifact_sha256'],
            'resident_geometry_sha256': 'sha256:' + 'd' * 64,
            'operator_plan_sha256': 'sha256:' + 'e' * 64,
            'endpoint': 'session://phone/HTP1', 'calls': 7, 'payload_bytes': 224,
        }
        proof.update(phone_call_count=7, phone_payload_bytes=224, phone_calls_by_session=[session])
        seal(proof, 'proof_sha256')
        native = dict(session, h2d_bytes=224, endpoint_sha256='sha256:' + hashlib.sha256(
            session['endpoint'].encode('ascii')).hexdigest())
        del native['endpoint']
        terminal = {'status': 0, 'requests': 7, 'session_proofs': [native], 'queue_depth': 4}
        result['direct_phone_receipts'] = [{'terminal': terminal}]
        summary = transport_summary(result)
        self.assertEqual((summary['calls'], summary['payload_bytes']), (7, 224))
        self.assertEqual(summary['maximum_queue_depth'], 4)
        for key, bad in (('session_generation', 1), ('artifact_sha256', 'sha256:' + '0' * 64),
                         ('endpoint_sha256', 'sha256:' + '0' * 64), ('h2d_bytes', 225), ('calls', 6)):
            old = native[key]
            native[key] = bad
            with self.subTest(key=key), self.assertRaises(ComparisonError):
                transport_summary(result)
            native[key] = old
        result['direct_phone_receipts'].append({'terminal': terminal})
        with self.assertRaisesRegex(ComparisonError, 'generation-keyed'):
            transport_summary(result)

    def test_phone_preflight_identity_excludes_cleanup_but_includes_binaries(self):
        receipt = {
            'remote_hashes': {'resident_router': 'sha256:' + 'a' * 64, 'worker': 'sha256:' + 'b' * 64},
            'ffn_shard_indexes': {'parent': 'sha256:' + 'c' * 64},
            'phone_kernel_release': 'test-kernel', 'usb_close_sha256': 'sha256:' + 'd' * 64,
            'restoration': {'finished_at': 1},
        }
        with TemporaryDirectory() as directory:
            path = Path(directory)
            preflight = path / 'DIRECT_PHONE_PREFLIGHT.json'
            preflight.write_bytes(canonical(receipt))
            identity = phone_preflight_identity(path / 'RESULT.json')
            self.assertNotIn('restoration', identity)
            receipt['remote_hashes']['worker'] = 'sha256:' + 'e' * 64
            preflight.write_bytes(canonical(receipt))
            self.assertNotEqual(identity, phone_preflight_identity(path / 'RESULT.json'))
            receipt['usb_close_sha256'] = 'unknown'
            preflight.write_bytes(canonical(receipt))
            with self.assertRaises(ComparisonError):
                phone_preflight_identity(path / 'RESULT.json')

    def test_exact_declared_24_request_replay(self):
        baseline, adaptive, expected = fixture()
        validate_matched(baseline, adaptive, expected_replay=expected)
        with self.assertRaisesRegex(ComparisonError, 'model request counts differ'):
            result_counts(baseline)

    def test_partial_run_does_not_define_its_own_expected_count(self):
        baseline, adaptive, expected = fixture()
        for result in (baseline, adaptive):
            result['request_results'].pop()
            result['counts'].update(requests=23, terminals=23, llama=2)
        with self.assertRaisesRegex(ComparisonError, 'complete named replay'):
            validate_matched(baseline, adaptive, expected_replay=expected)

    def test_wrong_arrival_duplicate_request_and_changed_schedule_hash_reject(self):
        for mutation in ('arrival', 'duplicate', 'hash', 'count'):
            baseline, _, expected = fixture()
            if mutation == 'arrival':
                baseline['request_results'][0]['replay_arrival_us'] += 1
            elif mutation == 'duplicate':
                baseline['request_results'][1]['request_id'] = 'r0'
            elif mutation == 'hash':
                baseline['replay_schedule']['schedule_sha256'] = 'sha256:' + '0' * 64
            else:
                baseline['counts']['qwen'] -= 1
            with self.subTest(mutation=mutation), self.assertRaises(ComparisonError):
                result_counts(baseline, expected)

    def test_malformed_expected_schedule_rejects(self):
        baseline, _, expected = fixture()
        for row in ({'combined_request_index': 0, 'replay_arrival_us': -1},
                    {'combined_request_index': True, 'replay_arrival_us': 0},
                    expected['arrivals'][1]):
            malformed = deepcopy(expected)
            malformed['arrivals'][0] = row
            with self.subTest(row=row), self.assertRaises(ComparisonError):
                result_counts(baseline, malformed)

    def test_parent_and_qualification_differences_reject(self):
        baseline, adaptive, expected = fixture()
        params = adaptive['request_results'][0]['execution_command']['adapter_parameters']
        for key, value in list(params.items()):
            params[key] = value + 1 if type(value) is int else value + '-different'
            with self.subTest(key=key), self.assertRaises(ComparisonError):
                validate_matched(baseline, adaptive, expected_replay=expected)
            params[key] = value
        del params['capacity_parent_qualification_sha256']
        with self.assertRaisesRegex(ComparisonError, 'qualification is incomplete'):
            _desktop_parent_identity(adaptive['request_results'][0])

    def test_source_runtime_mode_and_energy_boundary_differences_reject(self):
        for field in ('source', 'graph', 'boundary', 'attribution', 'evidence'):
            baseline, adaptive, expected = fixture()
            if field == 'source':
                adaptive['execution_identity']['source_manifest_sha256'] = 'changed'
            elif field == 'graph':
                adaptive['execution_identity']['cuda_graph_mode_by_artifact']['sha256:' + '4' * 64] = 'default'
            else:
                key = {'boundary': 'energy_boundary_id', 'attribution': 'attribution_kind',
                       'evidence': 'measurement_evidence_ids'}[field]
                adaptive['trace_energy'][key] = 'changed'
            with self.subTest(field=field), self.assertRaises(ComparisonError):
                validate_matched(baseline, adaptive, expected_replay=expected)


if __name__ == '__main__':
    unittest.main()
