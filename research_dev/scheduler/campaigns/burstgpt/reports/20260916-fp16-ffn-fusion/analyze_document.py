"""Compare the same bounded relocated request with fusion disabled and enabled."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('physical', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
roots = {mode: args.physical / ('document-' + mode) for mode in ('disabled', 'enabled')}
results = {mode: read(root / 'gate-run-v1/REMOTE_RESIDENT_GATE.json') for mode, root in roots.items()}
timing = {mode: read(root / 'TIMING.json') for mode, root in roots.items()}
off, on = results['disabled'], results['enabled']
assert off['status'] == on['status'] == 'PASS'
matching = ('artifact_sha256', 'runtime_binary_sha256', 'runtime_libraries_sha256',
            'remote_layer_mask', 'resident_layer_mask', 'shards', 'requests', 'launch_contracts')
for field in matching:
    assert off[field] == on[field], field
assert off['selection']['selected_placement_sha256'] == on['selection']['selected_placement_sha256']
assert off['arms']['reduced']['ready_identity']['phone_shards'] == on['arms']['reduced']['ready_identity']['phone_shards']
assert read(roots['disabled'] / 'gate-run-v1/DOCUMENT_REQUESTS.json') == read(
    roots['enabled'] / 'gate-run-v1/DOCUMENT_REQUESTS.json')
experiments = {mode: read(root / 'EXPERIMENT.json') for mode, root in roots.items()}
assert experiments['disabled']['source_manifest_sha256'] == experiments['enabled']['source_manifest_sha256']
before = off['arms']['reduced']['requests'][0]
after = on['arms']['reduced']['requests'][0]
assert before['context_completion']['terminal']['generation_settings'] == after['context_completion']['terminal']['generation_settings']
arms = {}
for mode, result in results.items():
    arm = result['arms']['reduced']
    request, = arm['requests']
    terminal = read(roots[mode] / 'gate-run-v1/phone/TERMINAL.json')
    terminal_receipt, = [row['terminal'] for row in terminal['phone_receipts'] if 'terminal' in row]
    assert terminal_receipt['status'] == terminal_receipt['reset_recoveries'] == 0
    assert all(row['status'] == 'RESTORED' for row in terminal['usb_restoration'])
    native = timing[mode]['native_timings']['reduced']
    assert native['prompt_n'] == 5261 and native['predicted_n'] == 64
    assert request['proof']['phone_call_count'] == 1800
    calls = request['proof']['phone_calls_by_session']
    ready = {row['session_id']: row for row in arm['ready_identity']['phone_shards']}
    for row in calls:
        for key in ('artifact_sha256', 'operator_plan_sha256', 'resident_geometry_sha256',
                    'session_generation', 'layer_mask'):
            assert row[key] == ready[row['session_id']][key], key
    arms[mode] = {
        'prefill_s': native['prompt_ms'] / 1000, 'decode_s': native['predicted_ms'] / 1000,
        'prefill_plus_decode_s': (native['prompt_ms'] + native['predicted_ms']) / 1000,
        'desktop_control_timings': timing[mode]['native_timings']['full'],
        'functionfs_call_shapes': timing[mode]['shapes'],
        'request_plus_desktop_launch_s': request['duration_us'] / 1e6,
        'request_plus_desktop_launch_host_energy_j': request['energy']['server_compute_device_energy_j'],
        'phone_preparation_s': arm['load_duration_us'] / 1e6,
        'phone_preparation_host_energy_j': arm['load_energy']['server_compute_device_energy_j'],
        'shard_load_counts': arm['ready_identity']['load_count_by_session'],
        'phone_calls': request['proof']['phone_call_count'], 'session_calls': calls,
        'terminal_status': terminal_receipt['status'], 'usb_reset_recoveries': terminal_receipt['reset_recoveries'],
        'output_text': request['context_completion']['output_text'],
        'output_quality': request['context_completion']['output_quality'],
        'result_sha256': sha(roots[mode] / 'gate-run-v1/REMOTE_RESIDENT_GATE.json'),
        'terminal_sha256': sha(roots[mode] / 'gate-run-v1/phone/TERMINAL.json'),
    }
changes = {name: 100 * (1 - arms['enabled'][name] / arms['disabled'][name])
           for name in ('prefill_s', 'decode_s', 'prefill_plus_decode_s',
                        'request_plus_desktop_launch_host_energy_j')}
summary = {
    'schema': 's42-fp16-ffn-document-ab-v1', 'status': 'PASS',
    'compatibility_checks': list(matching) + ['desktop_placement', 'phone_shards', 'document_requests',
                                              'generation_settings', 'source_manifest'],
    'allowed_differences': ['Explicit fusion switch wrapper and its transport identity',
                            'Artifact directory, timestamps and live observation values'],
    'arms': arms, 'reduction_percent': changes,
    'fusion_mode_output_token_match': before['tokens'] == after['tokens'],
    'fusion_mode_agreeing_tokens': sum(a == b for a, b in zip(before['tokens'], after['tokens'], strict=True)),
    'notes': ['One bounded request per fusion mode, not a statistical savings estimate.',
              'Desktop/relocated token divergence is recorded by the unchanged semantic-sanity gate.',
              'Relocated request energy includes desktop launch; phone preload is separate.',
              'Reported energy is measured host CPU package plus GPU board, not measured phone power.',
              'No pooled intervals or overlap subtraction; no total fleet-energy claim.'],
}
with args.output.open('x') as stream:
    json.dump(summary, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(changes, indent=2))
print('mode token match', summary['fusion_mode_output_token_match'])
