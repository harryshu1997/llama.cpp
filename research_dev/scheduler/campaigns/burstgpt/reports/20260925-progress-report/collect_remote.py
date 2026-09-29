#!/usr/bin/env python3
"""Read completed results and live artifacts without changing the shared rig."""

import json
from pathlib import Path
import subprocess


REMOTE = r'''
import collections
import datetime
import hashlib
import json
from pathlib import Path
import re

root = Path('/mnt/storage/s43-two-phone-eval-20260925')
specs = {
    'dev_desktop': 'inputs-desktop-dev2a',
    'dev_op15_r1': 'inputs-op15-dev2a',
    'dev_op15_r2': 'inputs-op15-dev2b',
    'dev_two_r1': 'inputs-two-phone-dev2a',
    'dev_two_r2': 'inputs-two-phone-dev2b',
    'lt_legacy': 'inputs-desktop-legacy-lt1',
    'lt_dispatcher': 'inputs-desktop-lt1',
    'eval_legacy': 'inputs-desktop-legacy-ev2',
    'eval_two': 'inputs-two-phone-ev2',
    'eval_dispatcher': 'inputs-desktop-ev3',
    'eval_op15': 'inputs-op15-ev3',
}
fields = ('model_id', 'prompt_sha256', 'input_tokens', 'output_tokens',
          'seed', 'source_arrival_us', 'source_slo_us')


def stream(path):
    raw = path.read_bytes()
    tokens = []
    stopped = False
    for line in raw.decode().splitlines():
        if not line.startswith('data:') or line[5:].strip() == '[DONE]':
            continue
        obj = json.loads(line[5:])
        tokens.extend(obj.get('tokens', []))
        stopped |= bool(obj.get('stop'))
    return {'tokens': tokens, 'stop': stopped,
            'sha256': hashlib.sha256(raw).hexdigest(), 'path': str(path)}


def completed(run):
    path = run / 'RESULT.json'
    raw = path.read_bytes()
    result = json.loads(raw)
    domains = result['trace_energy']['fleet_energy_uj_by_domain']
    manifest_path = run.parent / 'SOURCE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    files = {entry['path']: entry['sha256'] for entry in manifest['files']}
    file_digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    requests = {}
    assisted = collections.Counter()
    for row in result['request_results']:
        s = stream(run / 'streams' / f"request-{row['combined_request_index']:03d}.raw")
        assert len(s['tokens']) == row['output_tokens'], s['path']
        requests[row['request_id']] = {key: row[key] for key in fields}
        requests[row['request_id']].update(s)
        if row.get('physical_execution_proof', {}).get('phone_call_count', 0):
            assisted[row['model_id']] += 1
    calls = collections.Counter()
    for proof in result.get('physical_execution_proofs', {}).values():
        for entry in proof.get('phone_calls_by_session', []):
            device = 'pixel' if entry['session_id'].startswith('PIXEL') else 'op15'
            calls[device] += entry.get('calls', 0)
    lifecycle_path = run / 'CO_HELPER_LIFECYCLE.json'
    lifecycle = json.loads(lifecycle_path.read_text()) if lifecycle_path.exists() else []
    return {
        'status': result['status'], 'path': str(path),
        'result_sha256': hashlib.sha256(raw).hexdigest(),
        'duration_s': result['duration_us'] / 1e6,
        'cpu_kj': domains['cpu-package'] / 1e9,
        'gpu_kj': domains['gpu-board'] / 1e9,
        'host_kj': (domains['cpu-package'] + domains['gpu-board']) / 1e9,
        'phone_kj_assumed': {k: v / 1e9 for k, v in domains.items()
                             if k not in ('cpu-package', 'gpu-board')},
        'counts': result['counts'], 'requests': requests,
        'assisted_requests_by_model': dict(assisted),
        'phone_calls': dict(calls), 'execution_identity': result['execution_identity'],
        'source_file_map_sha256': file_digest, 'source_file_count': len(files),
        'source_manifest_path': str(manifest_path),
        'co_helper_lifecycle': lifecycle,
        'cleanup_failure_present': (run / 'CLEANUP_FAILURE.json').exists(),
        'failure_present': (run / 'FAILURE.json').exists(),
    }


report = {'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'rig': 'zhihao@172.20.74.85', 'arms': {}}
for label, inputs in specs.items():
    run = root / inputs / 'run-eval/run'
    if (run / 'RESULT.json').exists():
        report['arms'][label] = completed(run)
    elif (run / 'FAILURE.json').exists():
        report['arms'][label] = {'status': 'FAIL', 'path': str(run),
                                'failure': json.loads((run / 'FAILURE.json').read_text())}
    else:
        report['arms'][label] = {'status': 'RUNNING' if run.exists() else 'NOT_STARTED',
                                'path': str(run)}
        if run.exists():
            live = {}
            for path in sorted((run / 'streams').glob('request-*.raw')):
                if re.fullmatch(r'request-\d+\.raw', path.name):
                    s = stream(path)
                    live[path.name] = {'tokens': len(s['tokens']), 'stop': s['stop']}
            report['arms'][label]['live_streams'] = live
            shapes = []
            for path in sorted(run.glob('large-model*.stderr')):
                with path.open() as handle:
                    for line in handle:
                        if 'S41SERVERFFNSHAPE {' in line:
                            payload = line.split('S41SERVERFFNSHAPE ', 1)[1].strip()
                            try:
                                shape = json.loads(payload)
                            except ValueError:
                                continue
                            shape['source'] = str(path)
                            shapes.append(shape)
            report['arms'][label]['shape_summaries'] = shapes
            lifecycle = run / 'CO_HELPER_LIFECYCLE.json'
            if lifecycle.exists():
                report['arms'][label]['co_helper_lifecycle'] = json.loads(lifecycle.read_text())

report['comparisons'] = {}
for base, treatments in [
    ('dev_desktop', ['dev_op15_r1', 'dev_op15_r2', 'dev_two_r1', 'dev_two_r2']),
    ('lt_legacy', ['lt_dispatcher']),
    ('eval_legacy', ['eval_two', 'eval_dispatcher', 'eval_op15']),
]:
    for treatment in treatments:
        b, t = report['arms'][base], report['arms'][treatment]
        if b['status'] != 'PASS' or t['status'] != 'PASS':
            continue
        same_ids = b['requests'].keys() == t['requests'].keys()
        diffs = []
        token_diffs = []
        identical = 0
        for key in b['requests'].keys() & t['requests'].keys():
            br, tr = b['requests'][key], t['requests'][key]
            for field in fields:
                if br[field] != tr[field]:
                    diffs.append({'request': key, 'field': field})
            identical += br['tokens'] == tr['tokens']
            if br['tokens'] != tr['tokens']:
                first = next((i for i, (x, y) in enumerate(zip(br['tokens'], tr['tokens'])) if x != y),
                             min(len(br['tokens']), len(tr['tokens'])))
                token_diffs.append({'request': key, 'model_id': tr['model_id'],
                                    'first_differing_token_zero_based': first})
        report['comparisons'][treatment] = {
            'baseline': base, 'same_request_ids': same_ids, 'input_differences': diffs,
            'identical_outputs': identical, 'requests': len(b['requests']),
            'output_differences': sorted(token_diffs, key=lambda row: row['request']),
            'strict_tokens': 'PASS' if same_ids and not diffs and identical == len(b['requests']) else 'FAIL',
            'host_saving_pct': 100 * (1 - t['host_kj'] / b['host_kj']),
            'duration_reduction_pct': 100 * (1 - t['duration_s'] / b['duration_s']),
            'same_execution_identity': b['execution_identity'] == t['execution_identity'],
            'same_source_files': b['source_file_map_sha256'] == t['source_file_map_sha256'],
            'same_native_binaries': b['execution_identity']['binaries'] == t['execution_identity']['binaries'],
        }
report['chain'] = [json.loads(line) for line in
                   (root / 'chains/CHAIN-ev2.jsonl').read_text().splitlines() if line.strip()]
report['prep_chain'] = [json.loads(line) for line in
                        (root / 'chains/CHAIN-ev2-prep.jsonl').read_text().splitlines() if line.strip()]
followup = root / 'chains/CHAIN-ev3.jsonl'
report['followup_chain'] = ([json.loads(line) for line in followup.read_text().splitlines()
                            if line.strip()] if followup.exists() else [])
report['capture_finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(report))
'''


def main():
    response = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
         'zhihao@172.20.74.85', 'python3 -'],
        input=REMOTE, text=True, capture_output=True, check=True,
    )
    data = json.loads(response.stdout)
    output = Path(__file__).resolve().parent / 'sources'
    output.mkdir(exist_ok=True)
    stamp = data['captured_at_utc'][:19].replace(':', '').replace('-', '')
    path = output / f'rig_audit_{stamp}.json'
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    print(path)
    for label, arm in data['arms'].items():
        comparison = data['comparisons'].get(label, {})
        print(label, arm['status'], arm.get('host_kj'),
              comparison.get('host_saving_pct'), comparison.get('identical_outputs'),
              comparison.get('same_execution_identity'))


if __name__ == '__main__':
    main()
