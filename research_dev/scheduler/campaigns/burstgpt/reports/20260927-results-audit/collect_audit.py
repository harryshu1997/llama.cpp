#!/usr/bin/env python3
"""Read saved campaign evidence and compare current scheduler sources."""

import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
REMOTE = r'''
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root = Path('/mnt/storage/s43-two-phone-eval-20260925')
specs = {
    'legacy_r1': 'inputs-desktop-legacy-ev2',
    'dispatcher_r1': 'inputs-desktop-ev3',
    'op15_r1': 'inputs-op15-ev3',
    'two_r1': 'inputs-two-phone-ev2',
    'legacy_r2': 'inputs-desktop-legacy-ev8',
    'dispatcher_r2': 'inputs-desktop-ev8',
    'op15_r2': 'inputs-op15-ev8',
    'two_r2': 'inputs-two-phone-ev8',
    'two_tp2': 'inputs-two-phone-tp2',
    'g1g': 'inputs-two-phone-g7-elastic',
    'g9': 'inputs-two-phone-g9-elastic',
    'g11': 'inputs-two-phone-g11-elastic',
}
fields = ('model_id', 'prompt_sha256', 'input_tokens', 'output_tokens',
          'seed', 'source_arrival_us', 'source_slo_us')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stream(path):
    tokens, stop = [], False
    for line in path.read_text().splitlines():
        if line.startswith('data:') and line[5:].strip() != '[DONE]':
            row = json.loads(line[5:])
            tokens.extend(row.get('tokens', []))
            stop |= bool(row.get('stop'))
    return {'tokens': tokens, 'stop': stop, 'raw_sha256': digest(path)}

def sources(directory):
    return {str(p.relative_to(directory)): digest(p)
            for p in sorted(directory.rglob('*.py'))
            if not {'reports', '__pycache__', '.venv'}.intersection(p.parts)}

arms = {}
manifests = {}
for name, inputs in specs.items():
    run = root / inputs / 'run-eval/run'
    path = run / 'RESULT.json'
    result = json.loads(path.read_text())
    domains = result['trace_energy']['fleet_energy_uj_by_domain']
    requests = {}
    calls = Counter()
    calls_by_model_device = Counter()
    for row in result['request_results']:
        request = {k: row[k] for k in fields}
        request.update(stream(run / 'streams' / f"request-{row['combined_request_index']:03d}.raw"))
        assert len(request['tokens']) == row['output_tokens']
        requests[row['request_id']] = request
    for proof in result.get('physical_execution_proofs', {}).values():
        request_id = proof['ticket_id'].split(':attempt:')[0]
        model = requests[request_id]['model_id']
        for session in proof.get('phone_calls_by_session', []):
            calls[session['session_id']] += session.get('calls', 0)
            device = 'pixel' if session['session_id'].startswith('PIXEL') else 'op15'
            calls_by_model_device[f'{model}/{device}'] += session.get('calls', 0)
    manifest_path = run.parent / 'SOURCE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    manifests[name] = {x['path']: x['sha256'] for x in manifest['files']}
    config = json.loads((run.parent / 'RESOLVED_CONFIGURATION.json').read_text())
    arms[name] = {
        'path': str(path), 'result_sha256': digest(path),
        'source_manifest_path': str(manifest_path),
        'source_manifest_sha256': digest(manifest_path),
        'status': result['status'], 'counts': result['counts'],
        'duration_s': result['duration_us'] / 1e6,
        'cpu_kj': domains['cpu-package'] / 1e9,
        'gpu_kj': domains['gpu-board'] / 1e9,
        'host_kj': (domains['cpu-package'] + domains['gpu-board']) / 1e9,
        'phone_kj_assumed': {k: v / 1e9 for k, v in domains.items()
                             if k not in ('cpu-package', 'gpu-board')},
        'requests': requests, 'phone_calls_by_session': dict(calls),
        'phone_calls_by_model_device': dict(calls_by_model_device),
        'execution_identity': result['execution_identity'],
        'thermal_policy_fields': {k: v for k, v in config.items() if 'thermal' in k},
        'paid_start_ns': result['paid_start_ns'],
        'failure_present': (run / 'FAILURE.json').exists(),
        'cleanup_failure_present': (run / 'CLEANUP_FAILURE.json').exists(),
        **{k: result.get(k, []) for k in (
            'helper_mask_events', 'helper_membership_events',
            'device_membership_events', 'thermal_deferral_events',
            'request_recovered_events')},
    }
    fault = run / 'FAULT_INJECTED.json'
    if fault.exists():
        arms[name]['fault'] = json.loads(fault.read_text())

baseline = arms['legacy_r1']
for name, arm in arms.items():
    same_inputs = baseline['requests'].keys() == arm['requests'].keys()
    mismatches = []
    for key, request in arm['requests'].items():
        base = baseline['requests'][key]
        same_inputs &= all(request[k] == base[k] for k in fields)
        if request['tokens'] != base['tokens']:
            pairs = zip(request['tokens'], base['tokens'])
            first = next((i for i, (a, b) in enumerate(pairs) if a != b),
                         min(len(request['tokens']), len(base['tokens'])))
            mismatches.append({'request_id': key, 'first_divergent_token': first})
    arm['comparison_vs_legacy_r1'] = {
        'same_inputs': same_inputs,
        'all_streams_stopped': all(x['stop'] for x in arm['requests'].values()),
        'output_tokens': sum(len(x['tokens']) for x in arm['requests'].values()),
        'identical_outputs': len(arm['requests']) - len(mismatches),
        'mismatches': mismatches,
        'host_saving_pct': (1 - arm['host_kj'] / baseline['host_kj']) * 100,
        'source_files_match': manifests[name] == manifests['legacy_r1'],
        'binary_hashes_match': (arm['execution_identity']['binaries'] ==
                                baseline['execution_identity']['binaries']),
    }
for arm in arms.values():
    for request in arm['requests'].values():
        tokens = request.pop('tokens')
        request['tokens_sha256'] = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()

for name in ('two_r1', 'two_r2', 'two_tp2'):
    run = root / specs[name] / 'run-eval/run'
    observations = []
    files = {}
    for path in (run / 'snapshots').glob('runtime-*.json'):
        if path.name.count('.') != 1:
            continue
        snapshot = json.loads(path.read_text())
        for state in snapshot.get('executors', []):
            if state['executor_id'] == 'physical:op15-phone':
                observations.append({
                    'at_us': snapshot['captured_at_us'],
                    'qualified': state.get('thermal_qualified'),
                    'status': state.get('thermal_status'),
                    'temperature_millic': state['temperature_millic'],
                    'source': path.name,
                })
                files[path.name] = digest(path)
    observations.sort(key=lambda x: x['at_us'])
    changes = []
    for row in observations:
        if not changes or row['qualified'] != changes[-1]['qualified']:
            changes.append(row)
    excluded_us, start = 0, None
    for row in changes:
        if row['qualified'] is False:
            start = row['at_us']
        elif start is not None:
            excluded_us += row['at_us'] - start
            start = None
    arms[name]['op15_snapshot_thermal'] = {
        'sample_count': len(observations), 'changes': changes,
        'first_observation_us': observations[0]['at_us'],
        'last_observation_us': observations[-1]['at_us'],
        'closed_exclusion_intervals_s': excluded_us / 1e6,
        'unclosed_interval_start_us': start,
        'max_temperature_millic': max(x['temperature_millic'] for x in observations),
        'raw_status_sample_count': sum(x['status'] is not None for x in observations),
        'source_files_map_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    }
    log = run / 'SCHEDULER_DECISION_LOG.json'
    raw = log.read_text()
    arms[name]['decision_log_markers'] = {
        'path': str(log), 'sha256': digest(log),
        'counts_are_string_occurrences_not_unique_events': True,
        'counts': {s: raw.count(s) for s in (
            'THERMAL_LIMIT', 'PHONE_HELPER_UNAVAILABLE',
            'HELPER_REMATERIALIZATION_FAILED', 'no helper opportunity',
            'replacement source is not ready')},
    }

path = root / specs['g11'] / 'run-eval/run/SCHEDULER_DECISION_LOG.json'
log = json.loads(path.read_text())
arms['g11']['recovery_dispatch_evidence'] = {
    'path': str(path), 'sha256': digest(path),
    'fallbacks': [
        {'ticket_id': r['ticket_id'], 'event_time_us': r['event_time_us'],
         'dispatch_policy': r.get('selected', {}).get('dispatch_policy')}
        for r in log['records'] if r.get('event_kind') == 'FALLBACK'
    ],
    'server_exited_string_occurrences': path.read_text().count('SERVER_EXITED'),
}

report = {
    'captured_at_utc': datetime.now(timezone.utc).isoformat(),
    'rig': 'zhihao@172.20.74.85', 'arms': arms,
    'current_scheduler_python_sources': {
        'stage': sources(root / 'stage/research_dev/scheduler'),
        'deploy': sources(Path('/mnt/storage/s42-trace-v2-20260921-prep/source/research_dev/scheduler')),
    },
}
print(json.dumps(report, sort_keys=True))
'''


def main():
    completed = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
         'zhihao@172.20.74.85', 'python3 -'],
        input=REMOTE, text=True, capture_output=True, check=True,
    )
    report = json.loads(completed.stdout)
    scheduler = REPO / 'research_dev/scheduler'
    local = {
        str(p.relative_to(scheduler)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(scheduler.rglob('*.py'))
        if not {'reports', '__pycache__', '.venv'}.intersection(p.parts)
    }
    assert local, scheduler
    maps = report.pop('current_scheduler_python_sources')
    report['current_source_comparison'] = {}
    for name, remote in maps.items():
        report['current_source_comparison'][name] = {
            'local_count': len(local), 'remote_count': len(remote),
            'match': local == remote,
            'local_only': sorted(local.keys() - remote.keys()),
            'remote_only': sorted(remote.keys() - local.keys()),
            'different': sorted(k for k in local.keys() & remote.keys()
                                if local[k] != remote[k]),
        }
    report['local_scheduler_python_sha256'] = local
    path = HERE / 'AUDIT.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(path)
    print(json.dumps(report['current_source_comparison'], indent=2))
    for name, arm in report['arms'].items():
        comparison = arm['comparison_vs_legacy_r1']
        print(name, arm['status'], round(arm['host_kj'], 6),
              round(comparison['host_saving_pct'], 3),
              comparison['identical_outputs'], comparison['same_inputs'])


if __name__ == '__main__':
    main()
