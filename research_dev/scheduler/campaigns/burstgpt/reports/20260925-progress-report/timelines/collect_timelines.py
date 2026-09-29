#!/usr/bin/env python3
"""Copy compact, hashed evidence from the two completed eval_v2 arms."""

import json
from pathlib import Path
import subprocess


REMOTE = r'''
import datetime
import hashlib
import json
from pathlib import Path

root = Path('/mnt/storage/s43-two-phone-eval-20260925')
data = {'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'arms': {}, 'sources': []}


def read(path, jsonl=False):
    raw = path.read_bytes()
    data['sources'].append({'path': str(path), 'bytes': len(raw),
                            'sha256': hashlib.sha256(raw).hexdigest()})
    if jsonl:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    return json.loads(raw)


for label, name in (('baseline', 'desktop-legacy-ev2'), ('treatment', 'two-phone-ev2')):
    run = root / ('inputs-' + name) / 'run-eval/run'
    result = read(run / 'RESULT.json')
    assert result['status'] == 'PASS'
    samples = read(run / 'resource-samples.jsonl', jsonl=True)
    arm = {k: result[k] for k in ('paid_start_ns', 'paid_end_ns', 'duration_us',
                                 'trace_energy', 'status')}
    arm['path'] = str(run)
    arm['requests'] = []
    for row in result['request_results']:
        request = {k: row[k] for k in ('request_id', 'model_id', 'output_tokens',
                                      'first_token_ns', 'combined_request_index',
                                      'physical_execution_proof')}
        request['finished_at_us'] = row['completion']['actual_end_us']
        request['started_at_us'] = row['completion']['execution_receipt']['started_us']
        arm['requests'].append(request)
    arm['samples'] = [
        {'gpu': {k: row['gpu'][k] for k in ('sample_t_ns', 'power_mw')},
         'rapl_package': {k: row['rapl_package'][k] for k in
                          ('sample_t_ns', 'energy_uj', 'max_energy_range_uj')}}
        for row in samples if row.get('gpu') and row.get('rapl_package')]
    arm['discarded_host_samples'] = len(samples) - len(arm['samples'])
    if label == 'treatment':
        store = read(run / 'ADAPTIVE_DECODE_OBSERVATIONS.json')
        hashes = {row['physical_execution_proof']['adaptive_grouped_observation_sha256']
                  for row in result['request_results']
                  if row.get('physical_execution_proof', {}).get('adaptive_grouped_observation_sha256')}
        arm['groups'] = [g for g in store['groups'] if g['grouped_observation_sha256'] in hashes]
        assert {g['grouped_observation_sha256'] for g in arm['groups']} == hashes
        arm['ignored_historical_groups'] = len(store['groups']) - len(arm['groups'])
        arm['token_events'] = [e for e in result['adaptive_timing_events']
                               if e['kind'] == 'DECODE_BOUNDARY_OBSERVED']
    data['arms'][label] = arm
print(json.dumps(data, separators=(',', ':')))
'''


def main():
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
         'zhihao@172.20.74.85', 'python3 -'],
        input=REMOTE, text=True, capture_output=True, check=True,
    )
    data = json.loads(result.stdout)
    stamp = data['captured_at_utc'][:19].replace('-', '').replace(':', '')
    output = Path(__file__).resolve().parent / 'sources'
    output.mkdir(exist_ok=True)
    path = output / f'timeline_evidence_{stamp}.json'
    path.write_text(json.dumps(data, separators=(',', ':')) + '\n')
    print(path)
    print('Bytes:', path.stat().st_size)
    for name, arm in data['arms'].items():
        print(name, 'samples:', len(arm['samples']), 'requests:', len(arm['requests']),
              'groups:', len(arm.get('groups', [])), 'token events:', len(arm.get('token_events', [])))


if __name__ == '__main__':
    main()
