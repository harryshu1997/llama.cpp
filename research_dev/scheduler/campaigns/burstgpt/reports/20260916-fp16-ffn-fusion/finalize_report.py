"""Verify frozen local evidence and record the final validation inventory."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
physical = ROOT / 'physical-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


native = {}
for mode in ('disabled', 'enabled'):
    directory = physical / ('native-' + mode + '-v4')
    result = json.loads((directory / 'RESULT.json').read_text())
    assert result['returncode'] == 0
    log = (directory / 'test.log').read_text()
    assert '28/28 tests passed' in log
    native[mode] = {'passed': 28, 'fused_operations': log.count('|hmx-ffn-glu '),
                    'log_sha256': sha(directory / 'test.log')}
assert native['disabled']['fused_operations'] == 0
assert native['enabled']['fused_operations'] == 10
profile = physical / 'worker-profiled-v4/worker.log'
lines = [line for line in profile.read_text().splitlines() if '|hmx-ffn-glu ' in line]
assert len(lines) == 12
assert all('vtcm 7534592 ' in line for line in lines)
worker = json.loads((ROOT / 'WORKER_COMPARISON.json').read_text())
document = json.loads((ROOT / 'DOCUMENT_COMPARISON.json').read_text())
assert worker['status'] == document['status'] == 'PASS'
assert document['fusion_mode_output_token_match']
assert worker['all_outputs_bit_identical']
assert subprocess.run(['git', 'diff', '--check'], cwd=ROOT.parents[5]).returncode == 0
for row in json.loads((ROOT / 'CHANGES.json').read_text())['files']:
    assert sha(ROOT.parents[5] / row['path']) == row['after_sha256']
record = {
    'schema': 's42-fp16-ffn-fusion-validation-v1', 'status': 'PASS',
    'native': native, 'scratch_bounds': 'PASS in both native invocations',
    'worker_calls_per_mode': worker['worker_calls_per_mode'],
    'real_weight_profiled_fused_operations': len(lines),
    'real_weight_profile_log_sha256': sha(profile),
    'document_gate_modes_passed': 2,
    'document_requests_per_mode': {'desktop_control': 1, 'relocated': 1},
    'document_shape': {'input_tokens': 5261, 'output_tokens': 64},
    'fusion_modes_identical_output_tokens': 64,
    'broad_suite_run': False, 'long_trace_run': False,
    'git_diff_check': 'PASS',
    'remote_artifact_root': '/mnt/storage/s42-fp16-ffn-fusion-20260916-v1-yNEEYR',
    'hashes': {name: sha(ROOT / name) for name in ('CHANGES.json', 'BUILD_SOURCE_MANIFEST.json',
        'WORKER_COMPARISON.json', 'DOCUMENT_COMPARISON.json', 'physical-v1/SOURCE_MANIFEST.json')},
}
with (ROOT / 'TESTS.json').open('x') as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(record, indent=2))
