"""Reproduce matched real-weight worker timings and exact-output checks."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('physical', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
directories = {mode: args.physical / ('worker-' + mode + '-v4')
               for mode in ('disabled', 'enabled')}
data = {mode: json.loads((path / 'RESULT.json').read_text()) for mode, path in directories.items()}
assert len(data['disabled']) == len(data['enabled']) == 48
maximum_error = 0.0
for off, on in zip(data['disabled'], data['enabled'], strict=True):
    for field in ('id', 'rows', 'columns', 'repeat', 'input_sha256', 'output_sha256'):
        assert off[field] == on[field], (field, off, on)
    arrays = [np.load(path / ('output-' + str(off['id']) + '.npy')) for path in directories.values()]
    assert np.array_equal(*arrays)
    maximum_error = max(maximum_error, float(np.max(np.abs(arrays[0] - arrays[1]))))
shapes = []
for rows, columns in sorted({(row['rows'], row['columns']) for row in data['disabled']}):
    shape = {'rows': rows, 'columns': columns, 'fraction': columns / 15360}
    for mode, records in data.items():
        selected = [row for row in records if (row['rows'], row['columns']) == (rows, columns)]
        shape[mode] = {key + '_median': statistics.median(row[key] for row in selected)
                       for key in ('compute_us', 'rpc_us')}
    shape['compute_reduction_percent'] = 100 * (1 - shape['enabled']['compute_us_median'] /
                                              shape['disabled']['compute_us_median'])
    shapes.append(shape)
test_log = args.physical / 'native-enabled-v4/test.log'
text = test_log.read_text()
assert '28/28 tests passed' in text
fused = [line[line.index('ggml-hex:'):] for line in text.splitlines() if '|hmx-ffn-glu ' in line]
assert len(fused) == 10
vtcm = [int(re.search(r'vtcm (\d+)', line)[1]) for line in fused]
assert max(vtcm) <= 8 * 1024**2
summary = {
    'schema': 's42-fp16-ffn-worker-ab-v1', 'status': 'PASS',
    'worker_calls_per_mode': 48, 'shapes': shapes,
    'maximum_absolute_error_between_modes': maximum_error,
    'all_outputs_bit_identical': True,
    'hardware_tests_passed': 28, 'profiled_fused_operations': len(fused),
    'fused_vtcm_bytes_min_max': [min(vtcm), max(vtcm)],
    'fused_profile_lines': fused,
    'hashes': {str(path.relative_to(args.physical)): digest(path)
               for path in [test_log, *[p / 'RESULT.json' for p in directories.values()]]},
    'notes': ['Median of three calls per shape, same four-HVX-thread binary, profiling disabled.',
              'TCP/ADB forwarding is diagnostic, not the production FunctionFS transport.',
              'Rows one and four use the unchanged HVX path; timing differences there are run variation.',
              'Native CPU comparison keeps the existing 0.005 NMSE tolerance.',
              'Scratch layout tests validate alignment, capacity and invalid dimensions.',
              'Graph arena intermediate allocations remain; fused intermediates avoid DDR writes/reads.'],
}
with args.output.open('x') as stream:
    json.dump(summary, stream, indent=2, sort_keys=True)
    stream.write('\n')
for shape in shapes:
    print(shape)
print('maximum error', maximum_error, 'fused operations', len(fused))
