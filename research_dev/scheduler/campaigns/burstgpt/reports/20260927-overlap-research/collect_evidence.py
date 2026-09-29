#!/usr/bin/env python3
"""Read saved overlap summaries without changing the shared rig."""

import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
REMOTE = r'''
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

root = Path('/mnt/storage/s43-two-phone-eval-20260925/inputs-two-phone-tp2/run-eval/run')
rows, shapes, restores, releases = [], [], [], []
sources = {}
resets = 0
for path in sorted(root.glob('large-model*.stderr')):
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for line in path.open():
        resets += 'S41SERVERFFNRESET ' in line
        for marker, target in [('S41SERVERFFN {', rows), ('S41SERVERFFNSHAPE {', shapes)]:
            if marker in line:
                row = json.loads(line[line.index(marker) + len(marker) - 1:])
                row['source'] = str(path)
                target.append(row)
        if 'dormant_host_share phase=' in line:
            row = dict(re.findall(r'(\w+)=(\S+)', line))
            row['source'] = str(path)
            (restores if row.get('phase') == 'local' else releases).append(row)

weighted = {}
for label, helper in [('qwen_op15', 'op15'), ('qwen_pixel', 'pixel10pro'), ('gemma_op15', None)]:
    selected = [r for r in rows if r.get('helper') == helper and r.get('calls', 0) > 0]
    count = sum(r['calls'] for r in selected)
    if not count:
        continue
    means = {key: sum(r['calls'] * r[key] for r in selected) / count for key in (
        'rpc_mean_ms', 'compute_mean_ms', 'host_mean_ms', 'wait_mean_ms',
        'useful_overlap_mean_ms')}
    weighted[label] = {'calls_in_shutdown_summaries': count, 'summary_count': len(selected), **means}

native = Path('/mnt/storage/s42-trace-v2-20260921-prep/source')
files = ('examples/layersplit/ffn-split-client.cpp', 'src/llama-graph.cpp',
         'src/llama-model.cpp', 'tools/server/server.cpp', 'ggml/src/ggml-backend.cpp')
report = {
    'captured_at_utc': datetime.now(timezone.utc).isoformat(),
    'run': str(root),
    'scope': 'Saved shutdown summaries; not all request proof calls or whole-server time',
    'source_sha256': sources, 'shutdown_summaries': rows, 'shape_summaries': shapes,
    'weighted_means': weighted, 'reset_log_lines': resets,
    'restores': restores, 'releases': releases,
    'restore_elapsed_s': sum(int(r['elapsed_us']) for r in restores) / 1e6,
    'restore_byte_mentions': sum(int(r['restored_bytes']) for r in restores),
    'release_elapsed_s': sum(int(r['elapsed_us']) for r in releases) / 1e6,
    'deploy_native_source_sha256': {f: hashlib.sha256((native / f).read_bytes()).hexdigest() for f in files},
}
print(json.dumps(report, sort_keys=True))
'''


def main():
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
         'zhihao@172.20.74.85', 'python3 -'],
        input=REMOTE, text=True, capture_output=True, check=True,
    )
    report = json.loads(result.stdout)
    report['local_native_source_matches_deploy'] = {
        name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() == digest
        for name, digest in report['deploy_native_source_sha256'].items()
    }
    assert all(report['local_native_source_matches_deploy'].values())
    (HERE / 'EVIDENCE.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report['weighted_means'], indent=2))
    print('Restore seconds:', report['restore_elapsed_s'])
    print('Native source comparison: PASS')


if __name__ == '__main__':
    main()
