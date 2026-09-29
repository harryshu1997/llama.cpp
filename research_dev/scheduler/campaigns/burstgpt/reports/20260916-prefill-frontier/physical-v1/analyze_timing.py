"""Join router diagnostics to native calls within terminal request-proof bounds."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def analyze(run, router_log):
    result = json.loads((run / 'REMOTE_RESIDENT_GATE.json').read_text())
    assert result['status'] == 'PASS'
    request, = result['arms']['reduced']['requests']
    proof = request['proof']
    first, last = proof['phone_first_request_id'], proof['phone_last_request_id']
    calls, usb, router = {}, {}, {}
    for path in (run / 'phone').glob('*.stderr'):
        for line in path.read_text().splitlines():
            if not line.startswith(('S41SERVERFFNUSB ', 'S41SERVERFFNCALL ')):
                continue
            row = dict(re.findall(r'(\w+)=(\S+)', line))
            key = int(row['request'])
            if first <= key <= last:
                target = usb if line.startswith('S41SERVERFFNUSB ') else calls
                assert key not in target
                target[key] = row
    for line in router_log.read_text().splitlines():
        if line.startswith('RESIDENTTIMING '):
            row = json.loads(line.split(' ', 1)[1])
            if first <= row['request_id'] <= last:
                assert row['request_id'] not in router
                router[row['request_id']] = row
    assert len(calls) == len(usb) == proof['phone_call_count']
    shapes = {}
    for tokens in sorted({int(row['tokens']) for row in calls.values()}):
        keys = [key for key, row in calls.items() if int(row['tokens']) == tokens]
        timed = [router[key] for key in keys if key in router]
        if tokens > 1:
            assert len(timed) == len(keys)
        for row in timed:
            assert row['tokens'] == tokens
            assert row['layer'] == int(calls[row['request_id']]['layer'])
        shape = {'calls': len(keys), 'router_samples': len(timed),
            'host_rpc_ms_mean': statistics.mean((int(usb[key]['d2h_completed_ns']) -
                int(usb[key]['started_ns'])) / 1e6 for key in keys),
            'host_h2d_ms_mean': statistics.mean((int(usb[key]['h2d_completed_ns']) -
                int(usb[key]['started_ns'])) / 1e6 for key in keys),
            'worker_compute_ms_mean': statistics.mean(int(usb[key]['compute_us']) / 1000 for key in keys)}
        if timed:
            shape['router_mean_ms'] = {key: statistics.mean(row[key] / 1000 for row in timed)
                for key in ('input_check_us', 'worker_send_us', 'worker_receive_us',
                            'worker_compute_us', 'output_check_us', 'usb_write_us')}
        shapes[str(tokens)] = shape
    return {'schema': 's42-prefill-stage-audit-v1', 'shapes': shapes,
        'proof_bounds': [first, last], 'phone_calls': len(calls),
        'result_sha256': hashlib.sha256((run / 'REMOTE_RESIDENT_GATE.json').read_bytes()).hexdigest(),
        'router_log_sha256': hashlib.sha256(router_log.read_bytes()).hexdigest(),
        'native_timings': {arm: result['arms'][arm]['requests'][0]['context_completion']['terminal']['timings']
                           for arm in ('full', 'reduced')},
        'notes': ['Worker compute includes graph construction, input staging, HTP compute and output staging.',
                  'Router receive including idle is excluded from active phase totals.',
                  'No timestamps from different machines are subtracted.',
                  'Prefill timing is exhaustive; decode router timing is sampled.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('router_log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.router_log)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, indent=2))
