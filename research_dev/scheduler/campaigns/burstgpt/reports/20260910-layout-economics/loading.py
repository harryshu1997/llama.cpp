"""Read-only loading diagnostics; overlapping spans are never additive energy."""

from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
import sys


def native_markers(path):
    markers = {}
    patterns = {
        'model_start': 'load_model: loading model',
        'metadata_ready': 'llama_model_loader: loaded meta data',
        'tensors_start': 'load_tensors: loading model tensors',
        'context_start': 'llama_context: constructing llama_context',
        'warmup_start': 'warming up the model with an empty run',
        'server_ready': 'llama_server: model loaded',
    }
    for index, line in enumerate(path.read_text().splitlines()):
        match = re.match(r'(\d+)\.(\d{2})\.(\d{3})\.(\d{3}) ', line)
        if match is None:
            continue
        minute, second, milli, micro = map(int, match.groups())
        stamp = ((minute * 60 + second) * 1000 + milli) * 1000 + micro
        for key, pattern in patterns.items():
            if pattern in line and (key not in markers or (
                    key in {'metadata_ready', 'tensors_start', 'context_start'}
                    and 'warmup_start' not in markers)):
                markers[key] = dict(native_relative_us=stamp, line=index + 1)
    return markers


def span(markers, start, end):
    if start not in markers or end not in markers:
        return None
    return markers[end]['native_relative_us'] - markers[start]['native_relative_us']


def union_duration(intervals):
    total, stop = 0, 0
    for start, end in sorted(intervals):
        total += max(0, end - max(start, stop))
        stop = max(stop, end)
    return total


def analyze(root):
    result = json.loads((root / 'run/RESULT.json').read_text())
    paid_start = result['paid_start_ns']
    samples = defaultdict(list)
    for line in (root / 'LOADING_PROCESS_SAMPLES.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['status'] == 'VALID':
            samples[row['pid']].append(row)
    database = sqlite3.connect('file:' + str(root / 'CUDA.sqlite') + '?mode=ro', uri=True)
    clock = database.execute('select systemClockNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0]
    global_ids = {pid: global_pid for global_pid, pid in database.execute('select globalPid,pid from PROCESSES')}
    transitions = [(request, transition) for request in result['request_results']
                   for transition in request['terminal_ticket']['transition_receipts']
                   if transition['device_id'].startswith('desktop')]
    servers = []
    for pid, rows in sorted(samples.items()):
        label = Path(rows[0]['stdout']).with_suffix('.stderr').name
        path = root / 'run' / label
        markers = native_markers(path)
        first_us = (rows[0]['monotonic_ns'] - paid_start) // 1000
        owner = next(((request, transition) for request, transition in transitions
                      if transition['started_us'] <= first_us <= transition['finished_us']), None)
        if owner is None:
            servers.append(dict(pid=pid, log=label, native_markers=markers,
                                attribution='startup_or_unmatched_process'))
            continue
        request, transition = owner
        start_ns, end_ns = (paid_start + transition[key] * 1000 for key in ('started_us', 'finished_us'))
        preparation = [row for row in rows if row['monotonic_ns'] <= end_ns]
        copies = [(max(start, start_ns - clock), min(end, end_ns - clock), count)
                  for start, end, count in database.execute(
                      'select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY '
                      'where globalPid=? and copyKind=1 and start<? and end>?',
                      (global_ids.get(pid, -1), end_ns - clock, start_ns - clock))]
        io_first, io_last = preparation[0]['io'], preparation[-1]['io']
        sample_start, sample_end = preparation[0]['monotonic_ns'], preparation[-1]['monotonic_ns']
        activity = [(left['monotonic_ns'], right['monotonic_ns'])
                    for left, right in zip(preparation, preparation[1:])
                    if right['io']['read_bytes'] > left['io']['read_bytes']]
        acquisition = request['dispatch_receipts'][-1]['observed_at_us']
        servers.append({
            'pid': pid, 'model_id': request['model_id'], 'request_id': request['request_id'],
            'log': label, 'native_markers': markers,
            'transition_start_us': transition['started_us'], 'transition_end_us': transition['finished_us'],
            'transition_duration_us': transition['finished_us'] - transition['started_us'],
            'request_queue_wait_us': sum(row['queue_wait_us'] for row in request['dispatch_receipts']),
            'acquisition_to_transition_us': transition['started_us'] - acquisition,
            'metadata_and_model_initialization_us': span(markers, 'model_start', 'tensors_start'),
            'tensor_read_repack_and_transfer_span_us': span(markers, 'tensors_start', 'context_start'),
            'context_initialization_us': span(markers, 'context_start', 'warmup_start'),
            'native_warmup_and_ready_verification_us': span(markers, 'warmup_start', 'server_ready'),
            'storage_read_bytes_sampled': io_last['read_bytes'] - io_first['read_bytes'],
            'logical_read_bytes_sampled': io_last['rchar'] - io_first['rchar'],
            'io_sample_span_us': (sample_end - sample_start) // 1000,
            'storage_activity_bin_duration_us': union_duration(activity) // 1000,
            'gpu_h2d_bytes': sum(row[2] for row in copies),
            'gpu_h2d_union_duration_us': union_duration([(a, b) for a, b, _ in copies]) // 1000,
            'gpu_h2d_copy_count': len(copies),
            'host_startup_and_post_ready_remainder_us': (
                None if 'server_ready' not in markers else transition['finished_us'] - transition['started_us']
                - markers['server_ready']['native_relative_us']),
        })
    database.close()
    return {'schema': 'scheduler-loading-diagnostics-v1', 'servers': servers,
            'caveats': [
                '100 ms process samples, discovered within 500 ms, omit early reads before discovery.',
                'Storage activity bins are not blocked I/O duration; storage, CPU repack and GPU copies overlap.',
                'Native timestamps are process-relative; remainder includes spawn and adapter health/proof work.',
                'Queue wait is separate from loading and is not additive across requests.',
                'No cache flush, prefetch, preload change, or subtraction from paid energy was performed.']}


if __name__ == '__main__':
    with Path(sys.argv[2]).open('x') as output:
        json.dump(analyze(Path(sys.argv[1])), output, sort_keys=True, indent=2)
        output.write('\n')
