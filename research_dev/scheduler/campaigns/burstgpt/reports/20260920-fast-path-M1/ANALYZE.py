"""Freeze component sums before reading held-out totals, then check M1."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from research_dev.scheduler._internal.profile_materializer import read_scheduler_inventory


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def digest(path):
    with path.open('rb') as stream:
        return 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()


def execution(root, kind, ubatch, fixture):
    arm = root / f'{kind}-{ubatch}'
    result = read(arm/'RESULT.json')
    request = read(arm/'REQUEST.json')
    record = read(next(arm.glob('EXECUTION-*.json')))
    identity = read(arm/'SERVER_IDENTITY.json')
    contract = identity['launch_contract']
    assert result['status'] == 'COMPLETED' and record['error'] is None
    assert len(request['prompt_tokens']) == 9737 and len(record['tokens']) == 64
    assert contract['ubatch_size'] == ubatch and contract['batch_size'] == 2048
    assert contract['parallel'] == 1
    assert contract['threads'] == contract['threads_batch'] == 8 and contract['cpu_affinity'] is None
    assert bool(contract['scheduler_trace_path']) == (kind in ('traced', 'diagnostic'))
    ready = result['memory_ready']['cgroup']
    memory = result['memory_finished']['cgroup']
    assert memory['memory.swap.max'] == 0
    assert memory['memory.events']['oom'] == memory['memory.events']['oom_kill'] == 0
    if fixture == 'pressure-limited':
        assert memory['memory.max'] == 19327352832
    else:
        assert memory['memory.max'] == 'max' or memory['memory.max'] >= memory['memory.peak'] + 4*1024**3
        assert ready['memory.events']['max'] == memory['memory.events']['max'] == 0
        assert ready['memory.events']['high'] == memory['memory.events']['high'] == 0
        reference = root.parent / f'{"traced" if kind == "diagnostic" else kind}-{ubatch}'
        old_identity = read(reference/'SERVER_IDENTITY.json')
        assert identity['runtime'] == old_identity['runtime']
        assert {k: v for k, v in contract.items() if k != 'scheduler_trace_path'} == {
            k: v for k, v in old_identity['launch_contract'].items() if k != 'scheduler_trace_path'}
        assert request == read(reference/'REQUEST.json')
        assert read(arm/'KV_PLAN.json') == read(reference/'KV_PLAN.json')
    memory = {**memory, 'events_max_ready': ready['memory.events']['max'],
              'events_max_finish': memory['memory.events']['max'],
              'events_max_delta': memory['memory.events']['max'] - ready['memory.events']['max']}
    return arm, record, identity, memory


def freeze(root, fixture):
    assert not any((root/f'heldout-{u}').exists() for u in (128, 1024))
    predictions = {}
    for ubatch in (128, 1024):
        arm, record, identity, memory = execution(root, 'traced', ubatch, fixture)
        kwargs = {'start_us': record['started_ns']//1000, 'end_us': record['finished_ns']//1000}
        prefill = read_scheduler_inventory(arm/'SCHED_TRACE.jsonl', phase='prefill', **kwargs)
        decode = read_scheduler_inventory(arm/'SCHED_TRACE.jsonl', phase='decode', **kwargs)
        assert sum(g['tokens'] for g in prefill['graphs']) == 9737
        assert len(decode['graphs']) == 63 and all(g['tokens'] == 1 for g in decode['graphs'])
        sequence = [g['tokens'] for g in prefill['graphs']]
        save(root/f'INVENTORY-{ubatch}.json', {'prefill': prefill, 'decode': decode})
        save(root/f'REPRESENTATIVE-{ubatch}.json', {'prefill': prefill['graphs'][0], 'decode': decode['graphs'][0]})
        predictions[str(ubatch)] = {
            'predicted_prefill_s': prefill['accounted_host_us']/1e6,
            'traced_prefill_s': record['prefill_s'],
            'ubatch_sequence': sequence,
            'prefill_components': {k: v for k, v in prefill.items() if k not in ('graphs', 'components', 'weight_transfers')},
            'decode_components': {k: v for k, v in decode.items() if k not in ('graphs', 'components', 'weight_transfers')},
            'trace_sha256': digest(arm/'SCHED_TRACE.jsonl'),
            'execution_sha256': digest(next(arm.glob('EXECUTION-*.json'))),
            'runtime': identity['runtime'], 'memory_peak_bytes': memory['memory.peak'],
            'memory_fixture': fixture, 'memory_events': {k: v for k, v in memory.items() if k.startswith('events_max_')},
        }
    frozen = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'single_runs': True,
        'method': 'Sum calibrated copy host calls, compute host calls, and existing scheduler completion waits. '
                  'Full prompt sequence including remainder; no fitted scaling or held-out totals.',
        'analyzer_sha256': digest(Path(__file__)),
        'reader_sha256': digest(Path(read_scheduler_inventory.__code__.co_filename)),
        'predictions': predictions}
    save(root/'FROZEN_PREDICTIONS.json', frozen)
    print(json.dumps(frozen, indent=2))


def check(root, fixture):
    frozen = read(root/'FROZEN_PREDICTIONS.json')
    assert frozen['analyzer_sha256'] == digest(Path(__file__))
    assert frozen['reader_sha256'] == digest(Path(read_scheduler_inventory.__code__.co_filename))
    results = {}
    for ubatch in (128, 1024):
        calibration = frozen['predictions'][str(ubatch)]
        assert calibration['memory_fixture'] == fixture
        traced, old, _, _ = execution(root, 'traced', ubatch, fixture)
        _, measured, identity, memory = execution(root, 'heldout', ubatch, fixture)
        assert calibration['trace_sha256'] == digest(traced/'SCHED_TRACE.jsonl')
        assert calibration['execution_sha256'] == digest(next(traced.glob('EXECUTION-*.json')))
        assert calibration['runtime'] == identity['runtime']
        predicted = calibration['predicted_prefill_s']
        error = abs(predicted - measured['prefill_s']) / measured['prefill_s']
        closure_error = abs(predicted - old['prefill_s']) / old['prefill_s']
        results[str(ubatch)] = {
            'predicted_prefill_s': predicted, 'measured_prefill_s': measured['prefill_s'],
            'relative_error': error, 'within_10_percent': error <= .1,
            'traced_prefill_s': old['prefill_s'], 'traced_accounting_relative_error': closure_error,
            'traced_accounting_within_10_percent': closure_error <= .1,
            'observed_traced_vs_heldout_time_change_fraction': old['prefill_s']/measured['prefill_s']-1,
            'token_outputs_identical': old['tokens'] == measured['tokens'],
            'heldout_prefill_host_J': measured['prefill_host_energy']['server_compute_device_energy_j'],
            'heldout_request_host_J': measured['request_host_energy']['server_compute_device_energy_j'],
            'heldout_decode_ms_per_token': measured['decode_ms_per_token'],
            'assumed_phone_idle_W': .875, 'assumed_phone_request_J': measured['request_s']*.875,
            'memory_peak_bytes': memory['memory.peak'],
            'memory_fixture': fixture, 'memory_events': {k: v for k, v in memory.items() if k.startswith('events_max_')},
        }
    p128, p1024 = (frozen['predictions'][str(u)]['prefill_components'] for u in (128, 1024))
    measured_delta = results['128']['measured_prefill_s'] - results['1024']['measured_prefill_s']
    weight_delta = (p128['weight_copy_host_excluding_wait_us'] - p1024['weight_copy_host_excluding_wait_us'])/1e6
    identified = p128['weight_copy_bytes'] > p1024['weight_copy_bytes'] and weight_delta > 0 and measured_delta > 0
    passed = identified and all(r['within_10_percent'] and r['traced_accounting_within_10_percent']
                                and r['token_outputs_identical'] for r in results.values())
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS' if passed else 'FAIL',
        'single_runs': True, 'frozen_predictions_sha256': digest(root/'FROZEN_PREDICTIONS.json'), 'arms': results,
        'weight_streaming_identified': identified,
        'weight_bytes_128': p128['weight_copy_bytes'], 'weight_bytes_1024': p1024['weight_copy_bytes'],
        'measured_prefill_delta_s': measured_delta, 'weight_copy_host_excluding_wait_delta_s': weight_delta,
        'weight_copy_delta_over_measured_delta': weight_delta/measured_delta if measured_delta else None,
        'timing_note': p128['timing_note'],
        'overhead_note': 'One traced/untraced run per ubatch; observed time change also contains run-to-run variation.'}
    save(root/'CHECK_M1.json', result)
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


def diagnose(root, fixture):
    failed = read(root/'CHECK_M1.json')
    assert failed['status'] == 'FAIL' and not failed['arms']['1024']['within_10_percent']
    old = read(root/'INVENTORY-1024.json')['prefill']
    arm, record, identity, memory = execution(root, 'diagnostic', 1024, fixture)
    frozen = read(root/'FROZEN_PREDICTIONS.json')['predictions']['1024']
    assert identity['runtime'] == frozen['runtime']
    new = read_scheduler_inventory(arm/'SCHED_TRACE.jsonl', phase='prefill',
                                  start_us=record['started_ns']//1000, end_us=record['finished_ns']//1000)
    assert [g['tokens'] for g in old['graphs']] == [g['tokens'] for g in new['graphs']]
    rows = []
    totals = {}
    for index, (before, after) in enumerate(zip(old['graphs'], new['graphs'])):
        assert len(before['splits']) == len(after['splits'])
        for a, b in zip(before['splits'], after['splits']):
            for key in ('split', 'backend', 'node_count', 'nodes', 'copied_bytes'):
                assert a[key] == b[key], (index, key)
            row = {'ubatch_index': index, 'tokens': before['tokens'], 'split': a['split'],
                   'backend': a['backend'], 'nodes': a['nodes'], 'copied_bytes': a['copied_bytes']}
            for key in ('copy_host_us', 'compute_host_us', 'wall_us'):
                row[key] = {'calibration': a[key], 'diagnostic': b[key], 'delta': b[key]-a[key]}
            for label, graph in (('calibration', before), ('diagnostic', after)):
                row.setdefault('copy_wait_us', {})[label] = sum(
                    c['wait_us'] for c in graph['copies'] if c['split'] == a['split'])
            waits = row['copy_wait_us']
            waits['delta'] = waits['diagnostic'] - waits['calibration']
            rows.append(row)
            total = totals.setdefault(str(a['split']), {'split': a['split'], 'backend': a['backend'],
                                      'node_names': [n['name'] for n in a['nodes']], 'graphs': 0})
            assert total['backend'] == a['backend']
            total['graphs'] += 1
            for key in ('copy_host_us', 'compute_host_us', 'wall_us', 'copy_wait_us'):
                sums = total.setdefault(key, {'calibration': 0, 'diagnostic': 0, 'delta': 0})
                for label, value in row[key].items():
                    sums[label] += value
    result = {'note': 'The failed held-out arm is untraced and has no per-split timings. These deltas use '
                      'a subsequent traced diagnostic repeat, not the failed request itself; no acceptance retry or refit.',
              'frozen_prediction_s': frozen['predicted_prefill_s'],
              'calibration_prefill_s': frozen['traced_prefill_s'],
              'failed_heldout_prefill_s': failed['arms']['1024']['measured_prefill_s'],
              'diagnostic_prefill_s': record['prefill_s'],
              'diagnostic_components': {k: v for k, v in new.items() if k not in ('graphs', 'components', 'weight_transfers')},
              'memory': memory, 'per_graph_split': rows, 'split_totals': list(totals.values())}
    save(root/'DIAGNOSTIC_INVENTORY-1024.json', new)
    save(root/'SPLIT_DELTAS.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_graph_split', 'split_totals')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('freeze', 'check', 'diagnose'))
    parser.add_argument('--physical', type=Path, required=True)
    parser.add_argument('--memory-fixture', choices=('pressure-limited', 'no-pressure'), default='pressure-limited')
    args = parser.parse_args()
    raise SystemExit({'freeze': freeze, 'check': check, 'diagnose': diagnose}[args.mode](
        args.physical, args.memory_fixture))
