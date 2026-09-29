"""Check the fixed multi-slot gate without fitting or discarding any request."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct

import numpy as np

from research_dev.scheduler.adapters.llama_server_contracts import parse_llama_server_ffn_call


def read(path):
    return json.loads(path.read_text())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


class LogitsTrace:
    """Index the native pre-sampling f32 rows without loading the whole trace."""

    def __init__(self, path, entries):
        self.path = path
        self.rows = {}
        tasks = {}
        expected = {(entry['slot_id'], step) for entry in entries
                    for step in range(1, len(entry['tokens']) + 1)}
        size = path.stat().st_size
        vocabulary = None
        with path.open('rb') as stream:
            require(stream.read(8) == b'S41LOG1\0', 'unsupported raw logits schema')
            while stream.tell() < size:
                header = stream.read(16)
                require(len(header) == 16, 'truncated logits header')
                slot, task, step, count = struct.unpack('<IIII', header)
                require(1 < count <= 1_000_000, 'invalid logits vocabulary size')
                require(vocabulary is None or vocabulary == count, 'logits vocabulary changed')
                vocabulary = count
                require((slot, step) in expected and (slot, step) not in self.rows,
                        'unexpected or duplicate logits slot/step')
                require(tasks.setdefault(slot, task) == task, 'multiple tasks reused a logits slot')
                require(stream.tell() + count * 4 <= size, 'truncated logits row')
                self.rows[slot, step] = stream.tell(), count
                stream.seek(count * 4, 1)
        require(self.rows.keys() == expected, 'logits coverage is incomplete')

    def row(self, slot, step):
        offset, count = self.rows[slot, step]
        with self.path.open('rb') as stream:
            stream.seek(offset)
            values = np.fromfile(stream, dtype='<f4', count=count).astype(np.float64)
        require(len(values) == count and np.isfinite(values).all(), 'invalid logits values')
        return values


def compare_tokens(assisted, baseline, phone_directory, host_directory):
    traces = None
    comparisons = []
    require(len(assisted) == len(baseline), 'paired request counts differ')
    for left, right in zip(assisted, baseline):
        require(left['index'] == right['index'] and left['prompt_tokens'] == right['prompt_tokens'],
                'paired requests differ')
        require(left['slot_id'] == right['slot_id'], 'paired slot assignment differs')
        require(len(left['tokens']) == len(right['tokens']), 'paired output lengths differ')
        differences = [i + 1 for i, (a, b) in enumerate(zip(left['tokens'], right['tokens'])) if a != b]
        row = {'request_index': left['index'], 'phone_slot': left['slot_id'], 'host_slot': right['slot_id'],
               'prompt_tokens': len(left['prompt_tokens']), 'tokens': len(left['tokens']),
               'matching_tokens': len(left['tokens']) - len(differences), 'differences': differences,
               'first_mismatch': next(iter(differences), None), 'mismatches': [],
               'acceptance': 'EXACT', 'accepted': True}
        if differences:
            try:
                if traces is None:
                    traces = (LogitsTrace(phone_directory/'LOGITS.bin', assisted),
                              LogitsTrace(host_directory/'LOGITS.bin', baseline))
                for step in differences:
                    phone = traces[0].row(left['slot_id'], step)
                    host = traces[1].row(right['slot_id'], step)
                    require(phone.shape == host.shape, 'paired logits vocabularies differ')
                    top = np.argsort(-host, kind='stable')[:2]
                    denominator = float(np.dot(host, host))
                    require(denominator > 0, 'zero host logits norm')
                    delta = phone - host
                    nmse = float(np.dot(delta, delta) / denominator)
                    margin = float(host[top[0]] - host[top[1]])
                    row['mismatches'].append({'step': step, 'host_token': right['tokens'][step-1],
                        'phone_token': left['tokens'][step-1], 'host_top1_token': int(top[0]),
                        'host_top2_token': int(top[1]), 'host_top1_logit': float(host[top[0]]),
                        'host_top2_logit': float(host[top[1]]), 'margin': margin, 'nmse': nmse,
                        'shared_token_prefix': step == differences[0],
                        'comparison': 'acceptance' if step == differences[0] else 'after_context_divergence'})
                first = row['mismatches'][0]
                row['accepted'] = first['margin'] <= 0.05 and first['nmse'] <= 5e-4
                row['acceptance'] = 'NEAR_TIE' if row['accepted'] else 'FAULT'
            except (ValueError, OSError, KeyError) as error:
                row.update(accepted=False, acceptance='FAULT', error=str(error))
        comparisons.append(row)
    return comparisons


def phone_timing(directory, result, n, minimum_steps=512):
    lines = read(directory/'SERVER_FFN_LINES.json')
    calls, transfers = {}, {}
    for line in lines:
        call = parse_llama_server_ffn_call(line.rstrip())
        if call is not None:
            require(call.request_id not in calls, 'duplicate logical phone call')
            calls[call.request_id] = call
        if line.startswith('S41SERVERFFNUSB '):
            row = {key: int(value) for key, value in (item.split('=') for item in line.split()[1:])}
            require(row['request'] not in transfers, 'duplicate USB request')
            transfers[row['request']] = row
    require(bool(calls) and calls.keys() == transfers.keys(), 'phone calls and USB transfers differ')
    members = set(result['cohort']['members'])
    full = []
    for rid, call in calls.items():
        row = transfers[rid]
        require((row['layer'], row['tokens'], row['columns']) == (call.layer, call.tokens, call.columns),
                'one coalesced transfer must match each native call')
        require(call.tokens == len(call.contexts) and call.tokens <= n, 'decode must have one row per slot')
        require(all(c.rows == 1 and c.plan_generation == 1 and c.scheduler_request_id in members for c in call.contexts),
                'unknown or non-decode context')
        require(row['started_ns'] <= row['h2d_completed_ns'] <= row['d2h_completed_ns'], 'USB timestamps reversed')
        if call.tokens == n:
            full.append((call, row))
    per_layer = Counter(call.layer for call, row in full)
    require(set(per_layer) == set(range(18)), 'full cohort did not visit every owned layer')
    require(min(per_layer.values()) >= minimum_steps, f'fewer than {minimum_steps} full-cohort decode steps')
    compute = sum(row['compute_us'] for call, row in full)
    rpc = sum(row['d2h_completed_ns'] - row['started_ns'] for call, row in full)
    # Each generated token traverses 18 owned layers. Count a shared call once.
    slot_tokens = len(full) * n / 18
    return {'physical_calls': len(calls), 'rows': sum(call.tokens for call in calls.values()),
            'calls_by_batch': dict(sorted(Counter(call.tokens for call in calls.values()).items())),
            'full_batch_calls': len(full), 'full_batch_steps_by_layer': dict(sorted(per_layer.items())),
            'full_batch_slot_tokens': slot_tokens, 'phone_compute_us': compute, 'phone_rpc_ns': rpc,
            'phone_compute_ms_per_token_per_slot': compute / 1000 / slot_tokens,
            'phone_rpc_ms_per_token_per_slot': rpc / 1e6 / slot_tokens}


def arm(root, n, name, output_tokens=576, minimum_steps=512, directory=None):
    directory = directory or root/f'n{n}-{name}'
    require(not (directory/'WATCHDOG_FAILURE.json').exists(), 'decode watchdog fired')
    result = read(directory/'RESULT.json')
    identity = read(directory/'SERVER_IDENTITY.json')
    contract = identity['launch_contract']
    require(result['status'] == 'COMPLETED', 'arm did not complete')
    require(contract['parallel'] == n and contract['ubatch_size'] == 1024 and contract['batch_size'] == 2048,
            'arm batch geometry changed')
    require(contract['threads'] == contract['threads_batch'] == 8 and contract['cpu_affinity'] is None,
            'arm CPU placement changed')
    cohort = result['cohort']
    require(cohort['watchdog_passed'], 'watchdog did not pass')
    entries = sorted((read(path) for path in directory.glob('EXECUTION-*.json')), key=lambda entry: entry['index'])
    require(len(entries) == n and len({entry['slot_id'] for entry in entries}) == n, 'slot identities differ')
    require(all(len(entry['tokens']) == output_tokens and entry['error'] is None for entry in entries), 'request failed or truncated')
    ready, finish = result['memory_ready']['cgroup'], result['memory_finished']['cgroup']
    require(ready['memory.events']['max'] == finish['memory.events']['max'] == 0, 'memory.max reclaim occurred')
    require(finish['memory.max'] == 'max' and finish['memory.swap.max'] == 0, 'memory fixture changed')
    require(finish['memory.events']['oom'] == finish['memory.events']['oom_kill'] == 0, 'OOM occurred')
    energy = cohort['decode_host_energy']
    active = cohort['phone_active_union_s']
    row = {'parallel': n, 'arm': name, 'single_run': True,
        'request_host_j': cohort['request_host_energy']['server_compute_device_energy_j'],
        'decode_host_j': energy['server_compute_device_energy_j'], 'decode_s': cohort['decode_s'],
        'decode_host_w': energy['server_compute_device_energy_j']/cohort['decode_s'],
        'decode_ms_per_token_by_slot': [entry['decode_ms_per_token'] for entry in entries],
        'assumed_phone_decode_w': 4.5 if name == 'combined' else 0.875,
        'assumed_phone_idle_w': 0.875,
        'assumed_phone_request_j': active*4.5 + (cohort['request_s']-active)*0.875,
        'memory_peak_bytes': finish['memory.peak'], 'events_max_ready': ready['memory.events']['max'],
        'events_max_finish': finish['memory.events']['max'],
        'events_max_delta': finish['memory.events']['max']-ready['memory.events']['max']}
    if contract.get('logits_trace_path'):
        require(Path(contract['logits_trace_path']).name == 'LOGITS.bin', 'raw logits filename differs')
        trace = LogitsTrace(directory/'LOGITS.bin', entries)
        row['logits_capture'] = {'rows': len(trace.rows), 'bytes': trace.path.stat().st_size,
                                'schema': 's41-logits-v1', 'stage': 'before_sampling'}
    if name == 'combined':
        require(all(entry['exact'] for entry in result['phone_proof_summary'].values()), 'per-slot phone row count differs')
        require(len(result['phone_proof_summary']) == n, 'phone proof omitted a member')
        require(contract['ffn_environment']['S41_SERVER_FFN_USB_BATCH_PLAN'] == 'coalesced-batch', 'wrong packing mode')
        row['phone_timing'] = phone_timing(directory, result, n, minimum_steps)
    return row, entries, identity


def diagnostic(directory, steps=5, output_tokens=64, require_metrics=False):
    result = read(directory/'RESULT.json')
    require(result['status'] == 'COMPLETED', 'diagnostic arm did not complete')
    require(not (directory/'WATCHDOG_FAILURE.json').exists(), 'diagnostic watchdog fired')
    identity = read(directory/'SERVER_IDENTITY.json')
    require(identity['launch_contract']['ffn_row_diagnostic_steps'] == steps, 'diagnostic step contract differs')
    groups, member_steps = {}, {}
    for line in read(directory/'SERVER_FFN_LINES.json'):
        if not line.startswith('S41SERVERFFNROW '):
            continue
        row = dict(item.split('=', 1) for item in line.split()[1:])
        for name in ('call', 'layer', 'ubatch_row', 'member', 'slot_id', 'payload_row',
                     'position', 'decoded', 'applied', 'step'):
            row[name] = int(row[name])
        row['request_id'] = bytes.fromhex(row['request_id']).decode('ascii')
        require(1 <= row['step'] <= steps, 'row outside the requested assisted steps')
        require(row['decoded'] - row['applied'] + 1 == row['step'], 'diagnostic acknowledgement boundary differs')
        require(len(row['wire_sha256']) == len(row['f32_sha256']) == 64, 'invalid SHA-256 row record')
        key = row['call'], row['ubatch_row']
        group = groups.setdefault(key, {})
        require(row['stage'] not in group, 'duplicate diagnostic stage for a row')
        group[row['stage']] = row
        member_steps.setdefault((row['request_id'], row['layer']), set()).add(row['step'])
    require(groups, 'no diagnostic row records')
    executions = sorted((read(path) for path in directory.glob('EXECUTION-*.json')), key=lambda row: row['index'])
    members = {row['request_id']: row['slot_id'] for row in executions}
    require(len(members) == 2 and all(len(row['tokens']) == output_tokens for row in executions), 'diagnostic shape differs')
    require(set(member_steps) == {(rid, layer) for rid in members for layer in range(18)}, 'diagnostic omitted a member or layer')
    require(all(observed == set(range(1, steps + 1)) for observed in member_steps.values()), 'diagnostic coverage is incomplete')
    mismatches = []
    for key, group in sorted(groups.items()):
        require(set(group) == {'input', 'local', 'returned'}, 'diagnostic row lacks input, shadow or return')
        metadata = {k: v for k, v in group['input'].items() if k not in ('stage', 'f32_sha256', 'wire_sha256', 'local_rel_l2', 'local_max_abs', 'local_l2')}
        require(all({k: v for k, v in row.items() if k not in ('stage', 'f32_sha256', 'wire_sha256', 'local_rel_l2', 'local_max_abs', 'local_l2')} == metadata
                    for row in group.values()), 'diagnostic mapping changed within one call')
        require(members[metadata['request_id']] == metadata['slot_id'], 'diagnostic slot differs from its request')
        if group['local']['wire_sha256'] != group['returned']['wire_sha256']:
            alternatives = [other['local'] for (call, row), other in groups.items()
                            if call == key[0] and row != key[1]
                            and other['local']['wire_sha256'] == group['returned']['wire_sha256']]
            mismatches.append({**metadata, 'input': group['input'], 'local': group['local'],
                'returned': group['returned'], 'matching_other_local_rows': alternatives})
    numeric = numeric_rows(list(groups.values()), require_metrics)
    ready, finish = result['memory_ready']['cgroup'], result['memory_finished']['cgroup']
    require(ready['memory.events']['max'] == finish['memory.events']['max'] == 0, 'diagnostic memory.max reclaim occurred')
    return {'status': 'PASS', 'at_utc': datetime.now(timezone.utc).isoformat(),
        'check': 'diagnostic coverage and identity; hash disagreement is recorded separately',
        'directory': str(directory), 'rows': len(groups), 'members': members,
        'hash_mismatch_rows': len(mismatches), 'first_hash_mismatch': mismatches[0] if mismatches else None,
        'first_call_rows': [group for (call, row), group in sorted(groups.items())
                            if call == min(key[0] for key in groups)],
        'mismatch_counts_by_layer': dict(sorted(Counter(row['layer'] for row in mismatches).items())),
        'events_max_ready': ready['memory.events']['max'], 'events_max_finish': finish['memory.events']['max'],
        'events_max_delta': finish['memory.events']['max'] - ready['memory.events']['max'],
        'memory_peak_bytes': finish['memory.peak'], 'hash_mismatches': mismatches,
        'diagnostic_steps': steps, 'output_tokens_per_slot': output_tokens,
        'numeric': numeric, 'row_records': list(groups.values())}



def numeric_rows(groups, required):
    rows = []
    for group in groups:
        returned = group['returned']
        keys = ('local_rel_l2', 'local_max_abs', 'local_l2')
        if not all(key in returned for key in keys):
            require(not required, 'returned row lacks numeric shadow metrics')
            continue
        values = {key: float(returned[key]) for key in keys}
        require(all(math.isfinite(value) and value >= 0 for value in values.values()),
                'returned shadow metric is missing or non-finite')
        require(all(float(group[stage][key]) == -1 for stage in ('input', 'local') for key in keys),
                'input/local metric sentinel differs')
        rows.append({key: returned[key] for key in ('call', 'layer', 'ubatch_row', 'member',
                     'request_id', 'slot_id', 'payload_row', 'position', 'decoded', 'applied', 'step')} | values)
    above = [row for row in rows if row['local_rel_l2'] > 1e-2]
    return {'status': 'FAIL' if above else 'PASS' if rows else 'NOT_RECORDED', 'threshold': 1e-2,
            'max_local_rel_l2': max((row['local_rel_l2'] for row in rows), default=None),
            'worst_row': max(rows, key=lambda row: row['local_rel_l2'], default=None),
            'max_per_call_row': rows, 'rows_above_threshold': above}


def compare_wire_rows(first, second):
    def calls(groups):
        result = {}
        for group in groups:
            row = group['input']
            slots = result.setdefault(row['call'], {})
            require(row['payload_row'] not in slots, 'duplicate payload row')
            slots[row['payload_row']] = group
        return result
    left, right = calls(first), calls(second)
    records = []
    for call in sorted(left.keys() & right.keys()):
        a, b = left[call], right[call]
        same_geometry = (a.keys() == b.keys() and all(
            all(a[row]['input'][key] == b[row]['input'][key]
                for key in ('layer', 'ubatch_row', 'payload_row', 'wire_type')) for row in a))
        same_call_input = same_geometry and all(a[row]['input']['wire_sha256'] ==
                                                b[row]['input']['wire_sha256'] for row in a)
        for row in sorted(a.keys() & b.keys()):
            x, y = a[row], b[row]
            records.append({'call': call, 'payload_row': row, 'layer': x['input']['layer'],
                'same_call_geometry': same_geometry, 'same_call_input': same_call_input,
                'same_row_wire_input': x['input']['wire_sha256'] == y['input']['wire_sha256'],
                'same_row_f32_input': x['input']['f32_sha256'] == y['input']['f32_sha256'],
                'same_returned_wire': x['returned']['wire_sha256'] == y['returned']['wire_sha256'],
                'run1_input': x['input'], 'run2_input': y['input'],
                'run1_returned_wire_sha256': x['returned']['wire_sha256'],
                'run2_returned_wire_sha256': y['returned']['wire_sha256']})
    evidence = [row for row in records if row['same_call_input'] and not row['same_returned_wire']]
    comparable = [row for row in records if row['same_call_input']]
    matched_calls = {row['call'] for row in comparable}
    differing_calls = {row['call'] for row in evidence}
    return {'nondeterminism_observed': bool(evidence),
            'calls_with_identical_inputs': len(matched_calls),
            'matched_calls_with_identical_returns': len(matched_calls - differing_calls),
            'matched_call_return_agreement_fraction': (len(matched_calls - differing_calls) / len(matched_calls)
                                                      if matched_calls else None),
            'common_calls': len(left.keys() & right.keys()),
            'run1_only_calls': sorted(left.keys() - right.keys()),
            'run2_only_calls': sorted(right.keys() - left.keys()),
            'common_rows': len(records), 'rows_with_identical_call_inputs': len(comparable),
            'rows_with_identical_row_inputs': sum(row['same_row_wire_input'] for row in records),
            'differing_returns_with_identical_call_inputs': len(evidence),
            'first_evidence': next(iter(evidence), None), 'row_comparisons': records}



def token_reference(directory, reference):
    def entries(path):
        return sorted((read(p) for p in path.glob('EXECUTION-*.json')), key=lambda row: row['index'])
    actual, baseline = entries(directory), entries(reference)
    require(len(actual) == len(baseline) == 2, 'token reference must contain two requests')
    comparisons = []
    for a, b in zip(actual, baseline):
        require(a['prompt_tokens'] == b['prompt_tokens'], 'token reference prompts differ')
        count = min(len(a['tokens']), len(b['tokens']))
        differences = [i+1 for i in range(count) if a['tokens'][i] != b['tokens'][i]]
        comparisons.append({'request_index': a['index'], 'run_slot': a['slot_id'],
            'reference_slot': b['slot_id'], 'compared_tokens': count,
            'matching_tokens': count-len(differences), 'mismatch_positions': differences,
            'first_run_token': a['tokens'][differences[0]-1] if differences else None,
            'first_reference_token': b['tokens'][differences[0]-1] if differences else None})
    same_runtime = read(directory/'SERVER_IDENTITY.json')['runtime'] == read(reference/'SERVER_IDENTITY.json')['runtime']
    return {'reference': str(reference), 'same_native_runtime': same_runtime,
            'run_submission_order': read(directory/'RESULT.json')['cohort'].get('submission_order'),
            'reference_submission_order': read(reference/'RESULT.json')['cohort'].get('submission_order'),
            'token_comparisons': comparisons,
            'caveat': 'Historical comparisons are descriptive; runtime, slots and prefill history may differ.'}

def determinism(root, steps=64, output_tokens=576, ordered=False):
    directories = [root / name for name in ('run1', 'run2')]
    identities = [read(path/'SERVER_IDENTITY.json') for path in directories]
    results = [read(path/'RESULT.json') for path in directories]
    require(identities[0]['runtime'] == identities[1]['runtime'], 'native runtime changed between repetitions')
    require(all(result['cohort']['submission_order'] == ([0, 1] if ordered else []) for result in results),
            'repetition submission order differs')
    require(results[0]['cohort']['request_fixture_sha256'] == results[1]['cohort']['request_fixture_sha256'],
            'repeated request fixture differs')
    checks = [diagnostic(path, steps, output_tokens, True) for path in directories]
    require(all(check['numeric']['status'] == 'PASS' for check in checks), 'numeric row threshold exceeded')
    composition = []
    for check in checks:
        composition.append(sorted({(group['input']['request_id'], group['input']['slot_id'],
                                   group['input']['applied']) for group in check['row_records']}))
    if ordered:
        require(composition[0] == composition[1], 'ordered repetition slot assignments or acknowledgements differ')
    result = compare_wire_rows(*(check['row_records'] for check in checks))
    result.update({'status': 'PASS', 'at_utc': datetime.now(timezone.utc).isoformat(),
                   'diagnostic_steps': steps, 'output_tokens_per_slot': output_tokens,
                   'composition_by_run': composition, 'ordered': ordered, 'max_local_rel_l2_by_run':
                   [check['numeric']['max_local_rel_l2'] for check in checks]})
    return result

def matrix_case(root, n):
    check = {'status': 'FAIL', 'at_utc': datetime.now(timezone.utc).isoformat(), 'arms': [], 'errors': []}
    try:
        phone, assisted, phone_identity = arm(root, n, 'combined', 64, 1, root/'combined')
        host, baseline, host_identity = arm(root, n, 'control', 64, 1, root/'control')
        check['arms'] = [host, phone]
        require(phone_identity['runtime'] == host_identity['runtime'], 'native runtime differs between paired arms')
        left_result, right_result = read(root/'combined/RESULT.json'), read(root/'control/RESULT.json')
        order = left_result['cohort']['submission_order']
        check['submission_order'] = order
        require(order == right_result['cohort']['submission_order'], 'paired submission order differs')
        require(left_result['cohort']['request_fixture_sha256'] == right_result['cohort']['request_fixture_sha256'],
                'paired request fixture differs')
        for index, (left, right) in enumerate(zip(assisted, baseline)):
            require(left['slot_id'] == right['slot_id'] == n - 1 - order.index(index),
                    'submission order did not produce the expected slot assignment')
        comparisons = compare_tokens(assisted, baseline, root/'combined', root/'control')
        check['token_comparisons'] = comparisons
        require(all(row['accepted'] for row in comparisons), 'slot fails exact or first-mismatch near-tie check')
        check['status'] = 'PASS'
    except (ValueError, OSError, KeyError) as error:
        check['errors'].append(str(error))
    with (root/'CHECK_PAIR.json').open('x') as stream:
        json.dump(check, stream, indent=2)
        stream.write('\n')
    return check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path)
    parser.add_argument('--diagnostic', type=Path)
    parser.add_argument('--matrix-case', type=Path)
    parser.add_argument('--n', type=int, choices=(1, 2, 4, 8))
    parser.add_argument('--phone-only', action='store_true')
    parser.add_argument('--diagnostic-steps', type=int, choices=(5, 64), default=5)
    parser.add_argument('--output-tokens', type=int, choices=(64, 576), default=64)
    parser.add_argument('--require-metrics', action='store_true')
    parser.add_argument('--determinism', type=Path)
    parser.add_argument('--ordered', action='store_true')
    parser.add_argument('--regression', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    if args.regression:
        require(args.reference is not None, 'regression requires a saved reference')
        check = {'status': 'FAIL', 'at_utc': datetime.now(timezone.utc).isoformat(), 'errors': []}
        try:
            measurement, entries, identity = arm(args.regression.parent, 1, 'combined', 64, 1, args.regression)
            saved = [read(path) for path in args.reference.glob('EXECUTION-*.json')]
            require(len(saved) == 1 and len(entries[0]['prompt_tokens']) == 9737, 'pair-v1 regression fixture differs')
            require(entries[0]['prompt_tokens'] == read(args.reference/'REQUEST.json')['prompt_tokens'],
                    'pair-v1 regression prompt differs from saved reference')
            check['measurement'] = measurement
            check['reference'] = str(args.reference)
            check['native_runtime_matches_reference'] = identity['runtime'] == read(args.reference/'SERVER_IDENTITY.json')['runtime']
            differences = [i + 1 for i, (a, b) in enumerate(zip(entries[0]['tokens'], saved[0]['tokens'])) if a != b]
            check['differences'] = differences
            check['matching_tokens'] = 64 - len(differences)
            require(len(saved[0]['tokens']) == 64 and not differences, 'N=1 output changed from saved pair-v1 reference')
            check.update(status='PASS', acceptance='EXACT')
        except (ValueError, OSError, KeyError) as error:
            check['errors'].append(str(error))
        with (args.regression/'CHECK_REGRESSION.json').open('x') as stream:
            json.dump(check, stream, indent=2)
            stream.write('\n')
        print(json.dumps(check, indent=2))
        return 0 if check['status'] == 'PASS' else 1
    if args.determinism:
        check = determinism(args.determinism, args.diagnostic_steps, args.output_tokens, args.ordered)
        with (args.determinism/'CHECK_DETERMINISM.json').open('x') as stream:
            json.dump(check, stream, indent=2)
            stream.write('\n')
        print(json.dumps({k: v for k, v in check.items() if k != 'row_comparisons'}, indent=2))
        return 0
    if args.matrix_case:
        check = matrix_case(args.matrix_case, args.n or 2)
        print(json.dumps(check, indent=2))
        return 0 if check['status'] == 'PASS' else 1
    if args.diagnostic:
        try:
            check = diagnostic(args.diagnostic, args.diagnostic_steps, args.output_tokens, args.require_metrics)
            if args.require_metrics and check['numeric']['status'] != 'PASS':
                check['status'] = 'FAIL'
        except (ValueError, OSError, KeyError) as error:
            check = {'status': 'FAIL', 'error': str(error), 'at_utc': datetime.now(timezone.utc).isoformat()}
        destination = args.diagnostic/'CHECK_DIAGNOSTIC.json' if args.require_metrics else args.diagnostic.parent/'CHECK_STEP1.json'
        with destination.open('x') as stream:
            json.dump(check, stream, indent=2)
            stream.write('\n')
        print(json.dumps({key: value for key, value in check.items() if key not in ('hash_mismatches', 'row_records', 'numeric')}, indent=2))
        return 0 if check['status'] == 'PASS' else 1
    if args.root is None or args.n is None:
        parser.error('--root and --n are required for a pair check')
    output = args.root/f'CHECK_N{args.n}_{"phone" if args.phone_only else "pair"}.json'
    check = {'status': 'FAIL', 'at_utc': datetime.now(timezone.utc).isoformat(), 'arms': [], 'errors': []}
    try:
        phone, assisted, phone_identity = arm(args.root, args.n, 'combined')
        check['arms'].append(phone)
        if not args.phone_only:
            host, baseline, host_identity = arm(args.root, args.n, 'control')
            check['arms'].append(host)
            require(phone_identity['runtime'] == host_identity['runtime'], 'native runtime differs between paired arms')
            phone_directory, host_directory = args.root/f'n{args.n}-combined', args.root/f'n{args.n}-control'
            phone_fixture = read(phone_directory/'RESULT.json')['cohort']
            host_fixture = read(host_directory/'RESULT.json')['cohort']
            require(phone_fixture['request_fixture_sha256'] == host_fixture['request_fixture_sha256'],
                    'paired request fixture differs')
            comparisons = compare_tokens(assisted, baseline, phone_directory, host_directory)
            check['token_comparisons'] = comparisons
            require(all(row['accepted'] for row in comparisons), 'slot fails exact or first-mismatch near-tie check')
        check['status'] = 'PASS'
    except (ValueError, OSError, KeyError) as error:
        check['errors'].append(str(error))
    with output.open('x') as stream:
        json.dump(check, stream, indent=2)
        stream.write('\n')
    print(json.dumps(check, indent=2))
    return 0 if check['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
