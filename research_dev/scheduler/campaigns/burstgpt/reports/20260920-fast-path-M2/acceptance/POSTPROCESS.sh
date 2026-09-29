#!/usr/bin/env bash
set -euo pipefail
physical=${1:-/mnt/storage/s42-fast-path-M2-acceptance-20260921-c09ea6/physical}
python3 - "$physical" <<'PY'
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
read = lambda path: json.loads(path.read_text())
pairs = [read(root / f'CHECK_N{n}_pair.json') for n in (1, 2, 4, 8)]
regression = read(root / 'regression/CHECK_REGRESSION.json')
determinism = read(root / 'determinism/CHECK_DETERMINISM.json')
cleanup = read(root / 'CLEANUP.json')
curve = []
for n, pair in zip((1, 2, 4, 8), pairs):
    phone = next(row for row in pair['arms'] if row['arm'] == 'combined')
    host = next(row for row in pair['arms'] if row['arm'] == 'control')
    timing = phone['phone_timing']
    curve.append({'n': n,
        'phone_compute_ms_per_token_per_slot': timing['phone_compute_ms_per_token_per_slot'],
        'phone_rpc_ms_per_token_per_slot': timing['phone_rpc_ms_per_token_per_slot'],
        'host_only_decode_w': host['decode_host_w'], 'phone_arm_host_decode_w': phone['decode_host_w'],
        'minimum_full_cohort_steps': min(timing['full_batch_steps_by_layer'].values())})
changes = []
for left, right in zip(curve, curve[1:]):
    changes.append({'from_n': left['n'], 'to_n': right['n'],
        'compute_falls': right['phone_compute_ms_per_token_per_slot'] < left['phone_compute_ms_per_token_per_slot'],
        'rpc_falls': right['phone_rpc_ms_per_token_per_slot'] < left['phone_rpc_ms_per_token_per_slot'],
        'compute_change_percent': 100 * (right['phone_compute_ms_per_token_per_slot'] / left['phone_compute_ms_per_token_per_slot'] - 1),
        'rpc_change_percent': 100 * (right['phone_rpc_ms_per_token_per_slot'] / left['phone_rpc_ms_per_token_per_slot'] - 1)})
checks = {
    'ordered_determinism_composition': determinism['status'] == 'PASS' and determinism['ordered'],
    'n1_saved_output_regression': regression['status'] == 'PASS' and regression['acceptance'] == 'EXACT',
    'long_pairs_correctness_and_no_hang': all(pair['status'] == 'PASS' for pair in pairs),
    'at_least_512_full_cohort_steps': all(row['minimum_full_cohort_steps'] >= 512 for row in curve),
    'phone_compute_time_falls_with_n': all(row['compute_falls'] for row in changes),
    'phone_rpc_time_falls_with_n': all(row['rpc_falls'] for row in changes),
    'cleanup': cleanup['status'] == 'PASS',
}
result = {'status': 'PASS' if all(checks.values()) else 'FAIL',
    'at_utc': datetime.now(timezone.utc).isoformat(), 'single_pair_per_n': True,
    'checks': checks, 'curve': curve, 'adjacent_changes': changes,
    'matched_call_return_agreement_fraction': determinism['matched_call_return_agreement_fraction'],
    'acceptance_rule': 'First mismatch per slot: host top1-top2 <= 0.05 and raw-logit NMSE <= 5e-4; later differences are after context divergence.',
    'decision': 'Stop before M3; no rerun or implementation change without the user decision.'}
with (root / 'CHECK_M2.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')

with (root / 'UTILIZATION_CURVE.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
    writer.writeheader()
    for row in curve:
        writer.writerow({key: f'{value:.3f}' if isinstance(value, float) else value for key, value in row.items()})

lines = []
def table(headers, rows):
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('| ' + ' | '.join('---' for _ in headers) + ' |')
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
    lines.append('')

lines.append('## Utilization and power: FAIL\n')
table(['N', 'Phone compute ms/token/slot', 'Phone RPC ms/token/slot', 'Host-only decode W, measured', 'Phone-arm host decode W, measured', 'Phone W, assumed', 'Full-cohort steps/layer'],
      [[r['n'], f"{r['phone_compute_ms_per_token_per_slot']:.3f}", f"{r['phone_rpc_ms_per_token_per_slot']:.3f}", f"{r['host_only_decode_w']:.3f}", f"{r['phone_arm_host_decode_w']:.3f}", '4.5 active; 0.875 idle', r['minimum_full_cohort_steps']] for r in curve])
rise = changes[-1]
lines.append(f"N=4 to N=8 compute rises {rise['compute_change_percent']:.3f}%; RPC rises {rise['rpc_change_percent']:.3f}%. The plan requires phone time per token per slot to fall with N, so M2 fails. No cause is established by these single pairs.\n")
lines.append('The denominator is full-cohort physical calls * N / 18 owned layers. Partial start/drain calls remain in the records and are excluded from these fixed-N points.\n')

lines.append('## Per-slot correctness: PASS under the amended rule\n')
correctness = []
for n, pair in zip((1, 2, 4, 8), pairs):
    for row in pair['token_comparisons']:
        correctness.append([n, row['request_index'], row['phone_slot'], row['prompt_tokens'], row['acceptance'], f"{row['matching_tokens']}/{row['tokens']}", row['first_mismatch'] or '-'])
table(['N', 'Request', 'Slot, both arms', 'Prompt tokens', 'Acceptance', 'Matching positions', 'First mismatch'], correctness)
for n, pair in zip((1, 2, 4, 8), pairs):
    for row in pair['token_comparisons']:
        if not row['mismatches']:
            continue
        first = row['mismatches'][0]
        table(['N', 'Request / slot', 'Step', 'Host / phone token', 'Host top-1 logit', 'Host top-2 logit', 'Margin', 'NMSE', 'Later differing positions'],
              [[n, f"{row['request_index']} / {row['phone_slot']}", first['step'], f"{first['host_token']} / {first['phone_token']}", f"{first['host_top1_logit']:.3f}", f"{first['host_top2_logit']:.3f}", f"{first['margin']:.3e}", f"{first['nmse']:.3e}", len(row['mismatches']) - 1]])
lines.append('Both displayed N=4 logits round to the same value; the margin above uses the unrounded raw values. Every mismatch retains its step, both host logits, margin and NMSE in CHECK_N4_pair.json. Only the first mismatch decides acceptance; 434 later differences are labeled after_context_divergence.\n')

lines.append('## Single-pair energy and latency\n')
energy = []
for n, pair in zip((1, 2, 4, 8), pairs):
    for name in ('control', 'combined'):
        row = next(a for a in pair['arms'] if a['arm'] == name)
        energy.append([n, name, f"{row['request_host_j']/1000:.3f}", ' / '.join(f'{value:.3f}' for value in row['decode_ms_per_token_by_slot']), '0.875 idle' if name == 'control' else '4.5 active; 0.875 idle', f"{row['assumed_phone_request_j']/1000:.3f}"])
table(['N', 'Arm', 'Request host kJ, measured', 'Decode ms/token by request index', 'Phone W, assumed', 'Request phone kJ, assumed'], energy)
lines.append('Host energy is RAPL package plus NVML board, counted once per concurrent request group. Phone energy is separate and assumed over the union of active decode intervals plus idle time. Decode power uses the common decode interval; per-request ms/token is the server predicted_ms divided by output tokens. Both long arms write raw logits; these timings include that instrumentation. The phone arm has lower host power at every N but is slower end to end at N=4 and N=8.\n')

lines.append('## Memory and validation\n')
memory = []
for label in ('determinism/run1', 'determinism/run2', 'regression', 'n1-control', 'n1-combined', 'n2-control', 'n2-combined', 'n4-control', 'n4-combined', 'n8-control', 'n8-combined'):
    arm = read(root / label / 'RESULT.json')
    ready = arm['memory_ready']['cgroup']
    finish = arm['memory_finished']['cgroup']
    memory.append([label, f"{finish['memory.peak']/1024**3:.3f}", ready['memory.events']['max'], finish['memory.events']['max'], finish['memory.events']['oom_kill']])
table(['Arm', 'memory.peak GiB', 'memory.events.max ready', 'finish', 'oom_kill finish'], memory)
lines.append('All 11 arms use fresh MemoryMax=infinity, MemorySwapMax=0 scopes. All 100 rig unittests pass and pyflakes is clean before each arm. Cleanup passed for all 11 scopes and owned server PIDs; all seven phone closes report terminal status 0 and RESTORED. The rig lock is free, the GPU has no compute process, and OP15 is visible on ADB 5037. No other phone was used.\n')
(root / 'RESULT_TABLES.md').write_text('\n'.join(lines))
print(json.dumps({'status': result['status'], 'checks': checks, 'n4_to_n8_compute_rise_percent': round(rise['compute_change_percent'], 3)}, indent=2))
raise SystemExit(0 if result['status'] == 'PASS' else 1)
PY
