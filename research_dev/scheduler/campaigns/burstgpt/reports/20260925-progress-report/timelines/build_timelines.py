#!/usr/bin/env python3
"""Plot eval_v2 token coverage and energy from saved, request-scoped evidence."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import numpy as np  # noqa: E402


HERE = Path(__file__).resolve().parent
INK = '#182d3a'
BLUE = '#2a78d6'
ORANGE = '#eb6834'
GREEN = '#1b9970'
PURPLE = '#8953ad'
GRAY = '#858b93'
MODEL_NAMES = {'qwen': 'Qwen3 14B', 'gemma': 'Gemma 4 12B', 'llama': 'Llama 3.2 1B'}


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def write_csv(name, rows):
    with (HERE / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def policy_devices(policy):
    if policy['baseline']:
        return {}
    return dict(policy.get('device_layer_masks') or [('op15-phone', policy['layer_mask'])])


def token_events(arm):
    groups = {g['request_id']: g for g in arm['groups']}
    times = {}
    for event in arm['token_events']:
        key = (event['request_id'], event['token_index'])
        assert key not in times, ('duplicate token event', key)
        times[key] = event['token_observed_at_us'] / 1e6
    events, validations = [], []
    for request in arm['requests']:
        request_id = request['request_id']
        model = next(k for k in MODEL_NAMES if request['model_id'].startswith(k))
        count = request['output_tokens']
        policies = [{} for _ in range(count)]
        group = groups.get(request_id)
        if group:
            proof = request['physical_execution_proof']
            assert group['grouped_observation_sha256'] == proof['adaptive_grouped_observation_sha256']
            cursor = 1
            for window in group['windows']:
                assert window['token_start'] == cursor
                if window['applied_ack']:
                    assert window['applied_ack']['applied_token_index'] == window['token_start']
                    assert window['applied_ack']['policy_hash'] == window['policy']['policy_hash']
                cursor = window['token_end']
                devices = policy_devices(window['policy'])
                for index in range(window['token_start'], cursor):
                    policies[index] = devices
            assert cursor + group.get('unmeasured_tail_tokens', 0) == count
            for index in range(cursor, count):
                policies[index] = policy_devices(group['final_policy'])
            layer_rows, device_rows = Counter(), Counter()
            for devices in policies:
                for device, mask in devices.items():
                    device_rows[device] += mask.bit_count()
                    for layer in range(mask.bit_length()):
                        if mask & (1 << layer):
                            layer_rows[layer] += 1
            actual_layers = {x['layer']: x['calls'] for x in proof['phone_calls_by_layer']}
            assert dict(layer_rows) == actual_layers, request_id
            actual_devices = Counter()
            for session in proof['phone_calls_by_session']:
                device = 'pixel10pro-phone' if session['session_id'].startswith('PIXEL') else 'op15-phone'
                actual_devices[device] += session['rows']
            assert device_rows == actual_devices, request_id
            validations.append({'request_id': request_id, 'layer_counts_match': True,
                                'device_row_counts_match': True})
        else:
            assert request['physical_execution_proof']['phone_call_count'] == 0
        prior_time = 0
        for index, devices in enumerate(policies):
            when = times.get((request_id, index + 1))
            source = 'decode_boundary'
            if when is None:
                if index == 0 and request.get('first_token_ns'):
                    when = (request['first_token_ns'] - arm['paid_start_ns']) / 1e9
                    source = 'first_token'
                else:
                    when = request['finished_at_us'] / 1e6
                    source = 'request_completion'
            assert prior_time <= when <= (arm['paid_end_ns'] - arm['paid_start_ns']) / 1e9
            prior_time = when
            op15 = int('op15-phone' in devices)
            pixel = int('pixel10pro-phone' in devices)
            assert not pixel or op15
            events.append({'elapsed_s': when, 'request_id': request_id, 'model': model,
                           'output_token_index_zero_based': index,
                           'any_phone': int(bool(devices)), 'op15': op15, 'pixel': pixel,
                           'timestamp_source': source})
    events.sort(key=lambda x: (x['elapsed_s'], x['request_id'], x['output_token_index_zero_based']))
    assert len(events) == sum(r['output_tokens'] for r in arm['requests'])
    return events, validations


def energy_series(arm, grid):
    start, end = arm['paid_start_ns'], arm['paid_end_ns']
    duration = (end - start) / 1e9
    grid = grid[grid <= duration]
    assert grid[0] == 0 and grid[-1] == duration
    samples = arm['samples']
    rapl = sorted((r['rapl_package'] for r in samples), key=lambda x: x['sample_t_ns'])
    rt = np.array([(r['sample_t_ns'] - start) / 1e9 for r in rapl])
    counter = np.array([r['energy_uj'] for r in rapl], dtype=np.int64)
    maximum = rapl[0]['max_energy_range_uj']
    assert all(r['max_energy_range_uj'] == maximum for r in rapl)
    assert np.all(np.diff(rt) > 0) and rt[0] <= 0 and rt[-1] >= duration
    delta = np.diff(counter)
    delta[delta < 0] += maximum
    assert np.all((0 <= delta) & (delta < maximum))
    accumulated = np.r_[0, np.cumsum(delta)]
    cpu_kj = (np.interp(grid, rt, accumulated) - np.interp(0, rt, accumulated)) / 1e9

    gpu = sorted((r['gpu'] for r in samples), key=lambda x: x['sample_t_ns'])
    gt = np.array([(r['sample_t_ns'] - start) / 1e9 for r in gpu])
    watts = np.array([r['power_mw'] / 1000 for r in gpu])
    assert np.all(np.diff(gt) > 0) and gt[0] <= 0 and gt[-1] >= duration
    prefix = np.r_[0, np.cumsum(np.diff(gt) * (watts[:-1] + watts[1:]) / 2)]

    def integral(at):
        index = np.clip(np.searchsorted(gt, at, side='right') - 1, 0, len(gt) - 2)
        dt = at - gt[index]
        slope = (watts[index + 1] - watts[index]) / (gt[index + 1] - gt[index])
        return prefix[index] + watts[index] * dt + slope * dt ** 2 / 2

    gpu_kj = (integral(grid) - integral(0)) / 1000
    host_kj = cpu_kj + gpu_kj
    assert np.all(np.diff(cpu_kj) >= 0) and np.all(np.diff(gpu_kj) >= 0)
    domains = arm['trace_energy']['fleet_energy_uj_by_domain']
    errors_uj = {'cpu': float((cpu_kj[-1] - domains['cpu-package'] / 1e9) * 1e9),
                 'gpu': float((gpu_kj[-1] - domains['gpu-board'] / 1e9) * 1e9)}
    assert max(abs(x) for x in errors_uj.values()) < 1
    return np.column_stack((grid, cpu_kj, gpu_kj, host_kj)), errors_uj


def tidy(ax):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#cdd5db')
    ax.tick_params(length=0, pad=8)
    ax.grid(axis='y', color='#e5eaf0', linewidth=0.8)
    ax.set_axisbelow(True)


def header(fig, title, subtitle):
    fig.text(0.075, 0.947, title, fontsize=20, weight='bold', color=INK)
    fig.text(0.075, 0.912, subtitle, fontsize=10, color='#526575')


def export(fig, stem, pdf):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(HERE / f'{stem}.{ext}', dpi=190, facecolor='white')
    pdf.savefig(fig, facecolor='white')
    plt.close(fig)


def main():
    source = sorted((HERE / 'sources').glob('timeline_evidence_*.json'))[-1]
    data = json.loads(source.read_text())
    audit_path = HERE.parent / 'sources/rig_audit_20260925T163624.json'
    audit = json.loads(audit_path.read_text())
    for key in ('eval_legacy', 'eval_two'):
        result_source = next(s for s in data['sources'] if s['path'] == audit['arms'][key]['path'])
        assert result_source['sha256'] == audit['arms'][key]['result_sha256']
    comparison = audit['comparisons']['eval_two']
    assert comparison['identical_outputs'] == 13 and comparison['requests'] == 14
    baseline, treatment = (data['arms'][k] for k in ('baseline', 'treatment'))
    events, token_validation = token_events(treatment)
    duration = {k: (v['paid_end_ns'] - v['paid_start_ns']) / 1e9 for k, v in data['arms'].items()}
    grid = np.unique(np.r_[np.arange(0, max(duration.values()), 1), list(duration.values())])
    energy, energy_validation = {}, {}
    for name, arm in data['arms'].items():
        energy[name], energy_validation[name] = energy_series(arm, grid)
    curves, summary, coverage_rows = {}, {}, []
    for model in (*MODEL_NAMES, 'all'):
        rows = [r for r in events if model == 'all' or r['model'] == model]
        count = np.arange(1, len(rows) + 1)
        any_phone = np.cumsum([r['any_phone'] for r in rows])
        op15 = np.cumsum([r['op15'] for r in rows])
        pixel = np.cumsum([r['pixel'] for r in rows])
        curves[model] = np.column_stack(([r['elapsed_s'] for r in rows], 100 * any_phone / count,
                                        100 * pixel / count))
        summary[model] = {'output_tokens': len(rows), 'any_phone_tokens': int(any_phone[-1]),
                          'op15_tokens': int(op15[-1]), 'pixel_tokens': int(pixel[-1]),
                          'any_phone_pct': float(100 * any_phone[-1] / len(rows)),
                          'pixel_pct': float(100 * pixel[-1] / len(rows))}
        for i, row in enumerate(rows):
            coverage_rows.append({'model': model, 'elapsed_s': row['elapsed_s'],
                                  'generated_tokens': int(count[i]), 'any_phone_tokens': int(any_phone[i]),
                                  'op15_tokens': int(op15[i]), 'pixel_tokens': int(pixel[i]),
                                  'any_phone_pct': float(100 * any_phone[i] / count[i]),
                                  'pixel_pct': float(100 * pixel[i] / count[i])})
    b, t = energy['baseline'], energy['treatment']
    bgrid = np.interp(grid, b[:, 0], b[:, 3])
    tgrid = np.interp(grid, t[:, 0], t[:, 3])
    gap = bgrid - tgrid
    saving = 100 * (1 - t[-1, 3] / b[-1, 3])
    saving_rows = [{'elapsed_s': float(at), 'baseline_kj': float(bgrid[i]),
                    'treatment_kj': float(tgrid[i]), 'gap_kj': float(gap[i]),
                    'gap_pct': float(100 * gap[i] / bgrid[i]) if bgrid[i] else '',
                    'treatment_finished': bool(at > duration['treatment'])}
                   for i, at in enumerate(grid)]
    fallback = sum(r['timestamp_source'] == 'request_completion' for r in events)
    stats = {'trace': 'longtail_eval_v2', 'coverage': summary, 'duration_s': duration,
             'host_energy_kj': {k: float(v[-1, 3]) for k, v in energy.items()},
             'host_saving_kj': float(gap[-1]), 'host_saving_pct': float(saving),
             'token_timestamp_sources': dict(Counter(r['timestamp_source'] for r in events)),
             'source': str(source.relative_to(HERE)),
             'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'matched_audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
             'source_files': data['sources']}
    write_json('TIMELINE_SUMMARY.json', stats)
    write_json('VALIDATION.json', {'status': 'PASS', 'coverage_checks': token_validation,
                                 'energy_endpoint_error_microjoules': energy_validation,
                                 'complete_token_denominator': len(events),
                                 'historical_groups_excluded': treatment['ignored_historical_groups'],
                                 'result_hashes_match_exact_token_audit': True,
                                 'policy_acknowledgements_match_token_boundaries': True,
                                 'strict_output_identity': 'FAIL: 13/14 (from matched result audit)'})
    write_csv('token_events.csv', events)
    write_csv('token_coverage_timeline.csv', coverage_rows)
    write_csv('energy_savings_timeline.csv', saving_rows)
    write_csv('host_energy_timeline.csv', [
        {'arm': name, 'elapsed_s': float(row[0]), 'cpu_kj': float(row[1]),
         'gpu_kj': float(row[2]), 'host_kj': float(row[3])}
        for name, points in energy.items() for row in points])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})

    with PdfPages(HERE / 'eval_v2_timelines.pdf') as pdf:
        fig = plt.figure(figsize=(12.8, 8.5))
        header(fig, 'Phone-assisted token coverage over time',
               'longtail_eval_v2 | OP15 + Pixel | 14 requests | 3,604 output tokens | 25 September 2026')
        fig.text(0.075, 0.851, f"{summary['all']['any_phone_pct']:.1f}% overall coverage", fontsize=22,
                 weight='bold', color=BLUE)
        fig.text(0.485, 0.855, '2,985 / 3,604 tokens received FFN assistance', fontsize=12)
        ax = fig.add_axes((0.085, 0.31, 0.88, 0.47))
        specs = [('qwen', 1, BLUE, '-', 'Qwen: any phone / OP15'),
                 ('gemma', 1, ORANGE, '-', 'Gemma: OP15'),
                 ('qwen', 2, PURPLE, '--', 'Qwen: Pixel participation'),
                 ('llama', 1, GREEN, '-', 'Llama: any phone')]
        for model, col, color, style, label in specs:
            points = curves[model]
            x = np.r_[points[:, 0], duration['treatment']] / 60
            y = np.r_[points[:, col], points[-1, col]]
            ax.step(x, y, where='post', color=color, linestyle=style, linewidth=2.15,
                    label=f'{label} ({y[-1]:.1f}%)')
            ax.scatter([x[-1]], [y[-1]], color=color, s=22, zorder=4)
        ax.set(xlim=(0, 31.5), ylim=(-3, 104), ylabel='Cumulative assisted / generated tokens (%)',
               xlabel='Elapsed treatment time (minutes)', yticks=np.arange(0, 101, 20))
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.015), ncol=2, frameon=False,
                  fontsize=9.5, borderaxespad=0)
        tidy(ax)
        fig.text(0.085, 0.217, 'Qwen', fontsize=12, weight='bold', color=BLUE)
        fig.text(0.085, 0.198, '1,298 / 1,524 assisted\n1,196 also use Pixel',
                 fontsize=11, linespacing=1.6, va='top')
        fig.text(0.40, 0.217, 'Gemma', fontsize=12, weight='bold', color=ORANGE)
        fig.text(0.40, 0.198, '1,687 / 2,034 assisted\nOP15 only', fontsize=11, linespacing=1.6, va='top')
        fig.text(0.70, 0.217, 'Llama', fontsize=12, weight='bold', color=GREEN)
        fig.text(0.70, 0.198, '0 / 46 assisted\nDesktop only', fontsize=11, linespacing=1.6, va='top')
        fig.text(0.075, 0.115,
                 'Selected FFN layers are offloaded. A token using both phones is counted once; Pixel is a subset of Qwen coverage.\n'
                 f'All output tokens, including first tokens from prefill, are counted. {fallback} tokens lack individual timestamps and are placed at completion.\n'
                 'Counts match request-scoped physical call proofs. Exact-output check: 13/14 sequences; strict identity FAIL.',
                 fontsize=9, color='#526575', linespacing=1.7, va='top')
        export(fig, 'token_coverage_timeline', pdf)

        fig = plt.figure(figsize=(12.8, 8.8))
        header(fig, 'Host energy and savings over time',
               'longtail_eval_v2 | Legacy desktop vs OP15 + Pixel full configuration | Measured CPU package + GPU board')
        fig.text(0.075, 0.848, f'{saving:.1f}% less host energy', fontsize=23, weight='bold', color=GREEN)
        fig.text(0.51, 0.853, f'{gap[-1]:.1f} kJ saved | 18.3% shorter trace', fontsize=13)
        ax = fig.add_axes((0.085, 0.46, 0.88, 0.32))
        ax.plot(b[:, 0] / 60, b[:, 3], color=GRAY, linewidth=2.3, label='Legacy desktop')
        ax.plot(t[:, 0] / 60, t[:, 3], color=BLUE, linewidth=2.3, label='OP15 + Pixel')
        ax.plot([duration['treatment'] / 60, duration['baseline'] / 60], [t[-1, 3]] * 2,
                color=BLUE, linewidth=1.7, linestyle='--', label='Completed arm: final total held')
        ax.fill_between(grid / 60, bgrid, tgrid, where=gap >= 0, color=GREEN, alpha=0.13)
        ax.fill_between(grid / 60, bgrid, tgrid, where=gap < 0, color=ORANGE, alpha=0.13)
        ax.scatter([b[-1, 0] / 60, t[-1, 0] / 60], [b[-1, 3], t[-1, 3]],
                   color=[GRAY, BLUE], s=35, zorder=5)
        ax.annotate(f'{b[-1, 3]:.1f} kJ / {b[-1, 0] / 60:.1f} min', (b[-1, 0] / 60, b[-1, 3]),
                    xytext=(-5, 12), textcoords='offset points', ha='right', weight='bold')
        ax.annotate(f'{t[-1, 3]:.1f} kJ / {t[-1, 0] / 60:.1f} min', (t[-1, 0] / 60, t[-1, 3]),
                    xytext=(-9, -25), textcoords='offset points', ha='right', color=BLUE, weight='bold')
        ax.set(xlim=(0, 39), ylim=(0, 259), ylabel='Cumulative host energy (kJ)')
        ax.legend(loc='upper left', frameon=False, fontsize=9.5)
        tidy(ax)
        ax.tick_params(labelbottom=False)
        gap_ax = fig.add_axes((0.085, 0.218, 0.88, 0.185), sharex=ax)
        split = grid <= duration['treatment']
        gap_ax.plot(grid[split] / 60, gap[split], color=GREEN, linewidth=2.1)
        after = grid >= duration['treatment']
        gap_ax.plot(grid[after] / 60, gap[after], color=GREEN, linewidth=2.1, linestyle='--')
        gap_ax.fill_between(grid / 60, 0, gap, where=gap >= 0, color=GREEN, alpha=0.12)
        gap_ax.fill_between(grid / 60, 0, gap, where=gap < 0, color=ORANGE, alpha=0.16)
        gap_ax.axhline(0, color='#b7c0c8', linewidth=0.8)
        gap_ax.annotate(f'{gap[-1]:.1f} kJ saved ({saving:.1f}%)', (grid[-1] / 60, gap[-1]),
                        xytext=(-6, 10), textcoords='offset points', ha='right', color=GREEN, weight='bold')
        gap_ax.set(ylabel='Energy saved (kJ)', xlabel='Elapsed time from each arm\'s paid start (minutes)',
                   ylim=(min(-5, float(gap.min()) - 3), 153), xticks=np.arange(0, 40, 5))
        tidy(gap_ax)
        for panel in (ax, gap_ax):
            panel.axvline(duration['treatment'] / 60, color=BLUE, linewidth=0.8, alpha=0.45, linestyle=':')
        fig.text(0.075, 0.104,
                 'The time-aligned gap compares equal elapsed time, not equal completed work. Final totals compare the complete 14-request trace.\n'
                 'After the assisted arm finishes, its paid energy total is held flat; no idle energy is extrapolated. Phone energy is excluded.\n'
                 'Single matched pair; exact-output check 13/14 (strict identity FAIL). The Pixel-only contribution is not isolated by this comparison.',
                 fontsize=9, color='#526575', linespacing=1.7)
        export(fig, 'energy_savings_timeline', pdf)
    print(json.dumps({'coverage': summary, 'host_saving_pct': saving,
                      'completion_timestamp_fallback_tokens': fallback,
                      'energy_endpoint_error_microjoules': energy_validation}, indent=2))


if __name__ == '__main__':
    main()
