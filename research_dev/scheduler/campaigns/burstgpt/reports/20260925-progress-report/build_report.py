#!/usr/bin/env python3
"""Build the report and export figures from saved, auditable measurements."""

import csv
import hashlib
import json
from pathlib import Path
import statistics
import textwrap

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import numpy as np  # noqa: E402


HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent
CPU = '#236B8E'
GPU = '#83C9C1'
INK = '#182D3A'
GREY = '#9CAAB5'
TEAL = '#008577'
AMBER = '#98661B'


def read(path):
    return json.loads(path.read_text())


def csv_write(name, rows):
    with (HERE / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tidy(ax):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#CBD5DC')
    ax.tick_params(axis='both', length=0, pad=9)
    ax.grid(axis='y', color='#E5EAEF', linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name, pdf):
    fig.savefig(HERE / f'{name}.png', dpi=200, facecolor='white')
    fig.savefig(HERE / f'{name}.svg', facecolor='white')
    fig.savefig(HERE / f'{name}.pdf', facecolor='white')
    pdf.savefig(fig, facecolor='white')
    plt.close(fig)


def main():
    audit_path = sorted((HERE / 'sources').glob('rig_audit_*.json'))[-1]
    audit = read(audit_path)
    arms = audit['arms']
    compared = audit['comparisons']
    for arm in arms.values():
        if arm['status'] in ('PASS', 'FAIL'):
            continue
        events = [e for e in audit.get('followup_chain', [])
                  if e.get('inputs') and arm['path'].startswith(e['inputs'] + '/')]
        if events and events[-1].get('stage') == 'preflight':
            arm['status'] = 'PREFLIGHT'
    stamp = audit['capture_finished_at_utc'][:19].replace('T', ' ') + ' UTC'
    analysis_dir = REPORTS / '20260925-two-phone-eval/analysis'
    tcp_path = REPORTS / '20260925-pixel-cpu-gpu/physical/tcp1/TCP_TABLE.json'
    tcp = read(tcp_path)
    identity_path = REPORTS / '20260925-two-phone-eval/physical/server-identity-r1/RESULT.json'
    identity = read(identity_path)

    labels = {
        'lt_legacy': ('longtail_v1', 'Legacy desktop'),
        'lt_dispatcher': ('longtail_v1', 'Desktop + dispatcher'),
        'dev_desktop': ('longtail_dev_v2', 'Desktop + dispatcher'),
        'dev_op15_r1': ('longtail_dev_v2', 'OP15 r1'),
        'dev_op15_r2': ('longtail_dev_v2', 'OP15 r2'),
        'dev_two_r1': ('longtail_dev_v2', 'OP15 + Pixel r1'),
        'dev_two_r2': ('longtail_dev_v2', 'OP15 + Pixel r2'),
        'eval_legacy': ('longtail_eval_v2', 'Legacy desktop'),
        'eval_two': ('longtail_eval_v2', 'OP15 + Pixel'),
        'eval_dispatcher': ('longtail_eval_v2', 'Desktop + dispatcher'),
        'eval_op15': ('longtail_eval_v2', 'OP15'),
    }
    table = []
    for key, (trace, label) in labels.items():
        a = arms[key]
        c = compared.get(key, {})
        row = {'trace': trace, 'arm': label, 'status': a['status'],
               'requests': a.get('counts', {}).get('requests', ''),
               'cpu_kj': a.get('cpu_kj', ''), 'gpu_kj': a.get('gpu_kj', ''),
               'host_kj': a.get('host_kj', ''), 'duration_s': a.get('duration_s', ''),
               'host_saving_pct': c.get('host_saving_pct', ''),
               'duration_reduction_pct': c.get('duration_reduction_pct', ''),
               'identical_outputs': c.get('identical_outputs', ''),
               'strict_tokens': c.get('strict_tokens', 'reference' if a['status'] == 'PASS' else 'pending'),
               'op15_calls': a.get('phone_calls', {}).get('op15', 0) if a['status'] == 'PASS' else '',
               'pixel_calls': a.get('phone_calls', {}).get('pixel', 0) if a['status'] == 'PASS' else '',
               'result_path': a['path']}
        table.append(row)
        if a['status'] == 'PASS':
            assert abs(a['host_kj'] - a['cpu_kj'] - a['gpu_kj']) < 1e-8
        if c:
            assert c['same_source_files'] and c['same_native_binaries']
            assert c['same_request_ids'] and not c['input_differences']
    csv_write('trace_energy.csv', table)

    latency = []
    for batch in (1, 2, 4):
        before = [r for r in tcp if r['rows'] == batch and r['arm'] in ('a-prod', 'd-prod')]
        after = next(r for r in tcp if r['rows'] == batch and r['arm'] == 'c-boost-batch')
        baseline = statistics.mean(r['rpc_p50'] for r in before)
        assert after['identical_to_production'] and after['stop_exit'] == 0
        latency.append({
            'batch_rows': batch,
            'before_rpc_ms': baseline, 'after_rpc_ms': after['rpc_p50'],
            'speedup': baseline / after['rpc_p50'],
            'latency_reduction_pct': 100 * (1 - after['rpc_p50'] / baseline),
            'before_compute_ms': statistics.mean(r['compute_p50'] for r in before),
            'after_compute_ms': after['compute_p50'],
            'before_control_min_ms': min(r['rpc_p50'] for r in before),
            'before_control_max_ms': max(r['rpc_p50'] for r in before),
            'calls_per_arm': 72,
        })
    csv_write('pixel_latency.csv', latency)

    op15_mean = statistics.mean(arms[k]['host_kj'] for k in ('dev_op15_r1', 'dev_op15_r2'))
    two_mean = statistics.mean(arms[k]['host_kj'] for k in ('dev_two_r1', 'dev_two_r2'))
    baseline_dev = arms['dev_desktop']['host_kj']
    means = [
        {'configuration': 'OP15', 'repeats': 2, 'mean_host_kj': op15_mean,
         'saving_vs_desktop_pct': 100 * (1 - op15_mean / baseline_dev)},
        {'configuration': 'OP15 + Pixel', 'repeats': 2, 'mean_host_kj': two_mean,
         'saving_vs_desktop_pct': 100 * (1 - two_mean / baseline_dev)},
    ]
    csv_write('repeat_means.csv', means)
    incremental = None
    one, two = arms['eval_op15'], arms['eval_two']
    if one['status'] == two['status'] == 'PASS':
        assert one['source_file_map_sha256'] == two['source_file_map_sha256']
        assert one['execution_identity']['binaries'] == two['execution_identity']['binaries']
        incremental = {
            'host_saving_kj': one['host_kj'] - two['host_kj'],
            'host_saving_pct': 100 * (1 - two['host_kj'] / one['host_kj']),
            'duration_increase_s': two['duration_s'] - one['duration_s'],
            'duration_increase_pct': 100 * (two['duration_s'] / one['duration_s'] - 1),
            'identical_outputs': sum(one['requests'][k]['tokens'] == two['requests'][k]['tokens']
                                     for k in one['requests']),
        }
    source_paths = [audit_path, tcp_path, identity_path,
                    analysis_dir / 'DEV2_ANALYSIS.json', analysis_dir / 'LT_ALLON_STYLE.json']
    data = {'as_of_utc': stamp, 'trace_energy': table, 'pixel_latency': latency,
            'repeat_means': means, 'two_phone_vs_op15_mean_change_pct': 100 * (two_mean / op15_mean - 1),
            'eval_two_vs_op15': incremental,
            'provenance': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p in source_paths]}
    (HERE / 'REPORT_DATA.json').write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')

    def energy_rows(keys):
        lines = []
        for key in keys:
            a = arms[key]
            name = labels[key][1]
            if a['status'] != 'PASS':
                lines.append(f'| {name} | {a["status"]} | - | - | - | - | - | Pending |')
                continue
            c = compared.get(key)
            saving = f'{c["host_saving_pct"]:.1f}%' if c else 'Reference'
            exact = f'{c["identical_outputs"]}/{c["requests"]} (FAIL)' if c and c['strict_tokens'] == 'FAIL' else (
                f'{c["identical_outputs"]}/{c["requests"]} (PASS)' if c else 'Reference')
            lines.append(f'| {name} | PASS | {a["cpu_kj"]:.2f} | {a["gpu_kj"]:.2f} | '
                         f'{a["host_kj"]:.2f} | {a["duration_s"]:.1f} | {saving} | {exact} |')
        return '\n'.join(lines)

    header = ('| Arm | Completed | CPU kJ | GPU kJ | Host kJ | Duration s | Host saving | Exact sequences |\n'
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n')
    live = arms['eval_two']
    live_streams = live.get('live_streams', {})
    stopped = sum(s['stop'] for s in live_streams.values())
    pixel_shapes = [s for s in live.get('shape_summaries', []) if s.get('helper') == 'pixel10pro']
    live_note = (f'Two-phone preflight PASS at 16:03:01 UTC. At this snapshot, '
                 f'{stopped}/14 streams have a final stop record. '
                 f'{len(pixel_shapes)} saved native summaries identify real Pixel calls. '
                 'No RESULT.json or FAILURE.json exists yet; final energy, completion and output identity are pending. '
                 'Stream completion is not terminal scheduler proof acceptance.')
    if live['status'] == 'PASS':
        c = compared['eval_two']
        live_note = (f'Two-phone completion PASS, {live["host_kj"]:.3f} kJ in {live["duration_s"]:.1f} s; '
                     f'{c["host_saving_pct"]:.2f}% host saving. Exact output check {c["strict_tokens"]}: '
                     f'{c["identical_outputs"]}/{c["requests"]}.')
    elif live['status'] == 'FAIL':
        live_note = 'Two-phone execution FAIL. No completed-arm saving is available; see the saved failure in the rig audit.'
    latest_summary = ''
    latest_figure_link = ''
    current_details = ''
    if live['status'] == 'PASS':
        latest_summary = ('**New completed eval_v2 pair:** ' + live_note +
                          ' This compares the full configuration against legacy desktop; '
                          'the incremental Pixel effect still needs the same-trace OP15 control.\n')
        if incremental:
            latest_summary = ('**Three completed eval_v2 arms:** Legacy desktop '
                              f'{arms["eval_legacy"]["host_kj"]:.3f} kJ; OP15 {one["host_kj"]:.3f} kJ '
                              f'(-{compared["eval_op15"]["host_saving_pct"]:.2f}%); '
                              f'OP15+Pixel {two["host_kj"]:.3f} kJ '
                              f'(-{compared["eval_two"]["host_saving_pct"]:.2f}%). '
                              f'Two phones use {incremental["host_saving_pct"]:.2f}% less host energy '
                              'than OP15 in this single comparison. Exact outputs vs legacy: '
                              'OP15 12/14, two phones 13/14; both strict checks FAIL.\n')
        latest_figure_link = '- [Today\'s eval_v2 comparison](eval_v2_comparison.png), [vector SVG](eval_v2_comparison.svg).\n'
        c = compared['eval_two']
        base = arms['eval_legacy']
        current_details = (
            f'**Observed full-configuration saving: {base["host_kj"] - live["host_kj"]:.3f} kJ '
            f'({c["host_saving_pct"]:.3f}%).** Duration falls by '
            f'{base["duration_s"] - live["duration_s"]:.3f} s ({c["duration_reduction_pct"]:.3f}%). '
            f'Average host power is {base["host_kj"] * 1000 / base["duration_s"]:.1f} -> '
            f'{live["host_kj"] * 1000 / live["duration_s"]:.1f} W.\n\n'
            f'Phone-execution proof PASS: OP15 {live["phone_calls"].get("op15", 0):,} calls; '
            f'Pixel {live["phone_calls"].get("pixel", 0):,} calls. '
            'All 7 Qwen and all 6 Gemma requests are assisted. '
            'No request is rejected; all 3,604 requested output tokens are present.\n\n'
        )
        if c.get('output_differences'):
            current_details += 'Strict identity FAIL; differences:\n\n' + '\n'.join(
                f'- `{d["request"]}`: first difference at zero-based output token '
                f'{d["first_differing_token_zero_based"]}.' for d in c['output_differences']) + '\n\n'
            current_details += 'No first-divergence logits or quality evaluation establish an acceptance exception.\n\n'
        current_details += (
            f'Treatment phone-energy estimates are **{live["phone_kj_assumed"].get("phone-system", 0):.3f} kJ OP15** '
            f'and **{live["phone_kj_assumed"].get("pixel10pro-phone-system", 0):.3f} kJ Pixel**. '
            'They are assumed-power bookkeeping, excluded from the measured host saving.\n'
        )
        if incremental:
            current_details += (
                f'\n**New OP15-only control:** completion PASS 14/14, {one["host_kj"]:.3f} kJ / '
                f'{one["duration_s"]:.3f} s. Relative to OP15, two phones save '
                f'**{incremental["host_saving_kj"]:.3f} kJ ({incremental["host_saving_pct"]:.3f}%)**, '
                f'while taking {incremental["duration_increase_s"]:.3f} s '
                f'({incremental["duration_increase_pct"]:.3f}%) longer. '
                'Request inputs, source-file hashes and native binaries match across all three arms. '
                'This is an observed single-run configuration difference; repeatability is unverified. '
                f'OP15 vs two-phone exact sequences: {incremental["identical_outputs"]}/14 (FAIL). '
                'OP15 differs from legacy at request 003 token 203 and request 004 token 233 '
                '(zero-based).\n'
            )
    chain_stops = [e for e in audit['chain'] if e.get('stage') == 'exception']
    chain_note = ''
    if chain_stops:
        event = chain_stops[-1]
        chain_note = (f'The outer chain stopped at **{event["at"][:19].replace("T", " ")} UTC** with '
                      f'`{event["error"]}`. The two-phone arm had already reported PASS / exit 0. '
                      'The saved cleanup check found zero Pixel workers and zero ADB forwards. '
                      'The remaining controls were subsequently reordered under CHAIN-ev3. '
                      'This reporting session did not request cancellation.\n')
    if incremental:
        chain_note += ('\nCHAIN-ev3 completed OP15-only at **17:13:24 UTC**, exit 0 and clean worker/forward checks. '
                       f'Desktop+dispatcher is **{arms["eval_dispatcher"]["status"]}** at this snapshot.\n')

    controls = [r for r in identity['requests'] if not r['columns']]
    pixel_full = next(r for r in identity['requests'] if r['columns'] == 17408)
    control_j = statistics.mean(r['request_host_energy']['server_compute_device_energy_j'] for r in controls)
    pixel_j = pixel_full['request_host_energy']['server_compute_device_energy_j']
    pixel_save = 100 * (1 - pixel_j / control_j)
    latency_md = '\n'.join(
        f'| {r["batch_rows"]} | {r["before_rpc_ms"]:.2f} | {r["after_rpc_ms"]:.2f} | '
        f'{r["speedup"]:.2f}x | {r["latency_reduction_pct"]:.1f}% |'
        for r in latency)
    evidence_note = (
        'Host energy is measured CPU-package RAPL plus GPU-board NVML over the paid trace interval. '
        'It is not wall-plug energy and excludes phone energy. Phone power is only modeled '
        '(4.5 W active, 0.875 W idle), so no measured whole-system saving is claimed. '
        'Compare percentages only within a trace and its stated baseline. '
        'The two longtail desktop arms and the dev_v2 baseline are single measurements; '
        'there are two repeats of each dev_v2 phone configuration. No confidence interval is established.')
    report = f'''# Scheduler and phone-offload progress - 2026-09-25

Read-only result audit: **{stamp}**. No campaign, deployment, phone state or existing worker was changed.

## What can be reported today

{latest_summary}
- **Measured host savings exceed 25%:** dispatcher-only longtail_v1 saves **26.7%**, and the recent
  dev_v2 phone arms save **34.6-46.1% (OP15)** and **24.4-44.3% (OP15+Pixel)** against their matched desktop controls.
- **Strict exact-output acceptance FAILS on those completed comparisons:** 14/31 for the longtail
  desktop comparison, 5/9 for both OP15 repeats, and 4/9 then 5/9 for OP15+Pixel.
  All completed arms retain the requested output lengths and matched inputs. These are measured
  workload-energy results, not a new all-token-identical acceptance result or a quality-equivalence proof.
- **Short-trace repeats did not show a consistent Pixel benefit.** Across the two dev_v2 repeats,
  OP15 averages **{op15_mean:.2f} kJ** and OP15+Pixel **{two_mean:.2f} kJ**:
  two phones use **{100 * (two_mean / op15_mean - 1):.1f}% more** host energy on the mean.
  The pairwise changes are +40.3% and -14.9%; assistance availability and model-load timing vary substantially.
- **Pixel itself is much faster:** layer RPC latency improves **2.72-3.04x** with byte-identical
  kernel outputs. A separate four-request server test is 4/4 token-identical; its full-width Pixel
  request saves **{pixel_save:.1f}%** host request energy against the mean of bracketing desktop controls.
  This short controlled test is not a trace result.

## Today's longtail_eval_v2 run

14 requests (7 Qwen, 6 Gemma, 1 Llama), 3,604 output tokens, maximum output 614,
1,675 s arrival span. The 3 outputs above 512 tokens carry about 49.5% of output work.
The new window and current replan/fail-fast fixes make this a separate experiment from longtail_v1.

{live_note}

{header}{energy_rows(['eval_legacy', 'eval_op15', 'eval_two', 'eval_dispatcher'])}

The legacy baseline uses **159.872 CPU + 68.663 GPU = 228.535 kJ**, over **2,244.348 s**
(**101.827 W** average). Current run order: legacy desktop, OP15+Pixel, OP15, desktop+dispatcher.
The dispatcher-only control separates the scheduling contribution on this trace.
No partial energy extrapolation is presented as a final saving.

{current_details}
{chain_note}
## Completed longer trace: scheduling benefit without phones

longtail_v1: 31 requests, 8,207 output tokens, maximum output 1,100. Same source files and
native binaries within the pair; no phone calls in either arm.

{header}{energy_rows(['lt_legacy', 'lt_dispatcher'])}

The dispatcher saves **143.282 kJ (26.716%)** and **1,603.192 s (32.164%)**.
Model reloads fall **15 -> 8**, model switches **10 -> 4**, and logged model-load time
**630.3 -> 225.2 s**. Average host power rises **107.6 -> 116.2 W**; finishing sooner
more than offsets that increase. This supports the scheduling mechanism, not a phone-energy claim.

The OP15 arm of this older chain failed; OP15+Pixel was not run on longtail_v1.
No energy result is inferred for either missing arm. The subsequent replan fix is tested offline;
today's eval_v2 chain is its current hardware follow-up.

## Completed short trace: show both phone repeats

longtail_dev_v2: 9 requests, 1,420 output tokens, maximum output 617. All five runs complete;
every comparison has matching request IDs, prompts, lengths, seeds, source arrivals and source SLOs,
matching source-file hashes and matching native binaries. Baseline is **desktop + dispatcher**.

{header}{energy_rows(['dev_desktop', 'dev_op15_r1', 'dev_op15_r2', 'dev_two_r1', 'dev_two_r2'])}

| Configuration | Repeats | Mean host kJ | Saving vs desktop + dispatcher |
| --- | ---: | ---: | ---: |
| OP15 | 2 | {op15_mean:.2f} | {means[0]['saving_vs_desktop_pct']:.1f}% |
| OP15 + Pixel | 2 | {two_mean:.2f} | {means[1]['saving_vs_desktop_pct']:.1f}% |

Pixel execution PASS: **1,608 / 1,842** verified FFN calls in its two repeats. All 4 Qwen
requests are assisted in both. OP15 r2 loses all Qwen assistance; the two-phone r1 loses the
first two Gemma assists. The result reports identify helper readiness and model-cache timing
as major differences. Two repeats do not separate those effects from an incremental Pixel benefit.

The Qwen execution-window energies are OP15 r1 **9.99 kJ**, OP15+Pixel **10.01 / 9.11 kJ**.
These model windows can overlap and are not additive trace-energy components. They suggest a
possible local benefit, not a controlled whole-trace attribution.

## Recent implementation and qualification

| Milestone | Status | Evidence and limit |
| --- | --- | --- |
| Automatic two-phone FFN execution | PASS on dev_v2 | Pixel calls 1,608 / 1,842; OP15 remains the primary helper |
| Device-set policies per batch composition | PASS on dev_v2 | Start OP15-only, probe OP15+Pixel, keep it only within energy/latency bounds; B1 and B4 challengers accepted |
| Pixel CPU worker qualification | PASS | Byte-identical rows 1-4; 4/4 token-identical 64-token server outputs; 744 Pixel calls |
| Pixel CPU+GPU concurrency target >=1.4x | FAIL | Best measured cell 1.01x; every emulated server-cadence dual configuration slower than CPU-only |
| Replan crash correction and fail-fast | PASS in tests/replay | 141 modules / 2,031 tests, exit 0 in saved suite report; formerly failing replays complete 31/31 |
| Current eval_v2 matched two-phone energy | {live['status']} | {('Final result available above' if live['status'] == 'PASS' else 'Final matched saving not yet available')} |

The integrated Pixel helper uses the **packed CPU/NEON path over ADB TCP**, with a per-process
CPU utilization floor, bounded thread-pool polling, and batched weight-row reuse. OP15 uses its
qualified HTP/USB path. Pixel GPU, TPU, and the experimental accessory USB path are not the
backend or transport used in these two-phone traces.

## Pixel latency improvement

Per-layer RPC at server-like cadence, Qwen layers 18-23, width 17,408, ADB TCP.
Before = mean of two control-arm medians; after = one optimized-arm median, 72 calls per batch size.
These are layer-call latencies, not whole-model token latency.

| Rows per call | Before ms | After ms | Speedup | Latency reduction |
| ---: | ---: | ---: | ---: | ---: |
{latency_md}

The slow production path was dominated by low CPU/DSU clocks between short bursts. The adopted
fix keeps the CPU ready for those bursts and reuses decoded weight rows across batched inputs.
The controlled server test's full-width Pixel request uses **{pixel_j:.1f} J** vs desktop control
mean **{control_j:.1f} J**, and takes **{pixel_full['request_s']:.2f} s** vs **42.33 s** desktop mean.

## Interpretation limits

{evidence_note}

Output differences also occur in host-only comparisons. This is consistent with scheduling/batch-dependent
floating-point differences, but these artifacts do not establish a first-divergence logits tolerance
or task-quality equivalence. No exact-token exception is granted by this report.

Saved execution-identity hashes differ because source manifests include arm-specific resolved
configuration. The audit separately compares all source-file hashes and native-binary hashes;
both match within each completed comparison.

## Files and provenance

- [Why the headline changed from about 25% to 58.7%](SAVINGS_EXPLANATION.md): expanded layer sets,
  CPU energy reduction, scheduling and workload differences.
- [Recent results as report-ready tables](RECENT_RESULTS_TABLES.md).
- [New token coverage and energy timelines](timelines/README.md), with per-phone coverage,
  [two-page PDF](timelines/eval_v2_timelines.pdf) and downloadable CSV data.
{latest_figure_link}
- [Energy comparison figure](energy_comparison.png), [vector SVG](energy_comparison.svg).
- [Pixel latency figure](pixel_latency.png), [vector SVG](pixel_latency.svg).
- [Printable report](progress_report.pdf).
- [Trace table CSV](trace_energy.csv), [repeat means CSV](repeat_means.csv), [Pixel latency CSV](pixel_latency.csv).
- [Structured data and source hashes](REPORT_DATA.json).
- [Raw-result audit with exact saved tokens]({audit_path.relative_to(HERE)}).
- Source reports: [two-phone evaluation](../20260925-two-phone-eval/README.md),
  [Pixel qualification](../20260925-pixel-cpu-gpu/README.md),
  [replan diagnosis](../20260925-replan-reprojection-fix/PROGRESS.md),
  [fail-fast validation](../20260925-replan-reprojection-fix/failfast/README.md).

Reproduce without running hardware:

```sh
python3 collect_remote.py
python3 build_report.py
```

The collector only reads remote files. The builder uses saved artifacts locally. Each capture
is retained with its UTC timestamp. No commit, push or production code change was made.
'''
    (HERE / 'README.md').write_text(report)

    coverage = read(HERE / 'timelines/TIMELINE_SUMMARY.json')['coverage']
    coverage_rows = [{'model': name, **coverage[key]} for key, name in
                     [('qwen', 'Qwen3 14B'), ('gemma', 'Gemma 4 12B'),
                      ('llama', 'Llama 3.2 1B'), ('all', 'All models')]]
    csv_write('recent_token_coverage.csv', coverage_rows)
    coverage_md = '\n'.join(
        f'| {r["model"]} | {r["output_tokens"]:,} | {r["any_phone_tokens"]:,} | '
        f'{r["any_phone_pct"]:.2f}% | {r["pixel_tokens"]:,} | {r["pixel_pct"]:.2f}% |'
        for r in coverage_rows)
    incremental_note = '' if not incremental else (
        f'The two-phone arm uses **{incremental["host_saving_kj"]:.3f} kJ '
        f'({incremental["host_saving_pct"]:.2f}%) less** host energy than OP15 alone, '
        f'with **{incremental["duration_increase_pct"]:.2f}% longer** duration. '
        'This is one observation per configuration; repeatability is not established. '
        f'The two phone configurations match each other on {incremental["identical_outputs"]}/14 exact sequences.')
    tables = f'''# Recent results - {stamp}

Measured host energy is CPU-package RAPL plus GPU-board NVML. Phone energy is excluded.
Savings are relative to the baseline named for each trace; percentages across traces do not add.

## Latest matched trace: longtail_eval_v2

14 requests, 3,604 output tokens. Inputs, source-file hashes and native binaries match.
Every completed arm finished 14/14 requests with zero rejections.

{header}{energy_rows(['eval_legacy', 'eval_op15', 'eval_two', 'eval_dispatcher'])}

{incremental_note}

Strict output identity FAIL for both phone arms. No first-divergence logits or quality evaluation
justify calling these differences harmless. The dispatcher-only control remains pending.

## Token coverage in the two-phone arm

| Model | Output tokens | Any phone | Coverage | Also uses Pixel | Pixel coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
{coverage_md}

Coverage reconstruction PASS. Tokens using both phones count once. OP15 serves every assisted
token in this run; Pixel participates on a subset of Qwen tokens. These counts describe FFN
assistance on selected layers, not whole-model offload or the percentage of model FLOPs offloaded.

## Longer trace: longtail_v1

31 requests, 8,207 output tokens. Baseline is legacy desktop. All requests complete;
strict exact-output identity FAIL for the dispatcher comparison.

{header}{energy_rows(['lt_legacy', 'lt_dispatcher'])}

Model loads fall from 15 to 8, switches from 10 to 4, and load time from 630.3 to 225.2 s.
Neither arm uses a phone. This is a scheduling result.

## Short trace and repeat spread: longtail_dev_v2

9 requests, 1,420 output tokens. Baseline is desktop with the new dispatcher.
All arms complete 9/9; strict exact-output checks FAIL on all phone comparisons.

{header}{energy_rows(['dev_desktop', 'dev_op15_r1', 'dev_op15_r2', 'dev_two_r1', 'dev_two_r2'])}

| Configuration | Repeats | Mean host kJ | Saving vs desktop + dispatcher |
| --- | ---: | ---: | ---: |
| OP15 | 2 | {op15_mean:.2f} | {means[0]['saving_vs_desktop_pct']:.1f}% |
| OP15 + Pixel | 2 | {two_mean:.2f} | {means[1]['saving_vs_desktop_pct']:.1f}% |

On this shorter trace the two-phone mean uses {100 * (two_mean / op15_mean - 1):.1f}% more energy.
Helper readiness and model-load timing vary across repeats; do not report a universal Pixel gain.

## Pixel kernel/runtime improvement

Controlled Qwen FFN layer RPCs over ADB TCP at server-like cadence; 72 calls per batch size.
Before is the mean of two control medians; after is the optimized-arm median.

| Rows per call | Before ms | After ms | Speedup | Latency reduction |
| ---: | ---: | ---: | ---: | ---: |
{latency_md}

Numerical check PASS: byte-identical kernel outputs. Separate four-request server check PASS:
4/4 token-identical outputs. Its full-width Pixel arm saves {pixel_save:.1f}% host request energy
against the mean of bracketing desktop controls; this is a short controlled test, not a trace result.

## Download and provenance

- [All trace energy rows CSV](trace_energy.csv).
- [Repeat means CSV](repeat_means.csv).
- [Token coverage CSV](recent_token_coverage.csv).
- [Pixel latency CSV](pixel_latency.csv).
- [Exact saved-token audit]({audit_path.relative_to(HERE)}).
- [Full report and figures](README.md); [PDF report](progress_report.pdf).
- [Coverage and energy timeline graphs](timelines/README.md).

The OP15-only completion was checked in CHAIN-ev3 at 17:13:24 UTC. All activity for this
report was read-only on the rig; no campaign was launched, stopped or changed.
'''
    (HERE / 'RECENT_RESULTS_TABLES.md').write_text(tables)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'text.color': INK, 'axes.labelcolor': INK,
                         'xtick.color': INK, 'ytick.color': INK,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    with PdfPages(HERE / 'progress_report.pdf') as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.065, .925, 'Scheduler + phone FFN: progress report', size=24, weight='bold')
        fig.text(.065, .88, stamp + '  |  RTX 4060 Ti + i9-12900K + OP15 + Pixel 10 Pro', size=10, color=GREY)
        bullets = [
            ('26.7% less host energy on the longer trace',
             'Dispatcher-only: 536.32 -> 393.04 kJ; 4,984 -> 3,381 s. '
             '31/31 requests complete; 14/31 exact output sequences. One matched pair.'),
            ('Short-trace phone savings exceed 25%, with substantial variation',
             'OP15: 34.6-46.1% (mean 40.3%). OP15 + Pixel: 24.4-44.3% (mean 34.3%). '
             'Same desktop + dispatcher reference: 63.97 kJ. Only 4-5/9 exact sequences.'),
            ('Pixel integration works; additional trace savings remain unproven',
             f'Repeat means: OP15 {op15_mean:.2f} kJ; two phones {two_mean:.2f} kJ. '
             'Pixel made 1,608 / 1,842 verified FFN calls. Faster packed CPU path adopted; GPU not adopted.'),
            ('Pixel layer round trips are 2.72-3.04x faster',
             'B1/B2/B4: 38.42/61.44/88.39 -> 13.56/20.20/32.45 ms. '
             f'Kernel outputs unchanged. Separate server check: {pixel_save:.1f}% host saving, 4/4 token-identical outputs.'),
            ("Today's 14-request evaluation", live_note + ' Baseline: 228.535 kJ / 2,244.348 s.'),
        ]
        if live['status'] == 'PASS':
            bullets[-1] = (f'{compared["eval_two"]["host_saving_pct"]:.1f}% lower host energy on today\'s trace',
                           bullets[-1][1])
            bullets = bullets[-1:] + bullets[:-1]
        if incremental:
            bullets[3] = ('New OP15-only control: positive two-phone energy difference',
                          f'OP15 {one["host_kj"]:.2f} -> two phones {two["host_kj"]:.2f} kJ '
                          f'({incremental["host_saving_pct"]:.1f}% less); '
                          f'{incremental["duration_increase_pct"]:.1f}% longer duration. '
                          'One run each. Exact outputs vs legacy: 12/14 and 13/14.')
        y = .795
        for title, body in bullets:
            fig.text(.07, y, title, fontsize=14, weight='bold')
            fig.text(.07, y - .033, textwrap.fill(body, 113), fontsize=11, va='top', linespacing=1.45)
            y -= .134
        fig.text(.07, .055, textwrap.fill(evidence_note, 145), size=8.5, color='#586A76', va='bottom')
        pdf.savefig(fig)
        plt.close(fig)

        if live['status'] == 'PASS':
            fig, ax = plt.subplots(figsize=(10, 6.5))
            fig.subplots_adjust(left=.12, right=.95, top=.72, bottom=.25)
            fig.suptitle("Today's completed two-phone trace", x=.12, y=.95,
                         ha='left', size=21, weight='bold')
            fig.text(.12, .875, 'longtail_eval_v2: 14 requests / 3,604 output tokens  |  Measured host energy',
                     size=11, color='#586A76')
            keys = ['eval_legacy', 'eval_op15', 'eval_two'] if incremental else ['eval_legacy', 'eval_two']
            names = (['Legacy desktop', 'Dispatcher + OP15', 'Dispatcher + OP15 + Pixel']
                     if incremental else ['Legacy desktop', 'Dispatcher + OP15 + Pixel'])
            x = np.arange(len(keys))
            total = [arms[k]['host_kj'] for k in keys]
            cpu = [arms[k]['cpu_kj'] for k in keys]
            gpu = [arms[k]['gpu_kj'] for k in keys]
            ax.bar(x, cpu, .5, color=CPU, label='CPU package')
            ax.bar(x, gpu, .5, bottom=cpu, color=GPU, label='GPU board')
            tidy(ax)
            ax.set_xticks(x, names, fontsize=9)
            ax.set_ylabel('Host energy (kJ)')
            ax.set_ylim(0, max(total) * 1.3)
            ax.set_xlim(-.55, len(keys) - .45)
            for i, key in enumerate(keys):
                ax.text(i, total[i] + max(total) * .025, f'{total[i]:.2f}',
                        ha='center', size=15, weight='bold')
                ax.text(i, -max(total) * .17, f'{arms[key]["duration_s"]:.1f} s',
                        ha='center', size=11, color='#586A76')
            c = compared['eval_two']
            ax.text(len(keys) - 1, total[-1] + max(total) * .115, f'{c["host_saving_pct"]:.1f}% less energy',
                    ha='center', size=13, weight='bold', color=TEAL)
            fig.legend(*ax.get_legend_handles_labels(), frameon=False,
                       loc='upper left', bbox_to_anchor=(.11, .825), ncol=2, fontsize=10)
            fig.text(.12, .105, f'Completion PASS 14/14; strict output identity {c["strict_tokens"]} '
                     f'{c["identical_outputs"]}/14.', size=11, weight='bold',
                     color=TEAL if c['strict_tokens'] == 'PASS' else AMBER)
            figure_note = ('Single run per configuration. Phone energy excluded. '
                           f'Two phones use {incremental["host_saving_pct"]:.1f}% less than OP15;\n'
                           'OP15 exact outputs 12/14; two-phone exact outputs 13/14. Both strict checks FAIL.'
                           if incremental else 'Single matched pair. Phone energy excluded. '
                           'This combines dispatcher and phone effects;\nincremental Pixel benefit requires the OP15-only control.')
            fig.text(.12, .055, figure_note,
                     size=9, color='#586A76', linespacing=1.5)
            save(fig, 'eval_v2_comparison', pdf)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1.65]})
        fig.subplots_adjust(left=.065, right=.98, top=.78, bottom=.24, wspace=.25)
        fig.suptitle('Measured host energy: compare within each trace', x=.065, y=.96,
                     ha='left', size=21, weight='bold')
        fig.text(.065, .902, 'CPU package (RAPL) + GPU board (NVML)  |  Lower is better  |  Phones excluded',
                 size=11, color='#586A76')
        panels = [
            (axes[0], ['lt_legacy', 'lt_dispatcher'], ['Legacy\ndesktop', 'Desktop +\ndispatcher'],
             'A  Longtail: 31 requests / 8,207 tokens'),
            (axes[1], ['dev_desktop', 'dev_op15_r1', 'dev_op15_r2', 'dev_two_r1', 'dev_two_r2'],
             ['Desktop +\ndispatcher', 'OP15\nr1', 'OP15\nr2', 'OP15 + Pixel\nr1', 'OP15 + Pixel\nr2'],
             'B  Development: 9 requests / 1,420 tokens'),
        ]
        for ax, keys, names, title in panels:
            tidy(ax)
            x = np.arange(len(keys))
            cpu = [arms[k]['cpu_kj'] for k in keys]
            gpu = [arms[k]['gpu_kj'] for k in keys]
            total = [arms[k]['host_kj'] for k in keys]
            ax.bar(x, cpu, .64, color=CPU, label='CPU package')
            ax.bar(x, gpu, .64, bottom=cpu, color=GPU, label='GPU board')
            ax.set_xticks(x, names, fontsize=10)
            ax.set_ylabel('Host energy (kJ)')
            ax.set_title(title, loc='left', size=12, pad=17, weight='bold')
            ax.set_ylim(0, max(total) * 1.28)
            ax.set_xlim(-.6, len(keys) - .4)
            for i, (k, value) in enumerate(zip(keys, total)):
                c = compared.get(k)
                ax.text(i, value + max(total) * .025, f'{value:.1f}', ha='center', fontsize=12, weight='bold')
                if c:
                    ax.text(i, value + max(total) * .09, f'-{c["host_saving_pct"]:.1f}%',
                            ha='center', fontsize=11, color=TEAL, weight='bold')
                ax.text(i, -max(total) * .215, f'{arms[k]["duration_s"]:.0f} s',
                        ha='center', size=9, color='#586A76')
        axes[0].legend(frameon=False, loc='upper right', fontsize=9)
        fig.text(.065, .105, 'A: completion 31/31; exact outputs 14/31.  B: completion 9/9; exact outputs 4-5/9.',
                 size=10, color=AMBER, weight='bold')
        fig.text(.065, .061, 'One longtail pair; two phone repeats on dev_v2. No confidence intervals. '
                 'Different traces and baselines; savings do not add.', size=9, color='#586A76')
        save(fig, 'energy_comparison', pdf)

        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.subplots_adjust(left=.10, right=.96, top=.75, bottom=.24)
        fig.suptitle('Pixel FFN: fix the cost of short CPU bursts', x=.10, y=.95,
                     ha='left', size=21, weight='bold')
        fig.text(.10, .895, 'Server-like cadence, real Qwen layers 18-23, packed CPU weights, ADB TCP',
                 size=11, color='#586A76')
        tidy(ax)
        x = np.arange(3)
        before = np.array([r['before_rpc_ms'] for r in latency])
        after = np.array([r['after_rpc_ms'] for r in latency])
        ax.bar(x - .18, before, .32, color=GREY, label='Previous worker')
        ax.bar(x + .18, after, .32, color=TEAL, label='CPU floor + polling + batched reuse')
        ax.set_xticks(x, ['1 row', '2 rows', '4 rows'])
        ax.set_xlabel('Decode rows per phone call')
        ax.set_ylabel('Per-layer round trip (ms)')
        ax.set_ylim(0, 113)
        for i, r in enumerate(latency):
            ax.text(i-.18, before[i]+2, f'{before[i]:.2f}', ha='center', size=11)
            ax.text(i+.18, after[i]+2, f'{after[i]:.2f}', ha='center', size=11, weight='bold')
            ax.text(i, 103, f'{r["speedup"]:.2f}x faster', ha='center', size=12, color=TEAL, weight='bold')
        fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc='upper left',
                   bbox_to_anchor=(.09, .85), fontsize=10, ncol=2)
        fig.text(.10, .066, 'Before: mean of two control medians. After: optimized median; 72 calls per row count.\n'
                 'Byte-identical kernel outputs. These are layer-call latencies, not whole-model token latencies.',
                 size=9, color='#586A76', linespacing=1.5)
        save(fig, 'pixel_latency', pdf)

    figure_count = 3 if live['status'] == 'PASS' else 2
    print(f'Wrote report, PDF, {figure_count} figures, 3 CSVs; snapshot {stamp}')
    print(f'Mean host savings: OP15 {means[0]["saving_vs_desktop_pct"]:.3f}%; '
          f'two phones {means[1]["saving_vs_desktop_pct"]:.3f}%')


if __name__ == '__main__':
    main()
