"""Summarize a pulled Pixel bench suite: per arm/segment latency, engines, clocks, rel-L2.

usage: analyze_suite.py RUN_DIR [--ref ARM] [--json OUT.json] [--md OUT.md]
RUN_DIR holds one sub-directory per arm (replay.csv, replay.<segment>.f16, worker.log).
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

F16_FULL_LAYER_BYTES = 3 * 5120 * 17408 * 2
SAMPLE_NAMES = ['cpu2', 'cpu5', 'cpu7', 'gpu', 'gmc', 'dsu', 'memss', 'fabhbw']
DUAL = re.compile(r'S43DUALFFN request=(\d+) layer=(\d+) tokens=(\d+) columns=(\d+) primary_columns=(\d+) '
                  r'secondary_columns=(\d+) primary_us=(\d+) secondary_us=(\d+) wait_us=(\d+) merge_us=(\d+) total_us=(\d+)')


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float('nan')


def load_arm(path):
    rows = list(csv.DictReader(open(path / 'replay.csv')))
    dual = {}
    log = path / 'worker.log'
    if log.exists():
        for line in log.read_text(errors='replace').splitlines():
            m = DUAL.search(line)
            if m:
                dual[int(m.group(1))] = dict(primary_us=int(m.group(7)), secondary_us=int(m.group(8)),
                                             wait_us=int(m.group(9)), total_us=int(m.group(11)),
                                             secondary_columns=int(m.group(6)))
    return rows, dual


def summarize(run_dir, ref):
    out = {}
    arms = sorted(p for p in run_dir.iterdir() if p.is_dir() and (p / 'replay.csv').exists())
    for arm in arms:
        rows, dual = load_arm(arm)
        segments = {}
        for index, row in enumerate(rows, start=1):
            row['request'] = index
            segments.setdefault(row['segment'], []).append(row)
        result = {}
        for name, seg in segments.items():
            measured = [r for r in seg if int(r['step']) >= 0]
            compute = np.array([int(r['compute_us']) for r in measured]) / 1000.0
            rpc = np.array([int(r['rpc_us']) for r in measured]) / 1000.0
            rows_per_call = int(measured[0]['rows'])
            entry = dict(n=len(measured), rows=rows_per_call, compute_p50=pct(compute, 50), compute_p90=pct(compute, 90),
                         compute_mean=float(compute.mean()), rpc_p50=pct(rpc, 50),
                         f16_equiv_GBps=F16_FULL_LAYER_BYTES / (pct(compute, 50) / 1000.0) / 1e9)
            # first call of each token vs the rest (cadence effect)
            first = [int(r['compute_us']) / 1000.0 for r in measured if r['layer'] == measured[0]['layer']]
            rest = [int(r['compute_us']) / 1000.0 for r in measured if r['layer'] != measured[0]['layer']]
            entry['first_layer_p50'] = pct(first, 50)
            entry['other_layers_p50'] = pct(rest, 50)
            for k, sample in enumerate(SAMPLE_NAMES):
                key = f's{k}_mean'
                if key in measured[0]:
                    vals = np.array([float(r[key]) for r in measured if float(r[key]) >= 0])
                    scale = 1e6 if sample in ('gpu', 'gmc', 'dsu', 'memss', 'fabhbw') else 1e3
                    entry[f'{sample}_MHz'] = float(vals.mean() / scale) if len(vals) else float('nan')
            d = [dual[r['request']] for r in measured if r['request'] in dual]
            if d:
                entry['primary_p50'] = pct([x['primary_us'] / 1000 for x in d], 50)
                entry['secondary_p50'] = pct([x['secondary_us'] / 1000 for x in d], 50)
                entry['wait_p50'] = pct([x['wait_us'] / 1000 for x in d], 50)
                entry['secondary_columns'] = d[0]['secondary_columns']
            hashes = {}
            for r in measured:
                hashes.setdefault(r['layer'], set()).add(r['hash'])
            entry['deterministic'] = all(len(v) == 1 for v in hashes.values())
            dump = arm / f'replay.{name}.f16'
            entry['_dump'] = str(dump) if dump.exists() else None
            result[name] = entry
        out[arm.name] = result
    if ref and ref in out:
        for arm, segments in out.items():
            for name, entry in segments.items():
                ref_entry = out[ref].get(name)
                if not ref_entry or not entry['_dump'] or not ref_entry['_dump']:
                    continue
                a = np.fromfile(entry['_dump'], dtype=np.float16).astype(np.float64)
                b = np.fromfile(ref_entry['_dump'], dtype=np.float16).astype(np.float64)
                if a.shape != b.shape:
                    continue
                entry['rel_l2_vs_ref'] = float(np.linalg.norm(a - b) / np.linalg.norm(b))
                entry['max_abs_vs_ref'] = float(np.abs(a - b).max())
                entry['identical_vs_ref'] = bool(np.array_equal(a, b))
    for segments in out.values():
        for entry in segments.values():
            entry.pop('_dump', None)
    return out


def to_markdown(summary):
    cols = ['n', 'compute_p50', 'compute_p90', 'first_layer_p50', 'other_layers_p50', 'rpc_p50', 'f16_equiv_GBps',
            'primary_p50', 'secondary_p50', 'cpu2_MHz', 'cpu5_MHz', 'cpu7_MHz', 'gpu_MHz', 'gmc_MHz', 'dsu_MHz',
            'rel_l2_vs_ref', 'identical_vs_ref']
    lines = ['| arm | segment | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 2)]
    for arm, segments in summary.items():
        for name, e in segments.items():
            cells = []
            for c in cols:
                v = e.get(c, '')
                if isinstance(v, float):
                    cells.append(f'{v:.3g}' if c.startswith('rel') else f'{v:.2f}' if v < 100 else f'{v:.0f}')
                else:
                    cells.append(str(v))
            lines.append(f'| {arm} | {name} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--ref')
    parser.add_argument('--json', type=Path)
    parser.add_argument('--md', type=Path)
    args = parser.parse_args()
    summary = summarize(args.run_dir, args.ref)
    md = to_markdown(summary)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1) + '\n')
    if args.md:
        args.md.write_text(md)
    print(md)


if __name__ == '__main__':
    main()
