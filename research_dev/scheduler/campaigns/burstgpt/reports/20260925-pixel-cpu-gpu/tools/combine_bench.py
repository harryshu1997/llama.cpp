"""Combine the three repetitions of the CPU+GPU sweep into BENCH_TABLE.{md,json}.

rep1 = physical/d2r1 (+ GPU-only from physical/d1 h-gpu-f16-uclamp), rep2/rep3 = physical/d2r23.
"""

import json
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent.parent
F16_LAYER = 3 * 5120 * 17408 * 2
PACKED_LAYER = 158064640  # mean unique packed Q4_K/Q6_K bytes per layer (layers 18-23)
SHARES = {'s10': 1792, 's15': 2560, 's25': 4352, 's25-t5': 4352, 's25-t4': 4352, 's33': 5632, 's40': 6912,
          's50': 8704, 's60': 10496}


def load(path):
    return json.loads((BASE / path).read_text())


def main():
    d1 = load('physical/d1/SUMMARY.json')
    r1 = load('physical/d2r1/SUMMARY.json')
    r23 = load('physical/d2r23/SUMMARY.json')
    reps = {1: {'cpu': r1['a-cpu'], 'gpu': d1['h-gpu-f16-uclamp'], **{k: r1[k] for k in r1 if k.startswith('s')}}}
    for rep in (2, 3):
        reps[rep] = {name[3:]: value for name, value in r23.items() if name.startswith(f'r{rep}-')}
    configs = ['cpu', 'cpu-batch', 'cpu-poll-only', 'gpu', 's10', 's15', 's25', 's25-t5', 's25-t4', 's33', 's40',
               's50', 's60', 's25-noboost']
    segments = ['burst-m1', 'burst-m2', 'burst-m4', 'prod-m1', 'prod-m2', 'prod-m4']
    table = []
    for config in configs:
        for segment in segments:
            cells = [(rep, reps[rep][config][segment]) for rep in reps if config in reps[rep] and segment in reps[rep][config]]
            if not cells:
                continue
            walls = [e['compute_p50'] for _, e in cells]
            speed = [reps[rep]['cpu'][segment]['compute_p50'] / e['compute_p50'] for rep, e in cells]
            gpu_cols = SHARES.get(config.replace('-noboost', ''), 17408 if config == 'gpu' else 0)
            share = gpu_cols / 17408
            wall = statistics.median(walls)
            physical = PACKED_LAYER * (1 - share) + F16_LAYER * share
            row = dict(config=config, segment=segment, reps=len(cells), gpu_share=round(share, 3),
                       wall_p50=wall, wall_min=min(walls), wall_max=max(walls),
                       wall_p90=statistics.median(e['compute_p90'] for _, e in cells),
                       cpu_leg=statistics.median(e['primary_p50'] for _, e in cells) if 'primary_p50' in cells[0][1] else None,
                       gpu_leg=statistics.median(e['secondary_p50'] for _, e in cells) if 'secondary_p50' in cells[0][1] else None,
                       speedup_vs_cpu=statistics.median(speed), speedup_min=min(speed), speedup_max=max(speed),
                       logical_GBps=F16_LAYER / wall / 1e6, physical_GBps=physical / wall / 1e6,
                       rel_l2_vs_cpu=max(e.get('rel_l2_vs_ref') or 0.0 for _, e in cells),
                       identical_vs_cpu=all(e.get('identical_vs_ref') for _, e in cells))
            table.append(row)
    (BASE / 'BENCH_TABLE.json').write_text(json.dumps(table, indent=1) + '\n')
    lines = ['| config | GPU share | segment | reps | wall p50 ms [min-max] | wall p90 | CPU leg | GPU leg | '
             'x vs CPU-only [min-max] | logical GB/s | physical GB/s | rel-L2 vs CPU |',
             '|---|---:|---|---:|---|---:|---:|---:|---|---:|---:|---|']
    for r in table:
        leg = lambda v: f'{v:.2f}' if v is not None else '-'
        rel = 'identical' if r['identical_vs_cpu'] else f"{r['rel_l2_vs_cpu']:.1e}"
        lines.append(f"| {r['config']} | {r['gpu_share']:.3f} | {r['segment']} | {r['reps']} | "
                     f"{r['wall_p50']:.2f} [{r['wall_min']:.2f}-{r['wall_max']:.2f}] | {r['wall_p90']:.2f} | "
                     f"{leg(r['cpu_leg'])} | {leg(r['gpu_leg'])} | {r['speedup_vs_cpu']:.2f} "
                     f"[{r['speedup_min']:.2f}-{r['speedup_max']:.2f}] | {r['logical_GBps']:.1f} | "
                     f"{r['physical_GBps']:.1f} | {rel} |")
    (BASE / 'BENCH_TABLE.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
