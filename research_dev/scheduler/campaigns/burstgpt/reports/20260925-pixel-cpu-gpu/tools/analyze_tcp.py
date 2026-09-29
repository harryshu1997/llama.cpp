"""Summarize an over-TCP qualification run (qualify_tcp.py output) into TCP_TABLE.{md,json}."""

import json
from pathlib import Path
import statistics
import sys


def p(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q / 100 * len(values)))]


def main():
    run = Path(sys.argv[1])
    out = Path(sys.argv[2])
    arms = [d for d in sorted(run.iterdir()) if (d / 'RESULT.json').exists()]
    reference = next(d for d in arms if 'prod' in d.name)
    table = []
    for arm in arms:
        result = json.loads((arm / 'RESULT.json').read_text())
        calls = json.loads((arm / 'CALLS.json').read_text())
        for segment in sorted({c['segment'] for c in calls}):
            measured = [c for c in calls if c['segment'] == segment and c['step'] >= 0]
            compute = [c['compute_us'] / 1000 for c in measured]
            rpc = [c['rpc_us'] / 1000 for c in measured]
            first = [c['compute_us'] / 1000 for c in measured if c['layer'] == 18]
            other = [c['compute_us'] / 1000 for c in measured if c['layer'] != 18]
            ref_dump = (reference / f'{segment}.f16').read_bytes()
            table.append(dict(arm=arm.name, segment=segment, rows=measured[0]['rows'], calls=len(measured),
                              compute_p50=statistics.median(compute), compute_p90=p(compute, 90),
                              rpc_p50=statistics.median(rpc), rpc_p90=p(rpc, 90),
                              overhead_p50=statistics.median(c['overhead_us'] / 1000 for c in measured),
                              first_layer_p50=statistics.median(first), other_layers_p50=statistics.median(other),
                              six_layer_rpc_ms=6 * statistics.median(rpc),
                              identical_to_production=(arm / f'{segment}.f16').read_bytes() == ref_dump,
                              stop_exit=result['stop']['exit_code'], boot_unchanged=result['stop']['boot_unchanged'],
                              forward_removed=result['stop']['forward_removed']))
    out.with_suffix('.json').write_text(json.dumps(table, indent=1) + '\n')
    lines = ['| arm | rows | calls | compute p50 / p90 ms | RPC p50 / p90 ms | non-compute p50 ms | first layer / others ms | '
             '6-layer RPC ms | outputs = production | exit / boot / forward |',
             '|---|---:|---:|---|---|---:|---|---:|---|---|']
    for r in table:
        lines.append(f"| {r['arm']} | {r['rows']} | {r['calls']} | {r['compute_p50']:.2f} / {r['compute_p90']:.2f} | "
                     f"{r['rpc_p50']:.2f} / {r['rpc_p90']:.2f} | {r['overhead_p50']:.2f} | "
                     f"{r['first_layer_p50']:.2f} / {r['other_layers_p50']:.2f} | {r['six_layer_rpc_ms']:.1f} | "
                     f"{'byte-identical' if r['identical_to_production'] else 'DIFFERENT'} | "
                     f"{r['stop_exit']} / {'same' if r['boot_unchanged'] else 'CHANGED'} / {'removed' if r['forward_removed'] else 'LEFT'} |")
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
