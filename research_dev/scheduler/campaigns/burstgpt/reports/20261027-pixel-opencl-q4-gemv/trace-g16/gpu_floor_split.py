# Six-run comparison with the GPU idle-floor energy removed (and, for context, the whole host idle floor removed).
# Floors = lowest 60 s mean power window inside the paid window (generality_gap.py method), per run.
import sys, json
sys.path.insert(0, '/home/myid/zs89458/Documents/llama.cpp-release/research_dev/scheduler/campaigns/burstgpt/tools')
import fleet_energy, generality_gap
from pathlib import Path
S = '/mnt/data_h16t/zs89458/xband-runs/second-desktop'
RUNS = [('g62nph', 'inputs-desktop-g62nph', 'g62nph-desktop', 'desktop 10-26'), ('g62new', 'inputs-derived-mpaoa-g62new', 'g62new-derived-mpaoa', 'CPU-Pixel 10-26'),
        ('g161nph', 'inputs-desktop-g161nph', 'g161nph-desktop', 'desktop 10-28'), ('g16r1new', 'inputs-derived-mpaoa-g16r1new', 'g16r1new-derived-mpaoa', 'OpenCL-Pixel, Gemma unassisted'),
        ('g16c1new', 'inputs-derived-mpaoa-g16c1new', 'g16c1new-derived-mpaoa', 'CPU-Pixel 10-28'), ('g16o1new', 'inputs-derived-mpaoa-g16o1new', 'g16o1new-derived-mpaoa', 'OpenCL-Pixel 10-28')]
WINDOW = 60.0
rows = {}
for label, d, m, note in RUNS:
    directory = Path(f'{S}/{d}'); meter = Path(f'{S}/{m}/METER.json')
    base = fleet_energy.account_run(label, directory, meter_path=meter if meter.exists() else None)
    result = json.loads(fleet_energy.find_result(directory).read_text())
    start_s, end_s = result['paid_start_ns'] / 1e9, result['paid_end_ns'] / 1e9
    samples = fleet_energy.load_samples(directory) or []
    gpu = sorted((int(r['gpu']['sample_t_ns']) / 1e9, float(r['gpu']['power_mw']) / 1000) for r in samples if 'gpu' in r)
    gpu = [(t, w) for t, w in gpu if start_s <= t <= end_s]
    gfloor = generality_gap.idle_floor(gpu, WINDOW)
    host = generality_gap.host_power_series(samples, start_s, end_s)
    hfloor = generality_gap.idle_floor(host, WINDOW)
    dur = base['duration_s']; cpu = base['desktop']['cpu_package_j']; gpuj = base['desktop']['gpu_board_j']; hostj = base['desktop']['host_j']
    gidle = gfloor['window_min_w'] * dur; hidle = hfloor['window_min_w'] * dur
    rows[label] = dict(note=note, dur=dur, cpu=cpu, gpu=gpuj, host=hostj, phones=base['phones_j'], fleet=base['fleet_j'],
                       gpu_floor_w=gfloor['window_min_w'], gpu_p05_w=gfloor['p05_w'], gpu_idle=gidle, gpu_dyn=gpuj - gidle, host_ex_gpu_floor=hostj - gidle,
                       host_floor_w=hfloor['window_min_w'], host_idle=hidle, host_dyn=hostj - hidle, gpu_mean_w=gpuj / dur)
def kj(v): return f'{v/1000:.1f}'
def pct(a, b): return f'{(a/b-1)*100:+.1f} %'
B = rows['g161nph']
print('| run | GPU mean W | GPU floor W (p05) | GPU kJ = idle + dynamic | host kJ | host ex GPU floor | vs g161nph | host floor W | host dynamic kJ | vs g161nph (dyn) | fleet ex GPU floor kJ | vs g161nph |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for label, *_ in RUNS:
    r = rows[label]
    print(f"| {label} ({r['note']}) | {r['gpu_mean_w']:.1f} | {r['gpu_floor_w']:.1f} ({r['gpu_p05_w']:.1f}) | {kj(r['gpu'])} = {kj(r['gpu_idle'])} + {kj(r['gpu_dyn'])} | {kj(r['host'])} | {kj(r['host_ex_gpu_floor'])} | {pct(r['host_ex_gpu_floor'], B['host_ex_gpu_floor'])} | {r['host_floor_w']:.1f} | {kj(r['host_dyn'])} | {pct(r['host_dyn'], B['host_dyn'])} | {kj(r['fleet'] - r['gpu_idle'])} | {pct(r['fleet'] - r['gpu_idle'], B['fleet'] - B['gpu_idle'])} |")
print()
for a, b in (('g16c1new', 'g16o1new'), ('g62nph', 'g62new')):
    ra, rb = rows[a], rows[b]
    print(f'{b} vs {a}: host {pct(rb["host"], ra["host"])}, host ex GPU floor {pct(rb["host_ex_gpu_floor"], ra["host_ex_gpu_floor"])}, host dynamic {pct(rb["host_dyn"], ra["host_dyn"])}, GPU dynamic {kj(ra["gpu_dyn"])} -> {kj(rb["gpu_dyn"])} kJ, CPU {kj(ra["cpu"])} -> {kj(rb["cpu"])} kJ')
json.dump(rows, open(f'{S}/analysis-g16ab/gpu-floor-split.json', 'w'), indent=1)
