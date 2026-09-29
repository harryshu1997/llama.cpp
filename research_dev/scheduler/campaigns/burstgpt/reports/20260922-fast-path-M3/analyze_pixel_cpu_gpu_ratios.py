"""Summarize the four archived Pixel CPU/GPU ratio experiments."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics

import numpy as np

B = Path(__file__).resolve().parent
STAGES = ('smoke', 'coarse', 'refine', 'confirm')

def read(p):
    return json.loads(p.read_text())

def calls(root, name):
    return [json.loads(line) for line in (root/name/'CALLS.jsonl').read_text().splitlines()]

def quantiles(root, names, width):
    ms = [r['worker_us']/1000 for name in names for r in calls(root,name) if r['warm'] and r['columns']==int(width)]
    return {'samples':len(ms),'p50_ms':float(np.quantile(ms,.5)),'p90_ms':float(np.quantile(ms,.9)),
            'p99_ms':float(np.quantile(ms,.99)),'max_ms':max(ms)}

out = {'timestamp_utc':datetime.now(timezone.utc).isoformat(), 'stages':{}, 'ratios':{},
       'measurement_scope':'Phone-local one-row FFN execution; excludes USB, desktop and energy.',
       'energy_measured':False,'server_tokens_verified':False,'production_integration':False}
control_checks = {}
previous = B/'physical/pixel10pro-cpu-gpu-confirm-1/run1'
unique_ratios = set()
for stage in STAGES:
    root=B/f'physical/pixel10pro-cpu-gpu-ratio-{stage}-1/run1'
    audit=read(root/'SWEEP_AUDIT.json');dual=read(root/'DUAL_COVERAGE.json');cfg=read(root/'CONFIG.json')
    assert audit['status']=='PASS' and read(root/'CPU_AFFINITY.json')['status']=='PASS'
    assert read(root/'CLEANUP.json')['status']=='PASS'
    numeric=read(root/'RESULT.json')
    temperatures=[]
    for p in (root/'raw').glob('*/BATTERY_*.txt'):
        temperatures += [int(x)/10 for x in re.findall(r'^\s*temperature: (\d+)$',p.read_text(),re.M)]
    out['stages'][stage]={'path':str(root.relative_to(B)),'arms':len(cfg['arms']),'phone_calls':audit['phone_calls'],
        'numerical_status':'PASS','maximum_relative_l2':audit['maximum_relative_l2'],
        'dual_calls':sum(a['calls'] for a in dual['arms'].values()),
        'overlapping_calls':sum(a.get('overlapping_calls',a['calls']) for a in dual['arms'].values()),
        'overlap_status':dual['status'],'affinity_status':'PASS','cleanup_status':'PASS',
        'warmup_repeats':cfg.get('warmup_repeats',2),'battery_C_range':[min(temperatures),max(temperatures)]}
    for arm in cfg['arms']:
        reference=None
        if 'cpu_half_columns' in arm:
            unique_ratios.add(arm['cpu_half_columns'])
            if arm['cpu_half_columns']==4352:reference='03-dual4-pinned'
        elif 'gpu_block_mask' in arm:reference='03-dual4-pinned'
        elif arm.get('backend')=='CPU':reference='01-cpu6'
        elif arm.get('backend')=='Vulkan0':reference='00-gpu'
        if reference:
            actual=calls(root,arm['name']);ref=calls(previous,reference)[:len(actual)]
            exact=sum(a['output_sha256']==b['output_sha256'] for a,b in zip(actual,ref))
            assert exact==len(actual)
            control_checks[stage+'/'+arm['name']]={'status':'PASS','calls':exact,'previous_arm':reference}
    assert numeric['status']=='PASS'
root=B/'physical/pixel10pro-cpu-gpu-ratio-confirm-1/run1'
audit=read(root/'SWEEP_AUDIT.json');cfg=read(root/'CONFIG.json');dual=read(root/'DUAL_COVERAGE.json')['arms']
comparisons={a['name']:a for a in audit['comparisons']}
summary=audit['arms'];arms=cfg['arms']
cpu_controls=[i for i,a in enumerate(arms) if a.get('backend')=='CPU' and 'gpu_block_mask' not in a]
gpu_names=[a['name'] for a in arms if a.get('backend')=='Vulkan0']
for c in sorted({a['cpu_half_columns'] for a in arms if 'cpu_half_columns' in a}):
    indices=[i for i,a in enumerate(arms) if a.get('cpu_half_columns')==c]
    names=[arms[i]['name'] for i in indices]
    assert len(names)==2
    refs=[calls(root,n) for n in names]
    exact=sum(a['output_sha256']==b['output_sha256'] for a,b in zip(*refs));assert exact==240
    entry={'cpu_half_columns':c,'gpu_half_columns':8704-c,'cpu_percent':100*c/8704,'gpu_percent':100*(8704-c)/8704,
           'arms':names,'repeat_exact_calls':exact,'maximum_relative_l2':max(summary[n]['maximum_relative_l2'] for n in names),'widths':{}}
    matched_cpu=[[arms[j]['name'] for j in (max(x for x in cpu_controls if x<i),min(x for x in cpu_controls if x>i))] for i in indices]
    matched_50=[comparisons[n]['controls'] for n in names]
    for width in ('8704','17408'):
        repeat_ms=[summary[n][width]['worker']['mean_ms'] for n in names]
        mean_ms=statistics.mean(repeat_ms)
        cpu_per_repeat=[statistics.mean(summary[n][width]['worker']['mean_ms'] for n in pair) for pair in matched_cpu]
        equal_per_repeat=[comparisons[n]['widths'][width]['control_mean_ms'] for n in names]
        cpu_ms=statistics.mean(cpu_per_repeat);equal_ms=statistics.mean(equal_per_repeat)
        gpu_ms=statistics.mean(summary[n][width]['worker']['mean_ms'] for n in gpu_names)
        qs=quantiles(root,names,width)
        qc=quantiles(root,sorted({n for p in matched_cpu for n in p}),width)
        qe=quantiles(root,sorted({n for p in matched_50 for n in p}),width)
        qg=quantiles(root,gpu_names,width)
        branches={k:statistics.mean(dual[n]['widths'][width]['mean_ms'][k] for n in names) for k in dual[names[0]]['widths'][width]['mean_ms']}
        entry['widths'][width]={'cpu_columns':c*(int(width)//8704),'gpu_columns':(8704-c)*(int(width)//8704),
            'mean_ms':mean_ms,'repeat_ms':repeat_ms,'equal_control_ms':equal_ms,'equal_control_names':matched_50,'equal_control_ms_per_repeat':equal_per_repeat,
            'cpu_control_ms':cpu_ms,'cpu_control_names':matched_cpu,'gpu_reference_ms':gpu_ms,
            'saving_vs_equal_pct':100*(1-mean_ms/equal_ms),'saving_vs_cpu_pct':100*(1-mean_ms/cpu_ms),'saving_vs_gpu_pct':100*(1-mean_ms/gpu_ms),
            'each_repeat_saving_vs_equal_pct':[100*(1-v/ctrl) for v,ctrl in zip(repeat_ms,equal_per_repeat)],
            'each_repeat_saving_vs_cpu_pct':[100*(1-v/ctrl) for v,ctrl in zip(repeat_ms,cpu_per_repeat)],
            'mean_speed_vs_equal_status':'PASS' if all(v<ctrl for v,ctrl in zip(repeat_ms,equal_per_repeat)) else 'FAIL',
            'mean_speed_vs_cpu_status':'PASS' if all(v<ctrl for v,ctrl in zip(repeat_ms,cpu_per_repeat)) else 'FAIL',
            'quantiles':qs,'equal_quantiles':qe,'cpu_quantiles':qc,'gpu_quantiles':qg,
            'p99_vs_cpu_status':'PASS' if qs['p99_ms']<qc['p99_ms'] else 'FAIL',
            'p99_vs_equal_status':'PASS' if qs['p99_ms']<qe['p99_ms'] else 'FAIL',
            'branch_mean_ms':branches,'matrix_gflops_per_worker_second':6*5120*int(width)/(mean_ms*1e6),
            'effective_weight_GB_per_second':6*5120*int(width)/(mean_ms*1e6)}
    out['ratios'][str(c)]=entry
out['best_per_width']={w:min(out['ratios'],key=lambda c:out['ratios'][c]['widths'][w]['mean_ms']) for w in ('8704','17408')}
out['best_control_normalized_per_width']={w:max(out['ratios'],key=lambda c:out['ratios'][c]['widths'][w]['saving_vs_equal_pct']) for w in ('8704','17408')}
eligible = [c for c, entry in out['ratios'].items() if all(
    width[key] == 'PASS' for width in entry['widths'].values()
    for key in ('mean_speed_vs_equal_status', 'mean_speed_vs_cpu_status', 'p99_vs_cpu_status', 'p99_vs_equal_status'))]
out['recommended_shared_ratio'] = min(eligible, key=lambda c: out['ratios'][c]['widths']['17408']['mean_ms']) if eligible else None
out['shared_ratio_selection'] = 'Lowest full mean among tested ratios passing repeated mean and observed p99 comparisons at both widths versus CPU and50/50.'
out.update(numerical_status='PASS',phone_calls=sum(a['phone_calls'] for a in out['stages'].values()),
           dual_calls=sum(a['dual_calls'] for a in out['stages'].values()),overlapping_calls=sum(a['overlapping_calls'] for a in out['stages'].values()),
           mixed_ratios_tested=[{'cpu_half_columns':c,'cpu_percent':100*c/8704} for c in sorted(unique_ratios)],
           control_checks=control_checks,maximum_relative_l2=max(a['maximum_relative_l2'] for a in out['stages'].values()),
           mean_control_method='Mean of nearest bracketing controls for each candidate, then mean across reversed repeats.',
           tail_control_method='Pool distinct matched control arms; do not duplicate a shared middle control.',
           final_warmup_repeats=10,prior_warmup_repeats=2)
out['software_sha256']={name:hashlib.sha256((B/path).read_bytes()).hexdigest() for name,path in {
    'worker':'software/pixel10pro-cpu-gpu-v3/llama-ffn-split-worker',
    'vulkan':'software/pixel10pro-dense-gemv-ratio-v1/libggml-vulkan.so',
    'cpu_affinity':'software/pixel10pro-cpu-tune-v3/libggml-cpu.so'}.items()}
(B/'PIXEL_CPU_GPU_RATIO_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('phone_calls','dual_calls','overlapping_calls','best_per_width','best_control_normalized_per_width')},indent=2))
for c,e in out['ratios'].items():
    print(c,round(e['cpu_percent'],4),*[round(e['widths'][w]['mean_ms'],4) for w in ('8704','17408')],*[round(e['widths'][w]['saving_vs_equal_pct'],2) for w in ('8704','17408')])
