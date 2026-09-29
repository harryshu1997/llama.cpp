import json
from pathlib import Path
import statistics
import sys
import numpy as np

root=Path(sys.argv[1])
warmup=int(sys.argv[2]) if len(sys.argv)>2 else 10
cfg=json.loads((root/'CONFIG.json').read_text())
controls=[i for i,a in enumerate(cfg['arms']) if a['name'] in cfg['comparison_controls']]
rows={a['name']:[json.loads(x) for x in (root/a['name']/'CALLS.jsonl').read_text().splitlines()] for a in cfg['arms']}
means={name:{str(w):statistics.mean(r['worker_us'] for r in rr if r['repeat']>=warmup and r['columns']==w)/1000 for w in (8704,17408)} for name,rr in rows.items()}
out={'warmup_repeats':warmup,'arm_means_ms':means,'candidates':[]}
for i,a in enumerate(cfg['arms']):
    if i in controls or a.get('reference_only'):continue
    names=[cfg['arms'][j]['name'] for j in (max(x for x in controls if x<i),min(x for x in controls if x>i))]
    item={'name':a['name'],'cpu_percent':a['cpu_half_columns']/8704*100,'cpu_half_columns':a['cpu_half_columns'],'controls':names,'widths':{}}
    for w in ('8704','17408'):
        c=statistics.mean(means[n][w] for n in names)
        ms=[r['worker_us']/1000 for r in rows[a['name']] if r['repeat']>=warmup and str(r['columns'])==w]
        item['widths'][w]={'mean_ms':means[a['name']][w],'control_mean_ms':c,'saving_pct':100*(1-means[a['name']][w]/c),'p50_ms':float(np.quantile(ms,.5)),'p90_ms':float(np.quantile(ms,.9)),'p99_ms':float(np.quantile(ms,.99)),'samples':len(ms)}
    out['candidates'].append(item)
print(json.dumps(out,indent=2))
