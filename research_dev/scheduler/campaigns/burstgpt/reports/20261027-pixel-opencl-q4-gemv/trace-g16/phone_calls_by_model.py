# completed phone calls per request (max cumulative PHYSICAL_WORK_RECORDED) and per model, for several arms
import json, sys, collections
S = '/mnt/data_h16t/zs89458/xband-runs/second-desktop'
name = {'sha256:d89e9e8': 'qwen', 'sha256:ed76f21': 'gemma'}
for arm in sys.argv[1:]:
    try: r = json.load(open(f'{S}/inputs-derived-mpaoa-{arm}/run-eval/run/RESULT.json'))
    except Exception as ex: print(arm, 'missing', ex); continue
    ev = r['request_helper_events']; last = {}; models = {}
    for e in ev:
        rid = e.get('request_id')
        if rid and rid not in models and (e.get('base') or {}).get('artifact_sha256'): models[rid] = e['base']['artifact_sha256'][:14]
        if e.get('kind') == 'PHYSICAL_WORK_RECORDED': last[rid] = max(last.get(rid, 0), e.get('completed_phone_calls', 0))
    tot = collections.Counter(); 
    for rid, v in last.items(): tot[name.get(models.get(rid, ''), '?')] += v
    reasons = collections.Counter(e.get('reason') for e in ev if e.get('kind') == 'ASSISTANCE_DECISION')
    print(f'{arm:9s} phone calls: qwen {tot["qwen"]:6d}  gemma {tot["gemma"]:6d} | PHONE_HELPER_UNAVAILABLE {reasons["PHONE_HELPER_UNAVAILABLE"]}, HELPER_PHONE_SESSION_LOAD {reasons["HELPER_PHONE_SESSION_LOAD"]}')
