#!/usr/bin/env bash
set -euo pipefail
cd /home/myid/zs89458/Documents/llama.cpp-release
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import csv
import importlib.util
import json
root=Path('research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2')
physical=root/'physical/step3-numeric'
spec=importlib.util.spec_from_file_location('m2_check',root/'ANALYZE.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
arms=[];diagnostics=[]
for run in ('run1','run2'):
    p=physical/run
    measurement,entries,identity=m.arm(physical,2,'combined',576,512,p)
    diagnostic=m.read(p/'CHECK_DIAGNOSTIC.json')
    m.require(diagnostic['status']=='PASS' and diagnostic['numeric']['status']=='PASS','numeric diagnostic failed')
    close=m.read(p/'PHONE_CLOSE.json')
    m.require(close['terminal']['status']==0 and close['restoration']['status']=='RESTORED','phone close failed')
    (p/'ARM_CHECK.json').write_text(json.dumps({'status':'PASS','measurement':measurement},indent=2)+'\n')
    rows=diagnostic['numeric']['max_per_call_row']
    with (p/'MAX_LOCAL_REL_L2_PER_CALL_ROW.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    calls=0;count=0;release=[]
    for lineno,line in enumerate((p/'combined.stderr').read_text().splitlines(),1):
        if line.startswith('S41SERVERFFNCALL '):calls+=1
        if line.startswith('S41SERVERFFNROW stage=returned '):count+=1
        if 'S41SERVERFFN dormant_host_share phase=decode ' in line:
            release.append({'line_number':lineno,'line':line,'phone_calls_before_release':calls,
                            'diagnostic_returned_rows_before_release':count})
    m.require(len(release)==1 and release[0]['phone_calls_before_release']==1152 and
              release[0]['diagnostic_returned_rows_before_release']==2304,'release preceded the diagnostic window')
    (p/'DELAYED_RELEASE.json').write_text(json.dumps(release,indent=2)+'\n')
    refs=m.read(p/'EXACT_TOKENS.json')
    m.require(all(row['matching_tokens']==row['compared_tokens']==576 for row in refs[0]['token_comparisons']),
              'historical host token mismatch')
    ack=m.read(p/'COHORT_CONTROL.json')['ack']['cohort_members']
    result=m.read(p/'RESULT.json')
    arms.append({'run':run,'numeric_check':'PASS','historical_exact_token_check':'PASS','hang_check':'PASS',
                 'slots_by_request':[row['slot_id'] for row in entries],
                 'ack_by_request':[row['applied_token_index'] for row in ack],
                 'diagnostic_rows':len(rows),'max_local_rel_l2':diagnostic['numeric']['max_local_rel_l2'],
                 'first_step_max_local_rel_l2':max(row['local_rel_l2'] for row in rows if row['step']==1),
                 'worst_row':diagnostic['numeric']['worst_row'],'rows_above_threshold':[],
                 'measurement':measurement,'native_runtime':identity['runtime'],
                 'request_fixture_sha256':result['cohort']['request_fixture_sha256'],
                 'submission_order':result['cohort']['submission_order']})
    diagnostics.append(diagnostic)
a,b=({(row['input']['call'],row['input']['request_id']):row for row in check['row_records']} for check in diagnostics)
comparisons=[]
for key in sorted(a.keys() & b.keys()):
    x,y=a[key],b[key]
    comparisons.append({'call':key[0],'request_id':key[1],'run1_input':x['input'],'run2_input':y['input'],
                        'same_input_wire':x['input']['wire_sha256']==y['input']['wire_sha256'],
                        'same_returned_wire':x['returned']['wire_sha256']==y['returned']['wire_sha256']})
secondary={'aligned_rows':len(comparisons),'identical_input_rows':sum(row['same_input_wire'] for row in comparisons),
           'different_returns_for_identical_inputs':sum(row['same_input_wire'] and not row['same_returned_wire'] for row in comparisons),
           'rows':comparisons}
(physical/'REQUEST_ALIGNED_WIRE_COMPARISON.json').write_text(json.dumps(secondary,indent=2)+'\n')
wire=m.read(physical/'CHECK_DETERMINISM.json')
m.require(arms[0]['native_runtime']==arms[1]['native_runtime'],'repetition native runtime differs')
m.require(arms[0]['request_fixture_sha256']==arms[1]['request_fixture_sha256'],'request configuration differs')
m.require(all(arm['submission_order']==[] for arm in arms),'submission was not concurrent')
summary={'at_utc':datetime.now(timezone.utc).isoformat(),'status':'INCONCLUSIVE',
         'step3_numeric_and_historical_exact_token_check':'PASS',
         'step3_determinism_check':'INCONCLUSIVE: no identical input rows between repetitions',
         'common_calls':wire['common_calls'],'compared_row_positions':wire['common_rows'],
         'identical_call_input_rows':wire['rows_with_identical_call_inputs'],
         'identical_row_inputs_at_same_payload_position':wire['rows_with_identical_row_inputs'],
         'identical_row_inputs_aligned_by_request':secondary['identical_input_rows'],
         'nondeterminism_observed':wire['nondeterminism_observed'],
         'tolerance_amendment_condition_met':False,'m2_acceptance':'PENDING; original exact-token rule remains',
         'arms':arms,'cleanup':m.read(physical/'CLEANUP.json')['status']}
m.require(wire['rows_with_identical_call_inputs']==0 and secondary['identical_input_rows']==0,
          'update the conclusion: comparable inputs exist')
(physical/'STEP3_CHECK.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({**{k:v for k,v in summary.items() if k!='arms'},'arms':[
    {k:v for k,v in row.items() if k not in ('native_runtime','measurement')} for row in arms]},indent=2))
PY
