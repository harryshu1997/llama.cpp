"""Configure/calibrate a larger document using the existing canonical gates."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PARENT = Path('/mnt/storage/s42-prefill-affinity-20260916-v2-nuxeVS')
PRIOR = Path('/mnt/storage/s42-ffn-microbatch-20260916-v4-arwuw3')
sys.path.insert(0, str(ROOT / 'source'))
from research_dev.scheduler import RuntimeCapabilityCatalog
from research_dev.scheduler._internal.types import canonical_json


def save(path, value):
    with path.open('x') as stream:
        stream.write(canonical_json(value) + '\n')


def digest(path):
    return 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()


mode = sys.argv[1]
context = int(sys.argv[2])
assert context in (16384, 32768, 65536, 131072, 262144)
if mode == 'initialize':
    command = json.loads((PRIOR / 'calibrate-command.json').read_text())
    command[1] = str(ROOT / 'source/research_dev/scheduler/campaigns/burstgpt/desktop_parent_calibration.py')
    for key, value in {'--context-size': str(context), '--output': str(ROOT / 'calibration-v1'),
        '--source-manifest': str(PARENT / 'SOURCE_MANIFEST.json'),
        '--phone-session-root': '/data/local/tmp/' + ROOT.name,
        '--phone-remote-hash-cache': str(ROOT / 'PHONE_HASH_CACHE.json')}.items():
        command[command.index(key) + 1] = value
    save(ROOT / 'CALIBRATE_COMMAND.json', command)
elif mode == 'configure':
    evidence = ROOT / 'calibration-v1/DESKTOP_PARENT_CALIBRATION.json'
    measured = json.loads(evidence.read_text())
    assert measured['status'] == 'PASS'
    assert measured['launch_contract']['context_size'] == context
    assert measured['launch_contract']['gpu_layers'] == 23
    plans = ROOT / 'calibration-v1/MEASURED_DESKTOP_BASELINE_PLANS_V1.json'
    plan = next(row for row in json.loads(plans.read_text())['plans']
                if row['artifact_sha256'] == measured['artifact_sha256'])
    catalog = json.loads((PARENT / 'CONTEXT_CATALOG.json').read_text())
    control = next(row for row in catalog['desktop_control_profiles']
                   if row['artifact_sha256'] == measured['artifact_sha256'])
    assert control['placement_sha256'] == measured['placement_sha256']
    source = next(row for row in catalog['composite_executors'] if row['executor_id'] == control['executor_id'])
    context_resource = source['adapter_parameters']['context_resource_id']
    quantum = source['adapter_parameters']['context_token_quantum']
    for row in catalog['composite_executors']:
        if row['executor_id'] == source['executor_id'] or row['baseline_executor_id'] == source['executor_id']:
            parameters = {key: value for key, value in row['adapter_parameters'].items()
                          if not key.startswith('capacity_parent_')}
            parameters.update(plan['adapter_parameters'])
            row['adapter_parameters'] = parameters
            row['maturity'] = 'QUALIFIED' if row['executor_id'] == source['executor_id'] else 'SHADOW'
            if row['executor_id'] == source['executor_id']:
                row['evidence_ids'] = [digest(evidence)]
    control['evidence_ids'] = [digest(evidence)]
    for row in catalog['resources']:
        if row['resource_id'] == context_resource:
            row['capacity'] = context // quantum
    catalog['catalog_id'] += ':context' + str(context)
    RuntimeCapabilityCatalog.from_json(catalog)
    save(ROOT / 'CONTEXT_CATALOG.json', catalog)
    original = (PRIOR / 'PROMPT.txt').read_text()
    header, body = original.split('BEGIN ARCHIVE\n', 1)
    body, question = body.rsplit('END ARCHIVE\n', 1)
    repeats = (context - 512) // 5200
    prompt = header + 'BEGIN ARCHIVE\n' + body * repeats + 'END ARCHIVE\n' + question
    with (ROOT / 'PROMPT.txt').open('x') as stream:
        stream.write(prompt)
    command = json.loads((PARENT / 'GATE_COMMAND.json').read_text())
    command[1] = str(ROOT / 'source/research_dev/scheduler/campaigns/burstgpt/remote_resident_gate.py')
    for key, value in {'--capability-catalog': str(ROOT / 'CONTEXT_CATALOG.json'),
        '--desktop-baseline-plans': str(plans), '--output': str(ROOT / 'gate-run-v1'),
        '--phone-session-root': '/data/local/tmp/' + ROOT.name,
        '--phone-remote-hash-cache': str(ROOT / 'PHONE_HASH_CACHE.json'),
        '--prompt-file': str(ROOT / 'PROMPT.txt'), '--minimum-input-tokens': str(context // 2)}.items():
        command[command.index(key) + 1] = value
    save(ROOT / 'GATE_COMMAND.json', command)
    save(ROOT / 'CONTEXT_SPEC.json', {'context_size': context, 'document_repetitions': repeats,
        'prompt_sha256': digest(ROOT / 'PROMPT.txt'), 'output_tokens': 64,
        'physical_desktop_calibration_sha256': digest(evidence),
        'parent_placement_sha256': measured['placement_sha256'],
        'changes': ['context allocation', 'actual prompt length'],
        'unchanged': ['artifacts', 'binaries', 'GPU placement', 'shards', 'batch', 'ubatch', 'seed',
                      'output limit', 'CUDA graph mode', 'process-scoped phone CPU affinity']})
else:
    raise ValueError(mode)
