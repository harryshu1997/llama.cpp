"""Bind the diagnostic router to the previous canonical bounded experiment."""

import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PRIOR = Path('/mnt/storage/s42-ffn-microbatch-20260916-v4-arwuw3')
sys.path.insert(0, str(ROOT / 'source'))
from research_dev.scheduler.adapters.transport_profiles import (
    build_transport_qualification_identity, materialize_measured_usb_links,
)
from research_dev.scheduler import ResourceProfile, RuntimeCapabilityCatalog
from research_dev.scheduler._internal.types import canonical_json, canonical_sha256


def save(path, value):
    with path.open('x') as stream:
        stream.write(canonical_json(value) + '\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()


command = json.loads((PRIOR / 'run-command.json').read_text())
value = lambda key: command[command.index(key) + 1]
adb = [value('--adb'), '-P', value('--adb-port'), '-s', value('--phone-usb-serial')]
phone_hash = lambda path: 'sha256:' + subprocess.check_output(
    adb + ['shell', 'sha256sum ' + path], text=True, stdin=subprocess.DEVNULL).split()[0]
router = '/data/local/tmp/' + ROOT.name + '/resident-router.android'
assert phone_hash(router) == sha(ROOT / 'resident-router.android')
old = json.loads((PRIOR / 'TRANSPORT_IDENTITY.json').read_text())
affinity_path = ROOT / 'CPU_AFFINITY.json'
affinity = json.loads(affinity_path.read_text())['mask'] if affinity_path.exists() else None
hardware = dict(old['hardware_identity'])
if affinity is not None:
    hardware['phone_cpu_affinity'] = affinity
assert json.loads((ROOT / 'TRANSPORT_RESULT.json').read_text())['status'] == 'PASS'
dependencies = dict(word.split('=', 1) for i, word in enumerate(command)
                    if i and command[i - 1] == '--transport-host-dependency')
identity = build_transport_qualification_identity(
    identity_id=ROOT.name + ':timed-router',
    transport_generation=old['transport_generation'], hardware_identity=hardware,
    phone_session_sha256=phone_hash(value('--phone-session')),
    phone_worker_sha256=phone_hash(value('--phone-worker')),
    qualification_phone_session_sha256=old['software_identity']['qualification_phone_session_sha256'],
    qualification_phone_worker_sha256=old['software_identity']['qualification_phone_worker_sha256'],
    host_binary_path=Path(value('--server')), qualification_binary_path=ROOT / 'ffs_dmabuf_host',
    transport_client_source_path=ROOT / 'source/examples/layersplit/ffn-split-usb-client.cpp',
    qualified_allocators=('devmem',), receipt_paths=tuple(sorted((ROOT / 'transport').glob('*.json'))),
    minimum_usb_speed_mbps=5000,
    host_dependency_paths={key: Path(path) for key, path in dependencies.items()},
    phone_resident_workers_path=PRIOR / 'resident-workers.android',
    phone_resident_router_path=ROOT / 'resident-router.android',
)
save(ROOT / 'TRANSPORT_IDENTITY.json', identity.to_json())
links = materialize_measured_usb_links((ROOT / 'transport',), identity,
    host_device_id='desktop-cpu', phone_device_id='op15-phone')
for name, prior_name in [('CONTEXT_CATALOG.json', 'CONTEXT_CATALOG.json'),
                         ('PRELOAD_CATALOG.json', 'GATE_CATALOG.json')]:
    catalog = json.loads((PRIOR / prior_name).read_text())
    catalog['placement_profile']['links'] = [row for row in catalog['placement_profile']['links']
        if not row.get('transport_generation', '').startswith('functionfs')] + [dataclasses.asdict(row) for row in links]
    resources = {row['resource_id']: row for row in catalog['resources']}
    for link in links:
        key = 'link:' + link.link_id
        resources.setdefault(key, dataclasses.asdict(ResourceProfile(
            resource_id=key, kind='transport', capacity=1, ready=True, identity=link.link_id)))
    catalog['resources'] = [resources[key] for key in sorted(resources)]
    RuntimeCapabilityCatalog.from_json(json.loads(canonical_json(catalog)))
    save(ROOT / name, catalog)
source = {'schema': 's42-prefill-diagnostic-source-v1',
    'prior_source_manifest': str(PRIOR / 'SOURCE_MANIFEST.json'),
    'prior_source_manifest_sha256': sha(PRIOR / 'SOURCE_MANIFEST.json'),
    'router_source_sha256': sha(ROOT / 'source/examples/layersplit/ffn-split-resident-router.cpp'),
    'router_binary_sha256': sha(ROOT / 'resident-router.android'),
    'phone_cpu_affinity': affinity,
    'python_binding_sources': {name: sha(ROOT / 'source/research_dev/scheduler' / name) for name in (
        'adapters/phone_session_contracts/configuration.py',
        'adapters/phone_session_contracts/receipts.py', 'adapters/phone_session_ops/launch.py',
        'campaigns/burstgpt/arguments.py', 'campaigns/burstgpt/runner.py')},
    'desktop_and_worker_binaries_unchanged': True,
    'transport_identity_sha256': identity.identity_sha256}
source['manifest_sha256'] = canonical_sha256(source)
save(ROOT / 'SOURCE_MANIFEST.json', source)
for flag, replacement in {
    '--source-manifest': str(ROOT / 'SOURCE_MANIFEST.json'),
    '--capability-catalog': str(ROOT / 'CONTEXT_CATALOG.json'),
    '--preload-capability-catalog': str(ROOT / 'PRELOAD_CATALOG.json'),
    '--phone-resident-router': router,
    '--phone-session-root': '/data/local/tmp/' + ROOT.name,
    '--phone-remote-hash-cache': str(ROOT / 'PHONE_HASH_CACHE.json'),
    '--usb-qualification-identity': str(ROOT / 'TRANSPORT_IDENTITY.json'),
    '--output': str(ROOT / 'gate-run-v1'),
}.items():
    command[command.index(flag) + 1] = replacement
command[1] = str(ROOT / 'source/research_dev/scheduler/campaigns/burstgpt/remote_resident_gate.py')
if affinity is not None:
    command += ['--phone-cpu-affinity', affinity]
save(ROOT / 'GATE_COMMAND.json', command)
