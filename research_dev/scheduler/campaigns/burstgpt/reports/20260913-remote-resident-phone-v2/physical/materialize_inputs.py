"""Freeze this bounded experiment's newly measured transport and launch inputs."""
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/mnt/storage/s42-remote-resident-phone-20260913-v2-WTUd4V')
sys.path.insert(0, str(ROOT / 'source'))
from research_dev.scheduler.adapters.transport_profiles import (
    build_transport_qualification_identity, materialize_measured_usb_links,
)
from research_dev.scheduler._internal.types import canonical_json

command = json.loads((ROOT / 'PREVIOUS_COMMAND.json').read_text())
def value(flag):
    return command[command.index(flag) + 1]

def save(path, record):
    with path.open('x') as stream:
        stream.write(canonical_json(record) + '\n')

def digest(path):
    return 'sha256:' + hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()

adb = [value('--adb'), '-P', value('--adb-port'), '-s', value('--phone-usb-serial')]
def phone_hash(path):
    output = subprocess.check_output(adb + ['shell', 'sha256sum ' + path], text=True)
    return 'sha256:' + output.split()[0]

workers = ROOT / 'resident-workers.android'
router = ROOT / 'resident-router.android'
for remote, local in ((value('--phone-resident-workers'), workers),
                      (value('--phone-resident-router'), router)):
    if local.exists():
        raise RuntimeError('refusing to overwrite native evidence')
    subprocess.run(adb + ['pull', remote, str(local)], check=True)
    assert digest(local) == phone_hash(remote)
old = json.loads(Path(value('--usb-qualification-identity')).read_text())
dependencies = dict(word.split('=', 1) for index, word in enumerate(command)
                    if index and command[index - 1] == '--transport-host-dependency')
identity = build_transport_qualification_identity(
    identity_id=ROOT.name + ':fresh-transport',
    transport_generation=old['transport_generation'],
    hardware_identity=old['hardware_identity'],
    phone_session_sha256=phone_hash(value('--phone-session')),
    phone_worker_sha256=phone_hash(value('--phone-worker')),
    qualification_phone_session_sha256=phone_hash('/data/local/tmp/s41-ffs-dmabuf-v1/phone_gadget_session.sh'),
    qualification_phone_worker_sha256=phone_hash('/data/local/tmp/s41-ffs-dmabuf-v1/ffs_dmabuf_phone.android'),
    host_binary_path=Path(value('--server')),
    qualification_binary_path=ROOT / 'ffs_dmabuf_host',
    transport_client_source_path=ROOT / 'source/examples/layersplit/ffn-split-usb-client.cpp',
    qualified_allocators=('devmem',),
    receipt_paths=tuple(sorted((ROOT / 'transport-v2b').glob('*.json'))),
    minimum_usb_speed_mbps=5000,
    host_dependency_paths={name: Path(path) for name, path in dependencies.items()},
    phone_resident_workers_path=workers, phone_resident_router_path=router,
)
save(ROOT / 'TRANSPORT_IDENTITY.json', identity.to_json())
catalog = json.loads(Path(value('--capability-catalog')).read_text())
links = materialize_measured_usb_links((ROOT / 'transport-v2b',), identity,
                                     host_device_id='desktop-cpu', phone_device_id='op15-phone')
catalog['placement_profile']['links'] = [
    row for row in catalog['placement_profile']['links']
    if not row.get('transport_generation', '').startswith('functionfs')
] + [dataclasses.asdict(row) for row in links]
artifact = json.loads(Path(value('--gemma-manifest')).read_text())['artifact_sha256']
modified = []
for row in catalog['composite_executors']:
    if row['artifact_sha256'] == artifact and row['assisted_operator_kind'] == 'ffn':
        parameters = row['adapter_parameters']
        parameters['ffn_max_tokens'] = parameters['ubatch_size']
        parameters['memory_workspace_minimum_bytes:op15-phone'] = 805306368
        modified.append(row['executor_id'])
save(ROOT / 'GATE_CATALOG.json', catalog)
save(ROOT / 'INPUT_DIFF.json', {
    'source_catalog': value('--capability-catalog'),
    'changed_helper_launch_contracts': modified,
    'qualification_note': 'Fresh USB measurements only; prefill FFN execution remains unqualified until the physical gate.',
    'workspace_note': '768 MiB conservative reservation, not a measured allocation.',
    'usb_identity_sha256': identity.identity_sha256,
})
updates = {
    '--capability-catalog': str(ROOT / 'GATE_CATALOG.json'),
    '--usb-qualification-identity': str(ROOT / 'TRANSPORT_IDENTITY.json'),
    '--phone-session-root': '/data/local/tmp/' + ROOT.name,
    '--phone-remote-hash-cache': str(ROOT / 'PHONE_HASH_CACHE.json'),
    '--capacity-context-sizes': '',
    '--output': str(ROOT / 'gate-v1'),
}
for flag, replacement in updates.items():
    command[command.index(flag) + 1] = replacement
command[1] = str(ROOT / 'source/research_dev/scheduler/campaigns/burstgpt/remote_resident_gate.py')
command.remove('--preflight-only')
save(ROOT / 'GATE_COMMAND.json', command)
print(identity.identity_sha256)
