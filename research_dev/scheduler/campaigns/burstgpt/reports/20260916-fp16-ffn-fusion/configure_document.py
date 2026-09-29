"""Bind the existing bounded gate to paired backend binaries and an explicit switch."""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PRIOR = Path('/mnt/storage/s42-prefill-affinity-20260916-v2-nuxeVS')
PHONE = '/data/local/tmp/s42-fp16-fusion-20260916-v4'
sys.path.insert(0, str(PRIOR / 'source'))
from research_dev.scheduler.adapters.transport_profiles import (
    build_transport_qualification_identity, materialize_measured_usb_links,
)
from research_dev.scheduler import ResourceProfile, RuntimeCapabilityCatalog
from research_dev.scheduler._internal.types import canonical_json, canonical_sha256


def save(path, value):
    with path.open('x') as stream:
        stream.write(canonical_json(value) + '\n')


def sha(path):
    with path.open('rb') as stream:
        return 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('disabled', 'enabled'), required=True)
args = parser.parse_args()
output = ROOT / ('document-' + args.mode)
output.mkdir()
command = json.loads((PRIOR / 'GATE_COMMAND.json').read_text())
value = lambda key: command[command.index(key) + 1]
adb = [value('--adb'), '-P', value('--adb-port'), '-s', value('--phone-usb-serial')]


def phone_hash(path):
    return 'sha256:' + subprocess.check_output(
        adb + ['shell', 'sha256sum ' + path], text=True, stdin=subprocess.DEVNULL).split()[0]


bundle = {path.name: sha(path) for path in sorted((ROOT / 'bundle-v4').iterdir())}
for name, digest in bundle.items():
    assert phone_hash(PHONE + '/' + name) == digest, name
session = PHONE + '/session-fusion-' + args.mode + '.sh'
assert phone_hash(session) == sha(ROOT / ('session-fusion-' + args.mode + '.sh'))
old = json.loads((PRIOR / 'TRANSPORT_IDENTITY.json').read_text())
assert phone_hash(value('--phone-session')) == old['software_identity']['phone_session_sha256']
dependencies = dict(word.split('=', 1) for i, word in enumerate(command)
                    if i and command[i - 1] == '--transport-host-dependency')
identity = build_transport_qualification_identity(
    identity_id=ROOT.name + ':ffn-fusion-' + args.mode,
    transport_generation=old['transport_generation'], hardware_identity=old['hardware_identity'],
    phone_session_sha256=phone_hash(session),
    phone_worker_sha256=bundle['llama-ffn-split-worker'],
    qualification_phone_session_sha256=old['software_identity']['qualification_phone_session_sha256'],
    qualification_phone_worker_sha256=old['software_identity']['qualification_phone_worker_sha256'],
    host_binary_path=Path(value('--server')), qualification_binary_path=PRIOR / 'ffs_dmabuf_host',
    transport_client_source_path=PRIOR / 'source/examples/layersplit/ffn-split-usb-client.cpp',
    qualified_allocators=('devmem',), receipt_paths=tuple(sorted((PRIOR / 'transport').glob('*.json'))),
    minimum_usb_speed_mbps=5000,
    host_dependency_paths={key: Path(path) for key, path in dependencies.items()},
    phone_resident_workers_path=Path('/mnt/storage/s42-ffn-microbatch-20260916-v4-arwuw3/resident-workers.android'),
    phone_resident_router_path=PRIOR / 'resident-router.android',
)
save(output / 'TRANSPORT_IDENTITY.json', identity.to_json())
links = materialize_measured_usb_links((PRIOR / 'transport',), identity,
    host_device_id='desktop-cpu', phone_device_id='op15-phone')
for name in ('CONTEXT_CATALOG.json', 'PRELOAD_CATALOG.json'):
    catalog = json.loads((PRIOR / name).read_text())
    catalog['placement_profile']['links'] = [row for row in catalog['placement_profile']['links']
        if not row.get('transport_generation', '').startswith('functionfs')] + [dataclasses.asdict(row) for row in links]
    resources = {row['resource_id']: row for row in catalog['resources']}
    for link in links:
        key = 'link:' + link.link_id
        resources.setdefault(key, dataclasses.asdict(ResourceProfile(
            resource_id=key, kind='transport', capacity=1, ready=True, identity=link.link_id)))
    catalog['resources'] = [resources[key] for key in sorted(resources)]
    RuntimeCapabilityCatalog.from_json(json.loads(canonical_json(catalog)))
    save(output / name, catalog)
manifest = ROOT / 'SOURCE_MANIFEST.json'
if not manifest.exists():
    source = {'schema': 's42-fp16-fused-source-v1',
        'scheduler_source_manifest': str(PRIOR / 'SOURCE_MANIFEST.json'),
        'scheduler_source_manifest_sha256': sha(PRIOR / 'SOURCE_MANIFEST.json'),
        'scheduler_source_root': str(PRIOR / 'source'),
        'native_sources': {str(path.relative_to(ROOT / 'source-final')): sha(path)
            for path in sorted((ROOT / 'source-final').rglob('*')) if path.is_file()},
        'android_bundle': bundle,
        'session_wrappers': {mode: sha(ROOT / ('session-fusion-' + mode + '.sh'))
                            for mode in ('disabled', 'enabled')},
        'wrapped_session_sha256': old['software_identity']['phone_session_sha256'],
        'desktop_binary_sha256': sha(Path(value('--server'))),
        'desktop_source_and_binaries_unchanged': True}
    source['manifest_sha256'] = canonical_sha256(source)
    save(manifest, source)
for flag, replacement in {
    '--source-manifest': str(manifest),
    '--capability-catalog': str(output / 'CONTEXT_CATALOG.json'),
    '--preload-capability-catalog': str(output / 'PRELOAD_CATALOG.json'),
    '--phone-session': session,
    '--phone-worker': PHONE + '/llama-ffn-split-worker',
    '--phone-session-root': PHONE + '/document-' + args.mode,
    '--phone-remote-hash-cache': str(output / 'PHONE_HASH_CACHE.json'),
    '--usb-qualification-identity': str(output / 'TRANSPORT_IDENTITY.json'),
    '--output': str(output / 'gate-run-v1'),
}.items():
    command[command.index(flag) + 1] = replacement
save(output / 'GATE_COMMAND.json', command)
save(output / 'EXPERIMENT.json', {
    'fusion': args.mode, 'GGML_HEXAGON_OPFUSION': 1 if args.mode == 'enabled' else 0,
    'profile': 0, 'hvx_threads': 4, 'phone_cpu_affinity': value('--phone-cpu-affinity'),
    'prompt_sha256': sha(Path(value('--prompt-file'))), 'output_tokens': 64,
    'transport_receipts_reused_from': str(PRIOR / 'transport'),
    'transport_unchanged': True, 'desktop_unchanged': True,
    'measurement': 'Compare reduced request boundaries across modes; full and reduced arm boundaries differ.',
    'source_manifest_sha256': sha(manifest),
})
save(output / 'TRANSPORT_BOOT.json', json.loads((PRIOR / 'TRANSPORT_BOOT.json').read_text()))
print(output)
