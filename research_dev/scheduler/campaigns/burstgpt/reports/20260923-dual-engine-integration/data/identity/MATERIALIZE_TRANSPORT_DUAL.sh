#!/usr/bin/env bash
# Materialize a NEW transport qualification identity for the S43 dual-engine (NPU+GPU) phone
# worker deployment. The existing TRANSPORT_QUALIFICATION_IDENTITY.json is never touched.
#
# What changes vs the production identity: phone_worker_sha256 (new worker binary) and
# phone_session_sha256 (session script exporting the S43_* environment). Everything else must be
# byte-identical to the production identity and is asserted: hardware identity (kernel, boot image,
# USB), host server binary + host dependencies (cuda-build unchanged), the qualification binary
# (ffs_dmabuf_host), the receipt set, the qualification_phone_* digests (the transport was
# qualified with ffs_dmabuf_phone.android, not with the FFN worker), resident workers and router
# (unchanged production binaries; resident workers fork+exec the worker path they are given).
set -euo pipefail
deploy=/mnt/storage/s42-trace-v2-20260921-prep
exec 9>/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
flock -w 3600 9
cd "$deploy/source"
export PYTHONDONTWRITEBYTECODE=1
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

deploy = Path('/mnt/storage/s42-trace-v2-20260921-prep')
previous = json.loads((deploy / 'TRANSPORT_QUALIFICATION_IDENTITY.json').read_text())
hardware = previous['hardware_identity']
software = previous['software_identity']
output = deploy / 'TRANSPORT_QUALIFICATION_IDENTITY_DUAL.json'
if output.exists():
    raise RuntimeError('dual identity already exists: ' + str(output))
phone_dir = deploy / 'phone-identities-dual'
phone_dir.mkdir(exist_ok=False)
adb = ['/usr/bin/adb', '-P', '5037', '-s', hardware['phone_usb_serial']]
kernel = subprocess.check_output([*adb, 'shell', 'uname', '-r'], text=True).strip()
if kernel != hardware['phone_kernel_release']:
    raise RuntimeError('phone kernel changed since qualification')

DUAL_BIN = '/data/local/tmp/s43-dual-ffn-worker-20260923-v1-bin'
DUAL_SESSION = '/data/local/tmp/s43-dual-session-20260923-v1/direct_phone_ffn_session.sh'
# digests of the bundle as built on the workstation (bundle/SHA256SUMS.txt)
expected = json.loads((deploy / 'phone-bundle-dual/EXPECTED_SHA256.json').read_text())

production = {
    'phone_session': '/data/local/tmp/s42-hal-runtime-probe-20260906-v4/direct_phone_ffn_session.sh',
    'phone_worker': '/data/local/tmp/s42-ffn-shards-20260904-v1-bin/llama-ffn-split-worker',
    'phone_resident_workers': '/data/local/tmp/s42-per-session-correctness-20260903-v2-bin/llama-ffn-split-resident-workers',
    'phone_resident_router': '/data/local/tmp/s42-ready-subset-router-20260905-v2/llama-ffn-split-resident-router',
}
current = {
    'phone_session': DUAL_SESSION,
    'phone_worker': DUAL_BIN + '/llama-ffn-split-worker',
    'phone_resident_workers': production['phone_resident_workers'],
    'phone_resident_router': production['phone_resident_router'],
}


def digest(path):
    with path.open('rb') as stream:
        return 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()


# production stack must be untouched
for name, remote in production.items():
    subprocess.run([*adb, 'pull', remote, str(phone_dir / ('production_' + name))], check=True)
    if digest(phone_dir / ('production_' + name)) != software[name + '_sha256']:
        raise RuntimeError('PRODUCTION phone binary changed: ' + name)
# current (dual) stack: pull, and check against what was pushed
for name, remote in current.items():
    subprocess.run([*adb, 'pull', remote, str(phone_dir / name)], check=True)
current_digest = {name: digest(phone_dir / name) for name in current}
for name, key in (('phone_worker', 'llama-ffn-split-worker'), ('phone_session', 'direct_phone_ffn_session.sh')):
    if current_digest[name] != 'sha256:' + expected[key]:
        raise RuntimeError('pushed dual phone file differs from the built bundle: ' + name)
for name in ('phone_resident_workers', 'phone_resident_router'):
    if current_digest[name] != software[name + '_sha256']:
        raise RuntimeError('resident binary differs from production: ' + name)
if current_digest['phone_worker'] == software['phone_worker_sha256']:
    raise RuntimeError('dual worker is byte-identical to the production worker')
# the worker's shared libraries are not part of the identity schema; record them beside it
library_digests = {}
for library in ('libggml.so', 'libggml-base.so', 'libggml-cpu.so', 'libggml-hexagon.so',
                'libggml-opencl.so', 'libggml-htp-v81.so', 'libomp.so'):
    subprocess.run([*adb, 'pull', DUAL_BIN + '/' + library, str(phone_dir / library)], check=True)
    library_digests[library] = digest(phone_dir / library)
    if library_digests[library] != 'sha256:' + expected[library]:
        raise RuntimeError('pushed dual library differs from the built bundle: ' + library)

qualification = Path('/home/zhihao/s41-ffs-dmabuf-async-v2-20260815/ffs_dmabuf_host')
if digest(qualification) != software['qualification_binary_sha256']:
    raise RuntimeError('qualification binary changed')
receipts = sorted(Path('/home/zhihao/s42-mixed-residency-priority-transport-20260829-v2/receipts').glob('mixed-v6-*.json'))
if sorted(map(digest, receipts)) != sorted(previous['receipt_sha256s']):
    raise RuntimeError('qualification receipt set changed')
host_binary = deploy / 'cuda-build/bin/llama-server'
if digest(host_binary) != software['host_binary_sha256']:
    raise RuntimeError('host server binary changed; server side must stay unchanged')
libraries = ('ggml', 'ggml-base', 'ggml-cpu', 'ggml-cuda', 'llama', 'llama-common', 'llama-server-impl')
for library in libraries:
    if digest(deploy / f'cuda-build/bin/lib{library}.so') != software['host_dependency_sha256:' + library]:
        raise RuntimeError('host dependency changed: ' + library)
client_source = deploy / 'source/examples/layersplit/ffn-split-usb-client.cpp'
if digest(client_source) != software['transport_client_source_sha256']:
    raise RuntimeError('transport client source changed')

command = ['python3', '-m', 'research_dev.scheduler.adapters.materialize_transport_qualification',
           '--identity-id', 's43-dual-engine-20260923-prep',
           '--transport-generation', previous['transport_generation']]
for key, value in hardware.items():
    command += ['--' + key.replace('_', '-'), value]
command += ['--phone-session-sha256', current_digest['phone_session'],
            '--phone-worker-sha256', current_digest['phone_worker'],
            '--qualification-phone-session-sha256', software['qualification_phone_session_sha256'],
            '--qualification-phone-worker-sha256', software['qualification_phone_worker_sha256'],
            '--host-binary', str(host_binary),
            '--qualification-binary', str(qualification),
            '--transport-client-source', str(client_source),
            '--phone-resident-workers', str(phone_dir / 'phone_resident_workers'),
            '--phone-resident-router', str(phone_dir / 'phone_resident_router'),
            '--qualified-allocator', 'devmem', '--minimum-usb-speed-mbps', '5000']
for library in libraries:
    command += ['--host-dependency', f'{library}={deploy}/cuda-build/bin/lib{library}.so']
for receipt in receipts:
    command += ['--receipt', str(receipt)]
command += ['--output', str(output)]
(deploy / 'software/TRANSPORT_PREVIOUS_FOR_DUAL.json').write_text(json.dumps(previous, indent=2) + '\n')
(deploy / 'software/MATERIALIZE_COMMAND_DUAL.json').write_text(json.dumps(command, indent=2) + '\n')
(deploy / 'software/MATERIALIZE_COMMAND_DUAL.txt').write_text(shlex.join(command) + '\n')
(deploy / 'software/DUAL_PHONE_STACK.json').write_text(json.dumps({
    'phone_worker_path': current['phone_worker'],
    'phone_session_path': current['phone_session'],
    'phone_resident_workers_path': current['phone_resident_workers'],
    'phone_resident_router_path': current['phone_resident_router'],
    'digests': current_digest,
    'worker_library_digests': library_digests,
    'production_worker_sha256': software['phone_worker_sha256'],
    'production_session_sha256': software['phone_session_sha256'],
}, indent=2, sort_keys=True) + '\n')
subprocess.run(command, check=True)
PY
