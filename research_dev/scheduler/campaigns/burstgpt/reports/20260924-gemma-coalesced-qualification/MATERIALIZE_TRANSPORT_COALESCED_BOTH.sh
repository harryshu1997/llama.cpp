#!/usr/bin/env bash
# Materialize a NEW transport qualification identity that covers coalesced multi-row phone FFN calls
# of BOTH assisted models: the 2026-09-22 task1 receipts (7,680 / 10,240 / 40,960 bytes, depth 4) plus
# the 2026-09-24 coalesced-both receipts (38,400 / 61,440 bytes, depth 4). Production identity files
# (TRANSPORT_QUALIFICATION_IDENTITY.json, *_DUAL.json) and the coherent arm's identity are never touched.
#
# Everything except identity id, output path and the receipt set is byte-identical to the coherent arm's
# identity (s42-trace-longtaildev-coherent-20260924) and is asserted: hardware identity (candidate boot
# f13c7c03, kernel, USB), the deploy host server + host dependencies, the qualifier binary
# (stage ffs_dmabuf_host 61bc2907, the one that produced the task1 and the new receipts), the qualification
# phone session/worker digests, the production phone stack (session, worker, resident workers, router).
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
previous_path = Path('/home/zhihao/s42-trace-longtaildev-coherent-20260924-inputs/TRANSPORT_QUALIFICATION_IDENTITY.json')
previous = json.loads(previous_path.read_text())
production = json.loads((deploy / 'TRANSPORT_QUALIFICATION_IDENTITY.json').read_text())
hardware = previous['hardware_identity']
software = previous['software_identity']
output = deploy / 'TRANSPORT_QUALIFICATION_IDENTITY_COALESCED_BOTH.json'
if output.exists():
    raise RuntimeError('coalesced-both identity already exists: ' + str(output))
phone_dir = deploy / 'phone-identities-coalesced-both'
phone_dir.mkdir(exist_ok=False)
adb = ['/usr/bin/adb', '-P', '5037', '-s', hardware['phone_usb_serial']]


def digest(path):
    with path.open('rb') as stream:
        return 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()


def shell(command):
    return subprocess.run([*adb, 'shell', '-n', 'su -c ' + shlex.quote(command)], check=True,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout.replace('\r', '').strip()


# phone: kernel + candidate boot identity unchanged since the receipts were measured
if shell('uname -r') != hardware['phone_kernel_release']:
    raise RuntimeError('phone kernel changed since qualification')
notes = [row.split()[0] for row in shell('sha256sum /sys/kernel/notes /sys/kernel/btf/vmlinux').splitlines()]
candidate = json.loads(Path('/mnt/storage/s42-dmabuf-cancel-20260914-v1-gBJdFx/CANDIDATE_BOOT_RESULT.json').read_text())
if notes != [candidate['identity']['notes'], candidate['identity']['btf']]:
    raise RuntimeError('phone kernel notes/BTF differ from the candidate boot identity')
if 'sha256:' + candidate['image_sha256'] != hardware['phone_boot_image_sha256']:
    raise RuntimeError('candidate boot image digest differs from the identity')

# production phone stack unchanged (pulled to a new directory, never modified on the phone)
production_files = {
    'phone_session': '/data/local/tmp/s42-hal-runtime-probe-20260906-v4/direct_phone_ffn_session.sh',
    'phone_worker': '/data/local/tmp/s42-ffn-shards-20260904-v1-bin/llama-ffn-split-worker',
    'phone_resident_workers': '/data/local/tmp/s42-per-session-correctness-20260903-v2-bin/llama-ffn-split-resident-workers',
    'phone_resident_router': '/data/local/tmp/s42-ready-subset-router-20260905-v2/llama-ffn-split-resident-router',
}
for name, remote in production_files.items():
    subprocess.run([*adb, 'pull', remote, str(phone_dir / name)], check=True, stdin=subprocess.DEVNULL)
    if digest(phone_dir / name) != software[name + '_sha256']:
        raise RuntimeError('PRODUCTION phone binary changed: ' + name)
    if software[name + '_sha256'] != production['software_identity'][name + '_sha256']:
        raise RuntimeError('coherent and production identities disagree on ' + name)

# qualification stack: the binary and phone-side scripts that produced the task1 AND the new receipts
qualification = Path('/mnt/storage/s42-ffn-microbatch-20260916-v4-arwuw3/ffs_dmabuf_host')
if digest(qualification) != software['qualification_binary_sha256']:
    raise RuntimeError('qualification binary changed')
remote = {row.split()[1]: 'sha256:' + row.split()[0] for row in shell(
    'sha256sum /data/local/tmp/s42-rrphone-20260914-v3b/functionfs_transport_session.sh'
    ' /data/local/tmp/s41-ffs-dmabuf-v1/ffs_dmabuf_phone.android'
    ' /data/local/tmp/s43-transport-qual-20260924/ffs_dmabuf_phone.android').splitlines()}
if remote['/data/local/tmp/s42-rrphone-20260914-v3b/functionfs_transport_session.sh'] != software['qualification_phone_session_sha256']:
    raise RuntimeError('qualification phone session script changed')
for path in ('/data/local/tmp/s41-ffs-dmabuf-v1/ffs_dmabuf_phone.android',
             '/data/local/tmp/s43-transport-qual-20260924/ffs_dmabuf_phone.android'):
    if remote[path] != software['qualification_phone_worker_sha256']:
        raise RuntimeError('qualification phone worker differs: ' + path)

# receipts: the 9 task1 receipts bound by the coherent identity + the 6 new coalesced-both receipts
task1 = sorted(Path('/mnt/storage/s42-task1-transport-20260922-v2/receipts').glob('task1-20260922-*.json'))
if sorted(map(digest, task1)) != sorted(previous['receipt_sha256s']):
    raise RuntimeError('task1 receipt set changed')
new_dir = Path('/home/zhihao/s43-transport-receipts-61440-20260924/receipts')
new = sorted(new_dir.glob('coalesced-both-20260924-*.json'))
if len(new) != 6:
    raise RuntimeError('expected 6 new receipts, found %d' % len(new))
result = json.loads(Path('/home/zhihao/s43-transport-receipts-61440-20260924/RESULT.json').read_text())
if result.get('status') != 'PASS' or result.get('cases') != 6 or result.get('kernel_changed_by_this_test'):
    raise RuntimeError('coalesced-both qualification did not pass: ' + json.dumps(result)[:400])
receipts = task1 + new

# host stack unchanged
host_binary = deploy / 'cuda-build/bin/llama-server'
if digest(host_binary) != software['host_binary_sha256']:
    raise RuntimeError('host server binary changed')
libraries = ('ggml', 'ggml-base', 'ggml-cpu', 'ggml-cuda', 'llama', 'llama-common', 'llama-server-impl')
for library in libraries:
    if digest(deploy / f'cuda-build/bin/lib{library}.so') != software['host_dependency_sha256:' + library]:
        raise RuntimeError('host dependency changed: ' + library)
client_source = deploy / 'source/examples/layersplit/ffn-split-usb-client.cpp'
if digest(client_source) != software['transport_client_source_sha256']:
    raise RuntimeError('transport client source changed')

command = ['python3', '-m', 'research_dev.scheduler.adapters.materialize_transport_qualification',
           '--identity-id', 's43-coalesced-both-20260924',
           '--transport-generation', previous['transport_generation']]
for key, value in hardware.items():
    command += ['--' + key.replace('_', '-'), value]
for key in ('phone_session_sha256', 'phone_worker_sha256',
            'qualification_phone_session_sha256', 'qualification_phone_worker_sha256'):
    command += ['--' + key.replace('_', '-'), software[key]]
command += ['--host-binary', str(host_binary),
            '--qualification-binary', str(qualification),
            '--transport-client-source', str(client_source),
            '--phone-resident-workers', str(phone_dir / 'phone_resident_workers'),
            '--phone-resident-router', str(phone_dir / 'phone_resident_router'),
            '--qualified-allocator', 'devmem', '--minimum-usb-speed-mbps', str(previous['minimum_usb_speed_mbps'])]
for library in libraries:
    command += ['--host-dependency', f'{library}={deploy}/cuda-build/bin/lib{library}.so']
for receipt in receipts:
    command += ['--receipt', str(receipt)]
command += ['--output', str(output)]
(deploy / 'software/TRANSPORT_PREVIOUS_FOR_COALESCED_BOTH.json').write_text(json.dumps(previous, indent=2) + '\n')
(deploy / 'software/MATERIALIZE_COMMAND_COALESCED_BOTH.json').write_text(json.dumps(command, indent=2) + '\n')
(deploy / 'software/MATERIALIZE_COMMAND_COALESCED_BOTH.txt').write_text(shlex.join(command) + '\n')
subprocess.run(command, check=True, stdin=subprocess.DEVNULL)
identity = json.loads(output.read_text())
print('identity', output, 'receipts', len(identity['receipt_sha256s']))
PY
