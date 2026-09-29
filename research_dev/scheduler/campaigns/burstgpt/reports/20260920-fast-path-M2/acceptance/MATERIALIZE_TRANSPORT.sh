#!/usr/bin/env bash
set -euo pipefail
deploy=/mnt/storage/s42-fast-path-M2-acceptance-20260921-c09ea6
exec 9>/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
flock -n 9
cd "$deploy/native-source"
export PYTHONDONTWRITEBYTECODE=1
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

deploy = Path('/mnt/storage/s42-fast-path-M2-acceptance-20260921-c09ea6')
previous = json.loads(Path('/home/zhihao/s42-dormant-trace-20260917-v1-inputs/TRANSPORT_QUALIFICATION_IDENTITY.json').read_text())
hardware = previous['hardware_identity']
software = previous['software_identity']
phone_dir = deploy / 'phone-identities'
phone_dir.mkdir(exist_ok=False)
adb = ['/usr/bin/adb', '-P', '5037', '-s', hardware['phone_usb_serial']]
kernel = subprocess.check_output([*adb, 'shell', 'uname', '-r'], text=True).strip()
if kernel != hardware['phone_kernel_release']:
    raise RuntimeError('phone kernel changed since qualification')
files = {
    'phone_session': '/data/local/tmp/s42-hal-runtime-probe-20260906-v4/direct_phone_ffn_session.sh',
    'phone_worker': '/data/local/tmp/s42-ffn-shards-20260904-v1-bin/llama-ffn-split-worker',
    'phone_resident_workers': '/data/local/tmp/s42-per-session-correctness-20260903-v2-bin/llama-ffn-split-resident-workers',
    'phone_resident_router': '/data/local/tmp/s42-ready-subset-router-20260905-v2/llama-ffn-split-resident-router',
}
def digest(path):
    with path.open('rb') as stream:
        return 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()
for name, remote in files.items():
    subprocess.run([*adb, 'pull', remote, str(phone_dir / name)], check=True)
    if digest(phone_dir / name) != software[name + '_sha256']:
        raise RuntimeError('phone binary changed since qualification: ' + name)
qualification = Path('/home/zhihao/s41-ffs-dmabuf-async-v2-20260815/ffs_dmabuf_host')
if digest(qualification) != software['qualification_binary_sha256']:
    raise RuntimeError('qualification binary changed')
receipts = sorted(Path('/home/zhihao/s42-mixed-residency-priority-transport-20260829-v2/receipts').glob('mixed-v6-*.json'))
if sorted(map(digest, receipts)) != sorted(previous['receipt_sha256s']):
    raise RuntimeError('qualification receipt set changed')
command = ['python3', '-m', 'research_dev.scheduler.adapters.materialize_transport_qualification',
           '--identity-id', 's42-fast-path-M2-acceptance-20260921-c09ea6',
           '--transport-generation', previous['transport_generation']]
for key, value in hardware.items():
    command += ['--' + key.replace('_', '-'), value]
for key in ('phone_session_sha256', 'phone_worker_sha256',
            'qualification_phone_session_sha256', 'qualification_phone_worker_sha256'):
    command += ['--' + key.replace('_', '-'), software[key]]
command += ['--host-binary', str(deploy / 'cuda-build/bin/llama-server'),
            '--qualification-binary', str(qualification),
            '--transport-client-source', str(deploy / 'native-source/examples/layersplit/ffn-split-usb-client.cpp'),
            '--phone-resident-workers', str(phone_dir / 'phone_resident_workers'),
            '--phone-resident-router', str(phone_dir / 'phone_resident_router'),
            '--qualified-allocator', 'devmem', '--minimum-usb-speed-mbps', '5000']
for library in ('ggml', 'ggml-base', 'ggml-cpu', 'ggml-cuda', 'llama', 'llama-common', 'llama-server-impl'):
    command += ['--host-dependency', f'{library}={deploy}/cuda-build/bin/lib{library}.so']
for receipt in receipts:
    command += ['--receipt', str(receipt)]
command += ['--output', str(deploy / 'TRANSPORT_QUALIFICATION_IDENTITY.json')]
(deploy / 'software/TRANSPORT_PREVIOUS.json').write_text(json.dumps(previous, indent=2) + '\n')
(deploy / 'software/MATERIALIZE_COMMAND.json').write_text(json.dumps(command, indent=2) + '\n')
(deploy / 'software/MATERIALIZE_COMMAND.txt').write_text(shlex.join(command) + '\n')
subprocess.run(command, check=True)
PY
