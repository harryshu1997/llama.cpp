import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['launch', 'status', 'fetch'])
parser.add_argument('root', type=Path)
args = parser.parse_args()
root = args.root.resolve()
b = root.parents[2]
host = 'zhihao@172.20.74.85'
adb = ['adb', '-P', '5037', '-s', '5A040DLCH004ES']
config = json.loads((root / 'CONFIG.json').read_text())
phone = config['phone_dir']
remote = '/mnt/storage/s42-' + root.parent.name + '-20260923'
commands = []
def run(command):
    commands.append(command)
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    return result.stdout.strip()
def remote_adb(*command):
    return run(['ssh', host, shlex.join([*adb, *command])])
if args.action == 'launch':
    stage = root.parent / 'stage'
    stage.mkdir()
    for name in ['EXPECTED_HASHES.sha256', 'REQUESTS.bin', 'RUN_PHONE.sh']:
        shutil.copy2(root / name, stage / name)
    shutil.copy2(b / 'software/pixel10pro-dense-gemv-ratio-v1/libggml-vulkan.so', stage / 'libggml-vulkan.so')
    shutil.copy2(b / 'software/pixel10pro-cpu-gpu-v3/llama-ffn-split-worker', stage / 'llama-ffn-split-worker-dual')
    shutil.copy2(b / 'software/pixel10pro-cpu-tune-v3/llama-ffn-split-worker', stage / 'llama-ffn-split-worker')
    for subdir, source, name in [
        ('cpu-affinity', 'pixel10pro-cpu-tune-v3', 'libggml-cpu.so'),
        ('previous', 'pixel10pro-dense-gemv-v1', 'libggml-vulkan.so')]:
        (stage / subdir).mkdir()
        shutil.copy2(b / 'software' / source / name, stage / subdir / name)
    run(['ssh', host, shlex.join(['mkdir', remote])])
    run(['rsync', '-rlptc', str(stage) + '/', host + ':' + remote + '/'])
    remote_adb('shell', shlex.join(['mkdir', phone]))
    for source in sorted(stage.iterdir()):
        remote_adb('push', remote + '/' + source.name, phone + '/' + source.name)
    remote_adb('shell', shlex.join(['chmod', '755', phone + '/llama-ffn-split-worker', phone + '/llama-ffn-split-worker-dual']))
    script = 'sh ' + shlex.quote(phone + '/RUN_PHONE.sh') + ' > ' + shlex.quote(phone + '/RUN.log') + ' 2>&1; result=$?; echo "$result" > ' + shlex.quote(phone + '/EXIT.txt') + '; exit "$result"'
    launch = 'nohup sh -c ' + shlex.quote(script) + ' </dev/null >/dev/null 2>&1 & echo $!'
    pid = remote_adb('shell', launch)
    (root / 'LAUNCH.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(), 'pid':pid, 'remote':remote, 'phone':phone, 'commands':commands},indent=2)+'\n')
    shutil.copy2(__file__, root / 'TRANSPORT_SCRIPT.py')
elif args.action == 'status':
    remote_adb('shell', 'cat ' + shlex.quote(phone + '/RUN.log') + '; if test -f ' + shlex.quote(phone + '/EXIT.txt') + '; then echo EXIT; cat ' + shlex.quote(phone + '/EXIT.txt') + '; fi')
else:
    assert remote_adb('shell', 'cat ' + shlex.quote(phone + '/EXIT.txt')) == '0'
    run(['ssh', host, shlex.join(['mkdir', remote + '/results'])])
    for name in ['raw', 'RUN.log', 'EXIT.txt']:
        remote_adb('pull', phone + '/' + name, remote + '/results/' + name)
    run(['rsync', '-rlptc', host + ':' + remote + '/results/', str(root) + '/'])
    (root / 'FETCH.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(), 'commands':commands},indent=2)+'\n')
