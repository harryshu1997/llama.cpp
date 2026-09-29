"""Exclusive launch and same-boot postflight of one canonical bounded gate."""

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, '/mnt/storage/s42-dmabuf-cancel-20260914-v1-gBJdFx/validator-v2')
from src.run_direct_usb_device import execution_lock, preflight, shell


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


mode = sys.argv[1]
assert mode in {'preflight', 'run', 'calibrate'}
command = json.loads((ROOT / ('CALIBRATE_COMMAND.json' if mode == 'calibrate' else 'GATE_COMMAND.json')).read_text())
if mode == 'preflight':
    command[command.index('--output') + 1] = str(ROOT / 'gate-preflight-v1')
    command += ['--preflight-only']
candidate = json.loads((ROOT / 'TRANSPORT_BOOT.json').read_text())['candidate']
with execution_lock():
    before = ROOT / (mode + '-idle-check')
    before.mkdir()
    boot = preflight(before)
    assert boot == candidate['boot_id']
    actual = shell('sha256sum /sys/kernel/notes /sys/kernel/btf/vmlinux').splitlines()
    assert [row.split()[0] for row in actual] == [candidate['identity']['notes'], candidate['identity']['btf']]
    save(ROOT / (mode + '-command.json'), command)
    started = time.time_ns()
    with (ROOT / (mode.upper() + '.log')).open('x') as log:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    for attempt in range(3):
        after = ROOT / (mode + '-postflight' + ('' if attempt == 0 else '-retry-' + str(attempt)))
        after.mkdir()
        try:
            status = 'PASS' if preflight(after) == boot else 'FAIL'
            break
        except RuntimeError as error:
            save(after / 'FAILURE.json', {'error': str(error)})
            if not str(error).startswith('target GPU is not idle:') or attempt == 2:
                raise
            time.sleep(2)
    save(ROOT / (mode + '-launch.json'), {'started_epoch_ns': started,
        'finished_epoch_ns': time.time_ns(), 'returncode': result.returncode, 'postflight': status})
    print(mode, result.returncode, status, flush=True)
    raise SystemExit(result.returncode or (0 if status == 'PASS' else 1))
