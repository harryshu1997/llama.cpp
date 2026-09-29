"""Preserve raw phone logs after the gate's normal terminal cleanup."""

import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
command = json.loads((ROOT / 'GATE_COMMAND.json').read_text())
value = lambda key: command[command.index(key) + 1]
terminal = json.loads((ROOT / 'gate-run-v1/phone/TERMINAL.json').read_text())
receipt, = [row for row in terminal['phone_receipts'] if 'terminal' in row]
remote = value('--phone-session-root') + '/' + receipt['launch']['session_id']
adb = [value('--adb'), '-P', value('--adb-port'), '-s', value('--phone-usb-serial')]
for name in ('router.log', 'worker.log', 'session.log', 'launch.log'):
    with (ROOT / name).open('xb') as stream:
        subprocess.run(adb + ['shell', 'su -c ' + shlex.quote('cat ' + shlex.quote(remote + '/' + name))],
            stdin=subprocess.DEVNULL, stdout=stream, check=True)
subprocess.run([sys.executable, str(ROOT / 'analyze_timing.py'), str(ROOT / 'gate-run-v1'),
                str(ROOT / 'router.log'), '--output', str(ROOT / 'TIMING.json')], check=True)
