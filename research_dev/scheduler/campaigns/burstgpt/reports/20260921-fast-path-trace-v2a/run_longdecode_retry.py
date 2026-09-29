#!/usr/bin/env python3
"""Run the unchanged treatment after correcting the FFN proof log parser."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


SOURCE = Path('/mnt/storage/s42-trace-v2-20260921-prep/source')
BASE = Path('/home/zhihao/s42-trace-longdecode-treatment-r2-20260922-inputs')
LOCK = '/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock'


def main():
    if '--locked' not in sys.argv:
        return subprocess.run(['flock', '-w', '900', LOCK, 'python3', '-u',
                               str(Path(__file__).resolve()), '--locked'],
                              stdin=subprocess.DEVNULL).returncode
    os.environ.update(LANG='C.UTF-8', S42_UNIFIED_REPO_ROOT=str(SOURCE),
                      LD_LIBRARY_PATH='/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64')
    for stage in ('preflight', 'run'):
        if (BASE / 'CANCEL').exists():
            return 1
        output = BASE / ('preflight-1' if stage == 'preflight' else 'run-treatment-1')
        if output.exists():
            raise RuntimeError(f'output already exists: {output}')
        command = ['python3', 'research_dev/scheduler/campaigns/burstgpt/launch.py',
                   str(BASE / 'campaign.json'), str(output)]
        if stage == 'preflight':
            command.append('--preflight-only')
        print(json.dumps({'at': datetime.now(timezone.utc).isoformat(),
                          'stage': stage, 'status': 'STARTED'}), flush=True)
        with (BASE / f'{stage.upper()}.log').open('x') as log:
            code = subprocess.run(command, cwd=SOURCE, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        (BASE / f'{stage.upper()}_EXIT.txt').write_text(str(code) + '\n')
        print(json.dumps({'at': datetime.now(timezone.utc).isoformat(),
                          'stage': stage, 'exit_code': code}), flush=True)
        if code:
            return code
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
