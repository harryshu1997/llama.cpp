#!/usr/bin/env python3
"""Run the prepared, unfiltered real-window pair on the desktop after M4a5."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SOURCE = Path('/mnt/storage/s42-trace-v2-20260921-prep/source')
PREVIOUS = Path('/home/zhihao/s42-trace-v2a-m4a5-coalesced-20260922-inputs/run-treatment-1/run')
LOCK = '/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock'
PAIR = Path('/home/zhihao/s42-trace-longdecode-pair-20260922')
ARMS = ('baseline', 'treatment')


def note(**values):
    print(json.dumps({'at': datetime.now(timezone.utc).isoformat(), **values}), flush=True)


def main():
    if '--locked' not in sys.argv:
        while not (PREVIOUS / 'RESULT.json').exists():
            if (PAIR / 'CANCEL').exists() or (PREVIOUS / 'FAILURE.json').exists():
                note(status='STOPPED', reason='cancelled or previous physical failure needs review')
                return 1
            time.sleep(10)
        return subprocess.run(['flock', '-w', '900', LOCK, 'python3', str(Path(__file__).resolve()),
                               '--locked'], stdin=subprocess.DEVNULL).returncode
    os.environ.update(LANG='C.UTF-8', S42_UNIFIED_REPO_ROOT=str(SOURCE),
                      LD_LIBRARY_PATH='/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64')
    results = {}
    for arm in ARMS:
        base = Path(f'/home/zhihao/s42-trace-longdecode-{arm}-20260922-inputs')
        campaign = json.loads((base / 'campaign.json').read_text())
        assert not campaign.get('adaptive_decode_overrides', {}).get('server_policy_coherence', False)
        for stage in ('preflight', 'run'):
            if (PAIR / 'CANCEL').exists():
                note(status='STOPPED', arm=arm, stage=stage)
                return 1
            output = base / ('preflight-1' if stage == 'preflight' else f'run-{arm}-1')
            if output.exists():
                raise RuntimeError(f'output already exists: {output}')
            command = ['python3', 'research_dev/scheduler/campaigns/burstgpt/launch.py',
                       str(base / 'campaign.json'), str(output)]
            if stage == 'preflight':
                command.append('--preflight-only')
            note(arm=arm, stage=stage, status='STARTED')
            with (base / f'{stage.upper()}.log').open('x') as log:
                code = subprocess.run(command, cwd=SOURCE, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT).returncode
            (base / f'{stage.upper()}_EXIT.txt').write_text(str(code) + '\n')
            note(arm=arm, stage=stage, exit_code=code)
            if code:
                return code
        result = json.loads((output / 'run/RESULT.json').read_text())
        results[arm] = {'status': result['status'], 'result': str(output / 'run/RESULT.json')}
        if result['status'] != 'PASS':
            return 1
    with (PAIR / 'PAIR_RESULT.json').open('x') as stream:
        json.dump({'status': 'PASS', 'arms': results}, stream, indent=2)
        stream.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
