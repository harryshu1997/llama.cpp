"""Recheck an already successful calibration's transient GPU-idle postflight."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, '/mnt/storage/s42-dmabuf-cancel-20260914-v1-gBJdFx/validator-v2')
from src.run_direct_usb_device import execution_lock, preflight

result = json.loads((ROOT / 'calibration-v1/RESULT.json').read_text())
assert result['status'] == 'PASS'
with execution_lock():
    path = ROOT / 'calibrate-postflight-recheck'
    path.mkdir()
    boot = preflight(path)
    assert boot == json.loads((ROOT / 'TRANSPORT_BOOT.json').read_text())['candidate']['boot_id']
    with (ROOT / 'CALIBRATION_POSTFLIGHT_RECOVERY.json').open('x') as stream:
        json.dump({'calibration': result, 'postflight': 'PASS', 'boot_id': boot,
            'prior_failure': 'target GPU is not idle: NVIDIA GeForce RTX 4060 Ti, 2',
            'calibration_rerun': False, 'idle_threshold_unchanged': True}, stream, indent=2)
        stream.write('\n')
