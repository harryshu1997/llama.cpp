"""Repeat the unchanged dev3 run after restoring its qualified phone kernel."""

import importlib.util
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path('/mnt/storage/s42-layout-economics-20260910-v3-restored-host')
SOURCE = Path('/mnt/storage/s42-layout-economics-20260910-v2-clean-host/EXPERIMENT_FINAL.py')
spec = importlib.util.spec_from_file_location('quiet_retest', SOURCE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.ROOT = ROOT
base.previous.ROOT = ROOT


def prepare():
    verification = base.verify_sources()
    rig = base.previous.read(base.PREVIOUS / 'inputs/rig.json')
    phone = rig['phone']
    kernel = subprocess.check_output([
        rig['binaries']['adb'], '-P', str(phone['adb_port']), '-s', phone['serial'],
        'shell', 'uname', '-r'], text=True, timeout=10).strip()
    assert kernel == phone['kernel_release'], 'Qualified phone kernel is not running'
    ROOT.mkdir()
    inputs = ROOT / 'inputs'
    inputs.mkdir()
    phone['session_root'] = '/data/local/tmp/' + ROOT.name
    phone['remote_hash_cache_path'] = str(inputs / 'PHONE_HASH_CACHE.json')
    base.previous.write_new(inputs / 'rig.json', rig)
    base.previous.write_new(inputs / 'PHONE_HASH_CACHE.json', base.previous.read(
        base.previous.FROZEN / 'PHONE_HASH_CACHE.json'))
    campaign = base.previous.read(base.PREVIOUS / 'inputs/campaign.json')
    campaign['rig_manifest_path'] = str(inputs / 'rig.json')
    base.previous.write_new(inputs / 'campaign.json', campaign)
    base.previous.write_new(ROOT / 'RESOLVED_CONFIGURATION.json', base.previous.configuration().to_json())
    base.previous.write_new(ROOT / 'SOURCE_REUSE_VERIFICATION.json', verification)
    base.previous.write_new(ROOT / 'PHONE_KERNEL_BEFORE.json', {'kernel_release': kernel})
    base.previous.write_new(ROOT / 'RETEST_SPEC.json', {
        'previous_attempt': str(base.PREVIOUS), 'deployment': str(base.previous.REPO),
        'blocked_attempt': '/mnt/storage/s42-layout-economics-20260910-v2-clean-host',
        'kernel_restoration': '/mnt/storage/s42-phone-kernel-restore-20260910-v1',
        'workload': 'burstgpt_dev3_long_v1.json', 'production_changes': [],
        'differences': ['fresh artifact/session namespaces', 'read-only host activity sampling'],
        'cache_policy': 'unchanged; no cache flush or prefetch',
        'phone_powers_mw': [3000, 4500, 6000], 'baseline_rerun': False, 'longer_trace': False})
    base.quiet_check('HOST_BEFORE_PREFLIGHT.json')


if __name__ == '__main__':
    {'prepare': prepare, 'preflight': base.previous.preflight,
     'freeze': base.previous.freeze_execution, 'run': base.run}[sys.argv[1]]()
