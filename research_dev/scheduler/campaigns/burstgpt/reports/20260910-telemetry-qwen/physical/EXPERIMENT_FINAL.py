"""One dev3 retest with frozen inputs and read-only loading/host observations."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path('/mnt/storage/s42-telemetry-qwen-20260910-v1')
REPO = Path(str(ROOT) + '-deploy')
os.chdir(REPO)
sys.path.insert(0, str(REPO))

# Bind canonical modules to this deployment before reusing the measurement wrapper.
from research_dev.scheduler.config import load_scheduler_configuration
from research_dev.scheduler.campaigns.burstgpt.launch import _source_manifest

SOURCE = Path('/mnt/storage/s42-layout-economics-20260910-v2-clean-host/EXPERIMENT_FINAL.py')
spec = importlib.util.spec_from_file_location('quiet_retest', SOURCE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.ROOT = base.previous.ROOT = ROOT
base.previous.REPO = REPO
os.chdir(REPO)


def verify_sources():
    path = REPO / 'DEPLOYMENT_SHA256_EXECUTION.json'
    if not path.exists():
        path = REPO / 'DEPLOYMENT_SHA256.json'
    manifest = base.previous.read(path)
    mismatches = [path for path, expected in manifest.items()
                  if hashlib.sha256((REPO / path).read_bytes()).hexdigest() != expected]
    assert not mismatches, mismatches
    return {'file_count': len(manifest), 'mismatches': mismatches, 'manifest_path': str(path)}


base.verify_sources = verify_sources


def prepare():
    verification = verify_sources()
    rig = base.previous.read(base.PREVIOUS / 'inputs/rig.json')
    phone = rig['phone']
    kernel = subprocess.check_output([
        rig['binaries']['adb'], '-P', str(phone['adb_port']), '-s', phone['serial'],
        'shell', 'uname', '-r'], text=True, timeout=10).strip()
    assert kernel == phone['kernel_release'], 'Qualified phone kernel is not running'
    ROOT.mkdir()
    inputs = ROOT / 'inputs'
    inputs.mkdir()
    rig['repo_root'] = str(REPO)
    rig['binaries']['close_helper'] = str(REPO / 'research_dev/scheduler/adapters/close_resident_bridge.py')
    phone['session_root'] = '/data/local/tmp/' + ROOT.name
    phone['remote_hash_cache_path'] = str(inputs / 'PHONE_HASH_CACHE.json')
    base.previous.write_new(inputs / 'rig.json', rig)
    base.previous.write_new(inputs / 'PHONE_HASH_CACHE.json', base.previous.read(
        base.previous.FROZEN / 'PHONE_HASH_CACHE.json'))
    campaign = base.previous.read(base.PREVIOUS / 'inputs/campaign.json')
    campaign['rig_manifest_path'] = str(inputs / 'rig.json')
    assert campaign['fixed_phone_residency'] is None
    assert campaign['selection_mode'] == 'energy-aware'
    assert campaign['include_startup_preparation']
    base.previous.write_new(inputs / 'campaign.json', campaign)
    cfg = base.previous.configuration()
    base.previous.write_new(ROOT / 'RESOLVED_CONFIGURATION.json', cfg.to_json())
    base.previous.write_new(ROOT / 'SOURCE_DEPLOYMENT_VERIFICATION.json', verification)
    base.previous.write_new(ROOT / 'PHONE_KERNEL_BEFORE.json', {'kernel_release': kernel})
    base.previous.write_new(ROOT / 'RETEST_SPEC.json', {
        'previous_successful_attempt': str(base.PREVIOUS), 'deployment': str(REPO),
        'previous_failed_attempt': '/mnt/storage/s42-layout-economics-20260910-v3-restored-host',
        'workload': 'burstgpt_dev3_long_v1.json',
        'source_changes': base.previous.read(REPO / 'SOURCE_DELTA.json'),
        'cache_policy': 'unchanged; no cache flush or prefetch',
        'phone_powers_mw': [3000, 4500, 6000], 'baseline_rerun': False, 'longer_trace': False,
        'comparison_kind': 'historical-reference', 'references_source': 'references-source-v7',
        'shared_behavior_changed': 'Probe lifecycle changes also apply to fixed phone arms.',
        'not_claimed': 'Fresh matched A/B or isolated dynamic-placement superiority.',
    })
    base.previous.write_new(inputs / 'SOURCE_MANIFEST.json', _source_manifest(cfg))
    base.quiet_check('HOST_BEFORE_PREFLIGHT.json')


def freeze():
    base.previous.write_new(ROOT / 'SOURCE_DELTA_EXECUTION.json', base.previous.read(
        REPO / 'SOURCE_DELTA_EXECUTION.json'))
    base.previous.freeze_execution()


if __name__ == '__main__':
    {'prepare': prepare, 'preflight': base.previous.preflight,
     'freeze': freeze, 'run': base.run}[sys.argv[1]]()
