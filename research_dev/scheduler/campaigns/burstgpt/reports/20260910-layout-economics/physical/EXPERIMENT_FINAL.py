"""Configuration and persistence for one normal-scheduling dev3 experiment."""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path('/mnt/storage/s42-layout-economics-20260910-v1')
REPO = Path(str(ROOT) + '-deploy')
FROZEN = Path('/mnt/storage/s42-cuda-graph-v1-20260909/reference-inputs')
NSYS = '/mnt/storage/s21_deps/nsys-2026.3.1/opt/nvidia/nsight-systems-cli/2026.3.1/bin/nsys'
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from research_dev.scheduler.config import load_scheduler_configuration
from research_dev.scheduler.campaigns.burstgpt.launch import (
    _run_streamed, _source_manifest, command_manifest, file_sha256,
    preflight_command, runner_command, write_new,
)


def read(path):
    return json.loads(Path(path).read_text())


def configuration():
    return load_scheduler_configuration(ROOT / 'inputs/campaign.json', environ={})


def prepare():
    ROOT.mkdir()
    inputs = ROOT / 'inputs'
    inputs.mkdir()
    rig = read(FROZEN / 'matched-rig.json')
    rig['repo_root'] = str(REPO)
    rig['binaries']['close_helper'] = str(REPO / 'research_dev/scheduler/adapters/close_resident_bridge.py')
    rig['phone']['session_root'] = '/data/local/tmp/s42-layout-economics-20260910-v1'
    rig['phone']['remote_hash_cache_path'] = str(inputs / 'PHONE_HASH_CACHE.json')
    write_new(inputs / 'PHONE_HASH_CACHE.json', read(FROZEN / 'PHONE_HASH_CACHE.json'))
    write_new(inputs / 'rig.json', rig)
    campaign = read(FROZEN / 'fixed-ggg.json')
    campaign['campaign_id'] = 'layout-economics-adaptive-v1'
    campaign['fixed_phone_residency'] = None
    campaign['rig_manifest_path'] = str(inputs / 'rig.json')
    assert campaign['selection_mode'] == 'energy-aware'
    assert campaign['include_startup_preparation']
    write_new(inputs / 'campaign.json', campaign)
    cfg = configuration()
    catalog = FROZEN / 'matched-CATALOG.json'
    write_new(inputs / 'SOURCE_MANIFEST.json', _source_manifest(cfg))
    command = runner_command(cfg, catalog_path=catalog,
                             source_manifest_path=inputs / 'SOURCE_MANIFEST.json',
                             output_path=ROOT / 'run', execute=True)
    write_new(ROOT / 'RESOLVED_CONFIGURATION.json', cfg.to_json())
    write_new(ROOT / 'COMMAND_MANIFEST.json', command_manifest(cfg, command, catalog))
    write_new(ROOT / 'RUN_COMMAND.json', list(command))
    write_new(ROOT / 'REFERENCE_COMPATIBILITY_INTENT.json', {
        'comparison_kind': 'historical-reference',
        'references_source': 'references-source-v7',
        'catalog_sha256': file_sha256(catalog),
        'initial_evidence': {name: file_sha256(FROZEN / name) for name in (
            'INITIAL_AUTOMATED_OBSERVATIONS.json', 'INITIAL_ADAPTIVE_OBSERVATIONS.json')},
        'differences': ['whole-layout retained-helper revalidation opportunity and cost',
                        'complete comparable-measurement blocks before fraction refinement',
                        'adaptive residency instead of fixed assignment',
                        'fresh artifact and phone session namespace',
                        'read-only 100 ms process-I/O loading diagnostics'],
        'shared_behavior_changed': 'Adaptive controller changes also apply to fixed phone arms.',
        'not_claimed': 'Fresh matched A/B or isolated dynamic-placement superiority.',
    })


def loading_samples(stop):
    """Observe only this attempt's server stdout owners; never alter cache state."""
    tracked = {}
    next_scan = 0.0
    with (ROOT / 'LOADING_PROCESS_SAMPLES.jsonl').open('x') as output:
        while not stop.is_set():
            now = time.monotonic()
            if now >= next_scan:
                for path in Path('/proc').iterdir():
                    if not path.name.isdigit() or path.name in tracked:
                        continue
                    try:
                        stdout = (path / 'fd/1').readlink()
                        if stdout.is_relative_to(ROOT / 'run') and stdout.suffix == '.stdout':
                            tracked[path.name] = str(stdout)
                    except (FileNotFoundError, PermissionError, ProcessLookupError):
                        pass
                next_scan = now + 0.5
            for pid, stdout in tuple(tracked.items()):
                sample = dict(pid=int(pid), stdout=stdout, epoch_ns=time.time_ns(), monotonic_ns=time.monotonic_ns())
                try:
                    proc = Path('/proc') / pid
                    sample['io'] = {key: int(value) for key, value in (
                        line.split(':') for line in (proc / 'io').read_text().splitlines())}
                    sample['stat'] = (proc / 'stat').read_text().strip()
                    sample['status'] = 'VALID'
                except (OSError, ValueError) as exc:
                    sample.update(status='UNAVAILABLE', reason=type(exc).__name__ + ':' + str(exc))
                    tracked.pop(pid, None)
                output.write(json.dumps(sample, sort_keys=True) + '\n')
            output.flush()
            stop.wait(0.1)


def preflight():
    from research_dev.scheduler.adapters import verify_android_usb_restored
    cfg = configuration()
    output = ROOT / 'preflight'
    output.mkdir()
    receipt = verify_android_usb_restored(
        serial=cfg.rig.phone.serial, adb_port=cfg.rig.phone.adb_port,
        minimum_speed_mbps=cfg.rig.phone.minimum_usb_speed_mbps, timeout_s=60)
    write_new(output / 'PHONE_USB_BEFORE.json', receipt.to_json())
    command = preflight_command(cfg, catalog_path=FROZEN / 'matched-CATALOG.json',
                               normal_usb_receipt_path=output / 'PHONE_USB_BEFORE.json',
                               output_path=output / 'PHYSICAL_PREFLIGHT.json')
    write_new(output / 'COMMAND.json', list(command))
    _run_streamed(command, cwd=REPO, log_path=output / 'RUN.log')


def freeze_execution():
    """Keep the preflight inputs immutable while recording final scheduler sources."""
    cfg = configuration()
    source = ROOT / 'inputs/SOURCE_MANIFEST_EXECUTION.json'
    write_new(source, _source_manifest(cfg))
    catalog = FROZEN / 'matched-CATALOG.json'
    command = runner_command(cfg, catalog_path=catalog, source_manifest_path=source,
                             output_path=ROOT / 'run', execute=True)
    write_new(ROOT / 'RUN_COMMAND_EXECUTION.json', list(command))
    write_new(ROOT / 'COMMAND_MANIFEST_EXECUTION.json', command_manifest(cfg, command, catalog))


def run():
    assert read(ROOT / 'preflight/PHYSICAL_PREFLIGHT.json')['status'] == 'PASS'
    assert not (ROOT / 'PROFILE_COMMAND.json').exists(), 'This attempt has already started'
    os.environ.pop('GGML_CUDA_DISABLE_GRAPHS', None)
    command = (NSYS, 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
               '--cuda-graph-trace=graph', '--cuda-trace-all-apis=true',
               '--output', str(ROOT / 'CUDA'), *read(ROOT / 'RUN_COMMAND_EXECUTION.json'))
    write_new(ROOT / 'PROFILE_COMMAND.json', list(command))
    stop = threading.Event()
    sampler = threading.Thread(target=loading_samples, args=(stop,), daemon=True)
    sampler.start()
    try:
        _run_streamed(command, cwd=REPO, log_path=ROOT / 'RUN.log')
    finally:
        stop.set()
        sampler.join(timeout=5)
    subprocess.run([NSYS, 'export', '--type=sqlite', '--output', str(ROOT / 'CUDA.sqlite'),
                    str(ROOT / 'CUDA.nsys-rep')], check=True)


if __name__ == '__main__':
    {'prepare': prepare, 'preflight': preflight, 'freeze': freeze_execution, 'run': run}[sys.argv[1]]()
