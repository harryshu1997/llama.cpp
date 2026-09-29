"""Repeat the frozen dev3 gate; add read-only host-activity evidence."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
PREVIOUS = Path('/mnt/storage/s42-layout-economics-20260910-v1')
ROOT = Path('/mnt/storage/s42-layout-economics-20260910-v2-clean-host')
spec = importlib.util.spec_from_file_location('previous_experiment', PREVIOUS / 'EXPERIMENT_FINAL.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
previous.ROOT = ROOT
COMPILERS = {'cc1plus', 'cc1', 'cicc', 'nvcc', 'cmake', 'gmake', 'make', 'ninja'}


def write_observation(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def verify_sources():
    manifest = previous.read(PREVIOUS / 'inputs/SOURCE_MANIFEST_EXECUTION.json')
    mismatches = []
    for row in manifest['files']:
        path = previous.REPO / row['path']
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != row['sha256'].removeprefix('sha256:'):
            mismatches.append(row['path'])
    assert not mismatches, mismatches
    return {'file_count': len(manifest['files']), 'mismatches': mismatches,
            'previous_source_manifest_sha256': previous.file_sha256(
                PREVIOUS / 'inputs/SOURCE_MANIFEST_EXECUTION.json')}


def host_sample():
    processes = {}
    failures = 0
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / 'stat').read_text()
            left, right = stat.index('('), stat.rindex(')')
            fields = stat[right + 2:].split()
            processes[path.name] = {
                'name': stat[left + 1:right], 'ppid': int(fields[1]),
                'cpu_ticks': int(fields[11]) + int(fields[12]),
                'start_ticks': int(fields[19]), 'state': fields[0]}
        except (OSError, ValueError, IndexError):
            failures += 1
    aggregate = [int(value) for value in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
    return {'epoch_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(),
            'cpu_ticks': aggregate, 'processes': processes, 'unreadable_process_count': failures,
            'clock_ticks_per_second': os.sysconf('SC_CLK_TCK'), 'cpu_count': os.cpu_count()}


def activity(before, after):
    duration = (after['monotonic_ns'] - before['monotonic_ns']) / 1e9
    frequency = after['clock_ticks_per_second']
    delta = [b - a for a, b in zip(before['cpu_ticks'], after['cpu_ticks'])]
    busy = sum(delta[:8]) - delta[3] - delta[4]
    rows = []
    for pid, row in after['processes'].items():
        old = before['processes'].get(pid)
        ticks = row['cpu_ticks'] - old['cpu_ticks'] if old and old['start_ticks'] == row['start_ticks'] else 0
        if ticks > 0 or row['name'] in COMPILERS:
            rows.append(dict(pid=int(pid), name=row['name'], ppid=row['ppid'],
                             cpu_seconds=ticks / frequency, cpu_cores=ticks / frequency / duration))
    return {'duration_s': duration, 'busy_core_equivalents': busy / frequency / duration,
            'iowait_core_equivalents': delta[4] / frequency / duration,
            'process_activity': sorted(rows, key=lambda row: (-row['cpu_cores'], row['pid'])),
            'compiler_pids': [int(pid) for pid, row in after['processes'].items() if row['name'] in COMPILERS],
            'caveat': 'Processes ending between samples are absent from per-process deltas; aggregate CPU includes them.'}


def quiet_check(name):
    before = host_sample()
    time.sleep(5)
    after = host_sample()
    summary = activity(before, after)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu,power.draw',
                          '--format=csv,noheader'], check=True, text=True, capture_output=True).stdout.strip()
    summary['gpu'] = gpu
    summary['passed'] = (summary['busy_core_equivalents'] < 1.0 and not summary['compiler_pids']
                         and all(row['cpu_cores'] < 0.25 for row in summary['process_activity']
                                 if row['pid'] != os.getpid()))
    write_observation(ROOT / name, {'before': before, 'after': after, 'summary': summary})
    print('HOST_QUIET_CHECK', json.dumps(summary, sort_keys=True), flush=True)
    assert summary['passed'], 'Host not quiet; no experiment launched'


def prepare():
    verification = verify_sources()
    ROOT.mkdir()
    inputs = ROOT / 'inputs'
    inputs.mkdir()
    rig = previous.read(PREVIOUS / 'inputs/rig.json')
    rig['phone']['session_root'] = '/data/local/tmp/s42-layout-economics-20260910-v2-clean-host'
    rig['phone']['remote_hash_cache_path'] = str(inputs / 'PHONE_HASH_CACHE.json')
    previous.write_new(inputs / 'rig.json', rig)
    previous.write_new(inputs / 'PHONE_HASH_CACHE.json', previous.read(previous.FROZEN / 'PHONE_HASH_CACHE.json'))
    campaign = previous.read(PREVIOUS / 'inputs/campaign.json')
    campaign['rig_manifest_path'] = str(inputs / 'rig.json')
    previous.write_new(inputs / 'campaign.json', campaign)
    cfg = previous.configuration()
    previous.write_new(ROOT / 'RESOLVED_CONFIGURATION.json', cfg.to_json())
    previous.write_new(ROOT / 'SOURCE_REUSE_VERIFICATION.json', verification)
    previous.write_new(ROOT / 'RETEST_SPEC.json', {
        'previous_attempt': str(PREVIOUS), 'deployment': str(previous.REPO),
        'workload': 'burstgpt_dev3_long_v1.json', 'changed_production_files': [],
        'differences': ['fresh artifact paths and phone session namespace', 'read-only host CPU/process activity sampling'],
        'cache_policy': 'unchanged; no flush, prefetch or preload changes',
        'phone_powers_mw': [3000, 4500, 6000], 'baseline_rerun': False, 'longer_trace': False})
    quiet_check('HOST_BEFORE_PREFLIGHT.json')


def monitor_host(stop):
    with (ROOT / 'HOST_ACTIVITY.jsonl').open('x') as output:
        while not stop.is_set():
            output.write(json.dumps(host_sample(), sort_keys=True) + '\n')
            output.flush()
            stop.wait(1)


def run():
    verify_sources()
    quiet_check('HOST_BEFORE_RUN.json')
    stop = threading.Event()
    sampler = threading.Thread(target=monitor_host, args=(stop,), daemon=True)
    sampler.start()
    try:
        previous.run()
    finally:
        stop.set()
        sampler.join(timeout=5)
        previous.write_new(ROOT / 'HOST_AFTER_RUN.json', host_sample())


def record_failure():
    rig = previous.read(ROOT / 'inputs/rig.json')
    phone = rig['phone']
    adb = ['/usr/bin/adb', '-P', str(phone['adb_port']), '-s', phone['serial'], 'shell']
    observed = {}
    for key, command in {'kernel_release': ['uname', '-r'], 'uptime': ['uptime'],
                         'usb_config': ['getprop', 'sys.usb.config']}.items():
        reply = subprocess.run(adb + command, capture_output=True, text=True, timeout=10, check=True)
        observed[key] = reply.stdout.strip()
    log = (ROOT / 'RUN.log').read_text()
    reason = 'phone kernel is not qualified for direct DMA-BUF: ' + observed['kernel_release']
    assert reason in log and not (ROOT / 'run/RESULT.json').exists()
    write_observation(ROOT / 'FAILURE.json', {
        'schema': 'layout-economics-retest-blocker-v1', 'status': 'BLOCKED_PHONE_KERNEL_IDENTITY',
        'observed_at_epoch_ns': time.time_ns(), 'reason': reason,
        'expected_kernel_release': phone['kernel_release'], 'observed_phone': observed,
        'declared_qualified_boot_image_sha256': phone['boot_image_sha256'],
        'hardware_catalog_preflight_status': previous.read(ROOT / 'preflight/PHYSICAL_PREFLIGHT.json')['status'],
        'direct_phone_preflight_status': 'FAILED', 'requests_started': 0, 'phone_shard_loads': 0,
        'paid_interval_started': False, 'fleet_energy_uj': None,
        'source_verification': verify_sources(), 'production_changes': [],
        'preserved_checks': ['exact kernel identity', 'transport qualification', 'memory capacity'],
        'actions_not_taken': ['phone reboot', 'boot image change', 'USB reset', 'process termination',
                              'qualification override', 'baseline rerun', 'longer trace'],
        'setup_note': 'Initial host-activity JSON serialization rejected floats before inference; recorder corrected without changing scheduler code.',
        'required_next_action': 'Restore the previously qualified phone kernel, or explicitly authorize separate qualification of the changed kernel.'})


if __name__ == '__main__':
    {'prepare': prepare, 'quiet': lambda: quiet_check('HOST_BEFORE_PREFLIGHT.json'), 'preflight': previous.preflight,
     'freeze': previous.freeze_execution, 'run': run, 'record-failure': record_failure}[sys.argv[1]]()
