"""Persist the retest's observed energy, host activity, and artifact identities."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / '20260910-layout-economics'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def host_activity(result):
    samples = [json.loads(line) for line in (ROOT / 'physical/HOST_ACTIVITY.jsonl').read_text().splitlines()]
    samples = [row for row in samples if result['paid_start_ns'] <= row['monotonic_ns'] <= result['paid_end_ns']]
    totals = defaultdict(float)
    names = {}
    max_cores = 0
    compilers = {}
    known = {'cc1plus', 'cc1', 'cicc', 'nvcc', 'cmake', 'gmake', 'make', 'ninja'}
    for sample in samples:
        for pid, row in sample['processes'].items():
            if row['name'] in known:
                compilers[pid] = row['name']
    for before, after in zip(samples, samples[1:]):
        seconds = (after['monotonic_ns'] - before['monotonic_ns']) / 1e9
        frequency = after['clock_ticks_per_second']
        delta = [b - a for a, b in zip(before['cpu_ticks'], after['cpu_ticks'])]
        max_cores = max(max_cores, (sum(delta[:8]) - delta[3] - delta[4]) / frequency / seconds)
        for pid, row in after['processes'].items():
            old = before['processes'].get(pid)
            if old is not None and row['start_ticks'] == old['start_ticks']:
                identity = (pid, row['start_ticks'])
                totals[identity] += max(0, row['cpu_ticks'] - old['cpu_ticks']) / frequency
                names[identity] = row['name']
    database = sqlite3.connect('file:' + str(ROOT / 'physical/CUDA.sqlite') + '?mode=ro', uri=True)
    captured = dict(database.execute("select name,count(*) from PROCESSES where name in "
                                    "('cc1plus','cc1','cicc','nvcc','cmake','gmake','make','ninja') group by name"))
    database.close()
    return {'paid_sample_count': len(samples), 'maximum_observed_busy_cores': max_cores,
            'compiler_pids_in_periodic_samples': compilers, 'compiler_processes_in_cuda_capture': captured,
            'before_preflight': read(ROOT / 'physical/HOST_BEFORE_PREFLIGHT.json')['summary'],
            'before_run': read(ROOT / 'physical/HOST_BEFORE_RUN.json')['summary'],
            'top_process_cpu_seconds': [dict(pid=int(key[0]), start_ticks=key[1], name=names[key], cpu_seconds=value)
                                        for key, value in sorted(totals.items(), key=lambda item: -item[1])[:25]],
            'caveat': 'Per-process totals omit processes that start and exit between samples; no joules are subtracted.'}


def artifact_index():
    files = {str(path.relative_to(ROOT / 'physical')): {'sha256': digest(path), 'bytes': path.stat().st_size}
             for path in sorted((ROOT / 'physical').rglob('*')) if path.is_file()}
    write('ARTIFACTS.json', {'file_count': len(files), 'files': files,
                           'remote_root': '/mnt/storage/s42-layout-economics-20260910-v2-clean-host'})
    print('RETEST_SUMMARY_WRITTEN', len(files), 'artifacts')


def main():
    if not (ROOT / 'physical/run/RESULT.json').exists():
        failure = read(ROOT / 'physical/FAILURE.json')
        failure['host_before_run'] = read(ROOT / 'physical/HOST_BEFORE_RUN.json')['summary']
        failure['failure_sha256'] = digest(ROOT / 'physical/FAILURE.json')
        write('RETEST_SUMMARY.json', failure)
        artifact_index()
        return
    result = read(ROOT / 'physical/run/RESULT.json')
    comparison = read(ROOT / 'COMPARISON.json')
    previous = read(PREVIOUS / 'COMPARISON.json')
    activity = host_activity(result)
    current_manifest = read(ROOT / 'physical/inputs/SOURCE_MANIFEST_EXECUTION.json')
    previous_manifest = read(PREVIOUS / 'physical/inputs/SOURCE_MANIFEST_EXECUTION.json')
    assert current_manifest['files'] == previous_manifest['files'], 'Retest source files differ'
    write('RETEST_SUMMARY.json', {
        'schema': 'layout-economics-clean-host-retest-v1', 'execution_status': result['status'],
        'counts': result['counts'], 'duration_us': result['duration_us'],
        'previous_duration_us': previous['duration_us'], 'unchanged_source_files': len(current_manifest['files']),
        'production_changes': [], 'host_activity': activity,
        'current_energy_uj_by_domain': comparison['energy_uj_by_domain'],
        'previous_energy_uj_by_domain': previous['energy_uj_by_domain'],
        'phone_power_sensitivity': comparison['phone_power_sensitivity'],
        'raw_saving_vs_frozen_references_percent': {
            key: value['adaptive_saving_percent'] for key, value in comparison['frozen_references'].items()},
        'comparison_kind': comparison['comparison_kind'],
        'strict_matched_validator_rejection': comparison['matched_validator_rejection'],
        'assistance': comparison['assistance'], 'cuda_graphs': comparison['cuda_graphs'],
        'layout_event_counts': dict(Counter(row['kind'] for row in result['phone_residency_events'])),
        'load_count_by_session': result['phone_residency_at_completion']['load_count_by_session'],
        'result_sha256': digest(ROOT / 'physical/run/RESULT.json'), 'longer_trace_started': False,
        'tests': 'No suites rerun; identical 378-file deployment previously passed 303 focused/replay tests.',
    })
    artifact_index()


if __name__ == '__main__':
    main()
