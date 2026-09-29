"""Persist artifact integrity and the failed, confounded performance decision."""

import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[5]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def compilers(path):
    database = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
    rows = dict(database.execute("select name,count(*) from PROCESSES "
                                "where name in ('cc1plus','cicc','nvcc','cmake','gmake') group by name"))
    database.close()
    return rows


def main():
    comparison = json.loads((ROOT / 'COMPARISON.json').read_text())
    result = json.loads((ROOT / 'physical/run/RESULT.json').read_text())
    previous = ROOT.parent / '20260910-context-continuity/physical'
    before = json.loads((previous / 'run/RESULT.json').read_text())
    write('GATE_DECISION.json', {
        'schema': 'layout-economics-small-gate-v1',
        'execution_status': result['status'], 'request_count': result['counts']['terminals'],
        'performance_status': 'FAIL_CONFOUNDED', 'longer_trace_authorized': False,
        'energy_saving_target_percent': 25,
        'comparison_kind': 'historical-reference; unrelated compiler activity also differs',
        'nominal_raw_fleet_energy_uj': comparison['phone_power_sensitivity']['4500']['fleet_energy_uj'],
        'nominal_raw_saving_vs_matched_percent': comparison['frozen_references']['desktop-matched-cuda']['adaptive_saving_percent']['4500'],
        'nominal_raw_saving_vs_fixed_ggg_percent': comparison['frozen_references']['fixed-ggg']['adaptive_saving_percent']['4500'],
        'compiler_processes_in_capture': compilers(ROOT / 'physical/CUDA.sqlite'),
        'previous_compiler_processes_in_capture': compilers(previous / 'CUDA.sqlite'),
        'fixed_ggg_compiler_processes_in_capture': compilers(
            REPO / 'research_dev/scheduler/baselines/cuda_graph_v1/references-source-v7/fixed-ggg/CUDA.sqlite'),
        'previous_energy_uj_by_domain': before['trace_energy']['fleet_energy_uj_by_domain'],
        'current_energy_uj_by_domain': result['trace_energy']['fleet_energy_uj_by_domain'],
        'interpretation': [
            'Raw RAPL package energy includes unrelated CPU work. No energy is subtracted.',
            'Compiler activity is observed, but its exact joules are not isolated.',
            'The Gemma retained-layout tail fix is exercised; energy superiority is not established.',
            'Qwen still has incomplete probing and same-batch membership resets.',
            'Do not promote this result, rerun baselines, or launch a longer trace.',
        ],
        'tests': {'passed': 303, 'duration_s': 87.849, 'full_harness_run': False},
        'unchanged_replay_goldens': {
            'v3': 'ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d',
            'v8': '965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d'},
        'result_sha256': digest(ROOT / 'physical/run/RESULT.json'),
    })
    files = {str(path.relative_to(ROOT / 'physical')): {
        'sha256': digest(path), 'bytes': path.stat().st_size}
        for path in sorted((ROOT / 'physical').rglob('*')) if path.is_file()}
    write('ARTIFACTS.json', {'local_root': str(ROOT / 'physical'),
                           'remote_root': '/mnt/storage/s42-layout-economics-20260910-v1',
                           'file_count': len(files), 'files': files})
    backup = Path('/tmp/s42-layout-economics.BEN85g')
    changes = {}
    for old in sorted(backup.rglob('*.py')):
        relative = old.relative_to(backup)
        current = REPO / relative
        if current.is_file() and digest(old) != digest(current):
            changes[str(relative)] = {'before_sha256': digest(old), 'after_sha256': digest(current),
                                     'before_lines': len(old.read_text().splitlines()),
                                     'after_lines': len(current.read_text().splitlines())}
    new_test = Path('research_dev/scheduler/tests/test_layout_revalidation_economics.py')
    changes[str(new_test)] = {'before_sha256': None, 'after_sha256': digest(REPO / new_test),
                              'before_lines': 0, 'after_lines': len((REPO / new_test).read_text().splitlines())}
    write('CODE_CHANGES.json', changes)
    print('REPORT_INTEGRITY_WRITTEN', len(files), 'physical files;', len(changes), 'code/test files')


if __name__ == '__main__':
    main()
