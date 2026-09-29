"""Historical-reference reporting; never relax the matched A/B validator."""

from collections import Counter
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from research_dev.scheduler.campaigns.burstgpt.compare_ab import (
    ComparisonError, canonical, cuda_graph_evidence, energy_domains, file_sha256,
    frozen_reference_identity, load_adaptive_windows, load_object, phone_preflight_identity,
    reference_assistance_summary, reference_phone_power_sensitivity, request_rows,
    require, require_clean_execution, result_counts, validate_matched,
)


def differences(before, after, prefix=''):
    if isinstance(before, dict) and isinstance(after, dict):
        return [row for key in sorted(before.keys() | after.keys())
                for row in differences(before.get(key), after.get(key), prefix + '/' + key)]
    return [] if before == after else [{'field': prefix, 'reference': before, 'adaptive': after}]


def main():
    result_path = Path(sys.argv[1])
    output = Path(sys.argv[2])
    frozen = REPO / 'research_dev/scheduler/baselines/cuda_graph_v1'
    comparison = load_object(frozen / 'COMPARISON.json')
    result = load_object(result_path)
    require(result['status'] == 'PASS', 'adaptive execution did not pass')
    require_clean_execution(result, 'adaptive')
    replay = load_object(REPO / 'research_dev/scheduler/campaigns/burstgpt/data/traces/burstgpt_dev3_long_v1.json')
    result_counts(result, replay)
    identity = frozen_reference_identity(result)
    references = {}
    compatibility = {}
    power = reference_phone_power_sensitivity(result)
    for name, summary in comparison['references'].items():
        path = frozen / 'references-source-v7' / name / 'run/RESULT.json'
        require(file_sha256(path) == summary['result_sha256'], 'frozen reference changed: ' + name)
        reference = load_object(path)
        expected = frozen_reference_identity(reference)
        diff = differences(expected, identity)
        compatibility[name] = {'decoded_differences': diff}
        for field in ('workload', 'model_artifacts', 'model_roles', 'trace_identity',
                      'replay_schedule', 'initial_observation_inputs', 'adaptive_controller_configuration',
                      'preparation_accounting', 'energy_boundary', 'maximum_latency_ppm'):
            require(identity[field] == expected[field], 'unrelated comparison mismatch: ' + field)
        if name != 'desktop-default-cuda':
            require(identity['catalog_sha256'] == expected['catalog_sha256'], 'matched catalog differs')
            require(identity['desktop_parents'] == expected['desktop_parents'], 'matched desktop parent differs')
            require(identity['execution_identity']['binaries'] == expected['execution_identity']['binaries'],
                    'matched execution binaries differ')
        require(phone_preflight_identity(result_path) == phone_preflight_identity(path),
                'phone artifacts or worker/transport identity differ')
        references[name] = summary
        references[name]['adaptive_saving_percent'] = {
            p: 100 * (1 - power[p]['fleet_energy_uj'] / summary['phone_power_sensitivity'][p]['fleet_energy_uj'])
            for p in power}
    matched = load_object(frozen / 'references-source-v7/desktop-matched-cuda/run/RESULT.json')
    try:
        validate_matched(matched, result, expected_replay=replay)
    except ComparisonError as error:
        matched_rejection = str(error)
    else:
        raise AssertionError('The historical reference must not be labeled a fresh matched A/B')
    requests = []
    windows = load_adaptive_windows(result_path, result)
    for row in request_rows(result):
        events = [event for event in result['request_helper_events']
                  if event.get('request_id') == row['request_id']]
        receipt = row['terminal_ticket']['execution_receipt']
        requests.append({
            'request_id': row['request_id'], 'model_id': row['model_id'],
            'service_latency_us': row['actual_latency_us'], 'arrival_us': row['replay_arrival_us'],
            'start_us': receipt['started_us'], 'completion_us': receipt['finished_us'],
            'phone_call_count': row['physical_execution_proof']['phone_call_count'],
            'helper_event_counts': dict(Counter(event['kind'] for event in events)),
            'zero_policy_decisions': [event for event in events
                                     if event['kind'] == 'ASSISTANCE_DECISION'
                                     and event.get('selected_fraction_ppm') == 0],
            'retained_evidence_rebinds': [event for event in events
                                        if event.get('retained_evidence_layers_by_plan')],
            'zero_policy_windows': [window for window in windows
                                    if window['request_id'] == row['request_id']
                                    and window['policy']['split_fraction_ppm'] == 0],
        })
    summary = {
        'schema': 'retained-helper-historical-reference-comparison-v1',
        'comparison_kind': 'historical-reference; not a fresh matched A/B',
        'matched_validator_rejection': matched_rejection,
        'frozen_comparison_sha256': file_sha256(frozen / 'COMPARISON.json'),
        'adaptive_result_sha256': file_sha256(result_path),
        'duration_us': result['duration_us'], 'energy_uj_by_domain': energy_domains(result),
        'phone_power_sensitivity': power, 'requests': requests,
        'assistance': reference_assistance_summary(result_path, result),
        'cuda_graphs': cuda_graph_evidence(result_path.parent.parent / 'CUDA.sqlite'),
        'physical_residency': result['phone_residency_at_completion'],
        'session_events': result['phone_residency_events'],
        'session_physical_phases': result['phone_residency_phase_events'],
        'session_call_timestamps': result['phone_residency_call_events'],
        'compatibility': compatibility, 'frozen_references': references,
        'scope': 'One adaptive run and six immutable source-v7 historical references; preparation and cleanup included.',
    }
    with output.open('xb') as stream:
        stream.write(canonical(summary))
    print('HISTORICAL_REFERENCE_REPORT', output, file_sha256(output))


if __name__ == '__main__':
    main()
