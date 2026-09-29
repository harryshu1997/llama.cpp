## Dispatch, loads, pairs, probe reasons

| arm | dispatch_policy statistics | bypass counts | displacements (notes) | REPLAN reasons | wake reasons |
| --- | --- | --- | --- | --- | --- |
| desktop_dispatch | {"affinity_displaced_attempts": 0, "affinity_displacements": 0, "affinity_refusals": 0, "early_capacity_promotions": 1, "publication_replans": 4, "published_work_promotions": 2} | {} | 0 | {"DESKTOP_BASELINE_CONTROL": 11} | {"replanned": 7, "residency_observation_changed": 4, "capacity_released_early": 3, "calendar_elapsed": 2, "residency_transition_completed": 2, "predecessor_completion": 1, "predecessor_replan": 1} |
| allon | {"affinity_displaced_attempts": 0, "affinity_displacements": 0, "affinity_refusals": 0, "early_capacity_promotions": 2, "publication_replans": 1, "published_work_promotions": 7} | {} | 0 | {"QUALIFIED_FALLBACK_NO_SAFE_ALTERNATIVE": 20, "READY_DESKTOP_HELPER_RUNTIME_UNAVAILABLE": 3} | {"capacity_released_early": 10, "replanned": 8, "residency_transition_completed": 3, "calendar_elapsed": 1, "residency_observation_changed": 1, "preparation_phase_completed": 1} |

| arm | large-model launches (hot/cold/control) | load s by role | total load s | model_reload_count / transition_count | execution sequence | switches |
| --- | --- | --- | ---: | --- | --- | ---: |
| desktop_dispatch | {"cold": 3, "control": 1, "hot": 2} | {"hot": [46.7, 80.3], "cold": [3.4, 37.0, 35.8], "control": [0.3]} | 203.5 | 6 / 6 | gemma -> qwen -> gemma -> qwen -> gemma | 4 |
| allon | {"cold": 3, "control": 1, "hot": 2} | {"hot": [48.8, 54.7], "cold": [22.1, 38.2, 46.4], "control": [0.3]} | 210.5 | 6 / 6 | gemma -> qwen -> gemma -> qwen -> gemma | 4 |

| arm | request | model | arrival s | acquired s | exec start s | exec end s | load wait s | attempts | wake reasons |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| desktop_dispatch | 000 | gemma | 1.0 | 2.0 | 7.7 | 73.2 | 5.7 | 1 | calendar_elapsed |
| desktop_dispatch | 001 | gemma | 1.0 | 7.6 | 7.7 | 31.7 | 0.1 | 2 | residency_observation_changed,replanned |
| desktop_dispatch | 00 | llama | 3.8 | 73.2 | 75.3 | 75.4 | 2.1 | 3 | capacity_released_early,residency_transition_completed,replanned |
| desktop_dispatch | 002 | qwen | 79.0 | 79.5 | 129.7 | 211.1 | 50.2 | 1 | calendar_elapsed |
| desktop_dispatch | 003 | qwen | 86.2 | 134.0 | 134.1 | 216.5 | 0.1 | 2 | residency_observation_changed,replanned |
| desktop_dispatch | 004 | qwen | 99.0 | 129.2 | 129.7 | 273.3 | 0.5 | 2 | residency_observation_changed,replanned |
| desktop_dispatch | 005 | gemma | 109.4 | 273.4 | 317.2 | 625.6 | 43.8 | 2 | predecessor_completion,replanned |
| desktop_dispatch | 006 | qwen | 111.8 | 625.7 | 715.4 | 804.7 | 89.7 | 4 | residency_observation_changed,predecessor_replan,capacity_released_early,replanned |
| desktop_dispatch | 007 | gemma | 127.8 | 804.9 | 847.2 | 863.6 | 42.3 | 3 | capacity_released_early,residency_transition_completed,replanned |
| allon | 000 | gemma | 1.0 | 5.7 | 32.3 | 94.1 | 26.6 | 1 | calendar_elapsed |
| allon | 001 | gemma | 1.0 | 32.0 | 32.3 | 57.6 | 0.3 | 2 | residency_observation_changed,replanned |
| allon | 00 | llama | 3.8 | 94.2 | 102.5 | 102.8 | 8.2 | 3 | capacity_released_early,residency_transition_completed,replanned |
| allon | 002 | qwen | 79.0 | 102.8 | 156.0 | 228.5 | 53.2 | 3 | capacity_released_early,residency_transition_completed,replanned |
| allon | 003 | qwen | 86.2 | 154.7 | 155.6 | 232.3 | 0.9 | 3 | capacity_released_early,capacity_released_early,replanned |
| allon | 004 | qwen | 99.0 | 155.8 | 156.5 | 280.5 | 0.7 | 2 | capacity_released_early,replanned |
| allon | 005 | gemma | 109.4 | 280.8 | 325.9 | 593.8 | 45.1 | 3 | capacity_released_early,capacity_released_early,replanned |
| allon | 006 | qwen | 111.8 | 594.2 | 659.1 | 735.2 | 64.9 | 3 | capacity_released_early,capacity_released_early,replanned |
| allon | 007 | gemma | 127.8 | 735.4 | 787.1 | 804.4 | 51.7 | 4 | capacity_released_early,preparation_phase_completed,residency_transition_completed,replanned |

| arm | execution pairs (same model) | overlap s by model | decode-window overlaps | window tokens at batch>=2 | active_slots_peak / slots max |
| --- | --- | --- | --- | --- | --- |
| desktop_dispatch | gemma 000+001 24.0, qwen 002+004 81.4, qwen 002+003 77.1, qwen 004+003 82.5 | {"gemma": 24.0, "qwen": 241.0} | - | {} | 3 / 3 |
| allon | gemma 000+001 25.2, qwen 003+002 72.5, qwen 003+004 75.9, qwen 002+004 72.0 | {"gemma": 25.2, "qwen": 220.4} | gemma 000+001 14.1, qwen 003+004 67.9, qwen 003+002 64.3, qwen 004+002 64.3 | {"gemma": 60, "qwen": 355} | 1 / 1 |

| arm | server_policy reason by model (decisions) | per-batch reasons (last seen) | zero_assistance_reason by model | raw |
| --- | --- | --- | --- | --- |
| desktop_dispatch | {} | {} | {} | {"RESULT.json": {}, "SCHEDULER_DECISION_LOG.json": {}, "ADAPTIVE_DECODE_OBSERVATIONS.json": {}} |
| allon | {"gemma|SERVER_COMPARISON_HOST_WINDOW": 3, "qwen|SERVER_COMPARISON_HOST_WINDOW": 3} | {"gemma|1|SERVER_COMPARISON_HOST_WINDOW": 3, "qwen|1|SERVER_COMPARISON_HOST_WINDOW": 3} | {} | {"RESULT.json": {"SERVER_COMPARISON_HOST_WINDOW": 12, "SERVER_POLICY_COHERENCE": 324}, "SCHEDULER_DECISION_LOG.json": {"SERVER_COMPARISON_HOST_WINDOW": 12, "SERVER_POLICY_COHERENCE": 324}, "ADAPTIVE_DECODE_OBSERVATIONS.json": {}} |

