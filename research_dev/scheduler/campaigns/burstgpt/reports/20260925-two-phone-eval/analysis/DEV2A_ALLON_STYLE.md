## Dispatch, loads, pairs, probe reasons

| arm | dispatch_policy statistics | bypass counts | displacements (notes) | REPLAN reasons | wake reasons |
| --- | --- | --- | --- | --- | --- |
| desktop | {"affinity_displaced_attempts": 9, "affinity_displacements": 2, "affinity_refusals": 0, "early_capacity_promotions": 3, "publication_replans": 0, "published_work_promotions": 8} | {"burstgpt_longtail_dev_v2:002": 2, "burstgpt_longtail_dev_v2:003": 2, "burstgpt_longtail_dev_v2:004": 2, "burstgpt_longtail_dev_v2:006": 1, "burstgpt_longtail_dev_v2:llama-3.2-1b-overlay:00": 2} | 4 | {"DESKTOP_BASELINE_CONTROL": 24} | {"model_affinity_displaced": 9, "capacity_released_early": 8, "replanned": 7, "predecessor_replan": 3, "calendar_elapsed": 2, "preparation_phase_completed": 1, "residency_transition_completed": 1} |
| op15_r1 | {"affinity_displaced_attempts": 2, "affinity_displacements": 1, "affinity_refusals": 0, "early_capacity_promotions": 2, "publication_replans": 1, "published_work_promotions": 6} | {"burstgpt_longtail_dev_v2:005": 1, "burstgpt_longtail_dev_v2:007": 1} | 1 | {"QUALIFIED_FALLBACK_NO_SAFE_ALTERNATIVE": 16, "READY_DESKTOP_HELPER_RUNTIME_UNAVAILABLE": 2} | {"replanned": 8, "capacity_released_early": 6, "preparation_phase_completed": 3, "residency_transition_completed": 2, "calendar_elapsed": 1, "residency_observation_changed": 1, "model_affinity_displaced": 1} |
| twophone_r1 | {"affinity_displaced_attempts": 2, "affinity_displacements": 1, "affinity_refusals": 0, "early_capacity_promotions": 2, "publication_replans": 1, "published_work_promotions": 2} | {"burstgpt_longtail_dev_v2:005": 1, "burstgpt_longtail_dev_v2:007": 1} | 3 | {"QUALIFIED_FALLBACK_NO_SAFE_ALTERNATIVE": 12} | {"replanned": 7, "capacity_released_early": 3, "calendar_elapsed": 2, "preparation_phase_completed": 2, "residency_observation_changed": 1, "residency_transition_completed": 1, "model_affinity_displaced": 1} |

| arm | large-model launches (hot/cold/control) | load s by role | total load s | model_reload_count / transition_count | execution sequence | switches |
| --- | --- | --- | ---: | --- | --- | ---: |
| desktop | {"cold": 1, "control": 1, "hot": 1} | {"hot": [44.9], "cold": [67.7], "control": [0.4]} | 113.0 | 3 / 3 | gemma -> qwen | 1 |
| op15_r1 | {"cold": 2, "control": 1, "hot": 1} | {"hot": [75.4], "cold": [33.2, 36.5], "control": [0.3]} | 145.4 | 4 / 4 | gemma -> qwen -> gemma | 2 |
| twophone_r1 | {"cold": 2, "control": 1, "hot": 1} | {"hot": [94.1], "cold": [3.0, 93.7], "control": [0.4]} | 191.2 | 4 / 4 | gemma -> qwen -> gemma | 2 |

| arm | request | model | arrival s | acquired s | exec start s | exec end s | load wait s | attempts | wake reasons |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| desktop | 000 | gemma | 1.0 | 2.0 | 74.2 | 150.1 | 72.2 | 1 | calendar_elapsed |
| desktop | 001 | gemma | 1.0 | 73.9 | 74.2 | 98.7 | 0.3 | 2 | preparation_phase_completed,replanned |
| desktop | 00 | llama | 3.8 | 422.4 | 427.8 | 428.1 | 5.5 | 6 | capacity_released_early,model_affinity_displaced,model_affinity_displaced,capacity_released_early,capacity_released_early,replanned |
| desktop | 002 | qwen | 79.0 | 428.1 | 478.0 | 561.1 | 49.9 | 5 | model_affinity_displaced,model_affinity_displaced,capacity_released_early,residency_transition_completed,replanned |
| desktop | 003 | qwen | 86.2 | 477.7 | 478.0 | 566.7 | 0.2 | 5 | model_affinity_displaced,model_affinity_displaced,predecessor_replan,capacity_released_early,replanned |
| desktop | 004 | qwen | 99.0 | 477.9 | 478.2 | 623.7 | 0.3 | 5 | model_affinity_displaced,model_affinity_displaced,predecessor_replan,capacity_released_early,replanned |
| desktop | 005 | gemma | 109.4 | 111.8 | 112.1 | 422.3 | 0.2 | 1 | calendar_elapsed |
| desktop | 006 | qwen | 111.8 | 478.1 | 478.2 | 570.4 | 0.1 | 4 | model_affinity_displaced,predecessor_replan,capacity_released_early,replanned |
| desktop | 007 | gemma | 127.8 | 150.2 | 150.5 | 166.8 | 0.3 | 2 | capacity_released_early,replanned |
| op15_r1 | 000 | gemma | 1.0 | 5.9 | 41.5 | 100.5 | 35.6 | 1 | calendar_elapsed |
| op15_r1 | 001 | gemma | 1.0 | 41.3 | 41.6 | 65.1 | 0.3 | 2 | residency_observation_changed,replanned |
| op15_r1 | 00 | llama | 3.8 | 100.6 | 104.2 | 104.6 | 3.6 | 3 | capacity_released_early,residency_transition_completed,replanned |
| op15_r1 | 002 | qwen | 79.0 | 104.9 | 184.0 | 263.4 | 79.1 | 3 | capacity_released_early,residency_transition_completed,replanned |
| op15_r1 | 003 | qwen | 86.2 | 183.8 | 184.5 | 268.0 | 0.6 | 3 | capacity_released_early,preparation_phase_completed,replanned |
| op15_r1 | 004 | qwen | 99.0 | 184.4 | 184.7 | 317.8 | 0.4 | 3 | capacity_released_early,preparation_phase_completed,replanned |
| op15_r1 | 005 | gemma | 109.4 | 318.1 | 360.2 | 625.7 | 42.1 | 4 | model_affinity_displaced,capacity_released_early,capacity_released_early,replanned |
| op15_r1 | 006 | qwen | 111.8 | 184.4 | 184.7 | 270.9 | 0.4 | 1 | replanned |
| op15_r1 | 007 | gemma | 127.8 | 360.0 | 360.4 | 385.4 | 0.4 | 2 | preparation_phase_completed,replanned |
| twophone_r1 | 000 | gemma | 1.0 | 5.4 | 10.4 | 74.9 | 5.1 | 1 | calendar_elapsed |
| twophone_r1 | 001 | gemma | 1.0 | 10.3 | 10.4 | 32.9 | 0.1 | 2 | residency_observation_changed,replanned |
| twophone_r1 | 00 | llama | 3.8 | 74.9 | 77.7 | 77.9 | 2.7 | 3 | capacity_released_early,residency_transition_completed,replanned |
| twophone_r1 | 002 | qwen | 79.0 | 79.5 | 177.1 | 262.4 | 97.6 | 1 | calendar_elapsed |
| twophone_r1 | 003 | qwen | 86.2 | 176.9 | 177.1 | 271.6 | 0.2 | 2 | preparation_phase_completed,replanned |
| twophone_r1 | 004 | qwen | 99.0 | 176.9 | 177.6 | 318.6 | 0.7 | 1 | replanned |
| twophone_r1 | 005 | gemma | 109.4 | 318.8 | 428.6 | 718.7 | 109.8 | 4 | model_affinity_displaced,capacity_released_early,capacity_released_early,replanned |
| twophone_r1 | 006 | qwen | 111.8 | 176.9 | 177.6 | 274.1 | 0.7 | 1 | replanned |
| twophone_r1 | 007 | gemma | 127.8 | 423.0 | 428.5 | 455.8 | 5.6 | 2 | preparation_phase_completed,replanned |

| arm | execution pairs (same model) | overlap s by model | decode-window overlaps | window tokens at batch>=2 | active_slots_peak / slots max |
| --- | --- | --- | --- | --- | --- |
| desktop | gemma 001+000 24.5, gemma 000+005 38.1, gemma 005+007 16.3, qwen 003+002 83.1, qwen 003+004 88.5, qwen 003+006 88.5, qwen 002+004 82.9, qwen 002+006 82.9, qwen 004+006 92.2 | {"gemma": 78.9, "qwen": 518.1} | - | {} | 4 / 4 |
| op15_r1 | gemma 000+001 23.5, qwen 002+003 79.0, qwen 002+006 78.7, qwen 002+004 78.7, qwen 003+006 83.3, qwen 003+004 83.3, qwen 006+004 86.1, gemma 005+007 25.0 | {"gemma": 48.5, "qwen": 489.1} | gemma 000+001 11.9, gemma 005+007 10.9, qwen 002+006 69.7, qwen 002+003 69.0, qwen 002+004 68.4, qwen 006+003 74.0, qwen 006+004 76.5, qwen 003+004 73.4 | {"gemma": 115, "qwen": 491} | 1 / 1 |
| twophone_r1 | gemma 001+000 22.5, qwen 003+002 85.3, qwen 003+004 94.0, qwen 003+006 94.0, qwen 002+004 84.8, qwen 002+006 84.8, qwen 004+006 96.6, gemma 007+005 27.2 | {"gemma": 49.7, "qwen": 539.5} | gemma 001+000 13.0, gemma 007+005 11.3, qwen 003+002 73.2, qwen 003+004 82.4, qwen 003+006 82.4, qwen 002+004 73.2, qwen 002+006 73.2, qwen 004+006 85.5 | {"gemma": 106, "qwen": 496} | 1 / 1 |

| arm | server_policy reason by model (decisions) | per-batch reasons (last seen) | zero_assistance_reason by model | raw |
| --- | --- | --- | --- | --- |
| desktop | {} | {} | {} | {"RESULT.json": {"model_affinity_displaced": 9}, "SCHEDULER_DECISION_LOG.json": {"model_affinity_displaced": 9, "MODEL_AFFINITY_DISPLACEMENT": 4}, "ADAPTIVE_DECODE_OBSERVATIONS.json": {}} |
| op15_r1 | {"gemma|SERVER_COMPARISON_HOST_WINDOW": 3, "qwen|SERVER_COMPARISON_HOST_WINDOW": 3} | {"gemma|1|SERVER_COMPARISON_HOST_WINDOW": 3, "qwen|1|SERVER_COMPARISON_HOST_WINDOW": 3} | {} | {"RESULT.json": {"SERVER_COMPARISON_HOST_WINDOW": 12, "SERVER_POLICY_COHERENCE": 343, "model_affinity_displaced": 1}, "SCHEDULER_DECISION_LOG.json": {"SERVER_COMPARISON_HOST_WINDOW": 12, "SERVER_POLICY_COHERENCE": 343, "model_affinity_displaced": 1, "MODEL_AFFINITY_DISPLACEMENT": 1}, "ADAPTIVE_DECODE_OBSERVATIONS.json": {}} |
| twophone_r1 | {"gemma|SERVER_COMPARISON_HOST_WINDOW": 3, "qwen|SERVER_COMPARISON_HOST_WINDOW": 3} | {"gemma|1|SERVER_COMPARISON_HOST_WINDOW": 3, "qwen|1|SERVER_COMPARISON_HOST_WINDOW": 3} | {} | {"RESULT.json": {"SERVER_COMPARISON_HOST_WINDOW": 12, "SERVER_POLICY_COHERENCE": 315, "model_affinity_displaced": 1}, "SCHEDULER_DECISION_LOG.json": {"SERVER_COMPARISON_HOST_WINDOW": 12, "SERVER_POLICY_COHERENCE": 315, "model_affinity_displaced": 1, "MODEL_AFFINITY_DISPLACEMENT": 3}, "ADAPTIVE_DECODE_OBSERVATIONS.json": {}} |

- desktop displacement at 109.4 s (DECISION, requests ['005']): {"blocking_request_ids": ["002", "003", "004", "00"], "bypass_counts": {"burstgpt_longtail_dev_v2:002": 1, "burstgpt_longtail_dev_v2:003": 1, "burstgpt_longtail_dev_v2:004": 1, "burstgpt_longtail_dev_v2:llama-3.2-1b-overlay:00": 1}, "bypassed_request_ids": ["002", "003", "004", "00"], "displaced_request_ids": ["002", "003", "004", "00"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 109437931, "reserved_start_us": 109437931, "reserved_start_without_displacement_us": 737379510}
- desktop displacement at 111.8 s (ACQUIRED, requests ['005']): {"blocking_request_ids": ["002", "003", "004", "00"], "bypass_counts": {"burstgpt_longtail_dev_v2:002": 1, "burstgpt_longtail_dev_v2:003": 1, "burstgpt_longtail_dev_v2:004": 1, "burstgpt_longtail_dev_v2:llama-3.2-1b-overlay:00": 1}, "bypassed_request_ids": ["002", "003", "004", "00"], "displaced_request_ids": ["002", "003", "004", "00"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 109437931, "reserved_start_us": 109437931, "reserved_start_without_displacement_us": 737379510}
- desktop displacement at 127.8 s (DECISION, requests ['007']): {"blocking_request_ids": ["002", "003", "004", "006", "00"], "bypass_counts": {"burstgpt_longtail_dev_v2:002": 2, "burstgpt_longtail_dev_v2:003": 2, "burstgpt_longtail_dev_v2:004": 2, "burstgpt_longtail_dev_v2:006": 1, "burstgpt_longtail_dev_v2:llama-3.2-1b-overlay:00": 2}, "bypassed_request_ids": ["002", "003", "004", "006", "00"], "displaced_request_ids": ["002", "003", "004", "006", "00"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 127840765, "reserved_start_us": 204270842, "reserved_start_without_displacement_us": 1238003350}
- desktop displacement at 422.3 s (COMPLETED, requests ['005']): {"blocking_request_ids": ["002", "003", "004", "00"], "bypass_counts": {"burstgpt_longtail_dev_v2:002": 1, "burstgpt_longtail_dev_v2:003": 1, "burstgpt_longtail_dev_v2:004": 1, "burstgpt_longtail_dev_v2:llama-3.2-1b-overlay:00": 1}, "bypassed_request_ids": ["002", "003", "004", "00"], "displaced_request_ids": ["002", "003", "004", "00"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 109437931, "reserved_start_us": 109437931, "reserved_start_without_displacement_us": 737379510}
- op15_r1 displacement at 182.4 s (REPLAN, requests ['006']): {"blocking_request_ids": ["005"], "bypass_counts": {"burstgpt_longtail_dev_v2:005": 1, "burstgpt_longtail_dev_v2:007": 1}, "bypassed_request_ids": ["005", "007"], "displaced_request_ids": ["005", "007"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 182419988, "replan_reason": "priority_compaction_follower"}
- twophone_r1 displacement at 175.6 s (REPLAN, requests ['006']): {"blocking_request_ids": ["005"], "bypass_counts": {"burstgpt_longtail_dev_v2:005": 1, "burstgpt_longtail_dev_v2:007": 1}, "bypassed_request_ids": ["005", "007"], "displaced_request_ids": ["005", "007"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 175595180, "replan_reason": "priority_compaction_follower"}
- twophone_r1 displacement at 176.9 s (ACQUIRED, requests ['006']): {"blocking_request_ids": ["005"], "bypass_counts": {"burstgpt_longtail_dev_v2:005": 1, "burstgpt_longtail_dev_v2:007": 1}, "bypassed_request_ids": ["005", "007"], "displaced_request_ids": ["005", "007"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 175595180, "replan_reason": "priority_compaction_follower"}
- twophone_r1 displacement at 274.1 s (COMPLETED, requests ['006']): {"blocking_request_ids": ["005"], "bypass_counts": {"burstgpt_longtail_dev_v2:005": 1, "burstgpt_longtail_dev_v2:007": 1}, "bypassed_request_ids": ["005", "007"], "displaced_request_ids": ["005", "007"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 175595180, "replan_reason": "priority_compaction_follower"}
