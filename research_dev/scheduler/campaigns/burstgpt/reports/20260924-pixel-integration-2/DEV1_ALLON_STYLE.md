## Dispatch, loads, pairs, probe reasons

| arm | dispatch_policy statistics | bypass counts | displacements (notes) | REPLAN reasons | wake reasons |
| --- | --- | --- | --- | --- | --- |
| desktop | {"affinity_displaced_attempts": 3, "affinity_displacements": 2, "affinity_refusals": 1, "early_capacity_promotions": 2, "publication_replans": 1, "published_work_promotions": 4} | {"burstgpt_longtail_dev_v1:001": 1, "burstgpt_longtail_dev_v1:004": 1, "burstgpt_longtail_dev_v1:llama-3.2-1b-overlay:00": 1} | 6 | {"DESKTOP_BASELINE_CONTROL": 14} | {"replanned": 5, "capacity_released_early": 4, "preparation_phase_completed": 3, "residency_transition_completed": 2, "model_affinity_displaced": 2, "calendar_elapsed": 1, "residency_observation_changed": 1} |
| twophone | {"affinity_displaced_attempts": 2, "affinity_displacements": 2, "affinity_refusals": 0, "early_capacity_promotions": 1, "publication_replans": 1, "published_work_promotions": 2} | {"burstgpt_longtail_dev_v1:001": 1, "burstgpt_longtail_dev_v1:llama-3.2-1b-overlay:00": 1} | 4 | {"QUALIFIED_FALLBACK_NO_SAFE_ALTERNATIVE": 12} | {"replanned": 5, "capacity_released_early": 3, "model_affinity_displaced": 2, "residency_transition_completed": 2, "calendar_elapsed": 1, "preparation_phase_completed": 1, "residency_observation_changed": 1, "predecessor_replan": 1} |

| arm | large-model launches (hot/cold/control) | load s by role | total load s | model_reload_count / transition_count | execution sequence | switches |
| --- | --- | --- | ---: | --- | --- | ---: |
| desktop | {"hot": 2, "cold": 1, "control": 1} | {"hot": [56.5, 78.7], "cold": [68.0], "control": [1.5]} | 204.7 | 4 / 4 | qwen -> gemma -> qwen | 2 |
| twophone | {"hot": 2, "cold": 1, "control": 1} | {"hot": [76.3, 149.8], "cold": [36.0], "control": [0.4]} | 262.5 | 4 / 4 | qwen -> gemma -> qwen | 2 |

| arm | request | model | arrival s | acquired s | exec start s | exec end s | load wait s | attempts | wake reasons |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| desktop | 000 | qwen | 1.0 | 1.7 | 61.1 | 149.5 | 59.5 | 1 | calendar_elapsed |
| desktop | 001 | gemma | 27.0 | 149.6 | 222.0 | 494.6 | 72.4 | 4 | preparation_phase_completed,capacity_released_early,residency_transition_completed,replanned |
| desktop | 002 | qwen | 56.0 | 61.2 | 61.4 | 96.0 | 0.2 | 2 | preparation_phase_completed,replanned |
| desktop | 00 | llama | 207.0 | 494.9 | 507.0 | 507.4 | 12.1 | 5 | preparation_phase_completed,model_affinity_displaced,capacity_released_early,residency_transition_completed,replanned |
| desktop | 003 | gemma | 391.0 | 393.5 | 393.8 | 423.7 | 0.3 | 2 | residency_observation_changed,replanned |
| desktop | 004 | qwen | 393.0 | 507.5 | 589.1 | 711.3 | 81.6 | 4 | model_affinity_displaced,capacity_released_early,capacity_released_early,replanned |
| twophone | 000 | qwen | 1.0 | 5.2 | 92.0 | 193.1 | 86.7 | 1 | calendar_elapsed |
| twophone | 001 | gemma | 27.0 | 193.3 | 235.2 | 452.2 | 41.9 | 5 | preparation_phase_completed,model_affinity_displaced,capacity_released_early,residency_transition_completed,replanned |
| twophone | 002 | qwen | 56.0 | 92.0 | 92.5 | 136.1 | 0.5 | 2 | residency_observation_changed,replanned |
| twophone | 00 | llama | 207.0 | 479.2 | 487.7 | 488.1 | 8.5 | 3 | model_affinity_displaced,capacity_released_early,replanned |
| twophone | 003 | gemma | 391.0 | 452.6 | 453.6 | 479.0 | 1.0 | 2 | capacity_released_early,replanned |
| twophone | 004 | qwen | 393.0 | 488.3 | 645.3 | 776.3 | 157.0 | 3 | predecessor_replan,residency_transition_completed,replanned |

| arm | execution pairs (same model) | overlap s by model | decode-window overlaps | window tokens at batch>=2 | active_slots_peak / slots max |
| --- | --- | --- | --- | --- | --- |
| desktop | qwen 000+002 34.6, gemma 001+003 29.9 | {"gemma": 29.9, "qwen": 34.6} | - | {} | 2 / 2 |
| twophone | qwen 000+002 43.6 | {"qwen": 43.6} | qwen 000+002 33.1 | {"gemma": 0, "qwen": 94} | 1 / 1 |

| arm | server_policy reason by model (decisions) | per-batch reasons (last seen) | zero_assistance_reason by model | raw |
| --- | --- | --- | --- | --- |
| desktop | {} | {} | {} | {"RESULT.json": {"model_affinity_displaced": 2}, "SCHEDULER_DECISION_LOG.json": {"model_affinity_displaced": 2, "MODEL_AFFINITY_DISPLACEMENT": 6}, "ADAPTIVE_DECODE_OBSERVATIONS.json": {}} |
| twophone | {"qwen|SERVER_COMPARISON_HOST_WINDOW": 3} | {"qwen|1|SERVER_COMPARISON_HOST_WINDOW": 3} | {} | {"RESULT.json": {"SERVER_COMPARISON_HOST_WINDOW": 6, "SERVER_POLICY_COHERENCE": 248, "model_affinity_displaced": 2}, "SCHEDULER_DECISION_LOG.json": {"SERVER_COMPARISON_HOST_WINDOW": 6, "SERVER_POLICY_COHERENCE": 248, "model_affinity_displaced": 2, "MODEL_AFFINITY_DISPLACEMENT": 4}, "ADAPTIVE_DECODE_OBSERVATIONS.json": {}} |

- desktop displacement at 61.1 s (REPLAN, requests ['002']): {"blocking_request_ids": ["001"], "bypass_counts": {"burstgpt_longtail_dev_v1:001": 1}, "bypassed_request_ids": ["001"], "displaced_request_ids": ["001"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 61076737, "replan_reason": "preparation_phase_completed"}
- desktop displacement at 61.2 s (ACQUIRED, requests ['002']): {"blocking_request_ids": ["001"], "bypass_counts": {"burstgpt_longtail_dev_v1:001": 1}, "bypassed_request_ids": ["001"], "displaced_request_ids": ["001"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 61076737, "replan_reason": "preparation_phase_completed"}
- desktop displacement at 96.0 s (COMPLETED, requests ['002']): {"blocking_request_ids": ["001"], "bypass_counts": {"burstgpt_longtail_dev_v1:001": 1}, "bypassed_request_ids": ["001"], "displaced_request_ids": ["001"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 61076737, "replan_reason": "preparation_phase_completed"}
- desktop displacement at 393.5 s (REPLAN, requests ['003']): {"blocking_request_ids": ["00"], "bypass_counts": {"burstgpt_longtail_dev_v1:004": 1, "burstgpt_longtail_dev_v1:llama-3.2-1b-overlay:00": 1}, "bypassed_request_ids": ["004", "00"], "displaced_request_ids": ["004", "00"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 393466783, "replan_reason": "residency_observation_changed"}
- desktop displacement at 393.5 s (ACQUIRED, requests ['003']): {"blocking_request_ids": ["00"], "bypass_counts": {"burstgpt_longtail_dev_v1:004": 1, "burstgpt_longtail_dev_v1:llama-3.2-1b-overlay:00": 1}, "bypassed_request_ids": ["004", "00"], "displaced_request_ids": ["004", "00"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 393466783, "replan_reason": "residency_observation_changed"}
- desktop displacement at 423.7 s (COMPLETED, requests ['003']): {"blocking_request_ids": ["00"], "bypass_counts": {"burstgpt_longtail_dev_v1:004": 1, "burstgpt_longtail_dev_v1:llama-3.2-1b-overlay:00": 1}, "bypassed_request_ids": ["004", "00"], "displaced_request_ids": ["004", "00"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 393466783, "replan_reason": "residency_observation_changed"}
- twophone displacement at 91.9 s (REPLAN, requests ['002']): {"blocking_request_ids": ["001"], "bypass_counts": {"burstgpt_longtail_dev_v1:001": 1}, "bypassed_request_ids": ["001"], "displaced_request_ids": ["001"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 91927031, "replan_reason": "residency_observation_changed"}
- twophone displacement at 92.0 s (ACQUIRED, requests ['002']): {"blocking_request_ids": ["001"], "bypass_counts": {"burstgpt_longtail_dev_v1:001": 1}, "bypassed_request_ids": ["001"], "displaced_request_ids": ["001"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 91927031, "replan_reason": "residency_observation_changed"}
- twophone displacement at 136.1 s (COMPLETED, requests ['002']): {"blocking_request_ids": ["001"], "bypass_counts": {"burstgpt_longtail_dev_v1:001": 1}, "bypassed_request_ids": ["001"], "displaced_request_ids": ["001"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 91927031, "replan_reason": "residency_observation_changed"}
- twophone displacement at 391.3 s (DECISION, requests ['003']): {"blocking_request_ids": ["00"], "bypass_counts": {"burstgpt_longtail_dev_v1:llama-3.2-1b-overlay:00": 1}, "bypassed_request_ids": ["00"], "displaced_request_ids": ["00"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 391283236, "reserved_start_us": 719048443, "reserved_start_without_displacement_us": 732464336}
