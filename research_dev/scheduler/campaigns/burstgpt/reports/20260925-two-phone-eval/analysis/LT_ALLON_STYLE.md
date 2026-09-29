## Dispatch, loads, pairs, probe reasons

| arm | dispatch_policy statistics | bypass counts | displacements (notes) | REPLAN reasons | wake reasons |
| --- | --- | --- | --- | --- | --- |
| legacy | - | - | 0 | {"DESKTOP_BASELINE_CONTROL": 37} | {"replanned": 29, "capacity_released_early": 18, "predecessor_completion": 7, "preparation_phase_completed": 3, "calendar_elapsed": 2, "predecessor_replan": 1} |
| desktopDP | {"affinity_displaced_attempts": 51, "affinity_displacements": 12, "affinity_refusals": 12, "early_capacity_promotions": 6, "publication_replans": 0, "published_work_promotions": 25} | {"burstgpt_longtail_v1:004": 10, "burstgpt_longtail_v1:011": 4, "burstgpt_longtail_v1:014": 2, "burstgpt_longtail_v1:015": 2, "burstgpt_longtail_v1:016": 2, "burstgpt_longtail_v1:018": 1, "burstgpt_longtail_v1:020": 2, "burstgpt_longtail_v1:llama-3.2-1b-overlay:01": 10} | 18 | {"DESKTOP_BASELINE_CONTROL": 86} | {"model_affinity_displaced": 33, "capacity_released_early": 26, "replanned": 24, "predecessor_replan": 14, "preparation_phase_completed": 9, "calendar_elapsed": 6, "predecessor_completion": 4, "residency_projection_invalid": 3, "residency_transition_completed": 2} |

| arm | large-model launches (hot/cold/control) | load s by role | total load s | model_reload_count / transition_count | execution sequence | switches |
| --- | --- | --- | ---: | --- | --- | ---: |
| legacy | {"control": 3, "hot": 7, "cold": 5} | {"hot": [94.3, 85.6, 5.8, 43.8, 45.7, 90.8, 43.3], "cold": [31.3, 34.5, 35.0, 58.6, 58.3], "control": [0.3, 1.4, 1.6]} | 630.3 | 15 / 15 | qwen -> gemma -> qwen -> gemma -> qwen -> gemma -> qwen -> gemma -> qwen -> gemma -> qwen | 10 |
| desktopDP | {"control": 2, "hot": 3, "cold": 3} | {"hot": [19.1, 44.4, 50.9], "cold": [33.5, 3.3, 72.5], "control": [0.3, 1.2]} | 225.2 | 8 / 8 | qwen -> gemma -> qwen -> gemma -> qwen | 4 |

| arm | request | model | arrival s | acquired s | exec start s | exec end s | load wait s | attempts | wake reasons |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| legacy | 00 | llama | 1.0 | 1.2 | 2.9 | 3.1 | 1.7 | 1 | calendar_elapsed |
| legacy | 000 | qwen | 13.0 | 13.7 | 112.9 | 283.0 | 99.2 | 1 | calendar_elapsed |
| legacy | 001 | qwen | 79.0 | 283.2 | 283.3 | 424.0 | 0.1 | 2 | capacity_released_early,replanned |
| legacy | 002 | qwen | 115.0 | 283.2 | 283.3 | 450.7 | 0.1 | 1 | replanned |
| legacy | 003 | qwen | 168.0 | 424.4 | 424.6 | 551.2 | 0.2 | 2 | capacity_released_early,replanned |
| legacy | 004 | gemma | 248.0 | 551.4 | 588.0 | 894.6 | 36.6 | 2 | predecessor_completion,replanned |
| legacy | 005 | qwen | 257.0 | 894.7 | 986.3 | 1060.3 | 91.6 | 2 | predecessor_completion,replanned |
| legacy | 006 | qwen | 343.0 | 1062.2 | 1062.4 | 1185.5 | 0.2 | 2 | preparation_phase_completed,replanned |
| legacy | 01 | llama | 418.0 | 1185.6 | 1192.0 | 1192.3 | 6.4 | 2 | capacity_released_early,replanned |
| legacy | 007 | qwen | 430.0 | 1192.5 | 1204.4 | 1345.5 | 11.9 | 2 | preparation_phase_completed,replanned |
| legacy | 008 | qwen | 440.0 | 1353.3 | 1353.4 | 1523.9 | 0.1 | 4 | capacity_released_early,capacity_released_early,capacity_released_early,replanned |
| legacy | 009 | qwen | 517.0 | 1353.3 | 1353.4 | 1445.0 | 0.1 | 1 | replanned |
| legacy | 010 | qwen | 588.0 | 1524.2 | 1524.4 | 2061.8 | 0.2 | 2 | predecessor_completion,replanned |
| legacy | 011 | gemma | 593.0 | 2062.7 | 2107.1 | 2174.8 | 44.4 | 3 | capacity_released_early,capacity_released_early,replanned |
| legacy | 012 | qwen | 610.0 | 2177.3 | 2235.7 | 2320.6 | 58.4 | 2 | predecessor_completion,replanned |
| legacy | 013 | qwen | 638.0 | 2322.9 | 2324.2 | 2368.3 | 1.4 | 2 | predecessor_completion,replanned |
| legacy | 014 | gemma | 648.0 | 2368.6 | 2413.6 | 2440.8 | 45.0 | 3 | predecessor_replan,capacity_released_early,replanned |
| legacy | 015 | gemma | 661.0 | 2442.1 | 2442.3 | 2986.8 | 0.2 | 2 | preparation_phase_completed,replanned |
| legacy | 016 | gemma | 681.0 | 2442.1 | 2442.3 | 2477.7 | 0.2 | 1 | replanned |
| legacy | 017 | qwen | 683.0 | 2987.2 | 3047.0 | 3145.8 | 59.8 | 2 | capacity_released_early,replanned |
| legacy | 018 | gemma | 752.0 | 3155.3 | 3227.3 | 3288.3 | 72.0 | 2 | capacity_released_early,replanned |
| legacy | 019 | qwen | 755.0 | 3291.3 | 3397.3 | 3472.9 | 106.1 | 2 | capacity_released_early,replanned |
| legacy | 020 | qwen | 870.0 | 3475.6 | 3475.7 | 3582.5 | 0.2 | 2 | capacity_released_early,replanned |
| legacy | 021 | gemma | 961.0 | 3582.7 | 3654.1 | 4096.8 | 71.4 | 2 | capacity_released_early,replanned |
| legacy | 022 | gemma | 991.0 | 4099.3 | 4099.6 | 4366.5 | 0.4 | 2 | predecessor_completion,replanned |
| legacy | 023 | gemma | 1083.0 | 4099.5 | 4099.6 | 4160.4 | 0.1 | 2 | capacity_released_early,replanned |
| legacy | 024 | gemma | 1134.0 | 4366.7 | 4367.0 | 4454.7 | 0.3 | 2 | capacity_released_early,replanned |
| legacy | 025 | gemma | 1167.0 | 4366.7 | 4367.0 | 4537.4 | 0.3 | 1 | replanned |
| legacy | 026 | gemma | 1200.0 | 4538.8 | 4539.0 | 4726.0 | 0.1 | 2 | capacity_released_early,replanned |
| legacy | 02 | llama | 1515.0 | 4733.0 | 4740.3 | 4740.7 | 7.3 | 2 | capacity_released_early,replanned |
| legacy | 027 | qwen | 1580.0 | 4741.1 | 4794.3 | 4981.0 | 53.2 | 2 | predecessor_completion,replanned |
| desktopDP | 00 | llama | 1.0 | 1.2 | 2.7 | 2.9 | 1.5 | 1 | calendar_elapsed |
| desktopDP | 000 | qwen | 13.0 | 13.7 | 35.4 | 216.5 | 21.8 | 1 | calendar_elapsed |
| desktopDP | 001 | qwen | 79.0 | 80.8 | 81.0 | 222.4 | 0.1 | 1 | calendar_elapsed |
| desktopDP | 002 | qwen | 115.0 | 216.6 | 216.7 | 380.1 | 0.1 | 2 | capacity_released_early,replanned |
| desktopDP | 003 | qwen | 168.0 | 222.5 | 222.6 | 343.1 | 0.1 | 2 | capacity_released_early,replanned |
| desktopDP | 004 | gemma | 248.0 | 1156.1 | 1197.1 | 1509.2 | 41.1 | 15 | model_affinity_displaced,model_affinity_displaced,capacity_released_early,model_affinity_displaced,model_affinity_displaced,capacity_released_early,model_affinity_displaced,model_affinity_displaced,model_affinity_displaced,model_affinity_displaced,model_affinity_displaced,model_affinity_displaced,capacity_released_early,capacity_released_early,replanned |
| desktopDP | 005 | qwen | 257.0 | 343.2 | 343.4 | 424.1 | 0.2 | 2 | capacity_released_early,replanned |
| desktopDP | 006 | qwen | 343.0 | 380.2 | 380.4 | 504.5 | 0.2 | 2 | capacity_released_early,replanned |
| desktopDP | 01 | llama | 418.0 | 1511.4 | 1515.2 | 1515.5 | 3.8 | 16 | model_affinity_displaced,model_affinity_displaced,model_affinity_displaced,model_affinity_displaced,model_affinity_displaced,model_affinity_displaced,model_affinity_displaced,model_affinity_displaced,predecessor_replan,model_affinity_displaced,model_affinity_displaced,residency_projection_invalid,residency_projection_invalid,capacity_released_early,residency_transition_completed,replanned |
| desktopDP | 007 | qwen | 430.0 | 430.2 | 430.4 | 582.0 | 0.2 | 1 | calendar_elapsed |
| desktopDP | 008 | qwen | 440.0 | 504.6 | 504.9 | 673.6 | 0.3 | 2 | capacity_released_early,replanned |
| desktopDP | 009 | qwen | 517.0 | 673.7 | 673.8 | 759.9 | 0.1 | 3 | model_affinity_displaced,capacity_released_early,replanned |
| desktopDP | 010 | qwen | 588.0 | 588.2 | 588.5 | 1156.0 | 0.3 | 1 | calendar_elapsed |
| desktopDP | 011 | gemma | 593.0 | 1197.2 | 1197.4 | 1269.5 | 0.2 | 6 | model_affinity_displaced,model_affinity_displaced,model_affinity_displaced,model_affinity_displaced,predecessor_replan,replanned |
| desktopDP | 012 | qwen | 610.0 | 859.9 | 860.0 | 945.7 | 0.2 | 2 | capacity_released_early,replanned |
| desktopDP | 013 | qwen | 638.0 | 1026.5 | 1026.6 | 1073.1 | 0.1 | 2 | capacity_released_early,replanned |
| desktopDP | 014 | gemma | 648.0 | 1269.8 | 1269.9 | 1296.0 | 0.2 | 5 | model_affinity_displaced,model_affinity_displaced,predecessor_replan,capacity_released_early,replanned |
| desktopDP | 015 | gemma | 661.0 | 1515.7 | 1520.9 | 2061.1 | 5.2 | 5 | model_affinity_displaced,model_affinity_displaced,predecessor_replan,predecessor_completion,replanned |
| desktopDP | 016 | gemma | 681.0 | 1520.8 | 1520.9 | 1556.4 | 0.1 | 5 | model_affinity_displaced,model_affinity_displaced,predecessor_replan,capacity_released_early,replanned |
| desktopDP | 017 | qwen | 683.0 | 760.0 | 760.2 | 859.7 | 0.2 | 2 | capacity_released_early,replanned |
| desktopDP | 018 | gemma | 752.0 | 2061.6 | 2061.7 | 2121.4 | 0.1 | 4 | model_affinity_displaced,predecessor_replan,predecessor_completion,replanned |
| desktopDP | 019 | qwen | 755.0 | 945.8 | 945.9 | 1026.3 | 0.1 | 2 | capacity_released_early,replanned |
| desktopDP | 020 | qwen | 870.0 | 2121.5 | 2176.9 | 2284.1 | 55.4 | 4 | preparation_phase_completed,capacity_released_early,capacity_released_early,replanned |
| desktopDP | 021 | gemma | 961.0 | 2284.4 | 2367.9 | 2813.1 | 83.6 | 5 | preparation_phase_completed,preparation_phase_completed,residency_projection_invalid,residency_transition_completed,replanned |
| desktopDP | 022 | gemma | 991.0 | 2367.8 | 2367.9 | 2651.1 | 0.1 | 4 | preparation_phase_completed,predecessor_replan,preparation_phase_completed,replanned |
| desktopDP | 023 | gemma | 1083.0 | 2903.6 | 2903.8 | 2963.1 | 0.3 | 8 | predecessor_replan,capacity_released_early,capacity_released_early,preparation_phase_completed,capacity_released_early,capacity_released_early,capacity_released_early,predecessor_completion |
| desktopDP | 024 | gemma | 1134.0 | 2813.7 | 2813.9 | 2902.0 | 0.2 | 5 | predecessor_replan,predecessor_replan,preparation_phase_completed,capacity_released_early,replanned |
| desktopDP | 025 | gemma | 1167.0 | 2963.3 | 2963.4 | 3127.8 | 0.1 | 5 | model_affinity_displaced,predecessor_replan,preparation_phase_completed,capacity_released_early,replanned |
| desktopDP | 026 | gemma | 1200.0 | 2813.7 | 2813.9 | 3010.3 | 0.2 | 3 | predecessor_replan,predecessor_replan,replanned |
| desktopDP | 02 | llama | 1515.0 | 1515.1 | 1515.2 | 1515.6 | 0.1 | 1 | calendar_elapsed |
| desktopDP | 027 | qwen | 1580.0 | 3128.0 | 3192.0 | 3377.8 | 64.0 | 4 | preparation_phase_completed,predecessor_replan,predecessor_completion,replanned |

| arm | execution pairs (same model) | overlap s by model | decode-window overlaps | window tokens at batch>=2 | active_slots_peak / slots max |
| --- | --- | --- | --- | --- | --- |
| legacy | qwen 002+001 140.7, qwen 002+003 26.2, qwen 008+009 91.5, gemma 015+016 35.5, gemma 023+022 60.8, gemma 025+024 87.7 | {"gemma": 184.0, "qwen": 258.4} | - | {} | 2 / 2 |
| desktopDP | qwen 000+001 135.5, qwen 001+002 5.6, qwen 002+003 120.5, qwen 002+005 36.7, qwen 005+006 43.7, qwen 006+007 74.2, qwen 007+008 77.1, qwen 008+010 85.2, qwen 010+009 86.1, qwen 010+017 99.5, qwen 010+012 85.6, qwen 010+019 80.4, qwen 010+013 46.5, gemma 004+011 72.1, gemma 004+014 26.0, gemma 015+016 35.5, gemma 021+022 283.1, gemma 024+026 88.0, gemma 026+023 59.3, gemma 026+025 46.9 | {"gemma": 610.9, "qwen": 976.6} | - | {} | 2 / 2 |

| arm | server_policy reason by model (decisions) | per-batch reasons (last seen) | zero_assistance_reason by model | raw |
| --- | --- | --- | --- | --- |
| legacy | {} | {} | {} | {"RESULT.json": {}, "SCHEDULER_DECISION_LOG.json": {}, "ADAPTIVE_DECODE_OBSERVATIONS.json": {}} |
| desktopDP | {} | {} | {} | {"RESULT.json": {"model_affinity_displaced": 33}, "SCHEDULER_DECISION_LOG.json": {"model_affinity_displaced": 34, "MODEL_AFFINITY_DISPLACEMENT": 18}, "ADAPTIVE_DECODE_OBSERVATIONS.json": {}} |

- desktopDP displacement at 257.0 s (DECISION, requests ['005']): {"blocking_request_ids": ["004"], "bypass_counts": {"burstgpt_longtail_v1:004": 1}, "bypassed_request_ids": ["004"], "displaced_request_ids": ["004"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 257042130, "reserved_start_us": 431953233, "reserved_start_without_displacement_us": 1136608022}
- desktopDP displacement at 343.0 s (DECISION, requests ['006']): {"blocking_request_ids": ["004"], "bypass_counts": {"burstgpt_longtail_v1:004": 2}, "bypassed_request_ids": ["004"], "displaced_request_ids": ["004"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 343043097, "reserved_start_us": 472174902, "reserved_start_without_displacement_us": 1275953650}
- desktopDP displacement at 430.0 s (DECISION, requests ['007']): {"blocking_request_ids": ["004", "01"], "bypass_counts": {"burstgpt_longtail_v1:004": 3, "burstgpt_longtail_v1:llama-3.2-1b-overlay:01": 1}, "bypassed_request_ids": ["004", "01"], "displaced_request_ids": ["004", "01"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 430036379, "reserved_start_us": 430036379, "reserved_start_without_displacement_us": 1241843373}
- desktopDP displacement at 430.2 s (ACQUIRED, requests ['007']): {"blocking_request_ids": ["004", "01"], "bypass_counts": {"burstgpt_longtail_v1:004": 3, "burstgpt_longtail_v1:llama-3.2-1b-overlay:01": 1}, "bypassed_request_ids": ["004", "01"], "displaced_request_ids": ["004", "01"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 430036379, "reserved_start_us": 430036379, "reserved_start_without_displacement_us": 1241843373}
- desktopDP displacement at 440.0 s (DECISION, requests ['008']): {"blocking_request_ids": ["004", "01"], "bypass_counts": {"burstgpt_longtail_v1:004": 4, "burstgpt_longtail_v1:llama-3.2-1b-overlay:01": 2}, "bypassed_request_ids": ["004", "01"], "displaced_request_ids": ["004", "01"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 440027701, "reserved_start_us": 579039573, "reserved_start_without_displacement_us": 1360080549}
- desktopDP displacement at 582.0 s (COMPLETED, requests ['007']): {"blocking_request_ids": ["004", "01"], "bypass_counts": {"burstgpt_longtail_v1:004": 3, "burstgpt_longtail_v1:llama-3.2-1b-overlay:01": 1}, "bypassed_request_ids": ["004", "01"], "displaced_request_ids": ["004", "01"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 430036379, "reserved_start_us": 430036379, "reserved_start_without_displacement_us": 1241843373}
- desktopDP displacement at 588.0 s (DECISION, requests ['010']): {"blocking_request_ids": ["004", "01"], "bypass_counts": {"burstgpt_longtail_v1:004": 5, "burstgpt_longtail_v1:llama-3.2-1b-overlay:01": 3}, "bypassed_request_ids": ["004", "01"], "displaced_request_ids": ["004", "009", "01"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 588029332, "reserved_start_us": 588029332, "reserved_start_without_displacement_us": 1447132559}
- desktopDP displacement at 588.2 s (ACQUIRED, requests ['010']): {"blocking_request_ids": ["004", "01"], "bypass_counts": {"burstgpt_longtail_v1:004": 5, "burstgpt_longtail_v1:llama-3.2-1b-overlay:01": 3}, "bypassed_request_ids": ["004", "01"], "displaced_request_ids": ["004", "009", "01"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 588029332, "reserved_start_us": 588029332, "reserved_start_without_displacement_us": 1447132559}
- desktopDP displacement at 588.3 s (REPLAN, requests ['009']): {"blocking_request_ids": ["004", "01"], "bypass_counts": {"burstgpt_longtail_v1:004": 6, "burstgpt_longtail_v1:llama-3.2-1b-overlay:01": 4}, "bypassed_request_ids": ["004", "01"], "displaced_request_ids": ["004", "01"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 588308024, "replan_reason": "model_affinity_displaced"}
- desktopDP displacement at 610.0 s (DECISION, requests ['012']): {"blocking_request_ids": ["004", "011", "01"], "bypass_counts": {"burstgpt_longtail_v1:004": 7, "burstgpt_longtail_v1:011": 1, "burstgpt_longtail_v1:llama-3.2-1b-overlay:01": 5}, "bypassed_request_ids": ["004", "011", "01"], "displaced_request_ids": ["004", "011", "01"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 610044022, "reserved_start_us": 949715252, "reserved_start_without_displacement_us": 2387480504}
- desktopDP displacement at 638.0 s (DECISION, requests ['013']): {"blocking_request_ids": ["004", "011", "01"], "bypass_counts": {"burstgpt_longtail_v1:004": 8, "burstgpt_longtail_v1:011": 2, "burstgpt_longtail_v1:llama-3.2-1b-overlay:01": 6}, "bypassed_request_ids": ["004", "011", "01"], "displaced_request_ids": ["004", "011", "01"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 638038875, "reserved_start_us": 1250458785, "reserved_start_without_displacement_us": 2393679326}
- desktopDP displacement at 683.0 s (DECISION, requests ['017']): {"blocking_request_ids": ["004", "011", "014", "015", "016", "01"], "bypass_counts": {"burstgpt_longtail_v1:004": 9, "burstgpt_longtail_v1:011": 3, "burstgpt_longtail_v1:014": 1, "burstgpt_longtail_v1:015": 1, "burstgpt_longtail_v1:016": 1, "burstgpt_longtail_v1:llama-3.2-1b-overlay:01": 7}, "bypassed_request_ids": ["004", "011", "014", "015", "016", "01"], "displaced_request_ids": ["004", "011", "014", "015", "016", "01"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 683039412, "reserved_start_us": 927525481, "reserved_start_without_displacement_us": 1695555819}
- desktopDP displacement at 755.0 s (DECISION, requests ['019']): {"blocking_request_ids": ["004", "011", "014", "015", "016", "018", "01"], "bypass_counts": {"burstgpt_longtail_v1:004": 10, "burstgpt_longtail_v1:011": 4, "burstgpt_longtail_v1:014": 2, "burstgpt_longtail_v1:015": 2, "burstgpt_longtail_v1:016": 2, "burstgpt_longtail_v1:018": 1, "burstgpt_longtail_v1:llama-3.2-1b-overlay:01": 8}, "bypassed_request_ids": ["004", "011", "014", "015", "016", "018", "01"], "displaced_request_ids": ["004", "011", "014", "015", "016", "018", "01"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 755036171, "reserved_start_us": 1230201961, "reserved_start_without_displacement_us": 3388301069}
- desktopDP displacement at 1156.0 s (COMPLETED, requests ['010']): {"blocking_request_ids": ["004", "01"], "bypass_counts": {"burstgpt_longtail_v1:004": 5, "burstgpt_longtail_v1:llama-3.2-1b-overlay:01": 3}, "bypassed_request_ids": ["004", "01"], "displaced_request_ids": ["004", "009", "01"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 588029332, "reserved_start_us": 588029332, "reserved_start_without_displacement_us": 1447132559}
- desktopDP displacement at 1196.9 s (REPLAN, requests ['011']): {"blocking_request_ids": ["01"], "bypass_counts": {"burstgpt_longtail_v1:020": 1, "burstgpt_longtail_v1:llama-3.2-1b-overlay:01": 9}, "bypassed_request_ids": ["020", "01"], "displaced_request_ids": ["014", "015", "016", "018", "020", "021", "022", "023", "024", "025", "01"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 1196897261, "replan_reason": "predecessor_replan"}
- desktopDP displacement at 1197.2 s (ACQUIRED, requests ['011']): {"blocking_request_ids": ["01"], "bypass_counts": {"burstgpt_longtail_v1:020": 1, "burstgpt_longtail_v1:llama-3.2-1b-overlay:01": 9}, "bypassed_request_ids": ["020", "01"], "displaced_request_ids": ["014", "015", "016", "018", "020", "021", "022", "023", "024", "025", "01"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 1196897261, "replan_reason": "predecessor_replan"}
- desktopDP displacement at 1197.3 s (REPLAN, requests ['014']): {"blocking_request_ids": ["01"], "bypass_counts": {"burstgpt_longtail_v1:020": 2, "burstgpt_longtail_v1:llama-3.2-1b-overlay:01": 10}, "bypassed_request_ids": ["020", "01"], "displaced_request_ids": ["015", "016", "018", "020", "021", "022", "023", "024", "025", "01"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 1197311625, "replan_reason": "predecessor_replan"}
- desktopDP displacement at 1269.5 s (COMPLETED, requests ['011']): {"blocking_request_ids": ["01"], "bypass_counts": {"burstgpt_longtail_v1:020": 1, "burstgpt_longtail_v1:llama-3.2-1b-overlay:01": 9}, "bypassed_request_ids": ["020", "01"], "displaced_request_ids": ["014", "015", "016", "018", "020", "021", "022", "023", "024", "025", "01"], "kind": "MODEL_AFFINITY_DISPLACEMENT", "observed_at_us": 1196897261, "replan_reason": "predecessor_replan"}
