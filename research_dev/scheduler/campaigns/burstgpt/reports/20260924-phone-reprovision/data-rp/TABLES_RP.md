| arm | status | duration s | CPU kJ | GPU kJ | host kJ | host W | host kJ / duration vs earlier arms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| base | PASS | 1154.3 | 64.896 | 31.852 | 96.749 | 83.8 | - |
| plainEF | PASS | 1035.1 | 46.373 | 29.933 | 76.306 | 73.7 | base -21.1 % / -10.3 % |
| coherentEF | PASS | 924.5 | 43.502 | 28.355 | 71.858 | 77.7 | base -25.7 % / -19.9 %; plainEF -5.8 % / -10.7 % |
| coherentEF2 | PASS | 1042.8 | 39.908 | 29.729 | 69.637 | 66.8 | base -28.0 % / -9.7 %; plainEF -8.7 % / +0.7 %; coherentEF -3.1 % / +12.8 % |
| coherentRP | PASS | 898.1 | 26.034 | 27.652 | 53.686 | 59.8 | base -44.5 % / -22.2 %; plainEF -29.6 % / -13.2 %; coherentEF -25.3 % / -2.9 %; coherentEF2 -22.9 % / -13.9 % |

| arm | model | output tokens | phone-policy window tokens | share | requests with phone calls |
| --- | --- | ---: | ---: | ---: | ---: |
| base | gemma | 797 | 0 | 0.0 | 0 |
| base | qwen | 595 | 0 | 0.0 | 0 |
| plainEF | gemma | 797 | 692 | 0.868 | 4 |
| plainEF | qwen | 595 | 278 | 0.467 | 4 |
| coherentEF | gemma | 797 | 696 | 0.873 | 4 |
| coherentEF | qwen | 595 | 234 | 0.393 | 3 |
| coherentEF2 | gemma | 797 | 756 | 0.949 | 4 |
| coherentEF2 | qwen | 595 | 342 | 0.575 | 4 |
| coherentRP | gemma | 797 | 653 | 0.819 | 3 |
| coherentRP | qwen | 595 | 522 | 0.877 | 4 |

| arm | server | mixed passes | SHAPE calls by rows | USB calls by rows | multi-row USB calls |
| --- | --- | ---: | --- | --- | ---: |
| base | cold | 0 | {} | {} | 0 |
| base | hot | 0 | {} | {} | 0 |
| plainEF | cold | 0 | {'1': 11760} | {'1': 11760} | 0 |
| plainEF | hot | 121 | {'1': 1692} | {'1': 1692} | 0 |
| coherentEF | cold | 0 | {'1': 11856} | {'1': 11856} | 0 |
| coherentEF | hot | 1 | {'1': 96, '2': 660} | {'1': 96, '2': 660} | 660 |
| coherentEF2 | cold | 0 | {'1': 13320} | {'1': 13320} | 0 |
| coherentEF2 | hot | 5 | {'1': 1962, '2': 66} | {'1': 1962, '2': 66} | 66 |
| coherentRP | cold | 0 | {'1': 15816} | {'1': 15816} | 0 |
| coherentRP | hot | 7 | {'1': 5423, '2': 1802} | {'1': 5423, '2': 1802} | 1802 |

| arm | model | window tokens by policy and active batch | same-model decode overlaps (s) |
| --- | --- | --- | --- |
| plainEF | gemma | {"host": {"1": 93}, "phone": {"1": 692}} | - |
| plainEF | qwen | {"host": {"1": 120, "2": 187}, "phone": {"1": 219, "2": 59}} | 004+003 111.8 |
| coherentEF | gemma | {"host": {"1": 89}, "phone": {"1": 696}} | - |
| coherentEF | qwen | {"host": {"1": 325, "2": 26}, "phone": {"1": 18, "2": 216}} | 003+004 72.7 |
| coherentEF2 | gemma | {"host": {"1": 28}, "phone": {"1": 756}} | - |
| coherentEF2 | qwen | {"host": {"1": 25, "2": 215}, "phone": {"1": 318, "2": 24}} | 003+004 77.1 |
| coherentRP | gemma | {"host": {"1": 132}, "phone": {"1": 653}} | - |
| coherentRP | qwen | {"host": {"1": 26, "2": 34}, "phone": {"1": 310, "2": 212}} | 004+003 68.7 |

| arm | model | policy | batch | scope | windows | tokens | fleet J/slot-token | fleet J/produced token | host J/produced token | ms/slot-token | rows/call |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| plainEF | gemma | host | 1 | all | 25 | 93 | 56.74 | 56.74 | 55.08 | 456.0 | - |
| plainEF | gemma | host | 1 | eligible | 19 | 73 | 59.03 | 59.03 | 57.08 | 475.7 | - |
| plainEF | gemma | phone | 1 | all | 175 | 692 | 33.9 | 33.9 | 33.12 | 425.9 | 1.0 |
| plainEF | gemma | phone | 1 | eligible | 159 | 636 | 33.8 | 33.8 | 33.02 | 428.1 | 1.0 |
| plainEF | qwen | host | 1 | all | 30 | 117 | 74.11 | 74.11 | 73.57 | 614.0 | - |
| plainEF | qwen | host | 1 | eligible | 21 | 84 | 75.24 | 75.24 | 74.7 | 616.5 | - |
| plainEF | qwen | host | 2 | all | 48 | 187 | 96.5 | 48.25 | 47.89 | 806.5 | - |
| plainEF | qwen | host | 2 | eligible | 41 | 164 | 99.8 | 49.9 | 49.53 | 835.8 | - |
| plainEF | qwen | phone | 1 | all | 56 | 219 | 60.42 | 60.42 | 59.72 | 576.0 | 1.0 |
| plainEF | qwen | phone | 1 | eligible | 44 | 176 | 59.67 | 59.67 | 58.96 | 573.5 | 1.0 |
| plainEF | qwen | phone | 2 | all | 20 | 59 | 145.24 | 72.62 | 72.01 | 1232.0 | 1.0 |
| plainEF | qwen | phone | 2 | eligible | 6 | 24 | 140.93 | 70.47 | 69.87 | 1199.2 | 1.0 |
| coherentEF | gemma | host | 1 | all | 24 | 89 | 56.97 | 56.97 | 55.42 | 456.0 | - |
| coherentEF | gemma | host | 1 | eligible | 15 | 57 | 59.44 | 59.44 | 57.65 | 476.6 | - |
| coherentEF | gemma | phone | 1 | all | 175 | 696 | 32.85 | 32.85 | 32.06 | 427.9 | 1.0 |
| coherentEF | gemma | phone | 1 | eligible | 165 | 660 | 33.1 | 33.1 | 32.32 | 429.1 | 1.0 |
| coherentEF | qwen | host | 1 | all | 81 | 321 | 74.98 | 74.98 | 74.44 | 617.1 | - |
| coherentEF | qwen | host | 1 | eligible | 74 | 296 | 75.03 | 75.03 | 74.49 | 615.1 | - |
| coherentEF | qwen | host | 2 | all | 7 | 26 | 73.93 | 36.97 | 36.71 | 591.7 | - |
| coherentEF | qwen | host | 2 | eligible | 2 | 8 | 75.88 | 37.94 | 37.67 | 615.3 | - |
| coherentEF | qwen | phone | 1 | all | 5 | 18 | 60.46 | 60.46 | 59.74 | 591.4 | 1.0 |
| coherentEF | qwen | phone | 1 | eligible | 1 | 4 | 61.78 | 61.78 | 61.03 | 618.0 | 1.0 |
| coherentEF | qwen | phone | 2 | all | 54 | 216 | 61.7 | 30.85 | 30.47 | 596.4 | 1.0 |
| coherentEF | qwen | phone | 2 | eligible | 49 | 196 | 61.34 | 30.67 | 30.3 | 595.0 | 1.0 |
| coherentEF2 | gemma | host | 1 | all | 8 | 28 | 52.67 | 52.67 | 52.29 | 431.5 | - |
| coherentEF2 | gemma | host | 1 | eligible | 2 | 8 | 59.93 | 59.93 | 59.5 | 486.8 | - |
| coherentEF2 | gemma | phone | 1 | all | 190 | 756 | 32.31 | 32.31 | 31.5 | 430.3 | 1.0 |
| coherentEF2 | gemma | phone | 1 | eligible | 183 | 732 | 32.39 | 32.39 | 31.58 | 430.6 | 1.0 |
| coherentEF2 | qwen | host | 1 | all | 7 | 25 | 73.15 | 73.15 | 72.61 | 610.2 | - |
| coherentEF2 | qwen | host | 1 | eligible | 1 | 4 | 74.1 | 74.1 | 73.57 | 607.0 | - |
| coherentEF2 | qwen | host | 2 | all | 55 | 215 | 77.41 | 38.71 | 38.43 | 632.9 | - |
| coherentEF2 | qwen | host | 2 | eligible | 48 | 192 | 77.16 | 38.58 | 38.3 | 631.0 | - |
| coherentEF2 | qwen | phone | 1 | all | 80 | 318 | 60.13 | 60.13 | 59.4 | 581.5 | 1.0 |
| coherentEF2 | qwen | phone | 1 | eligible | 74 | 296 | 60.12 | 60.12 | 59.4 | 581.5 | 1.0 |
| coherentEF2 | qwen | phone | 2 | all | 7 | 24 | 75.46 | 37.73 | 37.3 | 699.7 | 1.0 |
| coherentEF2 | qwen | phone | 2 | eligible | 1 | 4 | 62.23 | 31.11 | 30.75 | 578.9 | 1.0 |
| coherentRP | gemma | host | 1 | all | 35 | 132 | 57.9 | 57.9 | 56.44 | 463.5 | - |
| coherentRP | gemma | host | 1 | eligible | 25 | 97 | 59.51 | 59.51 | 57.89 | 477.4 | - |
| coherentRP | gemma | phone | 1 | all | 164 | 653 | 24.04 | 24.04 | 23.06 | 423.5 | 1.0 |
| coherentRP | gemma | phone | 1 | eligible | 150 | 600 | 24.14 | 24.14 | 23.16 | 426.6 | 1.0 |
| coherentRP | qwen | host | 1 | all | 7 | 26 | 72.5 | 72.5 | 71.95 | 625.3 | - |
| coherentRP | qwen | host | 1 | eligible | 2 | 8 | 75.6 | 75.6 | 75.01 | 666.7 | - |
| coherentRP | qwen | host | 2 | all | 10 | 34 | 78.14 | 39.07 | 38.77 | 669.8 | - |
| coherentRP | qwen | host | 2 | eligible | 2 | 8 | 77.0 | 38.5 | 38.21 | 661.2 | - |
| coherentRP | qwen | phone | 1 | all | 78 | 310 | 36.64 | 36.64 | 35.57 | 518.1 | 1.0 |
| coherentRP | qwen | phone | 1 | eligible | 68 | 272 | 37.09 | 37.09 | 36.01 | 525.6 | 1.0 |
| coherentRP | qwen | phone | 2 | all | 54 | 212 | 39.3 | 19.65 | 19.09 | 542.3 | 1.0 |
| coherentRP | qwen | phone | 2 | eligible | 46 | 184 | 37.39 | 18.7 | 18.15 | 525.5 | 1.0 |

- base: CANDIDATE_REQUALIFIED decisions {}; window ineligible reasons {}; raw {'RESULT.json': {}, 'ADAPTIVE_DECODE_OBSERVATIONS.json': {}, 'adaptive-timing-events.json': {}, 'SCHEDULER_DECISION_LOG.json': {}}; lockout {'helper_events_by_model': {}, 'phone_residency_events': {'PROPOSED': 1}}; coherence host decisions 0 {}, phone 0
- plainEF: CANDIDATE_REQUALIFIED decisions {}; window ineligible reasons {'gemma|phone|TOKEN_STREAM_CATCH_UP': 1}; raw {'RESULT.json': {'HELPER_PHONE_SESSION_LOAD': 32}, 'ADAPTIVE_DECODE_OBSERVATIONS.json': {'TOKEN_STREAM_CATCH_UP': 1, 'measurement_ineligible_reason': 1}, 'adaptive-timing-events.json': {}, 'SCHEDULER_DECISION_LOG.json': {'HELPER_PHONE_SESSION_LOAD': 32}}; lockout {'helper_events_by_model': {'gemma|ATTACHED': 4, 'gemma|DETACHED': 4, 'gemma|PREPARATION_READY': 3, 'qwen|ATTACHED': 4, 'qwen|DETACHED': 4, 'qwen|PREPARATION_READY': 1}, 'phone_residency_events': {'PROPOSED': 4, 'READY': 4, 'SESSION_READY': 4}}; coherence host decisions 0 {}, phone 4
- coherentEF: CANDIDATE_REQUALIFIED decisions {}; window ineligible reasons {'gemma|phone|TOKEN_STREAM_CATCH_UP': 2}; raw {'RESULT.json': {'HELPER_PHONE_SESSION_LOAD': 28}, 'ADAPTIVE_DECODE_OBSERVATIONS.json': {'TOKEN_STREAM_CATCH_UP': 2, 'measurement_ineligible_reason': 2}, 'adaptive-timing-events.json': {}, 'SCHEDULER_DECISION_LOG.json': {'HELPER_PHONE_SESSION_LOAD': 28}}; lockout {'helper_events_by_model': {'gemma|ATTACHED': 4, 'gemma|DETACHED': 4, 'gemma|PREPARATION_READY': 3, 'qwen|ATTACHED': 4, 'qwen|DETACHED': 4, 'qwen|PREPARATION_READY': 1}, 'phone_residency_events': {'PROPOSED': 4, 'READY': 4, 'SESSION_READY': 4}}; coherence host decisions 84 {'None': 56, 'SERVER_PAIR_NOT_IMPROVED': 28}, phone 235
- coherentEF2: CANDIDATE_REQUALIFIED decisions {}; window ineligible reasons {}; raw {'RESULT.json': {}, 'ADAPTIVE_DECODE_OBSERVATIONS.json': {}, 'adaptive-timing-events.json': {}, 'SCHEDULER_DECISION_LOG.json': {}}; lockout {'helper_events_by_model': {'gemma|ATTACHED': 4, 'gemma|DETACHED': 4, 'gemma|PREPARATION_READY': 3, 'qwen|ATTACHED': 4, 'qwen|DETACHED': 4, 'qwen|PREPARATION_READY': 1}, 'phone_residency_events': {'PROPOSED': 4, 'READY': 4, 'SESSION_READY': 4}}; coherence host decisions 56 {'SERVER_PAIR_NOT_IMPROVED': 48, 'SERVER_COMPARISON_HOST_WINDOW': 8}, phone 283
- coherentRP: CANDIDATE_REQUALIFIED decisions {}; window ineligible reasons {'gemma|phone|TOKEN_STREAM_CATCH_UP': 8, 'qwen|phone|TOKEN_STREAM_CATCH_UP': 2}; raw {'RESULT.json': {'HELPER_PHONE_SESSION_LOAD': 29}, 'ADAPTIVE_DECODE_OBSERVATIONS.json': {'TOKEN_STREAM_CATCH_UP': 10, 'measurement_ineligible_reason': 10}, 'adaptive-timing-events.json': {}, 'SCHEDULER_DECISION_LOG.json': {'HELPER_PHONE_SESSION_LOAD': 29}}; lockout {'helper_events_by_model': {'gemma|ATTACHED': 4, 'gemma|DETACHED': 4, 'gemma|PREPARATION_READY': 9, 'qwen|ATTACHED': 4, 'qwen|DETACHED': 4, 'qwen|PREPARATION_READY': 6}, 'phone_residency_events': {'PROPOSED': 15, 'READY': 15, 'SESSION_READY': 15}}; coherence host decisions 8 {'SERVER_COMPARISON_HOST_WINDOW': 8}, phone 300

- base decisions: {}
- plainEF decisions: {'gemma|HELPER_PHONE_SESSION_LOAD|host': 16, 'gemma|INITIAL_BASELINE|host': 2, 'gemma|PHONE_HELPER_UNAVAILABLE|host': 3, 'gemma|POLICY_CHANGE_REQUIRED|phone': 6, 'gemma|VERIFICATION_MONITORING|phone': 2, 'gemma|WINDOW_OPENED|phone': 171, 'qwen|CONTEXT_CHANGED|host': 4, 'qwen|EXECUTION_CONTEXT_UNAVAILABLE|host': 1, 'qwen|EXTERNAL_ACTIVITY_CHANGED|host': 21, 'qwen|INCUMBENT_NO_LONGER_BENEFICIAL|host': 12, 'qwen|INITIAL_BASELINE|host': 4, 'qwen|MEASURED_REJECTION|host': 18, 'qwen|POLICY_CHANGE_REQUIRED|phone': 18, 'qwen|PROBE_CANDIDATE_REJECTED|host': 2, 'qwen|PROBE_CANDIDATE_REJECTED|phone': 4, 'qwen|SERVER_POLICY_COHERENCE|phone': 4, 'qwen|VERIFICATION|host': 3, 'qwen|WINDOW_OPENED|phone': 64}
- coherentEF decisions: {'gemma|HELPER_PHONE_SESSION_LOAD|host': 15, 'gemma|INITIAL_BASELINE|host': 2, 'gemma|PHONE_HELPER_UNAVAILABLE|host': 3, 'gemma|POLICY_CHANGE_REQUIRED|phone': 2, 'gemma|SERVER_POLICY_COHERENCE|phone': 177, 'qwen|CONTEXT_CHANGED|host': 3, 'qwen|EXECUTION_CONTEXT_UNAVAILABLE|host': 1, 'qwen|INITIAL_BASELINE|host': 2, 'qwen|POLICY_CHANGE_REQUIRED|phone': 2, 'qwen|SERVER_POLICY_COHERENCE|host': 84, 'qwen|SERVER_POLICY_COHERENCE|phone': 58}
- coherentEF2 decisions: {'gemma|INITIAL_BASELINE|host': 4, 'gemma|POLICY_CHANGE_REQUIRED|phone': 2, 'gemma|SERVER_POLICY_COHERENCE|phone': 193, 'qwen|EXECUTION_CONTEXT_UNAVAILABLE|host': 1, 'qwen|INITIAL_BASELINE|host': 2, 'qwen|POLICY_CHANGE_REQUIRED|phone': 1, 'qwen|SERVER_POLICY_COHERENCE|host': 56, 'qwen|SERVER_POLICY_COHERENCE|phone': 90}
- coherentRP decisions: {'gemma|HELPER_PHONE_SESSION_LOAD|host': 15, 'gemma|INITIAL_BASELINE|host': 2, 'gemma|INSUFFICIENT_OPPORTUNITY|host': 7, 'gemma|PHONE_HELPER_UNAVAILABLE|host': 8, 'gemma|POLICY_CHANGE_REQUIRED|phone': 2, 'gemma|SERVER_POLICY_COHERENCE|phone': 165, 'qwen|EXECUTION_CONTEXT_UNAVAILABLE|host': 1, 'qwen|INITIAL_BASELINE|host': 4, 'qwen|POLICY_CHANGE_REQUIRED|phone': 2, 'qwen|SERVER_POLICY_COHERENCE|host': 8, 'qwen|SERVER_POLICY_COHERENCE|phone': 135}

| arm | request | model | out tokens | arrival s | acquired s | end s | phone calls |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | 00 | llama | 28 | 3.8 | 178.4 | 180.8 | 0 |
| base | 000 | gemma | 119 | 1.0 | 2.0 | 161.2 | 0 |
| base | 001 | gemma | 32 | 1.0 | 161.3 | 178.4 | 0 |
| base | 002 | qwen | 118 | 79.0 | 180.8 | 349.5 | 0 |
| base | 003 | qwen | 126 | 86.2 | 350.3 | 433.7 | 0 |
| base | 004 | qwen | 219 | 99.0 | 350.8 | 491.8 | 0 |
| base | 005 | gemma | 617 | 109.4 | 491.9 | 878.4 | 0 |
| base | 006 | qwen | 132 | 111.8 | 880.4 | 1053.3 | 0 |
| base | 007 | gemma | 29 | 127.8 | 1060.4 | 1148.7 | 0 |
| plainEF | 00 | llama | 28 | 3.8 | 82.6 | 85.1 | 0 |
| plainEF | 000 | gemma | 119 | 1.0 | 2.0 | 67.2 | 1008 |
| plainEF | 001 | gemma | 32 | 1.0 | 67.5 | 82.5 | 672 |
| plainEF | 002 | qwen | 118 | 79.0 | 85.2 | 243.5 | 636 |
| plainEF | 003 | qwen | 126 | 86.2 | 245.3 | 365.4 | 138 |
| plainEF | 004 | qwen | 219 | 99.0 | 245.3 | 421.4 | 216 |
| plainEF | 005 | gemma | 617 | 109.4 | 421.9 | 758.8 | 9680 |
| plainEF | 006 | qwen | 132 | 111.8 | 759.2 | 936.4 | 702 |
| plainEF | 007 | gemma | 29 | 127.8 | 937.9 | 1026.8 | 400 |
| coherentEF | 00 | llama | 28 | 3.8 | 83.8 | 86.7 | 0 |
| coherentEF | 000 | gemma | 119 | 1.0 | 2.0 | 68.3 | 1104 |
| coherentEF | 001 | gemma | 32 | 1.0 | 68.7 | 83.7 | 672 |
| coherentEF | 002 | qwen | 118 | 79.0 | 86.7 | 213.2 | 66 |
| coherentEF | 003 | qwen | 126 | 86.2 | 213.6 | 295.2 | 660 |
| coherentEF | 004 | qwen | 219 | 99.0 | 213.6 | 351.1 | 690 |
| coherentEF | 005 | gemma | 617 | 109.4 | 351.3 | 662.5 | 9680 |
| coherentEF | 006 | qwen | 132 | 111.8 | 663.0 | 832.4 | 0 |
| coherentEF | 007 | gemma | 29 | 127.8 | 833.7 | 916.0 | 400 |
| coherentEF2 | 00 | llama | 28 | 3.8 | 148.5 | 156.2 | 0 |
| coherentEF2 | 000 | gemma | 119 | 1.0 | 2.0 | 131.7 | 2568 |
| coherentEF2 | 001 | gemma | 32 | 1.0 | 132.0 | 148.4 | 672 |
| coherentEF2 | 002 | qwen | 118 | 79.0 | 156.3 | 309.2 | 636 |
| coherentEF2 | 003 | qwen | 126 | 86.2 | 309.5 | 396.1 | 66 |
| coherentEF2 | 004 | qwen | 219 | 99.0 | 311.1 | 451.7 | 624 |
| coherentEF2 | 005 | gemma | 617 | 109.4 | 452.1 | 770.6 | 9680 |
| coherentEF2 | 006 | qwen | 132 | 111.8 | 771.0 | 952.7 | 768 |
| coherentEF2 | 007 | gemma | 29 | 127.8 | 955.2 | 1039.9 | 400 |
| coherentRP | 00 | llama | 28 | 3.8 | 86.8 | 89.3 | 0 |
| coherentRP | 000 | gemma | 119 | 1.0 | 2.0 | 70.3 | 624 |
| coherentRP | 001 | gemma | 32 | 1.0 | 70.7 | 86.7 | 672 |
| coherentRP | 002 | qwen | 118 | 79.0 | 89.4 | 226.2 | 1802 |
| coherentRP | 003 | qwen | 126 | 86.2 | 231.1 | 308.8 | 1836 |
| coherentRP | 004 | qwen | 219 | 99.0 | 232.2 | 355.5 | 3349 |
| coherentRP | 005 | gemma | 617 | 109.4 | 356.2 | 674.9 | 14520 |
| coherentRP | 006 | qwen | 132 | 111.8 | 675.5 | 837.7 | 2040 |
| coherentRP | 007 | gemma | 29 | 127.8 | 838.1 | 895.9 | 0 |

| arm | request | model | windows | eligible (phone/host) | batches | host J/t | phone J/t | final ppm | host despite better phone | top decisions |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| plainEF | 000 | gemma | 30 | 26 (8/18) | [1] | 56.5 | 20.0 | 1000000 |  | HELPER_PHONE_SESSION_LOAD@0 x16; WINDOW_OPENED@1000000 x10; PHONE_HELPER_UNAVAILABLE@0 x3; POLICY_CHANGE_REQUIRED@1000000 x1 |
| plainEF | 001 | gemma | 8 | 4 (4/0) | [1] | 38.5 | 22.0 | 1000000 |  | WINDOW_OPENED@1000000 x7; VERIFICATION_MONITORING@1000000 x1 |
| plainEF | 002 | qwen | 30 | 19 (18/1) | [1] | 62.8 | 60.6 | 1000000 |  | WINDOW_OPENED@1000000 x17; INITIAL_BASELINE@0 x2; POLICY_CHANGE_REQUIRED@1000000 x2; WINDOW_OPENED@750000 x2 |
| plainEF | 003 | qwen | 34 | 24 (2/22) | [2] | 98.6 | 145.7 | 0 |  | INCUMBENT_NO_LONGER_BENEFICIAL@0 x12; MEASURED_REJECTION@0 x4; INITIAL_BASELINE@0 x2; WINDOW_OPENED@1000000 x2 |
| plainEF | 004 | qwen | 58 | 42 (4/38) | [1, 2] | 83.8 | 142.9 | 0 |  | EXTERNAL_ACTIVITY_CHANGED@0 x21; MEASURED_REJECTION@0 x14; POLICY_CHANGE_REQUIRED@1000000 x6; CONTEXT_CHANGED@0 x4 |
| plainEF | 005 | gemma | 155 | 144 (143/1) | [1] | 54.2 | 34.5 | 1000000 |  | WINDOW_OPENED@1000000 x142; INITIAL_BASELINE@0 x2; POLICY_CHANGE_REQUIRED@1000000 x2; WINDOW_OPENED@750000 x2 |
| plainEF | 006 | qwen | 33 | 27 (26/1) | [1] | 69.5 | 58.9 | 1000000 |  | WINDOW_OPENED@1000000 x28; VERIFICATION@0 x3; POLICY_CHANGE_REQUIRED@1000000 x2 |
| plainEF | 007 | gemma | 7 | 4 (4/0) | [1] | 39.6 | 31.6 | 1000000 |  | WINDOW_OPENED@1000000 x6; VERIFICATION_MONITORING@1000000 x1 |
| coherentEF | 000 | gemma | 30 | 23 (9/14) | [1] | 56.8 | 20.4 | 1000000 |  | HELPER_PHONE_SESSION_LOAD@0 x15; SERVER_POLICY_COHERENCE@1000000 x11; PHONE_HELPER_UNAVAILABLE@0 x3; POLICY_CHANGE_REQUIRED@1000000 x1 |
| coherentEF | 001 | gemma | 8 | 3 (3/0) | [1] | 39.7 | 22.6 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x8 |
| coherentEF | 002 | qwen | 30 | 25 (1/24) | [1] | 72.5 | 59.3 | 0 | YES | SERVER_POLICY_COHERENCE@0 x25; INITIAL_BASELINE@0 x2; SERVER_POLICY_COHERENCE@1000000 x2; POLICY_CHANGE_REQUIRED@1000000 x1 |
| coherentEF | 003 | qwen | 31 | 25 (24/1) | [1, 2] | 81.0 | 61.3 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x27; CONTEXT_CHANGED@0 x3; EXECUTION_CONTEXT_UNAVAILABLE@0 x1; POLICY_CHANGE_REQUIRED@1000000 x1 |
| coherentEF | 004 | qwen | 55 | 45 (25/20) | [1, 2] | 74.8 | 60.6 | 0 | YES | SERVER_POLICY_COHERENCE@1000000 x29; SERVER_POLICY_COHERENCE@0 x26 |
| coherentEF | 005 | gemma | 154 | 150 (149/1) | [1] | 55.1 | 33.3 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x151; INITIAL_BASELINE@0 x2; POLICY_CHANGE_REQUIRED@1000000 x1 |
| coherentEF | 006 | qwen | 33 | 31 (0/31) | [1] | 74.7 | None | 0 |  | SERVER_POLICY_COHERENCE@0 x33 |
| coherentEF | 007 | gemma | 7 | 4 (4/0) | [1] | 39.1 | 33.9 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x7 |
| coherentEF2 | 000 | gemma | 29 | 26 (25/1) | [1] | 55.6 | 23.4 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x27; INITIAL_BASELINE@0 x2; POLICY_CHANGE_REQUIRED@1000000 x1 |
| coherentEF2 | 001 | gemma | 8 | 5 (5/0) | [1] | 38.0 | 21.3 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x8 |
| coherentEF2 | 002 | qwen | 29 | 25 (24/1) | [1] | 66.7 | 58.5 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x26; INITIAL_BASELINE@0 x2; POLICY_CHANGE_REQUIRED@1000000 x1 |
| coherentEF2 | 003 | qwen | 32 | 25 (1/24) | [1, 2] | 78.4 | 62.1 | 0 | YES | SERVER_POLICY_COHERENCE@0 x28; SERVER_POLICY_COHERENCE@1000000 x3; EXECUTION_CONTEXT_UNAVAILABLE@0 x1 |
| coherentEF2 | 004 | qwen | 55 | 44 (20/24) | [1, 2] | 76.0 | 63.1 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x28; SERVER_POLICY_COHERENCE@0 x28 |
| coherentEF2 | 005 | gemma | 154 | 150 (149/1) | [1] | 55.4 | 33.3 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x151; INITIAL_BASELINE@0 x2; POLICY_CHANGE_REQUIRED@1000000 x1 |
| coherentEF2 | 006 | qwen | 33 | 30 (30/0) | [1] | 53.6 | 59.8 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x33 |
| coherentEF2 | 007 | gemma | 7 | 4 (4/0) | [1] | 43.1 | 32.3 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x7 |
| coherentRP | 000 | gemma | 30 | 23 (4/19) | [1] | 57.5 | 20.4 | 1000000 |  | HELPER_PHONE_SESSION_LOAD@0 x15; PHONE_HELPER_UNAVAILABLE@0 x8; SERVER_POLICY_COHERENCE@1000000 x6; POLICY_CHANGE_REQUIRED@1000000 x1 |
| coherentRP | 001 | gemma | 8 | 3 (3/0) | [1] | 39.5 | 22.2 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x8 |
| coherentRP | 002 | qwen | 29 | 25 (24/1) | [1] | 66.5 | 34.3 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x26; INITIAL_BASELINE@0 x2; POLICY_CHANGE_REQUIRED@1000000 x1 |
| coherentRP | 003 | qwen | 32 | 24 (23/1) | [2] | 74.4 | 39.3 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x29; SERVER_POLICY_COHERENCE@0 x4 |
| coherentRP | 004 | qwen | 55 | 42 (41/1) | [1, 2] | 82.7 | 36.2 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x50; SERVER_POLICY_COHERENCE@0 x4; EXECUTION_CONTEXT_UNAVAILABLE@0 x1 |
| coherentRP | 005 | gemma | 154 | 144 (143/1) | [1] | 54.7 | 23.2 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x151; INITIAL_BASELINE@0 x2; POLICY_CHANGE_REQUIRED@1000000 x1 |
| coherentRP | 006 | qwen | 33 | 27 (26/1) | [1] | 70.5 | 37.0 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x30; INITIAL_BASELINE@0 x2; POLICY_CHANGE_REQUIRED@1000000 x1 |
| coherentRP | 007 | gemma | 7 | 5 (0/5) | [1] | 55.2 | None | 0 |  | INSUFFICIENT_OPPORTUNITY@0 x7 |
