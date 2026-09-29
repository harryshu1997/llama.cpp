| arm | status | duration s | CPU kJ | GPU kJ | host kJ | host W | host kJ / duration vs earlier arms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| base | PASS | 1154.3 | 64.896 | 31.852 | 96.749 | 83.8 | - |
| plainEF | PASS | 1035.1 | 46.373 | 29.933 | 76.306 | 73.7 | base -21.1 % / -10.3 % |
| coherentEF | PASS | 924.5 | 43.502 | 28.355 | 71.858 | 77.7 | base -25.7 % / -19.9 %; plainEF -5.8 % / -10.7 % |
| plain | PASS | 963.8 | 51.164 | 30.65 | 81.814 | 84.9 | base -15.4 % / -16.5 %; plainEF +7.2 % / -6.9 %; coherentEF +13.9 % / +4.2 % |
| coherentsr | PASS | 1107.5 | 45.078 | 32.403 | 77.481 | 70.0 | base -19.9 % / -4.0 %; plainEF +1.5 % / +7.0 %; coherentEF +7.8 % / +19.8 %; plain -5.3 % / +14.9 % |
| coherent | PASS | 1088.3 | 44.595 | 32.738 | 77.333 | 71.1 | base -20.1 % / -5.7 %; plainEF +1.4 % / +5.1 %; coherentEF +7.6 % / +17.7 %; plain -5.5 % / +12.9 %; coherentsr -0.2 % / -1.7 % |

| arm | model | output tokens | phone-policy window tokens | share | requests with phone calls |
| --- | --- | ---: | ---: | ---: | ---: |
| base | gemma | 797 | 0 | 0.0 | 0 |
| base | qwen | 595 | 0 | 0.0 | 0 |
| plainEF | gemma | 797 | 692 | 0.868 | 4 |
| plainEF | qwen | 595 | 278 | 0.467 | 4 |
| coherentEF | gemma | 797 | 696 | 0.873 | 4 |
| coherentEF | qwen | 595 | 234 | 0.393 | 3 |
| plain | gemma | 797 | 666 | 0.836 | 3 |
| plain | qwen | 595 | 112 | 0.188 | 4 |
| coherentsr | gemma | 797 | 692 | 0.868 | 3 |
| coherentsr | qwen | 595 | 338 | 0.568 | 4 |
| coherent | gemma | 797 | 661 | 0.829 | 3 |
| coherent | qwen | 595 | 0 | 0.0 | 0 |

| arm | server | mixed passes | SHAPE calls by rows | USB calls by rows | multi-row USB calls |
| --- | --- | ---: | --- | --- | ---: |
| base | cold | 0 | {} | {} | 0 |
| base | hot | 0 | {} | {} | 0 |
| plainEF | cold | 0 | {'1': 11760} | {'1': 11760} | 0 |
| plainEF | hot | 121 | {'1': 1692} | {'1': 1692} | 0 |
| coherentEF | cold | 0 | {'1': 11856} | {'1': 11856} | 0 |
| coherentEF | hot | 1 | {'1': 96, '2': 660} | {'1': 96, '2': 660} | 660 |
| plain | cold | 0 | {'1': 10488} | {'1': 10488} | 0 |
| plain | hot | 109 | {'1': 672} | {'1': 672} | 0 |
| coherentsr | cold | 0 | {'1': 10856} | {'1': 10856} | 0 |
| coherentsr | hot | 5 | {'1': 1932, '2': 66} | {'1': 2064} | 0 |
| coherent | cold | 0 | {'1': 15400} | {'1': 15400} | 0 |
| coherent | hot | 0 | {} | {} | 0 |

| arm | model | window tokens by policy and active batch | same-model decode overlaps (s) |
| --- | --- | --- | --- |
| plainEF | gemma | {"host": {"1": 93}, "phone": {"1": 692}} | - |
| plainEF | qwen | {"host": {"1": 120, "2": 187}, "phone": {"1": 219, "2": 59}} | 004+003 111.8 |
| coherentEF | gemma | {"host": {"1": 89}, "phone": {"1": 696}} | - |
| coherentEF | qwen | {"host": {"1": 325, "2": 26}, "phone": {"1": 18, "2": 216}} | 003+004 72.7 |
| plain | gemma | {"host": {"1": 119}, "phone": {"1": 666}} | - |
| plain | qwen | {"host": {"1": 282, "2": 191}, "phone": {"1": 58, "2": 54}} | 003+004 107.0 |
| coherentsr | gemma | {"host": {"1": 93}, "phone": {"1": 692}} | - |
| coherentsr | qwen | {"host": {"1": 21, "2": 224}, "phone": {"1": 314, "2": 24}} | 004+003 79.8 |
| coherent | gemma | {"host": {"1": 124}, "phone": {"1": 661}} | - |
| coherent | qwen | {"host": {"1": 340, "2": 243}} | 003+004 77.4 |

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
| plain | gemma | host | 1 | all | 31 | 118 | 59.71 | 59.71 | 58.63 | 468.4 | - |
| plain | gemma | host | 1 | eligible | 24 | 93 | 62.35 | 62.35 | 61.19 | 490.2 | - |
| plain | gemma | phone | 1 | all | 169 | 666 | 36.41 | 36.41 | 35.58 | 436.7 | 1.0 |
| plain | gemma | phone | 1 | eligible | 150 | 600 | 35.11 | 35.11 | 34.32 | 436.3 | 1.0 |
| plain | qwen | host | 1 | all | 71 | 279 | 76.9 | 76.9 | 76.36 | 626.1 | - |
| plain | qwen | host | 1 | eligible | 60 | 240 | 77.07 | 77.07 | 76.52 | 625.5 | - |
| plain | qwen | host | 2 | all | 49 | 191 | 96.44 | 48.22 | 47.87 | 785.8 | - |
| plain | qwen | host | 2 | eligible | 43 | 172 | 98.85 | 49.43 | 49.07 | 807.8 | - |
| plain | qwen | phone | 1 | all | 16 | 58 | 65.82 | 65.82 | 65.15 | 580.6 | 1.0 |
| plain | qwen | phone | 1 | eligible | 6 | 24 | 65.71 | 65.71 | 65.01 | 590.5 | 1.0 |
| plain | qwen | phone | 2 | all | 18 | 54 | 145.45 | 72.72 | 72.13 | 1194.2 | 1.0 |
| plain | qwen | phone | 2 | eligible | 6 | 24 | 143.98 | 71.99 | 71.4 | 1175.9 | 1.0 |
| coherentsr | gemma | host | 1 | all | 24 | 92 | 57.7 | 57.7 | 56.88 | 457.2 | - |
| coherentsr | gemma | host | 1 | eligible | 17 | 68 | 59.99 | 59.99 | 59.16 | 477.6 | - |
| coherentsr | gemma | phone | 1 | all | 174 | 692 | 38.51 | 38.51 | 37.5 | 490.1 | 1.0 |
| coherentsr | gemma | phone | 1 | eligible | 164 | 656 | 38.41 | 38.41 | 37.42 | 491.3 | 1.0 |
| coherentsr | qwen | host | 1 | all | 6 | 21 | 71.28 | 71.28 | 70.75 | 597.2 | - |
| coherentsr | qwen | host | 1 | eligible | 1 | 4 | 73.85 | 73.85 | 73.32 | 606.1 | - |
| coherentsr | qwen | host | 2 | all | 57 | 224 | 78.73 | 39.37 | 39.09 | 636.1 | - |
| coherentsr | qwen | host | 2 | eligible | 50 | 200 | 78.4 | 39.2 | 38.92 | 631.7 | - |
| coherentsr | qwen | phone | 1 | all | 79 | 314 | 60.38 | 60.38 | 59.62 | 582.2 | 1.0 |
| coherentsr | qwen | phone | 1 | eligible | 72 | 288 | 60.31 | 60.31 | 59.55 | 582.0 | 1.0 |
| coherentsr | qwen | phone | 2 | all | 7 | 24 | 78.53 | 39.26 | 38.61 | 753.5 | 1.0 |
| coherentsr | qwen | phone | 2 | eligible | 1 | 4 | 64.36 | 32.18 | 31.59 | 645.9 | 1.0 |
| coherent | gemma | host | 1 | all | 32 | 123 | 59.23 | 59.23 | 58.33 | 467.0 | - |
| coherent | gemma | host | 1 | eligible | 23 | 92 | 61.09 | 61.09 | 60.23 | 481.4 | - |
| coherent | gemma | phone | 1 | all | 166 | 661 | 27.24 | 27.24 | 26.05 | 467.2 | 1.0 |
| coherent | gemma | phone | 1 | eligible | 159 | 636 | 26.94 | 26.94 | 25.77 | 467.8 | 1.0 |
| coherent | qwen | host | 1 | all | 85 | 339 | 76.78 | 76.78 | 76.24 | 618.2 | - |
| coherent | qwen | host | 1 | eligible | 77 | 308 | 76.09 | 76.09 | 75.56 | 612.2 | - |
| coherent | qwen | host | 2 | all | 61 | 243 | 79.31 | 39.65 | 39.38 | 636.8 | - |
| coherent | qwen | host | 2 | eligible | 57 | 228 | 79.52 | 39.76 | 39.48 | 638.6 | - |

- base: CANDIDATE_REQUALIFIED decisions {}; window ineligible reasons {}; raw {'RESULT.json': {}, 'ADAPTIVE_DECODE_OBSERVATIONS.json': {}, 'adaptive-timing-events.json': {}, 'SCHEDULER_DECISION_LOG.json': {}}; lockout {'helper_events_by_model': {}, 'phone_residency_events': {'PROPOSED': 1}}; coherence host decisions 0 {}, phone 0
- plainEF: CANDIDATE_REQUALIFIED decisions {}; window ineligible reasons {'gemma|phone|TOKEN_STREAM_CATCH_UP': 1}; raw {'RESULT.json': {'HELPER_PHONE_SESSION_LOAD': 32}, 'ADAPTIVE_DECODE_OBSERVATIONS.json': {'TOKEN_STREAM_CATCH_UP': 1, 'measurement_ineligible_reason': 1}, 'adaptive-timing-events.json': {}, 'SCHEDULER_DECISION_LOG.json': {'HELPER_PHONE_SESSION_LOAD': 32}}; lockout {'helper_events_by_model': {'gemma|ATTACHED': 4, 'gemma|DETACHED': 4, 'gemma|PREPARATION_READY': 3, 'qwen|ATTACHED': 4, 'qwen|DETACHED': 4, 'qwen|PREPARATION_READY': 1}, 'phone_residency_events': {'PROPOSED': 4, 'READY': 4, 'SESSION_READY': 4}}; coherence host decisions 0 {}, phone 4
- coherentEF: CANDIDATE_REQUALIFIED decisions {}; window ineligible reasons {'gemma|phone|TOKEN_STREAM_CATCH_UP': 2}; raw {'RESULT.json': {'HELPER_PHONE_SESSION_LOAD': 28}, 'ADAPTIVE_DECODE_OBSERVATIONS.json': {'TOKEN_STREAM_CATCH_UP': 2, 'measurement_ineligible_reason': 2}, 'adaptive-timing-events.json': {}, 'SCHEDULER_DECISION_LOG.json': {'HELPER_PHONE_SESSION_LOAD': 28}}; lockout {'helper_events_by_model': {'gemma|ATTACHED': 4, 'gemma|DETACHED': 4, 'gemma|PREPARATION_READY': 3, 'qwen|ATTACHED': 4, 'qwen|DETACHED': 4, 'qwen|PREPARATION_READY': 1}, 'phone_residency_events': {'PROPOSED': 4, 'READY': 4, 'SESSION_READY': 4}}; coherence host decisions 84 {'None': 56, 'SERVER_PAIR_NOT_IMPROVED': 28}, phone 235
- plain: CANDIDATE_REQUALIFIED decisions {}; window ineligible reasons {}; raw {'RESULT.json': {}, 'ADAPTIVE_DECODE_OBSERVATIONS.json': {}, 'adaptive-timing-events.json': {}, 'SCHEDULER_DECISION_LOG.json': {}}; lockout {'helper_events_by_model': {'gemma|ATTACHED': 4, 'gemma|DETACHED': 4, 'gemma|PREPARATION_READY': 3, 'qwen|ATTACHED': 4, 'qwen|DETACHED': 4, 'qwen|PREPARATION_READY': 1}, 'phone_residency_events': {'PROPOSED': 4, 'READY': 4, 'SESSION_READY': 4}}; coherence host decisions 0 {}, phone 0
- coherentsr: CANDIDATE_REQUALIFIED decisions {}; window ineligible reasons {}; raw {'RESULT.json': {}, 'ADAPTIVE_DECODE_OBSERVATIONS.json': {}, 'adaptive-timing-events.json': {}, 'SCHEDULER_DECISION_LOG.json': {}}; lockout {'helper_events_by_model': {'gemma|ATTACHED': 4, 'gemma|DETACHED': 4, 'gemma|PREPARATION_READY': 3, 'qwen|ATTACHED': 4, 'qwen|DETACHED': 4, 'qwen|PREPARATION_READY': 1}, 'phone_residency_events': {'PROPOSED': 4, 'READY': 4, 'SESSION_READY': 4}}; coherence host decisions 57 {'SERVER_PAIR_NOT_IMPROVED': 49, 'SERVER_COMPARISON_HOST_WINDOW': 8}, phone 261
- coherent: CANDIDATE_REQUALIFIED decisions {}; window ineligible reasons {}; raw {'RESULT.json': {}, 'ADAPTIVE_DECODE_OBSERVATIONS.json': {}, 'adaptive-timing-events.json': {}, 'SCHEDULER_DECISION_LOG.json': {}}; lockout {'helper_events_by_model': {'gemma|ATTACHED': 4, 'gemma|DETACHED': 4, 'gemma|PREPARATION_READY': 3, 'qwen|PREPARATION_FAILED': 43}, 'phone_residency_events': {'PROPOSED': 47, 'READY': 3, 'SESSION_FAILED': 43, 'SESSION_READY': 3, 'TRANSITION_FAILED': 43}}; coherence host decisions 0 {}, phone 166

- base decisions: {}
- plainEF decisions: {'gemma|HELPER_PHONE_SESSION_LOAD|host': 16, 'gemma|INITIAL_BASELINE|host': 2, 'gemma|PHONE_HELPER_UNAVAILABLE|host': 3, 'gemma|POLICY_CHANGE_REQUIRED|phone': 6, 'gemma|VERIFICATION_MONITORING|phone': 2, 'gemma|WINDOW_OPENED|phone': 171, 'qwen|CONTEXT_CHANGED|host': 4, 'qwen|EXECUTION_CONTEXT_UNAVAILABLE|host': 1, 'qwen|EXTERNAL_ACTIVITY_CHANGED|host': 21, 'qwen|INCUMBENT_NO_LONGER_BENEFICIAL|host': 12, 'qwen|INITIAL_BASELINE|host': 4, 'qwen|MEASURED_REJECTION|host': 18, 'qwen|POLICY_CHANGE_REQUIRED|phone': 18, 'qwen|PROBE_CANDIDATE_REJECTED|host': 2, 'qwen|PROBE_CANDIDATE_REJECTED|phone': 4, 'qwen|SERVER_POLICY_COHERENCE|phone': 4, 'qwen|VERIFICATION|host': 3, 'qwen|WINDOW_OPENED|phone': 64}
- coherentEF decisions: {'gemma|HELPER_PHONE_SESSION_LOAD|host': 15, 'gemma|INITIAL_BASELINE|host': 2, 'gemma|PHONE_HELPER_UNAVAILABLE|host': 3, 'gemma|POLICY_CHANGE_REQUIRED|phone': 2, 'gemma|SERVER_POLICY_COHERENCE|phone': 177, 'qwen|CONTEXT_CHANGED|host': 3, 'qwen|EXECUTION_CONTEXT_UNAVAILABLE|host': 1, 'qwen|INITIAL_BASELINE|host': 2, 'qwen|POLICY_CHANGE_REQUIRED|phone': 2, 'qwen|SERVER_POLICY_COHERENCE|host': 84, 'qwen|SERVER_POLICY_COHERENCE|phone': 58}
- plain decisions: {'gemma|INCONCLUSIVE|host': 6, 'gemma|INITIAL_BASELINE|host': 2, 'gemma|INSUFFICIENT_OPPORTUNITY|host': 17, 'gemma|PHONE_HELPER_UNAVAILABLE|host': 3, 'gemma|POLICY_CHANGE_REQUIRED|phone': 10, 'gemma|VERIFICATION_INCONCLUSIVE|host': 1, 'gemma|VERIFICATION_MONITORING|phone': 1, 'gemma|WINDOW_OPENED|phone': 161, 'qwen|CONTEXT_CHANGED|host': 4, 'qwen|EXECUTION_CONTEXT_UNAVAILABLE|host': 1, 'qwen|EXTERNAL_ACTIVITY_CHANGED|host': 21, 'qwen|INCONCLUSIVE|host': 25, 'qwen|INCUMBENT_NO_LONGER_BENEFICIAL|host': 17, 'qwen|INITIAL_BASELINE|host': 4, 'qwen|MEASURED_REJECTION|host': 34, 'qwen|POLICY_CHANGE_REQUIRED|phone': 17, 'qwen|PROBE_CANDIDATE_REJECTED|host': 2, 'qwen|PROBE_CANDIDATE_REJECTED|phone': 4, 'qwen|VERIFICATION|host': 3, 'qwen|VERIFICATION_INCONCLUSIVE|host': 1, 'qwen|WINDOW_OPENED|phone': 22}
- coherentsr decisions: {'gemma|INITIAL_BASELINE|host': 4, 'gemma|INSUFFICIENT_OPPORTUNITY|host': 19, 'gemma|POLICY_CHANGE_REQUIRED|phone': 3, 'gemma|SERVER_POLICY_COHERENCE|phone': 173, 'qwen|EXECUTION_CONTEXT_UNAVAILABLE|host': 1, 'qwen|INITIAL_BASELINE|host': 2, 'qwen|POLICY_CHANGE_REQUIRED|phone': 1, 'qwen|SERVER_POLICY_COHERENCE|host': 57, 'qwen|SERVER_POLICY_COHERENCE|phone': 88}
- coherent decisions: {'gemma|INITIAL_BASELINE|host': 2, 'gemma|INSUFFICIENT_OPPORTUNITY|host': 25, 'gemma|PHONE_HELPER_UNAVAILABLE|host': 4, 'gemma|POLICY_CHANGE_REQUIRED|phone': 2, 'gemma|SERVER_POLICY_COHERENCE|phone': 166, 'qwen|PHONE_HELPER_UNAVAILABLE|host': 147}

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
| plain | 00 | llama | 28 | 3.8 | 86.9 | 88.4 | 0 |
| plain | 000 | gemma | 119 | 1.0 | 2.0 | 69.2 | 408 |
| plain | 001 | gemma | 32 | 1.0 | 69.5 | 86.9 | 0 |
| plain | 002 | qwen | 118 | 79.0 | 88.4 | 206.2 | 240 |
| plain | 003 | qwen | 126 | 86.2 | 206.5 | 322.0 | 108 |
| plain | 004 | qwen | 219 | 99.0 | 207.1 | 379.8 | 216 |
| plain | 005 | gemma | 617 | 109.4 | 380.2 | 705.3 | 9680 |
| plain | 006 | qwen | 132 | 111.8 | 705.6 | 855.9 | 108 |
| plain | 007 | gemma | 29 | 127.8 | 857.3 | 954.2 | 400 |
| coherentsr | 00 | llama | 28 | 3.8 | 96.3 | 98.0 | 0 |
| coherentsr | 000 | gemma | 119 | 1.0 | 2.0 | 78.5 | 776 |
| coherentsr | 001 | gemma | 32 | 1.0 | 78.8 | 96.3 | 0 |
| coherentsr | 002 | qwen | 118 | 79.0 | 98.0 | 250.1 | 636 |
| coherentsr | 003 | qwen | 126 | 86.2 | 253.5 | 341.5 | 78 |
| coherentsr | 004 | qwen | 219 | 99.0 | 254.0 | 393.7 | 582 |
| coherentsr | 005 | gemma | 617 | 109.4 | 394.1 | 785.2 | 9680 |
| coherentsr | 006 | qwen | 132 | 111.8 | 794.4 | 1003.7 | 768 |
| coherentsr | 007 | gemma | 29 | 127.8 | 1005.1 | 1098.6 | 400 |
| coherent | 00 | llama | 28 | 3.8 | 90.6 | 92.7 | 0 |
| coherent | 000 | gemma | 119 | 1.0 | 2.1 | 72.4 | 280 |
| coherent | 001 | gemma | 32 | 1.0 | 72.7 | 90.3 | 0 |
| coherent | 002 | qwen | 118 | 79.0 | 92.8 | 270.5 | 0 |
| coherent | 003 | qwen | 126 | 86.2 | 270.8 | 356.4 | 0 |
| coherent | 004 | qwen | 219 | 99.0 | 272.3 | 414.1 | 0 |
| coherent | 005 | gemma | 617 | 109.4 | 444.1 | 824.5 | 14520 |
| coherent | 006 | qwen | 132 | 111.8 | 825.1 | 987.9 | 0 |
| coherent | 007 | gemma | 29 | 127.8 | 988.9 | 1078.8 | 600 |

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
| plain | 000 | gemma | 31 | 20 (3/17) | [1] | 59.7 | 52.6 | 0 | YES | INSUFFICIENT_OPPORTUNITY@0 x9; INCONCLUSIVE@0 x6; PHONE_HELPER_UNAVAILABLE@0 x3; WINDOW_OPENED@250000 x3 |
| plain | 001 | gemma | 8 | 6 (0/6) | [1] | 58.4 | None | 0 |  | INSUFFICIENT_OPPORTUNITY@0 x8 |
| plain | 002 | qwen | 30 | 20 (4/16) | [1] | 74.9 | 66.8 | 0 | YES | INCUMBENT_NO_LONGER_BENEFICIAL@0 x17; WINDOW_OPENED@1000000 x3; INITIAL_BASELINE@0 x2; POLICY_CHANGE_REQUIRED@1000000 x2 |
| plain | 003 | qwen | 33 | 25 (2/23) | [1, 2] | 101.2 | 144.6 | 0 |  | MEASURED_REJECTION@0 x14; POLICY_CHANGE_REQUIRED@1000000 x10; CONTEXT_CHANGED@0 x2; WINDOW_OPENED@1000000 x2 |
| plain | 004 | qwen | 58 | 43 (4/39) | [1, 2] | 83.2 | 144.1 | 0 |  | EXTERNAL_ACTIVITY_CHANGED@0 x21; MEASURED_REJECTION@0 x20; INITIAL_BASELINE@0 x2; WINDOW_OPENED@1000000 x2 |
| plain | 005 | gemma | 155 | 144 (143/1) | [1] | 57.2 | 34.6 | 1000000 |  | WINDOW_OPENED@1000000 x142; INITIAL_BASELINE@0 x2; POLICY_CHANGE_REQUIRED@1000000 x2; WINDOW_OPENED@750000 x2 |
| plain | 006 | qwen | 34 | 27 (2/25) | [1] | 76.1 | 61.4 | 0 | YES | INCONCLUSIVE@0 x25; WINDOW_OPENED@1000000 x3; VERIFICATION@0 x3; POLICY_CHANGE_REQUIRED@1000000 x2 |
| plain | 007 | gemma | 7 | 4 (4/0) | [1] | 40.0 | 31.8 | 1000000 |  | WINDOW_OPENED@1000000 x6; VERIFICATION_MONITORING@1000000 x1 |
| coherentsr | 000 | gemma | 30 | 21 (11/10) | [1] | 58.0 | 40.2 | 0 | YES | SERVER_POLICY_COHERENCE@1000000 x15; INSUFFICIENT_OPPORTUNITY@0 x11; INITIAL_BASELINE@0 x2; POLICY_CHANGE_REQUIRED@1000000 x2 |
| coherentsr | 001 | gemma | 8 | 6 (0/6) | [1] | 57.6 | None | 0 |  | INSUFFICIENT_OPPORTUNITY@0 x8 |
| coherentsr | 002 | qwen | 29 | 25 (24/1) | [1] | 67.1 | 58.7 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x26; INITIAL_BASELINE@0 x2; POLICY_CHANGE_REQUIRED@1000000 x1 |
| coherentsr | 003 | qwen | 32 | 25 (0/25) | [2] | 77.3 | 88.1 | 0 |  | SERVER_POLICY_COHERENCE@0 x28; SERVER_POLICY_COHERENCE@1000000 x4 |
| coherentsr | 004 | qwen | 55 | 44 (19/25) | [1, 2] | 79.4 | 60.9 | 1000000 |  | SERVER_POLICY_COHERENCE@0 x29; SERVER_POLICY_COHERENCE@1000000 x25; EXECUTION_CONTEXT_UNAVAILABLE@0 x1 |
| coherentsr | 005 | gemma | 154 | 150 (149/1) | [1] | 54.8 | 37.4 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x151; INITIAL_BASELINE@0 x2; POLICY_CHANGE_REQUIRED@1000000 x1 |
| coherentsr | 006 | qwen | 33 | 30 (30/0) | [1] | 52.4 | 59.8 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x33 |
| coherentsr | 007 | gemma | 7 | 4 (4/0) | [1] | 39.6 | 32.2 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x7 |
| coherent | 000 | gemma | 30 | 22 (6/16) | [1] | 58.6 | 45.2 | 0 | YES | INSUFFICIENT_OPPORTUNITY@0 x17; SERVER_POLICY_COHERENCE@1000000 x8; PHONE_HELPER_UNAVAILABLE@0 x4; POLICY_CHANGE_REQUIRED@1000000 x1 |
| coherent | 001 | gemma | 8 | 6 (0/6) | [1] | 60.1 | None | 0 |  | INSUFFICIENT_OPPORTUNITY@0 x8 |
| coherent | 002 | qwen | 29 | 27 (0/27) | [1] | 76.5 | None | 0 |  | PHONE_HELPER_UNAVAILABLE@0 x29 |
| coherent | 003 | qwen | 31 | 28 (0/28) | [1, 2] | 79.7 | None | 0 |  | PHONE_HELPER_UNAVAILABLE@0 x31 |
| coherent | 004 | qwen | 54 | 48 (0/48) | [1, 2] | 77.6 | None | 0 |  | PHONE_HELPER_UNAVAILABLE@0 x54 |
| coherent | 005 | gemma | 154 | 150 (149/1) | [1] | 55.6 | 25.1 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x151; INITIAL_BASELINE@0 x2; POLICY_CHANGE_REQUIRED@1000000 x1 |
| coherent | 006 | qwen | 33 | 31 (0/31) | [1] | 75.1 | None | 0 |  | PHONE_HELPER_UNAVAILABLE@0 x33 |
| coherent | 007 | gemma | 7 | 4 (4/0) | [1] | 43.5 | 21.5 | 1000000 |  | SERVER_POLICY_COHERENCE@1000000 x7 |
