### plainEF

- event kinds {'EVALUATED': 32, 'PREPARING': 4, 'PROPOSED': 4, 'READY': 4, 'SELECTION_CONFIRMED': 1, 'SELECTION_OBSERVED': 2, 'SESSION_DRAINING': 1, 'SESSION_LOADING': 4, 'SESSION_READY': 4, 'SESSION_VERIFIED': 4}
- EVALUATED reasons {'PHONE_RESIDENCY_DEMAND_UNAVAILABLE': 1, 'PHONE_RESIDENCY_LEARNING_EXPLORATION': 4, 'PHONE_RESIDENCY_LEARNING_RETAINED': 14, 'PHONE_RESIDENCY_REVALIDATION_UNAFFORDABLE': 10, 'PHONE_RESIDENCY_TRANSITION_IN_PROGRESS': 3}
- helper deferrals {}; WAITING_FOR_HELPER_RELEASE 0
- per model phone layers while decoding {'gemma': {'decode_union_s': 337.1, 'mean_phone_layers_while_decoding': 16.01}, 'qwen3': {'decode_union_s': 311.6, 'mean_phone_layers_while_decoding': 6.0}}

| READY s | gen | changed | layers by model | sessions by model | selection |
| ---: | ---: | --- | --- | --- | --- |
| 20.4 | 1 | HTP0 | {'gemma': 8} | {'gemma': 1} | PHONE_RESIDENCY_LEARNING_EXPLORATION |
| 36.9 | 2 | HTP1 | {'gemma': 16} | {'gemma': 2} | PHONE_RESIDENCY_LEARNING_EXPLORATION |
| 49.4 | 3 | HTP2 | {'gemma': 24} | {'gemma': 3} | PHONE_RESIDENCY_LEARNING_EXPLORATION |
| 103.0 | 4 | HTP2 | {'gemma': 16, 'qwen3': 6} | {'gemma': 2, 'qwen3': 1} | PHONE_RESIDENCY_LEARNING_EXPLORATION |

| session | gen | model | GB | loading s | verified s | s | MB/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| HTP0 | 1 | gemma | 2.83 | 2.0 | 20.4 | 18.3 | 154.3 |
| HTP1 | 2 | gemma | 2.83 | 20.7 | 36.9 | 16.2 | 174.6 |
| HTP2 | 3 | gemma | 2.83 | 37.5 | 49.4 | 11.9 | 237.3 |
| HTP2 | 4 | qwen3 | 3.21 | 83.8 | 103.0 | 19.2 | 166.9 |

| request | model | out | acquired s | exec start s | first/last window s | end s | layers at first window | max layers while decoding | layers called | phone calls | phone/all window tokens |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 00 | llama | 28 | 82.6 | 84.9 | None/None | 85.1 | - | - | 0 | 0 | -/- |
| 000 | gemma | 119 | 2.0 | 7.5 | 15.0/66.4 | 67.2 | 0 | 24 | 24 | 1008 | 40/116 |
| 001 | gemma | 32 | 67.5 | 67.8 | 70.8/81.7 | 82.5 | 24 | 24 | 24 | 672 | 26/29 |
| 002 | qwen3 | 118 | 85.2 | 173.1 | 176.4/242.4 | 243.5 | 6 | 6 | 6 | 636 | 104/115 |
| 003 | qwen3 | 126 | 245.3 | 246.5 | 252.4/364.2 | 365.4 | 6 | 6 | 6 | 138 | 23/123 |
| 004 | qwen3 | 219 | 245.3 | 246.6 | 250.3/421.3 | 421.4 | 6 | 6 | 6 | 216 | 36/218 |
| 005 | gemma | 617 | 421.9 | 484.0 | 494.2/757.9 | 758.8 | 16 | 16 | 16 | 9680 | 603/614 |
| 006 | qwen3 | 132 | 759.2 | 852.7 | 860.7/935.3 | 936.4 | 6 | 6 | 6 | 702 | 115/129 |
| 007 | gemma | 29 | 937.9 | 1009.7 | 1014.8/1025.9 | 1026.8 | 16 | 16 | 16 | 400 | 23/26 |

### coherentEF

- event kinds {'EVALUATED': 35, 'PREPARING': 4, 'PROPOSED': 4, 'READY': 4, 'SELECTION_CONFIRMED': 1, 'SELECTION_OBSERVED': 2, 'SESSION_DRAINING': 1, 'SESSION_LOADING': 4, 'SESSION_READY': 4, 'SESSION_VERIFIED': 4}
- EVALUATED reasons {'PHONE_RESIDENCY_DEMAND_UNAVAILABLE': 1, 'PHONE_RESIDENCY_LEARNING_EXPLORATION': 4, 'PHONE_RESIDENCY_LEARNING_RETAINED': 15, 'PHONE_RESIDENCY_REVALIDATION_UNAFFORDABLE': 13, 'PHONE_RESIDENCY_TRANSITION_IN_PROGRESS': 2}
- helper deferrals {}; WAITING_FOR_HELPER_RELEASE 0
- per model phone layers while decoding {'gemma': {'decode_union_s': 338.4, 'mean_phone_layers_while_decoding': 16.3}, 'qwen3': {'decode_union_s': 283.3, 'mean_phone_layers_while_decoding': 6.0}}

| READY s | gen | changed | layers by model | sessions by model | selection |
| ---: | ---: | --- | --- | --- | --- |
| 20.0 | 1 | HTP0 | {'gemma': 8} | {'gemma': 1} | PHONE_RESIDENCY_LEARNING_EXPLORATION |
| 31.7 | 2 | HTP1 | {'gemma': 16} | {'gemma': 2} | PHONE_RESIDENCY_LEARNING_EXPLORATION |
| 46.2 | 3 | HTP2 | {'gemma': 24} | {'gemma': 3} | PHONE_RESIDENCY_LEARNING_EXPLORATION |
| 98.5 | 4 | HTP2 | {'gemma': 16, 'qwen3': 6} | {'gemma': 2, 'qwen3': 1} | PHONE_RESIDENCY_LEARNING_EXPLORATION |

| session | gen | model | GB | loading s | verified s | s | MB/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| HTP0 | 1 | gemma | 2.83 | 2.0 | 20.0 | 18.0 | 157.4 |
| HTP1 | 2 | gemma | 2.83 | 20.4 | 31.7 | 11.3 | 251.3 |
| HTP2 | 3 | gemma | 2.83 | 32.2 | 46.2 | 14.0 | 202.5 |
| HTP2 | 4 | qwen3 | 3.21 | 85.0 | 98.5 | 13.5 | 237.8 |

| request | model | out | acquired s | exec start s | first/last window s | end s | layers at first window | max layers while decoding | layers called | phone calls | phone/all window tokens |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 00 | llama | 28 | 83.8 | 86.4 | None/None | 86.7 | - | - | 0 | 0 | -/- |
| 000 | gemma | 119 | 2.0 | 8.0 | 16.1/67.5 | 68.3 | 0 | 24 | 24 | 1104 | 44/116 |
| 001 | gemma | 32 | 68.7 | 69.0 | 72.1/82.9 | 83.7 | 24 | 24 | 24 | 672 | 26/29 |
| 002 | qwen3 | 118 | 86.7 | 139.5 | 142.4/212.9 | 213.2 | 6 | 6 | 6 | 66 | 11/117 |
| 003 | qwen3 | 126 | 213.6 | 214.3 | 217.9/292.8 | 295.2 | 6 | 6 | 6 | 660 | 108/123 |
| 004 | qwen3 | 219 | 213.6 | 214.3 | 220.1/349.9 | 351.1 | 6 | 6 | 6 | 690 | 115/216 |
| 005 | gemma | 617 | 351.3 | 387.6 | 397.1/661.7 | 662.5 | 16 | 16 | 16 | 9680 | 603/614 |
| 006 | qwen3 | 132 | 663.0 | 743.9 | 750.9/831.7 | 832.4 | 6 | 6 | 0 | 0 | 0/129 |
| 007 | gemma | 29 | 833.7 | 898.2 | 903.5/915.1 | 916.0 | 16 | 16 | 16 | 400 | 23/26 |

### coherentEF2

- event kinds {'EVALUATED': 165, 'PREPARING': 4, 'PROPOSED': 4, 'READY': 4, 'SELECTION_CONFIRMED': 1, 'SELECTION_OBSERVED': 3, 'SESSION_DRAINING': 1, 'SESSION_LOADING': 4, 'SESSION_READY': 4, 'SESSION_VERIFIED': 4}
- EVALUATED reasons {'PHONE_RESIDENCY_DEMAND_UNAVAILABLE': 1, 'PHONE_RESIDENCY_LEARNING_EXPLORATION': 6, 'PHONE_RESIDENCY_LEARNING_RETAINED': 22, 'PHONE_RESIDENCY_REVALIDATION_ENERGY_UNKNOWN': 12, 'PHONE_RESIDENCY_REVALIDATION_UNAFFORDABLE': 123, 'PHONE_RESIDENCY_TRANSITION_IN_PROGRESS': 1}
- helper deferrals {}; WAITING_FOR_HELPER_RELEASE 0
- per model phone layers while decoding {'gemma': {'decode_union_s': 337.4, 'mean_phone_layers_while_decoding': 17.43}, 'qwen3': {'decode_union_s': 276.0, 'mean_phone_layers_while_decoding': 6.0}}

| READY s | gen | changed | layers by model | sessions by model | selection |
| ---: | ---: | --- | --- | --- | --- |
| 20.9 | 1 | HTP0 | {'gemma': 8} | {'gemma': 1} | PHONE_RESIDENCY_LEARNING_EXPLORATION |
| 31.9 | 2 | HTP1 | {'gemma': 16} | {'gemma': 2} | PHONE_RESIDENCY_LEARNING_EXPLORATION |
| 43.3 | 3 | HTP2 | {'gemma': 24} | {'gemma': 3} | PHONE_RESIDENCY_LEARNING_EXPLORATION |
| 174.9 | 4 | HTP2 | {'gemma': 16, 'qwen3': 6} | {'gemma': 2, 'qwen3': 1} | PHONE_RESIDENCY_LEARNING_EXPLORATION |

| session | gen | model | GB | loading s | verified s | s | MB/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| HTP0 | 1 | gemma | 2.83 | 2.0 | 20.9 | 18.9 | 149.7 |
| HTP1 | 2 | gemma | 2.83 | 22.2 | 31.9 | 9.6 | 294.4 |
| HTP2 | 3 | gemma | 2.83 | 32.0 | 43.3 | 11.4 | 248.7 |
| HTP2 | 4 | qwen3 | 3.21 | 150.0 | 174.9 | 24.9 | 129.0 |

| request | model | out | acquired s | exec start s | first/last window s | end s | layers at first window | max layers while decoding | layers called | phone calls | phone/all window tokens |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 00 | llama | 28 | 148.5 | 156.0 | None/None | 156.2 | - | - | 0 | 0 | -/- |
| 000 | gemma | 119 | 2.0 | 72.9 | 81.6/130.2 | 131.7 | 24 | 24 | 24 | 2568 | 104/115 |
| 001 | gemma | 32 | 132.0 | 132.6 | 135.6/147.6 | 148.4 | 24 | 24 | 24 | 672 | 26/29 |
| 002 | qwen3 | 118 | 156.3 | 239.2 | 242.0/308.0 | 309.2 | 6 | 6 | 6 | 636 | 104/115 |
| 003 | qwen3 | 126 | 309.5 | 311.2 | 314.9/394.3 | 396.1 | 6 | 6 | 6 | 66 | 11/123 |
| 004 | qwen3 | 219 | 311.1 | 311.5 | 317.2/449.5 | 451.7 | 6 | 6 | 6 | 624 | 101/215 |
| 005 | gemma | 617 | 452.1 | 492.6 | 503.1/769.7 | 770.6 | 16 | 16 | 16 | 9680 | 603/614 |
| 006 | qwen3 | 132 | 771.0 | 868.2 | 876.3/951.6 | 952.7 | 6 | 6 | 6 | 768 | 126/129 |
| 007 | gemma | 29 | 955.2 | 1025.0 | 1028.9/1039.1 | 1039.9 | 16 | 16 | 16 | 400 | 23/26 |

### coherentRP

- event kinds {'EVALUATED': 1225, 'PREPARING': 15, 'PROPOSED': 15, 'READY': 15, 'SESSION_DRAINING': 12, 'SESSION_LOADING': 15, 'SESSION_READY': 15, 'SESSION_VERIFIED': 15}
- EVALUATED reasons {'PHONE_RESIDENCY_DEMAND_UNAVAILABLE': 1, 'PHONE_RESIDENCY_DESKTOP_REPROVISION': 10, 'PHONE_RESIDENCY_PROPORTIONAL_SPLIT': 3, 'PHONE_RESIDENCY_REPROVISION_HOLD_IDLE_MODEL': 4, 'PHONE_RESIDENCY_REPROVISION_RETAINED': 1202, 'PHONE_RESIDENCY_TRANSITION_IN_PROGRESS': 4, 'PHONE_RESIDENT_MODEL_REPROVISIONING_CONFIGURED': 1}
- helper deferrals {}; WAITING_FOR_HELPER_RELEASE 0
- per model phone layers while decoding {'gemma': {'decode_union_s': 337.7, 'mean_phone_layers_while_decoding': 22.17}, 'qwen3': {'decode_union_s': 245.9, 'mean_phone_layers_while_decoding': 17.0}}

| READY s | gen | changed | layers by model | sessions by model | selection |
| ---: | ---: | --- | --- | --- | --- |
| 24.3 | 1 | HTP0 | {'gemma': 8} | {'gemma': 1} | PHONE_RESIDENCY_PROPORTIONAL_SPLIT |
| 43.4 | 2 | HTP1 | {'gemma': 16} | {'gemma': 2} | PHONE_RESIDENCY_PROPORTIONAL_SPLIT |
| 56.1 | 3 | HTP2 | {'gemma': 24} | {'gemma': 3} | PHONE_RESIDENCY_PROPORTIONAL_SPLIT |
| 105.0 | 4 | HTP1 | {'gemma': 16, 'qwen3': 6} | {'gemma': 2, 'qwen3': 1} | PHONE_RESIDENCY_DESKTOP_REPROVISION |
| 120.8 | 5 | HTP0 | {'gemma': 8, 'qwen3': 12} | {'gemma': 1, 'qwen3': 2} | PHONE_RESIDENCY_DESKTOP_REPROVISION |
| 133.1 | 6 | HTP2 | {'qwen3': 17} | {'qwen3': 3} | PHONE_RESIDENCY_DESKTOP_REPROVISION |
| 369.0 | 7 | HTP2 | {'gemma': 8, 'qwen3': 12} | {'gemma': 1, 'qwen3': 2} | PHONE_RESIDENCY_PROPORTIONAL_SPLIT |
| 380.5 | 8 | HTP0 | {'gemma': 16, 'qwen3': 6} | {'gemma': 2, 'qwen3': 1} | PHONE_RESIDENCY_DESKTOP_REPROVISION |
| 391.3 | 9 | HTP1 | {'gemma': 24} | {'gemma': 3} | PHONE_RESIDENCY_DESKTOP_REPROVISION |
| 690.8 | 10 | HTP1 | {'gemma': 16, 'qwen3': 6} | {'gemma': 2, 'qwen3': 1} | PHONE_RESIDENCY_PROPORTIONAL_SPLIT |
| 705.2 | 11 | HTP0 | {'gemma': 8, 'qwen3': 12} | {'gemma': 1, 'qwen3': 2} | PHONE_RESIDENCY_DESKTOP_REPROVISION |
| 721.2 | 12 | HTP2 | {'qwen3': 17} | {'qwen3': 3} | PHONE_RESIDENCY_DESKTOP_REPROVISION |
| 849.0 | 13 | HTP1 | {'gemma': 8, 'qwen3': 11} | {'gemma': 1, 'qwen3': 2} | PHONE_RESIDENCY_DESKTOP_REPROVISION |
| 859.5 | 14 | HTP2 | {'gemma': 16, 'qwen3': 6} | {'gemma': 2, 'qwen3': 1} | PHONE_RESIDENCY_DESKTOP_REPROVISION |
| 870.2 | 15 | HTP0 | {'gemma': 24} | {'gemma': 3} | PHONE_RESIDENCY_DESKTOP_REPROVISION |

| session | gen | model | GB | loading s | verified s | s | MB/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| HTP0 | 1 | gemma | 2.83 | 2.0 | 24.3 | 22.3 | 127.0 |
| HTP1 | 2 | gemma | 2.83 | 25.1 | 43.4 | 18.3 | 155.0 |
| HTP2 | 3 | gemma | 2.83 | 43.9 | 56.1 | 12.2 | 232.5 |
| HTP1 | 4 | qwen3 | 3.21 | 89.4 | 105.0 | 15.6 | 206.1 |
| HTP0 | 5 | qwen3 | 3.21 | 105.7 | 120.8 | 15.1 | 213.0 |
| HTP2 | 6 | qwen3 | 2.67 | 121.6 | 133.1 | 11.5 | 232.9 |
| HTP2 | 7 | gemma | 2.83 | 356.4 | 369.0 | 12.7 | 223.5 |
| HTP0 | 8 | gemma | 2.83 | 369.5 | 380.5 | 11.0 | 257.6 |
| HTP1 | 9 | gemma | 2.83 | 381.2 | 391.3 | 10.1 | 280.5 |
| HTP1 | 10 | qwen3 | 3.21 | 676.4 | 690.8 | 14.4 | 222.4 |
| HTP0 | 11 | qwen3 | 3.21 | 691.4 | 705.2 | 13.8 | 232.4 |
| HTP2 | 12 | qwen3 | 2.67 | 707.9 | 721.2 | 13.3 | 201.4 |
| HTP1 | 13 | gemma | 2.83 | 838.2 | 849.0 | 10.8 | 261.8 |
| HTP2 | 14 | gemma | 2.83 | 849.4 | 859.5 | 10.1 | 280.9 |
| HTP0 | 15 | gemma | 2.83 | 859.7 | 870.2 | 10.5 | 269.9 |

| s (count) | req | reason | mode | followed | source | leader | load window s | target layers | selected layers | fits | stage/target swap s | blocked | rate MB/s (n) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 000 | PHONE_RESIDENCY_PROPORTIONAL_SPLIT | PROPORTIONAL |  | None | None | [None, None] | {'gemma': 24} | {'gemma': 24} | None | 42.5/42.5 | - | 200.0 (0) |
| 56.2-70.4 (3) | 000 | PHONE_RESIDENCY_REPROVISION_RETAINED | FOLLOW | gemma | executing | None | [None, None] | {} | {'gemma': 24} | None | 0.0/0.0 | - | 161.1 (3) |
| 79.1-86.2 (22) | 002 | PHONE_RESIDENCY_REPROVISION_RETAINED | FOLLOW | gemma | executing | None | [None, None] | {} | {'gemma': 24} | None | 0.0/0.0 | - | 161.1 (3) |
| 86.7-86.8 (2) | 00 | PHONE_RESIDENCY_REPROVISION_HOLD_IDLE_MODEL | HOLD | gemma | executing | None | [None, None] | {} | {'gemma': 24} | None | 0.0/0.0 | - | 161.1 (3) |
| 89.3 | 002 | PHONE_RESIDENCY_REPROVISION_HOLD_IDLE_MODEL | HOLD | gemma | retained | None | [None, None] | {} | {'gemma': 24} | None | 0.0/0.0 | - | 161.1 (3) |
| 89.4 | 002 | PHONE_RESIDENCY_DESKTOP_REPROVISION | FOLLOW | qwen3 | loading | 002 | [89.4, 144.4] | {'qwen3': 18} | {'qwen3': 6, 'gemma': 16} | False | 19.9/59.8 | - | 161.1 (3) |
| 105.2 | 002 | PHONE_RESIDENCY_DESKTOP_REPROVISION | FOLLOW | qwen3 | loading | 002 | [89.4, 144.4] | {'qwen3': 17} | {'qwen3': 12, 'gemma': 8} | True | 18.7/34.3 | - | 171.4 (4) |
| 120.9 | 003 | PHONE_RESIDENCY_DESKTOP_REPROVISION | FOLLOW | qwen3 | loading | 002 | [89.4, 144.4] | {'qwen3': 17} | {'qwen3': 17} | True | 14.9/14.9 | - | 178.9 (5) |
| 133.3 | 004 | PHONE_RESIDENCY_REPROVISION_RETAINED | FOLLOW | qwen3 | loading | 002 | [89.4, 144.4] | {} | {'qwen3': 17} | None | 0.0/0.0 | - | 185.4 (6) |
| 166.4-170.4 (8) | 002 | PHONE_RESIDENCY_REPROVISION_RETAINED | FOLLOW | qwen3 | executing | None | [None, None] | {} | {'qwen3': 17} | None | 0.0/0.0 | - | 185.4 (6) |
| 172.8-224.7 (103) | 002 | PHONE_RESIDENCY_REPROVISION_RETAINED | FOLLOW | qwen3 | executing | None | [None, None] | {} | {'qwen3': 17} | None | 0.0/0.0 | - | 185.4 (6) |
| 226.3-238.9 (5) | 003 | PHONE_RESIDENCY_REPROVISION_RETAINED | FOLLOW | qwen3 | executing | None | [None, None] | {} | {'qwen3': 17} | None | 0.0/0.0 | - | 185.4 (6) |
| 239.5-355.5 (321) | 004 | PHONE_RESIDENCY_REPROVISION_RETAINED | FOLLOW | qwen3 | executing | None | [None, None] | {} | {'qwen3': 17} | None | 0.0/0.0 | - | 185.4 (6) |
| 356.2 | 005 | PHONE_RESIDENCY_PROPORTIONAL_SPLIT | PROPORTIONAL | qwen3,gemma | loading | 005 | [356.2, 423.9] | {'qwen3': 6, 'gemma': 16} | {'qwen3': 12, 'gemma': 8} | True | 15.3/30.5 | - | 185.4 (6) |
| 369.2 | 005 | PHONE_RESIDENCY_DESKTOP_REPROVISION | FOLLOW | gemma | loading | 005 | [356.2, 423.9] | {'gemma': 24} | {'qwen3': 6, 'gemma': 16} | True | 14.9/29.8 | - | 189.9 (7) |
| 380.7 | 007 | PHONE_RESIDENCY_DESKTOP_REPROVISION | FOLLOW | gemma | loading | 005 | [356.2, 423.9] | {'gemma': 24} | {'gemma': 24} | True | 14.4/14.4 | - | 196.2 (8) |
| 391.5 | 005 | PHONE_RESIDENCY_REPROVISION_RETAINED | FOLLOW | gemma | loading | 005 | [356.2, 423.9] | {} | {'gemma': 24} | None | 0.0/0.0 | - | 202.8 (9) |
| 414.3-417.2 (8) | 005 | PHONE_RESIDENCY_REPROVISION_RETAINED | FOLLOW | gemma | executing | None | [None, None] | {} | {'gemma': 24} | None | 0.0/0.0 | - | 202.8 (9) |
| 419.1-674.9 (603) | 005 | PHONE_RESIDENCY_REPROVISION_RETAINED | FOLLOW | gemma | executing | None | [None, None] | {} | {'gemma': 24} | None | 0.0/0.0 | - | 202.8 (9) |
| 675.5 | 006 | PHONE_RESIDENCY_PROPORTIONAL_SPLIT | PROPORTIONAL | qwen3,gemma | loading | 006 | [675.5, 730.5] | {'qwen3': 12, 'gemma': 8} | {'qwen3': 6, 'gemma': 16} | True | 15.8/31.6 | - | 202.8 (9) |
| 691.0 | 006 | PHONE_RESIDENCY_DESKTOP_REPROVISION | FOLLOW | qwen3 | loading | 006 | [675.5, 730.5] | {'qwen3': 17} | {'qwen3': 12, 'gemma': 8} | True | 15.7/28.7 | - | 204.8 (10) |
| 705.6 | 006 | PHONE_RESIDENCY_DESKTOP_REPROVISION | FOLLOW | qwen3 | loading | 006 | [675.5, 730.5] | {'qwen3': 17} | {'qwen3': 17} | True | 12.9/12.9 | - | 207.2 (11) |
| 721.3 | 006 | PHONE_RESIDENCY_REPROVISION_RETAINED | FOLLOW | qwen3 | loading | 006 | [675.5, 730.5] | {} | {'qwen3': 17} | None | 0.0/0.0 | - | 206.8 (12) |
| 769.1-772.8 (8) | 006 | PHONE_RESIDENCY_REPROVISION_RETAINED | FOLLOW | qwen3 | executing | None | [None, None] | {} | {'qwen3': 17} | None | 0.0/0.0 | - | 206.8 (12) |
| 775.3-836.5 (117) | 006 | PHONE_RESIDENCY_REPROVISION_RETAINED | FOLLOW | qwen3 | executing | None | [None, None] | {} | {'qwen3': 17} | None | 0.0/0.0 | - | 206.8 (12) |
| 837.7 | 007 | PHONE_RESIDENCY_REPROVISION_HOLD_IDLE_MODEL | HOLD | qwen3 | executing | None | [None, None] | {} | {'qwen3': 17} | None | 0.0/0.0 | - | 206.8 (12) |
| 838.1 | 007 | PHONE_RESIDENCY_DESKTOP_REPROVISION | FOLLOW | qwen3,gemma | loading | 007 | [838.1, 905.7] | {'gemma': 24} | {'qwen3': 11, 'gemma': 8} | True | 13.7/41.1 | - | 206.8 (12) |
| 849.2 | 007 | PHONE_RESIDENCY_DESKTOP_REPROVISION | FOLLOW | gemma | loading | 007 | [838.1, 905.7] | {'gemma': 24} | {'qwen3': 6, 'gemma': 16} | True | 13.5/27.0 | - | 210.1 (13) |
| 859.6 | 007 | PHONE_RESIDENCY_DESKTOP_REPROVISION | FOLLOW | gemma | loading | 007 | [838.1, 905.7] | {'gemma': 24} | {'gemma': 24} | True | 13.2/13.2 | - | 213.8 (14) |
| 870.4 | 007 | PHONE_RESIDENCY_REPROVISION_RETAINED | FOLLOW | gemma | loading | 007 | [838.1, 905.7] | {} | {'gemma': 24} | None | 0.0/0.0 | - | 216.7 (15) |

| request | model | out | acquired s | exec start s | first/last window s | end s | layers at first window | max layers while decoding | layers called | phone calls | phone/all window tokens |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 00 | llama | 28 | 86.8 | 89.1 | None/None | 89.3 | - | - | 0 | 0 | -/- |
| 000 | gemma | 119 | 2.0 | 8.0 | 15.5/69.5 | 70.3 | 0 | 24 | 24 | 624 | 24/116 |
| 001 | gemma | 32 | 70.7 | 71.0 | 74.0/85.8 | 86.7 | 24 | 24 | 24 | 672 | 26/29 |
| 002 | qwen3 | 118 | 89.4 | 162.4 | 165.7/225.2 | 226.2 | 17 | 17 | 17 | 1802 | 104/115 |
| 003 | qwen3 | 126 | 231.1 | 232.3 | 238.3/307.0 | 308.8 | 17 | 17 | 17 | 1836 | 105/122 |
| 004 | qwen3 | 219 | 232.2 | 232.3 | 236.1/354.4 | 355.5 | 17 | 17 | 17 | 3349 | 195/216 |
| 005 | gemma | 617 | 356.2 | 403.1 | 413.8/674.0 | 674.9 | 24 | 24 | 24 | 14520 | 603/614 |
| 006 | qwen3 | 132 | 675.5 | 759.8 | 768.4/836.5 | 837.7 | 17 | 17 | 17 | 2040 | 118/129 |
| 007 | gemma | 29 | 838.1 | 879.6 | 883.3/895.0 | 895.9 | 24 | 24 | 0 | 0 | 0/26 |

