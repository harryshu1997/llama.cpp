# All-request runtime scheduling audit

The 74 FP16 requests inherit the placement selected from a live startup snapshot. The ten Llama 1B requests invoke the cost estimator and lease policy independently at arrival. Therefore only the latter have per-request candidate costs and scheduling overhead.

| # | Request | Arrival s | Model | Scheduling scope | Selected binding | Candidate result | Decision cost | Completion s | SLO |
| ---: | --- | ---: | --- | --- | --- | --- | ---: | ---: | --- |
| 0 | fp16-burstgpt:0 | 1.950 | gemma-4-12b-q40-dequant-f16 | runtime_placement_inherited | cuda0:25layers+desktop-cpu+op15:htp0 | inherited gemma-suffix-layers-0-22-m1-m16 | n/a | 2455.424 | miss |
| 1 | llama1b-overlay:0 | 2.150 | llama-3.2-1b-instruct-q4_0 | per_request_runtime_cost_and_lease | desktop-cpu | CPU SELECTED; CUDA EXECUTOR_ABSENT; phone SLO_INFEASIBLE | 48.749 ms | 81.509 | miss |
| 2 | fp16-burstgpt:1 | 3.000 | gemma-4-12b-q40-dequant-f16 | runtime_placement_inherited | cuda0:25layers+desktop-cpu+op15:htp0 | inherited gemma-suffix-layers-0-22-m1-m16 | n/a | 2286.087 | miss |
| 3 | fp16-burstgpt:2 | 3.600 | gemma-4-12b-q40-dequant-f16 | runtime_placement_inherited | cuda0:25layers+desktop-cpu+op15:htp0 | inherited gemma-suffix-layers-0-22-m1-m16 | n/a | 2670.988 | miss |
| 4 | fp16-burstgpt:3 | 4.300 | gemma-4-12b-q40-dequant-f16 | runtime_placement_inherited | cuda0:25layers+desktop-cpu+op15:htp0 | inherited gemma-suffix-layers-0-22-m1-m16 | n/a | 2413.221 | miss |
| 5 | fp16-burstgpt:4 | 6.450 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 103.434 | miss |
| 6 | fp16-burstgpt:5 | 7.200 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 137.002 | miss |
| 7 | fp16-burstgpt:6 | 7.450 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 47.549 | miss |
| 8 | fp16-burstgpt:7 | 7.600 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 37.364 | miss |
| 9 | fp16-burstgpt:8 | 7.650 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 89.639 | miss |
| 10 | fp16-burstgpt:9 | 7.750 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 75.929 | miss |
| 11 | fp16-burstgpt:10 | 7.850 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 123.542 | miss |
| 12 | llama1b-overlay:1 | 7.850 | llama-3.2-1b-instruct-q4_0 | per_request_runtime_cost_and_lease | desktop-cpu | CPU SELECTED; CUDA EXECUTOR_ABSENT; phone SLO_INFEASIBLE | 34.720 ms | 29.753 | met |
| 13 | fp16-burstgpt:11 | 8.050 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 148.980 | miss |
| 14 | fp16-burstgpt:12 | 8.200 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 166.540 | miss |
| 15 | fp16-burstgpt:13 | 8.250 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 178.680 | miss |
| 16 | fp16-burstgpt:14 | 8.400 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 190.788 | miss |
| 17 | fp16-burstgpt:15 | 8.650 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 203.286 | miss |
| 18 | fp16-burstgpt:16 | 8.800 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 221.709 | miss |
| 19 | fp16-burstgpt:17 | 8.850 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 236.031 | miss |
| 20 | fp16-burstgpt:18 | 8.900 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 235.981 | miss |
| 21 | fp16-burstgpt:19 | 8.950 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 249.932 | miss |
| 22 | llama1b-overlay:2 | 9.000 | llama-3.2-1b-instruct-q4_0 | per_request_runtime_cost_and_lease | desktop-cpu | CPU SELECTED; CUDA EXECUTOR_ABSENT; phone SLO_INFEASIBLE | 35.422 ms | 9.776 | met |
| 23 | fp16-burstgpt:20 | 9.050 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 272.425 | miss |
| 24 | fp16-burstgpt:21 | 9.100 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 291.412 | miss |
| 25 | fp16-burstgpt:22 | 9.200 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 281.746 | miss |
| 26 | fp16-burstgpt:23 | 9.300 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 313.376 | miss |
| 27 | fp16-burstgpt:24 | 9.400 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 327.544 | miss |
| 28 | llama1b-overlay:3 | 9.600 | llama-3.2-1b-instruct-q4_0 | per_request_runtime_cost_and_lease | desktop-cpu | CPU SELECTED; CUDA EXECUTOR_ABSENT; phone SLO_INFEASIBLE | 29.230 ms | 9.177 | met |
| 29 | fp16-burstgpt:25 | 9.650 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 359.193 | miss |
| 30 | fp16-burstgpt:26 | 9.950 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 347.188 | miss |
| 31 | fp16-burstgpt:27 | 10.150 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 554.311 | miss |
| 32 | fp16-burstgpt:28 | 10.150 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 377.678 | miss |
| 33 | fp16-burstgpt:29 | 10.400 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 384.332 | miss |
| 34 | fp16-burstgpt:30 | 10.500 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 396.464 | miss |
| 35 | fp16-burstgpt:31 | 12.050 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 411.057 | miss |
| 36 | fp16-burstgpt:32 | 12.050 | gemma-4-12b-q40-dequant-f16 | runtime_placement_inherited | cuda0:25layers+desktop-cpu+op15:htp0 | inherited gemma-suffix-layers-0-22-m1-m16 | n/a | 2179.022 | miss |
| 37 | llama1b-overlay:4 | 12.250 | llama-3.2-1b-instruct-q4_0 | per_request_runtime_cost_and_lease | desktop-cpu | CPU SELECTED; CUDA EXECUTOR_ABSENT; phone SLO_INFEASIBLE | 39.504 ms | 86.445 | miss |
| 38 | fp16-burstgpt:33 | 12.550 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 443.064 | miss |
| 39 | fp16-burstgpt:34 | 13.550 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 421.891 | miss |
| 40 | fp16-burstgpt:35 | 14.300 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 469.016 | miss |
| 41 | fp16-burstgpt:36 | 14.750 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 452.147 | miss |
| 42 | fp16-burstgpt:37 | 15.150 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 531.557 | miss |
| 43 | fp16-burstgpt:38 | 15.450 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 1049.873 | miss |
| 44 | fp16-burstgpt:39 | 15.500 | gemma-4-12b-q40-dequant-f16 | runtime_placement_inherited | cuda0:25layers+desktop-cpu+op15:htp0 | inherited gemma-suffix-layers-0-22-m1-m16 | n/a | 2508.750 | miss |
| 45 | fp16-burstgpt:40 | 17.350 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 508.609 | miss |
| 46 | fp16-burstgpt:41 | 17.350 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 571.243 | miss |
| 47 | llama1b-overlay:5 | 17.550 | llama-3.2-1b-instruct-q4_0 | per_request_runtime_cost_and_lease | desktop-cpu | CPU SELECTED; CUDA EXECUTOR_ABSENT; phone SLO_INFEASIBLE | 38.910 ms | 30.198 | miss |
| 48 | fp16-burstgpt:42 | 17.900 | gemma-4-12b-q40-dequant-f16 | runtime_placement_inherited | cuda0:25layers+desktop-cpu+op15:htp0 | inherited gemma-suffix-layers-0-22-m1-m16 | n/a | 2021.924 | miss |
| 49 | fp16-burstgpt:43 | 17.950 | gemma-4-12b-q40-dequant-f16 | runtime_placement_inherited | cuda0:25layers+desktop-cpu+op15:htp0 | inherited gemma-suffix-layers-0-22-m1-m16 | n/a | 2318.057 | miss |
| 50 | fp16-burstgpt:44 | 21.050 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 641.745 | miss |
| 51 | fp16-burstgpt:45 | 21.700 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 635.596 | miss |
| 52 | fp16-burstgpt:46 | 21.850 | gemma-4-12b-q40-dequant-f16 | runtime_placement_inherited | cuda0:25layers+desktop-cpu+op15:htp0 | inherited gemma-suffix-layers-0-22-m1-m16 | n/a | 2183.988 | miss |
| 53 | fp16-burstgpt:47 | 22.050 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 587.496 | miss |
| 54 | fp16-burstgpt:48 | 22.100 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 614.067 | miss |
| 55 | fp16-burstgpt:49 | 22.250 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 639.798 | miss |
| 56 | llama1b-overlay:6 | 22.450 | llama-3.2-1b-instruct-q4_0 | per_request_runtime_cost_and_lease | desktop-cpu | CPU SELECTED; CUDA EXECUTOR_ABSENT; phone SLO_INFEASIBLE | 32.836 ms | 23.487 | met |
| 57 | fp16-burstgpt:50 | 24.550 | gemma-4-12b-q40-dequant-f16 | runtime_placement_inherited | cuda0:25layers+desktop-cpu+op15:htp0 | inherited gemma-suffix-layers-0-22-m1-m16 | n/a | 2208.261 | miss |
| 58 | fp16-burstgpt:51 | 24.900 | gemma-4-12b-q40-dequant-f16 | runtime_placement_inherited | cuda0:25layers+desktop-cpu+op15:htp0 | inherited gemma-suffix-layers-0-22-m1-m16 | n/a | 2494.509 | miss |
| 59 | fp16-burstgpt:52 | 27.300 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 635.495 | miss |
| 60 | fp16-burstgpt:53 | 27.800 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 652.807 | miss |
| 61 | fp16-burstgpt:54 | 28.750 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 673.666 | miss |
| 62 | fp16-burstgpt:55 | 29.650 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 690.369 | miss |
| 63 | fp16-burstgpt:56 | 34.150 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 946.591 | miss |
| 64 | fp16-burstgpt:57 | 37.000 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 865.136 | miss |
| 65 | llama1b-overlay:7 | 37.200 | llama-3.2-1b-instruct-q4_0 | per_request_runtime_cost_and_lease | desktop-cpu | CPU SELECTED; CUDA EXECUTOR_ABSENT; phone SLO_INFEASIBLE | 39.859 ms | 31.890 | miss |
| 66 | fp16-burstgpt:58 | 37.900 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 1825.700 | miss |
| 67 | fp16-burstgpt:59 | 39.600 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 996.168 | miss |
| 68 | fp16-burstgpt:60 | 39.850 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 962.739 | miss |
| 69 | fp16-burstgpt:61 | 39.950 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 1082.516 | miss |
| 70 | fp16-burstgpt:62 | 41.050 | gemma-4-12b-q40-dequant-f16 | runtime_placement_inherited | cuda0:25layers+desktop-cpu+op15:htp0 | inherited gemma-suffix-layers-0-22-m1-m16 | n/a | 2339.741 | miss |
| 71 | fp16-burstgpt:63 | 43.600 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 1125.605 | miss |
| 72 | fp16-burstgpt:64 | 45.500 | gemma-4-12b-q40-dequant-f16 | runtime_placement_inherited | cuda0:25layers+desktop-cpu+op15:htp0 | inherited gemma-suffix-layers-0-22-m1-m16 | n/a | 2656.018 | miss |
| 73 | fp16-burstgpt:65 | 45.550 | gemma-4-12b-q40-dequant-f16 | runtime_placement_inherited | cuda0:25layers+desktop-cpu+op15:htp0 | inherited gemma-suffix-layers-0-22-m1-m16 | n/a | 2204.903 | miss |
| 74 | llama1b-overlay:8 | 45.750 | llama-3.2-1b-instruct-q4_0 | per_request_runtime_cost_and_lease | desktop-cpu | CPU SELECTED; CUDA EXECUTOR_ABSENT; phone SLO_INFEASIBLE | 30.987 ms | 58.682 | miss |
| 75 | fp16-burstgpt:66 | 48.450 | gemma-4-12b-q40-dequant-f16 | runtime_placement_inherited | cuda0:25layers+desktop-cpu+op15:htp0 | inherited gemma-suffix-layers-0-22-m1-m16 | n/a | 2407.210 | miss |
| 76 | fp16-burstgpt:67 | 49.450 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 1291.066 | miss |
| 77 | fp16-burstgpt:68 | 50.300 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 1223.941 | miss |
| 78 | fp16-burstgpt:69 | 51.850 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 1314.432 | miss |
| 79 | fp16-burstgpt:70 | 53.550 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 1264.727 | miss |
| 80 | fp16-burstgpt:71 | 54.250 | qwen3-14b-q4km-dequant-f16 | runtime_placement_inherited | cuda0:18layers+desktop-cpu+op15:htp1,htp2 | inherited qwen-full-ffn-layers-0-11-m1-m4 | n/a | 1289.960 | miss |
| 81 | fp16-burstgpt:72 | 55.800 | gemma-4-12b-q40-dequant-f16 | runtime_placement_inherited | cuda0:25layers+desktop-cpu+op15:htp0 | inherited gemma-suffix-layers-0-22-m1-m16 | n/a | 2608.349 | miss |
| 82 | fp16-burstgpt:73 | 57.700 | gemma-4-12b-q40-dequant-f16 | runtime_placement_inherited | cuda0:25layers+desktop-cpu+op15:htp0 | inherited gemma-suffix-layers-0-22-m1-m16 | n/a | 2295.938 | miss |
| 83 | llama1b-overlay:9 | 57.900 | llama-3.2-1b-instruct-q4_0 | per_request_runtime_cost_and_lease | desktop-cpu | CPU SELECTED; CUDA EXECUTOR_ABSENT; phone SLO_INFEASIBLE | 31.408 ms | 57.369 | miss |
