## Host-side S41SERVERFFNSHAPE (per server process, per shape)

| process | tokens | columns | calls | compute_mean_ms | compute_p50_ms | rpc_mean_ms | overlap_mean_ms | reference_ms | speedup |
|---|---|---|---|---|---|---|---|---|---|
| large-model-2-physical-hot-desktop.stderr | 1 | 8704 | 168 | 5.128 | 4.598 | 5.859 | 9.487 | 4.404 | 0.859 |
| large-model-2-physical-hot-desktop.stderr | 1 | 17408 | 132 | 9.237 | 8.443 | 9.989 | 10.023 | 9.72 | 1.052 |
| large-model-5-physical-cold-desktop.stderr | 1 | 3840 | 66 | 3.152 | 2.106 | 3.824 | 9.399 | 1.694 | 0.538 |
| large-model-5-physical-cold-desktop.stderr | 1 | 7680 | 66 | 4.324 | 3.455 | 4.947 | 6.892 | 3.309 | 0.765 |
| large-model-5-physical-cold-desktop.stderr | 1 | 11520 | 66 | 5.714 | 5.088 | 6.293 | 6.293 | 4.915 | 0.860 |
| large-model-5-physical-cold-desktop.stderr | 1 | 15360 | 108 | 7.212 | 6.688 | 7.873 | 7.882 | 6.59 | 0.914 |
| large-model-6-physical-hot-desktop.stderr | 1 | 17408 | 468 | 9.631 | 9.703 | 10.324 | 10.361 | 9.72 | 1.009 |
| large-model-8-physical-cold-desktop.stderr | 1 | 15360 | 6 | 7.615 | 6.508 | 8.279 | 8.312 | 6.59 | 0.865 |
| large-model-9-physical-hot-desktop.stderr | 1 | 17408 | 2172 | 9.624 | 9.770 | 10.315 | 10.354 | 9.72 | 1.010 |

## Host-side call-weighted mean per shape

| tokens | columns | calls | compute_mean_ms | rpc_mean_ms | reference_ms | speedup |
|---|---|---|---|---|---|---|
| 1 | 3840 | 66 | 3.152 | 3.824 | 1.694 | 0.538 |
| 1 | 7680 | 66 | 4.324 | 4.947 | 3.309 | 0.765 |
| 1 | 8704 | 168 | 5.128 | 5.859 | 4.404 | 0.859 |
| 1 | 11520 | 66 | 5.714 | 6.293 | 4.915 | 0.860 |
| 1 | 15360 | 114 | 7.233 | 7.895 | 6.59 | 0.911 |
| 1 | 17408 | 2772 | 9.607 | 10.301 | 9.72 | 1.012 |

## Phone-side S43DUALFFN per worker log and shape (us)

| log | tokens | columns | calls | primary_cols | secondary_cols | total p50 | total mean | total p90 | primary p50 | secondary p50 | wait p50 | merge p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| worker.log | 1 | 3840 | 66 | 3264 | 576 | 2072 | 3123 | 7665 | 1548 | 1880 | 499 | 14 |
| worker.log | 1 | 7680 | 66 | 6528 | 1152 | 3438 | 4293 | 9142 | 3174 | 3072 | 1 | 13 |
| worker.log | 1 | 8704 | 168 | 7424 | 1280 | 4609 | 5106 | 5400 | 4487 | 4008 | 0 | 16 |
| worker.log | 1 | 11520 | 66 | 9792 | 1728 | 5075 | 5691 | 9823 | 4864 | 4133 | 0 | 12 |
| worker.log | 1 | 15360 | 114 | 13056 | 2304 | 6653 | 7211 | 11434 | 6116 | 5158 | 0 | 13 |
| worker.log | 1 | 17408 | 2772 | 14848 | 2560 | 9666 | 9576 | 10263 | 9206 | 7453 | 0 | 17 |

## Phone-side warm-up lines

- {"rounds": 3, "layers": 6, "calls": 18, "first_total_us": 10776, "last_total_us": 10614, "elapsed_us": 188405, "log": "worker.log"}
- {"rounds": 3, "layers": 6, "calls": 18, "first_total_us": 8976, "last_total_us": 8779, "elapsed_us": 160400, "log": "worker.log"}
- {"rounds": 3, "layers": 6, "calls": 18, "first_total_us": 5730, "last_total_us": 5953, "elapsed_us": 110177, "log": "worker.log"}

wrote /mnt/storage/s43-dual-prep/analysis/pair-1-1/treatment_timings.json
