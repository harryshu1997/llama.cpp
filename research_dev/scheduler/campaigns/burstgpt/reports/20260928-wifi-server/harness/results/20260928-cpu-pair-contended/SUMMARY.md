# server_bench wifibafc5a

| arm | c | step ms | ms/token | decode tok/s | e2e tok/s | GPU W | GPU J/tok | CPU J/tok | identical vs cpu |
|---|---|---|---|---|---|---|---|---|---|
| cpu-helpers | 1 | 253.5 | 253.5 | 3.95 | 3.89 | 107.3 | 25.32 | - | - |
| cpu-helpers | 4 | 264.7 | 66.2 | 15.11 | 14.45 | 111.7 | 7.02 | - | - |
| cpu | 1 | 246.9 | 246.9 | 4.05 | 3.95 | 112.3 | 26.15 | - | - |
| cpu | 4 | 267.4 | 66.8 | 14.96 | 14.32 | 111.6 | 7.07 | - | - |
