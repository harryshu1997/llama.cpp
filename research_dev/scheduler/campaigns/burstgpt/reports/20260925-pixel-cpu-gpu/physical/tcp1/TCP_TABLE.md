| arm | rows | calls | compute p50 / p90 ms | RPC p50 / p90 ms | non-compute p50 ms | first layer / others ms | 6-layer RPC ms | outputs = production | exit / boot / forward |
|---|---:|---:|---|---|---:|---|---:|---|---|
| a-prod | 1 | 72 | 28.01 / 32.42 | 39.12 / 43.66 | 10.48 | 28.18 / 27.82 | 234.7 | byte-identical | 0 / same / removed |
| a-prod | 2 | 72 | 46.05 / 51.73 | 61.66 / 67.92 | 15.31 | 48.39 / 43.71 | 370.0 | byte-identical | 0 / same / removed |
| a-prod | 4 | 72 | 72.00 / 81.56 | 88.14 / 97.93 | 17.36 | 81.74 / 67.00 | 528.9 | byte-identical | 0 / same / removed |
| b-boost | 1 | 72 | 6.43 / 15.20 | 13.31 / 24.92 | 7.03 | 15.39 / 6.40 | 79.9 | byte-identical | 0 / same / removed |
| b-boost | 2 | 72 | 10.15 / 19.08 | 20.65 / 32.42 | 10.73 | 19.36 / 9.84 | 123.9 | byte-identical | 0 / same / removed |
| b-boost | 4 | 72 | 20.19 / 24.26 | 34.23 / 39.16 | 13.96 | 21.91 / 20.10 | 205.4 | byte-identical | 0 / same / removed |
| c-boost-batch | 1 | 72 | 6.46 / 14.58 | 13.56 / 23.20 | 7.07 | 15.16 / 6.40 | 81.4 | byte-identical | 0 / same / removed |
| c-boost-batch | 2 | 72 | 9.67 / 17.11 | 20.20 / 29.68 | 10.83 | 16.92 / 9.46 | 121.2 | byte-identical | 0 / same / removed |
| c-boost-batch | 4 | 72 | 18.66 / 22.11 | 32.45 / 37.69 | 13.66 | 20.48 / 18.34 | 194.7 | byte-identical | 0 / same / removed |
| d-prod | 1 | 72 | 27.56 / 30.89 | 37.72 / 42.25 | 9.86 | 27.55 / 27.57 | 226.3 | byte-identical | 0 / same / removed |
| d-prod | 2 | 72 | 45.03 / 51.64 | 61.22 / 67.12 | 15.17 | 47.25 / 43.73 | 367.3 | byte-identical | 0 / same / removed |
| d-prod | 4 | 72 | 72.03 / 81.84 | 88.63 / 98.21 | 17.50 | 82.24 / 67.31 | 531.8 | byte-identical | 0 / same / removed |
| e-boost | 1 | 72 | 6.46 / 15.52 | 13.41 / 26.15 | 7.00 | 15.82 / 6.37 | 80.4 | byte-identical | 0 / same / removed |
| e-boost | 2 | 72 | 10.43 / 17.88 | 21.70 / 31.54 | 11.03 | 17.54 / 9.87 | 130.2 | byte-identical | 0 / same / removed |
| e-boost | 4 | 72 | 21.06 / 23.76 | 35.62 / 39.67 | 13.98 | 21.92 / 20.47 | 213.7 | byte-identical | 0 / same / removed |
