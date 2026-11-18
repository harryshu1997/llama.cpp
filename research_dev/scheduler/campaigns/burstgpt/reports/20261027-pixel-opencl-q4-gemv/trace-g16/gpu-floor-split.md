| run | GPU mean W | GPU floor W (p05) | GPU kJ = idle + dynamic | host kJ | host ex GPU floor | vs g161nph | host floor W | host dynamic kJ | vs g161nph (dyn) | fleet ex GPU floor kJ | vs g161nph |
|---|---|---|---|---|---|---|---|---|---|---|---|
| g62nph (desktop 10-26) | 34.3 | 20.3 (20.3) | 58.9 = 34.9 + 24.0 | 179.6 | 144.8 | -4.4 % | 60.1 | 76.5 | -4.2 % | 146.3 | -3.4 % |
| g62new (CPU-Pixel 10-26) | 32.3 | 14.1 (14.0) | 55.2 = 24.0 + 31.1 | 155.3 | 131.3 | -13.3 % | 54.0 | 63.0 | -21.2 % | 142.4 | -6.0 % |
| g161nph (desktop 10-28) | 31.2 | 12.9 (12.8) | 53.6 = 22.2 + 31.4 | 173.7 | 151.5 | +0.0 % | 54.6 | 79.9 | +0.0 % | 151.5 | +0.0 % |
| g16r1new (OpenCL-Pixel, Gemma unassisted) | 31.0 | 13.0 (12.9) | 53.0 = 22.2 + 30.9 | 165.6 | 143.5 | -5.3 % | 54.8 | 71.9 | -10.0 % | 148.5 | -2.0 % |
| g16c1new (CPU-Pixel 10-28) | 32.1 | 13.2 (13.1) | 54.9 = 22.6 + 32.3 | 159.1 | 136.5 | -9.9 % | 55.1 | 64.9 | -18.8 % | 147.7 | -2.5 % |
| g16o1new (OpenCL-Pixel 10-28) | 31.7 | 12.9 (12.8) | 54.0 = 22.1 + 31.9 | 158.3 | 136.2 | -10.1 % | 54.9 | 64.6 | -19.1 % | 146.0 | -3.6 % |

g16o1new vs g16c1new: host -0.5 %, host ex GPU floor -0.2 %, host dynamic -0.4 %, GPU dynamic 32.3 -> 31.9 kJ, CPU 104.2 -> 104.3 kJ
g62new vs g62nph: host -13.5 %, host ex GPU floor -9.3 %, host dynamic -17.7 %, GPU dynamic 24.0 -> 31.1 kJ, CPU 120.7 -> 100.2 kJ
