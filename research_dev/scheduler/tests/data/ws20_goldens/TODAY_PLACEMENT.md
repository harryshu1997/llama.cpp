| variant | Qwen owners | Gemma owners | saving J/token Qwen / Gemma | kJ/run | % fleet | FFN chain ms (rows 1) Qwen / Gemma, now -> plan | priors |
|---|---|---|---|---|---|---|---|
| measured-adb | op15 0-17, pixel10pro 18-23 | op15 0-25 | 0.00 / 2.23 | 3.3 | 2.9 | 257 -> 257 / 194 -> 182 | 0 |
| measured-aoa-risk | op15 0-17, pixel10pro 18-23 | op15 0-25 | 0.00 / 2.23 | 3.3 | 2.9 | 257 -> 226 / 194 -> 182 | 1 |
| measured-aoa-expected | op15 0, pixel10pro 1-23 | op15 0-25 | 1.01 / 2.23 | 4.4 | 3.9 | 257 -> 199 / 194 -> 182 | 1 |
| measured-aoa-envelope | op15 0-11, pixel10pro 12-23 | op15 0-25 | 0.36 / 2.23 | 3.7 | 3.2 | 257 -> 217 / 194 -> 182 | 1 |
| op15-capacity-lowest-live | op15 0-16, pixel10pro 17-23 | op15 0-25 | -0.21 / 2.23 | 3.1 | 2.7 | 257 -> 261 / 194 -> 182 | 0 |
| op15-gemma-held-24-q4_0 | op15 0-17, pixel10pro 18-23 | op15 0-23, pixel10pro 24-25 | 0.00 / 1.10 | 1.7 | 1.4 | 257 -> 257 / 194 -> 200 | 6 |
| op15-gemma-held-24-f16 | op15 0-17, pixel10pro 18-23 | op15 0-23, pixel10pro 24-25 | 0.00 / 0.98 | 1.5 | 1.3 | 257 -> 257 / 194 -> 202 | 6 |
| op15-absent | , CPU 0-23 | , CPU 0-25 | -8.90 / 0.29 | -8.9 | -7.7 | 257 -> 449 / 194 -> 341 | 0 |
| pixel-absent | op15 0-17, CPU 18-23 | op15 0-25 | 0.11 / 2.29 | 3.6 | 3.1 | 257 -> 289 / 194 -> 182 | 0 |

Assisted tokens per run (block 1 mean): Gemma 1499, Qwen 1043
