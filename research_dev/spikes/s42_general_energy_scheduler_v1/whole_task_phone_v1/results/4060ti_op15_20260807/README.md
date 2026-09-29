# RTX 4060 Ti plus OP15 whole-task result

Device and workload identity:

- Desktop: `zhihao-Z690-C-ac`, RTX 4060 Ti UUID
  `GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08`.
- Phone: OP15 serial `3C15AU002CL00000`, Adreno 840, kernel
  `6.12.23-android16-5-o-g227664cbe007-4k`.
- Model SHA-256:
  `4b90b1d7ae7324676194755a6dfce11cb6e457982c4c01a1db2857be1ed064ad`
  on both devices.
- Requests: mixed-trace indices 2, 50, and 99, with respective input/output
  lengths 857/192, 915/292, and 1421/342 tokens.
- Repetitions: three per route, nine paid observations per route.

Artifacts:

- `RESIDENT_ROUTE_RESULT.json`: synchronized physical samples and identity.
  SHA-256 `5413dece30fec0b5a9f262357007a584fc68520126bd27fc2844b2f2dcf05ac4`.
- `CUDA_TAIL_RESULT.json`: three post-response CUDA-tail measurements.
  SHA-256 `fee4c62215d7ad830286bdd69020047021d4ee5a830300b0280afd60b93c12ce`.
- `ANALYSIS_LIFECYCLE.json`: fitted costs, comparisons, policy states, and
  decisions. SHA-256
  `081bde3c5fc380dec2c70d20f2253a7e72800eadb4595fdfa1261ceea157eb38`.
- `SCHEDULER_PROFILE_ISOLATED.json`: CUDA epoch-open profile with one measured
  tail charge. SHA-256
  `7be07a2d3ff0af223b0ca9a8d25ae83001d2592414f94f7d11cbdd1e7d6dda1f`.
- `SCHEDULER_PROFILE_TAIL_REUSED.json`: CUDA active-epoch incremental profile.
  SHA-256
  `e1954d965daa86928fd14066221ad739ee033a0eb2aaf4bfffcd3018557c5dba`.

Energy is CPU package plus GPU board plus whole-phone energy during the
request window. CUDA post-response accounting adds only GPU dynamic energy
above measured resident P8 idle; ordinary resident idle after completion is
not assigned to the completed request.

