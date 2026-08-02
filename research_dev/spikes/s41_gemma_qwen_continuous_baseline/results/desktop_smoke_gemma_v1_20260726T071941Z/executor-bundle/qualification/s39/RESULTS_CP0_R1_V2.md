# S39 CP0-R1 v2 evidence hardening

Verdict:

`CP0_R1_V2_EVIDENCE_MECHANICS_PASS; NO_MODEL_PREFLIGHT_PASS; TWO_ROUTE_ELIGIBILITY_NOT_RUN; CYCLE_BLOCKED`

This checkpoint replaces no CP0-R1 v1 artifact. It adds a versioned raw
evidence contract and identity-only preflight. Model qualification, candidate
download, switching, trace replay, controller integration, and energy
acquisition remain out of scope.

## Evidence boundary

The v1 evaluator accepted summary records whose artifact fields could contain
arbitrary SHA-shaped strings. V2 accepts no summary or reported verdict. It
derives eligibility only after reopening all 26 required roles, checking each
byte count and SHA-256, and parsing the same bytes that were hashed. Paths are
relative to a pinned root; symlinks, hardlinks, path reuse, digest reuse,
mount crossing, and files that change during reading are rejected.

The required raw roles separately cover:

- the candidate and prospectively frozen route geometry;
- phone history, position, ownership, and cleanup mechanics;
- a tested CUDA replay path and an independent path-matched monolithic CUDA
  oracle;
- per-model B8 CUDA memory and a measured pair co-residency attempt;
- one corpus plus CUDA and phone outputs for every quality item;
- common-host-clock phone publication and CUDA readiness;
- OP15 and OP12 node-level realized placement and memory state;
- direct OP15-to-OP12 activation transfer;
- complete A-to-B and B-to-A phone release and local-UFS reprepare.

Phone-vs-CUDA greedy-token and phone-vs-CUDA call-shape agreement are derived
diagnostics. Exact continuation agreement is required only between the tested
CUDA path and its independent CUDA oracle at the same input and call geometry.

## Status

The frozen parent bindings are:

- v1 contract:
  `ffb2abeb33e818477e8a7181e177a6d296ed3b021767bc83a8df0d2450e5d095`;
- v1 candidate:
  `ee3196ca660fa7eb7ea293260dc98dd6fdbf14571a4d4c5aece5343fb29b28d8`;
- v1 manifest:
  `480cf8836e0a83ea9102c19e82bf5ec01e2f78b0ea2a3d9061be71bed4b31275`;
- v2 evidence contract:
  `a1ae7f58a63dfbfeae7e29619b63da875a4e411ce60c11973907b0831b48d534`.

The focused suite passes 37/37. It covers candidate rebinding,
missing and aliased roles, unsafe paths, digest rebinding, same-program and
cross-geometry oracle substitutions, forged CUDA and phone memory summaries,
per-item quality regressions, mixed clocks, late publication, CPU fallback,
host-relayed activation, incomplete release, wrong shards, USB weight traffic,
float/integer confusion, supplied verdict fields, and mutations of persisted
preflight stdout with its digest rebound. The complete S39 suite passes
291/291.

## No-model preflight

The identity-only run is frozen under
`results/cp0_r1_preflight_v2/run_20260725T163055Z/`.

- RTX 4060 Ti host: `zhihao-Z690-C-ac`;
- GPU UUID: `GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08`;
- GPU memory: 16,380 MiB;
- OP15: serial `3C15AU002CL00000`, product/model `CPH2749`, device
  `OP611FL1`;
- OP12: serial `5ae7a43d`, product/model `CPH2583`, device `OP595DL1`;
- both USB phones were found on ADB port 5038;
- ADB port 5037 was empty;
- OP15 also exposed a WiFi ADB endpoint, recorded separately;
- preflight artifact SHA-256:
  `a23013e9d5ff43563faa77d1a7d7daa088a9674468036046dc7862a6047085fa`.

`validate_cp0_r1_preflight_v2.py` independently reopened the preflight and its
manifest, recomputed the GPU and ADB identities from saved stdout, checked the
collector source digest, and emitted `NO_MODEL_PREFLIGHT_VALIDATED`.

No model was loaded, executed, downloaded, or transferred. No route has passed
the v2 evidence gate. The next ordered gate remains complete Qwen3 14B
qualification, but it was not started in this checkpoint.
