# S39 CP0-R1 V2.6 results

Status: `MECHANICS_PASS; V24_PLAN_SET_MATERIALIZED; A_ONLY_ATTEMPTED_STAGE_5_OF_10_REFUSED; NO_MODEL_EXECUTED`

Completed:

- managed-plan gate: 20/20 focused tests;
- capture execution receipt: 4/4 focused tests;
- runtime inventory security: 4/4 focused tests;
- V2.6 authority contract checks: 4/4 focused tests;
- production materialization: 27/27 focused tests, including adversarial
  altered-command, altered-hash, altered-path, boot-ID, phase-ID, timestamp,
  topology, duplicate-publication, stale-record, and partial-output cases;
- V2.4 focused suites: 93/93;
- V2.5 production suites: 132/132;
- V2.4 desktop deployment suites: 132/132;
- all CP0-R1 and W9 SHA-256 manifests verify; `git diff --check` is clean;
- V2.6 contract builder produces a canonical contract.

The desktop-controlled no-model topology probe passes against the exact V2.4
contract. Artifact:
`results/no_model_topology_final.json` (sha256
`d073b80da1107b447147c3c31cadaeafef8c5ec84b3c8107ed06647124c4b6ff`).

## Materialized runtime inventory and phase lock

`materialize_production_v26.py` (sha256
`72e7e9e5e472562c1e1c4c3baa22eb4055d44fd3860d045fd9d675d1af4c1591`) is the new
bounded production materializer. It reuses `runtime_inventory_v26.py`
verbatim: it builds the canonical
`s39-cp0-r1-v26-runtime-inventory-spec-v1` spec and calls
`materialize_runtime_inventory()`; it does not introduce another inventory
format. One real run completed on 2026-07-26 with no model process launched:

- evidence root:
  `results/production_materialization_20260726T170521Z/`
  (manifest `SHA256SUMS.txt` sha256
  `5104cd7f11190b1f2b5ce42a2475b2a05dd00cbe0c73bab1b0dd726f843d9470`);
- phase ID: `cp0-r1-v26-a-only-20260726T170336Z-9fbfbe34`;
- spec sha256
  `5ec1ec841970111c2a61e9991eda008fa174bfe7720fcbb53b44639783dcddfa`;
- runtime inventory sha256
  `139e91a91f8137ca293a1886536d046428c45a4cba9b4c83f5dff07dc958b809`
  (38 components, 5 bundles, 4 managed launch-plan pairs);
- A_ONLY phase lock sha256
  `eebe7b96e8ff7cea29e731a17a9d144a2ffd8539f8cfdde7f873ab20c1b8668b`.

The run probed live identities fail-closed against the frozen V2.6 contract
and the validated topology receipt: RTX 4060 Ti UUID, name, and memory; both
phone serial/product/model/device identities; and the current boot IDs
(cuda `2f68fcf5-54e1-4306-ba25-63359be572c2`,
op12 `2ca4b7a3-c9c1-4614-a0d2-5746c57c8c4d`,
op15 `3eb99d7e-0b35-41a5-9e30-21867dc5dec7`). Both phone shards and the
desktop Qwen3-14B artifact were re-hashed live and match their frozen
digests; no caller-supplied summary or stored verdict was accepted. All five
runtime bundle closures were observed twice with double-hash and
before/after-stat TOCTOU discipline: the desktop
`/home/zhihao/s39-v26-a-only/runtime-v1/{cuda-monolithic,cuda-route}` roots
over hardened SSH and the three on-phone roots over ADB. The four capture
entrypoints inside the cuda bundles match the contract-pinned producer bytes,
and the seven cuda-monolithic components match the pinned
`cuda-monolithic-launch.json` digests. The two managed launchers were
provisioned to `/home/zhihao/s39-v26-a-only/launcher-v1/` and verified against
their frozen digests (`52c4e1f2...` USB adapter, `b97941dc...` frozen v23
launcher).

The phase lock binds the contract, spec, inventory, topology-receipt, and
cuda-monolithic-launch digests, the three device boot IDs, the device
identities, the three model-artifact identities, the producer self-pin, and a
`HOST_MONOTONIC_RAW` interval with `started_ns < completed_ns <= event_ns`.
Publication is exclusive (`O_EXCL` plus fsync) and all-or-nothing.

Independent validation passed in a fresh process
(`materialize_production_v26.py validate`): it re-reads every published file
once, re-derives every digest, re-runs `validate_runtime_inventory` and the
managed-plan gate, and recomputes all four managed launch plans from the
contract, topology receipt, bound boot IDs, and observed components, so even
a digest-consistent argv rewrite is rejected. `live-check` then re-probed all
three boot IDs and confirmed the lock is live
(`V2_6_PRODUCTION_BOOT_IDENTITY_LIVE`). Known limit: offline validation
binds record consistency and every pinned anchor; device content freshness is
bound at materialization time and re-checked by `live-check`, and
`android.adb_sha256` has no external anchor beyond forced consistency across
the three android plans.

## Fresh V2.6 phase after route-bundle extension

The desktop cuda-route bundle was extended with verified copies of
`artifact_root_capture_v1.py` and `cuda_monolithic_v1.py` (all four capture
producers now live inside the deployment), `ROUTE_FILES` was extended to the
12-file closure, and one fresh production materialization ran with no model
process:

- current root: `results/production_materialization_20260726T220718Z/`;
- phase ID: `cp0-r1-v26-a-only-20260726T220529Z-6176d8d7`;
- 40 components, 5 bundles, 4 managed launch-plan pairs; independent
  `validate` and `live-check` both pass.

The earlier root `production_materialization_20260726T170521Z/` is
SUPERSEDED: its closure predates the two route copies and the strict
producer self-pin rejects it against the edited producer. It remains on disk
as the historical record of its phase.

## V2.4 plan-set materialization (the V2.6-to-V2.4 adapter)

`materialize_v24_plan_set_v26.py` (sha256
`d4b6f555eb56afa6a47428595b1ae6fe92e54e7b9d87f0711bab4298abf2acf8`) closes
the missing "concrete production plan" gap. From the validated V2.6
evidence it derives the three inputs the historical (by-design-refusing)
capture path never produced - the runtime-bundle closure input, the desktop
operator input, and the desktop inventory - then mirrors the byte-verified
frozen V2.4 tool closure to `/home/zhihao/s39-v26-a-only/repo-v1/s39` and
drives the UNMODIFIED frozen chain
`materialize_a_only_inputs_v1.py -> originate_runtime_v1.py` on the
acquisition desktop, keeping every frozen validator in the path (USB-launcher
compatibility with the frozen v23 launcher, live-identity exclusion,
originator revalidation, authority `validate_runtime_plan`). 15 focused
tests pass, including one end-to-end test that runs the real frozen chain
locally and adversarial cases for live-IP injection, prebound boot IDs,
`ssh`-key plans, mono/capture hash tampering, short mechanism matrices,
duplicate publication, and byte tampering.

The real run passed with no model process:

- evidence root: `results/v24_plan_set_20260726T221842Z/` (inputs, outputs,
  spec, dry-run report, capture log, `SHA256SUMS.txt`);
- bound to V2.6 phase `cp0-r1-v26-a-only-20260726T220529Z-6176d8d7`
  (lock sha256 bound inside the materialization record);
- artifacts now present at `v24_readiness/results/prephase_20260726T0915Z/`
  and byte-identically on the desktop mirror:
  `cuda-route-launch.json` sha256 `f8c7fb0b...aa32e18`,
  `joint-capture-plan.json` `b6b7e215...0a67b659`,
  `phone-route-launch.json` `6c6dc78e...42b7cebc`,
  `prospective-runtime-root.json` `4ede29fe...f5b76ca`,
  `runtime-bundle-plan.json` `933bbf49...66c1de72`;
- independent validation passes in a fresh process
  (`V24_PLAN_SET_VALIDATION_PASS`): the local V2.4 authority re-runs
  `validate_runtime_plan`, launch schemas and the mechanism matrix are
  re-checked, and no artifact contains a `v23_readiness` reference;
- consumability is proven one stage further: the frozen
  `materialize_config_v1.py` on the desktop mirror accepted the five
  artifacts and built the A_ONLY orchestration config and plan (exit 0,
  scratch root `scratch-config-check/`, no execution).

Two earlier adapter attempts refused fail-closed and were superseded (their
evidence roots `v24_plan_set_20260726T221208Z/` and the mono-stat refusal
before it are retained): the first refusal exposed second-granularity remote
stats versus the exact-nanosecond frozen pins (fixed by capturing desktop
stats via remote `os.stat`), the second exposed the orchestration
source-pin rule that capture entrypoint components must reference the repo
producer sources, satisfied by widening the cuda_route bundle root to
`/home/zhihao/s39-v26-a-only` so both the runtime binaries and the mirrored
producer sources fall under one confinement root.

Known executor seams recorded for the next gate (not blockers for the plan
set itself): the frozen phone-route process argv is the 5-element
no-`--boot-id` form the V2.4 validators require, but the current USB
launcher demands `--boot-id`, and the frozen probe argv cannot satisfy
`phone_runtime_probe_v1.py`'s per-run `--pid/--start-ticks/--boot-id`
interface. One bounded execution adapter honoring
`boot_id_source=phase_fresh_snapshot` is required before the joint capture
can run. Shared-fleet note: an unrelated foreign `llama-cli` process
(`/data/local/tmp/hyzheng/elastic/...`, CPU-only) was observed on OP15
during the final sweep; it is not part of this work and was left running,
and the acquisition preflight gates on phone memory will fail closed if
foreign load persists at acquisition time.

Suites after the changes: V2.6 focused 74/74 (20+4+4+4+27+15); V2.4 focused
93/93; V2.4 desktop deployment 132/132; V2.5 production 132/132; all
CP0-R1/W9 manifests and both new evidence-root manifests verify;
`git diff --check` is clean. Zero llama processes from this work and zero
CUDA compute apps remained on the desktop, OP12, OP15, and the controller.

## First real end-to-end acquisition attempts (2026-07-26)

Two real orchestrated A_ONLY runs were launched on the hardware through the
frozen `orchestration_v1` sequence. Neither completed; both refused
fail-closed, and both refusals are recorded rather than worked around.

Prerequisite closed first: the frozen phone-route argv seams. Two bounded
wrappers were added and pinned into the plan set - the snapshot launcher
(`managed_runtime_launcher_snapshot_v1.py`: validates the plan through the
FROZEN launcher stack, snapshots the phone's live boot ID per
`boot_id_source=phase_fresh_snapshot`, then delegates to the frozen
`execute()`) and the snapshot probe
(`phone_runtime_probe_snapshot_v1.py`: derives pid/start-ticks/boot live,
runs the FROZEN v23 probe in `capture_compatible` mode, and projects its row
onto the exact V2.4 22-key `s39-cp0-r1-v24-phone-runtime-probe-v1` shape).
`originate_from_inventory_v26.py` keeps every frozen MAT validator and adds
only the snapshot-executor pinning. The regenerated plan set validates
(`V24_PLAN_SET_VALIDATION_PASS`, evidence root
`results/v24_plan_set_20260726T234741Z/`) and `materialize_config_v1.py`
accepts it (exit 0).

Attempt 1, run root `a-only-run-1-refused-hostkey/` on the desktop:
refused at stage 1 (`artifact_root`) with
`E_EXIT ... expected 0, got 255`. Cause was environmental, not evidential:
the desktop's own `~/.ssh/known_hosts` no longer contained its self-SSH
host pin, which the frozen stage environment exposes because the embedded
capture-common runs a bare `ssh`. The ED25519 fingerprint
`SHA256:gxnHOeYIL1FPe2BIUfxQArV9Vmy9ToXkY9j/7U5fIpg` was cross-verified from
two independent observation paths (the controller's existing pin and a
loopback `ssh-keyscan` on the desktop) before the pin was restored, and the
self-SSH was reverified under the exact frozen stage environment.

Attempt 2, run root `a-only-run-2/` on the desktop: stage 1
`artifact_root` PASSED for real - it hash-pinned the 9,001,752,960-byte
desktop Qwen3-14B artifact, both phone shards, and every runtime component
across all three devices. Stage 2 `preparation` then executed the real
physical phone reboots (op12 `2ca4b7a3-c9c1-4614-a0d2-5746c57c8c4d` ->
`5ed2c12b-2c39-4c49-bc43-c4bbe7ee1465`, op15
`3eb99d7e-0b35-41a5-9e30-21867dc5dec7` ->
`ef44ed15-7f3f-4569-b9ee-940f92fa7004`) and refused afterwards at
`V24_PREPARATION_REFUSED: E_FIELD: cuda.status`.

Root cause, localized and reproduced: the frozen contract-pinned
`v24_readiness/production_plan_v1/preparation_v1.py` (sha256
`2aa020e1fe8e6396cd73b8710eac5cf36291b8e027290c815a0e2f27766ef0d2`) builds
its CUDA probe as
`["ssh", ..., target, "sh", "-c", CUDA_STATUS]`. OpenSSH joins remote argv
with spaces, so the remote login shell executes `sh -c set` with `-eu` and
the script lines as positional arguments: the probe returns 28 lines of
shell variables on stdout, and the `IFS` continuation line carries no `=`,
which is exactly the `E_FIELD` trigger in `parse_assignments`. Reproduced
byte-for-byte with the same argv under the same stage environment (rc 0, 28
stdout lines, empty stderr). The sibling frozen capture-common does this
correctly via `ssh_python_argv`, which `shlex.quote`s the whole script into
ONE argument. This is a latent defect in frozen V2.4 production code that
only a first real execution could expose; it is NOT reachable or fixable
from outside that file, because the orchestration source-pin rule forces the
stage entrypoint to be exactly the contract-pinned bytes.

Correct fail-closed side effect: the phone reboots invalidated the
materialized V2.6 phase lock, and `materialize_production_v26.py live-check`
now refuses with `E_VALUE: live.op12.boot_id`. The plan set still validates,
since it binds no boot IDs. After both attempts, zero llama processes and
zero CUDA compute apps remained on the desktop, OP12, OP15, and the
controller.

## V2.4.1 successor and post-fix acquisition

The human-authorized V2.4.1 repair changed only the remote-shell quoting in
`preparation_v1.py`. The parent bytes are preserved under
`v26_readiness/frozen_parents_v24/`. The exact successor digests are:

```text
preparation_v1.py                  1e02e19b589c8d6badacdf8c596d9a25ea44605d4a17777be28663302300a443
CP0_R1_EVIDENCE_CONTRACT_V2_4.json 264d16b33d56176ee6d3ac84471b3616d4b17e5bea785e08d03723ceb73a439f
artifact_root_capture_v1.py        a70f72bf1a9e90eb2821e9c8ab9ded00389fb608e95a8b6be42b65c8804f9475
fast_fresh_capture_v1.py           bb77fa9f281cee68463360f84ead05b304a4f976ad1d84350aea171b251c68cc
verify_topology_v1.py              602dc531001cea06fc30161879c1bcf454b06f92c6c9942bfc33571ed3885c66
materialize_a_only_inputs_v1.py    09e41d84893ccf641c9e14cc70269b880bde5800d043a8aac26d1b134d7ab9fb
CP0_R1_EVIDENCE_CONTRACT_V2_6.json 8eaad17ace65d4e5c5d1a2621e6d942fc4f76df3ed28d606e9421588a3e57082
```

V2.4.1 then advanced the real acquisition through stages 1-4 repeatedly:
`artifact_root`, physical reboot preparation, `phase_lock`, and
`identity_binding` passed. Every complete post-fix attempt refused at stage 5
`fresh_readiness`, before the durable paid-start marker:

- desktop `a-only-run-5/`: OP12 thermalservice broken pipe;
- desktop `a-only-20260727T024144Z/`: OP15 changed from
  `192.168.1.97` to `172.20.173.218` after identity binding;
- desktop `a-only-20260727T024604Z/`: OP15 thermalservice broken pipe;
- desktop `a-only-20260727T025317Z/`: OP12 thermalservice broken pipe;
- desktop `a-only-20260727T025706Z/`: OP15 changed from
  `192.168.1.97` to `172.20.173.218`;
- desktop `a-only-20260727T030126Z/`: OP12 thermalservice broken pipe.

One inherited background setup attempt and a newly started refresh overlapped.
The acquisition `a-only-20260727T025029Z/` refused at `preparation` because
the plan-set cleanup removed `runtime-bundle-plan.json`; the reciprocal
refresh rooted at
`results/production_materialization_20260727T025317Z/` refused publication
with `E_PLAN_EXISTS: cuda-route-launch.json`. Both are excluded setup
collisions, neither reached a paid stage, and no result is claimed from them.
Subsequent attempts ran after the overlap ended.

The three clean refreshed roots consumed by the last bounded loop are:

```text
root                                                  SHA256SUMS.txt sha256
production_materialization_20260727T025451Z/           2f3189909eec48df9b204dbbb57e8e88650cf73d656711ccb493fad69e468e57
production_materialization_20260727T025859Z/           036eb980832855d8d3861fde88eac7dc9061c1bccd50ecb9c744d09e92924775
production_materialization_20260727T030320Z/           5ac1e138d9d5fd3af4a8e8e1318027e8e7a3050e56de712d238a8e0676dd7ab4
```

All three plan sets independently reported
`V24_PLAN_SET_VALIDATION_PASS`. At the final stop, both phones were present
on ADB port 5038, all run roots contained zero paid-marker files, and the
desktop had zero `llama-layersplit` or `llama-stage-direct-relay` processes.
No source byte changed during these live attempts. The previously reported
focused and full-suite test counts therefore remain the implementation
verification baseline; no new qualification test result is inferred from
them.

## Honest A_ONLY verdict

No Qwen3-14B model execution, phone qualification, task-quality result, or
energy result is claimed. The bounded qualification has reached the
`fresh_readiness` stage, but it has never reached
`readiness_projection`, `cuda_monolithic`, `joint_phone_cuda`, `fan_in`, or
`authority`. The V2.4.1 quoting blocker is closed. The current blockers are
the frozen Android thermalservice capture race and OP15 changing Wi-Fi
networks after the post-reboot identity is bound.

The pipeline is structurally at stage 5 of 10, with four stages passing, but
all model-execution evidence remains ahead. A first end-to-end result needs
one stable pre-paid attempt through `fresh_readiness` and
`readiness_projection`, then one non-retryable paid chain through
`cuda_monolithic`, `joint_phone_cuda`, `fan_in`, and the V2.4 authority,
followed by the V2.6 outer receipts and authorization. Repeated blind retries
are not justified while the two setup races remain. No qualification result
is manufactured.

## Separate no-reboot prototype result

The statement above applies to the frozen qualification path. A separate,
explicitly non-qualification prototype subsequently executed Qwen3-14B on
the real RTX 4060 Ti, OP15, and OP12.

The run root is
`/home/zhihao/s39-v26-a-only/prototype-v1/run_20260727T034430Z`.
Its `RESULT.json` SHA-256 is
`ffddf6c0ba3a486c437f2adbd1cf856e9328352d8c315d5a9a2e5dfeb03fbfa3`.
The verdict is
`JOINT_B8_PROTOTYPE_EXECUTION_PASS_TOKEN_DIVERGENCE`.

Both routes completed eight requests with eight continuation tokens each.
The phone route took 45.239 seconds and concurrent CUDA took 1.601 seconds.
The direct phone relay carried 19 batches, 763 activation rows, and
15,626,240 activation bytes. Placement and session certificates passed on
CUDA0, OP15 OpenCL, and OP12 OpenCL. Both routes returned from eight to zero
live sequences, and no S39 process remained.

Phone and CUDA agreed on 60/64 greedy tokens. This comparison is diagnostic:
the phone and CUDA kernels are different numerical paths. A paid handoff gate
must use a path-matched replay oracle. Full details and artifact hashes are in
`prototype_v1/RESULTS.md`.
