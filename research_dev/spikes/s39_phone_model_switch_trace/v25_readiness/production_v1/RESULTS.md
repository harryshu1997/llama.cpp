# V2.5 A_ONLY production materializer

Status: `A_ONLY_PLANS_MATERIALIZED_NO_HARDWARE_RUN`

This directory contains the no-model materializer for the V2.5 A_ONLY
acquisition. It does not reboot phones, discover devices, run a model, create
the outer phase lock, execute a stage, or authorize acquisition.

## Inputs

`materialize_a_only_v1.py` consumes:

- the SHA-256-pinned production inventory;
- the fresh V2.5 reboot-preparation record;
- the linked post-reboot discovery record;
- the fresh artifact-identity record;
- controller-readable V2.4 inputs, including the V2.4 orchestration plan and
  the distinct identity-binding receipt, stage receipt, and stdout
  attestation records;
- separately attested RTX copies of every remote fan-in input.

The materializer reopens controller files and checks bytes, SHA-256, and stat
identity. Remote inputs are content-bound to their controller sources. Local
and remote source programs are bound to the V2.5 contract composition.

## Outputs

The materialization root schema is
`s39-v25-a-only-materialization-v1`. A successful root has
`fan_in_materialized=true` and declares these stage descriptors:

- `remote_history`
- `phone_guard_before`
- `cuda_monolithic`
- `joint_phone_cuda`
- `remote_fan_in`
- `phone_guard_after`

Each descriptor contains only `argv`, `cwd`, `entrypoint`, `environment`,
`expected_output`, `support`, and `timeout_seconds`. The orchestrator owns
stage ordering and execution.

The materializer emits phase preparation/discovery copies, the required
`inner.*` V2.4 records, the remote-history plan, both remote CUDA plans, the
remote-phone guard plan, and the remote fan-in plans. Files are created with
`O_EXCL`, fsynced, and removed as a unit on failure.

The full `inner.identity_binding_stage_receipt` has schema
`s39-cp0-r1-v24-stage-receipt-v1`. The separate
`inner.identity_binding_attestation` has schema
`s39-cp0-r1-v24-identity-binding-attestation-v1`. They are not
interchangeable.

## Verification

On 2026-07-26:

- production materializer tests: 13/13 pass;
- complete V2.5 test discovery: 102/102 pass;
- remote fan-in contract tests: 17/17 pass;
- remote fan-in executor tests: 21/21 pass;
- `git diff --check`: pass;
- non-ASCII scan of `production_v1`: clean.

No hardware command, model execution, acquisition, commit, or push was
performed. The outer phase lock, stage receipts, final manifest, and authority
decision remain the orchestration stream's responsibility.
