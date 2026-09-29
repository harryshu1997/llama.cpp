# Bounded long-context relocation test

Status: prepared, but physical calibration has not started. On the first
attempt the exclusive rig lock rejected the launch (`LOCK_DENIED.json`).
On continuation at 23:44 EDT the lock was free, but idle preflight rejected
another campaign's active NPU worker. See `PREFLIGHT_WORKER_BLOCKED.json` and
`physical/calibrate-idle-check/preflight_snapshot.json`. No other process was
stopped, no check was bypassed, and no phone mode/kernel operation was performed
by this test.

## Frozen scope

- Context: 8192, versus 2560 in the prior relocation gate.
- One document request with 64 generated tokens on each parent. The document
  is tokenized once using the actual full-parent server, persisted, and reused
  verbatim for the relocated parent. It must exceed 4096 prompt tokens and fit
  with generation inside the context; no slicing or truncation is allowed.
- Identical Gemma artifact, existing server/libraries, CUDA graphs enabled,
  GPU suffix of 23 layers, batch 2048, ubatch 512, parallel 1, seed 42 and
  temperature 0. All three existing F16 shards cover CPU FFNs 0-23 at full width.
- Two short cold/hot calibration requests first, using the existing desktop
  calibration entry point at context 8192. Only successful physical evidence
  may update the experiment-local desktop launch qualification. Assisted
  long-context execution and energy do not inherit old qualification.
- No long trace, native rebuild, new shard, driver change, reboot, or flashing.

`PROMPT.txt` is a frozen copy of the architecture document with a retrieval
question. SHA256: `b131c1dd0497fae6a50273b9d5a9732dcce4bca9ae7097c50095e31cee29efc5`.
The exact token count is not yet measured. Output validation includes native
completion counts, semantic sanity and a separate read of the answer; exact
cross-hardware token equality is diagnostic, as in the prior accepted gate.

## Narrow changes

- `campaigns/burstgpt/remote_resident_gate.py`: optional document input; tokenize
  once without truncating; require a terminal `truncated=false` and exact
  prompt/generation counts; persist full completion metadata, host high-water
  RSS, native KV allocation lines and raw host samples. Long-document mode
  also requires host admission including the full file during loading,
  configured weight factor, KV, workspace and reserve before any launch.
- `tests/test_remote_resident_gate.py`: document identity/shape and terminal
  accounting regressions.
- This report's `run_bounded.py`: frozen configuration and invocation only;
  reuses canonical calibration, catalog validation, gate and exclusive lock.
- `research_dev/talks.md`: progress and blocker.

The scheduler, helper/session lifecycle, native workers and wire formats are
unchanged. Before-images and the working prior physical results are preserved.

## Validation and deployment

66 focused tests and both replay tests pass; both golden hashes are unchanged.
See `TESTS.json`. Compile and whitespace checks pass. No broad suite was run.

Fresh deployment:
`/mnt/storage/s42-remote-resident-long-context-20260914-v1-OUqyHs/`.
It copies the prior working three-session deployment and overlays only the
gate and its tests. `before.tar.gz` preserves those two prior source files.
No completed physical reference is overwritten.

When the other job releases the rig, the bounded order from that directory is:

```sh
python3 -u run_bounded.py calibrate
python3 run_bounded.py configure
python3 -u run_bounded.py preflight
python3 -u run_bounded.py run
```

Completed stage directories must not be reused for retries. Preserve failures
and allocate a fresh attempt before rerunning a stage that wrote artifacts.
The continuation wrote `calibrate-idle-check`, so the next authorized attempt
needs a fresh artifact directory. The other campaign must first release its
session or the user must authorize its normal shutdown.

## What is not established yet

There is no new long-context performance, memory or energy result. Earlier
8.49 GB tensor omission is host memory, not GPU VRAM. The new test must report
per-pool KV allocation and measured host/GPU peaks, full prompt processing,
phone calls, generation-bound terminal proofs, preparation and execution
separately. If both parents fit, it proves execution with relocated weights
at the longer context, not an increase in maximum feasible context.
