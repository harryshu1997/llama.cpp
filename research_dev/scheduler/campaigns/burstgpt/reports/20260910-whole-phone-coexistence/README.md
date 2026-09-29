# Whole-phone control recovery through FunctionFS

The final bounded gate passed inference, allocation observation and cleanup.
No trace, baseline, native rebuild, commit or push was performed. GDM and unrelated
processes were preserved. This is a transport/co-residency diagnostic, not a
concurrent-compute or energy-saving qualification.

## Physical result

The existing scheduler's calibration mode selected the whole-phone route. The
runner did not force a route or weaken memory, artifact or execution proof checks.
The diagnostic used an existing 915-token Llama prompt with a new, bounded
32-output-token request, not a shortened trace result.

| Measurement | Final gate |
| --- | ---: |
| Whole-phone Llama output | 32/32 tokens, semantic validation passed |
| Request service time | 9.428042 s |
| First HTP session READY from preload epoch | 17.367658 s |
| Physical shard read/init interval | 8.761313 s |
| FFN resident / file bytes | 2,831,155,200 / 2,831,157,472 |
| Shard loads / online reloads | 1 / 0 |
| Retained HTP generation / whole-endpoint generation | 1 / 3 |
| Valid allocation observations / unique source samples | 32 / 24 |
| Allocation observation span / maximum age | 15.107116 s / 0.859082 s |
| Observed process PSS range | 366,210,048 - 441,638,912 bytes |
| Fallbacks / USB reset recoveries | 0 / 0 |
| Owned phone workers after cleanup | 0 |

The selected shard was Gemma HTP0, with its existing F16 FFN file, layers 0-7
and 15,360 stored/active columns. It remained READY and unchanged throughout
the Llama request. It received **zero FFN calls** in this diagnostic. Llama ran
entirely on the phone through its OpenCL server; the proof's `phone_call_count`
field counts FFN helper calls, not whole-model execution, so that field is zero.

The 32 allocation observations include repeated reads of the background monitor;
there are 24 distinct source timestamps. `physical/SUMMARY.json` calls the former
`allocation_samples`; the distinction is explicit here. Each sample is bound to
artifact, endpoint generation, launch identity, PID, boot ID and process start
ticks. PSS is partial resident process memory, not the full OpenCL peak. The
3,000,000,000-byte whole-service declaration was a conservative reservation, not
a measured or qualified peak. Unobserved memory retains its reservation.

The files were already staged and the OS cache was not flushed. These load times
are not a cold-storage benchmark. The separate FFN preparation receipt records
551.098881 J fleet energy, including 73.349042 J assumed phone energy. Whole-model
execution records 30.427847 J CPU, 70.793651 J GPU and 42.423370 J assumed phone
energy at 4.5 W. These diagnostic intervals are not a matched comparison or a
whole-experiment energy total. No savings claim is made.

## Causes and repairs

1. The OP15 was back on the stock kernel. Restored the exact qualified image with
   `fastboot boot`, RAM-only. Nothing was flashed, erased, unlocked or re-slotted.
2. Android init stopped adbd when FunctionFS disabled the normal Android gadget.
   The TCP listener therefore disappeared despite its persistent port setting.
   A bounded, hash-verified bootstrap restores only the existing authenticated
   TCP service after the changeover. It does not toggle USB for telemetry.
3. The existing Android launcher now declares `adb-usb` or `adb-ncm` explicitly.
   The latter verifies phone/boot identity and uses NCM for process control, logs,
   memory probes and inference forwarding. Forwarding cannot overwrite an
   existing binding. NCM costs remain estimated, not copied USB qualification.
4. The health fallback still named the disappeared USB serial. It now resolves
   the verified NCM connection while FunctionFS is active, with unchanged sample
   freshness and conservative admission. The diagnostic waits for fresh health
   before submitting its calibration request; execution is not forced.
5. A diagnostic ownership error disconnected NCM before server cleanup. The old
   stop routine ignored remote command failures. Stop now checks process identity,
   requires command success and confirms exit. Failed stops retain the endpoint
   record, and cleanup preserves control while an endpoint still owns it.
   PASS publication now follows successful cleanup.

The preceding six-field `phone_power_*` launch-normalization repair was preserved
and retested, including the ASSUMED_4P5W snapshot round-trip regression. Energy
annotations remain in costing and physical execution records.

## Preserved attempts

All directories below are under `physical/` locally and the remote root below.

| Attempt | Result and cause |
| --- | --- |
| `kernel-restore` | PASS: exact qualified kernel/BTF restored |
| `transport-attempt-1` | FAIL: NCM TCP connection refused after adbd stopped; one shard loaded |
| `transport-attempt-2` | FAIL before loading: ADB push into root-owned directory denied |
| `transport-attempt-3` | PASS: control/telemetry across FunctionFS, 12 probes |
| `whole-attempt-1` | Diagnostic failed on an incorrect ticket attribute; corrected to `ticket.decision.route_id` |
| `whole-attempt-2` | Phone rejected for unavailable telemetry; scheduler correctly executed desktop |
| `whole-attempt-3` | Inference passed, cleanup failed after premature disconnect; overall FAIL |
| `whole-attempt-4` | PASS: whole-phone inference, fresh allocation telemetry and cleanup |

The original `whole-attempt-3/RESULT.json` was published before cleanup and is
preserved. Its `POST_RUN_AUDIT.json` explicitly marks the overall attempt FAIL
and documents identity-checked cleanup of that attempt's PID 28178. No unrelated
process was signalled. The final attempt's PID 6619 was stopped normally and
its exit verified. The post-gate audit found no research phone workers; desktop
VRAM returned to 3,178 MiB used / 12,770 MiB free, with GDM PID 6871 unchanged.

## Software validation and source identity

228 focused and related tests passed in 105.282 s:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_dev/scheduler/tests:. python3 -m unittest test_android_ncm_control test_phone_allocation_snapshot test_phone_memory_cap test_phone_power_probe test_telemetry_recovery test_physical_residency test_llama_server_adapter test_physical_adapter test_runtime_resources test_catalog_materialization test_campaign_inputs test_replay_determinism
```

One additional failure-only shutdown guard and regression were added while the
last gate was running. All 16 NCM tests then passed (0.009 s), including the 15
already in the larger run: 229 distinct tests, not 244. The last guard was
software-tested, not physically fault-injected. Its complete decoded diff and
before/after hashes are in `POST_GATE_SOURCE_DELTA.json`. The gate's executed
source copies and hashes are preserved in `physical/SOURCE_GATE/` and
`physical/SOURCE_GATE.json`; they are not presented as the later source.

Both replay goldens remain unchanged:

- v3: `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`
- v8: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

Shell syntax and whitespace checks passed. No complete scheduler harness was
run. The worktree remains dirty and user-owned; Git HEAD is
`5f89a2d9d33be547a1bdef5fd0f504a279c50800`. Source-file hashes, not that commit
alone, identify the executed code.

Changed production files, relative to `research_dev/scheduler/`:

- `adapters/android_llama_server.py`
- `adapters/heterogeneous_rig.py`
- `adapters/probes.py`
- `adapters/catalog_materialization.py`
- `adapters/native/android_ncm_adb_control.sh` (new)
- `configuration/rig.py`
- `campaigns/burstgpt/arguments.py`
- `campaigns/burstgpt/catalog.py`
- `campaigns/burstgpt/launch.py`
- `campaigns/burstgpt/runner.py`

Also added `tests/test_android_ncm_control.py`, the diagnostic, owned-cleanup
and audit scripts in this report directory, this report and source-delta record.
Updated `ARCHITECTURE.md` and `research_dev/talks.md`. Existing HTP transactions,
shard generators, native workers, wire formats, replay fixtures and frozen
baselines were not changed.

## Artifacts and limits

Remote root: `/mnt/storage/s42-whole-phone-coexistence-20260910-v1`.
Passing run: `whole-attempt-4`. Deployment:
`/mnt/storage/s42-whole-phone-coexistence-20260910-v1-deploy`.
All 763 files in `physical/ARTIFACT_HASHES.json` were copied and verified locally
with zero mismatches. Selected SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| Final RESULT.json | `340faf1c99e0114024bdae635854b39c1200b99fc0b4210c7ea55cc27ce095a2` |
| WHOLE_EXECUTION.json | `09ee4ae97a46ae2f127e1ba52e3fa238789ed27fcaefaa1b953b7c2403f6a848` |
| Final CLEANUP.json | `cfc70e6172706f5dae24cebf001197bd517f01cc852087636886c8b6ce2dae8b` |
| Artifact hash manifest | `0162d70e044b84ef72fb434993b82dc825d571c8628664fd3dc1f1562ff28f38` |
| Qualified boot image | `26e8d41808b10b70264d958582bb6f6c9fba634c34275e0bc28fc820a6d3fb8d` |
| Verified kernel BTF | `f3afcf985b24de3eb5a1d5453bf4ffd95963b806ce99a59430c5a8bf7d201d17` |
| New control bootstrap | `fcb949486d164f6d65ac090ca2e14b307b0836a6cd4238dbeed77612d32e2b15` |

Remaining, not established by this gate:

- Simultaneous HTP calls and Adreno inference. The current whole-model capability
  still reserves `op15-htp`; no concurrency claim is justified by residency alone.
- Full GPU/KV/workspace/OpenCL-prepack high-water qualification and memory credit.
- Independent NCM lifetime without an FFN transport owner and idle-only shard
  resizing, already documented separately in the architecture guide.
- Energy advantage over a matched desktop route. No baseline was run here.

The qualified kernel is a temporary RAM boot; a normal reboot can restore the
stock kernel. Subsequent preflight must continue checking its exact identity.
