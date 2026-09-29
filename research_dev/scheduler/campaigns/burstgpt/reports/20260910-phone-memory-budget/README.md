# Whole-phone memory budget measurement

The old 3,000,000,000-byte Llama reservation was configuration, not a measured
peak. A bounded physical probe used the current whole-phone binary, NCM control,
the unchanged Llama 37 prompt and all 292 output tokens, context 4096, batch 1024,
ubatch 256, parallel 1 and 16 GPUOpenCL layers. One Gemma HTP shard remained
resident. No trace, baseline campaign, native rebuild, USB reset, GDM change or
unrelated process termination was performed.

The first request completed physically on the phone. The second was selected
on desktop after the phone route exceeded its predicted latency upper bound.
The two-phone-request reuse gate therefore FAILED; that failure is preserved.
This is not a successful three-request mixed run or a qualified energy profile.

## Measurement and provisional configuration

| Quantity | Bytes |
| --- | ---: |
| Llama tensor weights | 763,097,216 |
| CPU process RSS high-water mark | 1,105,862,656 |
| GPU driver kernel-allocation high-water mark | 1,430,216,704 |
| Conservative sum of lifetime peaks | 2,536,079,360 |
| Proposed tested-shape reservation | 2,801,795,072 |
| Old configured reservation | 3,000,000,000 |
| Bytes recovered by the proposal | 198,204,928 |

The proposed budget is the measured peak sum x 1.10, rounded up to 16 MiB.
The CPU and GPU peaks need not coincide and mapped allocations can overlap;
no overlap is subtracted. All 108 samples had zero imported GPU memory and
zero process swap. Missing counters, PID reuse or boot/lifetime mismatch fail
closed. Swapped or imported allocations explicitly mark peak accounting as
incomplete. Peak observations do not receive current resident-memory credit.
The existing generation-bound PSS observation and admission checks are unchanged.

This budget is provisional for the tested shape and launch contract, not proof
for arbitrary contexts, batches, binaries, per-shape prepack accumulation or
repeated phone requests. Do not promote the memory budget or energy evidence
to qualified status from this failed reuse attempt.

The 10,000,000,000-byte declared pool and 805,306,368-byte safety reserve are
unchanged. With the proposed whole-model budget, the aggregate HTP cap would be
6,392,898,560 bytes including its workspace. Reserving 4,951,552 workspace bytes
leaves 6,387,947,008 bytes for FFN weights. Session and layer selection remain
owned by the existing scheduler; no session is dedicated to the whole model.

## Code changes

- `adapters/probes.py`: separate process-lifetime-checked CPU/KGSL peak parser.
- `adapters/android_llama_server.py`: read-only peak probe over the existing
  USB/NCM control path; no new endpoint or reset path.
- `configuration/campaign.py`, `config.py`, `campaigns/burstgpt/arguments.py`,
  `campaigns/burstgpt/launch.py`, `campaigns/burstgpt/runner.py`: expose the existing
  `set_phone_htp_memory_cap` API as explicit typed campaign configuration. No
  runner-side shard selection or route policy; default configurations unchanged.
- `_unified/phone_residency_ops/economics.py`: capped static budgets preserve the
  larger catalog/live reserve even when physical RAM exceeds the declared pool.
- `_unified/phone_residency_ops/publication.py`: a cold unqualified endpoint may
  pass memory admission when actual retained allocations, its complete peak and
  fresh live memory prove capacity. Energy qualification is not substituted for
  that memory proof; cold requests still reserve every byte of the peak and
  energy/latency route selection remains unchanged.
- `campaigns/burstgpt/phone_memory_probe.py`: bounded measurement using existing
  preload, scheduler-owned calibration, request tickets and cleanup.
- Tests: `test_phone_allocation_snapshot.py`, `test_phone_memory_cap.py`,
  `test_campaign_inputs.py`.

## Remaining blocker

The phone request used 65.547653 s for 292 tokens. The subsequent identical
desktop request used 1.534981 s. Phone execution energy was 1,024.358 J versus
191.100 J desktop at assumed phone power 4.5 W, with physically measured CPU/GPU
energy. These are diagnostic execution windows, not matched trace totals, and
exclude the separately recorded preparations. The first completion recorded
`prediction_late`, an upper-bound overrun of 46.262139 s, full lease coverage and
`disabled_for_remaining_run`. The second request was not forced back to phone.

Normal energy-aware scheduling also still lacks qualified cold-route energy
evidence. A memory admission improvement must not override either protection.
The requested three-request phone demonstration requires an explicitly labelled
calibration path and corrected, measured latency estimates, or a genuinely
competitive phone route. No such change is silently made here.

Remote attempt: `/mnt/storage/s42-phone-memory-20260910-v1`.
Deployment: `/mnt/storage/s42-phone-memory-20260910-v1-deploy`.
The `physical/` copy preserves both terminal executions, failure, cleanup,
memory samples, submissions, snapshots and commands/logs. Exact hashes and final
test results are recorded below.

87 focused tests passed in 95.603 s, including both byte-identical replay
goldens. No complete scheduler harness was run. AST and whitespace checks passed.
The later probe-only persistence change also records incomplete accounting and
cleanup failures explicitly; the deployed v1 probe and failed result are retained.

- v3 replay: `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`
- v8 replay: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

## Decision-only capacity result

The existing offline planner, using the saved initial physical snapshot, current
catalog/artifacts and Gemma's actual parent, generated three nonempty sessions:

| Session | CPU FFN layers | Resident bytes |
| --- | --- | ---: |
| HTP0 | 0-1 | 707,788,800 |
| HTP1 | 8-15 | 2,831,155,200 |
| HTP2 | 16-23 | 2,831,155,200 |

Total FFN weights: 6,370,099,200 bytes. Adding 4,951,552 workspace,
2,801,795,072 whole-phone reservation and 805,306,368 reserve gives
9,982,152,192 bytes, leaving 17,847,808 bytes inside the declared 10 GB pool.
The assignment was selected by the existing planner, not forced in the runner.
This is a decision-only candidate, not physical READY state or released memory.
Layout geometry: `048913e6e4bfaacbc55a0bf2fcc4a49f1ccdda44db265126a62cfb09b1519abc`.

## Physical artifact hashes

- Phone execution: `d352533181ba15e53b100a4123d6206b693dae6de5cee81562830414b22dd252`
- Desktop execution: `27c0a95606c1a1fe5040d22a9a55ef6ed620e589d5d942b6d2394c01733da486`
- Memory samples: `4ab859523d5625d960cb0d693eca64ba6a6887004630a7bea83a769e16212c03`
- Failure: `95fa603d431c96183fb4291bb83142ed4e11dcdee38c9c57b91375741bf5a8ec`
- Cleanup: `74e64413468ea142ee243fba6eb1ec4f6a00c687fc657217c3e055e6cdf1d6b4`

Read-only post-cleanup audit: normal `ptp,adb`, zero remaining owned phone or
desktop inference workers, original GDM worker still running, GPU 3,178 MiB used
and 12,770 MiB free. No reset or manual process termination was used.
