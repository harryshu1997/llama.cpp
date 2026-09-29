# Prefill overhead and real context frontier

Status: tests completed through context 32768, then stopped. Physical execution
and memory checks pass, but both 32k outputs are degenerate: useful 32k inference
is NOT demonstrated. No workload trace was run.

Preserve the 20260916-ffn-microbatch-attribution result and all earlier attempts.
Keep model/shard files, 23 GPU layers, default CUDA graphs, seed, output budget,
batch 2048, ubatch 512 and strict execution proofs unchanged. No kernel changes,
USB resets, unrelated process interference, commits or pushes.

## Plan and admission

1. Time the existing resident router's checksum, worker forwarding/response and
   USB write stages. Its worker is TCP-connected; `compute_us` includes graph
   construction and tensor staging, not just NPU arithmetic. Never subtract
   timestamps from different hosts. Receive wait includes desktop idle time.
2. Use that evidence to make a narrow optimization without changing payload
   hashes, geometry, session identity, finite checks or wire formats. Recheck the
   original 5261-token prompt on the real phone.
3. Test progressively larger actual prompts with identical contexts/placements
   in both arms. Enforce live CPU/GPU capacity and stop before unsafe admission.
   An untested or merely allocated context is not a demonstrated context limit.
   Host RAM savings do not count as GPU VRAM savings.

Initial fresh remote directory:
`/mnt/storage/s42-prefill-frontier-20260916-v1-LRKTEg/`.

Desktop and phone binaries/shards are frozen from the prior passing gate except
for explicitly recorded diagnostic/optimization binaries. Preserve intermediate
attempts separately. Phone active power remains assumed at 3/4.5/6 W.

## Prefill diagnosis and narrow repair

The resident router forwards to TCP workers, not the worker's direct DMA-BUF
execution branch. New `RESIDENTTIMING` records isolate router input/output
checksum, worker send/receive and USB write durations. Worker `compute_us`
already includes graph construction and tensor staging. Receive wait is not
reported as active transfer because it also includes desktop idle time.

The instrumented default-scheduling attempt (physical-v1) passed its execution
and memory checks, but was slower than the earlier gate: 78.264 s prefill versus
the earlier 46.435 s. At 512 tokens the router's two FNV checks alone averaged
81.136 ms. This variability is preserved, not substituted for the earlier
reference. An exact-hash loop-unrolling microbenchmark showed no gain; that
optimization was not applied.

A process-scoped high-capacity-core test reduced the same 3,932,160-byte hash
from roughly 30-42 ms to 6.5 ms without changing its result. The canonical phone
configuration now has an optional `cpu_affinity` hexadecimal mask, exposed as
`--phone-cpu-affinity`. It must match the transport qualification identity; an
unqualified or mismatched mask fails closed. It prefixes only the owned phone
session subprocess tree with `taskset`, is recorded in the launch receipt, and
does not change governors, global affinity, other users' processes or the default
when unset. The experiment selected mask `c0` from the live CPUs with the highest
reported capacity, not from a model-specific rule. Phone-worker/shard binaries,
payload hashes, finite checks, protocol and session identities are unchanged.

| Original 5261-token document | Desktop | Relocated |
| --- | ---: | ---: |
| Prior prefill | 22.710 s | 46.435 s |
| Timed default-scheduling prefill | 22.380 s | 78.264 s |
| Process-affinity prefill | 22.802 s | 36.008 s |
| Process-affinity decode, 64 tokens | 29.524 s | 25.586 s |

The affinity attempt (physical-v2) has 1800 proved request calls, one load for
each session, generation 1 throughout, zero resets/fallback, terminal status 0,
and same-boot idle postflight PASS. Its independent audit passes. Router
checksum total at 512 rows falls to 15.537 ms; host-observed round trip falls
from 258.134 to 88.121 ms in the two instrumented attempts. The earlier passing
run was 130.539 ms. Do not claim all default runs incur the diagnostic attempt's
larger cost. Prefill is still slower than desktop, and no total speedup is claimed.

Phone energy is still assumed. Changing CPU affinity may change its actual
power; 3/4.5/6 W sensitivity is not a physical phone-power measurement. Request
energy boundaries also remain asymmetric (remote includes its desktop launch).

## Validation

31 focused tests PASS, including five new affinity tests. Both replay tests PASS
unchanged (85.131 s). No broad suite or workload trace. The earlier 11-test
timing-only check also passed. The initial new test fixture incorrectly used
`dataclasses.replace` on an identity containing immutable mapping proxies; the
fixture now supplies a plain software-identity dict as the existing API requires.

Larger contexts receive fresh canonical cold/hot desktop qualification before
the document gate. The 16k calibration passed; its immediate postflight observed
2% GPU utilization. A later unchanged idle check passed, with its evidence saved
in `CALIBRATION_POSTFLIGHT_RECOVERY.json`. No calibration rerun, threshold
relaxation or foreign process intervention was used.

## Bounded context test design

The user explicitly bounded this experiment at context 32768. No 64k-262k
physical run is authorized by this gate. `physical-v1/CAPACITY_SCREEN.json` is
only a read-only memory prediction, not a physical success at those sizes.

The 16k and 32k documents repeat the prior archive body three and six times.
Both arms receive the same tokenized prompt at each size, with no prefix cache
or truncation, and generate 64 tokens. This tests execution of the real token
count, not merely successful allocation of a context buffer. The archive-key
and semantic-sanity checks do not qualify task accuracy over long documents.

The fixed desktop placement has 23 GPU layers, 25-47. Full-width FFNs in layers
0-23 are owned by the phone; layer 24 remains on the host. This is actual host
weight omission, not a partial-fraction helper that retains a local copy. CPU
attention and GPU placement are unchanged. Each larger context gets fresh
desktop qualification; old-context performance qualification is not copied.

All physical attempts are separate directories. `physical-v2` reuses the nine
transport receipts collected in `physical-v1`, not newly measured affinity-mode
USB throughput. Its new identity records the explicit affinity and new launch
source hashes. The actual model timings are fresh calibration measurements, not
inherited route qualification. The worker, payload hashing, transfer protocol,
model and shard bytes are unchanged.

## Reproduction and changed files

`CHANGES.json` lists exact production/test files and before/after hashes. The
before-images are under `before/`. Apart from these six production files and
one test file, this turn adds report/evidence scripts and updates talks.md.
Unrelated dirty-worktree changes are preserved. No commit or push.

The canonical gate is still `campaigns/burstgpt/remote_resident_gate.py`.
The report scripts only supply configuration, launch its existing calibration,
preflight and execution modes under the rig lock, collect logs and summarize.
They do not choose scheduling policy or replace its memory/session/proof checks.
`GATE_COMMAND.json`, the mode-specific command files, `SOURCE_MANIFEST.json`,
`CONTEXT_SPEC.json` and native receipts retain the exact remote invocations.

After mirroring the completed attempts, regenerate the report into a new file:

```sh
python3 research_dev/scheduler/campaigns/burstgpt/reports/20260916-prefill-frontier/summarize.py --output /tmp/prefill-frontier-summary.json
```

The output uses exclusive creation. `analyze_timing.py` joins host/router records
within physical request proof bounds; it never subtracts timestamps from
different hosts. The independent `audit_result-v2.py` used here is the unchanged
auditor from the prior microbatch-attribution report.

## Final physical results

`SUMMARY-v2.json` is the final summary including direct output inspection.
`SUMMARY.json` preserves the first draft before that inspection; do not use it
as an output-quality acceptance record.

| Context | Real input tokens | Desktop prefill s | Relocated prefill s | Desktop decode s | Relocated decode s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 | 5261 | 22.802 | 36.008 | 29.524 | 25.586 |
| 16384 | 15589 | 64.672 | 105.219 | 30.617 | 27.306 |
| 32768 | 31081 | 129.559 | 210.481 | 36.258 | 31.013 |

Every arm generated 64 tokens, with zero prompt-cache reuse or truncation.
Decode takes 10.8-14.5% less time with relocation, but prefill takes 1.58-1.63x
as long. Combined prefill plus decode remains slower; phone preparation adds
further cold cost.

### Output limitation found during final review

The 8k and 16k outputs contain the required archive key, though their 64-token
explanations are incomplete. At 32k both arms emit repeated channel markers and
downward arrows, and neither contains the key. The existing semantic-sanity
check nevertheless accepts both: marker text contributes lexical characters
and the mixed repetitions miss its narrow repetition thresholds. Its PASS is
not semantic correctness. SUMMARY-v2 records failed archive-key retrieval for
both arms; original gate results and outputs are unchanged. The cause of the
shared desktop/phone output degeneration is not diagnosed here. No validator,
prompt or token budget was changed to pass. Useful 32k inference is unproven.

### Memory and preparation at 32k

| Measurement | Desktop full | Relocated |
| --- | ---: | ---: |
| Process RSS snapshot, decimal GB | 15.409 | 7.115 |
| Process RSS high-water, decimal GB | 24.278 | 15.785 |
| Process VRAM snapshot, decimal GB | 13.063 | 13.059 |
| Native CPU KV buffer, MiB | 520 | 520 |
| Native GPU KV buffer, MiB | 472 | 472 |

The kernel confirms 8,493,170,688 complete-page bytes unmapped with zero VMA
overlap, from 8,493,465,600 omitted tensor bytes. RSS is 8.295 GB lower. VRAM is
effectively unchanged: these are CPU-resident FFNs. Observed total-device VRAM
peaks are 13.888/13.884 GB, including other processes. Both layouts fit every
tested context; a larger maximum context than desktop is not established.

Phone weights total 8,493,465,600 bytes plus 805,306,368 bytes of workspace
reservation: 9,298,771,968 bytes under the declared 10 GB pool. This is not a
measured phone peak. Actual FFN shard paths/hashes and bytes read are in each
audit's `weight_sources`; there is no full-GGUF fallback.

| Session | Weight read s | HTP init s | Upload s | Phone load-to-READY s | Request calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| HTP0 | 8.002 | 0.099 | 0.675 | 8.873 | 1000 |
| HTP1 | 10.278 | 0.105 | 1.096 | 11.625 | 1000 |
| HTP2 | 11.068 | 0.080 | 1.202 | 12.479 | 1000 |

From the first host preparation stage, first post-verification is 17.577 s,
all post-verification 42.338 s, outer preparation 42.352 s. Host READY receipt
times are separately 17.406/29.462/42.295 s. Phone phase durations and host
offsets use their own clocks, never cross-host timestamp subtraction.

One load per session per fresh gate, generation 1 throughout. At 32k: 1488
prefill plus 1512 decode calls, proof range 97-3096 excluding 96 startup calls.
At 8k/16k: 1800/2280 request calls. Independent audits verify assignment,
artifact, geometry, operator plan, session generations, actual microbatch rows
and leases. No within-request reload, fallback or reset; terminal statuses
zero. Each isolated gate performs normal cleanup: Android USB restored at
5 Gbps, same boot ID, idle postflight PASS. No owner-loss/replacement injection.

### Energy: diagnostic phase measurements, not matched savings

| Context | Desktop request server kJ | Relocated request server kJ | Separate phone-preparation server kJ |
| --- | ---: | ---: | ---: |
| 8192 | 5.412 | 3.563 | 0.604 |
| 16384 | 9.318 | 7.549 | 0.654 |
| 32768 | 15.926 | 13.857 | 0.619 |

CPU is measured with RAPL; GPU with integrated board power. Desktop request
excludes desktop launch; relocated request includes it. Cleanup and inter-phase
gaps are excluded. These are not matched end-to-end savings, and the 32k outputs
are not useful answers.

At 32k, assuming active phone power over the entire relocated request gives
14.598/14.968/15.339 kJ at 3/4.5/6 W. Separate phone preparation adds
0.746/0.810/0.873 kJ. Desktop request plus assumed 0.875 W idle phone is
16.071 kJ. These conservative phase sensitivities are diagnostic; no savings
percentage or break-even is claimed from unequal boundaries. SUMMARY-v2
contains all contexts and preserves measured CPU/GPU components separately.

### Artifacts and conclusion

Remote attempts, mirrored under the matching `physical-vN/` folders:

- `/mnt/storage/s42-prefill-frontier-20260916-v1-LRKTEg/`: timed default, preserved.
- `/mnt/storage/s42-prefill-affinity-20260916-v2-nuxeVS/`: 8k affinity.
- `/mnt/storage/s42-context16384-20260916-v3-86SDY3/`: 16k.
- `/mnt/storage/s42-context32768-20260916-v4-9bEpE6/`: 32k.

Gate SHA-256:

- 8k: `61f8937385f605394a8a357c23635d131e569c50c75acd2fbf9f8a0830b5b76a`
- 16k: `53ad3ed64b9f9f6e317873fbab1b9d885a410ca25b15a2ba5830f1aabf70368c`
- 32k: `9eae2f33dff1638f2764179440d9d30fb7a971d9d083e023dfa1e2b6d6aa6e71`

Final SUMMARY-v2 SHA-256:
`50a9cfc91ea451fcaa5a26fc44aaf1589a2ab039550e9e61c82ac087288fb6b3`.
Native/model/shard and parent/assignment hashes are in the audits and summary.
Six production file hashes match the deployed manifest. 33 focused/replay tests
pass with unchanged goldens. No larger context, trace, broad suite, kernel
change, commit or push.

Remaining, not started: diagnose useful-output failure using a validated
long-document task, and reduce remaining serialized prefill forwarding,
staging and checksum costs. At 512 rows the optimized round trip is still
about 89 ms versus roughly 32 ms inside the worker, which itself includes
graph/staging work. Host RAM release does not free GPU KV capacity.
