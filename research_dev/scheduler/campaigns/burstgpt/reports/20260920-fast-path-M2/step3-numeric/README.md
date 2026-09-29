# M2 revised Step 3: row checks pass; determinism is inconclusive

**Step 3.1: PASS** for the requested 64-step row-error window and historical exact-token comparison in both runs. **Step 3.2: INCONCLUSIVE**: the concurrent repetitions produced no identical input-row hashes, including after alignment by request. This does not establish either phone determinism or nondeterminism. The condition for replacing exact tokens with logits NMSE <= 5e-4 is unmet, so the M2 correctness rule is unchanged and the N=1/4/8 curve remains pending. No third physical run was added.

Two runs used the same native runtime hashes and request-fixture digest, N=2, prompts (256,257), concurrent submission, 576 outputs per slot, ubatch 1024, eight unpinned threads, keep-cache + populate restore. OP15 alone served all phone work on ADB 5037. Pack/unpack order and acknowledgement behavior remain unchanged. The withdrawn mapping build in `../step3/` was not used for any physical acceptance run.

| Single diagnostic run | Request slots 0 / 1 | Own ack indices 0 / 1 | Request 0 vs historical host | Request 1 vs historical host | Max local_rel_l2 | Rows > 1e-2 / checked | Full-cohort steps per layer |
| --- | --- | --- | --- | --- | ---: | --- | ---: |
| run1 | 1 / 0 | 3 / 3 | 576/576 | 576/576 | 3.195e-4 | 0 / 2304 | 573 |
| run2 | 0 / 1 | 3 / 4 | 576/576 | 576/576 | 3.566e-4 | 0 / 2304 | 572 |

Both runs completed without a watchdog event, with exact per-request phone proofs. Run 1 made 10,314 two-row calls. Run 2 made 10,296 two-row calls and 18 one-row tail calls. The original request slots [1,0] and acknowledgements [4,3] did not recur together: Run 1 had [1,0] and [3,3]; Run 2 had [0,1] and [3,4]. The original transient remains unreproduced.

The 64-step diagnostic retained resident host weights until all 1,152 diagnostic calls and 2,304 returned rows completed in each run. Only then did each arm release 9,625,706,496 mapped host bytes. The row windows cover every owned layer (0-17) for both requests. No row exceeded 1e-2. The maxima and their exact mappings are:

| Run | Call | Layer | Ubatch row | Request / slot / member | Payload row | Assisted step | local_rel_l2 | local_max_abs | local_l2 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| run1 | 977 | 4 | 0 | 1 / 0 / 1 | 0 | 55 | 3.195e-4 | 1.923e-3 | 7.227 |
| run2 | 1067 | 4 | 1 | 1 / 1 / 1 | 1 | 60 | 3.566e-4 | 1.72e-3 | 5.163 |

Every per-call/row maximum is retained in [Run 1 CSV](../physical/step3-numeric/run1/MAX_LOCAL_REL_L2_PER_CALL_ROW.csv) and [Run 2 CSV](../physical/step3-numeric/run2/MAX_LOCAL_REL_L2_PER_CALL_ROW.csv). The corresponding `CHECK_DIAGNOSTIC.json` files retain all input/local/returned hashes, mapping fields, numeric distances and the empty threshold-exceedance lists. Result summaries use at most three decimal places, with scientific notation for small errors. Raw records retain the precision emitted by the native logger.

| Determinism comparison | Count |
| --- | ---: |
| Corresponding call IDs | 1152 |
| Compared payload-row positions | 2304 |
| Identical input-row wire hashes at the same payload position | 0 |
| Identical input-row wire hashes after alignment by request | 0 |
| Rows in calls with identical full inputs | 0 |
| Differing returned hashes for identical inputs | 0 (no eligible comparisons) |

[CHECK_DETERMINISM.json](../physical/step3-numeric/CHECK_DETERMINISM.json) records every call/row comparison. Its `status: PASS` means the structural analysis completed; with zero eligible inputs it is not a positive determinism result. [REQUEST_ALIGNED_WIRE_COMPARISON.json](../physical/step3-numeric/REQUEST_ALIGNED_WIRE_COMPARISON.json) also checks the rows by logical request. [STEP3_CHECK.json](../physical/step3-numeric/STEP3_CHECK.json) records the experiment verdict as **INCONCLUSIVE**. Native/runtime identity equality and equal request configuration did not produce equal runtime rows under concurrent admission.

Both runs match all 576 tokens per request against the original host reference at ubatch 1024. That host reference has different native hashes; this is a historical comparison, not a fresh host/phone acceptance pair. Both runs also match both 64-token prefixes of the reversed Step 2 host reference. Against the normal Step 2 host, request 0 matches 64/64 and request 1 matches 42/64: current token 4 is 198, while that host's token 4 is 271. The two Step 2 host-only references already reproduce this difference under identical native hashes. Slot assignment and prefill history differ together; these records do not isolate their individual effects. Full positions and reference identities remain in each run's `EXACT_TOKENS.json`.

The energy below is measured host RAPL package + NVML board energy. Assumed phone power and energy are separate. These are single diagnostic runs with 64 host-shadow steps, not utilization-curve measurements. They do not replace the original 65.8 W versus 123.4 W and 21.2 kJ versus 45.0 kJ results, which remain a **single run from a failed correctness pair**.

| Run | Request host J, measured | Decode host J, measured | Decode host W, measured | ms/token request 0 / 1 | Phone W, assumed | Request phone J, assumed |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| run1 | 23940.106 | 23387.188 | 73.738 | 552.844 / 552.844 | 4.5 active; 0.875 idle | 1434.582 |
| run2 | 24583.511 | 23827.006 | 74.838 | 554.914 / 558.881 | 4.5 active; 0.875 idle | 1442.211 |

| Run | memory.peak bytes | memory.events.max ready | finish | delta |
| --- | ---: | ---: | ---: | ---: |
| run1 | 30105604096 | 0 | 0 | 0 |
| run2 | 30274908160 | 0 | 0 | 0 |

Both scopes used MemoryMax=infinity and MemorySwapMax=0. High-limit, OOM and OOM-kill events stayed zero. Cleanup passed at 2026-09-21 00:03:28 UTC: both scopes inactive, both owned server PIDs gone, GPU idle, lock acquired nonblocking and then released, OP15 restored on port 5037 with the expected kernel. Both phone terminal records have status 0. [Cleanup](../physical/step3-numeric/CLEANUP.json). The first report-only cleanup reader looked for an abort receipt; its retained error log is `CLEANUP_CHECK_FIRST_ATTEMPT.log`. Reading the normal close receipt corrected the check without touching a process or rerunning either arm.

The only native change for these repetitions extends the existing diagnostic bound to 64 (0 and 5 remain supported). The typed launch field defaults to 0, enters the existing environment/runtime digest when enabled, and retains the full-width, dormant, runtime-control and unsupported-mode checks. The shared numeric metrics are preserved. The tiny-model test covers 5 and 64 steps, unchanged logits and zero distances with an exact worker. All 96 rig unittests pass before each arm; pyflakes is clean. The local 96 tests pass with one server-only skip. [Isolated changes](software/IMPLEMENTATION.patch) and before/after source hashes are under `software/`.

Deployment: `/mnt/storage/s42-fast-path-M2-numeric-20260920-2d61e7`. Fresh CUDA build and transport materialization completed before Run 1; both arms verify the identity before launch. Qualification identity: `sha256:ba097dac079947e59b01a90d3973d0804fb90fa6af359a1af2bf687deb40a8a3`. All desktop models, traces, logs and temporary test files are under `/mnt/storage`. Every arm holds the rig lock and uses a free port with server PID/argv verification; cleanup uses the existing normal close path. No phone worker was force-killed; no second phone was used.

Exact commands from the controller repo root:

```sh
bash research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/step3-numeric/DEPLOY.sh
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M2-numeric-20260920-2d61e7/BUILD.sh'
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M2-numeric-20260920-2d61e7/MATERIALIZE_TRANSPORT.sh'
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M2-numeric-20260920-2d61e7/RUN_REPEATS.sh'
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M2-numeric-20260920-2d61e7/CLEANUP.sh'
rsync -a zhihao@172.20.74.85:/mnt/storage/s42-fast-path-M2-numeric-20260920-2d61e7/physical/ research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/physical/step3-numeric/
bash research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/step3-numeric/POSTPROCESS.sh
```

`RUN_REPEATS.sh` runs the touched modules' unittests and pyflakes through `CHECK.sh` before each arm. Each command record includes the exact argv, config and gate source. Each run retains all per-call/row maxima, every threshold exceedance, row wire hashes, controls, token traces, energy and memory events. The build is requalified before either trace starts.
