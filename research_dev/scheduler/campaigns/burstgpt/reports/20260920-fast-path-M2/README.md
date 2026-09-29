# M2: multi-slot phone decode acceptance

**Current M2 check: FAIL on utilization**, after the final pair completed at
2026-09-21 02:38 UTC. [Amended acceptance report](acceptance/README.md).
All 15 slots pass correctness (14 EXACT, one first-mismatch NEAR_TIE), N=1
matches the saved outputs, and every long arm finishes without a worker hang.
Phone compute time rises at N=8, failing the plan's decreasing-time requirement.
M3 has not started; work stops for the user's decision.

| N, single pair | Phone compute ms/token/slot | Phone RPC ms/token/slot | Host-only decode W, measured | Phone-arm host decode W, measured | Phone W, assumed |
| --- | --- | --- | --- | --- | --- |
| 1 | 164.651 | 194.934 | 124.304 | 66.846 | 4.5 active; 0.875 idle |
| 2 | 82.999 | 103.113 | 124.962 | 67.197 | 4.5 active; 0.875 idle |
| 4 | 51.488 | 67.398 | 133.308 | 66.018 | 4.5 active; 0.875 idle |
| 8 | 64.171 | 77.996 | 145.646 | 57.087 | 4.5 active; 0.875 idle |

Ordered Step 3.2 is closed: slots 1/0, own acknowledgements 4/3 in both runs;
90/90 matched-input calls return identical hashes (fraction 1.000). The N=4
near-tie is at output 80, request 2 / slot 1: margin 3.891e-4, NMSE 8.226e-7.
All 434 later mismatches are recorded after context divergence. The amended
checker, raw logits, energy/latency and per-arm memory records are linked in
the report. All 100 rig tests and pyflakes pass before each arm; cleanup passes.
Historical results below retain their original failed/single-run labels.

## Historical initial check and investigation

**Initial M2 check: FAIL**, 2026-09-20 19:15:50 UTC. The N=2 pair completed without a hang, but the
257-token prompt diverged at output token 4. M2 stopped for the user's decision.
N=1 regression, N=1/4/8 pairs and the complete utilization curve were not run.
M3 has not started and still requires OP12 to be moved physically to the desktop.
All original observations below are **single runs from a failed correctness pair**.
The user subsequently authorized a four-step row-mapping investigation.
[Step 1](diagnostic/README.md) passed instrumentation coverage but failed to
localize a faulty row: all 180 phone/local hashes differ, even though both
64-token historical host prefixes match. The [ordered matrix](diagnostic/MATRIX.md)
passed all eight matched slot/submission cases (64/64 in every slot); mapping
and acknowledgement behavior remain unchanged. Its new host-only (256,257) references reproduce the
original token-4 difference when submission order and slot assignment flip,
under identical native runtime hashes. The original discrepancy is therefore
not isolated to the phone path. The full recheck was outstanding at that point.

The user replaced the planned mapping change with [two concurrent repetitions using
64-step numeric row diagnostics](step3-numeric/README.md). Pack/unpack and acknowledgement
behavior remain unchanged. The withdrawn mapping build was not used for a physical
acceptance run. Both repetitions pass the 64-step row threshold and match all 576 historical host
tokens per request. The determinism check is inconclusive: no input-row hashes match
between repetitions, including after request alignment. At that point the conditional tolerance amendment was unmet. The later user amendment
and final recheck above supersede that decision; these diagnostic observations remain unchanged.

| Check | Result |
| --- | --- |
| At least 512 full-cohort decode steps without a worker hang | PASS: 572 steps on every owned layer; no watchdog event |
| One call carries both slots, with exact per-request proof rows | PASS: 10,296 two-row calls plus 18 one-row tail calls; one USB transfer per call |
| Every slot's tokens equal its matched host reference | FAIL: 256-token prompt 576/576; 257-token prompt 51/576 matching positions |
| N=1 output unchanged; N=4/8 correctness; decreasing N=1/2/4/8 time curve | Not run after the failed N=2 pair |

The requests are matched by their prompt tokens and request index. Native slot
assignment differs between arms:

| Request index | Prompt tokens | Host slot | Phone slot | Matching positions / outputs | First mismatch, one-based | Host / phone token ID |
| ---: | ---: | ---: | ---: | --- | ---: | --- |
| 0 | 256 | 0 | 1 | 576 / 576 | none | - |
| 1 | 257 | 1 | 0 | 51 / 576 | 4 | 198 / 271 |

The common phone control acknowledged output indices 4 and 3 respectively.
The second request's first mismatch is its first assisted output. Its outputs
5-42 match again, with further divergence starting at 43. Root cause remains
undiagnosed. Full mismatch positions, token hashes and slot identities are in
[FAILURE_DETAILS.json](physical/FAILURE_DETAILS.json); the unchanged checker
returned exit 1 in [CHECK_N2_pair.json](physical/CHECK_N2_pair.json).

| Arm, N=2 | Request host J, measured | Decode host W, measured | Decode ms/token, request 0 / 1 | Phone W, assumed | Request phone J, assumed |
| --- | ---: | ---: | --- | --- | ---: |
| Host-only | 44996.318 | 123.402 | 627.573 / 627.572 | 0.875 idle | 322.706 |
| 100% phone FFN split | 21178.192 | 65.801 | 546.3 / 542.107 | 4.5 active; 0.875 idle | 1409.263 |

Host energy is measured RAPL package plus NVML board. Request energy is counted
once for the concurrent group. Decode power uses the intersection of the slots'
decode intervals, beginning after the common control acknowledgement in the phone
arm. Per-slot latency uses the server's `predicted_ms / output_tokens`. Phone
energy uses assumed power over the union of assisted decode time and idle power
over the remaining request interval. It is not added to the measured host column.

The one available curve point, N=2, is 83.063 ms of phone compute and
103.551 ms of host-observed phone RPC time per generated token per slot.
The fixed normalization is the sum over full-cohort physical calls divided by
`full_batch_calls * N / 18`. All partial start/drain calls are retained in the
records; the 18 one-row tail calls are excluded from this fixed-N point.

| Arm | memory.peak bytes | events.max at ready | at finish | delta | Request major-fault delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Host-only | 29870563328 | 0 | 0 | 0 | 101 |
| Phone split | 30134321152 | 0 | 0 | 0 | 100 |

Both arms used fresh `MemoryMax=infinity`, `MemorySwapMax=0` scopes. High-limit,
OOM and OOM-kill events stayed zero. Model paging remains visible in the recorded
memory statistics. The phone arm released 9,625,706,496 mapped host bytes in
33,419 us with keep-cache + populate selected.

Deployment: `/mnt/storage/s42-fast-path-M2-20260920-d7BA1a`. Both arms used the
same native runtime hashes and KV plan: Qwen3-14B F16, ubatch 1024, batch 2048,
eight unpinned threads, 16 GPU layers, context 32768, parallel 2. Prompts are exact
256/257-token document-prefix-plus-task-suffix variants, with 576 generated tokens
per request, seed 17 and greedy sampling. Phone layers 0-17 use all 17,408 FFN
columns, the existing three HTP sessions, max_tokens 4 and `coalesced-batch`.
Transport identity after the rebuild is
`sha256:d66ed4b2f2ee775039bad2916cd00520836912b34a4894fe8f6c6bb296116a15`.
The phone binaries and kernel are unchanged; all ADB operations used port 5037.

The existing client already packed contiguous rows. Implementation extends the
existing cohort control to eight members, bounds coalesced admission by worker
rows and the execution contract, and rejects undersized launch/payload geometry.
Default split-row admission remains limited to four. Existing typed batch/row
fields enter the runtime and transport digests. The existing gate gains concurrent
issuance, per-slot proofs, shared energy accounting and a 60-second progress
watchdog that preserves the rig lock/session while draining is unconfirmed.
No phone kernels were changed. The isolated diff is
[IMPLEMENTATION.patch](software/IMPLEMENTATION.patch).

Software checks: all 90 rig tests pass; local 90 tests pass with one skip because
the local CPU build has no server binary. Pyflakes is clean. The tiny native probe
checks 1/2/4/8-sequence argmax identity and logits within 1e-4; the rig also tests
the actual server's eight-member HTTP validation. `RUN_ARM.sh` ran these checks
before each physical arm. Logs and source hashes are in `software/` and the copied
per-arm `physical/*.unit.log` / `*.pyflakes.log` records.

Exact setup commands from the controller:

```bash
bash research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/DEPLOY.sh
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M2-20260920-d7BA1a/BUILD.sh'
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M2-20260920-d7BA1a/MATERIALIZE_TRANSPORT.sh'
```

Physical runs and checks, on the desktop, in order:

```bash
bash /mnt/storage/s42-fast-path-M2-20260920-d7BA1a/RUN_ARM.sh 2 combined
cd /mnt/storage/s42-fast-path-M2-20260920-d7BA1a/native-source
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 ../ANALYZE.py --root /mnt/storage/s42-fast-path-M2-20260920-d7BA1a/physical --n 2 --phone-only
bash /mnt/storage/s42-fast-path-M2-20260920-d7BA1a/RUN_ARM.sh 2 control
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 ../ANALYZE.py --root /mnt/storage/s42-fast-path-M2-20260920-d7BA1a/physical --n 2
```

The phone-only check returned 0; the paired check returned 1. Expanded systemd/gate
argv and exact command-source text are in `physical/n2-*.command.json`; checker
commands are in [ANALYZE_COMMANDS.json](physical/ANALYZE_COMMANDS.json). Raw streams,
native call/USB logs, phone receipts, power samples and memory observations remain
in `physical/n2-combined/` and `physical/n2-control/`.

Cleanup verified at 19:16:51 UTC: both scopes inactive, no owned server PID alive,
GPU idle, rig lock free, OP15 visible on ADB 5037. The phone restored normally;
no worker was force-killed. See [CLEANUP.json](physical/CLEANUP.json). Code and
reports remain uncommitted.
