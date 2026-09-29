# M2 row mapping diagnostic

**Step 1: instrumentation PASS; fault localization FAIL**, 2026-09-20 20:40:48 UTC. The user requested this investigation after the failed
N=2 correctness pair. The original measurements and records remain in the parent
report, labeled as a single run from a failed correctness pair.

The instrumented client records physical ubatch row, sorted context member,
request, slot, payload row, input position, member acknowledgement and current
output index. Input, local-shadow and returned rows have SHA-256 hashes in both
F32 and the configured wire format. For this diagnostic the existing payload
pack/unpack order is unchanged. The host computes a full FFN shadow and delays
its dormant release for the first five assisted steps. The shadow is discarded.
The default is zero diagnostic steps; no shadow graph or hash records are made.
The typed launch field is `ffn_row_diagnostic_steps`; its nondefault value enters
the runtime digest. The launcher requires a native support acknowledgement.
Partial-width and remote-resident diagnostic configurations fail closed.

| Software check | Result |
| --- | --- |
| Local touched-module unit suite | PASS: 92 tests, one server-only skip |
| Tiny model, diagnostic enabled and disabled against local reference | PASS: same argmax, logits within 1e-4, local/returned row SHA-256 equal |
| Five-step coverage and no records when disabled | PASS |
| Pyflakes | PASS |
| Rig suite | PASS: all 92 tests; pyflakes clean |
| Physical diagnostic coverage | PASS: 180 row triples, five steps on all 18 layers for both requests |
| Isolate a faulty mapping | FAIL: all 180 returned/local F16 hashes differ, with no cross-row equality |

Deployment: `/mnt/storage/s42-fast-path-M2-diagnostic-20260920-a81c09`.
All phone operations use ADB 5037. The runner retains the existing rig lock,
server port/PID/argv validation, watchdog and cleanup. Each scope has
`MemoryMax=infinity` and `MemorySwapMax=0`; ready/finish memory events are kept.

Commands from the controller:

```bash
bash research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/diagnostic/DEPLOY.sh
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M2-diagnostic-20260920-a81c09/BUILD.sh'
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M2-diagnostic-20260920-a81c09/MATERIALIZE_TRANSPORT.sh'
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M2-diagnostic-20260920-a81c09/RUN_DIAGNOSTIC.sh'
```

The diagnostic arm uses N=2, prompts (256,257), 64 outputs each, ubatch 1024,
batch 2048, eight unpinned threads, 16 GPU layers, keep-cache + populate and the
existing phone kernel/workers. The submitted command and source snapshots are
saved in the arm's `.command.json`. Measured host energy is RAPL package plus
NVML board; assumed phone power is separately reported at 4.5 W active and
0.875 W idle. Diagnostic timings include shadow execution and are not acceptance
performance measurements.

The first disagreement is call 1, layer 0, step 1. Both rows disagree:

| Ubatch row | Request index | Native slot | Member index | Payload row | Position | Decoded / applied |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 0 | 0 | 258 | 3 / 3 |
| 1 | 1 | 1 | 1 | 1 | 259 | 3 / 3 |

Row 0 local F16 SHA-256 is
`2b9bb3cf12e123930b1a64fa594ee0ae520cb909fcc43b1be36549dbbe63153e`;
returned F16 SHA-256 is
`05b72793081cfb881c9a0ce0acfb5718ba6ebb4f7ecbd3f39ca14e1df5c843f5`.
All hashes, mappings and cross-row comparisons are in
[CHECK_STEP1.json](../physical/diagnostic/CHECK_STEP1.json).
This hash test does not isolate a single bad row or a swap. Both 64-token outputs
match the corresponding historical host prefix, but this is not a fresh matched
pair because the native build changed. This run also has acknowledgements 3/3
and aligned slot/member order, unlike the original 4/3 reversed-order failure.
Continue to the requested ordered matrix; no mapping correction is applied yet.

| Single diagnostic arm | Request host J, measured | Decode host W, measured | Decode ms/token, request 0 / 1 | Phone W, assumed |
| --- | ---: | ---: | --- | --- |
| N=2, five shadow steps | 2866.335064577404 | 69.81190634552053 | 538.945203125 / 538.94071875 | 4.5 active; 0.875 idle |

| memory.peak bytes | events.max at ready | at finish | delta |
| ---: | ---: | ---: | ---: |
| 29844824064 | 0 | 0 | 0 |

[STEP1_DIAGNOSIS.json](../physical/diagnostic/STEP1_DIAGNOSIS.json) contains the
status and historical comparisons. Cleanup at 20:41:42 UTC confirmed the scope
inactive, server gone, GPU idle, lock free, and OP15 restored on ADB 5037.
[Cleanup record](../physical/diagnostic/STEP1_CLEANUP.json).

## Step 2: ordered prompt matrix (PASS)

All eight pairs passed on 2026-09-20 at 23:08:37 UTC, with 64/64 exact outputs
in every slot before any mapping change. [Per-slot exactness, findings and
measured-arm tables](MATRIX.md).

Eight cases use prompts (256,256), (257,257), (256,257), (257,256), each with
normal [0,1] and reversed [1,0] submission. A request waits for its predecessor's
observed slot allocation and prefill-start log before submission. The successor
is submitted while the first prefill is in flight; this uses no HTTP polling or
wait for a decoded token. The first free slot on a fresh
server is the highest index (the server's LRU tie rule), so N=2 normal order maps
requests to [1,0] and reversed order maps them to [0,1]. Control member order
remains [request 0, request 1]. This reproduces the reversed member/ubatch order
without relying on a race between simultaneous submissions.

The request contract's `cohort_submission_order` defaults to empty (concurrent
issuance), validates a bounded permutation, and enters the request fixture hash
and the gate's runtime environment digest when nondefault. Both arms use the
same native hashes, request tokens, order and KV geometry. Each has a fresh
uncapped scope; no source or phone-binary changes occur between paired arms.
The phone arm retains the five-step diagnostic. Every request generates 64 tokens.

The 94 local tests pass (one server-only skip), and pyflakes is clean.
The same 94-test suite runs on the rig before each arm. The matrix checks every slot, expected assignment and full
completion, then records exact-token failures without dropping cases. A fixture
or watchdog failure stops the driver. Each physical arm is preceded by the unit
suite, pyflakes and verification of the materialized native identity.

```bash
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M2-diagnostic-20260920-a81c09/RUN_MATRIX.sh'
```

Per-arm exact commands and source snapshots are in
`physical/diagnostic/step2/<case>/<arm>.command.json`; paired checks are
`physical/diagnostic/step2/<case>/CHECK_PAIR.json`. The runner and checker sources
are preserved in [STEP2_SOURCE.json](software/STEP2_SOURCE.json).

The first two pairs are preserved as fixture pilots in
[step2-pilot](../physical/diagnostic/step2-pilot/PILOT_SUMMARY.json). Both orders
of (256,256) matched all 64 tokens in both slots. That runner waited for the
first decode progress report before submitting its successor; the normal phone
pair acknowledged indices 6/3, wider than the original 4/3 gap. A second fixture pilot polled the server's `/slots` allocation response. Its
normal (256,256) pair also matched all 64 tokens in both slots, but acknowledged
indices 5/3. Its reversed host arm completed; no reversed phone arm was launched.
These records and sources remain in `physical/diagnostic/step2-http-pilot` and
`software/STEP2_HTTP_PILOT_SOURCE.json`. Cleanup at 21:33 UTC confirmed the
owned servers gone, GPU idle, the rig lock free and OP15 restored on ADB 5037.

The final matrix uses existing native slot-allocation and matching prefill-start
log lines. The request fixture digest includes
`submission_boundary=native-prefill-start-log`; each arm records those lines,
their timestamps and prompt counts. Tests check parsing against the native line
format and verify that submission does not wait for decoded-token progress.
This changes the diagnostic request fixture, with no mapping or native-binary
changes between the pilots and final matrix.
The held driver let the last pilot complete normally, then cleanup at
2026-09-20 21:12:00 UTC verified no owned servers, GPU idle, the lock free and
OP15 restored on ADB 5037. No worker was killed. All pilots and their original
sources remain reviewable; neither is substituted for a final matrix case.
