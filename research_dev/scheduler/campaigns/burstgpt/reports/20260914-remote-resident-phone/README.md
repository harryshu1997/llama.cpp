# Real-phone FFN weight relocation

This is the bounded follow-on to the candidate-kernel cancellation prerequisite.
No trace, native rebuild, shard regeneration, flash, reboot, GDM change or
unrelated process control is part of this gate. No commit, push or PR.

Result: actual OP15 FFN weight relocation and reuse are demonstrated, but the
overall gate is PARTIAL. Memory gate B passes; exact greedy-output gate A fails
on the third request. The failure is preserved, not waived.

## Inputs and scope

- Gemma F16 parent, the existing 23-GPU-layer placement and CUDA graphs enabled.
- Existing FFN shard for CPU layers 0-7, all FFN columns, 2,831,155,200 tensor bytes.
- Three existing short requests (indices 42, 46, 50; outputs 11, 14, 41), identical
  prompts, seed and decoding settings on the full and reduced parents.
- Canonical offline preparation, generation-bound READY ownership, scheduler
  calibration admission, actual request leases and memory reservations, physical
  execution proofs and normal terminal cleanup.
- The reduced route is CALIBRATION_PENDING. No qualification or energy result is
  inherited from the full desktop parent. USB measurements qualify transport,
  not FFN correctness or savings.

The gate does not attempt in-flight HTP cancellation. The prerequisite proved
idle USB cancellation only. Additional KV capacity and a savings comparison are
also outside this bounded execution/memory gate.

## Repairs on the existing path

Remote owners now participate in phone resources and workspace admission even
though the unmodified operator rows describe the desktop parent. Owner binding
reads exact per-session observations from the current snapshot, not stale
catalog residency annotations. Commit and ticket validation carry the READY
layout's audit generation alongside the owners' individual physical generations.

Phone reuse verifies the actually loaded shard path, file/index/parent hashes,
geometry, layer mask, columns, operator plan, generation and prefill capacity.
It reuses a compatible owner rather than restarting its endpoint. Execution
registers the ticket's proof generation and phone activity interval. The same
owner-to-shard conversion is used by reuse and terminal proof validation.

The existing gate now invokes the canonical preloader and physical ticket
executor. Its error-free-output check also requires the greedy token agreement
it already records. A measurement-only rig startup option avoids launching an
unrelated legacy Llama endpoint for this single-model test.

## Fresh transport qualification

Attempt v3a: host-to-phone passed. The following phone-to-host launch failed
before USB enumeration. Its launch log says `Device or resource busy`, and the
kernel log showed Android g1 racing g2 for the UDC. The old qualification script
did not disable Android's automatic gadget rebinding. No reboot occurred.

Attempt v3b reuses the existing canonical `functionfs_transport_session.sh` and
`restore_android_usb.sh` in a fresh directory. All nine cases passed: 7,680,
10,240 and 3,932,160 byte payloads, each in H2D, D2H and duplex directions.
No old receipts were relabelled. The same boot persisted throughout:
`fa74f551-fb1b-465e-b5a2-053b9869b1bb`.

Fresh transport identity:
`sha256:ced4229daf7607dbb5aae064a8016638efc1a2639a056c5ba9671b012c9af901`.
Candidate image:
`sha256:f13c7c033ce74f32ea4aa6349cb531704407708f5fb35756a6c361ffb48322a3`.

## Software and preflight

The focused owner/launch/gate/controller set passed 66 tests, including both
replays, byte-identical on repeated evaluation. The seven owner regressions
were rerun after the final gate configuration change and passed. The offline
preload suite also passed in the earlier 54-test focused run. No broad suite.

Replay goldens, unchanged:

- Session COW v3: `sha256:5d52e8673fc60f974ca167964b3ad579e59abea28feb3d66357cf39c1ca7caf4`.
- Sparse v8: `sha256:241924463ac27ead0c2d7bcb5da214e85097049c91d5800848e0b29effd2b917`.

Physical preflight v1 failed because the isolated deployment lacked `gguf-py`.
After deploying that existing package, preflight v2 passed all file, device and
binary identity checks. Both attempts and their source archives are preserved.

Remote artifacts:

- Failed transport v3a: `/mnt/storage/s42-remote-resident-phone-20260914-v3-Ps8oat/`.
- Fresh qualification and gate: `/mnt/storage/s42-remote-resident-phone-20260914-v3b-r9Rdsh/`.
- Gate source archive: `source-gate-run-v1.tar.gz`, SHA-256
  `ad2e000d3ead0e0f8cc3d4d157fd24208171727ab19f390251dc7fe609c1892a`.

## Physical result

Gate run v1 completed its three full-parent requests, then stopped before any
phone load: fresh measured USB link IDs had no corresponding catalog resource
entries. The missing six entries are now declared with unit capacity, preserving
all shared USB/FunctionFS/HTP resources. `CATALOG_REPAIR_REPLAY.json` records a
successful canonical offline plan using the failed run's actual initial snapshot.
Old catalog and failed run remain unchanged; the corrected catalog is v2.

The gate now persists each completed arm immediately, including its memory
measurements, so a subsequent failure cannot discard it. Fifteen focused
owner/gate tests pass after this repair. Gate run v2 uses source archive SHA-256
`dcd4c9318e4b99b4a6d6a8c7cdae215f637775db78c8ceba5f71445b976eabc1`.

Gate run v2 loaded the real FFN shard and published generation 1 READY. Its
physical transition receipt spans 17.630 seconds. Reduced-parent loading then
failed because its otherwise-local operator rows did not contribute the remote
owner's USB links. The adapter consequently tried an implicit bridge and rejected
the absent `bridge_allocator`. Reading phase events after closing the phone also
masked this error. The full-arm measurements and native failed attempt remain.

The compiler now includes ready measured links to the exact remote owner in its
resource/transport binding and rejects missing transport qualification before
launch. `TRANSPORT_BINDING_REPLAY.json` reproduces the old saved-ticket error,
then proves the corrected transport reuses the actual loaded router, including
the same qualification identity and 3,932,160-byte prefill capacity. Failure
reporting now records the runtime decision/controller state before cleanup and
reads native phase events while the session is still active.

Twenty-five focused tests pass after the transport repair. Both replay tests
also pass (149.370 seconds), with the goldens above unchanged. The old
decode-only synthetic admission fixture now explicitly supplies synthetic
prefill transport capacity; real-device qualification was not altered.

Gate run v3 source archive SHA-256:
`5dbf5219db1c99ecac3dcf1c7b7fc2755d04d9b8c0ba376a1faef1c832cfae6e`.

Gate run v3 reached the real phone owner but failed reduced-parent launch because
its experimental profile omitted `ffn_assistance_phase=decode`, required by the
existing launch contract. Failure persistence now retained the original exception
and normal cleanup evidence. The gate profile now supplies that declaration;
remote-resident prefill remains enabled by the existing runtime contract.
A focused regression builds and validates the full generated launch command.
No native binary, shard, model or acceptance threshold changed.

Gate run v4 completed all three full-parent and all three reduced-parent
requests. The final 29 owner/reuse/launch/gate tests pass (10.867 seconds).
These test sets overlap; their counts must not be added as unique tests.

### Memory and physical ownership

| Measurement | Full parent | Reduced parent |
| --- | ---: | ---: |
| Model-file mappings, bytes | 13,811,417,088 | 10,980,360,192 |
| Model-file RSS, bytes | 13,811,417,088 | 10,980,327,424 |
| Process RSS, bytes | 15,255,138,304 | 12,504,985,600 |
| Process GPU VRAM, bytes | 12,654,215,168 | 12,654,215,168 |

Twenty-four FFN tensors for layers 0-7, all 15,360 columns, are remote-owned:
2,831,155,200 weight bytes. Native omission proof reports 2,831,056,896 complete
pages unmapped, with zero VMA overlap for those pages and `warmup=validated`.
Boundary pages explain the difference from tensor bytes. This proves a 2.831 GB
host model-mapping reduction, not GPU VRAM reclamation, extra usable KV capacity,
or energy savings. No larger-context request has been run.

The worker opened the indexed FFN file, not the full GGUF:
`/data/local/tmp/s42-ffn-shards-20260904-v2/gemma24/HTP0.ffn.gguf`.
The file is 2,831,157,472 bytes including metadata. Stored and executed masks are
both `00000000000000ff`; stored and active widths are both 15,360.
HTP0 loaded once and remained generation 1 for all three requests. Each request
has a distinct ticket and fresh lease tokens. Lease role IDs may repeat; the
authorization tokens do not. Artifact, shard, geometry, resident operator plan,
request operator plan and positive physical generation are checked by the
canonical execution/proof paths and the saved summary audit.

### Requests and exact-output failure

| Request index | Output tokens | Phone FFN calls | Exact greedy agreement |
| --- | ---: | ---: | --- |
| 42 | 11 | 96 | Identical |
| 46 | 14 | 120 | Identical |
| 50 | 41 | 336 | First difference at token 37 |

All six executions reached terminal output lengths. The three reduced executions
record 552 calls, including prefill; the native terminal totals 584 including
32 warm-up calls. Native terminal status is 0 with zero reset recoveries.
There was no reload between requests, stale generation, fallback or internal
endpoint restart. Normal final shutdown restored Android USB at 5,000 Mbps.
The separate read-only postflight confirmed the same temporary candidate boot
and idle GPU. `gate-run-v4-postflight-partial/` preserves that check.

For request 50, the shared first 36 tokens end in `interplay between`. The full
parent continues `communication overhead and local processing`; the reduced
parent continues `hardware capabilities and the physical`. The wire path rounds
activations/results through F16, whereas the local reference does not use that
phone path. Rounding is a hypothesis, not a diagnosis: these artifacts contain
neither same-prefix logits nor per-layer numeric error. Exact-token gate A
remains FAIL, and overall status remains PARTIAL. Fluency does not establish
numerical correctness. The next bounded diagnostic should compare the same
prefix through both paths and inspect the first logit or intermediate divergence;
do not rerun a trace or merely relax the comparison.

The recorded full request durations are 6.994, 8.392 and 20.592 seconds. Reduced
durations are 13.165, 9.458 and 21.159 seconds, but the first reduced duration
includes its desktop load transition while the full arm records loading
separately. These are not a matched latency or energy comparison.

### Preparation and accounting

| Physical interval | Duration |
| --- | ---: |
| Phone storage read | 9.803470 s |
| HTP initialization | 0.096590 s |
| Weight upload to HTP | 0.755491 s |
| Native LOAD_AUTHORIZED to READY | 10.726816 s |
| Complete canonical phone preparation | 19.873230 s |

Native event timestamps, in the phone's monotonic microseconds:
LOAD_AUTHORIZED 6,735,957,361; read 6,735,969,613 to 6,745,773,083;
HTP initialization 6,745,773,104 to 6,745,869,694; upload 6,745,869,709 to
6,746,625,200; VERIFIED 6,746,684,137; READY 6,746,684,177. Their wall-clock
timestamps and exact identities are also in `physical/GATE_V4_SUMMARY.json`.
Do not subtract the phone monotonic clock from the host monotonic clock.
Canonical preparation additionally includes transport, observations and setup;
this gate does not attribute that difference to a single phase.

The preparation transition receipt spans 19.650504 s and records 126.038421 J
CPU package, 155.290587 J GPU board and 88.426175 J assumed phone energy:
369.755183 J fleet total at the 4.5 W phone assumption. It is explicitly
diagnostic (`ENERGY_LEDGER_UNAVAILABLE`), not qualification evidence. Its
receipt boundary differs slightly from the outer 19.873230 s preparation
interval, whose separately sampled server-only energy is 284.140570 J. These
values must not be combined into a new total. CPU/GPU energy is measured;
phone power is assumed, not measured.

### Immutable artifacts and limits

Local evidence is under `physical/`; the remote root is
`/mnt/storage/s42-remote-resident-phone-20260914-v3b-r9Rdsh/` on the desktop.
All earlier failed attempts remain alongside `gate-run-v4/`.

- Result: `gate-run-v4/REMOTE_RESIDENT_GATE.json`, SHA-256
  `9ca16f45a21e4015831f01e772211313e1eacd48b81c2862f4675f05846e25c5`.
- Summary: `GATE_V4_SUMMARY.json`, SHA-256
  `706204f1fbbbc76d7963f0896e04e70b142c5ee7622b6e127a4e61433c32ff6c`.
- Exact deployed scheduler source: `source-gate-run-v4.tar.gz`, SHA-256
  `0dc7afffcaa6a7161e264f1bc629a25825532970b78c8ac3a8af48ab1b505f22`.
- Existing desktop server: SHA-256
  `00220ffd27aa167752de240d82cdf1062104ec087ce62c388e2e193b8b1d974a`.
- Existing ARM64 worker: SHA-256
  `43adcb755f8ff10ba30073ae55a31c7af95ee4f14071cfd2a5ba961b8317da19`.
- FFN shard: SHA-256
  `ef2efcce04b08ee78b7a232d7d5f33f62ccebbee9add93669c5100e97ae041a0`.
- FFN index: SHA-256
  `5d0fc13e8be1875269a6c8ef2eeaaded8c968ecf262703de529bdbc71ba13681`.
- Parent artifact: SHA-256
  `ed76f2183d2d1d65091986033023e6c78d27f6276c1b0c5826cc92acf73538cf`.

The summary includes library, router and resident-manager hashes, exact request
proofs, lease tokens, phase events and preparation receipts. Local result and
summary hashes were verified equal to their remote originals. `CHANGES.json`
lists exact scheduler files and preserved before-image hashes; `TESTS.json`
lists the focused checks. No frozen replay golden was updated.

Remaining blockers: the numerical divergence is undiagnosed; useful added KV
capacity and matched savings are untested; in-flight HTP owner-loss recovery is
still outside the qualified cancellation envelope. No long trace, recovery
fault injection, native rebuild, reboot, flash, commit or push was performed.
