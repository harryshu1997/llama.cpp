# Remote-resident FFN numerical diagnostic

Follow-on to the preserved relocation v4 PARTIAL result, not a replacement for
it. Compare the same three short Gemma requests (42/46/50; 11/14/41 outputs) with
32 pre-sampling token log probabilities. Model, native binaries, FFN shard,
desktop placement, seeds, prompts, batching, graph mode and request ordering are
unchanged. No trace, native rebuild, new shard, recovery injection or reboot.

The existing `/completion` API supplies the observations. An opt-in payload
field and gate argument request them; default HTTP bodies remain unchanged.
Canonical phone preparation, ticket admission, leases, generation checks,
physical proofs and cleanup still execute. These runs are numerical diagnostics,
not new energy or latency qualification. Compare distributions only while the
preceding token IDs are identical, including the first divergent prediction.
After divergence, different contexts must not be interpreted as numeric error.

The existing server disables backend sampling to read pre-sampling logits.
The original run already recorded `backend_sampling=false`, so this does not
change that setting. Probabilities are normalized logits, not raw logits or
per-layer errors. Top-32 observations cannot establish full-vocabulary KL or
attribute a difference specifically to F16 rounding.

Source before-images: `before.tar.gz`, SHA-256
`63391f75d6a67cb19da3cf7559ee31fc642f9e30b34551abf60eb9582b0e9fed`.
The two before-images for deployed code match the preserved v4 source archive.
Deploy that archive into a fresh tree, then overlay only those two source files.

Remote directory:
`/mnt/storage/s42-remote-resident-numeric-20260914-v1-sdDNJN/`.
The launch wrapper acquires the existing exclusive device lock and checks the
same boot, kernel notes/BTF, idle GPU/phone and USB link. It never kills other
processes, changes GDM/drivers, flashes a partition or resets USB as recovery.

## Measured result

The physical run completed all three requests on each arm. Each output is
identical to the corresponding original v4 arm, reproducing its exact failure:
requests 42 and 46 match across arms; request 50 first diverges at output token
37. Both paths had the same 271 prompt tokens and 36 generated tokens at that
point, a 307-token context. No sampled-token difference precedes it.

| Candidate at output token 37 | Full desktop parent | Reduced parent + OP15 |
| --- | ---: | ---: |
| ` communication` (7283) | 37.4209% | 34.9117% |
| ` hardware` (14628) | 35.0206% | 35.5523% |
| Greedy choice | communication | hardware |

The communication-versus-hardware log-odds margin changes from +0.0662937164
to -0.0181826353, a -0.0844763517 shift. These are probabilities from the
pre-sampling logits, not sampling frequencies. The ranking reversal explains
the first different token under temperature 0. Later words have different
prefixes and are excluded from the numeric comparison.

Differences already exist on the first prediction, after prefill, in every
request. For request 50, the initial winning token `Co` has probability
80.4477% locally versus 79.9529% through the phone path. Across all 62
same-prefix predictions, the largest absolute log-probability difference on
shared top-32 candidates is 0.738658, including low-ranked candidates. This is
not a relative tensor error, full-vocabulary distance or accuracy tolerance.
Do not describe all differences as machine-epsilon noise or infer quality from
the 61 matching choices out of those 62 predictions.

This establishes reproducible numerical differences along the phone execution
path, rather than merely observing different prose after generation diverges.
There is no observed ticket, artifact, generation or transport failure. It does
not yet locate the numeric difference: F16 wire casts, NPU arithmetic/activation
implementation, reduction order and accumulated KV differences remain possible
contributors. The next diagnostic should feed identical captured FFN inputs
through local and phone computations and compare per-layer outputs, separating
wire rounding from computation. No native instrumentation or precision change
was made in this step, and none of these explanations is asserted as proven.

## Physical proof and limits

- One HTP0 load; generation 1 throughout; fresh request ticket/lease tokens.
- Phone calls: 96 / 120 / 336, totaling 552 request-attributed calls. The native
  terminal has 584 including warm-up, status 0 and zero reset recoveries.
- Worker, router, resident-manager and other native hashes equal the v4 hashes.
  Request and session operator plans, artifact, geometry, layer mask, endpoint
  and physical generation pass exact proof checks.
- Memory gate B PASS: 2,831,056,896 bytes unmapped, with zero omitted-page VMA
  overlap. Model mappings are 13,811,417,088 versus 10,980,360,192 bytes.
  GPU VRAM stays 12,654,215,168 bytes on both arms.
- Canonical phone preparation took 21.372611 s; native authorization to READY
  took 10.847870 s. Preparation energy is recorded in the raw receipts, not
  silently omitted or used for a diagnostic-run savings claim.
- Normal final USB restoration and read-only same-boot postflight PASS.
  No between-request reload, fallback, reset, in-flight owner-loss injection,
  unrelated process signal or trace.

Exact-output gate A is still FAIL, and overall relocation status is still
PARTIAL. The earlier result SHA-256 is unchanged:
`9ca16f45a21e4015831f01e772211313e1eacd48b81c2862f4675f05846e25c5`.
Neither harmless rounding, quality acceptance, added KV capacity nor savings
has been established.

## Changes and validation

Only these existing code files changed:

- `adapters/http_backend.py`: bounded opt-in diagnostic probability count;
  ordinary completion bodies are unchanged.
- `campaigns/burstgpt/remote_resident_gate.py`: explicit diagnostic argument
  applied equally to both measured arms; result labelled diagnostic.
- `tests/test_remote_resident_gate.py`: HTTP body, bounds and gate-forwarding
  regressions. Existing lease, identity and terminal checks remain in force.

Report-local launch/analysis scripts supply configuration, measurements and
persistence only. No scheduling, attachment or residency policy changed.

Validation:

- 60 focused owner/reuse/gate/HTTP-adapter tests PASS, 13.526 s.
- Both replay tests PASS, 79.154 s; goldens unchanged.
- Three probability-analyzer tests PASS, 0.003 s. They check identical streams,
  exclusion after divergent prefixes, and fail-closed missing observations.
- Repeating the analysis on the local artifact mirror produces byte-identical
  JSON to the remote analysis.
- Parse/compile and whitespace checks PASS. No broad suite or native rebuild.

Replay hashes:

- Session COW v3: `sha256:5d52e8673fc60f974ca167964b3ad579e59abea28feb3d66357cf39c1ca7caf4`.
- Sparse v8: `sha256:241924463ac27ead0c2d7bcb5da214e85097049c91d5800848e0b29effd2b917`.

## Reproduction and artifacts

The local mirror is `physical/`. Remote commands, source archive, raw streams,
receipts and phase/terminal proofs are in the remote directory above. Every
analysis checks model, binary/library hashes, shard, masks, launch contracts
and request identities against the original v4 gate before comparing scores.

Run the analyzer into a new output path:

```sh
python3 analyze_probabilities.py physical/gate-run-v1 \
  --previous ../20260914-remote-resident-phone/physical/gate-run-v4/REMOTE_RESIDENT_GATE.json \
  --output probability-comparison-new.json
```

Artifacts and SHA-256:

- `physical/PROBABILITY_COMPARISON.json`:
  `764b7a9955eb6d0820a72f6fc5f724f9b5e67351642e33b3c7219231ec1b7274`.
- `physical/gate-run-v1/REMOTE_RESIDENT_GATE.json`:
  `8c365c8ae3d4d1b48368515ed8b1c918c1342b182d0484f333a6dd7c9d8a6ad3`.
- `physical/source-numeric-v1.tar.gz`:
  `f035b68c951a9706800af721c2d54de5b7a42a326f8421cd695170b5d01d1cf6`.

Nothing committed or pushed. Native binaries, shard files and old failed
artifacts were not changed.
