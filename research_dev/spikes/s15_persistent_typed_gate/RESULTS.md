# S15 typed persistent physical results

Verdict: `TYPED_PERSISTENT_OP15_B32_PHYSICAL_PASS_ENERGY_UNKNOWN`.

## Result

The full typed path ran on real OP15 and A6000 hardware. It reused one phone
worker and one server-tail process across two B32 exchanges.

| Field | Exchange 1 | Exchange 2 |
|---|---:|---:|
| Session end | DETACH | STOP |
| Child elapsed | 3,208,313 us | 2,792,293 us |
| Route wall | 3,063,879 us | 2,722,829 us |
| Phone steps, session | 384 | 384 |
| Phone steps, cumulative | 384 | 768 |
| Exact requests | 32/32 | 32/32 |
| Boundary certificates | 32/32 | 32/32 |
| Host compute nodes | 9,288 CUDA0 | 9,288 CUDA0 |

Persistent identities:

- A6000 tail PID: `3815676` in both exchanges.
- OP15 worker PID: `21616` in both exchanges.
- OP15 worker nonce is unchanged; session ids are exactly `1,2`.
- DETACH reports `reset_applied=true`; STOP reports false and both processes
  terminate with status zero.

Placement:

- Phone layers `[0,8)` use HTP0. CPU appears only for the declared GET_ROWS
  seam. Missing-buffer compute nodes are zero.
- Server layers `[8,48)` use CUDA0 exclusively. Missing-buffer compute nodes
  are zero.
- HMX temperature rises from 32.9 C to 37.2 C, below the frozen gate.

## Verification

- Persistent live-launcher adversarial suite: 19/19.
- Physical mux tests: 6/6.
- Independent report and mutation tests: 5/5.
- Existing input-seam tests: 9/9.
- `validate_report.py` reopens bridge replies, artifact hashes, tokens, both
  placement certificates, raw mux streams, process metadata, and the run
  manifest.
- `run_manifest.json` binds 19 local artifacts, 10 deployed runtime files, the
  exact model and shard digests, OP15 boot id, selected GPU UUID, and report.

## Scope

This proves persistent physical execution mechanics through the typed runtime.
It does not yet prove mixed-workload relief or energy savings. Payload,
priority, and SLO sidecars are synthetic. Phone, USB, server-wall, and total
system energy remain unknown. The next claim-bearing checkpoint is repeated
matched BGE+Gemma control/treatment acquisition on one selected A6000.
