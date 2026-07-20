# S14 CP1: static resident mixed runtime

Status: 2026-07-17 LIVE-FRIENDLY OFFLINE + REAL DEVICE. Preview mechanics only;
CP1 gate incomplete, no energy, no committed device claim.

    verdict        STATIC_MIXED_MECHANICS_PASS_RELIEF_INSUFFICIENT
    bindings       island_catalog_hash sha256:3cf13792...  (CP0-c freeze)
                   mix_v1_output_sha256 sha256:567d4af1...  (real mix-v1, 177 rows)
    result         cp1_result.json  (byte-deterministic; result_digest inside)

## 1. What CP1 built

CP1 admits the real `mix-v1` trace (BurstGPT generation + RAGPulse RAG, 177
requests over ~897 s) and compares four controls through the S12-V2 reducer
(`two_level_vq.Simulator`, REUSED UNCHANGED; its 91 tests stay green):

    C0   server_only        optimized server-only, no phone
    C1   fixed_static_phone fixed all-phone placement, static READY residency
    C2   static_two_phone   mixed virtual queue, static READY residency
    C3p  dynamic_two_level  dynamic slow loop (preview; CP3 formalizes it)

The new pieces are `cp1_adapter.py` (the PLAN section 8 "measured-profile
adapter": frozen island catalog + mix-v1 -> reducer inputs, with honest MEASURED
labels instead of the reducer's synthetic ones) and `cp1_runtime.py` (runs all
four controls, computes throughput / relief / overlap / latency, and emits a
digest-bound result). The reducer's Simulator is driven directly so measured
LOWER_BOUND data is never relabelled as synthetic.

`cp1_runtime.py` now supports `--mode live` with bounded command execution:

- repeated `--live-device id,host,port,serial` or `S14_CP1_LIVE_DEVICES` for
  device descriptors;
- explicit fleet binary selection (CLI/env/build path resolution);
- strict JSON parse and schema checks on the final fleet output line;
- optional injection of live device metrics into per-control rows.
- device run path verified with both OP12+OP15 sessions (`REAL_FLEET_FFN_PASS` on C2/C3p)

This is not the planned CP1 gate yet. It is an offline single-island replay:
RAG rows are mapped to the same Gemma head island as generation, the BGE island
is excluded because it has no latency row, C1 is now a native fixed-all-phone policy
in the S12 reducer. The result validates the adapter and reducer accounting, with
live device coupling on the measured Gemma head island only.

## 2. Honest scope (what this is NOT)

- Every catalog row is dispatch-INELIGIBLE (LOWER_BOUND latency, no same-run
  placement certificate). This is MECHANICS, not a certified performance claim.
- Only ONE island has a measured phone latency (`gemma_head_0_2`, a 2-layer head
  on OP15). So all 177 requests share it; `bge_encoder_0_12` is EXCLUDED (no
  measured latency). This is a single-phone-island run, not two-island-diverse.
- A 2-layer head means `server_tail (156736 us) ~= server_full (156951 us)`: the
  phone offloads only 2 of 48 layers, so it can relieve at most 215 us of server
  compute per request BY CONSTRUCTION.
- `deadline_us`/`priority_class` are SYNTHETIC scheduling parameters (the real
  trace carries none); arrival-compression is a SYNTHETIC load stress. Both are
  labelled as such and never presented as real-trace claims.
- CP1's control C1 (fixed all-phone) is now native in S12.

## 3. Result

### Real median mix-v1 (177 requests, ~897 s, server ~3% utilised)

| policy | useful | phone | server_full | server_tail | p50 latency (us) | phone/server overlap (us) | HBM relief |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 server_only | 177 | 0 | 177 | 0 | 156951 | 0 | 0 |
| C1 fixed_static_phone | 0 | 1 | 0 | 0 | 0 | 0 | 901775360 |
| C2 static_two_phone | 177 | 170 | 7 | 170 | 311698 | 1084734 | 0 |
| C3p dynamic_two_level | 177 | 170 | 7 | 170 | 311698 | 1084734 | 0 |

- Throughput: C1 is not viable in this synthetic median because fixed-static
  routing cannot complete work at current timing. C2/C0 = 1.000x is also trivial
  here because server is only ~3% utilized.
- Latency: C2 DOUBLES p50 (156951 -> 311698 us). The phone route is serial
  (phone head 154962 us THEN server tail 156736 us), exactly the S11-E0 penalty.
- HBM relief: 0. The VQ let 7 requests fall through to `server_full` (all 48
  layers on the A6000), so the A6000 must keep layers [0,2) resident -- no
  exclusive offload, no HBM freed. Opportunistic routing does not guarantee the
  exclusive residency that HBM relief requires.
- Server-compute relief: 0.1 percent (215 us x 170 phone requests = 36.5 ms over
  a 897 s run). Negligible by construction (2-layer head).

### Synthetic arrival-compression sweep (mechanism under contention)

| den (compress) | window (us) | C2 useful ratio | C2 phone | C2 server_full | C2 rejected |
|---:|---:|---:|---:|---:|---:|
| 1 | 897000000 | 1.000 | 170 | 7 | 0 |
| 10 | 89700000 | 0.989 | 163 | 14 | 0 |
| 50 | 17940000 | **0.667** | 68 | 68 | 0 |
| 100 | 8970000 | 1.000 | 39 | 39 | 55 |
| 200 | 4485000 | 1.000 | 25 | 25 | 83 |
| 500 | 1794000 | 1.000 | 16 | 16 | 101 |

The load-bearing point is `den=50`: with no queue rejections, C2 has four useful
completions versus six for C0, a **0.667x useful-completion ratio**. Both policies
also produce tardy and timed-out terminals, so this is not a clean throughput
rate; it is a bounded useful-work retention signal showing that routing to the
slow 2-layer-head phone can no longer keep up. (At `den>=100` the ratio returns to 1.000 only
because BOTH C0 and C2 saturate and reject the same 55-101 requests; that 1.000
is saturation-equal, not a win, so those points are excluded from the contention
verdict.)

## 4. Verdict and why

`STATIC_MIXED_MECHANICS_PASS_RELIEF_INSUFFICIENT` is a preview label only;
`cp1_gate_complete=false` because the planned two-service/two-phone live gate
was not run.

- Replay mechanics PASS: exact routing and terminal conservation (all 177
  terminal outcomes, including timeouts, are accounted under every policy and
  load point), zero background transfer (static residency), and modeled
  phone/server interval overlap.
- Relief INSUFFICIENT: HBM relief 0 (opportunistic routing breaks exclusivity);
  server-compute relief 0.1 percent (2-layer head). Not a material mechanism.
- Contention FAIL: C2 drops to 0.667x C0 at den=50; latency doubles at every
  load. The mixed VQ over a 2-layer head does not help and hurts under load.

This is an informative negative-leaning result, not a hollow pass. It quantifies
exactly why the current frozen catalog cannot yet demonstrate a useful mixed
mechanism: the only measured island offloads 2 of 48 layers, so it costs ~155 ms
of serial phone time to save ~0.2 ms of server compute.

## 5. What unblocks a real CP1 pass

The single highest-value measurement (already named as CP0-c gap 1 in CATALOG.md
section 5): a COHERENT deeper-head run -- a larger `[0,k)` island where
`server_full - server_tail` is large (real compute to offload) -- emitting a
per-request latency AND a same-run scheduled-buffer `PLACEMENTCERT` on one
binary, so the island becomes dispatch-ELIGIBLE with a real compute-relief gap.
Only then can the mixed VQ retain throughput WHILE materially relieving the
server. Until then the mechanism is proven to work but has nothing worth
offloading.

## 6. Reproduce

    cd research_dev/spikes/s14_mixed_streaming_scheduler
    /usr/bin/python3 cp1_runtime.py               # byte-deterministic
    /usr/bin/python3 cp1_runtime.py --mode live \
      --live-device op12,127.0.0.1,19012,5ae7a43d \
      --live-device op15,127.0.0.1,19015,3C15AU002CL00000 \
      --output cp1_result_live.json
    /usr/bin/python3 tests/test_cp1.py            # 19/19

Files: `cp1_adapter.py`, `cp1_runtime.py`, `cp1_result.json`, `cp1_result_live.json`,
`fixtures/mix_v1.trace.jsonl` (staged real mix-v1, digest = frozen output_sha256),
`tests/test_cp1.py`. Reducer: `../s12_trace_vq/two_level_vq.py` (unchanged).
