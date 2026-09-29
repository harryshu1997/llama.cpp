# Decode-only FFN relocation with per-layer host KV (Qwen3-14B, RTX 4060 Ti + OP15)

## Outcome

Three measured results on Qwen3-14B (RTX 4060 Ti CPU parent + OP15, three phone sessions owning the FFN of
layers 0-17), 9,737-token prompt, one run per arm:

| Claim | Evidence |
| --- | --- |
| Prefill unchanged | Local in every arm, 287-316 s uncapped; no phone traffic before the first token |
| Energy saving | 50 % split: request host energy -16 % (decode -20 %, same host power, shorter time). 100 % split: -34 % (decode -44 %, host power 127 W -> 71 W, same time). With the phone assumed at 4.5 W: -14 % / -31 %. Over a 2,048-token generation the 100 % saving grows to -38 % |
| Longer context under a fixed host budget | 4.44 GB (50 %) or 9.63 GB (100 %) of FFN pages released at the first token, page-exact, in 86-316 ms; page-granular KV backing lets the generation grow into them, and the direct probe (`kv-headroom-v1`) shows a 3 GiB KV fill under an 18 GiB cap landing in the freed room with the remaining weights intact, where the control evicts 3 GiB of weights. Memory-equivalent: 9.63 GB is the CPU-tier KV of ~63k tokens (131,072 B each), but the usable context stays bounded by the configured 32,768 cells, the model's 40,960-token limit, GPU-tier KV (32,768 B/token) and workspace, and by restoring the weights before the next prompt; the measured request was 9,737 prompt + up to 2,048 generated tokens. At a budget 1 GiB below the control's prompt footprint the control decodes at 1.7 to 8.2 s/token with 23 M major faults and was stopped at 1,254 tokens; the combined arm served the full 2,048 tokens at 843 ms/token |

Not established: phone energy (assumed 3/4.5/6 W, not measured), run-to-run variance (single runs), token
identity (greedy paths diverge from the control after token 84), a larger *prompt* under a budget (prefill
still needs every weight resident: at the tight budget both arms' prefill slowed by a third or more).

## Why this stage exists

The 2026-09-17 `layer-kv-placement` stage measured its "phone-owned FFNs" arm in remote-prefill mode:
the desktop omitted the FFN weights of layers 0-17 before context allocation, so the phones also served
prefill (424 s vs 281 s) and the request cost 6.8 % more energy. That is the mode the decode-split work had
already ruled out. This stage keeps prefill exactly as the desktop control runs it and relocates the FFN
weights **only while decoding**:

1. prefill on the desktop with every weight resident (no phone traffic);
2. at the first generated token the `decode-boundary-v1` runtime control applies the phone split for
   layers 0-17 and the server releases the page tables and page cache of the phone-owned share of the FFN
   weights (dormant host share, `S41_SERVER_FFN_DORMANT_HOST_SHARE=1`);
3. before the next prompt the share is populated again.

The second enabler is new: **page-granular host KV backing**. `llama_kv_cache` used to `memset` every KV
buffer at construction, so a 4 GiB host KV allocation was resident from launch whether or not any token
used it. Plain CPU buffers are now zeroed with `MADV_DONTNEED` (zero-fill on first touch); the resident
footprint of the KV cache tracks the cells actually written, `clear(true)` gives the pages back, and the
weight pages released at the first token can back KV growth during the generation. Kill switch:
`LLAMA_KV_CACHE_EAGER_CLEAR=1`. Test: `research_dev/scheduler/tests/test_kv_lazy_backing.py` (native probe on a
tiny llama GGUF: eager minus lazy anonymous RSS equals the 128 MiB KV allocation, argmax traces identical).

Static KV allocation still cannot spend a decode-time release before the prefill that creates it. What this
stage claims is therefore: unchanged prefill, faster and cheaper decode, and a decode-phase footprint that is
several GiB smaller at the same allocated context — the room the generation grows into. It does not claim a
larger maximum prompt.

## Rig and protocol

Desktop `zhihao@172.20.74.85` (i9-12900K, 30 GB RAM, RTX 4060 Ti 16 GB, NVMe), OP15 (`3C15AU002CL00000`,
qualified kernel `6.12.23-android16-5-o-g227664cbe007-4k`, three resident HTP sessions owning layers 0-5 /
6-11 / 12-17). Model `Qwen3-14B-Q4KM-dequant-f16.gguf` (29.5 GB F16), 16 GPU layers (25-39 + output head),
CPU parent layers 0-24, context 32,768, batch 512, ubatch 128, KV plan CPU layers 0-31 (4 GiB) / GPU layers
32-39 (1 GiB), F16 KV = 131,072 B per token on the CPU tier and 32,768 B on the GPU tier. Column quantum
2,176 (17,408 / 8). Same deployed bundle for every arm (`RUNTIME.json` library hashes), model file dropped
from the page cache before every launch, no other GPU workload, rig execution lock held. Host energy: RAPL
CPU package + NVML GPU board at 10 Hz; phone power is assumed at 3 / 4.5 / 6 W (not measured), charged either
over the whole paid span (conservative) or only while assisting decode. The desktop bundle is
`/mnt/storage/s42-kv-decode-relocation-20260917-v1-eedc22/` (`cuda-build`, `BUILD-v3.log`).

Harness: `kv_decode_relocation_gate.py` (arms `control`, `combined`, `sweep`, `pair`; one request per split
ratio; per-token progress timestamps; 0.5 s memory timeline from `/proc/<pid>/status`, cgroup v2 and
`/proc/meminfo`; page-map residency of the released share; dormant proofs parsed from the server log;
split-aware phone proofs). `analyze_gate.py` renders the tables below.

## Split-ratio sweep (`sweep-v2`, 1,158-token prompt, 96 output tokens, dormant release on)

One server, three phone sessions, five requests in one process. Plain = phone attached but no control.

| phone share | host cols | decode ms/token | request host J | decode-phase host J | decode-phase host W | released during decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 % (plain) | 17,408 | 618.3 | 9,246 | 7,192 | 121 | – |
| 25 % | 13,056 | 542.0 | 8,340 | 6,345 | 122 | 1.93 GiB |
| 50 % | 8,704 | **464.7** | 7,636 | 5,455 | 122 | 4.13 GiB |
| 75 % | 4,352 | 525.4 | 6,833 | 4,274 | 85 | 6.42 GiB |
| 100 % | 0 | 532.7 | **5,743** | **3,465** | 68 | **8.97 GiB** |

Two optima: the fastest decode is the 50 % split (host and phone finish together); the cheapest and the
most memory-freeing point is 100 % phone FFN during decode (the host CPU idles through the FFN of 18
layers). The exact release proofs equal the model-derived figures: 2,076,033,024 / 4,435,329,024 /
6,888,996,864 / 9,625,706,496 bytes, all above their planning lower bounds. Restore before the next prompt
took 2.4 / 6.1 / 4.8 s. Every split request: control acknowledged at token index 2, share residency 1.0 at
prefill+8 s and 0.0 at decode+20 s, 564 phone calls per session per request, 94 assisted rows per owned layer
(= 96 tokens minus the two decoded before the control landed).

## Matched pair, uncapped (`pair-v1`, 9,737-token prompt, 1,024 output tokens, 50 % split)

One run per arm, combined first. Same bundle, model, prompt, KV plan (CPU 0-31 / GPU 32-39, 32,768 cells)
and greedy sampling; no memory cap (30 GB host, nothing else running). Prefill is local in both arms.

| Metric | Desktop control | Combined (50 % phone FFN in decode, dormant release) |
| --- | ---: | ---: |
| Desktop load to READY | 62.0 s | 73.8 s |
| Phone preload (3 sessions) | – | 41.9 s |
| Prefill (request start to first token) | 306.7 s | 286.7 s (-6.5 %) |
| Decode | 817.4 ms/token | 650.5 ms/token (-20.4 %) |
| Decode ms/token, first to last 128-token bucket | 810 to 834 | 637 to 667 |
| Request host energy (CPU package + GPU board) | 138.2 kJ | 115.6 kJ (-16.3 %) |
| of which prefill / decode | 32.0 / 106.2 kJ | 31.0 / 84.6 kJ |
| Decode-phase host power | 126.8 W | 126.9 W |
| Paid span (load + request) | 1,206.8 s | 1,079.1 s |
| Paid host energy | 140.3 kJ | 118.4 kJ (-15.6 %) |
| Paid host + assumed phone 3 / 4.5 / 6 W (active while assisting, idle 0.875 W otherwise) | 141.3 kJ | 120.8 / 121.8 / 122.8 kJ (-14.6 / -13.9 / -13.1 %) |
| Process RSS at the prefill peak | 18.07 GiB | 17.56 GiB |
| Process RSS while decoding | 16.8 to 18.1 GiB | 13.45 to 13.58 GiB |
| Released at the first token (proof, exact) | – | 4,435,329,024 B in 114 ms |
| Phone calls per session / assisted rows per owned layer | – | 6,132 / 1,022 (= 1,024 - 2) |
| Share residency prefill+8 s / decode+20 s / decode+120 s | – | 1.0 / 0.0 / 0.0 |
| Output tokens | 1,024 | 1,024, greedy path diverges at token 84 (HMX vs CPU arithmetic) |

Reading: the phone takes half of every FFN in layers 0-17 during decode, so the 50 % arm is 20 % faster
per token while the host draws the same power, hence 20 % less decode energy and 16 % less per request
with an unchanged prefill. Its decode footprint is 4.1 GiB below its own prefill footprint and about
4.5 GiB below the control's. As a memory equivalent (131,072 B per CPU-tier KV token) that is roughly 69k
tokens of room at a 22 GiB budget versus 32k-40k for the control; it is not a usable-context figure (see
the Outcome caveat: configured cells, model limit, GPU-tier KV and workspace, restoration before the next prompt). The control's file RSS moved between 15.5 and 16.8 GiB
during its generation while the host page cache was full (the whole 29.5 GB file is cached after load);
the combined arm's file RSS was steady at 11.88 GiB. Single runs: the prefill difference is within
run-to-run variation of this rig (pair-r3 measured 281 s for the same prompt), the decode and energy
differences are not.

## Second operating point: 100 % phone FFN during decode (`combined-h0-v1`, same prompt and tokens)

| Metric | Desktop control (pair-v1) | Combined, 100 % phone FFN in decode |
| --- | ---: | ---: |
| Prefill | 306.7 s | 289.8 s (-5.5 %) |
| Decode | 817.4 ms/token | 819.3 ms/token (+0.2 %) |
| Decode ms/token, first to last bucket | 810 to 834 | 795 to 837 |
| Request host energy | 138.2 kJ | 90.7 kJ (-34.4 %) |
| of which prefill / decode | 32.0 / 106.2 kJ | 31.3 / 59.4 kJ (decode -44.1 %) |
| Decode-phase host power | 126.8 W | 70.8 W |
| Paid host + assumed phone 3 / 4.5 / 6 W (assisting) | 141.3 kJ | 96.4 / 97.7 / 98.9 kJ (-31.8 / -30.9 / -30.0 %) |
| Process RSS while decoding | 16.8 to 18.1 GiB | 9.24 to 9.37 GiB |
| Released at the first token (proof, exact) | – | 9,625,706,496 B in 86 ms |
| Phone calls per session / assisted rows per owned layer | – | 6,132 / 1,022 (exact) |
| Share residency prefill+8 s / decode+20 s / decode+120 s | – | 1.0 / 0.0 / 0.0 |

Reading: with the whole FFN of layers 0-17 on the phone, host attention and phone FFN serialize inside
each layer, so at a 9.7k-token context the per-token time equals the control's (at the 1.2k-token sweep
context it was 14 % faster). The host idles through 18 FFNs per token, which is where the 44 % decode
energy saving comes from, and the decode footprint drops by 7.7 GiB against the control (memory-equivalent of 63k CPU-tier KV
tokens; the configured context is 32,768 cells and the model's limit 40,960, so this is room for the
generation and for other decode-phase consumers, not a larger usable context by itself). The choice between the two
points is a policy decision: 50 % for latency, 100 % for energy and memory; prefill is local in both.

## Budget-capped pair (`capped-v1`, 100 % split, 2,048 output tokens, `MemoryMax` 19,394,461,696 B)

Each arm runs alone in a transient user scope (`systemd-run --user --scope -p MemoryMax=... -p MemorySwapMax=0`);
the budget is the control's uncapped prefill-peak RSS (18.07 GiB) rounded down to 64 MiB, i.e. the budget
that exactly fits the control's prompt. The model file is dropped from the page cache before each launch, so
the scope is charged for every page it faults.

| Metric | Desktop control | Combined (100 % phone FFN in decode) |
| --- | ---: | ---: |
| Desktop load to READY | 130.5 s (1.25 M major faults: the load already runs at the cap) | 81.5 s |
| Prefill | 315.7 s | 285.2 s |
| Decode (2,048 tokens) | 824.6 ms/token | 838.0 ms/token |
| Decode ms/token, first to last 128-token bucket | 784 to 860 | 789 to 882 |
| Request host energy | 247.4 kJ | 152.7 kJ (-38.3 %) |
| Decode-phase host power | 127.6 W | 71.1 W |
| Paid host + assumed phone 4.5 W (assisting) | 253.2 kJ | 163.8 kJ (-35.3 %) |
| Charged memory during decode (`memory.current`) | 18.03 to 18.06 GiB, pinned at the cap | 12.0 GiB after the release |
| Process RSS during decode | 17.17 to 17.43 GiB | 8.27 to 8.53 GiB |
| Major faults during prefill / during decode | +46k / +36 | +1.0k / +37 |
| `memory.max` events over the request | 4,901 to 5,318 | 3,976 to 4,308 |
| OOM kills | 0 | 0 |

Reading: at the budget that exactly fits its prompt, the control still serves the 2,048-token generation at
full speed because the kernel reclaims the unmapped page cache of the GPU-layer weights (about 11 GB, faulted
during load) before it has to touch the mapped CPU-layer weights; the 256 MiB of KV growth fits in that
slack. So this budget shows equal service with 8.9 GiB less footprint and 38 % less host energy, not a control
failure. The control's cost is at load: with the cap already binding while the file streams in, the load took
130 s and 1.25 M major faults. `capped-v2` repeats the pair 1 GiB tighter (17.06 GiB) to find the point where
the control's working set no longer fits.

## Budget-capped pair, 1 GiB tighter (`capped-v2`, `MemoryMax` 18,320,719,872 B = 17.06 GiB)

Same protocol, budget 1 GiB below the control's prompt footprint, so the control's mapped working set no
longer fits and neither arm's prefill does.

| Metric | Desktop control | Combined (100 % phone FFN in decode) |
| --- | ---: | ---: |
| Desktop load to READY | 112.5 s | 78.8 s |
| Prefill | 406.8 s (+33 % vs uncapped) | 479.5 s (+68 % vs uncapped: the deficit is re-read on every ubatch) |
| Major faults during prefill | +229k | +295k |
| Decode (2,048 tokens) | stopped at 1,254 tokens after 3,141.6 s: 2,505 ms/token average, 1,702 / 1,861 / 1,687 / 1,622 / 1,673 / 1,951 / 1,729 / 2,244 / 3,528 / 8,215 ms/token per 128-token bucket | 842.9 ms/token (812 to 886 across the generation) |
| Major faults during decode | +23.2 M (`memory.max` events 10.6k to 955k) | +15k, flat after the release |
| Charged memory during decode | 15.5 to 17.06 GiB, pinned at the cap | 11.1 GiB after the 9.63 GB release (316 ms) |
| Process RSS during decode | 15.27 to 16.97 GiB (mapped weights evicted and re-read every token) | 8.08 to 8.44 GiB |
| Request host energy | 257.1 kJ for 1,254 tokens; decode 221.4 kJ = 176.6 J per generated token at 70.5 W (combined: 60.0 J per token) | 161.2 kJ (decode 122.8 kJ at 71.1 W) |
| Phone proof | – | exact: 12,276 calls per session, 2,046 assisted rows per owned layer |

Reading: 1 GiB below the control's prompt footprint, the control's mapped weights no longer fit. Its prefill
pays once per ubatch (+33 %), its decode pays on every token: 1.7 s per token from the start, 8.2 s per token
by token 1,254 as the growing KV displaces more weight pages, 23 million major faults, three times the
uncapped energy per generated token. The run was stopped deliberately at 17:33 after 62 minutes (server
`SIGTERM`; the harness kept the per-token timestamps, memory timeline and power samples in
`EXECUTION-*.json`, `MEMORY.jsonl`, `POWER_SAMPLES.json`, and `FAILURE.json` records the stop) because the
projected finish had moved past two hours; it is a partial record, not a completed request. The combined arm
under the same budget paid the same prefill penalty, released 9.63 GB at the first token and then served the
whole 2,048-token generation at 843 ms per token with flat faults. This is the "longer context under a fixed
host budget" evidence: at this budget the desktop alone cannot serve the growing context at usable speed, and
the decode-only relocation can, because the released FFN pages are exactly the room the KV grows into.

## Follow-ups closed after review

1. **Wording.** Every "additional KV tokens" figure above is a memory equivalent of the released bytes
   at 131,072 B per CPU-tier token. The measured request was 9,737 prompt tokens plus 1,024 or 2,048
   generated tokens inside a configured 32,768-cell context; the model's limit is 40,960; GPU-tier KV
   (32,768 B/token), workspace and the restoration of the share before the next prompt all still bound
   what a request can use.
2. **Automatic selection and admission** (`_internal/decode_split_selection.py`, 13 tests).
   - *Selection* reads the digest-pinned atlas (`campaigns/burstgpt/data/QWEN_DECODE_SPLIT_ATLAS.json`,
     schema v2, 12 rows from this report's records via `build_decode_split_atlas.py`). A row applies only when
     the execution environment matches exactly (artifact, GPU layers, context cells, batch and ubatch, KV
     plan digest, column quantum, phone session masks, runtime bundle digest) and the request lies inside
     the row's validated shape: prompt length within 25 % of the measured prompt (868-1,447 or 7,302-12,171
     tokens) and prompt plus expected output no larger than the measured total (1,254 or 10,761). A
     million-token prompt, a 5,000-token prompt in the unmeasured gap, a longer generation than measured or
     any environment difference fails closed; capped pressure-test rows are never profiles. Objectives:
     `latency` -> 50 % (both shapes), `energy` -> 100 % (host + assumed phone J/token), `memory` -> the largest
     release under a required release and an optional latency bound.
   - *Accounting* (`DecodeReleaseAccountant`): the share of one physical allocation (a server endpoint) is
     its own ledger owner, booked before the first prompt with a `ShareBinding` (endpoint, artifact, layer
     mask, host columns, expected bytes). A `phase=decode` proof credits it only if it comes from that
     endpoint, covers the bound layers and columns and carries a release generation never credited before;
     the same proof cannot credit a second owner and cannot be replayed. A shortfall against the reservation
     stays charged. Restoration before the next prompt is one ledger transaction (checkpoint/restore): when
     the room has been consumed, the ledger is left byte-for-byte unchanged, including the retained
     shortfall (19 GiB stays 19 GiB in the regression test), and the prompt is held.
   - *Execution path*: the gate's `admission` arm runs the two-request flow on one server inside a MemoryMax
     scope: share booked -> prompt 1 -> the server's decode proof is credited at the first token -> a tenant
     takes the released room (ledger growth reservation plus touched anonymous memory in the same cgroup)
     -> prompt 2 is refused (`preview_prompt_admission` false, `restore_before_prompt` raises, replay of proof
     1 refused) -> the tenant leaves -> the share is re-reserved -> prompt 2 runs: the server populates the
     share (`phase=local` proof), prefills, decodes and releases again as generation 2. Results: see
     "Two-request admission gate" below. `scheduler_qualified` stays false: this is the mechanics of admission
     on the real server, not a full trace under the automated scheduler.
3. **Reuse and recovery.** The probe gained `--clear-reuse` (decode, `clear(true)`, decode the same prompt
   on the same context) and `--dormant-consume` (release the share, occupy exactly the released bytes with
   touched anonymous memory, decode, restore). `test_kv_lazy_backing.py` checks that the lazy clear returns
   the written KV pages (16 MiB of a 4,096-token prompt on the tiny model; the eager clear returns none)
   with identical logits after reuse, and that the restore after the room was consumed is byte-exact with
   identical logits. Under a hard budget that sequence would fault or be OOM-killed, which is the case the
   accountant's prompt gate prevents and the admission arm exercises physically.

## Two-request admission gate (`admission-v1`, `MemoryMax` 19,394,461,696 B, `--select energy`)

One server, three phone sessions, 1,158-token prompt, 64 output tokens per request, inside an 18.06 GiB
scope. The selector chose 100 % (expected release 9,625,706,496 B). Ledger: capacity = MemoryMax, 256 MiB
safety reserve, server base 7.22 GB (RSS at READY minus the share), 1 GiB workspace, 160 MB KV, share via
the accountant. Times are from the share booking; "charged" is the scope's `memory.current`.

| t | event | charged | server RSS | ledger reserved | note |
| ---: | --- | ---: | ---: | ---: | --- |
| 0.0 s | share booked | 17.88 GiB | 15.69 GiB | 16.84 GiB | prompt 1 admitted with the share resident |
| 27.8 s | release credited, generation 1 | 12.04 GiB | 7.14 GiB | 7.88 GiB | server proof 9,625,706,496 B in 279 ms; share residency 0.0; headroom 10.67 GB |
| 61.8 s | request 1 done | 12.06 GiB | 7.15 GiB | 7.88 GiB | prefill 26.6 s, decode 544 ms/token, 62 assisted rows per layer (exact) |
| 64.2 s | tenant admitted | 18.05 GiB | 7.15 GiB | 15.84 GiB | 8,551,964,672 B of touched anonymous memory in the same cgroup; headroom 2.1 GB |
| 64.2 s | request 2 refused | 18.05 GiB | 7.15 GiB | 15.84 GiB | `preview_prompt_admission` false; `restore_before_prompt` -> capacity insufficient; state `restore-blocked`; a restore now would exceed the cap by 9.6 GB |
| 64.2 s | replay refused | | | | proof of generation 1 presented again -> refused |
| 84.2 s | held 20 s | 18.05 GiB | 7.15 GiB | 15.84 GiB | ledger unchanged while blocked |
| 85.6 s | request 2 admitted | 10.04 GiB | 7.15 GiB | 16.84 GiB | tenant terminated, growth released, share re-reserved, state `prefill-resident` |
| 135.4 s | release credited, generation 2 | 10.29 GiB | 7.15 GiB | 7.88 GiB | the server populated the share first (`phase=local`, 9,625,706,496 B in 23.4 s under the cap), prefilled (48.6 s incl. the populate), then released again in 61 ms |
| 168.4 s | request 2 done | 10.29 GiB | 7.15 GiB | 7.88 GiB | decode 533 ms/token, 62 assisted rows per layer (exact) |

Reading: the accountant's decision and the kernel's accounting agree at every step. While the tenant held
the released room the scope sat at its cap and the share could not have been populated without eviction
or an OOM kill, and the second prompt was refused on the ledger alone, before anything was sent to the
server. Once the room returned the share was re-reserved and the server's own populate ran with the room
available. The proof of generation 1 could not credit twice. The whole flow took 294 s of paid time; the
run is recorded in `physical/admission-v1/` (events with cgroup, server and ledger state in `RESULT.json`).
Not claimed: automatic scheduler routing of real traffic (`scheduler_qualified` false), repeated runs, or
the populate cost under a cap at long prompts (23.4 s here for 9.63 GB).

## Trace integration (in progress, 2026-09-17 evening)

Goal: automatic routing of the reduced 24-request BurstGPT trace (`burstgpt_sparse_locality24_v1`: 15
Qwen, 6 Gemma, 3 Llama-1B requests over 28 minutes) with the decode-only relocation on the Qwen parent,
under the automated scheduler. Built so far:

- Server: every control acknowledgement's `runtime_stats` now carries `dormant_release_generation`,
  `dormant_released_bytes`, `dormant_layer_mask`, `dormant_host_columns`, `dormant_release_elapsed_us`
  (`server_context::ffn_dormant_state`), so the scheduler credits releases from the acknowledgement it
  already reads instead of scraping logs.
- Adapters: `CanonicalHttpExecutionBackend` gained `on_control_ack` and `before_prompt` hooks (adaptive
  and static acknowledgement paths, main completion). `adapters/dormant_share_coordinator.py` is the rig's
  admission: one ledger over a host budget, every launched server booked with its resident footprint
  (non-strict: over-subscription is recorded, not fatal, because the scheduler already launched it), the
  dormant server's largest share booked with a `ShareBinding`, each acknowledgement generation credited
  once, and `before_prompt` re-reserving the share or holding the prompt until the room returns (fail
  closed at the hold timeout). Wired into `HeterogeneousPhysicalRig` (`host_memory_budget_bytes`), server
  publish and retire. 5 coordinator tests.
- Contract path: `ffn_host_share_release` is accepted in the dormant runtime contract (both whitelists) so
  a model's `phone_adapter_parameters` in `models.json` switch the release on for the trace's Qwen parent.
- Runner and campaign: `--host-memory-budget-bytes` / `host_memory_budget_bytes`.
- Selection: `DecodeSplitEnvironment` gained `parallel`; the gate sweep can run exact prompt lengths
  (`--prompt-tokens-list`) for a later calibration in the trace's shapes (context 4096, four slots, batch
  2048, ubatch 512). The trace path itself does not consult the atlas: the existing adaptive controller
  chooses the fraction online, the server releases whatever share that fraction implies, and the
  coordinator credits it.
- First run (`s42-dormant-trace-20260917-v1-run`) failed after 20 minutes with every request queued behind a
  residency-transition barrier: the new acknowledgement fields violated the adapter's runtime-stats whitelist
  (`_runtime_stats` accepts a fixed key set of non-negative integers), so every control acknowledgement raised
  `FFN runtime stats are invalid` and dispatch replanned forever. Fixed on both sides (flag emitted as 0/1, six
  dormant keys whitelisted); rerun as `-run2`.
- Second run failed at phone preflight (identity pinned the pre-rebuild bundle; re-materialized). Third run
  (`-run3`) flowed (7 completed, first release 3.21 GB for layers 0-5, populate 2.5 s) until the coordinator
  refused that release: the helper had attached to layers 0-5 while the booking covered layers 0-17, and the
  refusal was raised inside the acknowledgement path, which the backend reports as a control failure. Fixed:
  a proof may release any subset of the booked layers (the remainder stays charged), and a refused credit is
  an accounting event, never an exception in the request path. Rerun as `-run4`.
- Fourth run (`-run4`) reached 25 streams with releases of 3.21 / 4.59 / 6.42 GB (layers 0-5 and 0-11, host
  columns 0 and 4352, the controller's own fractions) and their populates (2.2-3.3 s), then failed on the
  adapter's monotonic-counter check when the release state reset to zero after a populate. The five state fields
  are now gauges (only `dormant_release_generation` is a counter). Rerun as `-run5`.
- Inputs `s42-dormant-trace-20260917-v1` on the desktop: the reduced-24 configuration set re-pinned to the
  new bundle, a 28 GiB host budget, Qwen `ffn_host_share_release: 1`; the transport qualification identity
  for the new bundle is the remaining prerequisite before preflight.

## KV growth into the freed memory, directly (`kv-headroom-v1`)

Does the KV cache really occupy the released pages? Tested without a phone and without a long generation:
the probe loads Qwen (16 GPU layers, CPU KV layers 0-31, 32,768 cells), decodes two tokens, then writes the
first 24,576 KV cells of every layer through the new test hook `llama_kv_touch_cells` (the same pages a real
cache fill of that many tokens occupies: 3.0 GiB on the host tier plus 0.75 GiB on the GPU tier), inside an
18 GiB `MemoryMax` scope after the model file is dropped from the page cache. The combined arm releases the
phone share of layers 0-17 (`llama_model_ffn_host_share_release`, host columns 0) before the fill.

| | control (no release) | combined (share released first) |
| --- | ---: | ---: |
| Weights resident before the fill | 8.95 GiB | 15.38 GiB before, 6.42 GiB after releasing 8.97 GiB in 0.31 s |
| KV fill (24,576 cells, 40 layers) | 3.75 GiB written, 1.67 s | 3.75 GiB written, 1.12 s |
| Anonymous RSS across the fill | 0.22 to 3.22 GiB | 0.22 to 3.22 GiB |
| Weights resident after the fill | 5.93 GiB (3.02 GiB evicted) | 6.42 GiB (unchanged) |
| Major faults across the fill | 0 | 0 |
| Scope `memory.max` events (mostly load) / OOM | 8,810 / 0 | 7,387 / 0 |
| Decode argmax | 4180, 13, 2160 | 4180, 13, 2160 |

Reading: under a budget that cannot hold weights plus this KV, the control's fill is paid for by evicting
3 GiB of weights it must re-read on every token; after the release the same fill lands in the freed room and
the remaining weights stay resident (the share is then populated back in 21.4 s). This is the direct evidence that the freed memory backs additional KV
context. Bound: it is memory, not tokens; the fill stands in for a cache of 24,576 tokens (the configured
context is 32,768 cells) and the released layers' FFN must be computed by the phone during decode for the
result to be usable. Script `kv_headroom_probe.sh`, records in `physical/kv-headroom-v1/`.

Next step, handed off: split-slice attention for GPU-resident layers so the host room also serves the
GPU layers' KV, plan in `research_dev/SPLIT_KV_ATTENTION_PLAN.md`.

Operational note: the probe needed a server rebuild (BUILD-v6, the KV touch hook), so the trace bundle's
transport identity in `/home/zhihao/s42-dormant-trace-20260917-v1-inputs/TRANSPORT_QUALIFICATION_IDENTITY.json`
pins a stale `libllama`; re-materialize it before any further trace run.

## Reduced BurstGPT trace under the automated scheduler (`s42-dormant-trace-20260917-v1-run5`)

The reduced 24-request trace (`burstgpt_sparse_locality24_v1`: 15 Qwen, 6 Gemma, 3 Llama-1B) ran to
completion under the automated scheduler with the decode-only relocation switched on for the Qwen parent
(`ffn_host_share_release: 1`), a 28 GiB host budget and the rig-level admission coordinator. Fifth attempt;
the four earlier attempts died on integration details recorded above (stats whitelist, identity after a
rebuild, subset-layer proofs raised in the acknowledgement path, state fields treated as counters).

| | baseline gate 2026-09-09 (no release, old bundle) | run5 (release on, new bundle) |
| --- | ---: | ---: |
| Requests terminal / rejected | 24 / 0 | 24 / 0 |
| Execution attempts | 49 | 47 |
| Trace duration | 1,689.1 s | 1,651.1 s |
| Sum of request latencies | 1,625.6 s | 1,473.3 s (-9.4 %) |
| Fleet energy, CPU package | 98.8 kJ | 77.0 kJ (-22.1 %) |
| Fleet energy, GPU board | 52.0 kJ | 50.5 kJ (-2.9 %) |
| Fleet energy, phone (assumed 4.5 W active / 0.875 W idle) | 1.8 kJ | 2.0 kJ |
| Fleet energy total | 152.6 kJ (90.3 W) | 129.4 kJ (78.4 W, -15.2 %) |
| Qwen requests decoded with phone assistance | 0 whole + adaptive windows | 1 whole + adaptive windows (executed fractions 1000000: 5, 750000 windows) |
| Gemma requests with phone assistance | 1 | 4 |

Release mechanics observed in the trace: 19 server-side decode releases and 19 populates on the Qwen
servers (three launches of the hot parent over the trace, 3.21 / 4.59 / 6.42 GB per release for the
controller's own layer subsets 0-5 and 0-11 and fractions 75 / 100 %; populate 2.2-3.3 s); 7 server
bookings and 6 retirements in the ledger, none over budget; 3 prompt restorations, all admitted without
waiting (the budget never bound at 28 GiB). The coordinator credited 3 of the 19 releases: credits ride on
control acknowledgements, which are composed before the release the acknowledged policy triggers, so most
releases were only visible in later statistics reads; one credit was refused because a relaunched server on
the same endpoint restarted its generation counter. Both are fixed after this run (credits from every FFN
statistics read; consumed events reset when a server is forgotten) and a sixth run re-checks them.

Run 6 (same inputs, credits taken from every FFN statistics read, consumed events reset per launch) passed
again: 24 / 24, 54 attempts, 1,678.3 s, fleet energy 132.4 kJ (CPU 78.0, GPU 52.4, phone 2.0 kJ; 78.9 W),
sum of latencies 1,916.6 s; 39 server releases and 38 populates, this time mostly the full 9.06 GB share
(all three sessions attached), 5 prompt restorations admitted without waiting. Credits rose to 7; 18 new
generations were refused as "no resident share" because the adaptive controller changes the fraction or layer
subset mid-request and the server re-releases without a prompt in between. The accountant now treats a new
generation while released as a re-release (the shortfall is re-sized to the new proof; if the room has been
consumed the previous accounting is kept and the event is reported); the next run re-checks rule B.

Run 7 (re-release accounting) passed with **zero refused credits**: 24 / 24, 46 attempts, 1,577.4 s, fleet
energy 127.0 kJ (CPU 77.0, GPU 48.2, phone 1.9 kJ; 80.5 W), sum of latencies 1,508.5 s; 27 releases and 27
populates (4.59 / 6.42 / 9.06 GB), 4 prompt restorations admitted without waiting, 9 release generations
observed and credited once each. The 27-versus-9 gap is observation, not accounting: with four server
slots a release is created and populated again between two statistics reads and is then never visible to
the coordinator. The checker's rule B is therefore observation-based (every observed generation credited
once, nothing refused) and reports coverage (0.33 here); complete coverage needs the server to publish
release events rather than state, which is the next server-side item.

| run | trace | duration | fleet energy | releases / populates | credited | refused | holds | checker |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline 09-09 | PASS | 1,689 s | 152.6 kJ | – | – | – | – | – |
| run 5 | PASS | 1,651 s | 129.4 kJ | 19 / 19 | 3 | 1 | 0 | FAIL (refusal) |
| run 6 | PASS | 1,678 s | 132.4 kJ | 39 / 38 | 7 | 18 | 0 | FAIL (refusals) |
| run 7 | PASS | 1,577 s | 127.0 kJ | 27 / 27 | 9 | 0 | 0 | PASS |

Caveats: single runs eight days apart on different bundles (the new bundle also carries page-granular KV
backing and the fused Hexagon FFN work), so the energy and latency differences are indicative, not a matched
A/B; phone energy is assumed; the adaptive controller chose more phone assistance in run5 than in the
baseline, which is part of the difference. `scheduler_qualified` for the release route stays false until a
matched pair on one bundle exists.

## Changes in this stage

- `src/llama-kv-cache.cpp`: `llama_kv_cache_clear_buffer` (lazy zeroing of plain host buffers; env kill switch).
- `examples/layersplit/ffn-remote-resident-probe.cpp`: `context_rss_anon_bytes`, `--clear-reuse`, `--dormant-consume`, `--kv-touch-tokens`, `--dormant-no-decode`.
- `src/llama-kv-cache.{h,cpp}`, `src/llama-context.cpp`, `include/llama.h`: `llama_kv_touch_cells` test hook.
- `research_dev/scheduler/tests/test_kv_lazy_backing.py` (new).
- `kv_decode_relocation_gate.py` (`--select`), `analyze_gate.py`, `build_decode_split_atlas.py`, `rig_config.json` (this directory, new).
- `research_dev/scheduler/_internal/decode_split_selection.py` (selector, `ShareBinding`, `DecodeReleaseAccountant`), `tests/test_decode_split_selection.py`, `campaigns/burstgpt/data/QWEN_DECODE_SPLIT_ATLAS.json` (schema v2) (new).
