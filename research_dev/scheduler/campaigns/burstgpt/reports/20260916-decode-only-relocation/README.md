# Decode-only FFN weight relocation: dormant host share (2026-09-16/17)

Decision (user, 2026-09-16): full relocation of the phone-owned FFN weights makes prefill
2.0-2.25x slower because the desktop would otherwise stream those weights to the GPU over
PCIe for every prompt microbatch while the phone path is bounded by USB 3 (a 512-row call
costs 87 ms, 30 ms of it NPU compute). So the weights stay on the desktop and are relocated
for decode only: the phone executes its share of the FFN columns of the split layers during
decode (`decode-boundary-v1` runtime control), and the desktop server releases the pages of
exactly that share while every slot decodes, populating them again before the next prompt.

Ratio: 75 % of the FFN columns on the phone (`20260916-decode-split-ratio/`), host columns
3,840 of 15,360.

## Mechanism

| Layer | What |
| --- | --- |
| `src/llama-mmap.cpp` | `release_fragments`: `MADV_DONTNEED` on inward page-aligned pieces (page-table entries dropped, exact RSS accounting) then `POSIX_FADV_DONTNEED` over the tensor (clean, now-unmapped pages leave the page cache; still-mapped host-prefix pages are skipped by the kernel); `populate_fragments`: `MADV_POPULATE_READ`. The mapping keeps its own duplicated descriptor because the loader closes the file after loading. `MADV_PAGEOUT` was tried first and rejected: it reclaims whole folios only, so an inward-aligned range lost pages at both ends |
| `src/llama-model-loader.cpp` | records every dense FFN weight's file range and geometry (`dense_ffn_weights`) for memory-mapped loads |
| `src/llama-model.cpp`, `include/llama.h` | `ffn_host_share_release(layer_mask, host_columns)` / `restore`: gate/up share = one contiguous suffix of rows per tensor, `down` share = the column suffix of every output row (3,840 pieces per layer); idempotent per geometry; `llama_model_ffn_host_share_*` C API |
| `tools/server/server-context.cpp` | `apply_dormant_host_share` after the split policy is applied to a batch: release when every processing slot is generating under a split policy, populate before any prompt processing; `S41_SERVER_FFN_DORMANT_HOST_SHARE=1` (requires runtime control, excludes remote-resident layers) |
| scheduler | adapter parameter `ffn_host_share_release=1` -> the environment flag (`adapters/llama_server_contracts.py`); `RuntimeHostShareReleaseProof` parses `S41SERVERFFN dormant_host_share phase=decode|local ...`; `host_share_release_lower_bound_bytes` gives the planning bound (`_internal/runtime_resources.py`) |
| tests | `tests/test_remote_resident_native.py::DormantHostShareNativeTests` (page-exact geometry against the GGUF, kernel accounting, bit-identical re-execution after release), `tests/test_host_share_release.py` (7 tests); canonical suite 1,490 tests, 0 failures |

Timing of the primitive on the desktop (Gemma `down` tensor, 3,840 row pieces of one layer):
2.4 ms single-threaded, 33 ms with 24 busy threads. A server-side release of the eight-layer
share (30,736 pieces) took 28 ms; populate 31 ms while the pages were still cached.

## Gate v4: one HTP session (layers 0-7), 5,261-token document, 96 output tokens

`physical-v4-one-session/` (`RESULT.json`, `SUMMARY.json`, per-request `EXECUTION-*.json`,
`SERVER_MEMORY.jsonl` with model-file RSS and `/proc/meminfo` every 0.5 s, `PHONE_HEALTH.jsonl`,
server logs, `PHONE_LOG-*.json`, `gate-v4.log`). Driver `dormant_gate.py`, summary
`analyze_gate.py`. Remote `/mnt/storage/s42-decode-only-relocation-20260916-v1-5c3bb8/gate-v4/`.
Expected page-exact release for the eight layers at host columns 3,840: 1,981,743,104 bytes
over 30,736 ranges (planning lower bound 1,918,828,544).

| Arm | Requests | Prefill ms | Decode ms/token | Model-file RSS during prefill | Model-file RSS during decode and idle | Share pages resident during decode (pagemap) | Phone calls by mid-decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| desktop only | 2 | 22,989 / 22,717 | 468.3 / 467.9 | 13.81 GB | 13.81 GB | 483,824 / 483,824 | - |
| split, weights resident | 2 | 23,305 / 23,016 | 417.7 / 417.2 | 13.81 GB | 13.81 GB | 483,824 / 483,824 | 224 |
| dormant (release during decode) | 3 back to back | 23,445 / 23,277 / 23,250 | 419.2 / 418.9 / 416.9 | 13.81 GB | 11.83 GB | 0 / 483,824 | 224 |

Proof lines from the dormant server, one pair per request: `phase=decode ... released_bytes=
1981743104 ranges=30736 elapsed_us=27796|28569|28241` and `phase=local ... restored_bytes=
1981743104 elapsed_us=31367|31677|31007`. The release is byte-exact against the geometry
computed independently from the GGUF, the resident set of the model file drops by the same
1.98 GB and stays there while the server idles, and the next prompt's populate cost is 31 ms
here because this build's page-cache drop was a no-op (the descriptor bug fixed afterwards),
so the pages came back from the page cache. Decode speed is unchanged by the release (419 vs
417 ms per token); the split itself is 10.7 % faster than desktop-only decode, as in the sweep.

Owner loss: the recovery step could not kill the worker because adb over USB is unavailable
while the phone is in FunctionFS mode, and an in-flight kill is deliberately not attempted
without the DMA-BUF cancellation qualification (a mid-transfer kill crashed the phone kernel
on 2026-09-13). The "victim" request therefore completed normally; the follow-up request
without a control ran fully local at 467.8 ms per token after the server populated the share
(proof line `phase=local`). Owner-loss handling for this mode is the populate path itself and
is exercised by every prefill; the fail-closed behaviour of the FFN client is unchanged.

Gate history: v1 launched a second server on the same direct worker session (the direct
worker completes when its host disconnects, so the second server's FFN connect hung 30 s
with zero phone calls); v2 and v3 tripped a post-request stats check that only answers while
a slot is active; v4 is the first complete run. All four are preserved on the desktop.

## Gate 3s: three canonical sessions (layers 0-23), same document and settings

`physical-3s-v1-three-sessions/`. The three resident sessions HTP0/1/2 are started through
`DirectPhoneFfnSession.start` (resident workers plus router, hand-built transition command
with three `RuntimePhoneShard` rows), one session set per server. Expected page-exact release
for the 24 layers at host columns 3,840: 5,945,229,312 bytes over 92,208 ranges (planning
lower bound 5,756,485,632). This run used the build with the working page-cache drop.

| Arm | Requests | Prefill ms | Decode ms/token | Model-file RSS prefill / decode | System Cached prefill -> decode | System MemFree prefill -> decode | Share pages resident during decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| desktop only | 2 | 23,151 / 22,975 | 468.2 / 467.7 | 13.81 / 13.81 GB | 28.9 -> 28.7 GB | 0.4 -> 0.5 GB | all |
| split, weights resident | 2 | 23,301 / 23,055 | 339.3 / 339.3 | 13.81 / 13.81 GB | 28.9 -> 28.9 GB | 0.3 -> 0.3 GB | all |
| dormant, back to back | 3 | 23,486 / 31,562 / 30,814 | 341.6 / 338.6 / 337.8 | 13.81 / 7.86 GB | 28.9 -> 24.1 GB | 0.3 -> 5.1 GB | 0 of 1,451,472 |

Proof lines per request: `phase=decode ... released_bytes=5945229312 ranges=92208
elapsed_us=186781|171390|168536` and `phase=local ... restored_bytes=5945229312
elapsed_us=8254301|7516495|7525093`. The release is byte-exact, the model file's resident set
falls by the released amount and stays there while idle, and about 4.8 GB of it actually
leaves the page cache (the rest sits on page boundaries shared with retained rows). The
phone-assisted split alone makes decode 27.5 % faster than desktop-only; the release changes
decode by nothing measurable. The cost is the populate before the next prompt: 7.5-8.3 s to
read 5.95 GB back from the NVMe through the row-interleaved `down` layout, visible as prefill
going from 23.1 s to 31 s. The build now issues a whole-tensor readahead before the
synchronous populate; the 64k-context run below measures the effect.

Owner loss: as in gate v4, the kill step could not reach the phone (adb unavailable in
FunctionFS mode) and an in-flight kill is not attempted without the DMA-BUF cancellation
qualification. The follow-up local request populated the share and ran at 468 ms per token.

## 64k context, uncapped reference (`physical-ctx64k-v1-uncapped/`)

Same three-session configuration, 57,871-token prompt (the document repeated 11 times),
context 65,536, batch and ubatch 2,048, 32 output tokens, one desktop-only request and two
dormant requests back to back. The run was launched inside a 12.5 GiB user-scope memory cgroup,
but the model's page-cache pages were already charged to earlier cgroups, so the cap did not
bind (`memory.current` 2.9-3.6 GB, no `max` events); it serves as the uncapped reference.

| Arm | Prefill ms | Decode ms/token (32 tokens at ~58k context) | Model-file RSS prefill / decode | System Cached prefill -> decode |
| --- | ---: | ---: | ---: | ---: |
| desktop only | 102,905 | 698.0 | 13.81 / 13.81 GB | 27.4 -> 27.4 GB |
| dormant, request 1 | 103,403 | 564.0 | 13.81 / 7.87 GB | 27.8 -> 23.1 GB |
| dormant, request 2 (populate first) | 110,606 | 565.6 | 13.81 / 7.85 GB | 23.3 -> 22.1 GB |

Prefill at ubatch 2,048 runs 572 tokens per second on this desktop, both arms alike; the
populate before the second dormant prompt cost 7.67 s (the row-interleaved `down` share
dominates, and the whole-tensor readahead did not change that). Release 178.8 / 150.5 ms.
After this run the build stopped dropping the `down` share from the page cache: only its
page-table entries are released (the pages stay reclaimable), so the populate becomes a
re-map for `down` and a sequential read for gate/up.

## Budget-matched 64k context under a 12.5 GiB host memory cap

`physical-ctx64k-v2-capped-plain/` (desktop-only arm; the run was stopped after it because
the runtime-control call at the first token still used the client's 5 s HTTP timeout and the
capped server needed longer to answer) and `physical-ctx64k-v3-capped-dormant/` (the dormant
arm with the control timeout raised); merged summary `SUMMARY_CAPPED_64K.json`
(`analyze_capped.py`). Both arms ran inside `systemd-run --user --scope -p MemoryMax=12.5G
-p MemorySwapMax=0`, and the model file was evicted from the page cache before each server
launch (`--drop-model-cache`, 21.1 -> 2.0 GB and 12.4 -> 2.2 GB cached) so that every page
the server faults in is charged to the scope. Same 57,871-token prompt, context 65,536,
ubatch 2,048, 32 output tokens, three phone sessions.

| Arm | Prefill | Decode ms/token | Scope memory.current during decode | `max` reclaim events during decode | Major faults during decode | Model-file RSS during decode | Share pages resident during decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| desktop only | 812.1 s | 5,791 | 12.27-13.42 GB (at the limit) | +63,745 | +760,689 | 9.5-10.8 GB | 90 % |
| dormant, request 1 | 797.1 s | 717.6 | 10.47-13.42 GB | +1,727 | +26,977 | 5.0 GB | 5 % |
| dormant, request 2 | 770.6 s | 714.5 | 10.40-13.42 GB | +3,108 | +30,983 | 5.4 GB | 5 % |

Under the cap both arms pay the same prefill penalty (812 vs 797/771 s against 103 s
uncapped): prefill needs every weight, so the desktop streams them from the NVMe through a
page cache that the cgroup keeps reclaiming. The difference is decode. The desktop-only
working set (13.8 GB of mapped weights plus about 2 GB of anonymous memory) does not fit, so
it decodes at 5.8 s per token with the scope pinned at its limit. The dormant arm releases
the 5.95 GB phone share at its first decode step (169 / 271 ms) and its working set fits, so
it decodes at 0.72 s per token, 8.1x faster, against 0.56 s uncapped. The populate before the
second prompt took 21.7 s under the cap (7.7 s uncapped) because the re-read competes with
the cap's reclaim.

What this does and does not show. Under a host memory budget that the full-weight parent
cannot meet for decode, the decode-only relocation serves a 58k-token context at a usable
decode rate while keeping every weight on the desktop for prefill; that is the "freed memory
supports longer context" claim in the form this rig can demonstrate. It does not raise
`n_ctx` by itself (the KV cache is allocated at launch and Gemma's sliding-window KV is small
at every tested context), it frees no VRAM, and the prefill under such a budget is 8x slower
in both arms. The relocation's own costs are the phone (three sessions, 5.95 GB of shards)
and the populate before each prompt.

## Follow-ups

- Owner loss while a request is decoding needs a control channel that survives FunctionFS
  mode, or the DMA-BUF cancellation qualification; the populate path (proof `phase=local`) is
  exercised by every prompt and is the recovery.
- Scheduler integration beyond the launch flag and the proof record: a phase-conditional
  host-memory credit in the ledger and a route policy that chooses the dormant mode.
- The populate cost: the row-interleaved `down` share (1/3 of the release) is now kept in the
  page cache, but under a cap the cache is reclaimed anyway; a repacked `down` layout would make
  the share contiguous.
