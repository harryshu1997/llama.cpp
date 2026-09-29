# Robustness plan — "robust to new traces and new devices" (2026-09-25)

Goal: turn the four open robustness gaps into mechanisms + hardware evidence, in this priority order.
P1 graceful degradation (device lost mid-run) · P2 automatic demotion / re-admission (device gets slow) ·
P3 trace families · P4 device onboarding. P5 (repeats / confidence intervals) is deferred.
Everything is opt-in, fail-closed, developed in an isolated copy, `git apply --check`-ed against main, merged
with pre-merge backups, and measured only from the synced deploy (standing rules in
`NEXT_AGENT_PROMPT_SCHEDULER.md`). Nothing is committed without the user.

## 0. What exists today (grounded in the code)

| layer | mechanism | file | gap |
| --- | --- | --- | --- |
| server | per-helper runtime layer masks with rollback (`apply_policy`), a helper with mask 0 is switched off, the dormant host share keeps the host weights | `tools/server/server.cpp` (`s41_ffn_helper`, `apply_policy`, `apply_context`) | a helper RPC failure sets `client->failed()` → `S41SERVERFFNERROR` and the request errors (`Compute aborted`, run 2 of longtail_v1: `LIBUSB_ERROR_NO_DEVICE`); no in-flight host fallback |
| scheduler | device sets per batch composition, verdicts from measured pairs; drop codes `SERVER_DEVICE_SET_NOT_IMPROVED`, `…_LATENCY_BOUND_EXCEEDED`, `…_PROBE_BUDGET_EXHAUSTED`, `DEVICE_SET_FAILED` (a failed set is dropped with its supersets for every composition); monitored verdicts are rejected by `_reject_verdict` when the elimination test fails | `_internal/adaptive_decode_ops/coherence.py` | no re-admission (a dropped set stays dropped), no device-level drift detector, a lifecycle failure aborts the run (`adapters/coordinator.py` fail-fast) |
| scheduler | phone session controller raises `PhysicalAdapterError` on transport/identity loss; USB restore on close | `adapters/phone_session_ops/{completion,identity,transport}.py`, `adapters/phone_helpers.py` | the error propagates into the request lifecycle → run abort, no quarantine |
| onboarding | scripts scattered across report dirs: `prepare_campaign_eval.py` (evidence bundle, prior kernel, cost receipt), `make_gate_config.py`, `make_server_identity_config.py`, `tools/qualify_pixel_server.py`, `tools/qualify_op11_tcp.py`, `adapters/materialize_transport_qualification.py`, `shard_gguf.py` | ~1 day per device, hand-driven |
| traces | `campaigns/burstgpt/build_realistic_trace.py` knobs: `--arrival-scale --prompt-cap --output-cap --min-input --min-output --long-tail-threshold/-tolerance --min-same-model-overlaps --min-requests-per-model --small-model-share --duration-s --start-offset` | only one trace family measured (one model mix, batch ≤ 4, moderate load) |

## P1. Graceful degradation — a device that disappears never fails a request or changes an output

**Claim to earn:** offload is a pure optimization. If a phone dies mid-token, the request finishes with the
desktop's output; the device is quarantined; the run continues; the device is re-admitted when it comes back.

### Code-map findings (2026-09-25, read-only pass) that change the design
- **CORRECTION 2026-09-26: a helper failure does NOT kill the server** — `server-context.cpp:3149-3159` catches the decode exception, errors the processing slots and keeps serving with the helper client latched failed (any later graph reaching that helper aborts). The text below predates that finding. **(Original statement:) A helper failure kills the whole server today.** The FFN split client's failure makes `llama_decode` return 2;
  `tools/server/server-context.cpp:4120-4136` then sends "Compute aborted." to EVERY processing slot, clears their
  prompts and `throw`s out of the update loop → the llama-server process dies, co-tenant requests die with it, and
  the next attempt needs a model reload (50–80 s). So scheduler-only recovery is correct but slow; the server must
  stop aborting for anything better.
- **Most of the scheduler fallback path already exists.** `adapters/runtime.py::_recover` →
  `_unified/automated_requests_ops/failure.py::fail_automated_request` → `_prepare_automated_failure_fallback`
  (desktop-baseline recovery route) → `_commit_automated_attempt(event_kind="FALLBACK")` → the adapter re-executes
  the SAME payload (same prompt) and the coordinator does not abort (test
  `tests/test_physical_adapter.py::test_adapter_owns_failure_replan_and_logs_the_fallback`). Blockers: (a) every
  mid-stream failure is `execution_started=True, retry_safe=False` → `fallback_allowed` false → terminal
  (`_internal/runtime_execution.py:63-65`); (b) nothing names the device that died (`failure_reason` is free text;
  `coherence.py:539-541` "which phone failed is not known"); (c) the stream file is opened `"xb"` and
  `LlamaCppCompletionPayload` rejects an existing `stream_path` → a second attempt fails "completion stream path
  is not new" (`adapters/http_backend.py:122,277`); (d) the HTTP client discards the server's error body
  (`http_backend.py:273-291`), so the failure kind cannot be read from it — classify from the captured server
  stderr (`adapters/llama_server.py` `stderr_lines` + `marker.stderr_index`) or from the phone session instead.
- **Drop scope today:** `server_policy_failed` drops the failed device SET and its supersets for all batch
  compositions, but only inside one server-policy group (model × placement × layout × geometry); other models /
  layouts never see it. Quarantine primitives exist for routes/resources (`RuntimeController._quarantined_routes/
  _quarantined_resources`, applied in admission and recovery selection) but phone resources are not in a
  desktop-parent ticket's binding (late-attached helpers), so `failed_resource_ids` must not name them
  (`completion.py:144-152`).
- **Phone sessions:** the OP15 USB restore runs synchronously on the transition thread (up to ~195 s) and blocks
  the request; the Pixel adb-tcp session has no liveness poll (`active` = process handle exists) and
  `co_helper_lifecycle.py` has no per-device stop.

### S1 — scheduler-level recovery (no C++; ~2 days)
1. **Classify.** New structured failure kind on `PhysicalBackendFailure` / the ticket: `HELPER_LOST:<device_id>`.
   Sources: server stderr since `marker.stderr_index` (`S41SERVERFFNERROR`, `LIBUSB_ERROR_*`, "Compute
   aborted"), phone session transport errors (`PhysicalAdapterError` of transport kind), Pixel worker exit / adb
   forward loss. Anything else keeps today's semantics. `fallback_allowed` becomes true for `HELPER_LOST` even
   though execution started (the recovery route is desktop-only, so re-execution from the prompt is exact).
2. **Quarantine the device** (`coherence.py`): drop every device set containing `<device_id>` in EVERY server
   policy group (today: one group), add a device-level `quarantined_devices` set consulted by route generation /
   candidate selection (like `_quarantined_resources`), mark the phone session dead, run the USB restore /
   worker stop on a background thread (never on the transition thread), add a Pixel liveness poll.
3. **Recover the request.** Reuse the FALLBACK path; fix the stream-path exclusivity by moving the partial
   `streams/request-NNN.raw` aside (`.attempt1`) so the canonical path holds the recovered tokens (the offline
   identity check reads the canonical path); keep `validate_attempt_chains` FALLBACK semantics; record
   `REQUEST_RECOVERED{request, device, tokens_discarded, penalty_us}`.
4. **Survive the server death.** The recovery route's server is the same process that just died: the fallback
   must go through the existing server relaunch (model load) — record `SERVER_RELAUNCHED_AFTER_HELPER_LOSS` and
   its load time as part of the penalty. Co-tenant requests on that server fail too → they take the same
   FALLBACK path (one `HELPER_LOST` event fans out).
5. **Fault injector.** Extend `helper_preparation_fault_injection` (`configuration/campaign.py:286`,
   `adapters/runtime.py:76-141`) with `in-flight-once:<device>` at `_execute_once` / after the first streamed
   token; tests for 1–4 plus "fail-fast still fires for a non-helper error".

### S2 — server keeps the token: retry the step with the helper masked out (C++; ~2 days)
Instead of an in-op host fallback (ggml changes), handle it in the server update loop: when `llama_decode`
returns 2 and `ffn_runtime.failed()` names a helper, (1) `apply_policy(mask & ~helper.layer_mask)` (the dormant
host share holds the weights; a helper with mask 0 is off), (2) `llama_memory_seq_rm` for the positions of the
failed ubatch in every slot of the batch (the step's tokens were never sampled, so a retry is exact), (3) retry
the same batch once host-side, (4) log `S41SERVERFFNFALLBACK{helper, slots, tokens}`. No process death, no
co-tenant failure, zero tokens lost, penalty = one failed step + one retry. NOT available in remote-resident
mode (desktop omits phone-owned weights) — keep S1. In-op host fallback (`src/llama-graph.cpp` split FFN op)
stays as the alternative if the seq_rm retry proves unsafe for some memory types.

### S3 — re-admission (~1 day)
When a quarantined device re-enumerates with the pinned identity (kernel release, boot image, worker sha, USB
identity — the same checks the preflight does), start ONE bounded re-qualification probe (the coherence probe
against the current owner) after a cooldown; cap re-admissions per run (hysteresis). Events
`DEVICE_READMISSION_PROBE`, `DEVICE_READMITTED`.

### Hardware gates (eval_v2, ~40 min each, both phones)
- G1 kill the Pixel worker at ~50 % of the longest Qwen request → run completes, 14/14 outputs identical or
  near-tie, host energy still below baseline, `REQUEST_RECOVERED` present. (Run it once BEFORE S1 as the "before"
  figure: expect run FAIL.)
- G2 OP15 loss (with the user present: unplug or `adb reboot`) → same as G1 for the primary phone.
- G3 re-plug → `DEVICE_READMITTED`, phone calls resume, saving recovers.

## P2. Automatic demotion / re-admission — a device that gets slow is switched off by evidence

Exists: monitored verdicts are rejected when the elimination test fails (`_reject_verdict`). Add (~1.5 days):
1. **Device drift detector as evidence context.** Per device × composition, compare the windowed RPC p50 with the
   qualified receipt bound; if it exceeds bound × (1 + δ) for K windows, invalidate that composition's verdict →
   the coherence probe re-runs (same code path as the initial probe). Mirror of the "measurement-context reset"
   already used for external desktop activity.
2. **Re-admission after cooldown** (shared with P1-S3): one bounded probe per cooldown period.
3. **Device-set timeline in RESULT** (`device_sets_by_composition`: set, verdict, reason, since_us) for the report
   figure "which devices assisted which batch sizes over time".

Hardware gates (eval_v2, ~40 min each):
- H1 degraded Pixel from the start: worker with the DVFS fix OFF (38 / 61 / 88 ms per layer) → expect the Pixel
  dropped for B ≥ 2 compositions (`NOT_IMPROVED` / `LATENCY_BOUND_EXCEEDED`), host energy within noise of
  OP15-only; no hard-coded rule involved.
- H2 mid-run degradation: CPU stress on the Pixel from t = 600 s → demotion event; stress off at 1,200 s →
  re-admission.

## P2b. Maximize per-phone utilization (user direction 2026-09-25: "maximize the utilization of each phone")

### What the code map says (read-only pass, 2026-09-25)
- A phone holds WHOLE layers but only a column SUFFIX of each (`native/ffn_shard_gguf.py`); `split_fraction` = the
  column share; all helpers get the SAME column count (`server.cpp::apply_policy`); ownership of a layer by a
  helper is fixed at server launch (`S41_SERVER_FFN_HELPER<k>_LAYER_MASK`, `eval_cb` routes by launch mask) —
  the runtime control can only switch a phone's own layers off (host computes them), never move a layer to the
  other phone.
- **Qwen: every phone-assistable layer is already offloaded at 100 %.** The desktop runs layers 24–39 on the GPU;
  only CPU-parent layers 0–23 can be assisted (`catalog.py:1146`, and the remote part is injected through the
  server's eval callback on host-memory tensors, `src/llama-graph.cpp::build_dense_ffn_split`
  "ffn_phone_partial"). OP15 owns 0–17 (its full-width ceiling: 3 HTP sessions × 6 F16 layers = 3.2 GB),
  the Pixel 18–23 (packing script hard-codes `range(18, 24)`). Layers execute in order, so the two phones are
  busy one after the other inside a token: the Pixel's 3.3 % is 6/24 of the phone chain, not idle capacity.
- **Gemma:** no Pixel shard (packing supports Qwen's Q4_K/Q6_K rows + SwiGLU only; Gemma is Q4_0-origin +
  GeGLU); OP15 holds 24 layers (a 26-layer shard set exists unused).
- **Where the two-phone arm's 94.4 kJ goes** (1,833 s): desktop idle floor ≈ 27.6 W × 1,833 s ≈ 51 kJ (fixed by
  the arrival span); GPU "running state" during decode ≈ 34 kJ (30.6 W average vs ≈ 12 W idle, while the GPU
  computes only 16 of 40 Qwen layers ≈ 40 ms of a ≈ 530 ms token); CPU active ≈ 10 kJ (attention of 24 layers
  ≈ 175 ms/token). Phone chain ≈ 265 ms/token (OP15 18 × 10.5 + Pixel 6 × 12.7 ms).
- Evidence keying hazard for any layer-share policy: `_policy_identity` omits `device_layer_masks` (two layouts
  with the same union pool evidence) and the server-policy group key ignores the Pixel; TCP HELLO requires
  `hello.layer_mask ⊇ --layers` while `apply_policy` connects with the owned subset (a partial first policy fails
  the handshake) — both must be fixed before any Pixel re-layout.

### Levers, by expected payoff (measure before coding)
| lever | what | payoff / risk | cost |
| --- | --- | --- | --- |
| U0 measurement | 1 h microbench on the rig: GPU + CPU power vs. (a) idle, (b) phone-assisted decode with 16 / 8 / 0 layers on the GPU (`--override-tensor 'blk\.(2[4-9]|3[0-9])\.ffn_.*=CPU'` keeps attention on the GPU, moves FFN weights to the host share), (c) all-CPU+phone decode with the GPU asleep | decides U1 vs U2; the 34 kJ GPU running-state energy is the largest addressable block | 1 h rig, no code |
| U1 more assistable layers | attention on the GPU for all 40 layers, FFN of 24–39 on host share → phones (Pixel shard 24–39: 16 × 158 MB = 2.5 GB, fits); needs: eval_cb to read/write GPU-resident activations (`ggml_backend_tensor_get/set`), catalog CPU-parent rule → "FFN-on-host" rule, generalized packing script + 5 Pixel receipts + calibration | phones busy ≈ 90 % of the token; host CPU attention → GPU; GPU energy only drops if its power falls when lightly loaded (U0 tells) | 3–4 days + rig |
| U2 GPU asleep during phone decode | all 40 layers CPU-parent while phones hold all FFN (OP15 18 + Pixel 22); GPU used for prefill only | −20 W × decode time if the GPU reaches P8 between prefill bursts; tokens slower (CPU attention 40 layers ≈ 290 ms + phone chain ≈ 470 ms) — arrival-bound runs hide most of it, long-tail requests don't | 2 days + rig; only if U0 shows the GPU really drops |
| U3 Pixel Gemma shard | packing for Q4_0 rows + GeGLU in the Pixel worker; Pixel takes Gemma layers beyond OP15's 24 | small (few layers) unless combined with U1 for Gemma | 1–2 days |
| U4 layer share as a policy dimension | `device_layer_masks` in `_policy_identity`/group key, per-(device, mask) drops, share axis in `_device_set_policy`; C++: TCP helpers connect with their full mask; optional true migration (per-helper masks in the runtime control, `eval_cb` by applied mask) | only useful once phones have spare layers to trade (after U1) | 2 days (option A) / 4 days (option B) |
| U5 per-helper column fractions | C++ runtime control carries per-helper columns | lets a slow phone take a narrower slice instead of being dropped | 1–2 days |
| batching | denser traces (P3-T1) raise rows per phone call at ≈ constant call time (weight-read-bound) | the cheapest utilization gain there is | builder only |

Recommendation: U0 tonight/tomorrow morning (no code), then U1 if the GPU power drops with fewer resident
layers, else U2; U4/U5 after; U3 in parallel if the Pixel worker supports GeGLU cheaply.

## P3. Trace families — the same scheduler on different workloads (builder exists; ~2 h rig per pair)

| trace | knobs | what it tests |
| --- | --- | --- |
| T1 `longtail_load_v1` | `--arrival-scale 0.5`, denser window, `--min-same-model-overlaps ≥ 12` | batch 2–4 phone calls (coalesced receipts cover ≤ 6 Qwen rows / ≤ 2 Gemma rows), dispatcher under load |
| T2 `longtail_cold_v1` | Gemma-heavy window (`--small-model-share`, window offset) | re-provisioning follows the cold model, OP15-only Gemma policy |
| T3 `longtail_prompt_v1` | `--min-input 2000 --prompt-cap 6000` (contexts are 24k / 32k) | prefill-heavy: decode-only offload must not slow prefill; admission of long shapes |
| T4 (optional, ~2 days) | third large model (needs shards, calibration, VRAM budget) | 3-model residency |

Each: baseline + OP15+Pixel, single runs → a 2 × N table; T1 first.

## P4. Device onboarding — from ~1 day to < 1 hour, scripted and fail-closed (~2 days)

`research_dev/scheduler/onboarding/onboard_helper.py DESCRIPTOR.json` where the descriptor names serial,
transport (`adb-tcp` | `functionfs`), worker binary + library dir, shard quantum / layer range, power model.
Steps (each writes a receipt; any failure → device not admissible):
1. push + hash worker/libs/shards; 2. worker smoke + token-identity receipt vs the desktop reference
(generalize `tools/qualify_pixel_server.py`); 3. latency calibration B1/B2/B4 → cost receipt + `prior:<device>:ffn`
kernel; 4. transport identity + evidence bundle (generalize `prepare_campaign_eval.py`); 5. rig/models patch
(co-helper entry, layer range); 6. `launch.py --preflight-only` dry run.
Demo: onboard a THIRD phone over adb-tcp (OP11 — `tools/qualify_op11_tcp.py` exists — or OP12) and run one
eval_v2 three-phone arm. Success = admissible in < 1 h, policy admits or refuses it by evidence.

## P5 (deferred). Repeats: eval_v2 × 3 per arm for confidence intervals (~6 h rig).

## Schedule (rig-bound; code and rig overlap)

| when | rig | code |
| --- | --- | --- |
| today after the chain (~18:00 UTC) | H1 degraded-Pixel arm (no code change); G1 "before" figure (expected FAIL) | — |
| day 1 | G1 (after) , H2 | P1-S1 + P2-1/2/3 in an isolated copy, tests, merge, sync stage+deploy |
| day 2 | T1, T2 pairs | P1-S2 server fallback (C++), rebuild, re-materialize transport identity |
| day 3 | G1/G2/G3 with S2 | P4 onboarding driver |
| day 4 | third-phone arm; T3 | P1-S3 / polish, report figures |

## Novelty — what these fixes add, honestly

| item | novelty | how to state it |
| --- | --- | --- |
| P1 + P2 together | **yes, as a system property**: "safe elasticity" — heterogeneous, untrusted edge helpers are admitted, demoted and re-admitted purely from measured evidence (sequential test with bounds + hysteresis), and offload can never fail a request or change its output, even when a device disappears or slows mid-token. Edge-assisted serving papers show speed/energy; few show correctness-preserving fallback under device loss with a decision procedure that is fail-closed. | one figure: device-set timeline with kill/slow/recover events + identical-output checkmarks + energy trace |
| P1-S2 (in-flight host fallback) | technical novelty is modest (host has the weights), but it is what makes "no token lost" true; without it the claim is "recovered by re-execution" | state the token-loss = 0 property and the latency penalty |
| P3 | not novelty — generality evidence the claim needs | 2 × N table across trace families |
| P4 | modest: a **qualification contract** — every device enters through measured transport receipts by payload size, identity pins and a calibrated prior, never vendor specs; it is what makes "any device" credible | onboarding time + a third device admitted/refused by evidence |
| P5 | none; statistics | CIs |

The strongest reviewer-facing outcome of this plan is P1+P2 with the mid-run kill/slow/recover figure. P3/P4 are
the evidence that the property generalizes.
