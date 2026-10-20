# Phase 2 target A: remote-resident OnePlus owner (2026-10-09)

Target: Qwen3-32B Q8_0 (`qwen3-32b-q8_0`, `/home/zhihao/models/Qwen3-32B-Q8_0.gguf`). The OnePlus 15 (`op15-phone`) owns
the dense FFN weights of layers 0-20 in three HTP sessions (HTP0 0-6, HTP1 7-13, HTP2 14-20; `GGML_HEXAGON_NDEV=3`,
one session per worker). The desktop server never loads them (`S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK`); prefill and
decode of those layers run on the phone over FunctionFS. No Pixel (target B, below, is a separate opt-in).

Everything is opt-in: without `remote_resident_owner` in the campaign no catalog, launch command, preflight check,
scheduler state or RESULT key changes.

## What the opt-in does

| stage | behaviour |
|---|---|
| catalog (`catalog.py`, or `launch.py` for a prematerialized catalog) | adds `<desktop control>:remote-owner` (CALIBRATION_PENDING): the desktop control plus `remote_resident_ffn_v1` built from the campaign's `fixed_phone_residency` (session geometry = the shard packer's), the model's FFN shard index and the primary FFN family's phone parameters (assisted-copy `ffn_host_share_*` keys dropped). Fails when the transport identity does not qualify a whole-ubatch payload |
| preflight | `remote-resident-route:<model>` BLOCKED unless the parent is generated and held back only by owner readiness (`REMOTE_RESIDENT_OWNER_NOT_READY:*`, the residency it brings) or calibration reasons (`ROUTE_NOT_QUALIFIED`, `ENERGY_UNKNOWN`, `COLD_RESIDENCY_BREAK_EVEN`, `SLO_UPPER_BOUND`); `remote-resident-transport:<model>` BLOCKED unless the identity qualifies the payload |
| scheduler | `configure_remote_resident_owner(model)`: the fixed assignment must equal the declared owners. Every selection of the model (dispatch, placement epochs, desktop fallbacks) returns the remote parent, admitted as a calibration sample (requests are submitted as `calibration`); not admissible -> `UnifiedScheduleError` with the reasons. The full-weight route is refused at commit (also a startup preload) and failures have no recovery route |
| runner | fixed-residency preload loads the owners (phone transitions only); the model's arrivals wait until it is READY (all sessions), at most `owner_ready_timeout_s`, then the run fails. RESULT gets `remote_resident_owner` (pin, owner readiness, ledger) |
| ledger | every reduced launch: the server's omission proof, VmRSS, host MemAvailable; `remote_resident_accounting` per launch; `desktop_ram_saving_bytes` = proof-verified omitted bytes (0 unless every launch proves the plan) |

## Operator fields (Q8_0, OnePlus owner of layers 0-20)

Masks: HTP0 layers 0-6 = 127, HTP1 7-13 = 16256, HTP2 14-20 = 2080768 (union 2097151). Q8_0 gate/up/down per layer
3 x 5120 x 25600 x 34/32 = 417,792,000 B; per session 2,924,544,000 B; omitted 8,773,632,000 B.

campaign.json (add; `<Q8_0_SHA256>` is the full parent `sha256:2c50eb8a...`):

```json
"selection_mode": "energy-aware",
"fixed_phone_residency": {
  "mode": "fixed",
  "reference_id": "q32-q8-op15-owner-0-20",
  "assignments": [
    {"session_id": "HTP0", "artifact_sha256": "<Q8_0_SHA256>", "layer_mask": 127, "maximum_columns": 25600},
    {"session_id": "HTP1", "artifact_sha256": "<Q8_0_SHA256>", "layer_mask": 16256, "maximum_columns": 25600},
    {"session_id": "HTP2", "artifact_sha256": "<Q8_0_SHA256>", "layer_mask": 2080768, "maximum_columns": 25600}
  ]
},
"remote_resident_owner": {"model_id": "qwen3-32b-q8_0", "owner_ready_timeout_s": 1800}
```

Not allowed with it: `selection_mode: "desktop-baseline"`, a `startup_desktop_parents` row of the model,
`phone_resident_model_reprovisioning`, `phone_htp_memory_caps`, `elastic_phones.helper_loss_recovery: "mask_out"`.

models.json, the 32B row (in addition to the Phase 1 fields):

```json
"phone_ffn_shard_index_path": "<dir>/FFN_SHARDS.json",
"phone_ffn_shard_directory": "/data/local/tmp/<q8-owner-shards>",
"runtime_parameters": {"ubatch_size": 512, "...": "..."},
"phone_adapter_parameters": {"ffn_activation": "swiglu", "ffn_timeout_ms": 120000, "ffn_fast_hash": 1, "...": "..."}
```

- FFN_SHARDS.json: three full-width rows (`session_id` HTP0/1/2, `layer_mask` hex 7f / 3f80 / 1fc000, `columns` =
  `n_ff` = 25600, `weight_type` Q8_0, FFN kernel, not fused), e.g. `ffn_shard_gguf.py ... --shard HTP0=0-6:25600
  --shard HTP1=7-13:25600 --shard HTP2=14-20:25600`.
- no `helper_phone_ffn_shards` (that is target B).
- `ffn_fast_hash: 1` is optional: it adds `S41_SERVER_FFN_FAST_HASH=1` (FunctionFS only; the phone worker and router of
  the bundle must support flag 4, e.g. `/data/local/tmp/s43-fastcall-v1`). Omit it for the plain hash.
- phone session discovery: HTP0..HTP2 on `op15-phone`, ready, `supported_data_types` with Q8_0, `maximum_columns` >=
  25600 with a `column_quantum` dividing it, `resident_memory_limit_bytes` >= 2,924,544,000, layers 0-20 supported.

Q4_0 variant (two sessions, 221,184,000 B per layer): HTP0 layers 0-10 = 2047, HTP1 11-20 = 2095104, `artifact_sha256`
the Q4_0 parent (`sha256:4b3aaa12...`), `model_id` `qwen3-32b-q4_0`, shards `weight_type` Q4_0.

## Transport identity the scheduler expects

The owner computes prefill, so one call carries `ffn_n_embd x ffn_max_tokens x 2` = 5120 x 512 x 2 = **5,242,880
bytes** (`S41_SERVER_FFN_USB_MAX_PAYLOAD_BYTES`). Route generation now requires a qualified link of at least this size
for remote-resident parents whatever the batch plan (split-row normally qualifies one row only), and the catalog/
preflight fail with "needs 5242880 bytes per call ... the transport identity qualifies at most N bytes".

A qualification script must produce, for the identity's `transport_generation` (`functionfs-*`) and one allocator in
`qualified_allocators`, receipts (schema `s41_ffs_dmabuf_transport_v2`, `device_mode` dmabuf, `host_mode` async,
`reset_recoveries` 0) with one configuration (`host_allocator`, `configured_queue_depth`, `queue_depth`,
`usbfs_available_bytes`, `slot_safety_bytes` all equal) at payload P = 5,242,880:

- duplex: `request_bytes` = `response_bytes` = P;
- h2d: `request_bytes` = P, `response_bytes` < P (with `h2d_payload_MBps` >= 1);
- d2h: `response_bytes` = P, `request_bytes` < P (with `d2h_payload_MBps` >= 1);
- `usbfs_available_bytes` >= (P x `ffn_transport_slot_payload_multiplier` + `slot_safety_bytes`) x `queue_depth`
  (multiplier 4, safety 65,536, depth 4: 84,148,224 B, so usbfs_memory_mb >= 81).

Their sha256 values go into the identity's `receipt_sha256s`, the files into the evidence
`transport_qualification_directories`; the identity's `software_identity` must name the deployed host server
(`host_binary_sha256` + `host_dependency_sha256:*`) and phone bundle (`phone_worker_sha256`, `phone_session_sha256`,
`phone_resident_workers_sha256`, `phone_resident_router_sha256`) as for FUSED3/ANCHOR. FUSED3 receipts at smaller
payloads remain valid for the Phase 1 arms.

## Changed files (target A)

| file | change |
|---|---|
| `configuration/campaign.py`, `config.py` | `RemoteResidentOwnerConfiguration`, campaign field + cross-checks |
| `adapters/catalog_materialization.py`, `adapters/__init__.py` | `materialize_remote_resident_owner`, `remote_resident_payload_qualification` |
| `_internal/phone_shards.py` | shared `phone_shard_resident_bytes` / `phone_shard_resident_geometry_sha256` (packer unchanged) |
| `campaigns/burstgpt/catalog.py`, `launch.py` | `remote_resident_owner_catalog`; launch passes `--remote-resident-owner-json` to runner and preflight |
| `adapters/preflight.py`, `campaigns/burstgpt/preflight.py` | remote-resident route and transport checks |
| `_unified/phone_residency_ops/fixed.py`, `_unified/phone_residency.py`, `scheduler.py`, `_unified/common.py` | the pin and its configuration |
| `_unified/automated_selection_ops/adaptive.py`, `calibration.py` | pinned selection |
| `_unified/automated_requests_ops/submission.py`, `commit.py`, `failure.py` | calibration submission, commit guard, no recovery |
| `_unified/phone_residency_ops/offline_inputs.py` | a fixed assignment may name a subset of the discovered sessions (crashed before) |
| `_internal/route_generation/costing_parameters.py`, `feasibility.py` | whole-ubatch payload qualification for remote parents |
| `adapters/phone_transport.py`, `phone_helpers.py`, `contracts.py`, `_unified/common.py` | opt-in `ffn_fast_hash` |
| `adapters/heterogeneous_rig.py`, `heterogeneous_rig_ops/transitions.py`, `host_runtime.py` | launch records (proof, RSS, MemAvailable) |
| `_internal/runtime_resources.py` | `remote_resident_run_ledger` |
| `campaigns/burstgpt/runner.py`, `arguments.py` | pin, owner-ready wait, phone-only preload check, RESULT section |
| `_internal/adaptive_decode_planning.py`, `_unified/adaptive_decode_control.py` | no adaptive policy over a remote group; prefill yield refuses a remote joiner |
| tests | `test_remote_resident_materialization`, `test_remote_resident_trace_arm`, `test_remote_resident_accounting` (ledger) |

## Remaining gaps (target A)

- No hardware run yet; the identity above must be qualified first.
- The first arrival of the model waits for the owner preload: its TTFT and the paid interval include the load.
- Costing still has no phone compute term for remote layers: predicted finish/SLO values of the remote parent are
  unreliable (the pin admits `SLO_UPPER_BOUND`).
- In-flight owner loss is not recoverable: the request fails, the run stops (fail-fast).
- The scheduler-level end-to-end tests use the F16 replay rig (three sessions); Q8_0 is covered by materialization and
  route generation tests on the synthetic Q8_0 harness, not by a Q8_0 preload replay.
- `remote_resident_gate.py` is unchanged (still F16-only); the trace arm does not use it.

# Target B: Pixel static decode co-helper (opt-in on top of A)

The Pixel 10 Pro serves decode of layers 21-40 (the desktop keeps those weights; prefill of those layers stays on the
desktop) with `S41_SERVER_FFN_HELPER1_MAX_TOKENS` = its `max_tokens` (4) while the owner takes 512-row ubatches.

Fields: campaign `"remote_resident_owner": {"model_id": "qwen3-32b-q8_0", "owner_ready_timeout_s": 1800,
"static_co_helpers": true}`; models.json `helper_phone_ffn_shards` for the Pixel (one full-width shard of layers 21-40)
and the rig `helper_phones` row (`max_tokens` 4, column quantum dividing 25600) with qualified helper evidence, as in
the Phase 1 two-phone arms. Without `static_co_helpers` a model with co-helper shards is refused for the arm.

| piece | change |
|---|---|
| materialization | the remote parent takes the primary family's static co-helpers (participants, resources) and `remote_resident_co_helper_decode=1`; elastic (`per_helper_outage`) or owner-overlapping co-helpers are refused |
| route generation | with the flag, a co-helper may cap its rows below the ubatch; its binding row carries `max_tokens` |
| launch | `PhoneHelperBinding.max_tokens` -> `S41_SERVER_FFN_HELPER<k>_MAX_TOKENS` (needs runtime control and a larger shared batch) |
| decode policy | `remote_resident_co_helper_policy`: the co-helper layers at full width from the first decode token, never a remote layer, applied as the static control |
| proof | owner calls on remote layers (every row) plus co-helper calls on the policy layers within the cap after the control; the Pixel session is proven when the policy applied before the last token |
| mask-out | the rig never masks a helper out of a remote parent (`REMOTE_RESIDENT_PARENT`); the campaign refuses `mask_out` |

Changed files (B): `_internal/plan_contracts/co_helpers.py`, `adapters/phone_helpers.py`, `adapters/ticket.py`,
`adapters/llama_server.py`, `adapters/llama_server_ops/proofs.py`, `adapters/heterogeneous_rig_ops/helper_masks.py`,
`_internal/route_generation/remote_resident.py`, `costing_parameters.py`, `adapters/catalog_materialization.py`,
`configuration/campaign.py`, `config.py`, `campaigns/burstgpt/catalog.py`; tests `test_remote_resident_co_helper_decode`.

Gaps (B): a thermally deferred or lost Pixel fails the request (no mask-out, no recovery); no hardware run; the C++
per-helper cap is the uncommitted server change in this tree.
