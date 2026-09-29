# Two-phone Stage A: gaps 1-3 closed in code, gap 4 designed, gaps 5-6 fail-closed

2026-09-24. Scratch deliverable. Nothing was committed, and nothing ran on a phone, over adb or on the rig.
This builds on `reports/20260924-two-phone-readiness/README.md`, whose `TWO_PHONE*.diff` are already
merged into main: the N-helper llama-server, the rig `helper_phones`, the `phone_helpers` launch hook,
the adb-tcp session and the M3 gate. That README's section 6 has the Stage A design and the ranked gap
list this work follows.

| Deliverable | Path (this directory) |
| --- | --- |
| Scheduler patch, 23 files: 19 edited, 4 new | `TWO_PHONE_GAPS.diff`, against the live main tree; `git apply --check` passes with stdin closed |
| Regenerate the diff and check it applies (refuses if main changed a patched file) | `make_diff.sh` |
| Progress log | `PROGRESS.md` |
| Trees | `base2/` = main at rebase time (after the dispatcher #2/#5 and server-probe merges); `root2/` = `base2/` + the patch |
| Test logs, experiment scripts | `logs/` (full suites, gap tests on the base tree), `scripts/` (route experiments, golden digests) |

## 1. Implemented vs designed

| Gap | State | Where |
| --- | --- | --- |
| 1 Catalog and route generation for the co-helper | IMPLEMENTED + tested; the campaign catalog validates the co-helper but withholds it until gaps 5-6 | `_internal/plan_contracts/co_helpers.py` (new), `adapters/catalog_materialization.py`, `_internal/capability_contracts/catalog.py`, `_internal/route_generation/{patterns,envelopes,costing_parameters}.py`, `campaigns/burstgpt/catalog.py` |
| 2 `phone_helpers` through the plan, dormant contract, residency, ticket and launch | IMPLEMENTED + tested end to end with fake executors | `_unified/common.py`, `adapters/contracts.py` (whitelists), `_unified/automated_selection_ops/{dormant,attachment}.py`, `adapters/residency.py`, `adapters/ticket.py`, `adapters/llama_server_contracts.py` |
| 3 Adaptive policies on the co-helper grid, per-device masks, per-device proofs | IMPLEMENTED + tested | `_internal/adaptive_decode_contracts.py`, `_internal/adaptive_decode_planning.py`, `adapters/llama_server_ops/proofs.py` |
| 4 Pixel worker lifecycle in a campaign | DESIGNED. The hook is written and tested with fakes but not wired in. No stop policy is implemented (user decision). | `adapters/co_helper_lifecycle.py` (new), section 4 |
| 5 Pixel transport identity and receipts | NOT DONE (hardware). Fail-closed: the Pixel executor is `SHADOW` unless listed in `qualified_helper_phone_ids`, and preflight reports BLOCKED. | section 5 |
| 6 Pixel kernel, link and power evidence | NOT DONE (hardware). Fail-closed: the campaign catalog withholds the co-helper, and preflight reports BLOCKED. | section 5 |
| Launch of a real two-phone run | Still refused (`launch.py`, unchanged) | |

## 2. Design (Stage A: static co-helper)

OP15 stays the scheduled primary phone: its sessions, shards, adaptive policies and re-provisioning are
unchanged. The Pixel is a participant of one model's phone-assisted composites, with its own executor,
resources and operator assignments. It is not an OP15 session. Its shard never enters `phone_shards`,
the OP15 residency layout, helper preparation or COW replacement, so the Stage B blockers
(`ready_plan` single parent, "sessions span multiple devices") are never reached.

**Declaration** (`phone_co_helpers_v1`, a canonical JSON composite adapter parameter;
`RuntimeCoHelperDeclaration`):
- It carries the primary phone's launch label and serial, plus one row per co-helper, in helper order.
  Helper order fixes the request-id range: helper `k` numbers its calls from `1 + k*2^24`.
- Each row holds: device, serial, label, proof session id (`PIXEL0`), static layer mask, column
  quantum, max tokens, shard sha256 and bytes, and the adb-tcp transport parameters. The transport is
  its own `adb forward` on a fixed host port.
- Rows are refused if they overlap or share identities, if the forward belongs to another serial, or if
  a quantum is not a multiple of 32.

**Catalog** (`adapters/catalog_materialization.py`):
- `RuntimePhysicalTopology.helper_phones`: new `RuntimeHelperPhoneTopology` rows (memory, transport
  and compute resources). Helper compute must not be shared with any other phone.
- `base_executor_capabilities`: one extra FFN-only executor per helper phone, `physical:<device>`.
  - It is never a generic placement target: operator, layer and whole-model placement and coordinated
    families are all off.
  - It is `QUALIFIED` only when listed in `qualified_helper_phone_ids`, and `SHADOW` otherwise. It is
    never qualified from OP15's receipts.
- `RuntimeModelEndpointCapability.co_helpers`: the declaration. It must use decode-boundary control on
  CPU-parent layers only (`layer_mask >> desktop_gpu_first_layer == 0`), because the server splits the
  FFN of CPU layers.
- Phone families (`operator_offload` and `operator_split` for the GPU and CPU parents) gain:
  - the co-helper as a participant, with its own `participant_resource_ids` and resources;
  - the declaration;
  - `ffn_column_quantum = lcm(primary, co-helper)`: 4352 for OP15 2176 plus Pixel 4352;
  - `operator_ids` that exclude co-helper layers. These are OP15's candidates only.
- Model transitions never list a co-helper in `prepares_device_ids`: a cold co-helper is rejected,
  never prepared.
- Catalog validation (`_validate_composite_co_helpers`): co-helpers must be FFN phone participants,
  distinct from the coordinator and from the primary helper.

**Route generation**:
- `patterns.py` assigns every FFN operator in a co-helper's layers to that co-helper. Split families
  get `(base, pixel, fraction)` and offload families get `(pixel, None, 0)`. Primary-phone candidates
  exclude those layers. Two-phone composites only emit decode-resident-envelope patterns.
- `envelopes.py`: the envelope's minimum quantum is the LCM, and OP15 envelopes never contain
  co-helper layers.
- `costing_parameters.py`:
  - Decode-resident plans get `ffn_resident_layer_mask = OP15 envelope | co-helper layers` and
    `phone_helpers` (the launch binding, first row = the ticket's phone with the OP15 envelope mask).
  - The co-helper's links are excluded when deriving the primary FunctionFS transport parameters.
  - A co-helper whose `max_tokens` differs from the plan's `ffn_max_tokens` rejects the candidate
    (`TRANSPORT_PROFILE_INCOMPLETE`), because every client's HELLO must match the server.

**Plan -> launch** (gap 2):
- `phone_helpers` is whitelisted in both dormant key sets (`_unified/common.py`, `adapters/contracts.py`).
  `dormant_phone_ffn_parameters` checks that its first row is the contract's phone and that the union
  equals `ffn_resident_layer_mask`.
- The desktop parent's dormant contract therefore carries both phones.
- `_dormant_phone_ffn_storage_superset` grows only the primary row, never over co-helper layers.
- `physical_residency_parameters_match` and `_dormant_phone_ffn_runtime_supports` accept a live server
  whose helpers are the same devices, labels, serials and transports in the same order, with each
  requested owner a subset of the live owner (`phone_helpers_support`).
- `ticket.py::_validate_phone_helpers`, for phone commands and desktop commands with a helper envelope,
  requires:
  - the first row is the contract phone and the `phone_device_id`;
  - the first row equals the OP15 shard mask;
  - the union equals the resident mask;
  - every helper is exactly one participant.
  The shard-geometry check accepts `shard mask | co-helper layers == resident mask`.
- `llama_server_launch_contract` excludes co-helper devices from the GPU-layer derivation. The existing
  hook turns `phone_helpers` into `S41_SERVER_FFN_HELPERS=2` plus `HELPER<k>_*`.

**Adaptive** (gap 3):
- `AdaptiveDecodePolicy.device_layer_masks` holds `(device, active owned layers)` in helper order. It is
  normalized to the policy's layers, so narrowing a policy with `replace()` also narrows the owners.
  Every active layer needs an owner, and baselines cannot carry one.
- The policy hash and JSON include the field only when it is non-empty, so single-phone policy hashes
  are unchanged. This is tested against a hash computed on base.
- `_policy` and `_envelope_subpolicies` fill the field from the plan's `phone_helpers`. Sub-policies use
  the envelope quantum (the LCM), so two-phone widths are 25/50/75/100 %.

**Proofs** (gap 3):
- `_execution_session_proofs` adds one proof shard per co-helper: session `PIXEL0`, endpoint
  `adb-tcp://<serial>/PIXEL0`, geometry = shard sha256, generation 1.
- It checks every call's request id against its owning helper's range. Co-helper shards must receive
  calls like OP15's.
- A proof without a co-helper declaration is unchanged.

**Campaign catalog** (`campaigns/burstgpt/catalog.py`):
- `helper_phone_co_helpers(rig, model, manifest)` builds the declaration from the rig helper phone and
  its `helper_phone_ffn_shards` index. It requires:
  - exactly one full-width shard of this artifact;
  - a fixed host forward port;
  - no overlap between the co-helper layers and the primary phone's shard index.
  The label and session come from the device id: `pixel10pro` and `PIXEL10PRO0`.
- `main()` validates this, plus the CPU-parent layer check, at resolve time. It then withholds the
  co-helper from the catalog and prints `withheld_co_helpers` in its summary. No Pixel kernel, link or
  power profile exists (gap 6), so its routes could be neither placed nor priced.

## 3. Tests

`tests/test_two_phone_gaps.py` has 29 tests. It uses `tests/two_phone_harness.py`, a synthetic rig
whose catalog is built by the real materialization functions:
- 8 layers and 128 columns;
- the desktop parent puts layers 0-5 on the CPU and 6-7 on the GPU;
- OP15 has quantum 16 and may assist layers 0-3;
- the Pixel has quantum 32 and owns layers 4-5. This mirrors 2176/4352 on 17408.

| Gap | Tests (class) | Result on base2 (main + test files only) | Result on root2 |
| --- | --- | --- | --- |
| contract | `CoHelperContractTests` (3) | ERROR: no `co_helpers` module | PASS |
| 1 | `CatalogCoHelperTests` (3 + 1 guard), `CampaignCoHelperTests` (2) | ERROR: no module / no `helper_phone_co_helpers` | PASS |
| 1 + 3 | `TwoPhoneRouteTests` (5): Pixel layers assigned to the Pixel, union mask, `phone_helpers`, quantum 32, fractions 0/25/50/75/100 %, policies 32/64/96/128 columns with per-device masks, a SHADOW Pixel gives no two-phone policy, a cold Pixel is rejected and not prepared, batch capacity must match | ERROR: no module | PASS |
| 2 | `TwoPhoneTicketLaunchTests` (3): adaptive ticket -> `validate_physical_execution_command` -> launch env `HELPERS=2`, `HELPER0` functionfs 15, `HELPER1` tcp `127.0.0.1:26991` 48, union 63, `gpu_layers=2`; the energy-aware desktop parent's dormant contract gives the same env; tampered binding or missing participant refused | ERROR: no module | PASS |
| 2 | `DormantContractTests` (3) | FAIL: base rejects the dormant contract (`phone_helpers` not whitelisted); base residency match returns False for the per-device subset | PASS |
| 3 | `AdaptivePolicyOwnerTests` (1 + 1 guard), `TwoPhoneProofTests` (2 + 1 guard) | ERROR: `device_layer_masks` unknown; base proofs fail on a Pixel call with "no unique shard owner" | PASS |
| 4 | `CoHelperLifecycleHookTests` (3): no stop policy means no start; a plugged fake policy stops; worker must match the declaration | ERROR: no module | PASS |
| single phone | `SinglePhoneUnchangedTests`, plus 4 guards above | PASS | PASS |

`SinglePhoneUnchangedTests` pins digests computed on the base tree for the single-phone catalog, the
candidate set, the adaptive policies, the adaptive-decode and energy-aware plans, and the launch
environments. They are identical on base, base2, root and root2.

Other checks:
- `tests/test_two_phone_helpers.py`: one assertion was updated for the two new BLOCKED preflight rows.
  All pass.
- pyflakes: clean on all 23 files.
- Full scheduler suite: see section 7. The adaptive, coherence, server-probe, dispatch-policy and
  two-phone test files all pass on root2: 157 tests, 1 skip.

## 4. Gap 4: lifecycle hook (designed only)

`adapters/co_helper_lifecycle.py::CoHelperLifecycle(declaration, sessions, stop_policy)`:
- **Construction**: each `AdbTcpPhoneWorkerSession` must match the catalog declaration: device, serial,
  layers, quantum, max tokens, phone port and fixed forward port.
- **`start_trace(log_dir)`**: refused without a stop policy. Otherwise it runs preflight and start for
  each co-helper, then checks that the live forward equals the declared one.
- **`residency_observations(manifest)`**: `hot` `ModelResidencyObservation` rows for started
  co-helpers. The snapshot builder must add them, or route generation rejects the two-phone routes.
- **`split_session_proofs(proofs)`**: primary rows go to OP15's owner and co-helper rows stay with the
  lifecycle.
- **`end_trace(served_calls_by_device)`**: delegates to `CoHelperStopPolicy.stop(session, served_calls=)`.

Either stop option plugs in as a `CoHelperStopPolicy`, and neither is implemented:
- **(a) Worker shutdown message.** A new protocol message in `ffn-split-worker.cpp`'s TCP loop (Pixel
  worker code, owned by the other agent). The policy connects once more and sends it after the
  server's client left.
- **(b) Opt-in idle SIGTERM.** `session.stop(allow_idle_signal=True)`, which already refuses while a
  client is connected.

Wiring points once the user decides (none are edited here):
1. `adapters/heterogeneous_rig.py` starts the lifecycle after preflight and ends it at `end_trace`.
2. The snapshot builder merges `residency_observations`, and gives the Pixel executor a telemetry state
   so a missing Pixel battery or thermal reading does not defer OP15 preparations (readiness gap 4).
3. `adapters/heterogeneous_rig_ops/observations.py` (about line 106) passes
   `execution_proof.phone_calls_by_session` through `split_session_proofs` before
   `direct_phone.record_execution_proof`. Today a `PIXEL0` row would reach OP15's session owner.
4. `heterogeneous_rig_ops/residency.py` records OP15's persistent residency from
   `phone_ffn_resident_contract`, which is now the union mask. It must record the primary row of
   `phone_helpers`.
5. `launch.py`: lift the helper-phone refusal only once 1-4 and gaps 5-6 are in place.

## 5. Remaining gaps and fail-closed behaviour

| Gap | What is missing | Today |
| --- | --- | --- |
| 4 | Lifecycle wiring (section 4) and the stop-policy decision | preflight `two-phone-dispatch` BLOCKED; `launch.py` refuses a real run with helper phones |
| 5 | Pixel `PhoneHelperTransportIdentity`: `server-token-identity` (new binary), `adb-forward-round-trip`, `scheduler-launched-session`; then pass the device in `qualified_helper_phone_ids` | preflight `helper-phone-transport-identity:<device>` BLOCKED; Pixel executor SHADOW -> no two-phone adaptive policy |
| 6 | Pixel kernel, link and power profile. Also an evidence input from which the campaign catalog adds the Pixel device, pool, kernel and links, `topology.helper_phones`, and `co_helpers` on the endpoint | preflight `helper-phone-cost-evidence:<device>` BLOCKED; catalog prints `withheld_co_helpers`, no Pixel rows |
| 8 | Per-helper runtime stats for the adaptive controller (optional) | aggregate stats; per-device accounting comes from proofs |
| Stage B | device-keyed residency/preparation/layout generations | not started |

Known properties of Stage A, for review:
- **Every phone route of the model uses both phones.** With a co-helper declared, every phone-assisted
  route of that model uses both phones. An unqualified or cold Pixel therefore disables phone assistance
  for that model entirely (fail-closed). An OP15-only arm for Qwen needs a catalog without the co-helper.
- **The dormant parent can carry a SHADOW Pixel.** The dormant desktop parent may carry the two-phone
  binding while the Pixel is still SHADOW, the same as OP15 today. The server then defers the Pixel
  connection until a policy targets it, which cannot happen without qualified policies.
- **`resident_model_identity_sha256` ignores co-helpers.** This is safe in Stage A, because a model's
  phone composites are either all one-phone or all two-phone in one catalog.
- **Fixed forward ports only.** The co-helper needs a fixed forward port (rig `forward_port` > 0),
  because its port is part of the plan and of the server environment.

## 6. Updated smoke plan

Each step waits for the user's authorization. The rules are those of the readiness plan: one run at a
time under the shared flock, adb server 5037 only, never kill an in-flight worker, new output directories.

0. **Read-only link check.** Unchanged from the readiness README. Additionally:
   - set the rig helper phone `forward_port` to a fixed free host port, e.g. 26991, instead of 0;
   - check that `/home/zhihao/s42-op11-qwen-shards-20260921-v1/qwen/FFN_SHARDS.json` has one shard
     with parent `sha256:d89e9e82...` (Qwen), layers 18-23, full width.
1. **Deploy and build.** As before, with `TWO_PHONE_GAPS.diff` applied after the merged two-phone diffs.
   Run `make_diff.sh` first. Run these tests plus pyflakes:
   - `test_two_phone_gaps`;
   - `test_two_phone_helpers`;
   - `test_two_phone_server_native` with `S42_LLAMA_BUILD_BIN`;
   - `test_adaptive_runtime` and `test_llama_server_adapter`.
2. **OP15 identity.** Unchanged.
3. **M3 mechanism gate.** Unchanged arms. This produces the gap-5 receipts and the per-helper SHAPE
   lines, which are raw material for gap 6.
4. **Resolve and preflight only, no run.** Run `launch.py ... --resolve-only`, then `--preflight-only`,
   with the rig and models additions. Expected:
   - the catalog summary lists `withheld_co_helpers: {"hot": sha256:...}`;
   - resolve fails loudly if the shard index, the forward port, CPU-parent layers or OP15 disjointness
     is wrong;
   - the preflight USB rows PASS;
   - `helper-phone-transport-identity`, `helper-phone-cost-evidence` and `two-phone-dispatch` are
     BLOCKED;
   - a real launch is refused.
5. **User decisions, then code.** In order:
   - choose the stop policy (gap 4 a or b) and wire section 4 items 1-4;
   - materialize the Pixel identity and pass `qualified_helper_phone_ids` (gap 5);
   - add the helper-phone evidence input and un-withhold the co-helper in `campaigns/burstgpt/catalog.py`
     main (gap 6).
   Each BLOCKED row flips only through its gap's code.
6. **dev_v2 trace.** As in the readiness plan, only after step 5 and a passed step 3. Pair it with an
   OP15-only treatment, which uses a catalog without the co-helper, and with the desktop baseline.
   Report Pixel energy as a separate assumption.

## 7. Full-suite results

`python3 -m unittest discover -s research_dev/scheduler/tests`, logs in `logs/suite-{base2,root2}.log`:

| Tree | Tests | Errors | Skipped |
| --- | ---: | ---: | ---: |
| base2 (main) | 1723 | 2 | 13 |
| root2 (main + patch) | 1752 (+29) | 2 | 13 |

The same 2 errors occur on both trees, and both are pre-existing:
- `test_resident_router_subset`: its setUpClass compiles `examples/layersplit/...`, which the scratch
  copy does not contain.
- `test_split_kv_attention`: a package-relative import under discover. As a module it runs with 5 skips.

The known-flaky tests passed in both runs.

Suggested talks.md entry (main tree not edited): `2026-09-24 - two-phone Stage A gaps 1-3 in code: Pixel
static co-helper in catalog/route generation (per-device executor, union mask on the 4352 grid,
phone_helpers through dormant parent/ticket/launch, per-device policy masks, PIXEL0 proofs); gap 4 hook
designed (stop policy undecided); gaps 5-6 fail-closed; no phone run.`
