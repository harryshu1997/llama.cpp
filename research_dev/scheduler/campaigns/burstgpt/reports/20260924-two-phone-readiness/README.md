# Two helper phones behind one server: OP15 (FunctionFS DMA-BUF) + Pixel 10 Pro (adb-tcp Vulkan worker)

2026-09-24. Scratch deliverable, nothing committed, nothing run on a phone or on the rig. Authorization boundary
of `reports/20260922-fast-path-M3/README.md` ("only item 1 authorized") is respected: this prepares code and a
plan; every physical step below waits for the user.

| Deliverable | Path (this directory) |
| --- | --- |
| C++ server/client patch | `TWO_PHONE_SERVER.diff` (server.cpp, ffn-split-client.{h,cpp}) |
| Scheduler patch (incl. tests, new adapters, M3 gate) | `TWO_PHONE.diff` |
| Regenerate + `git apply --check` both against the live tree | `make_diffs.sh` |
| Progress log | `PROGRESS.md` |
| Example gate config / rig additions | `smoke/TWO_PHONE_GATE_CONFIG.example.json`, `smoke/RIG_TWO_PHONE_ADDITIONS.example.json` |
| Local builds (CPU, `S41_SERVER_FFN_SPLIT=ON`) | `build-base/` (main tree), `build-root/` (`src/` = main + server patch) |

## 1. What is implemented and verified vs designed only

| Item | State | Evidence |
| --- | --- | --- |
| llama-server: N FFN helper clients, disjoint whole-layer ownership, one union policy | IMPLEMENTED | built warning-free; `tests/test_two_phone_server_native.py` 3/3 PASS on `build-root`, 3/3 FAIL on `build-base`; `test_remote_resident_native.py` 17/17 PASS (legacy path) |
| Two helpers give tokens identical to one helper owning all layers | VERIFIED locally | tiny llama, CPU workers over TCP, static policy 16:128 |
| Partial acceptance of a union policy is rolled back | VERIFIED locally | quanta 64/128, 64-column policy refused with `helper pixel:`, no 64-column call made |
| rig `helper_phones`, topology, models `helper_phone_ffn_shards`, `adb-tcp` transport contract | IMPLEMENTED + unit-tested | `tests/test_two_phone_helpers.py` (28 tests) |
| adb-tcp worker lifecycle (preflight, start, forward, finite-budget drain, idle-only signal) | IMPLEMENTED; tested against a real local CPU worker through a fake adb | same file, `AdbTcpSessionTests` |
| Launch-contract hook `phone_helpers` -> multi-helper env | IMPLEMENTED + tested through the real `llama_server_launch_contract` | `LaunchContractTests` |
| Per-device call accounting (layer owner + request-id range) | IMPLEMENTED + tested | `CallAttributionTests`, gate result |
| Per-helper transport identity + missing-receipt report | IMPLEMENTED (contract) | `TransportIdentityTests`; no identity file materialized for the Pixel |
| USB topology preflight (sysfs, read-only) | IMPLEMENTED + tested on a fake sysfs of today's desktop | `UsbTopologyTests`, `PreflightRowsTests` |
| M3 mechanism gate (`campaigns/burstgpt/two_phone_gate.py`) | IMPLEMENTED; mechanics tested with fakes; NEVER RUN | `TwoPhoneGateTests` |
| Campaign runner dispatching two phones on a trace (dev_v2) | DESIGNED ONLY (section 6); fail-closed today | preflight `two-phone-dispatch` BLOCKED; `launch.py` refuses a run with helper phones |
| Pixel energy, two-phone energy/latency, token identity on real models | NOT MEASURED | needs hardware |

Full scheduler suite on the patched copy: 1676 tests; the only errors are the same 14 as the unpatched copy
(scratch copy lacks `research_dev/spikes` data, pre-existing relative import); with the data linked the 7
affected modules (90 tests) pass.

## 2. Native design (`tools/server/server.cpp`, `examples/layersplit/ffn-split-client.*`)

Environment (legacy single-helper variables keep their exact meaning and proof lines):

```
S41_SERVER_FFN_HELPERS=2
S41_SERVER_FFN_LAYER_MASK=<union>                     # shared: ARTIFACT, COLUMNS, N_EMBD, RUNTIME_CONTROL, DORMANT_*, POLICY ...
S41_SERVER_FFN_HELPER0_{LABEL,LAYER_MASK,TRANSPORT=functionfs-usb,USB_*}   # OP15
S41_SERVER_FFN_HELPER1_{LABEL,LAYER_MASK,TRANSPORT=tcp,HOST,PORT}          # Pixel, via adb forward
```

- One `ffn_split::client` per helper (TCP or FunctionFS, unchanged client code). Masks must be nonempty,
  disjoint and cover the union; at most one `functionfs-usb` helper (the host USB client opens the first
  18d1:2d00 device); legacy HOST/TRANSPORT/PORT next to HELPERS, duplicate labels, remote-resident layers, tail
  fence and row diagnostic are refused at startup. Static table/shape policies work (each client reads them for
  its own layers) and are validated against every connected helper.
- `connection=deferred`: every helper defers; a policy connects only the owners of its active layers, then checks
  that connected helpers serve the same FFN suffix geometry.
- Eval callback: with several helpers the layer in `ffn_norm-<il>` / `ffn_phone_partial-<il>` selects the owner
  client; nothing else is observed. Layers run in order, so at most one client is pending (no cross-phone
  overlap inside a token; latencies add).
- `apply_policy(union mask, columns)`: owners connect first, each client gets `mask & owned` (or 0,0); if a later
  helper rejects its subset (e.g. width not on its quantum) the earlier helpers are restored and the control is
  refused (clean `success:false`, the slot keeps its policy). `column_quantum()` reports the LCM of connected
  quanta (1 while deferred); OP15 Qwen HELLO quantum 2176, Pixel 4352 -> union widths must be multiples of 4352
  (25/50/75/100 %).
- server-context is unchanged and needs no change: `can_batch_with` compares the union policy, the dormant host
  share releases/restores the union mask (both phones' layers) with the mixed-policy guard intact,
  `apply_ffn_split_ubatch_context` feeds every client the same row context.
- Proof lines: `S41SERVERFFNCALL` unchanged; helper k numbers its calls from `1 + k * 2^24` (new
  `client_config::first_request_id`), so ids are unique and identify the owner. One `S41SERVERFFN ready` line
  (`... connection=deferred helpers=2`), one `S41SERVERFFNHELPER label= layer_mask= transport= ...` line per helper,
  and per-helper `S41SERVERFFN {"helper":..,"layer_mask":..,...}` / `S41SERVERFFNSHAPE {"helper":..}` summaries.
  With one helper every line is byte-identical to today.
- Control-ack runtime stats stay aggregate (sum of calls/rows/bytes, call-weighted means): `http_backend.
  _runtime_stats` accepts only its fixed integer keys, so a nested per-helper object (the stopped agent's version)
  would have failed every acknowledgement. Per-device accounting comes from the proof lines.

## 3. Scheduler contracts added (all additive; single-phone manifests and tickets unchanged)

- `rig.json` keeps `phone` (the primary FunctionFS phone, every legacy field) and gains `helper_phones: [...]`
  plus `topology.helper_phones: [...]` (`configuration/rig.py`, `HelperPhoneRigConfiguration`,
  `HelperPhoneTopologyConfiguration`). A helper phone binds `device_id`, `serial`, `adb_port`,
  `transport: "adb-tcp"` (the only helper transport; `tcp` keeps its tensor-bridge meaning), `backend`,
  `worker_path`, `library_directories` (list; joined into LD_LIBRARY_PATH), `worker_environment`,
  `worker_port`, `forward_port` (0 = adb picks), `column_quantum`, `max_tokens`, `max_requests` (0 = resident),
  `kernel_release`, `minimum_usb_speed_mbps` (per phone), `usb_sysfs_device` (e.g. `2-9.2`). Refused: serial or
  device equal to the primary or duplicated, helper devices not of kind `phone` or not in the topology, shared
  compute/memory resources, forward port equal to the whole-phone forward, two helpers on one USB port,
  FunctionFS/bridge helpers, `:` in a library directory, LD_LIBRARY_PATH in the worker environment. A manifest
  without helper phones serializes byte for byte as before (tested on the dev2 rig.json).
- `models.json` model rows gain `helper_phone_ffn_shards: {device_id: {index_path, directory}}`; the loader
  refuses a device that is not a rig helper phone.
- `adapters/phone_transport.py`: `ffn_transport: "adb-tcp"` -> `PhoneTransportContract(transport="adb-tcp",
  adb_serial, adb_port, phone_worker_port, control_host/port = the forward)`; the server sees a plain TCP client.
- `adapters/phone_helpers.py`: `PhoneHelperBinding` (device, serial, owned layers, label, transport parameters),
  `validate_disjoint_ownership`, `helper_layer_masks` (per-phone sub-masks of one adaptive policy),
  `helper_server_environment` / `phone_helper_launch_environment`, `attribute_ffn_calls` (per-device calls, rows,
  bytes, layers; fails closed on unowned layers, repeated ids or an id outside the owner's range),
  `PhoneHelperTransportIdentity` + `validate_helper_identities`, `check_usb_topology` (sysfs).
- `adapters/phone_tcp_session.py`: `AdbTcpWorkerConfiguration` / `AdbTcpPhoneWorkerSession`: preflight refuses a
  running worker, an occupied phone port (LISTEN/ESTABLISHED) and any worker/library/shard hash mismatch; start
  runs the exact qualified command through `adb -s <serial> shell -T exec env LD_LIBRARY_PATH=... worker ...`,
  waits for `[ffn-worker] ready`, adds the forward; stop waits until the server's client left, drains the finite
  budget with zero-input protocol-v6 calls so the worker exits 0 (the M3 practice), removes the forward and
  checks the boot id. A resident worker is only signalled with `allow_idle_signal=True` and no client; an
  in-flight or undrained worker is left running, never killed.
- `adapters/llama_server_contracts.py`: a ticket carrying `phone_helpers` (JSON list; first = the ticket's own
  phone with empty transport parameters) launches the multi-helper environment; the plan's
  `S41_SERVER_FFN_LAYER_MASK` must equal the union of the helpers' layers.
- `campaigns/burstgpt/preflight.py`: `--helper-phone` rows add `phone-usb-port:*` and `phone-usb-topology` checks
  and an always-BLOCKED `two-phone-dispatch` check; `launch.py` passes the rows and refuses a real run (not
  `--resolve-only`/`--preflight-only`) of a rig with helper phones: a run would silently use OP15 only.
- `campaigns/burstgpt/two_phone_gate.py`: the M3 check (section 7, step 3).

## 4. Transport qualification identity for two phones

The existing `TransportQualificationIdentity` (`adapters/transport_profiles.py`) stays OP15's: FunctionFS
hardware (`functionfs_identity 18d1:2d00`, boot image, kernel, `a600000.dwc3`, serial, sysfs `2-2`), host server
and libraries, phone worker/session/router hashes, 9 DMA-BUF h2d/d2h/duplex receipts. A two-phone deployment
additionally needs one `PhoneHelperTransportIdentity` per helper, never derived from OP15's receipts:

| Pixel identity field | Value / source | Status |
| --- | --- | --- |
| hardware: serial, sysfs port, ADB-mode id, host controller, kernel | `5A040DLCH004ES`, `2-9.2`, `18d1:4ee7`, `0000:00:14.0`, `6.6.102-android15-8-g6eb5b2a8c46b-ab14739656-4k` | known (read-only sysfs today; M3 INVENTORY.json) |
| software: worker | `7cf01c7a...` (`s42-pixel10pro-ffn-coalesced-20260922-v1`) | known |
| software: libraries | libggml `601b8a7c...`, libggml-base `ec839665...`, libggml-cpu `6ca5bcc4...`, tuned libggml-vulkan `a97cb05d...` | known |
| software: shard | `HTP0.ffn.gguf` layers 18-23 `dd705a75...`; parent tensors verified (`physical/op11-tcp-1/SHARD_PARENT_IDENTITY.json`) | known; index still says `parent_verified: false` |
| software: worker environment | `S42_PIXEL_F16_{WG=128,ROWS=8,SUBGROUP=128}`, `--column-quantum 4352 --max-tokens 4 --backend Vulkan0` | known (`PIXEL_GEMV_SELECTED.json`) |
| software: host server + client source | new `llama-server` / `libllama-server-impl.so` (this patch), `ffn-split-client.cpp` | MISSING: must be built and hashed in the M3 deploy |
| receipt `usb-link-speed` | `software/pixel10pro-vulkan/INVENTORY.json` (5000M, 2-9.2) | exists |
| receipt `numerical-rows-1-2-4` | `physical/pixel10pro-server-1/run4-six-layer-qualification/RESULT.json` (72 calls/168 rows, max rel L2 3.27e-4, stock Vulkan lib); tuned lib: `PIXEL_GEMV_CONFIRM.json` (1680 calls byte-identical to the original kernel) | exists |
| receipt `server-token-identity` | `physical/pixel10pro-gemv-server-1/run1/RESULT.json` (4x64 tokens exact, 744 calls) | exists for the OLD server `ca975ce2...`; MISSING for the new binary |
| receipt `adb-forward-round-trip` | only derived (`PIXEL_SERVER_LATENCY_BREAKDOWN.json`: 4.3-5.0 ms outside the worker) | MISSING as a machine-checked receipt |
| receipt `scheduler-launched-session` | adapter launch + stop receipts with normal exit | MISSING (no scheduler has launched the Pixel) |
| two-phone concurrency (OP15 FunctionFS + Pixel adb on one xHCI) | none | MISSING |
| Pixel energy | none | NOT MEASURED (assumed power reported separately) |

OP15's own identity must be re-materialized after the server rebuild (`host_binary_sha256` and
`host_dependency_sha256:llama-server-impl` change; `transport_client_source_sha256` = `ffn-split-usb-client.cpp`
does not).

## 5. USB bus sharing (read-only `lsusb -t` and sysfs on 172.20.74.85, 2026-09-24)

```
Bus 002 root hub xhci_hcd 20000M/x2  (Alder Lake-S PCH USB 3.2 Gen 2x2, PCI 0000:00:14.0)
 |__ Port 002  22d9:2772 OnePlus 15   5000M   sysfs 2-2   (serial 3C15AU002CL00000; FunctionFS mode enumerates 18d1:2d00)
 |__ Port 009  174c:3074 ASM107x hub  5000M
      |__ Port 002  18d1:4ee7 Pixel 10 Pro  5000M   sysfs 2-9.2 (serial 5A040DLCH004ES)
```

- One xHCI controller, distinct root ports, no shared hub uplink: decode calls are 10-40 KiB each way
  (5120 x 1-4 rows x f16) and latency-bound, so bandwidth contention is not expected; `check_usb_topology` fails
  only for a shared root port and reports the shared controller.
- Real risks: (a) an OP15 gadget switch/restore or the adapter's one-shot USB reset re-enumerates port 2 while the
  Pixel's adb forward is live; the adb server (5037) must not be restarted and no second adb server started;
  (b) the host FunctionFS client selects by VID:PID 18d1:2d00; the Pixel is 18d1:4ee7 in ADB mode but would
  collide if it ever entered accessory mode, hence the native "at most one functionfs-usb helper" rule; binding
  by bus path is the durable fix; (c) the xHCI interrupt/latency path is shared, so the M3 gate must report
  OP15 RPC percentiles with and without the Pixel (the per-helper SHAPE lines give both); (d) both phones charge
  from the desktop; the Pixel sits behind the onboard hub, so phone thermals/DVFS drift remain uncontrolled.

## 6. Trace dispatch with two phones: design and ranked gap list (DESIGNED, not implemented)

The campaign runner still schedules exactly one phone (about 300 single-phone assumptions; a read-only map of the
request path was taken for this list). Two stages:

**Stage A, static co-helper (recommended next; enough for dev_v2).** OP15 stays the scheduled, adaptive,
re-provisioned helper. The Pixel is a trace-lifetime resident co-helper of one model (Qwen layers 18-23, the
only Pixel shard that exists), started after preflight and stopped at `end_trace`, never COW-replaced. A Qwen
phone-assisted route then carries both phones: its plan includes the Pixel layers and shard, its ticket carries
`phone_helpers`, its adaptive policies use the union mask on the 4352-column grid. Gemma/Llama routes are
unchanged (no Pixel shard).

**Stage B, per-device residency** (later): device-keyed session pools, layout generations, helper preparation
and re-provisioning; a second phone composite or composite `operator_placements` with a per-operator helper
device (`_internal/capability_contracts/composites.py` already has the field).

Desktop-follow re-provisioning (`_unified/phone_residency_ops/reprovision.py`, merged today) under each stage:
Stage A leaves it unchanged. It keeps moving OP15's HTP sessions between Qwen and Gemma as the desktop switches
model. The Pixel range stays outside its layout; the only invariant is per-model disjointness, and OP15's Qwen
shard index already covers only layers 0-17. When the desktop follows Gemma, the Pixel idles: a Gemma ticket has
no `phone_helpers`. Stage B generalizes the portfolio to `{device: (sessions, memory budget, layout generation,
learned swap rate)}`:
- FOLLOW gives each device's capacity to the followed model's layers, with at most one owner per layer per model
  across devices.
- PROPORTIONAL splits each device's sessions by remaining decode work.
- The in-use deferral and the SESSION_LOADING -> SESSION_VERIFIED swap estimate are per device. A Pixel swap
  means restarting the worker with another shard; that cost is unmeasured.
- HTP memory caps (`economics.py`) are already keyed by device.

| Rank | Gap (blocks a two-phone dev_v2 run) | Where | Stage |
| ---: | --- | --- | --- |
| 1 | No Pixel executor, links, shards or layers exist in the catalog/plan. With the rig additions, catalog materialization only registers OP15, so no ticket can carry `phone_helpers`, and proofs would reject Pixel calls ("no unique shard owner"). | `adapters/catalog_materialization.py` `base_executor_capabilities` 671-761, `_phone_family_*` 940-1091 (inject `phone_helpers`, add Pixel resources); `campaigns/burstgpt/catalog.py` 523-541 (re-adds USB links only for OP15), 613-688; `plan_contracts/operators.py` phone_shards already allow several endpoints | A |
| 2 | Ticket parameter plumbing for `phone_helpers`. | whitelist `_unified/common.py` 37-70; `adapters/residency.py` 278-320 (pop it like the dormant key); `adapters/ticket.py` 371-425 | A |
| 3 | Adaptive policy on the union. Union widths must lie on the LCM quantum 4352, since OP15's 2176 grid gives 12.5 % widths the Pixel refuses (a refused control keeps the host policy but wastes the window). The policy mask must be the union, or `completed_phone_calls % layer_count` fails. | `catalog_materialization.py` ~949 (`ffn_column_quantum`), `_internal/adaptive_decode_planning.py` 133-223, 404-450 | A |
| 4 | No rig-side helper lifecycle. Worker start and stop at trace boundaries are missing, the transition scope and helper-publication gates are FunctionFS-only, and a Pixel without phone telemetry would defer every OP15 preparation. | `adapters/heterogeneous_rig.py` 215-503, 863-1072; `heterogeneous_rig_ops/transitions.py` 311-345, 578-608; `_unified/phone_residency_ops/fixed.py` 111-147. Use `adapters/phone_tcp_session.py` (implemented). | A |
| 5 | Transport qualification. There is no Pixel identity file, and OP15's identity must be re-materialized after the server rebuild. The Pixel lacks `server-token-identity` for the new binary, `adb-forward-round-trip` and `scheduler-launched-session`. Step 3 below produces all three. | `adapters/transport_profiles.py` (single phone), `phone_session_ops/transport.py` 237-322; `phone_helpers.PhoneHelperTransportIdentity` (implemented) | A |
| 6 | Cost and energy evidence: no Pixel kernel/link profile and no power profile, so the energy-aware selector cannot price a two-phone route. Treat the combined route as SHADOW or learn it, and report Pixel power as a separate assumption. | `campaigns/burstgpt/catalog.py` 378-475, 545-575; evidence `phone_power` | A |
| 7 | Stop semantics of a trace-length Pixel worker. The TCP worker has no shutdown message, and the budget for a trace is unknown. Either add a TCP shutdown message to the worker (Pixel worker code belongs to the other agent), or accept an opt-in SIGTERM of an idle worker (`allow_idle_signal`, never while a client is connected). **User decision.** | `examples/layersplit/ffn-split-worker.cpp` TCP loop; `phone_tcp_session.stop` | A |
| 8 | Per-helper runtime stats for the adaptive controller (RPC per phone). Stats are aggregate today; `_runtime_stats` would need new whitelisted integer keys. | `adapters/http_backend.py` 431-478 | A (optional) |
| 9 | Device-keyed residency, preparation and layout generation. Blockers: `ready_plan.py` 290-310 `len(parents) != 1`, `phone_shards.py` 765/1304 "sessions span multiple devices", and the single generation in `model_placement_controller`. | as listed | B |
| 10 | Two FunctionFS phones: the host USB client selects by VID:PID. Not needed for OP15+Pixel. | `ffn-split-usb-client.cpp` | later |

Today, with the rig additions of `smoke/RIG_TWO_PHONE_ADDITIONS.example.json`, the dev_v2 preflight gives:
- `phone-usb-port:primary-phone`, `phone-usb-port:pixel10pro-phone` and `phone-usb-topology`: expected PASS. Sysfs
  today shows both at 5000M on root ports 2-2 and 2-9.
- `two-phone-dispatch`: **BLOCKED** by design.
- `launch.py` (not `--resolve-only` / `--preflight-only`): **refused**.
- After a server rebuild, OP15's transport identity check fails (`host_binary_sha256`) until it is re-materialized.
- Loading the manifest itself works (tested).

## 7. Smoke plan (each step waits for the user's authorization; nothing below has been run)

Rules as in the plan: one run at a time under `flock -w 900 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock`,
adb server 5037 only, never `pkill -f` a pattern in your own command line, never kill an in-flight worker, new
output directories only.

0. **Read-only link check (no lock).**
   - Desktop: run `lsusb -t` and
     `python3 -c 'from research_dev.scheduler.adapters.phone_helpers import check_usb_topology as c; print(c([("op15-phone","3C15AU002CL00000",None,5000),("pixel10pro-phone","5A040DLCH004ES","2-9.2",5000)]))'`.
     Expect both passed and root ports 2-2/2-9.
   - `adb -P 5037 devices -l` must list both serials.
   - On the Pixel: `sha256sum` of the worker, its 4 libraries and `HTP0.ffn.gguf` must match
     `smoke/TWO_PHONE_GATE_CONFIG.example.json`; no `llama-ffn-split-[w]orker` process may be running and port
     26990 must be free.
1. **Deploy and build (no phone).**
   - New deploy `/mnt/storage/s42-two-phone-M3-<date>-<id>`. Copy the main tree with both diffs applied, as in
     `reports/20260920-fast-path-M2/DEPLOY.sh`, including `cmake/build-info.cmake`, `common/build-info.cpp.in`,
     the 20260917 `kv_decode_relocation_gate.py` (the wrapper imports it; `reports/` is otherwise excluded) and
     the M0 `PROMPT.txt`.
   - Build `llama-server llama-ffn-split-worker` with CUDA and `-DS41_SERVER_FFN_SPLIT=ON`.
   - `strings cuda-build/bin/libllama-server-impl.so | grep -c S41SERVERFFN` must be >= 12 (the local CPU build gives 17).
   - CHECK: `test_two_phone_helpers`, `test_two_phone_server_native`, `test_kv_decode_relocation_gate`,
     `test_remote_resident_native` and `test_llama_server_adapter` with `S42_LLAMA_BUILD_BIN=<deploy>/cuda-build/bin`,
     plus pyflakes.
2. **OP15 identity (lock, OP15 only).** Re-materialize OP15's transport identity against the new server, as in
   `reports/20260920-fast-path-M2/MATERIALIZE_TRANSPORT.sh`: `rm -rf phone-identities`, move the old identity aside.
3. **M3 mechanism gate (lock, both phones; pair-v1 prompt, 9,737 prompt tokens, 64 outputs, OP15 HTP0-2 = layers
   0-17, Pixel = 18-23).** Launch `python3 -u research_dev/scheduler/campaigns/burstgpt/two_phone_gate.py --config
   <deploy>/config/two-phone.json --output <deploy>/physical/<arm> ...` from `<deploy>/native-source`. The config is
   `smoke/TWO_PHONE_GATE_CONFIG.example.json` with paths filled in; Pixel `max_requests` 1024. Run these arms in order:
   1. `--arm control` (desktop only)
   2. `--arm combined --host-columns 0 --without-helper-phone`: OP15 alone, 100 % of 18 layers = 313,344 column-layers
   3. `--arm combined --host-columns 4352`: two phones, 75 % of 24 layers = 313,344 column-layers, **the equal-fraction arm**
   4. `--arm combined --host-columns 0`: two phones, 24 layers at 100 %, the capacity arm (+33 % released bytes)
   5. `--arm control` again

   **PASS** requires all of:
   - tokens exact against control, or the amended M2 near-tie rule;
   - `phone_proof_summary.exact` for every owned layer, including the `PIXEL0` shard;
   - `TWO_PHONE_RESULT.calls_by_device` covers layers 0-17 on OP15 and 18-23 on the Pixel, with ids in the
     helper's range;
   - Pixel drained to exit 0, boot id unchanged, forward removed; OP15 close receipt normal;
   - dormant released bytes >= the union lower bound;
   - reported: host decode W, ms/token, released bytes, OP15 RPC percentiles with and without the Pixel (bus
     sharing), and assumed Pixel energy in a separate column.

   Stop at the first failure and report it. The arms also produce the missing Pixel `server-token-identity`,
   `adb-forward-round-trip` (from the per-helper SHAPE lines) and `scheduler-launched-session` receipts for
   gap 5. Expectation from M3 data, not a claim: the Pixel adds about 25-35 ms per assisted layer, in sequence
   with OP15's layers.
4. **dev_v2 trace** (`/mnt/storage/burstgpt-source/longtail_dev_v2`: 9 requests, 4 Qwen / 4 Gemma / 1 Llama,
   arrival scale 0.4; the plain arm took 964 s). Only after gaps 1-5 are closed and step 3 passed. Then:
   - Copy `/home/zhihao/s42-trace-longtaildev2-plain-20260924-inputs` to a new inputs directory; add the rig and
     models entries of `smoke/RIG_TWO_PHONE_ADDITIONS.example.json`, a new `campaign_id`, and the re-materialized
     identity.
   - Run `launch.py <campaign.json> <new dir> --resolve-only`, then `--preflight-only` (expect every row PASS,
     including `two-phone-dispatch` once gap 1 lands), then the run.
   - Pair it with an OP15-only treatment and the desktop baseline on the same deploy. Report the matched saving
     with the Pixel's assumed energy separate. Decide gap 7 before this step.

## 8. Risks and open questions

- Latency: phones execute in layer order within one token, so the Pixel's per-layer RPC (the qualified Vulkan
  worker is 33 ms at full width against about 18 ms for the desktop CPU) adds to OP15's. The energy win must come
  from the released host share; M3 single-phone Pixel data shows about 11 % host-energy saving and +18 % decode
  time for 6 layers. The other agent's faster private CPU candidates (about 6-8 ms B1 phone-local) are not
  server-qualified. The rig schema accepts any worker, libraries and environment, and each needs its own receipts.
- Column grid: union widths must be multiples of 4352. OP15-only work keeps its 2176 grid, so policies differ
  between one-phone and two-phone arms.
- USB: see section 5. The adb server is shared; OP15 gadget switches and resets happen next to a live Pixel forward.
- Phone memory: 6 Pixel layers = 3060 MiB of Vulkan weights (qualified). Other Qwen ranges on the Pixel need new
  shards: `ffn_shard_gguf.py` plus a push, which is a user-authorized adb step.
- Identity: the Pixel worker and libraries are isolated builds (`s42-pixel10pro-ffn-coalesced-20260922-v1`, tuned
  Vulkan `a97cb05d`). A worker rebuilt from the current tree changes the hashes and needs requalification.
- Energy: Pixel energy is unmeasured and not remotely measurable; always report it as a separate assumption.
- Concurrency with other agents: the patches were rebased onto the live tree on 2026-09-24 after the
  re-provisioning merge. `make_diffs.sh` regenerates them and runs `git apply --check`. Re-run it immediately
  before applying.

Suggested talks.md entry (main tree not edited): `2026-09-24 - two-phone prep: native N-helper FFN runtime +
scheduler contracts + M3 gate implemented and locally tested (no phone run); trace dispatch = gaps 1-5; plan in
two-phone/README.md.`
