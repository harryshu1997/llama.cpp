# Gemma coalesced-call transport qualification: 61,440 / 38,400-byte receipts + both-model identity (2026-09-24)

Goal (from `../20260924-coherent-policy-coalesced/README.md`): let BOTH assisted models run the `coalesced-batch`
phone plan under the coherent policy, so a batch-2 phone pass is one multi-row USB call per layer instead of two
sequential 1-row calls (the +15 % run-time cost of the coherence-only arm). The premise was that Gemma needs
61,440-byte receipts (8 rows x 3,840 x 2 B at `parallel=8`) or `parallel<=5` (38,400 B), while the identity the
coherent arm bound covers at most 40,960 B.

Status: receipts produced (PASS, 6/6), new identity materialized (15 receipts), admission verified offline for both
models. No campaign arm and no physical preflight were run. **One correction to the premise (section 4): the catalog
runs the Gemma desktop parent at `parallel=2`, so a Gemma coalesced call today is 2 rows = 15,360 bytes, which the
09-22 receipts already admitted; the Gemma lock-out in the coherent arm was the mixed batch-plan configuration, not
the receipts.** The new receipts make the identity future-proof for a re-planned Gemma parent at parallel 5-8.

## 1. Method (identical to the 2026-09-22 task1 receipts)

| item | value | check |
| --- | --- | --- |
| case runner | `/mnt/storage/s42-ffn-microbatch-20260916-v4-arwuw3/run_transport_case.sh dmabuf async devmem <req> <resp> 8 64 4 <case> <receipts>` | same argv shape as `task1-20260922-*-command.json` |
| host qualifier | stage `ffs_dmabuf_host`, sha256 `61bc2907...` | asserted before the run; = `qualification_binary_sha256` of the coherent identity (the deploy identity binds an older build `dea2f582...` in `/home/zhihao/s41-ffs-dmabuf-async-v2-20260815/`; the 08-29 mixed-v6 receipts were made with that one and stay in the deploy identity only) |
| phone worker | `ffs_dmabuf_phone.android` sha256 `e2c66e6b...` | byte-identical copy pushed to a NEW dir `/data/local/tmp/s43-transport-qual-20260924/` (verified on the phone); the production `/data/local/tmp/s41-ffs-dmabuf-v1/` was only read |
| phone session / restore | `/data/local/tmp/s42-rrphone-20260914-v3b/functionfs_transport_session.sh` (`fc9e91c0...` = identity `qualification_phone_session_sha256`), `restore_android_usb.sh` (`4c749c85...`) | asserted before the run |
| geometry | queue depth 4 (configured = active), warmup 8, 64 iterations, usbfs 16 MiB, slot safety 65,536 B, USB SuperSpeed 5000 Mbit/s (sysfs 2-2 `speed`), gadget 18d1:2d00 | same as task1; per case the runner asserts `complete status=0`, `worker_status=0`, no kernel faults, g1 restored, g2 empty, `super-speed` |
| directions | h2d (req=P, resp=64), d2h (req=64, resp=P), duplex (req=resp=P) for P in {61,440; 38,400} | what `transport_profiles.materialize_measured_usb_links` needs per payload (duplex + h2d + d2h of one configuration) |
| driver | `qualify_coalesced_both.py` (this dir; desktop copy `/home/zhihao/s43-coalesced-both-20260924-rig/`) | run as `flock -w 7200 <rig lock> env S43_UNDER_LOCK=1 python3 -u qualify_coalesced_both.py`; refuses to start if a desktop campaign/server/qualifier or a phone FFN worker is running, the g2 gadget is bound, or the kernel identity differs |

The only deviation from the 09-22 run is the phone session root (`S41_PHONE_ROOT`), so that no production `s41-*`/`s42-*`
phone directory is modified (every earlier qualification wrote its per-case session dirs into
`/data/local/tmp/s41-ffs-dmabuf-v1/`). The session script takes the binary path as an argument and writes only under
the session root, so the transport path is unchanged.

## 2. Receipts (`/home/zhihao/s43-transport-receipts-61440-20260924/receipts/`, copies in `data/receipts-run/`)

Phone: boot_id `ca2e7442-c4ed-4029-b85b-ad292e088c26` (same boot as the 09-22 receipts), kernel
`6.12.23-android16-5-o-g227664cbe007-4k`, `/sys/kernel/notes` `40bbacf7...` + BTF `77a8ce5a...` = candidate boot
`f13c7c03...`; unchanged after the run. Battery: level 80 %, USB powered at 500 mA, `battery_notify_code` 0 before,
after every case and after; temperature 25.7 C (PhoneTemp 250 -> 260). The unrelated retained
`direct_phone_service` (PID 25713, present since 09-22) was left untouched. No FFN worker was running.

| case | request B | response B | median ms | p90 ms | h2d MB/s | d2h MB/s | req/s | resets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coalesced-both-20260924-61440-h2d | 61,440 | 64 | 0.507 | 0.623 | 452.8 | - | 7,370 | 0 |
| coalesced-both-20260924-61440-d2h | 64 | 61,440 | 0.525 | 0.583 | - | 422.4 | 6,875 | 0 |
| coalesced-both-20260924-61440-duplex | 61,440 | 61,440 | 0.703 | 0.863 | 334.2 | 334.2 | 5,439 | 0 |
| coalesced-both-20260924-38400-h2d | 38,400 | 64 | 0.385 | 0.682 | 321.7 | - | 8,378 | 0 |
| coalesced-both-20260924-38400-d2h | 64 | 38,400 | 0.331 | 0.562 | - | 387.8 | 10,099 | 0 |
| coalesced-both-20260924-38400-duplex | 38,400 | 38,400 | 0.393 | 0.549 | 337.5 | 337.5 | 8,789 | 0 |
| (09-22) task1-20260922-40960-h2d | 40,960 | 64 | 0.343 | 0.421 | 430.2 | - | 10,502 | 0 |
| (09-22) task1-20260922-40960-d2h | 64 | 40,960 | 0.523 | 0.688 | - | 297.3 | 7,258 | 0 |
| (09-22) task1-20260922-40960-duplex | 40,960 | 40,960 | 0.523 | 0.763 | 267.9 | 267.9 | 6,540 | 0 |

An 8-row Gemma call (61,440 B each way) costs 0.70 ms median at depth 4 vs 0.52 ms for a 4-row Qwen call, i.e. the
per-row transport cost keeps falling with coalescing. All 6 receipts pass the `transport_profiles` receipt checks
(schema `s41_ffs_dmabuf_transport_v2`, dmabuf/async/devmem, generation `functionfs-dmabuf-async-ring-v2`,
`reset_recoveries=0`, depth 4/4, usbfs 16,777,216, slot safety 65,536). `RESULT.json`: status PASS, 6 cases,
`kernel_changed_by_this_test=false`, `maximum_payload_bytes=61440`.

## 3. Identity (`/mnt/storage/s42-trace-v2-20260921-prep/TRANSPORT_QUALIFICATION_IDENTITY_COALESCED_BOTH.json`)

- `identity_id` `s43-coalesced-both-20260924`; identity sha256 (canonical, what catalog links carry)
  `sha256:866f628e4605a1a73739568690b27b234a52b449e7373abc08dedff9158d8eb3`; file sha256
  `e95f0b945b8c14dd9a531532e506ecb9209c69cbb8b8af13aae3525385a1994c`.
- 15 receipts = the 9 task1 receipts bound by the coherent identity (7,680 / 10,240 / 40,960) + the 6 new ones.
- Software identity byte-identical to the coherent arm's identity except id and receipts: host server `ca975ce2...` + 7
  host libraries, transport client source `48be6dea...`, qualifier `61bc2907...`, qualification phone session
  `fc9e91c0...` / worker `e2c66e6b...`, production phone session `1118eb90...` / worker `43adcb75...` / resident workers
  `0e7736c0...` / router `25d27ff1...` (re-pulled to `phone-identities-coalesced-both/` and compared). Hardware: candidate
  boot `f13c7c03...`, kernel notes/BTF re-checked on the phone.
- Built by the new `MATERIALIZE_TRANSPORT_COALESCED_BOTH.sh` (deploy dir; copy here) under the rig lock through the
  production `research_dev.scheduler.adapters.materialize_transport_qualification` from the deploy source; command
  recorded in `software/MATERIALIZE_COMMAND_COALESCED_BOTH.{json,txt}`, log `materialize-coalesced-both.log`. Nothing
  in `materialize_transport_qualification.py` / `transport_profiles.py` was changed.
- Untouched (sha256 before = after): deploy `TRANSPORT_QUALIFICATION_IDENTITY.json` `82617a5c...`, `_DUAL.json`
  `4188c988...`, coherent arm identity `23b7f356...`.

## 4. Admission check (offline: `launch.py --resolve-only`, no hardware, no campaign)

Inputs `/home/zhihao/s42-trace-longtaildev2-coalescedboth-20260924-inputs` (copies in `data/admission-inputs/`), derived
with the deploy's `prepare_trace_inputs_v2.py` from the 09-23 long-tail treatment inputs + the dev_v2 trace +
`server_policy_coherence=true` + `--qualify-phone-batch-plan hot=coalesced-batch --qualify-phone-batch-plan
cold=coalesced-batch` (no `--allow-mixed-phone-batch-plans` needed: both models qualify the same plan) +
`--transport-receipts-dir .../receipts-all` (byte-identical copies of the 9 task1 + 6 new receipts; the identity binds
by sha256) + candidate boot `f13c7c03`; the new identity copied in as `TRANSPORT_QUALIFICATION_IDENTITY.json`.
Resolve PASS (catalog `sha256:9e16e801...`). `check_admission_both.py` (this dir) PASS, `TRANSPORT_ADMISSION-resolve-1.json`:

| check | result |
| --- | --- |
| one QUALIFIED `operator_split` helper per desktop parent, plan `coalesced-batch`, both models | Qwen `physical:hot:{cpu-,}phone-assisted:operator_split:coalesced-batch`, Gemma `physical:cold:{cpu-,}phone-assisted:operator_split:coalesced-batch`; split-row executors now SHADOW (still declared) |
| every FunctionFS link bound to the new identity | `sha256:866f628e...` (the NCM overlay link is bound to the whole-server identity, as before) |
| measured FunctionFS links | 7,680 / 10,240 / 38,400 / 40,960 / 61,440 B, depth 4, both directions |
| Gemma at models.json `parallel=8`: 61,440 B, 8 rows | `usb_max_payload_bytes=61440`, capacity `payload-61440:{h2d,d2h}`, depth 4, slot check 61,440x4+65,536 per slot x 4 < 16 MiB |
| Qwen `parallel=4`: 40,960 B, 4 rows | `usb_max_payload_bytes=40960`, capacity `payload-40960` |
| Gemma parallel-5 fallback: 38,400 B, 5 rows | capacity `payload-38400` |
| Gemma at the CATALOG parallel (2): 15,360 B, 2 rows | capacity `payload-38400` (new identity); `payload-40960` against the old coherent identity `06cd7bf4...`, i.e. **already admissible on 09-22** |
| negative control: 61,440 B against the old coherent catalog | refused, `phone transport maximum payload is not qualified` |

**Why the catalog says parallel 2.** `campaigns/burstgpt/catalog.py:623-640` builds a model's adapter parameters as
`{**config.runtime_parameters, **desktop_plan["adapter_parameters"]}`, and the measured desktop baseline plan
(`MEASURED_DESKTOP_BASELINE_PLANS_V1.json`, `plans[1]`) carries `{"parallel": 2}` for Gemma (Qwen: `{"context_size":
4096, "parallel": 4}`). The Gemma desktop server is launched accordingly (`large-model-4-physical-cold-desktop.stderr`:
`n_seq_max = 2`, `n_ctx = 32768`, `kv_unified = true`), and the plain arm's Gemma split-row helpers ran with
`ffn_max_tokens=2`, `usb_max_payload_bytes=15360`. The route compiler sizes a coalesced decode call as
`n_embd x min(ubatch, parallel) x 2` with that `parallel` (`_internal/route_generation/costing_parameters.py:487-508`),
and the runtime coalesced cohort is `min(4, parallel, ffn_max_tokens, maximum_batch_size)` = 2 rows for Gemma
(`_internal/runtime_decode_cohort.py:447-456`). So models.json's `parallel: 8` / `context_size: 32768` only sets the
context; the coalesced Gemma call the coherent controller would issue for a pair is exactly the 2-row call.

Consequence: the coherent arm's "Gemma never assisted" was not a receipt gap. Its inputs qualified only Qwen for
`coalesced-batch` (`--allow-mixed-phone-batch-plans`), and one phone session carries one batch plan. A both-coalesced
derivation (as above) puts both models on one plan with either identity; the new identity is what these inputs use.

## 5. Gemma parallel / context implications

- Today: `n_seq_max=2`, unified KV over `n_ctx=32768` (iSWA: non-SWA cache 32,768 cells for the 8 global layers, SWA
  cache for the 40 sliding-window layers, window 1,024; logged KV buffers 256 MiB CPU + 256 MiB CUDA0 at
  `gpu_layers=22`). Two Gemma slots share the 32,768 cells, so a pair has ample per-request context; the 2-row coalesced
  call (15,360 B) is what coherence buys for a Gemma pair.
- Raising Gemma to `parallel=5` (38,400 B) or `8` (61,440 B) is a desktop-parent contract change, not a transport one:
  the parent's measured baseline plan is at parallel 2, so a new `MEASURED_DESKTOP_BASELINE_PLANS` entry (desktop parent
  calibration) would be needed before the catalog admits it; with `kv_unified` the KV bytes stay those of 32,768 cells
  while the per-slot share drops to 32,768/5 or /8 (6,553 / 4,096 cells) if requests fill the slots. The receipts are
  ready for either choice; the 38,400 and 61,440 profiles are in the identity now.
- Under coherence the runtime cohort is capped at 4 rows (`min(4, ...)`), so `parallel=8` would still yield <= 4-row
  calls (30,720 B); only the route-compiler admission needs the 61,440 B profile.

## 6. Caveats

- Not run: any campaign arm, and `launch.py --preflight-only` (physical). The physical preflight re-verifies the
  identity against the live phone (boot image, session/worker digests) at session time
  (`adapters/phone_session_ops/transport.py`); the digests it compares are the ones asserted here, unchanged.
- The identity mixes receipts from two runs (09-22, 09-24) under the same boot_id, kernel and qualifier; the 08-29
  mixed-v6 receipts (older qualifier `dea2f582...`, older session script) are intentionally not included, as in the
  coherent identity.
- The derived admission inputs use a combined receipts directory with copied receipt files; `evidence.json` also
  accepts several directories (`transport_qualification_directories` is a list), which avoids the copies if preferred.
- `check_admission_both.py` first failed on an over-strict test (it required every phone link to carry one identity; the
  NCM overlay link carries the whole-server identity); the fix scopes the test to the FunctionFS generation. `ADMISSION-1.log`
  records that failure, `ADMISSION-2/3.log` the passes.
- The 61,440 / 38,400 medians are single 64-iteration campaigns like every earlier receipt; the identity uses
  `min(rates)` per direction for the link bandwidth, so these numbers set the cost model for those payloads.
