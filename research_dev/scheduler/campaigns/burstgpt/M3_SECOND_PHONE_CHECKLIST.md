# M3 preparation: a second phone on the RTX 4060 Ti desktop (OP11 arrived 2026-09-21; OP12 notes kept)

Written 2026-09-21 from the current tree and rig state. Status of each item is what is known today; the
person plugging the phone in should walk the list top to bottom. OP12 is OnePlus 12, Hexagon v75, serial
`5ae7a43d`, about 6 GB of usable RAM (OP15: v81, about 10 GB).

## 0. What is on the desk today (2026-09-21 14:00 EDT)

A **OnePlus 11 5G** (`22d9:2764`, serial `832358d4`, Snapdragon 8 Gen 2 = **Hexagon v73**) is attached, but:

- it enumerated at **480 Mbit/s on the USB 2.0 side of the ASMedia ASM107x hub** (`1-2.2`), not on the 5 Gbit/s
  side (`Bus 002`); the cable or contact is USB 2.0. Use a USB 3 data cable and re-seat; verify with `lsusb -t`.
- it exposes only an **MTP (Imaging) interface, no adb**: USB debugging is off or the USB mode is file transfer
  only. Enable Developer options -> USB debugging, accept the RSA prompt for this desktop, then `adb -P 5037 devices -l`.
- root, kernel and HTP library status unknown until adb works (section 2). The tree supports `opt_arch 73`
  (smaller VMEM path, `ggml-hexagon.cpp`), but **no v73 HTP library is built or deployed**: build with
  `DSP_VERSION=v73` in the snapdragon docker. v73 has no HMX, so its FFN calls run on HVX (slower per layer than
  OP15's v81); it still removes host CPU work, which is the energy lever.
- OnePlus 11 ships with 12 or 16 GB RAM; expect about 6-8 GB usable -> 6-12 Qwen F16 layers (535 MB each).
  With OP15 owning 18 layers, OP11 owning layers 18-23 releases all 24 CPU-resident FFN layers.

### Decisions 2026-09-21 17:00 EDT (user)

- The OnePlus 11 has a **USB 2.0 port**: it stays at 480 Mbit/s. Consequence: `minimum_usb_speed_mbps` (rig 5000,
  identity 5000) must become per-phone (OP11: 480), and its receipts are USB 2.0 receipts. Per-layer decode payloads
  are ~10 KB each way, so ~0.5 ms of wire time plus USB 2.0 scheduling latency per layer call; prefill payloads
  (hundreds of KB to a few MB) cost tens of ms. Decode-only relocation is the right mode for it.
- The phone **is rooted**: functionfs DMA-BUF transport is possible (needs the ffs/dmabuf kernel patch via a
  user-authorized `fastboot boot`, as on OP15) or TCP over adb as the fallback.
- Still blocked on **USB debugging**: the phone exposes only the MTP interface (class 06), adb does not list it.

## 1. Physical link (user, minutes)

- Any rear blue (USB 3.2 Gen1), red/teal (Gen2) or Type-C port is fast; only the two black USB 2.0 ports
  are not. The onboard ASMedia hub also feeds the case's front USB 3 ports. OP15 is on root port 2.
- Use a USB 3 data cable, not a charging cable. Verify on the desktop:
  `lsusb -t | grep -B1 -A1 -i "oppo\|vendor specific"` must show the new device under `Bus 002` at
  `5000M`. A `480M` line under `Bus 001` means the cable or contact fell back to USB 2.0.
- `adb -P 5037 devices -l` must list both serials. Do not start an adb server on any other port.

## 2. Phone software (blocking questions, decide before code)

| Item | OP15 today | OP12 status | Needed |
| --- | --- | --- | --- |
| Root / Magisk (`/data/adb/magisk/busybox`) | yes | unknown, earlier notes say no root | the functionfs DMA-BUF transport configures the USB gadget and needs root; without it OP12 must use the TCP transport (`S41_SERVER_FFN_TRANSPORT=tcp`, worker `--port`) over adb forward, which adds a few ms per layer call |
| Qualified kernel (`fastboot boot`, no flash) | `6.12.23-android16-5-o-g227664cbe007-4k` with the ffs/dmabuf patch, image pinned by sha256 in rig.json | none | only needed for functionfs; TCP works on the stock kernel |
| HTP library | `libggml-htp-v81.so` in the android bin dir | **no v75 library in any deployed bin dir** | build the Android/Hexagon binaries with the v75 skel inside the snapdragon docker (see memory "b98xx Android Hexagon build gotchas"), push to `/data/local/tmp/...` on OP12, record hashes |
| Resident workers/router binaries | pinned hashes in the transport identity | same source builds, must be rebuilt for v75 | hashes enter the identity |
| Model weights on phone | full F16 artifact at `/data/local/tmp/s41-opoffload-dmabuf-v1/` plus per-session shards | none | shards only (below); the full 29.5 GB artifact does not fit and is not needed |

## 3. Layer ownership and shards

- Qwen3-14B F16 FFN is 535 MB per layer. OP15 owns layers 0-17 (three HTP sessions of six layers,
  about 9.6 GB). OP12 can hold at most about eight layers; the natural split is OP12 owns layers 18-23
  (one or two sessions), so all 24 host layers' FFN leave the desktop during decode.
- Generate OP12's shards with `research_dev/scheduler/native/ffn_shard_gguf.py` (`--shard` per session,
  `--parent-sha256`, `--verify-parent`), push them, and register them in the shard index
  (`adapters/ffn_shards.py`, schema `s42-ffn-shard-index-v1`).
- OP12 is v75: the fused flash-attention bug there is attention-side and irrelevant to FFN work.

## 4. Scheduler and rig manifest (code, M3 proper)

- `rig.json` has a single `phone` block and one `op15-phone` device with `op15-htp`, `op15-adreno`,
  `op15-functionfs` resources and `*_phone` endpoints. Two phones need a list of phone blocks, a second
  device (`op12-phone`, 6e9 bytes), its resources, and endpoints; the rig coordinator books both, the
  runtime control applies one decode-boundary layer mask covering both ranges, and the dormant host
  share releases both. `can_batch_with` and the cohort logic are unchanged by this.
- The transport identity must carry both phones' binary hashes and both sets of receipts.

## 5. Transport qualification for OP12

- Fresh receipts on OP12's own port (the six `mixed-v6-{7680,10240}-{h2d,d2h,duplex}` receipts for the
  functionfs path, or the TCP equivalents if that is the transport), then
  `python3 -m research_dev.scheduler.adapters.materialize_transport_qualification` for the deploy.
- Both phones share the desktop's one xHCI controller. Payloads are about 20 KB per layer call, so this is
  latency-bound and the sharing is expected to be invisible; M3's phone-time check measures it.

## 6. M3 check (from the plan)

Decode split over both phones versus OP15 alone at the same total fraction: host decode power, ms per
token, released bytes; both phones' proof rows verified; outputs identical to host-only (or the amended
near-tie rule). Cohort size stays at most 4 per phone call.

## Order of work once the phone is on the desk

1. Link check (section 1). 2. Root and kernel decision (section 2) -> functionfs or TCP. 3. v75 Android
build and push. 4. Shards. 5. Rig manifest and coordinator for two phones. 6. Receipts and identity.
7. The M3 check.

## 7. OP11 state after adb came up (2026-09-21 21:45 EDT) and the implementation brief

Facts (adb, `adb root` works, Magisk app installed, busybox at `/data/adb/magisk/busybox`):
CPH2451, `kalama` / SM8550 (Hexagon v73, soc_id 519), Android 15 (SDK 35), kernel `5.15.189-g306c8fd4beb0`
(stock: `CONFIG_USB_CONFIGFS_F_FS=y`, `CONFIG_DMABUF_HEAPS=y`, dma_heap `system` present, UDC `a600000.dwc3`,
gadgets `g1`/`g2` exist), 15.5 GB RAM (12.7 GB available), `/data` **3.9 GB free of 218 GB** (`/data/local/tmp`
holds 209 GB of older experiments: test_bench 70 GB, hyzheng 38 GB, llamacpp_models 38 GB, ...). Pushed and
verified: `/data/local/tmp/s42-op11-20260921-bin/` = worker (sha eaa83a73...), resident-workers (043a95e5...),
router (6d513580...), `libggml-htp-v73.so` (16ae6da5...), libggml*/libllama* arm64 libs, `libomp.so` from NDK
r27c; `llama-ffn-split-worker --help` runs. Qwen shard for layers 18-23 (3.21 GB, sha dd705a75...) is on the
desktop at `/home/zhihao/s42-op11-qwen-shards-20260921-v1/qwen/HTP0.ffn.gguf`, NOT pushed: storage.

Blocking user decision: free >= 10 GB on OP11 `/data` (which of the old directories may go), or accept a 5-layer
shard (2.67 GB) that leaves ~1.2 GB.

Transport decision: the stock 5.15 kernel has no OP15-style ffs/dmabuf patch, so OP11 starts on the **TCP
transport** (worker `--port`, adb forward; `S41_SERVER_FFN_TRANSPORT=tcp`); the launch contract only forces
functionfs-usb for the coalesced-batch plan, and the rig adapter gates the direct transport on
`transport == "functionfs-usb"` in `heterogeneous_rig.py:887` and `heterogeneous_rig_ops/transitions.py`, which
the two-phone work must generalize. USB 2.0 -> per-phone `minimum_usb_speed_mbps` 480.

Implementation brief (agent, after the single-phone target run):
1. Rig manifest: `phone` becomes a list (or `phones`), each with serial, adb port, transport (`functionfs-usb` |
   `tcp`), speed floor, session root, binaries, shard dirs; a second device `op11-phone` (memory 12e9),
   resources `op11-htp`, `op11-tcp`; topology lists phone devices; `PhoneRigConfiguration.from_json` and
   `RigTopologyConfiguration` accept both shapes (single-phone manifests keep working).
2. Discovery/identity: phone session discovery per device; transport identity with both phones' receipts (OP11
   receipts over TCP/USB 2.0, measured); `minimum_usb_speed_mbps` per phone.
3. Layout: shard index per (model, device); planner treats sessions of both devices as one pool with device
   affinity; decode-boundary runtime control applies one mask covering both devices' layers; dormant release
   releases both; per-request proofs per device.
4. Executors/routes: `physical:*:phone-assisted:*` per device (or one composite with two phone endpoints); cost
   evidence per device (calibration on OP11: v73 HVX FFN per layer call, TCP RTT).
5. Checks: M3 check in the plan (two phones vs OP15 alone at equal total fraction; outputs identical; both
   proof rows verified), then the 24-request and realistic traces with both phones.

## 8. OP11 worker smoke tests (2026-09-21 22:40 EDT)

Shard `HTP0.ffn.gguf` (Qwen layers 18-23, 3.21 GB) is on the phone; user freed 19 GB by authorizing deletion of two
old models. Worker runs from `/data/local/tmp/s42-op11-20260921-bin/` with `LD_LIBRARY_PATH=.`:

| backend | layers | outcome |
| --- | --- | --- |
| HTP0 (v73) | 18-23 (6) | HTP session OK (`Hexagon Arch version v73`, hvx 4, vtcm 8 MB) but weight allocation fails: `HTP0 buffer mapping failed ... error 0x1` after ~28 x 94 MB rpcmem buffers; v73 has no `rpcmem_alloc2` and the backend caps mapped VA at 3000 MiB below v75 (`opt_vmem`) |
| HTP0 (v73) | 18-21 (4) | allocates (34 weight buffers) then aborts in the first graph compute: `GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS)` in `ggml_hexagon_opbatch::add_buffer` (16-buffer batch limit; `fit_op` bounds buffers but `add_buffer` still asserts); `GGML_HEXAGON_OPBATCH=0` -> `GGML_ASSERT(n_ops <= n_ops_max)`; `=1` -> abort in `flush_pending`; `=4` -> abort, no assert text |
| GPUOpenCL (Adreno 740) | 18-21 (4) | **ready** (`[ffn-worker] ready backend=QUALCOMM Adreno(TM) layers=4`), 34 weight buffers |

Decision: OP11 joins on **OpenCL over TCP** first (its energy lever is host CPU work removed, not phone speed).
The v73 HTP path is a ggml-hexagon debugging item (buffer chunking without `rpcmem_alloc2` + op-batch buffer
accounting) and stays out of the critical path. Phone-side `pkill -f` needs a bracketed pattern
(`llama-ffn-split-[w]orker`) in its own `adb shell` call, or it kills the shell that carries the pattern.
Open: end-to-end TCP check from the desktop (`adb forward tcp:26900 tcp:26900` + the FFN client), OP11 cost
calibration on OpenCL, then the two-phone rig work (section 7).

### v73 HTP crash signature (2026-09-21 23:15 EDT, after raising `HTP_OP_MAX_BUFS` to 64)

First op batch (171 ops, 35 buffers, 1.6 GB vmem) reaches the DSP; the user PD crashes:
`Process "/frpc/... llama-ffn-split" crashed ... due to TLBMISS RW`, `Bad VA 0x0`, call trace
`op_matmul+0x5818 <- worker_pool_run_jobs <- worker_pool_run_func <- op_matmul+0xCD4 <- htp_iface_start`
in `libggml-htp-v73.so`. The failing ops are the FFN `MUL_MAT f16 x f32` (5120x512 tiles, `hvx-tiled vtcm 673792`)
on an **unsigned PD** (`Unsigned:Y`, `fastrpc_shell_unsigned_3`). Host side then sees `dspqueue_read failed 0x2e`.
Null pointer in the HVX tiled matmul worker: check the VTCM scratch acquisition on v73 unsigned PDs (partition
size), and the kernel's assumptions that hold on v81 (OP15). The 6-layer shard also exceeds the v73 VA mapping
cap; use 3-layer sessions.

### v73 fault located (2026-09-21 23:30 EDT)

`hexagon-addr2line`/`objdump` on `libggml-htp-v73.so` (load 0x20000000): fault PC 0x19218 is the `dmpoll`
instruction inside `hvx_mv_2d` (inlined into `op_matmul`, `hvx-mm-kernels-tiled.h`), i.e. the user-DMA engine's
poll of its descriptor chain, at Bad VA 0. The per-thread DMA queues come from `main.c:428`
`ctx->dma[i] = dma_queue_create(256)` (`hex-dma.c`: `memalign(64, ...)`), and a NULL result is silently
tolerated (only `trace` is skipped). On this v73 unsigned PD the kernel therefore polls a descriptor chain at 0,
or user DMA itself is not usable from the unsigned PD. Fix candidates for the bring-up session, in order:
1. make a NULL `dma_queue_create` fatal with a FARF error (turns a crash into a diagnosable start failure);
2. add a non-DMA path for `__HVX_ARCH__ < 75` in the tiled kernels (vector loads instead of `dmstart`/`dmpoll`);
3. try a signed PD (test signature for serial 832358d4) to rule out the unsigned-PD DMA restriction.
Also: `HTP_OP_MAX_BUFS` raised 16 -> 64 (host-side only; DSP reads the buffer list by count) — keep; 3-layer
sessions for v73 (VA cap); `hmx_enabled = n_hmx` is set from hwinfo (`hmx 1` on v73) while the host disables HMX
below v75 -- check the DSP side does not enter an HMX path on v73.
OpenCL worker remains the working fallback for OP11.

**Correction (2026-09-22 22:00 UTC): the diagnosis above is wrong on both counts.** Raising `HTP_OP_MAX_BUFS`
was not host-side only: the DSP kept a 16-slot mapping table, so batches of more than 16 buffers left the later
bases silently at zero, and the kernel then loaded weights from address 0. That, not a NULL DMA descriptor chain,
is the Bad VA 0 fault; the `dmpoll` was polling a DMA whose source was the unmapped zero base. The 16:36-18:15
UTC work found it (`dma_queue_push_sync+0xc0` vector load in a synchronous-copy diagnostic, then the mapping table
itself), tied the DSP mapping capacity to `HTP_OP_MAX_BUFS` = 64 with a 64-bit reuse mask, slot-pressure eviction
and fatal guards for invalid or exhausted mappings, and qualified the NPU on layers 18-21 with normal DMA and HMX
off. See the 18:15 UTC entry in `research_dev/talks.md` and `reports/20260922-fast-path-M3/README.md`.

### OP11 TCP/OpenCL qualification (2026-09-22 16:40 UTC)

Functional **PASS** for layers 18-21 at 1/2/4 rows: 48 calls, 112 checked
rows, max relative L2 3.274e-4 against CPU (limit 0.01), matching identities
and payload hashes, normal exits. Parent and desktop/phone shard hashes
verified; all 18 stored tensors for layers 18-23 match the parent.

Batched-decode suitability **FAIL** on measured latency: phone round trips
68.652 / 4751.604 / 4749.101 ms at 1/2/4 rows versus CPU
18.254 / 18.850 / 19.653 ms. Most multi-row time is worker compute.
Full M3 integration is deferred by the user. Six-layer simultaneous OpenCL
capacity, real-token correctness and energy benefit are unverified.
See [qualification report](reports/20260922-fast-path-M3/README.md).

### OP11 NPU repair qualified (2026-09-22 18:15 UTC)

**PASS**, superseding the DMA-restriction diagnosis above for the tested FFN
path. Host batch capacity was raised to 64, but the DSP mapping table remained
16 entries. Exhaustion silently left later tensor addresses zero. A diagnostic
HVX-copy bypass also failed at a zero-address source load; normal DMA works
once mapping slots follow HTP_OP_MAX_BUFS, reuse uses a 64-bit mask, and slot
pressure triggers eviction. Invalid/exhausted mappings now fail explicitly.
DMA allocation null checks and cleanup were fixed independently. No signed PD,
phone kernel change or hardware-DMA workaround was needed.

Real Qwen layers 18-21, HMX explicitly disabled, 1/2/4 rows: 48 calls, 112 rows,
max relative L2 0.000167056 (limit 0.01), exact repeats, identity/hash checks and
normal exits PASS. Median NPU round trips 59.622/63.301/77.097 ms; worker
15.803/18.223/31.440 ms. This replaces OpenCL's 4.75 s batched call, but still
has substantial transport/protocol overhead and no measured energy benefit.
The host does not automatically disable HMX on v73 in this tree; qualification
uses GGML_HEXAGON_NHMX=0. Four-layer residency is now tested (2040.13 MiB),
while all six layers together remain unverified and exceed the configured
3000 MiB mapped-VA budget. Layers 22-23 were not numerically retested.

New isolated bin `/data/local/tmp/s42-op11-v73-mmap-20260922-v1`, skel SHA
0025a0e4c500e3f8f1d62dd08009651ce7681444a8f4df0d5e4dc65e345ea374.
Old bin retained. No active worker, forward or queued job remains. OP15 was
not changed. Wi-Fi not measured: OP11 has no wlan0 IPv4/route; user asked to
connect it to a network reachable from the rig. Full M3 integration, real
model token correctness and energy qualification remain deferred/unverified.
See [NPU repair evidence](reports/20260922-fast-path-M3/README.md).
