# WiFi transport + stronger-server energy test (plan, 2026-09-28)

Goal (user): show the phone helpers work over a local WiFi 7 network (today: USB FunctionFS for the OnePlus 15,
adb-forwarded TCP over USB for the Pixel), and that a stronger server (FCHLLX01: RTX A6000 ×2, Threadripper PRO
5995WX 64C, 125 GB RAM, CPU streams ~69 GB/s) still saves energy with phones.

## What already exists (no new phone code)
- Phone worker TCP mode: `llama-ffn-split-worker --backend <HTPn|CPU> --port P --bind ADDRESS ...`
  (`examples/layersplit/ffn-split-worker.cpp:359-441`; TCP and FunctionFS/dmabuf are mutually exclusive).
  `--bind 0.0.0.0` (or the phone's WLAN IP) serves over WiFi.
- Server helper client: `S41SERVERFFNHELPER ... transport=tcp host=<ip> port=<p>` — host is configurable (today
  127.0.0.1 via adb forward for the Pixel).
- Only activations cross the link: 10,240 B per row per call for Qwen3-14B (7,680 B for Gemma-4-12B). Bandwidth is
  irrelevant (1 Gb/s = 0.08 ms); the per-call ROUND-TRIP delay and its tail are what matter, because a token makes one
  synchronous exchange per phone-owned layer (18-24 per token today, one after another).

## Stage W1 — WiFi round trip (1 hour, go/no-go)
Tool: `wifi_rtt.py` (this directory). Phone: `toybox nc -L -p 7070 cat` (echo). Host: `python3 wifi_rtt.py --host
<phone-ip> --calls 5000 --gap-ms 8`. Run: each phone, idle; both phones at once; with the phone NPU busy (worker
running a local loop); server wired to the router. Reference: USB FunctionFS 1.5 ms transport per call (measured),
adb-forward TCP (Pixel) ~1.5-3 ms. **Go** if p99 ≤ 5 ms and p99.9 ≤ 15 ms per call; the per-token transport p99
(24 calls) is printed directly.

## Stage W2 — real FFN calls over WiFi (half day)
Run the worker with `--port 7070 --bind 0.0.0.0` on each phone (same shards as today) and replay the qualification
call pattern (rows 1/2/4) from the host; compare call round trip and per-token period with USB (OP15 9.6 ms per
Qwen call, 8.9 compute) and adb-forward TCP (Pixel). Check outputs byte-identical to the USB path.

## Stage W3 — end-to-end on the DESKTOP over WiFi (1 day incl. scheduler transport)
Scheduler: a `wlan-tcp` phone transport (host = phone WLAN IP, no adb forward; identity/qualification receipts
`wlan-round-trip` from W1/W2; liveness/join over TCP). Run the standard two-phone eval_v2 arm with both helpers on
WiFi vs the USB result (pe1 95.7 kJ, measured phone energy with `run_phone_energy.sh`). This isolates the WiFi
effect with everything else fixed. Expected: token period + (WiFi RTT − 1.5 ms) × calls; phone power + radio.

## Stage S1 — stronger server, mechanism microbenchmark (1 day)
Rebuild `build-cuda` on FCHLLX01 from main (current build is 2026-09-20 and lacks S2a mask-out/caps). Model:
the SAME artifact the phones already hold (`Qwen3-14B-Q4KM-dequant-f16.gguf`, copy from the desktop) with GPU
layers set so the phone-owned layers 0-17 are CPU-resident (forced spill ~13 GB) — mechanism only. Measure per
layer and per token: CPU alone vs phone alone vs CPU+phone column split (shares 25/50/75 %, column quantum 4352),
at batch 1/2/4, over WiFi. Expected: phone ≈ server CPU per byte (8.9 ms vs ~8.9 ms per Qwen FFN layer), so only the
split helps: CPU 69 + 2 × 55 GB/s ≈ 180 GB/s → CPU-resident FFN up to ~2.6× faster minus transport.

## Stage S2 — stronger server, realistic model (2-3 days)
A model whose spill ≈ what the phones can hold (~20 GB for two phones): Qwen3-32B f16 on one A6000 (~65 GB, spill
~20 GB) — same family/format the phones run, needs new shards + qualification; or two models, one per GPU, each
spilling (mirrors today's two-model desktop). Baseline = the same server without phones. Estimates (not measured):
−30-35 % time/energy at 3 ms per call, ~0 at 10 ms per call.

## Prerequisites the user must provide
1. The WiFi 7 router placed near the phones; both phones joined to it (keep USB to the desktop for power + adb).
2. FCHLLX01's free second port `enp2s0` cabled to a router LAN port (campus link on `enp1s0f0` untouched);
   the desktop wired to the router as well for W3. Router: one subnet, client/AP isolation OFF.
3. Energy on FCHLLX01: read access to RAPL (`/sys/class/powercap/intel-rapl:0/energy_uj` is root-only; e.g. a
   tmpfiles/udev rule or a sudoers rule like the desktop's) — ideally a metering smart plug for wall power
   (static power is the lever on this host).
4. A quiet window on FCHLLX01: another user's SGLang runs at ~3,200 % CPU (load avg ~40), which contaminates the CPU
   measurements this test is about.
