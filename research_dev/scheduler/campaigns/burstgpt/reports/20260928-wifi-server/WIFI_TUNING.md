# WiFi per-call latency tuning (2026-09-29 00:40-01:20 UTC)

Question (user): "anyway to speedup the wifi speed?"  Answer: bandwidth is not the limit (a 10 KB call is
~0.1 ms of airtime at the negotiated rates). The per-call cost comes from channel access in the best-effort
queue (OP15) and from the phones' CPUs sleeping between calls (both). Two runtime-only phone settings cut the
real workers' network time per call by ~30 % (OP15) and ~18 % (Pixel). `harness/wifi_tune.sh on|off|status`
applies / reverts them (run on the desktop, local adb). All settings were reverted after the measurements.

## Setup

FCHLLX01 (A6000 server) on the router's 10 GbE LAN port (192.168.0.73); both phones on "Boyd612_6G", 6 GHz
channel 69, 80 MHz, single MLO link (AP has no TID-to-link mapping). OP15 192.168.0.194: RSSI -51..-59 dBm,
rx EHT-MCS11 2SS 1,201 Mbps, tx MCS6 648 Mbps (link-layer stats: most uplink MPDUs at nss=1 MCS3-5, ~15 %
retries), power save off, Android low-latency lock held. Pixel 192.168.0.202: RSSI -50, 1,441/1,470 Mbps.
Host load: another user's CPU jobs on FCHLLX01 throughout (as in the S1 benchmark).

## What was tried

| lever | where | OP15 effect | Pixel effect |
|---|---|---|---|
| host DSCP EF (IP_TOS 0xb8, `wifi_rtt.py --tos`) | server socket, downlink only | none (echo p50 10.68 -> 10.53 ms) | none for echo; small pings WORSE (2.4 -> 5.5 ms avg) |
| call gap 0 / 2 / 8 / 20 / 100 ms | client | p50 9.5 / 9.0 / 10.7 / 12.0 / 12.7 ms | 8.0 / 9.9 / 10.5 / 10.6 / 10.3 ms |
| one CPU core busy on the phone | phone | ping avg 5.6 -> 4.8 ms | - |
| all 8 cores busy | phone | - | 10 KB ping 7.0 -> 4.7 ms |
| raise min freq of the WiFi-IRQ cores (policy5) | phone | - | 10 KB ping 6.9 -> 6.4 ms (only ~0.5 ms) |
| **phone-side DSCP 46 on replies** (iptables mangle OUTPUT) | phone uplink | **6/6 alternating rounds**: echo p50 10.64-10.75 -> 8.61-8.80 ms; 10 KB ping 6.16-6.25 -> 4.83-5.22 | none |
| **disable CPU idle states above WFI** | phone | (combined below) | 10 KB ping 6.97 -> 5.00 ms |

OP15 link-layer stats explain the DSCP result: best-effort contention time avg 1,198 us (max 10,170 us, 3,963
samples) vs voice 172 us, at 0.07 % channel busy. The Pixel's WiFi interrupts land on CPU 6 (policy5, idles at
400 MHz) and its deep idle state costs ~0.4-0.5 ms per wake; several wakes per 10 KB exchange.

## Real FFN workers (harness `server_bench.py probe`, 300 calls/rows, gap 8 ms)

rpc p50 / compute p50 / network+overhead p50, ms (files `harness/results/20260929-probe-ef/`):

| setting | op15-htp0..2 rows 1 | op15 rows 4 | pixel rows 1 | pixel rows 4 |
|---|---|---|---|---|
| default (2 rounds) | 14.5-14.8 / 7.7 / 6.7-7.1 | 18.3-18.5 / 11.2 / 7.1-7.3 | 10.7-10.8 / 6.1 / 4.60 | 18.0-18.2 / 13.1-13.3 / 4.7 |
| OP15 DSCP only (2 rounds) | 13.1-13.5 / 7.7 / 5.4-5.7 | 17.0-18.0 / 11.2 / 5.8-6.7 | 10.7 / 6.1 / 4.60 | 18.1-18.3 / 13.2-13.3 / 4.7 |
| Pixel idle off | (unchanged) | | 9.73 / 5.96 / 3.75 | 17.6 / 13.3 / 3.96 |
| **all tuned** (OP15 DSCP + idle off, Pixel idle off) | **12.5 / 7.6 / 4.83-4.87** | 17.1-17.4 / 11.1 / 5.9-6.2 | **9.77 / 5.94 / 3.77** | 17.9 / 13.4 / 4.1 |
| USB reference (desktop) | 9.6 / 8.9 / ~0.7-1.5 | | | |

Network time per call: OP15 -2.0 ms (-30 %), Pixel -0.85 ms (-18 %). WiFi is still ~3-5x the USB per-call cost.
p99 unchanged (~17 ms rows 1). Echo check through the script: OP15 p50 7.52 ms (default 10.6-10.8).

## What it means for the S1 server result (projection, NOT re-measured)

S1 (Qwen3-14B dequant-f16, -ngl 16, 24 CPU-resident layers, 18 OP15 + 6 Pixel calls/token): cpu 246.3 ms/step,
split-25 254.1 (+3 %), phone 423.5 (+72 %) at c=1.

* Tuning saves ~18 x 2.0 + 6 x 0.85 = ~41 ms/token on the phone arm -> ~382 ms, still ~+55 % vs cpu.
* Even at ZERO network time, full offload cannot win on this server: the phones' compute alone is
  6 x (7.75 + 7.82 + 7.82 + 6.75) = 181 ms/token vs the server CPU's 24 x 534.8 MB / 76.2 GB/s = 168 ms.
* split-25 tuned: OP15 call ~4.85 + 2.1 = ~7.0 ms vs the full-CPU layer ~7.0 ms (no gain on 18 layers);
  Pixel ~5.6 ms (~-1.4 ms on 6 layers) -> roughly break-even (~-3 %), inside the host-contamination noise.
* The GPU board draws ~100 W while it waits on the CPU part, so any slowdown costs GPU energy; server CPU
  package energy is not measurable (RAPL root-only), phone energy over WiFi not measured.

Conclusion: WiFi tuning is worth keeping (free for DSCP; idle-off costs phone power), but it does not change
the S1 verdict. The per-call network cost must be small relative to the phone's slice of a layer. That needs
bigger layers (a model whose CPU-resident FFN takes well over 10 ms per layer), fewer exchanges per token
(whole-block ownership), or both.

## Not done / open

* Host-side 10 KB echo tool is pessimistic (toybox `nc -L ... cat`, pipe hop); use the worker probe for claims.
* Phone power with idle states disabled: unmeasured.
* Router-side levers (TWT off, 160/320 MHz, WMM/game QoS) need the router UI (user). The OP15 supports 320 MHz
  (max 5,764 Mbps); the AP runs 80 MHz. Wider channels mostly cut airtime, which is not the bottleneck.
* S1 server arms not re-run with tuning.
