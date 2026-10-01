# Paper configuration v1 (frozen 2026-09-29)

The single scheduler configuration the paper's full-system arm uses. It is the desktop template
`template-eval2-s2` of the two-phone evaluation, copied here unchanged together with its trace, transport
identity and launch scripts. Code: the commit that adds this folder. Nothing here is tuned per run.

## What is frozen

| file | content |
|---|---|
| `template/campaign.json` | scheduler policy (below) |
| `template/models.json` | Qwen3-14B Q4_K_M dequantized to f16, Gemma-4-12B Q4_0 dequantized to f16, Llama-3.2-1B-Instruct Q4_0; phone FFN shard locations |
| `template/rig.json` | desktop RTX 4060 Ti 16 GB + CPU, OnePlus 15 over USB FunctionFS (DMA-BUF), binaries, pinned phone kernel and boot image |
| `template/evidence.json` | calibration and observation stores on the desktop (paths; hashed in `DESKTOP_MANIFEST.json`) |
| `template/CHANGES.txt` | how this template derives from the base eval template |
| `trace/longtail_eval_v2/` | BurstGPT-derived trace: 14 requests, 5,385 input / 3,604 output tokens, 1,675 s arrival span |
| `identity/` | transport qualification identity the run binds (`S43_TRANSPORT_IDENTITY`) |
| `scripts/` | desktop launch, readiness, phone-energy and analysis scripts as they ran |
| `DESKTOP_MANIFEST.json` | sha256 of every desktop file the template references (43 paths; two model files > 1 GiB are pinned by `models.json`) |
| `reference_run_s2a/` | evaluation output of the reference run of this exact configuration |

Scheduler policy (`template/campaign.json`):

| setting | value |
|---|---|
| selection mode | energy-aware |
| admission | work-conserving, model affinity |
| continuous join | on, barrier extension up to 120 s |
| residency hysteresis | 20 s |
| late helper adoption | on, at least 24 remaining tokens |
| server policy coherence | on |
| batch-growth verdict inheritance | on |
| phone re-provisioning | counts queued demand, starts early on a model transition |
| adaptive probes | at most 4 per context |
| latency bound | 1.25x the desktop-only estimate |
| minimum energy saving to offload | 1 % |
| host memory budget | 28 GiB |

Not part of v1: the GPU device-power controller (dp arms). It currently receives the next trace arrival before it
happens, which is future knowledge; it returns as a separate arm once it uses an online arrival estimate.

## How to run it (desktop `zhihao@172.20.74.85`)

1. OP15 on the qualified RAM-booted kernel `6.12.23-android16-5-o-g227664cbe007-4k` (boot image sha256
   `f13c7c03...`, recorder `boot_candidate.py` from a fresh copy), then `scripts/s43_g2_recreate.sh` as root on
   the phone. The Pixel needs no kernel change (adb forward transport).
2. `scripts/rig_ready.sh` must print READY: OP15 thermal status 0, at most 32 C, charger not latched, rig lock free.
3. `scripts/launch_arm.sh <attempt> template-eval2-s2` prepares the inputs and runs the two-phone arm under one rig lock.
4. `scripts/post_eval.sh <out-prefix> legacy legacy=<all-desktop inputs> <label>=<inputs> ...` evaluates the arms.

All paths inside the template are the desktop's absolute paths; this folder records them, it does not relocate them.

## Reference run s2a (2026-09-29 04:00-04:32 UTC)

| run | host kJ | vs all-desktop | identical outputs | median latency | slowest-10 % latency |
|---|---|---|---|---|---|
| all-desktop baseline | 228.5 | - | - | - | - |
| s2a (this config) | 94.6 | -58.6 % | 13/14 | 240 s | 478 s |
| s1c (without inheritance) | 83.9 | -63.3 % | 11/14 | 312 s | 803 s |
| s1d (without inheritance) | 103.1 | -54.9 % | 12/14 | 186 s | 468 s |

Host energy is CPU package plus GPU board. Phone energy in these rows is assumed, not measured. Latency is
arrival to completion. Single runs; the paper needs repeats.

Known gap in v1, corrected 2026-09-30: in s2a the Qwen pair 005/007 decoded about 170 s without phones. The
re-evaluation of the phone weights did run when the Gemma pair released the phone, but every OP15 route was refused
with THERMAL_LIMIT after the pair had driven the phone at full fraction, so the preparation never started, and the
refusal left no record (the thermal state between 1,053 s and 1,272 s is inferred). An earlier note here blamed a
missing retry; that was wrong. The opt-in `dispatch_policy.event_replanning` (added after v1) records such blocks
and re-evaluates on every helper release and device recovery.
