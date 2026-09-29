# Report data sheet — phone-assisted scheduler evaluation (2026-09-25)

Numbers a report can quote today. Every row is a real run on the rig; "host kJ" is the measured desktop
energy over the paid trace interval (RAPL CPU package + NVML GPU board). Phone energy is an assumed model
(4.5 W active / 0.875 W idle), NOT measured — quote host energy, mention the phone assumption.
Sources: `/mnt/storage/s43-two-phone-eval-20260925/analysis/{EV2,LT,DEV2}_ENERGY.txt`, `*_ANALYSIS.txt`.

2026-09-27 audit: saved energies and g11 same-server recovery PASS. Best measured host saving remains
58.687%; latest undisturbed tp2 saves 50.911%, g11 saves 46.862%. Strict output identity remains FAIL.
The fresh suite had one timing assertion failure (10.475 ms vs 10 ms), which passed in isolation;
463 scheduler Python files match the working checkout, stage and deploy. See the
[independent audit](../20260927-results-audit/README.md) for source hashes and reporting corrections.

## Setup

| item | value |
| --- | --- |
| desktop | RTX 4060 Ti (16 GB) + host CPU, Qwen3-14B F16 (dequantized Q4_K_M) as the hot model, Gemma-4-12B F16 as the cold model, Llama-3.2-1B overlay |
| phones | OnePlus 15 (Hexagon NPU, FunctionFS DMA-BUF over USB 3, 5 Gb/s); Pixel 10 Pro (packed-CPU worker over adb TCP, DVFS-fixed) |
| offload unit | dense FFN layers, decode only ("dormant host share"); server keeps prefill; outputs checked token-by-token vs the desktop |
| traces | BurstGPT logs: real arrival times and token counts, synthesized prompt text; long-tail share matched within 0.05 (log) |
| arms | **legacy**: all-desktop, old dispatcher · **desktop+DP**: all-desktop + work-conserving admission + model affinity (isolates the scheduler from the phones) · **OP15 all-on**: one phone, per-device policies · **OP15+Pixel**: two phones, per-device policies (device set × fraction chosen per batch composition from measured evidence) |

## A. `longtail_eval_v2` — 14 requests (7 Qwen / 6 Gemma / 1 Llama), 3,604 output tokens, 1,675 s arrival span (single runs, 2026-09-25)

| arm | status | duration s | host kJ | host W | vs baseline | phone FFN calls | identical outputs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **baseline** (legacy all-desktop) | PASS | 2,244 | **228.5** (CPU 159.9 + GPU 68.7) | 101.8 | — | 0 | — |
| **baseline + OP15** (all-on, per-device policy, one phone) | PASS | 1,822 (−19 %) | **104.5** (CPU 50.4 + GPU 54.1) | 57.3 | **−54.3 %** (fleet −52.8 %) | OP15: gemma 30,600 + qwen 26,262 | 12/14 |
| **baseline + OP15 + Pixel** (per-device policy, two phones) | PASS | 1,833 (−18 %) | **94.4** (CPU 38.3 + GPU 56.1) | 51.5 | **−58.7 %** (fleet −57.1 %) | OP15: gemma 40,488 + qwen 23,364; **Pixel: qwen 7,176** | 13/14 |
| desktop + dispatcher (scheduler only, no phones) | PASS | 1,835 (−18 %) | **178.2** (CPU 123.5 + GPU 54.7) | 97.1 | **−22.0 %** | 0 | 13/14 |

Adding the Pixel to the OP15 saves a further **9.622% host energy** in the first controls
(104.465 to 94.414 kJ); duration increases 0.609%. This incremental benefit is not yet repeatable evidence.

**Decomposition of the 134.1 kJ saved by the full system (baseline → +OP15+Pixel):** dispatcher (work-conserving
admission + model affinity, no phones) 50.3 kJ = 37 % — it is also what shortens the run (2,244 → 1,835 s);
OP15 offload 73.7 kJ = 55 %; Pixel offload 10.1 kJ = 8 %. Relative to the desktop+dispatcher control the phones
save −41.4 % (OP15) and −47.0 % (OP15+Pixel) at the same duration — the phones lower power, not time.
Model loads: 9 (425 s) baseline vs 7 (289 s / 338 s) in both phone arms; switches 5.

**Why the Pixel was used — the policy's own evidence (two-phone run, Qwen, fleet J/token incl. the assumed
phone power):**

| Qwen batch | OP15 only (probe) | OP15 + Pixel (chosen) | tokens served by the chosen set |
| --- | --- | --- | --- |
| B1 | 35.8 J/tok (12 probe tokens) | **27.3 J/tok** | 238 |
| B2 | 38.5 J/tok (32) | **27.3 J/tok** | 268 |
| B3 | 40.3 J/tok (36) | **30.9 J/tok** | 666 |

The scheduler probed OP15 alone first (its lower-bound rule), measured the two-phone set better on every Qwen
composition and switched to it; no device-specific rule is coded. Gemma stayed OP15-only by evidence.
Pixel per-layer RPC 12.7 / 14.2–14.7 / 17.4 ms at B1 / B2 / B3 (compute 8.2 / 9.5 / 11.9 ms) vs OP15
10.4–12.2 ms. (Pixel owns Qwen layers 18–23.)

## A2. Robustness gate H1 — the scheduler demotes a slow device by evidence (2026-09-25 18:32 UTC)

Same two-phone arm, same trace, but the Pixel runs its OLD slow worker (38 / 54 ms per layer at B1 / B2 instead of
13 / 14). The scheduler is told nothing.

| arm | Pixel worker | dur s | host kJ | vs baseline | Pixel calls | identical |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| + OP15 only | — | 1,822 | 104.5 | −54.3 % | 0 | 12/14 |
| + OP15 + Pixel (fast) | 13 ms/layer | 1,833 | 94.4 | −58.7 % | 7,176 | 13/14 |
| **H1: + OP15 + Pixel (slow)** | 38 ms/layer | 2,033 | **109.2** | −52.2 % | **240** | 13/14 |

What the policy did (from its own windows, Qwen, fleet J/token): probed OP15+Pixel at B1 (8 tokens: 39.3 vs
OP15-alone 34.7) and at B2 (24 tokens: 41.4 vs 36.7) → dropped the two-phone set for both compositions with
`SERVER_DEVICE_SET_NOT_IMPROVED`, then ran Qwen on the OP15 alone (753 + 578 tokens). Host energy lands within
4.5 % of the OP15-only run (noise band ≈ 3–5 %), outputs unchanged. No device-specific rule is involved — the
same code admitted the fast Pixel earlier in the day.

## A3. Robustness gates — phones dropping and joining at runtime (2026-09-25/26, `elastic_phones` on)

All on `longtail_eval_v2`, two-phone per-device arm. The fault tool SIGTERMs the Pixel FFN worker (authorized);
`active` = the kill lands while the Pixel is serving a layer call.

Here `penalty` is the discarded attempt's wall time, not recovery downtime or a matched end-to-end
latency increment. The recovery re-executes the prompt and discards the partial generated tokens.
Fault timing, affected requests and thermal conditions differ between gates; their energy differences
do not isolate the recovery implementation.

| gate | fault | run | host kJ | vs baseline | what the scheduler did | identical |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | Pixel 3× slower (old worker) | PASS 2,033 s | 109.2 | −52.2 % | probed OP15+Pixel at B1/B2, dropped it (`NOT_IMPROVED`), ran OP15-only | 13/14 |
| G1c | worker killed while idle | PASS 1,932 s | 102.2 | −55.3 % | loss flagged at the first liveness check (47 µs after the kill), quarantined, identity-verified re-join 65 s later, Pixel served 8,088 calls afterwards | 11/14 |
| **G1g** | **worker killed mid-call** | **PASS 1,795 s** | **112.9** | **−50.6 %** | `helper_lost` classified from the server's helper error → poisoned Qwen server retired (`SERVER_EXITED`, rc 0) → request 001 re-planned onto the desktop route and re-executed (95 tokens discarded, 60.6 s penalty, partial stream kept as `.attempt1`) → Qwen reloaded → co-tenants continued → Pixel quarantined at +5 s and re-admitted at +65 s → 7,338 Pixel calls afterwards | 10/14 |
| g9 (S2a `mask_out`) | **worker killed mid-call**, rebuilt server | PASS 1,948 s | 137.7 | −39.8 % | `HELPER_MASKED_OUT` → request 001 recovered on the SAME live server (`same_server_mask_out`, 60 tokens discarded, penalty 40 s), NO `SERVER_EXITED` → but the dispatcher switched the freed server to Gemma 4 s later (mask ended `SERVER_STOPPED`), so the reload was avoided only nominally; Pixel re-joined at +65 s; thermal gate active (512 THERMAL_LIMIT mentions) → low Qwen assistance dominates the energy | 11/14 |
| **g11 (S2a `mask_out` + dispatcher fix)** | **worker killed mid-call**, mask recorded at 505.5 s | **PASS 1,814 s** | **121.4** | **-46.9%** | Two in-flight Qwen requests (003, 004) recovered on the same live server with `RECOVERY_RETAINED_QUEUE_PLACE`; 71/72 tokens discarded, 47.0/47.2 s of failed-attempt time. No server stop or reload during recovery. Pixel live-reconnected 64.731 s after masking. OP15 thermal deferrals total 528.898 s; their share of the energy gap is not isolated. | 12/14 |
| undisturbed | — | PASS 1,833 s | 94.4 | −58.7 % | (reference) | 13/14 |

Output identity vs the all-desktop baseline (token-by-token): **the G1g recovered request 001 is identical** (124/124
tokens). The requests that differ have equal lengths; no saved logits establish an FP near-tie exception.
Undisturbed run: 004 (first
difference at token 137/319); G1g: 003 (203/262), 004 (233/319), 008 (82/168), 009 (24/53). G1g also assisted
only 2 of 6 Gemma requests (undisturbed: 6/6) while the Qwen server was being replaced — part of the +18 kJ.

Cost of a mid-run device loss on this trace: ≈ +18 kJ vs the undisturbed run (reload + 95 re-decoded tokens
+ less Gemma assistance while the Qwen server was replaced), still −50.6 % vs the all-desktop baseline. No
request failed, no run aborted. The path was hardened by four hardware iterations (G1 idle kill → G1b
control-path classification → G1d/G1e live-server retire + fresh snapshot → G1f stale adaptive registration),
each with a recorded regression test (`tests/test_elastic_*.py`, 145/145 modules). The mask-out path needed one
more iteration (g9 → g11): a recovery attempt re-entered the dispatch queue as a NEW admission, so a waiting
other-model switch was ordered ahead of it by arrival order and stopped the freed server 4 s after the mask;
the fix keeps the failed attempt's queue place for the recovery (`tests/test_elastic_recovery_dispatch.py`, 12
tests, 6 of which reproduce g9's order on the pre-fix code). Both recovery modes are now hardware-proven:
`retire` (G1g: reload, 60.6 s penalty) and `mask_out` (g11: no reload, ~47 s penalty, live re-attach).

## A4. Repeats (2026-09-26, `ev8`, elastic off) — run-to-run spread

| arm | run 1 | run 2 | spread |
| --- | --- | --- | --- |
| legacy all-desktop | 228.5 kJ / 2,244 s | 250.4 kJ / 2,429 s | +9.5 % energy, +8 % duration |
| OP15+Pixel | 94.4 kJ / 1,833 s | 120.6 kJ / 1,833 s | +28 % energy, same duration — run 2 assisted Qwen for only ~230 tokens (OP15 Qwen calls 2,940 vs 23,364): 469× `phone helper replacement source is not ready`, 85× `HELPER_REMATERIALIZATION_FAILED` ("ready layout produced no helper opportunity"), 301 `PHONE_HELPER_UNAVAILABLE` decisions → the OP15 re-provisioning Gemma→Qwen stalled; Gemma assistance normal (38,376 calls). Root cause: a thermal gate on the OP15 during the Gemma→Qwen load excluded every phone route at ticket time, the fallback then bound an UNQUALIFIED `split-row` batch plan to the desktop server for its whole life ("no helper opportunity"), and the two-phone envelope could never be extended ("immutable"). Fixed 2026-09-26 (F1/F2/F3, 15 replay tests); the thermal gate itself remains a variance source. |
| desktop + dispatcher | 178.2 kJ / 1,835 s | 174.1 kJ / 1,810 s | −2.3 % energy, −1.4 % duration (the scheduler-only arm is the most reproducible one) |
| OP15 all-on | 104.5 kJ / 1,822 s | 111.2 kJ / 1,836 s | +6.4 %; run 2 again barely assisted Qwen (OP15 Qwen calls 2,016 vs 26,262) while Gemma was fully assisted (47,400) — the same re-provisioning stall; Gemma assistance alone carries most of the OP15 saving |

**Confirming run 3 (tp2, 2026-09-26 08:04-08:35 UTC, provisioning fixes F1-F3 deployed, elastic off):**
112.186 kJ / 1848.969 s, 50.911% host saving versus the first legacy reference, 12/14 outputs exact.
The three provisioning-stall strings are absent from its decision log. OP15 served 42,216 Gemma calls
and 5,526 Qwen calls; Pixel served 1,566 Qwen calls. Snapshot qualification is false for 514.662 s
during the Qwen window, plus a later unclosed episode whose saved samples span another 35.649 s.
Maximum saved OP15 executor temperature is 67.8 C. The snapshots do not retain the raw Android severity,
so LIGHT/status 1 throughout that interval is not independently verified. Thermal gating is a confirmed
coverage loss; whether relaxing it closes the energy gap remains untested. The decision log contains
706 literal `THERMAL_LIMIT` and 272 `PHONE_HELPER_UNAVAILABLE` occurrences, not unique event counts;
the earlier 35,290/544 numbers are not reproduced by that scope. The 90 C temperature check is a fallback
when qualification is unknown, not a separate ceiling when a known verdict is accepted.

| two-phone run | host kJ | OP15 thermal-excluded | OP15 Qwen calls | note |
| --- | --- | --- | --- | --- |
| run 1 (09-25) | 94.4 | 0 s | 23,364 | reference |
| run 2 (ev8) | 120.6 | 570.296 s sampled interval | 2,940 | re-provisioning stall (fixed) |
| run 3 (tp2) | 112.2 | 514.662 s long interval plus later unclosed episode | 5,526 | stall gone; thermal intervention still untested |

**2026-09-28 gates (scheduler features, elastic off, OP15 cool at start):** cj2 (`dispatch_policy.continuous_join`, bound 120 s)
PASS 98.3 kJ / 1,832 s (−57.0 %), 13/14 identical: Qwen OP15 calls 23,274 ≈ run 1 (no thermal exclusion), a 3-row assisted
Qwen batch (4,266 calls at rows=3) and the 006→002 Gemma join — but the new bypass never engaged (affinity refused
004's displacement first and the code skipped the join bypass; fixed in main, cj3 pending). dp1 (`device_power`: GPU
210 MHz in idle gaps ≥ 60 s and during loads) PASS 97.3 kJ / 1,802 s (−57.4 %): loads ran at 15.1 W GPU instead of
~38 W, idle gaps at 14.4 W instead of 20-28 W, 27 state events, clocks restored at the end; decode (1,486 s, 32.6 W)
untouched → dp2 adds a 1,200 MHz SM cap during decode.

| gate | host kJ | vs legacy | identical | note |
| --- | --- | --- | --- | --- |
| s2a (s1 + batch-growth verdict inheritance; frozen as paper_config_v1) | 94.6 | −58.6 % | 13/14 | inheritance fired 240×, probe-budget exhaustion 0 (s1c 526, s1d 1,632); Gemma 002+006 both assisted through the join (Gemma OP15 47,208 calls); p50/p90 arrival-to-end 240/478 s; new gap: phone re-provision to Qwen deferred IN_USE at 1,052.9 s and never retried → Qwen 005/007 host-only ~170 s (133 PHONE_HELPER_UNAVAILABLE), Qwen OP15 17,244 calls |
| cj2 | 98.3 | −57.0 % | 13/14 | batching happened (rows=3), new bypass not reached |
| dp1 | 97.3 | −57.4 % | — | LOAD_MIN 222 s @ 15.1 W, IDLE_MIN 88 s @ 14.4 W |
| dp2 (dp1 + decode cap 1,200 MHz) | 125.0 | −45.3 % | 12/14 | decode GPU 32.6 → 27.3 W at unchanged token periods; prefill 1.3-1.9× slower; run took the same order fork as cj3 (003 solo, 012 waited 393 s) |
| s1d (+ publication re-plan) | 103.1 | −54.9 % | 12/14 | best latency (p50 214 s, p90 534 s, waits 1,557 s); late adoption fired ×2; Gemma pair dropped to host-only for ~200 s after the join (probe budget exhausted at batch 2) |
| s1c (step-1 fix: conditional hysteresis, retained late helper, queued-demand re-provision) | **83.9** | **−63.3 %** | 11/14 | best energy; 001 assisted from its first token; but 003/004 not re-evaluated during 001's decode → 4-row batch at 1,141 s; p50/p90 latency 312/894 s vs 197-223/539-591 |
| s1b (repeat of s1a, same code) | 97.7 | −57.2 % | 13/14 | same batch structure as s1a (001+003+004 …); Gemma OP15 calls 28,584 vs 47,568 → CPU +10 kJ |
| s1a (step 1: + hysteresis 20 s, late adoption, early re-provision) | **88.0** | **−61.5 %** | 11/14 | best of the campaign; 3-row Qwen batch 001+003+004; but hysteresis held 5× admitting 0, late adoption and early re-provision never fired → gain from timing (001 desktop-only held no phone lanes) |
| pe1 (standard arm, OP15 charging disabled) | 95.7 | −58.1 % | 10/14 | **measured OP15 energy 9.7 kJ (5.3 W mean: USB 2.47 W at the 500 mA cap + 5.2 kJ battery) vs 4.7 kJ assumed** → fleet 107.6 kJ, −53 % |
| cj5 (join-v3 fix) | 129.9 | −43.2 % | 11/14 | join precondition never arose (003 arrived 2 s after 001 ended → switch started first); Qwen phone assistance collapsed (4,302 calls; fractions [0]) |
| cj4 (join-refusal fix) | 115.8 | −49.3 % | 11/14 | lowest waits (1,743 s), 003+004 admitted as 001 finished; join bypass still 0 (refusal = joiner placed at the holder's lease horizon, desktop-parent route not chosen); Gemma window lost assistance |
| cj3 (join fix deployed) | 130.4 | −42.9 % | 12/14 | bypass engaged and refused 5× ("does not start earlier"); 003 waited 443 s → extra reload; run-to-run fork on whether 003 lands in 001's 2.5 s cohort window |

Token identity between the two all-desktop runs is 12/14. This establishes baseline repeat variability;
it does not waive the strict output-identity gate or establish near-ties for the treatment divergences.

## B. `longtail_v1` — 31 requests, 8,207 output tokens (single runs, 2026-09-25 03:44–06:20 UTC)

| arm | duration s | host kJ | host W | vs legacy | model loads | switches | identical outputs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| legacy all-desktop | 4,984 | 536.3 (CPU 377.0 + GPU 159.3) | 107.6 | — | 15 (630 s) | 10 | — |
| desktop + dispatcher | 3,381 | **393.0** (CPU 279.6 + GPU 113.5) | 116.2 | **−26.7 %** | 8 (225 s) | 4 | 14/31 (rest near-ties, same lengths) |
| OP15 all-on | crashed at 431 s (replan-path bug, fixed 09:00 UTC) | | | | | | |
| OP15+Pixel | not run (superseded by eval_v2) | | | | | | |

## C. `longtail_dev_v2` — 9 requests, 1,420 output tokens, arrival scale 0.4 (8 same-model overlaps), 2026-09-25 (matched runs, same tree)

| arm | duration s | host kJ | host W | vs desktop+DP | phone calls | identical outputs |
| --- | --- | --- | --- | --- | --- | --- |
| desktop + dispatcher | 627 | 64.0 | 102.0 | — | 0 | — |
| OP15 all-on r1 | 635 | **34.5** | 54.3 | **−46.1 %** | gemma 18,120 + qwen 9,432 | 5/9 |
| OP15 all-on r2 | 631 | 41.9 | 66.4 | −34.6 % | | |
| OP15+Pixel r1 | 729 | 48.4 | 66.3 | −24.4 % | gemma:op15 9,664, qwen:op15 8,041, qwen:pixel 1,608 | 4/9 |
| OP15+Pixel r2 | 622 | **35.6** | 57.3 | **−44.3 %** | | |

Run-to-run spread on this short trace is ~20 % (which requests get assisted depends on when phone
sessions become ready), so quote the pair of runs, not one. Earlier (2026-09-24) legacy all-desktop on
the same trace: 96.8 kJ; desktop+DP then 82.3 kJ → the dispatcher alone was −15 %.

## D. Per-device facts (for the "why" paragraphs)

| fact | value |
| --- | --- |
| OP15 FFN layer time (NPU, USB) | ~10–13 ms per layer at B1–B4 |
| Pixel FFN layer time before fix (DVFS at server cadence) | 38 / 61 / 88 ms at B1 / B2 / B4 |
| Pixel after fix (uclamp floor + spin poll + batch-pair) | 13.6 / 20.2 / 32.5 ms, outputs byte-identical |
| Pixel CPU+GPU column split | 1.01× (rejected: GPU 2.4–5× slower than packed CPU, interference) |
| OP15 NPU+GPU dual engine | 1.22× in the bench, ~1.0× over USB (parked) |
| OP15 charging on the desktop port | SDP 2.5 W; net-discharges under load; charger latch cleared by reboot + re-boot of the qualified kernel |
| phone power (1 Hz sysfs diagnostic) | OP15 4–6 W assisting; Pixel 1.3–2.0 W (the 4.5 W model overstates the Pixel ~2×) |

## E. Caveats to state

- Single runs per arm (eval_v2) — no confidence intervals yet; dev_v2 gives the spread.
- Phone energy is an assumed model; host energy is measured. Savings quoted are HOST energy.
- Output identity: phone-assisted decode is checked token-by-token against the desktop; "near-ties"
  are FP-accumulation-order differences (same length, same content up to a late token).
- Nothing is committed to git; runs live under `/mnt/storage/s43-two-phone-eval-20260925/`.
