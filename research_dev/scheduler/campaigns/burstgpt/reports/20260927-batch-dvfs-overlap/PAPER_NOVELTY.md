# Systems-paper framing (draft, 2026-09-28)

## One-sentence thesis
On a user-owned heterogeneous fleet (VRAM-bound desktop + phones as FFN helpers), LLM decode is bandwidth-bound
everywhere, so energy per token is decided by *coincidences* — whether independent requests share a step, whether
the helper holds the right shards when the step starts, whether idle silicon is asleep, whether the phone is
thermally admissible — and a scheduler that manufactures those coincidences instead of leaving them to arrival
timing recovers the 25-30 % of energy a work-conserving scheduler leaves on the table.

## Measured observations that carry the paper (all on hardware, RAPL + NVML, byte-identical outputs)
| # | observation | evidence |
|---|-------------|----------|
| O1 | decode step cost is flat in rows | desktop 611 ms at B=1 vs 615-632 at B=4; phone call 9.6 → 12.7 ms for 1 → 4 rows |
| O2 | phones replace CPU weight streaming at ~¼ the power | 228.5 → 94.4 kJ (−59 %) on the 14-request trace; token only 12-16 % faster |
| O3 | energy is decided by discrete timing forks | same trace 97-130 kJ across 6 runs; forks: cohort window (2.5 s), switch start vs same-model arrival (2 s), phone shards ready vs session start (2-30 s), thermal status flip |
| O4 | idle silicon dominates | GPU 8 % busy = 49 % of energy at 34 W; model switches 22 % of time; OP15 busy 17 % |
| O5 | devices churn | thermal exclusion ~500 s/run at Android status 1; USB worker loss; charger latch |

## Contributions (and what is genuinely new)
C1 **Phones as bandwidth donors.** FFN-only offload over USB keeps attention and KV on the host; per-layer synchronous
   helper contract with qualification receipts (row limits, payload sizes, identity of every binary) and per-device
   policies decided from measured evidence. New in topology and in the fail-closed evidence discipline; the energy
   mechanism (replace host DRAM streaming at 17-20 GB/s with phone DRAM at 55 GB/s and 4.5 W) is the paper's
   physical backbone.
C2 **Batch manufacturing under helper contracts** (the core scheduler contribution): continuous join of same-model
   arrivals into a decoding server, bounded bypass of a queued model switch (fairness bound), residency hysteresis,
   late-helper adoption, early helper re-provision overlapped with the desktop load, speculative rows to fill a lone
   request's free rows. Known relatives (Orca/vLLM continuous batching, Sarathi) assume weights are always present
   and optimize throughput; ours forms batches jointly with helper readiness and residency under an energy objective.
C3 **Calendar-driven device power states.** The scheduler, not a governor, decides when the GPU idles at 210 MHz
   (arrival calendar + queue), floors it during disk-bound loads, caps SM during bandwidth-bound decode, and restores
   predictively. Measured: loads 38 → 15 W, idle 28 → 14 W, decode 32.6 → 27.3 W at unchanged token period; prefill
   is compute-bound and must be exempt. Relative: DynamoLLM (GPU DVFS + instance sizing under SLO) — ours is
   device-agnostic, calendar-driven and coupled with offload; the phone DSP DCVS (pinned at max corner, idle 52 % of
   each token) is the next lever.
C4 **Elastic membership with exact recovery.** Helper drop/join at runtime, quarantine/readmit with identity checks,
   mask-out recovery that keeps the live server (47 s penalty vs 60 s reload), recovery keeps its queue place.
   Relative: Llumnix/SpotServe migrate requests; ours re-plans layers across devices mid-request with byte-identical
   output.
C5 **Method.** Per-token decomposition from RPC timestamps, per-state energy from device-power events, variance as a
   first-class metric, repeated runs with controlled phone thermal state.

## Related-work check (2026-09-28 web search) — the closest neighbours
Our phone design IS attention-FFN disaggregation (AFD): attention + KV on the host, FFN on the helpers. AFD is a
crowded 2025-2026 topic, so the positioning must be explicit:
| work | what it does | how we differ |
|------|--------------|---------------|
| MegaScale-Infer, Step-3 (AFD for MoE) | attention and FFN/expert pools on separate datacenter GPUs; microbatch ping-pong overlaps them | edge/consumer devices over USB; DENSE FFN that is bandwidth-bound at batch 1-4 (their FFN is compute-bound at large batch); we measured that ping-pong loses to batching when rows are free (2·max(p,d) ≥ p+d, phone rows +32 % for 4×) |
| AFlex (arXiv 2608.01891) | AFD + per-operator DVFS with an ILP global scheduler + local DVFS controller | datacenter GPUs, throughput/SLO; ours: calendar-driven device power states across GPU + phones on one desktop, coupled with batch formation and helper residency; measured with phone battery energy |
| OpWeave (arXiv 2609.14237), AFD-Ledger, OpScale | operator-level disaggregation planning across heterogeneous GPU groups (cost model + planner) | static planning for cost; ours: online, availability-churning helpers (thermal, drop/join, charger), exact recovery, energy objective |
| AFD challenge papers (2602.09721, 2605.28302) | when AFD helps for MoE | a dense-model, batch-1 regime they do not cover |
| DynamoLLM, throttLL'eM, GreenLLM, VoltanaLLM, EcoInfer, BiScale | GPU DVFS for serving (pool / request / iteration / phase level) | they tune one GPU type; we sleep the GPU from the arrival calendar and floor it in disk-bound loads and bandwidth-bound decode, and exempt compute-bound prefill (measured 1.3-1.9× prefill cost when not exempt) |
| LOIP/LIME (2512.21835), Jupiter, EdgeShard, Galaxy, EdgeInfer-TP | edge collaborative inference: pipeline/tensor parallel across edge devices (Jetsons), latency-first | phones as FFN bandwidth donors to a desktop, energy-first, per-layer synchronous helper contract with qualification receipts |
| PowerInfer, HeteGen, NEO, Dovetail | CPU/GPU split inside one machine (hot/cold neurons, async transfer, CPU offload of attention, CPU/GPU speculative) | the helpers are separate SoCs on USB with their own DRAM bandwidth and power; NEO's asymmetric pipelining is our rejected stagger |
| Orca, vLLM, Sarathi | continuous batching / chunked prefill | assume weights resident on one device; we form batches jointly with helper readiness, residency and power state under an energy objective |

**Consequence for the claims.** "Disaggregating attention and FFN" and "DVFS for serving" are NOT novel on their
own; the paper's novelty must be (a) the regime: dense models at batch 1-4 on a VRAM-bound consumer host with
phones as bandwidth donors over USB, where every stage is bandwidth-bound and rows are nearly free — the opposite of
datacenter AFD's compute-bound FFN; (b) the scheduler that exploits it: batch manufacturing coupled with helper
residency/readiness and device power states, under helper availability churn; (c) hardware energy evidence
including the phones' own batteries, with run-to-run variance explained by discrete timing forks. The measured
negative result "AFD ping-pong loses to batching at edge batch sizes" is a useful, citable contrast.

## Positioning, honestly
- Not new: continuous batching, DVFS for inference, speculative decoding, CPU/GPU offloading (FlexGen, HeteGen,
  PowerInfer), edge collaborative inference (EdgeShard, Galaxy). Each is one ingredient.
- New: the integration under a measured thesis (coincidence engineering), the phone-FFN-helper topology with
  fail-closed identity, decisions that couple batch formation with helper readiness/residency/power, and hardware
  energy evidence with repeated runs on a real desktop + phones.
- Rejected directions we can cite as negative results: dual-engine NPU+GPU on the phone (1.01×/0.91×), operator-type
  split (OpenCL attention too slow), stagger vs batch (batching dominates when rows are free), staging the next
  model during decode (RAM-capped, 38 % of load time), streamed tiles with deferred norm (≤ 2 ms/layer ceiling).

## Claims we can make today vs what must still be produced
| claim | status |
|-------|--------|
| −59 % host energy with two phones, identical outputs | measured (run 1, cj2 98.3, pe1 95.7 kJ); fleet −53 % with measured OP15 energy |
| decode is row-invariant on both stages | measured |
| device power controller saves 15-23 W in loads/idle, 16 % in decode | measured (dp1, dp2) |
| elastic drop/join, mask-out recovery | measured (G1g, g11) |
| batch manufacturing removes the timing forks and the 30 % spread | NOT yet: join bypass never fired; hysteresis/late-helper/early-reprovision in progress |
| speculative rows halve J/token for lone requests | NOT yet |
| phone energy | OP15 MEASURED 2026-09-28 (pe1): 9.7 kJ/run = 5.3 W mean (USB 2.47 W at the 500 mA cap + 5.2 kJ from the battery coulomb counter) vs 4.7 kJ assumed → fleet saving −53 % not −56 %; Pixel still assumed (2.2 kJ) — measure the same way |

## Evaluation plan
- Traces: eval_v2 (sparse, 14 req), a 2× and 4× denser variant, longtail_v1 (24 req); models Qwen3-14B + Gemma-4-12B (+ a third pair if time).
- Arms: all-desktop; work-conserving two-phone (today's best); + batch manufacturing; + device power; + speculative rows; each ×5 with the OP15 started cool and the thermal-status limit on.
- Metrics: host kJ and measured fleet kJ, J/token, p50/p99 latency and SLO attainment, identical outputs, variance across repeats.
- Microbenchmarks: row-cost curves (phone/desktop), per-token split, per-state power, decode-cap sweep (800/1200/1600 MHz), phone DCVS on/off.
- Robustness: worker kill mid-call (retire vs mask-out), thermal exclusion with/without the status policy.
- Sensitivity: RAM 30 vs 64 GB (switch cost), 1 vs 2 phones, arrival rate.

## Threats a reviewer will raise
Single rig; two models; sparse trace; assumed phone energy; f16 dequantized weights (chosen for exactness); the
helper's gain is only 12-16 % latency (we sell energy, not speed); the run-to-run variance (we make it a result).
