# Fast-path utilization plan: make the scheduler spend more of every request on the one path that pays

Current reporting status (2026-09-27 16:31 UTC): saved eval_v2 energy/recovery audit PASS; best two-phone 94.414 vs legacy 228.535 kJ (58.687% host saving), latest undisturbed tp2 112.186 kJ (50.911%), g11 121.439 kJ (46.862%). g11 same-server mask/reconnect PASS, 64.731 s; quoted 47 s is discarded attempt time. Strict identity FAIL: best 13/14, tp2/g11 12/14. Thermal intervention NOT VERIFIED and remains off; tp2 has a 514.662 s long exclusion plus a later episode, raw severity not retained. All 463 scheduler Python sources match checkout/stage/deploy. Fresh full runner FAIL on one 10.475 ms vs 10 ms timing assertion; isolated rerun PASS. No new hardware run. G2/G3 remain pending. [Audit and tables](scheduler/campaigns/burstgpt/reports/20260927-results-audit/README.md). Overlap code/log audit PASS; full-width local FFN overlap is about 0.012-0.014 ms versus 10.9-13.5 ms Qwen RPCs in tp2 shutdown summaries. Four overlap research proposals remain unverified; [research note](scheduler/campaigns/burstgpt/reports/20260927-overlap-research/README.md).

Written 2026-09-20. Historical milestone status: `M0 PASS (single pair); M1 PASS with amended check; M2 ACCEPTED for N<=4; cohort co-dispatch fix PASS; M4a Task 1 m4a8b ended FAIL 2026-09-22 15:58:47 UTC, transition failed without fallback, no RESULT/energy; shared-forward PASS (442 multi-row calls), observed batch>=2 median 38.134 J/token including assumed phone (19 windows, 3 eligible); coverage FAIL 4/15, 19/24 outputs complete and exact; prior 88121 tail-proof abort did not recur; 15 deployed source/test hashes verified, preflight PASS with split-row SHADOW/coalesced QUALIFIED; m4a8 coalesced-only preflight FAIL because calibration profiles reference removed executor; current 24-request exact-output best is 117.533 vs 132.253 kJ (-11.130%), earlier 14-16% variants have output mismatches; Task 2 matched pair PASS 203.348 -> 164.290 kJ (-19.208%, 19/19 exact), maximum output 472 so long-tail benefit unverified; historical v16/v16c verified -25.827% host/-25.118% with assumed phone, older baseline/deployment; current >25% target FAIL; M3 OP11 v73 NPU repair/qualification PASS 2026-09-22 18:11 UTC: DSP mapping slots corrected 16->64, normal DMA, HMX off; layers18-21 48 calls/112 rows, max relative L2 1.671e-4; median NPU round trips 59.622/63.301/77.097 ms at rows1/2/4 vs OpenCL 68.652/4751.604/4749.101; CPU 18-20 ms so end-to-end latency still worse; Wi-Fi NOT MEASURED (no OP11 IPv4/route); no energy/full-model token/two-phone proof; full integration deferred; cleanup PASS, isolated OP11 bin only; Pixel10Pro Vulkan qualification PASS 2026-09-22 19:25 UTC (layer18,12calls/28rows,max relative L2 0.000311882); latency improvement over OP11 FAIL, median103.110/142.184/201.093ms at rows1/2/4; USB5000M; Qwen TPU/Wi-Fi/energy/full tokens/integration unverified; Pixel Tensor TPU add round-trip PASS 2026-09-22 19:57 UTC, 180 calls exact, 128-element add only, no Qwen compiler SDK; split/single-send/split A-B-A median50.003/4.666/50.070ms, best p905.203ms, worker1.689ms; response coalescing proven only in isolated probe, production FFN unchanged; cleanup PASS; Pixel FFN TPU path reviewed 2026-09-22 20:38 UTC, protocol reusable but custom Qwen compilation blocked by missing Tensor SDK, no new hardware arm; Pixel Vulkan FFN tuning PASS 2026-09-22 20:51 UTC, quantum4352 plus coalesced TCP reply, one-row median35.008ms vs prior103.110ms, max relative L2 0.000311957; one-layer real server PASS 20:58 UTC, all4x64 tokens exact,124 phone calls, full-layer host request saving4.301% vs mean of two controls, decode+2.062-2.791%, phone energy unmeasured; six-layer numerical qualification PASS72calls/168rows, max relative L2 0.000326951; first six-layer server startup rejected occupied port before launch; corrected LISTEN guard/fresh port retry PASS 21:07 UTC, all4x64 outputs exact,744 phone calls; half/full columns on layers18-23 save10.659%/11.170% host request energy vs mean of two controls, decode+9.3%/+18.1%; one short measurement, phone energy/full trace/multi-phone integration unverified; cleanup PASS21:09 UTC; Pixel phone-stage profile PASS21:55 UTC,384 phone calls identical across four arms, CPU max relative L2 0.000325483; execution+sync97.644%/98.425% of half/full worker time, GPU matrix-vector timestamp share79.878%/89.797%; isolated diagnostic only, profiler adds observed16-30% overhead, no new energy claim, cleanup PASS; Pixel throughput arithmetic PASS22:20 UTC, full-width matrix timestamps18.150 GFLOP/s, stage-only worker16.123 GFLOP/s; optimization candidates unverified, no new physical run; Pixel complete shape/precision audit PASS22:33 UTC, all96 captured calls match operation counts and GEMV m4352/n1/k5120 or m5120/n1/k4352, FP16 weights with FP32 arithmetic, SwiGLU preserves4352 width; no new physical run or standalone bandwidth measurement; Pixel per-op profile PASS2026-09-23 00:29 UTC,384 exact phone calls, full-width unfused gate/up/down2.491/2.644/1.808ms timestamp intervals,17.888/16.855/24.650GFLOP/s; normal worker32.628ms vs profiled48.527ms, no pure-kernel or energy claim; initial slice diagnostic FAIL discarded, barrier/fence retry PASS with high overhead; cleanup PASS; Pixel GEMV tuning build PASS 2026-09-23 01:09 UTC, 132 shader hashes verified, six workgroup/row candidates plus quantum8704 staged; physical launch FAIL 2026-09-23 01:24 UTC after900s shared-lock timeout, zero phone calls; graph-reorder four-line candidate build PASS; correctness/speedup NOT VERIFIED, no tuning job queued at01:24; Pixel GEMV sweep numerical PASS 2026-09-23 02:39 UTC,1320 calls,max relative L2 0.000325483,full w128r8=32.322ms vs33.368ms controls (-3.135%) exploratory, larger workgroups/block size regress; cleanup PASS, repeat/server verification pending; graph reorder numerical PASS 2026-09-23 02:42 UTC,600calls,performance FAIL33.321ms vs33.055ms (+0.804%),candidate rejected,cleanup PASS; subgroup sweep numerical PASS 2026-09-23 02:51 UTC,1320calls,32/64-lane variants regress,128/128/8 repeats32.063ms exact,cleanup PASS; reversed-order confirmation PASS 2026-09-23 02:55 UTC,1680 exact calls,full128/128/8 32.399ms vs33.098ms matched controls (-2.111%),half18.397ms vs18.535ms (-0.743%,inconsistent across repeats),cleanup PASS; isolated tuned server PASS 2026-09-23 03:08 UTC,all4x64 tokens exact,744 calls,half/full phone worker19.848/28.642ms,host request energy -11.575%/-12.429% vs mean desktop controls with decode +8.326%/+15.340%,one short sample per split,no phone-energy/full-trace or paired stock-phone server speedup proof; cleanup PASS03:09,no queued jobs,selected config archived,production default unchanged; Pixel dense shader build PASS 2026-09-23 03:30 UTC,9 body variants/27 validated SPIR-V modules,FP32 accumulation,15-arm1800-call physical launch FAIL 2026-09-23 03:46 UTC,900s shared-lock timeout exit75,zero phone calls/no phone deployment,no tuning job queued; performance/correctness/server tokens NOT VERIFIED,staged experiment resumable; Pixel dense retest lock wait FAIL exit75 at2026-09-23 04:34 UTC,zero calls,user requested keep queued,persistent queue PID2282252 superseded per user Pixel-only clarification and cancelled05:05 UTC; Pixel local-loopback sweep+confirmation complete,3960 CPU checks PASS,maxL2=0.000325483,all2160 confirmation outputs exact; best repeated vec4_u1 full20.455 vs21.244ms (-3.714%),half11.696 vs12.277ms (-4.728%),small gain provisional due control drift; cleanup PASS,no host model work/RPC/energy/server-token measurement,private candidate only,old queued desktop run cancelled; Pixel partition-concurrency source review PASS2026-09-23 16:40 UTC,existing four4352-wide parts share input and sum outputs,explicit1/2/4-way overlap speedup NOT VERIFIED,no new hardware run; Pixel up-matvec/SwiGLU private build PASS2026-09-23 16:59 UTC,27 validated SPIR-V modules,840-call phone-local on/off benchmark running,numerical/speedup NOT VERIFIED; Pixel fusion first840 outputs exact PASS,maxL2=0.000325483,coverage FAIL264/384,diagnostic confirms one alias-rejected GLU/request; input-retention worker build PASS,2160-call confirmation running; Pixel fusion confirmation complete2026-09-23 17:11 UTC: numerical PASS2160/2160 exact,maxL2=0.000325483,coverage PASS744/744 per fused arm after input retention; speed FAIL full21.001 vs20.352ms (+3.191%),half11.819 vs11.652ms (+1.431%),candidate rejected,previous selection retained,cleanup PASS; no energy/USB/server-token claim; Pixel CPU/GPU comparison PASS2026-09-23 17:54 UTC,600 numerical checks,CPU4 half/full23.975/45.849ms vsGPU11.487/19.929ms,GPU2.09x/2.30x; CPU/GPU low-bit differences,maxCPU L2=0.0000988841,CPU tuning/fusion/energy/server tokens unverified,cleanup PASS; Pixel bandwidth/concurrency review PASS2026-09-23 17:59 UTC,effective GPU26.834/CPU11.664GBps,physical DRAM ceiling/bottleneck/combined execution NOT VERIFIED; ideal70/30 split13.891ms is a no-contention model only,no new hardware run; Pixel streaming sweep PASS2026-09-23 18:22 UTC,24 arms,512MiB,CPU26.737/GPU38.694GBps wall (GPU44.163 device),concurrent44.218GBps,copy24.348 read-plus-write,all checksums/cleanup PASS; no DRAM counters/theoretical peak;1200-call persistent-CPU-thread FFN sweep running,confirmation pending; Pixel bandwidth task COMPLETE 2026-09-23 18:42 UTC,55 bandwidth arms/8594 engine passes PASS,CPU dynamic30.421-30.565GBps,GPU confirmed37.25-38.04 wall/43.19-43.71 device,joint46.47-46.58 (+22.91% vs matched GPU),no physical DRAM counters/theoretical peak; FFN CPU6 persistent44.291->18.291ms (-58.70%,2.421x),2520 numerical checks PASS,all2040 CPU outputs exact; initial Android affinity coverage FAIL,private guard fix PASS; cleanup PASS,no worker/queue,production unchanged,energy/concurrent FFN/server tokens unverified; Pixel CPU/server overlap source/arithmetic PASS 2026-09-23 18:54 UTC,50% projected host9.552 vs CPU9.162+old overhead5.041 =>4.651ms wait/layer,100% projected23.066ms; physical retest NOT RUN/shared rig lock busy,no job queued,no new token/energy proof; Pixel CPU/GPU FFN review PASS 2026-09-23 19:10 UTC, full CPU18.291/GPU20.610ms, conditional equal-split model17.763ms before extra overhead, concurrent FFN speedup/energy unverified,no new hardware run; Pixel CPU/GPU private smoke PASS 2026-09-23 19:26 UTC,180 numerical/36 concurrent calls,initial speed FAIL22.128 vs18.280ms full,19-arm sweep pending,no energy/server proof; Pixel CPU/GPU19-arm sweep PASS 2026-09-23 19:35 UTC,2280 numerical/1440 dual calls,speed FAIL best19.349 vs18.214ms full,affinity confirmation running; Pixel CPU/GPU FFN COMPLETE 2026-09-23 19:45 UTC,5100 numerical/2916 concurrent calls PASS,pinnedCPU4+GPU50/50 full15.334 vsCPU18.210ms (-15.79%),half8.086 vs9.169 (-11.81%),p99 FAIL25.294 vs18.716ms,cleanup PASS,private candidate only,no energy/server-token/multi-row proof; Pixel unequal-ratio build PASS 2026-09-23 20:12 UTC,64-channel granularity,27 SPIR-V validated,180-call smoke staging,numerical/speed NOT VERIFIED; Pixel fine-ratio smoke PASS 2026-09-23 20:14 UTC,180 numerical/overlap checks,new50 byte-exact36/36,controls unstable,no speed claim,1800-call coarse sweep staging; Pixel ratio coarse numerical PASS 2026-09-23 20:22 UTC,1800 calls,strict overlap FAIL1/1560,CPU25/35.29 percent full15.135/14.583ms exploratory,4080-call refinement staging; Pixel ratio refinement PASS 2026-09-23 20:29 UTC,4080 numerical/3600 overlap checks,CPU38.235 percent full13.145/half7.084ms exploratory,5280-call reversed fine confirmation staging; Pixel unequal-ratio tuning COMPLETE 2026-09-23 20:42 UTC,20 ratios11340 numerical checks PASS,CPU39.706/GPU60.294 full13.058 vs50/50 15.379ms (-15.09%) vsCPU18.330 (-28.76%),half7.077 vs8.166 (-13.34%),selected p99 PASS14.029/8.104ms vsCPU19.034/9.582,final4080 overlap PASS,exploration overlap FAIL1/9420,cleanup PASS,private candidate only,no energy/server-token proof; Pixel bandwidth/compute assessment PASS 2026-09-23 20:51 UTC,full40.955GBps vs measured joint46.515 (88.05%),conditional11.497ms reference vs13.058,physical DRAM/sustained compute peaks NOT ESTABLISHED,no new hardware run; Pixel four-direction experiments active 2026-09-23 21:11 UTC,joint5040 calls PASS,existingCPU4 retained,coalescing144-call smoke PASS,new batch27SPIRV PASS,packed/batch physical and full confirmation pending; batch smoke1 FAIL client element-count header,graceful finite drain PASS,fix and fresh retry prepared; fresh batch672 numerical/dispatch PASS,maxL2=0.000417331;packedCPU allocationFAIL fixed in privatev5,3120-call confirmation running,14496-call batch sweep prepared; packed/coalescing3120 completed,nativeCPU5.87ms accuracyFAIL1.89percent,coalescing full1percent gain/half4.7percent loss,not promoted;batch14496 running,residual correction build/test underway; batch14496 numerical/dispatch PASS,CPU-only8row38.44ms leads,mixed63.23ms,shared-memory tile and steady-batch confirmation pending;residual4320 completed/audit pending; residual4320 accuracyPASS0.0005194,timing unstable,CPU8 performanceFAIL;shared-tile672 numericalPASS/speedFAIL;2160 pinned confirmation complete/audit pending,13536 final batch prepared; corrected packedCPU6 pinned confirmation2160 PASS,full10.792 vs matched13.694ms (-21.19percent),half5.485 vs7.442 (-26.30percent),maxL2=0.0005194,private candidate,no energy/tokens;final batch13536 running; batch13536 numericalPASS,maxL2=0.000562194,packedCPU B1/2 gains20.24/24.20percent but B8 regresses; mixedCPU76.47/GPU23.53 B4/8 gains39.07/38.98percent; unboundCPU stabilityFAIL and strict overlapFAIL1/8640; bounded6816 pinnedCPU endpoint confirmation prepared; Pixel four-direction tuning COMPLETE 2026-09-23 22:35 UTC,50976 completed-suite calls with native packed accuracyFAIL920 preserved; selected corrected packedCPU6 B1 full10.792/half5.485ms (-21.20/-26.30percent matched),selected F16 pinnedCPU6 B2/4/8 full19.829/21.697/35.423ms (-31.08/-53.39/-56.73percent matched); final6816 numerical/affinityPASS,4800 overlapPASS,repeat960 exact,absolute drift remains;coalescing speedFAIL,shared-tile single-arm exploratory;cleanupPASS,no queued jobs,private only,no energy/server/token/peak proof; Pixel Tensor SDK installed PASS 2026-09-24 01:16 UTC,compiler FFN6/6 ops,physical real layer18 tail512 PASS88calls,maxL2=0.000571125,invoke3.351/worker4.021/USB7.523ms,full-width test pending,no energy/full-token/integration proof; Pixel full-width TPU PASS 2026-09-24 01:24 UTC,B1/B4 88calls each,6/6 compiled/all NPU,maxL2=0.000547222,worker35.007/36.686ms USB40.832/47.593ms,initial speed goal FAIL vs historical CPU,paired CPU untested,BF16 pending; Pixel Tensor SDK investigation COMPLETE 2026-09-24 01:30 UTC,installation/dispatch/numerical PASS528FFNcalls1056rows+24ADD,FP16 full B1/B4 worker34.671/35.998ms USB40.646/47.213ms,maxL2=0.000571125,repeatsexact;BF16 accuracyPASS0.004686623/speedFAIL34.952ms;initial CPU speed goalFAIL,no matchedCPU/energy/tokens/serverintegration;cleanupPASS,no pending job; Pixel TPU runtime audit PASS 2026-09-24 01:43 UTC,48calls120rows,maxL2=0.000547222,vendor hardware B1/B4 27.337/27.419ms,invocation remainder5.680/5.625ms,one residentDISPATCH_OP;precise DRAM/MAC bottleneck unverified,sharding/performance tuning untested,cleanupPASS; Pixel lossless-layout smoke PASS936calls,maxL2=0.0005194,33SPIRV validated,CPU/GPU small gains exploratory,2026-09-24 03:24 UTC,FMLAL/device-format follow-ups active; Pixel FMLAL1152/device-format3456 numericalPASS,2026-09-24 03:34 UTC,bestmixed10.830ms vsCPU10.262ms speedFAIL,residual-pass fusion test active; Pixel fused residual smoke PASS864calls,CPU288/288 exact to oldpacked,full9.772vs11.658ms,final4560call confirmation active 2026-09-24 03:39 UTC; Pixel device-layout4560 and batch816 numerical PASS 2026-09-24 03:48 UTC,mixed94.118percent CPU full7.844/8.098ms (-22.71/-22.13percent matched),fusedCPU8.095/10.217ms unstable,focused2160 endpoint active; Pixel NEON/GPU/layout tuning COMPLETE 2026-09-24 03:56 UTC,86arms13944calls numerical/dispatchPASS,maxL2=0.000536187,5304overlaps,1632CPUoutputs exact;finalCPU8.122/4.135ms vsold10.179/5.189 (-20.21/-20.31percent),p99PASS8.873vs11.949; mixedCPU94.118percent full7.928ms (-22.11percent vsold,-2.39percent vsfusedCPU),p99vsCPUFAIL9.712/max20.993;CPUfusion privatecandidate,mixedexperimental;cleanup/buildPASS,noqueuedjob;peak/energy/server/tokens unverified; Pixel custom packedGPU tuning COMPLETE 2026-09-24 04:37 UTC,50arms6528calls numerical/dispatchPASS,maxL2=0.000519396;final nativeGPU20.200/11.157ms,newblock16WG256r8 19.854/11.088ms (-1.71/-0.62percent provisional),fusedCPU8.122/4.120ms;GPU-over-CPU speedFAIL2.44xCPU,stablegain unproven,no promotion;v5 eightSPIRV/build/lint/hashPASS;cleanupPASS,noqueuedjob;newbatch/energy/server/token/peak unverified; Pixel dynamicCPU/pairedNEON COMPLETE 2026-09-24 05:12 UTC,41arms7008calls/9648rows numericalPASS,5376optimizedoutputs exact,maxL2=0.000562194;B1/2/4/8 correctnessPASS;native8.094/4.119ms,pairedSDOT+dynamic64 6.325/3.239ms (-21.855/-21.365percent),fullp998.778->7.422PASS;native schedulingaloneFAIL8.186ms;halfmaxworse5.274vs4.511ms;build/lint/hash/cleanupPASS,noqueuedjob;privateB1candidate,server/USB/energy/fulltokens/physicalpeak/newmixed unverified; Pixel rooted AOA COMPLETE 2026-09-24 17:18 UTC,functional/numerical/build/cleanupPASS,9936FFNcalls14256rows exact plus6820echo;continuous B1 full10.565->6.152ms(-41.77percent),half7.858->3.918(-50.14percent),B2/B4 full-36.19/-28.15percent;idle robustnessFAIL,5msgap full29.824->28.179(-5.51percent),echo+46.60percent regression;timedwake remedyFAIL;normalADB/root/boot restored,no workers/forwards/wakelocks;private stock-accessory path,FunctionFS/server/energy/tokens/integration unverified`. Latest checkpoint 2026-09-24 20:48 UTC: v7 completion PASS9/9,863.663s,host61.905kJ; activation FAIL Pixel0calls/Qwen0of4; OP15Gemma15936calls. Strict outputs FAIL7of9 vs older desktop+dispatcher (different code/build). 225 partial-transition refusals; primary-phone contract still includes Pixel mask. Independent route-pruning reproducer yields0->4 policies with split preference, not a hardware fix. Two completed activation failures; retries stopped, cleanup PASS, no queued job. Integration remains isolated; rebased review diff applies cleanly,245 targeted checks passed after fixture restoration. Fresh controls and incremental Pixel benefit unverified. Owner: the implementing agent named in the handoff prompt at
the end. Build on the current tree; do not start parallel implementations. Nothing is committed without
the user's explicit approval (`AGENTS.md`).

## Why this plan

Everything measured in this project says one path saves energy: **dense FFN column slices with
pre-positioned weights on a phone, applied only during decode, with the host's copy released**
(`research_dev/scheduler/campaigns/burstgpt/reports/20260917-decode-relocation-kv-headroom/README.md`).
On Qwen3-14B the 100 % decode split takes host decode power from 121 W to 68 W, matched pairs saved
16 to 34 % of request host energy, and the reduced BurstGPT trace under the automated scheduler saved
17 % (run7, 127.0 vs 152.6 kJ). Everything else on a phone lost: whole operators as extra backends
(S5), attention (S4, split-KV), MoE experts (`reports/20260919-moe-edge-energy/`).

The same measurements show the fast path is under-used and its surroundings waste more than it saves:

| Waste or under-use | Evidence | Plan item |
| --- | --- | --- |
| Prefill re-streams host-layer weights to the GPU every 128-token ubatch; GPU 45 W, CPU 29 W for 201 s | `reports/20260918-split-kv-attention/physical/*/POWER_SAMPLES.json`; CUDA `offload_op` threshold 32 tokens | M0 |
| Decode uses 8 of 24 threads, about 20 GB/s of a 60+ GB/s bus; the phone split was compared against this | same arms; `reports/20260919-moe-edge-energy` shows the MoE decode is bandwidth-bound at 24 GB/s | M0 |
| 27 restores cost 152 s of a 1,577 s trace because release drops the page cache | `reports/20260917-decode-relocation-kv-headroom/physical/trace-run7/SERVER_FFN_PROOF_LINES.txt`; `src/llama-mmap.cpp` `release_fragments` (`POSIX_FADV_DONTNEED`) | M0 |
| Phone FFN calls carry at most 4 tokens, so one server slot at a time benefits | `tools/server/server.cpp:191` (`S41_SERVER_FFN_MAX_TOKENS`), gates run `--parallel 1` | M2 |
| One phone per server; OP12 idle while OP15 serves | run7 environment | M3 |
| Fraction chosen by a fixed adaptive policy; 50 % is latency-optimal, 100 % energy- and memory-optimal | Qwen sweep in the 20260917 report | M4 |
| Freed host RAM (9.63 GB at 100 %) backs KV, but nothing turns it into more served context or more slots | KV-headroom probe and split-KV Steps 1-4 | M4 |

Target for the plan: on the reduced BurstGPT trace (`campaigns/burstgpt/data/traces/burstgpt_dev4_5min_v1.json`,
the run7 configuration) beat run7's saving against a **retuned** host baseline, report fleet energy with the
phone's assumed power separated from the measured server energy, and keep request latency within the trace's
SLO. Every milestone has a check; a milestone that fails its check is reported as failed, not narrowed.

## Ground rules for the code

- Extend the existing modules: `research_dev/scheduler/adapters/` (rig, HTTP backend, contracts),
  `research_dev/scheduler/_internal/` (contracts, selection, accounting), `campaigns/burstgpt/` (runner,
  launch, checkers), native `tools/server/`, `src/`, `examples/layersplit/`. No second scheduler, no copy of
  a module with a suffix.
- Every new knob is a typed field in an existing contract with a default that reproduces today's behavior,
  is covered by the KV-plan or runtime digest where it changes numerics or memory, and is rejected fail-closed
  when unsupported.
- Tests next to the module they cover (`research_dev/scheduler/tests/`); `python3 -m unittest` for the
  touched modules and `pyflakes` clean before each check. Native changes get a `test-backend-ops` or probe
  test on the tiny model where the behavior is observable there.
- Each milestone writes a short report under `campaigns/burstgpt/reports/<date>-fast-path-M<k>/README.md`
  with the exact command lines, records under `physical/`, and a talks.md entry (newest first, timestamped,
  tables over prose). Single runs are labeled single runs.
- Energy: RAPL package + NVML board for the host, measured; phone power assumed at a stated value and reported
  in a separate column, never summed silently into "fleet".

## Milestones and checks

### M0. Retune the host baseline (no new mechanism)

Do:
1. Prefill: measure ubatch 128 / 256 / 512 / 1024 on the Qwen pair-v1 request (9,737 prompt tokens, 64
   output) with the current 16 GPU layers; keep batch 2048; confirm compute-buffer memory under the 18 GiB
   scope stays within budget.
2. Decode: sweep threads 8 / 12 / 16 / 24 (P-cores first, `--cpu-mask` if needed) on the same request.
3. Restore: stop calling `posix_fadvise(POSIX_FADV_DONTNEED)` in `release_fragments` unless a new contract
   flag asks for it; keep `MADV_DONTNEED`. Measure restore time with warm and with pressured page cache. Then
   try the second variant: no explicit `populate_fragments`, let the next prefill fault pages in on demand.
4. Rebuild the transport identity (`materialize_transport_qualification`) because the server changes.

Check M0 (all must hold before M1 starts):
- Prefill time and host energy at the chosen ubatch versus 128, in a table; the choice is the energy minimum
  that fits the memory scope.
- Decode ms/token and J/token per thread count; the choice is the J/token minimum.
- Restore time under no pressure below 2 s for the 9.63 GB share; correctness (argmax equality) unchanged.
- **Re-pair the phone split against the tuned host**: 0 % vs 100 % decode split on the pair-v1 request, host
  energy and ms/token. If the tuned host removes the phone win, stop the plan here and report that.

M0 passed on 2026-09-20 after the user restored OP15 on ADB 5037. The tuned
single pair used ubatch 1024, eight unpinned threads and keep-cache + populate:
22,967.922 -> 20,219.003 measured request host J (11.97% saved, single pair),
779.722 -> 789.379 ms/token. All 64 tokens matched at ubatch 1024.
Assumed phone power is separate: 0.875 W idle and 4.5 W during assisted decode;
measured host plus that assumption saves 11.046%. Warm restore was 0.226 s.
All M0 checks passed; proceed to M1. [Report](scheduler/campaigns/burstgpt/reports/20260920-fast-path-M0/README.md).

### M1. Split and transfer inventory (instrumentation only)

Do: env-gated (`GGML_SCHED_TRACE=path`) JSONL emitter in `ggml_backend_sched_compute_splits`: per split, the
backend, node count, op names, bytes copied in, wall time; per graph, the total. Zero overhead when unset. A
small reader in `research_dev/scheduler/_internal/` that turns one prefill ubatch and one decode step into
the operator inventory the planner consumes (device, bytes, time per layer and family).

Check M1: the inventory reproduces the measured prefill time at ubatch 128 and at the M0 choice within 10 %
from the sum of split times plus copies, and identifies the weight-streaming term as the difference.

**Amendment 2026-09-20 (after the first M1 check failed at 24 % on the 1024 held-out run):** every M0/M1
arm ran with `memory.current` pinned at the 18 GiB `memory.max` (19.30 of 19.33 GB, 4,000 to 6,700 `max`
events already at server-ready, more during the request). The F16 model's host RSS (16.5 GB) plus KV and
buffers does not fit the scope, so the kernel reclaims and re-faults model pages throughout prefill; the
traced run took 110 s and the held-out run 143 s for the same configuration. A split-plus-copy model cannot
predict reclaim. The timing calibration and its held-out check must run **without memory pressure** (no
scope, or a scope at least 4 GiB above the arm's `memory.peak`); the 18 GiB scope is reserved for the
memory-benefit arms (release, KV headroom), where reclaim is the phenomenon under test. Re-run the M1
held-out check that way before M2. Record `memory.events.max` deltas per arm in every report from now on.

The first M1 check failed on 2026-09-20 under the 18 GiB cap. Frozen component sums predicted 293.118 s at ubatch
128 and 108.788 s at 1024; held-out prefills took 321.685 s (8.880%
error, pass) and 143.271 s (24.069% error, fail). Both token comparisons
are exact and repeated weight streaming is identified, but the 1024 prediction
exceeds the 10% limit. Work stopped before M2 until the user authorized the
no-pressure recheck above. Both calibration and held-out arms were repeated
with the same native runtime hashes and fresh uncapped scopes. Original results
remain preserved; no prediction is fitted to held-out measurements.
The user paused work at 2026-09-20 17:30:49 UTC during the first no-pressure
traced-128 request. It finished and cleaned up normally while the launch driver
remained stopped, with no further arm launched. The user resumed work at
2026-09-20 17:50:17 UTC; runtime identity was verified before continuing the driver.
Pause/resume records
are in the report's `physical/no-pressure/PAUSED.json` and `RESUMED.json`.
M1 passed the amended check at 2026-09-20 18:15:46 UTC. Frozen predictions
291.319 s (128) and 111.980 s (1024) versus held-out 296.523 s and
113.785 s give 1.755% and 1.586% error. Both comparisons match all 64 tokens.
All four fresh uncapped scopes recorded `memory.events.max` 0 at ready and finish.
Repeated weight-copy time explains 84.702% of the held-out prefill difference.
The 73 local and 73 rig tests pass; pyflakes is clean. No refit or diagnostic
repeat was needed. Proceed to M2. [Report](scheduler/campaigns/burstgpt/reports/20260920-fast-path-M1/README.md).

### M2. Multi-slot decode through one phone session

Do: raise the phone FFN call to carry the decode tokens of all active slots in one call
(`S41_SERVER_FFN_MAX_TOKENS` bounded by the worker's `max_tokens` and the transport contract in
`adapters/llama_server_contracts.py`); server `--parallel N` with `N` in {2, 4, 8}; the client packs the
ubatch's rows (decode has one row per slot) into one request and unpacks the result. Keep the existing
`flash_attn_type=DISABLED` decode context and the HTP session/worker binaries; the static B=16 batch on
HTP was bit-correct in the dualengine work, and the n_parallel=2 hang was localized to the HTP backend's
flush path, so the first check is a hang watch, not a performance number.

Check M2:
- N=1 output identical to today; N in {2,4,8}: every slot's tokens identical to the host-only run of the
  same prompts; no worker hang over 512 decode steps (watchdog in the gate, never kill an in-flight worker).
- Phone time per token per slot falls with N (report the curve); host decode power with N slots at 100 %
  split versus N slots host-only.

M2 failed the N=2 exact-token check at 2026-09-20 19:15:50 UTC and stopped for
the user's decision. The 256-token prompt matched all 576 outputs; the 257-token
prompt first differed at output 4 (host token 198, phone 271) and matched 51/576
positions overall. Both requests completed, with 572 full-cohort steps on every
owned layer and no watchdog event. Single-run host decode power was
123.402 W host-only versus 65.801 W with the phone; assumed
phone power remains separate at 0.875 W idle / 4.5 W active. Both fresh uncapped
scopes recorded memory.events.max 0 at ready and finish. All 90 rig tests pass
and pyflakes is clean, but physical output equality failed. N=1/4/8 and the full
utilization curve were not run after this failure; M3 has not started.
[Report](scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/README.md).

M2 revised Step 3 finished at 2026-09-21 00:01 UTC. Two concurrent N=2, 576-token
runs with unchanged pack/unpack pass the first-64-step numeric check (max relative
L2 3.195e-4 and 3.566e-4; no row above 1e-2) and match all historical host tokens.
Own acknowledgements were 3/3 and 3/4, with request slots 1/0 and 0/1. Across 1,152
corresponding calls there are zero identical input-row hashes, even after request
alignment, so phone determinism is **inconclusive**. The user's conditional change
to logits NMSE <= 5e-4 requires observed nondeterminism and has not been applied.
The initial M2 acceptance failure is not cleared; N=1/4/8 and the curve remain
pending. Both scopes recorded memory.events.max 0/0 and cleaned up normally.
[Revised Step 3 report](scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/step3-numeric/README.md).

**M2 acceptance amendment 2026-09-20 (after the original 576-token mismatch at output 4 failed to reproduce in
8 matrix cases and 2 full-length reruns, all exact, with the per-row phone-vs-host relative L2 at 3.2e-4 to
3.6e-4 over 2,304 rows and nothing above 1e-2):** the phone FFN is f16 and its result is not bit-identical to
the host's, so exact tokens over long generations are expected to flip occasionally at near-ties. The check
becomes: per slot, tokens identical to the host reference, OR at the first mismatch the host logits' top-1 and
top-2 are within a near-tie margin (|l1 - l2| <= 0.05 after softmax-free logits, recorded) and the phone-arm
logits at that step are within the project tolerance (NMSE <= 5e-4) of the host's. Per the user's clarification,
acceptance is decided at the first mismatch in each slot; a slot failing that test is a fault and stops M2.
Later differing positions retain their host logits, margin and NMSE and are labeled as comparisons after
context divergence. The original failure is recorded as one unexplained event in 11 pairs; the determinism question (3.2) is closed only with two runs of identical slot assignment and ack
indices via the ordered submission path.


Ordered Step 3.2 closed on 2026-09-21 at 00:55:46 UTC: both N=2 runs used
slots 1/0, own acknowledgements 4/3, prompts 256/257 and 64 outputs. All 90
first-five-step call inputs and all 90 returned payloads match (agreement
fraction 1.000); no nondeterminism was observed in this sample. Both numeric
checks pass with maximum relative L2 2.896e-4.

**M2 final recheck: FAIL on utilization, 2026-09-21.** N=1 matches all 64 saved
pair-v1 outputs. All N=1/2/4/8 slots pass the amended correctness rule over 576
outputs: 14 EXACT and one N=4 near-tie at output 80 (request 2 / slot 1, margin
3.891e-4, NMSE 8.226e-7). Later differences are recorded after context divergence.
Every long run completes, with 573/572/571/567 full-cohort steps per layer and no
hang. Phone compute ms/token/slot is 164.651 / 82.999 / 51.488 / 64.171; N=4 to
N=8 rises 24.633%, and RPC also rises (67.398 to 77.996). The decreasing-time
check fails, so M3 has not started and the plan stops for the user's decision.
Each point is one single pair. All 11 scopes record memory.events.max 0/0,
all 100 rig tests and pyflakes pass before each arm, and cleanup passes. [Acceptance recheck](scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/acceptance/README.md).

**M2 disposition 2026-09-21:** correctness PASS under the amended rule (14 exact slots, one recorded near-tie
at N=4, ordered determinism 90/90 calls, no hangs over 567-573 steps). The utilization curve is
164.7 / 83.0 / 51.5 / 64.2 ms of phone compute per token per slot at N = 1 / 2 / 4 / 8, so the literal check
("falls with N") fails at N=8. The records explain it: every call is one USB transfer with all N rows, and
the phone's compute per call is 11.3 ms at 4 rows but 27.6 ms at 8 rows (2.45x), i.e. the HTP switches kernel
path above 4 rows (the HMX gate is M > 4; the HMX tile pads small M) and loses per-row efficiency. Host power
in the phone arm is 57 to 67 W at every N against 124 to 146 W host-only; request-group host energy is
-54 / -53 / -48 / -31 % at N = 1 / 2 / 4 / 8; per-token latency is better than host-only at N <= 2, +3 % at
N = 4 and +68 % at N = 8.
Decision: **M2 accepted with an operating envelope of N <= 4 slots per phone call.** The planner (M4) and the
trace (M5) must cap cohort size at 4 until the N = 8 knee is removed. The knee is a phone-side kernel-dispatch
question (keep the HVX GEMV path up to 8 rows, or make the HMX path efficient at M = 8) and is recorded as an
optional M2b, outside this plan's "no phone kernel work" rule; it needs a measured HTP profile of the FFN at
M = 4 and M = 8 before any change.

**Cohort queue prerequisite passed, 2026-09-21 03:42 UTC.** The unchanged cold-phone
cohort regression now acquires both members concurrently, with causal readiness
true and no active conflict. Both share eight cohort-owned lease tokens; a fifth
member waits until the last of four releases the reservation. Cohort capacity is
capped at four. All 133 requested runtime/controller/cohort tests and 41 phase/resource
tests pass; pyflakes is clean. No phone kernel changes or physical run were needed.
[Queue fix report](scheduler/campaigns/burstgpt/reports/20260921-cohort-co-dispatch/README.md).
Proceed to M4 while M3 remains blocked on the OP12 move.

### M3. Two phones, disjoint layer ownership, one server

**Update 2026-09-21:** the second phone is a **OnePlus 11 5G** (serial `832358d4`, Snapdragon 8 Gen 2,
Hexagon **v73**, USB **2.0 only**, **rooted**), already attached to the desktop's ASM107x hub at 480 Mbit/s;
adb is still off (MTP interface only). Ready on the controller: `libggml-htp-v73.so` and arm64
worker/resident-workers/router built from the current tree in `build-ffn-overlap-android`; on the desktop:
Qwen shard `HTP0.ffn.gguf` for layers 18-23 (3.21 GB, sha dd705a75...) under `s42-op11-qwen-shards-20260921-v1`.
Required code: per-phone `minimum_usb_speed_mbps` (480 for OP11), a second phone block/device/resources in the
rig manifest and topology, identity with both phones' receipts, runtime mask spanning both ranges. The single
phone assumption appears ~300 times in 55 files (`phone_device_id`), so plan it as its own milestone with the
checklist `campaigns/burstgpt/M3_SECOND_PHONE_CHECKLIST.md`. v73 has no HMX: FFN runs on HVX, slower per call,
still removes host CPU work. The OP12 paragraph below is kept for reference.

Do (original): OP15 and OP12 each own a disjoint layer range of the same model's FFN (the shard files already exist
per session). **Precondition (user action):** OP12 is currently attached to the controller host FCHLLX01
(usb 6-2), not to the desktop; the FFN fast path needs a direct USB link to the serving host, so OP12 must be
plugged into the desktop before M3. The desktop has one xHCI controller (Alder Lake PCH, 20 Gbps upstream) with
nine SuperSpeed root ports: OP15 is on port 2 at 5 Gbps and an empty 4-port 5 Gbps hub sits on port 9. Use a
free root port or that hub; both phones then share the one controller, which is fine for 20 KB latency-bound
calls, and M3's phone-time check measures any interference. OP12 needs fresh transport receipts on its desktop
port before its identity can be materialized. Check with `adb devices -l` and `lsusb -t` on the desktop. The rig
coordinator books both, the runtime control applies one decode-boundary mask covering both ranges, and the
dormant host share releases both. OP12 is Hexagon v75: keep its known limits (no fused FA, RMS/ROPE crash
cases are attention-side and do not apply to FFN).

Check M3: decode split over both phones versus OP15 alone at the same total fraction: host decode power,
ms/token, released bytes; both phones' proof rows verified; outputs identical to host-only.

### M4a (inserted 2026-09-21, do before the LP). Make the phone path usable for every eligible decode

Measured on 2026-09-21 (report `campaigns/burstgpt/reports/20260921-fast-path-trace-v2a/`): with the phone
engaged a Qwen decode window costs 41 J/token against 75 J/token on the host (medians, n=244/147), yet only 5
of 15 Qwen requests on the 24-request trace and ~61 % of tokens on the realistic trace ever used it. Three
mechanisms, all confirmed in logs, and the order to fix them:

1. **Server-wide dormant release (highest value).** `server_context` applies one FFN split policy per decode
   batch and `apply_dormant_host_share` releases only when the batched slots all carry the phone policy; any
   co-tenant prompt or non-eligible decode restores (110 ms) and the next batch releases again (45 ms). On the
   busy server 36 of 45 releases served one token. Do: per-sequence release semantics: keep the host FFN
   pages released while at least one eligible decode is running and serve the other slots' FFN from the phone
   as well (same weights, bit-exact as shown by 24/24 identical outputs), or if a slot cannot use the phone
   (prefill), run that prompt's FFN on the phone for short prompts or defer it (M4 hysteresis below). Check:
   releases per assisted token < 0.05 on the 24-request trace (was ~0.12), zero one-token releases.
2. **Probe measurements poisoned by the thrash.** The adaptive controller probes the phone policy at most
   `maximum_probe_attempts_per_context` = 2 times per context; a probe that lands during thrash measures
   116-202 J/token, the pair is judged not improved and the baseline becomes the cached winner for the whole
   context. Do: after (1), re-run; then raise the probe budget to 4 and lower `minimum_remaining_tokens` from 24
   to 8 via the campaign field `adaptive_minimum_remaining_tokens`. Check: >= 12 of 15 Qwen requests carry
   `CONTROL_ISSUED` with the phone policy; phone windows median stays <= 50 J/token.
3. **Layout demand for routes that never run.** `_learning_phone_demand` keeps a phone session for any model
   whose phone routes are in evidence state LEARNING even when they are never selected (Gemma held 8-9 layers
   all trace long in every arm; removing its shard index does not stop it because shards derive from the plan
   contract). Do: decay learning demand for a model whose phone routes have not been selected within the last
   N decisions, and give a model with `ffn_host_share_release == 1` priority for sessions in dormant campaigns.
   Check: with Gemma release off, all three OP15 sessions hold Qwen by the third Qwen request; with Gemma
   release on (variant D), the split follows the realized phone tokens per model.

Do not use campaign `fixed_phone_residency` for energy arms: it is an evaluation preload that arrived work
ignores (variant B). Do not use `phone_resident_limit_bytes` to evict a model: it is forwarded as
`maximum_helper_resident_weight_bytes` and nothing consumes it (variant A).

### M4. Planner: energy-with-idle-power LP and residency knapsack

Note added 2026-09-20 from the M2 code review: `server_slot::can_batch_with` already refuses to batch slots
whose FFN split policy differs, so a new prompt is never mixed into an assisted decode ubatch (correctness
holds). The cost is that with `--parallel N` and mixed traffic the server alternates prompt batches and
assisted decode batches, and `apply_dormant_host_share` populates the released share on every alternation
and releases it again once all slots decode. M4 must give the release a hysteresis rule (release only when
no prompt is admitted or expected within a window, or when the KV budget needs the room) and account the
populate cost per alternation; otherwise multi-slot traffic turns the 0.2 s warm restore into a per-batch tax.

Do: replace the fixed adaptive fraction policy (`adapters/http_backend.py` `_AdaptivePayloadController` and
`_internal/adaptive_decode*`) as the *source of the proposal* with a plan produced by an LP over the M1
inventory: variables are per-layer FFN fraction per phone, host thread count, ubatch; objective is host
idle power times token time plus dynamic energy plus assumed phone energy; constraints are the SLO, phone
weight residency (`ShareBinding`), host memory budget (the `DormantShareCoordinator` ledger, with released
bytes credited to KV cells or slots through the existing `DecodeReleaseAccountant`). Solve with `scipy`
if present, else a bounded grid over the discrete fractions the atlas validated. The existing accountant,
coordinator and admission hooks stay the enforcement path; the planner only proposes.

Check M4:
- Prediction within 10 % on every already-measured arm: Qwen 0/25/50/75/100 sweep, M0 ubatch and thread
  sweeps, split-KV arms, and the run7 per-request energies.
- On a synthetic mixed batch of prompt shapes the planner picks 50 % when the SLO binds and 100 % when it
  does not, and refuses shapes that exceed the memory budget (fail-closed, with the reason code).

**Pre-existing blocker that M5 will hit (found 2026-09-20 by running the full suite):**
`test_adaptive_runtime.test_cold_phone_cohort_shares_transition_identity` hangs. Two adaptive-decode requests
that should form one decode cohort are bound as causal predecessor and successor in `runtime_queue.py`
(`_bind_causal_predecessors`: their decisions conflict on the same compute lanes, so the later one waits until
the earlier one is no longer ACTIVE). The second member therefore cannot be dispatched while the first runs,
which is the opposite of a cohort. The 2026-09-11 epoch-helper-admission report already recorded this as
"outside its path, needs a separate phase/cohort audit". With `--parallel N` in the trace (M5) this is exactly
the path that must work: cohort members must be co-dispatchable, i.e. the queue must not treat members of
one decode cohort (shared transition identity) as lane conflicts. Fix belongs with M2's owner before M5; the
test is the acceptance check. Until then the automated scheduler can only serve one assisted request at a
time regardless of the server's slot count.

### M5. Trace acceptance

**Measured 2026-09-21 (single runs, same deploy, same trace, host = RAPL package + NVML board):**
24-request trace: baseline 132.3 kJ / 1,381 s; treatment 117.5 kJ (-11.1 %); layout variant A 111.2 kJ
(-15.9 %); variant C 112.7 kJ (-14.8 %); outputs bit-identical 24/24 in the treatment pair. Realistic trace
(`burstgpt_realistic30_v2`, 20 requests): baseline 340.0 kJ / 3,898 s vs treatment 281.2 kJ / 3,555 s
(-17.3 %, 61 % Qwen token coverage). Target 25 % not met; cause and fix order are in M4a. Variant D (Gemma
release on) running. Failures worth remembering: deploy built without `S41_SERVER_FFN_SPLIT` (identity now
refuses such a server); realistic manifest lacked `model_inventory` (builder fixed, validation added); stale
Gemma cache flags (parse-time validator added).

**Prepared 2026-09-21 (before the runs):** deploy `/mnt/storage/s42-trace-v2-20260921-prep` (current tree, CUDA
build, git snapshot, transport identity materialized against OP15). Input sets under `/home/zhihao/`:
`s42-trace-v2a-20260921-inputs` = run7's 24-request trace with the keep-cache restore policy and paths
repointed (resolve and preflight PASS); `s42-trace-v2r-20260921-inputs` = the realistic BurstGPT window
trace built by `campaigns/burstgpt/build_realistic_trace.py` (real inter-arrivals, ChatGPT -> Qwen,
GPT-4 -> Gemma, shortest ChatGPT share -> Llama overlay). Both are derived by
`campaigns/burstgpt/prepare_trace_inputs_v2.py` from run7's inputs so the diff is exactly the intended change.
Run order: v2a host-only baseline x3 and phone treatment x3 (parallel unchanged until the cohort co-dispatch
fix), then v2r the same way, then the parallel-4 variants once the queue fix has landed.

Do: reduced BurstGPT trace under the automated scheduler with M0 to M4 enabled, three runs, plus the
retuned host-only baseline, three runs. Same checker as run7 (`check_dormant_release_trace.py`, extended
for two phones and multi-slot credits).

Alongside these baseline runs, repeat M0's tuned pair-v1 comparison as three
alternating 0% / 100% decode-split pairs. Keep the 11.97% M0 energy saving labeled
as a single-pair result until those repeats are available; do not run them during
the M1 recheck.

Check M5: mean matched saving of server energy versus the retuned baseline, with the phone assumed-power
column; request latency distribution within the trace SLO; zero refused credits, zero holds over 2 s. Report
whether the saving beats run7's 17 % against its untuned baseline and by how much against the tuned one.

## Handoff prompt for the next agent (rewritten 2026-09-22)

> You are picking up the phone-assisted decode-energy work in `/home/myid/zs89458/Documents/llama.cpp-release`,
> branch `wip/unified-scheduler-cleanup-20260813`. The tree is deliberately dirty. Never commit, never push,
> never write commit messages or PR text (`AGENTS.md`). Ask the user before anything destructive; the desktop
> rig and both phones are shared hardware.
>
> READ FIRST, in this order, before writing code:
> 1. `research_dev/FAST_PATH_UTILIZATION_PLAN.md` in full (M0-M5, M4a, this prompt).
> 2. `research_dev/scheduler/campaigns/burstgpt/reports/20260921-fast-path-trace-v2a/README.md` in full: every
>    measured arm of 2026-09-21 and the root-cause analysis.
> 3. `research_dev/talks.md`, the newest 20 entries (newest first, timestamped; keep adding to it).
> 4. `research_dev/scheduler/campaigns/burstgpt/M3_SECOND_PHONE_CHECKLIST.md` sections 0, 7, 8, only if you touch
>    the second phone.
>
> MEASURED FACTS. Do not re-derive these; build on them.
> - Rig: desktop `zhihao@172.20.74.85` (RTX 4060 Ti, i9-12900K), OnePlus 15 `3C15AU002CL00000` on usb 2-2 and
>   OnePlus 11 `832358d4` on USB 2.0, both on adb port 5037 only. Deploy `/mnt/storage/s42-trace-v2-20260921-prep`
>   with `source/` (scheduler + C++) and `cuda-build/`.
> - A Qwen decode window costs 40 J/token with its FFN on the phone and 72-79 J/token on the host. Beside a
>   host-policy co-tenant it costs 128-138 J/token, because `server_slot::can_batch_with` refuses to batch slots
>   with different `ffn_split_policy`: the assisted request runs in alternating batches, its latency doubles and
>   its window is charged the co-tenant's energy, so the adaptive controller rejects the phone policy. That is why
>   only 4-6 of 15 Qwen requests ever attach.
> - 24-request trace, host energy against the 132.3 kJ / 1,381 s desktop baseline: as-designed 117.5 kJ (-11.1 %),
>   contiguous Qwen sessions 111.2 (-15.9 %), Gemma release on 111.5 (-15.7 %), server release rule 113.3 (-14.3 %),
>   cheap probe 125.3 (-5.3 %, worse), per-server coherence 113.9 (-13.8 %). Configuration and controller tuning
>   are exhausted at 14-16 %.
> - Realistic BurstGPT trace: baseline 340.0 kJ / 3,898 s against treatment 281.2 kJ / 3,555 s, -17.3 %, with 61 %
>   of Qwen tokens phone-assisted. Long decodes pay more because attach cost amortizes and the mix shifts to decode.
> - The real BurstGPT conversation log has median output 190 tokens, p90 671, p99 1,068. Requests above 512 output
>   tokens are 18.2 % of requests and carry 49.1 % of all output tokens, and our builder caps output at 512.
>
> UNCOMMITTED WORK ALREADY IN THE TREE. Keep it, extend it, or remove it with a stated reason.
> - `tools/server/server-context.cpp`: `apply_dormant_host_share` releases the host FFN only when every processing
>   slot decodes under the same phone policy; under a policy mix it stays local and counts
>   `ffn_dormant_release_skipped_mixed`.
> - `ggml/src/ggml-hexagon/htp/htp-ops.h`: `HTP_OP_MAX_BUFS` 16 -> 64 (host-side batch limit; the DSP reads the
>   buffer list by count).
> - `research_dev/scheduler/_internal/adaptive_decode_ops/coherence.py` and `tests/test_adaptive_coherence.py`:
>   a session exploiting the baseline adopts a co-tenant's phone policy. It fired 8 times in a run and never
>   produced a multi-token phone call, so it is necessary but not sufficient.
> - Campaign fields `adaptive_maximum_probe_attempts_per_context` and `adaptive_decode_overrides` (any
>   `AdaptiveDecodeConfig` field), and learning-demand decay in `_unified/phone_residency_ops/demand.py`.
>
> TASK 1, highest value: make co-tenants share one forward pass.
> Decide the FFN split policy per (model artifact, desktop parent placement, layout generation) instead of per
> request, so every decode slot of that model on that server runs the same policy from its first token; attach the
> helper at admission for a server that is already assisted, instead of after the first recorded window; confirm
> the server batches policy-equal slots and sends all their rows in one phone call; and compare only windows of
> equal batch composition, or normalize, so a probe is never rejected against a baseline measured at another
> concurrency. Files: `_internal/adaptive_decode*.py`, `_unified/adaptive_decode_control.py`,
> `_unified/helper_preparation*.py`, `_unified/automated_requests_ops/`, and `tools/server/server-context.cpp`
> around `update_slots` and `apply_ffn_split_ubatch_context`. Keep the dormant release rule above.
> PASS requires all of: `S41SERVERFFNUSB` lines with `tokens=` greater than 1 on the busy Qwen server; phone
> windows at `active_batch` 2 or more with median at most 55 J/token; at least 12 of 15 Qwen requests
> phone-assisted; host saving at least 22 % against the 132.3 kJ baseline; all 24 outputs token-identical to the
> baseline arm. Stop and report if two runs fail for the same reason.
>
> TASK 2, independent of task 1: a long-decode trace, which is the design's honest best case.
> Our 512-token output cap truncates half the real decode work. Rebuild the window with the real distribution:
>
>     cd /mnt/storage/s42-trace-v2-20260921-prep/source
>     LD_LIBRARY_PATH=/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64 \
>     python3 -m research_dev.scheduler.campaigns.burstgpt.build_realistic_trace \
>       --burstgpt-csv /mnt/storage/burstgpt-source/burstgpt_3.csv \
>       --output-dir /mnt/storage/burstgpt-source/longdecode_v1 --trace-name burstgpt_longdecode_v1 \
>       --codec /mnt/storage/s42-trace-v2-20260921-prep/cuda-build/bin/llama-token-codec \
>       --library-dir /mnt/storage/s42-trace-v2-20260921-prep/cuda-build/bin \
>       --qwen-tokenizer-model /home/zhihao/models/Qwen3-14B-Q4_K_M.gguf \
>       --gemma-tokenizer-model /home/zhihao/models/gemma-4-12B-it-Q4_0-op15-exact.gguf \
>       --llama-tokenizer-model /home/zhihao/models/Llama-3.2-1B-Instruct-Q4_0.gguf \
>       --prompt-cap 2048 --output-cap 1100 --duration-s 1800 --min-requests 12 --max-requests 20 \
>       --execution-artifact qwen3-14b-q4km-dequant-f16=/home/zhihao/models/Qwen3-14B-Q4KM-dequant-f16.gguf=text_decoder_f16_proxy \
>       --execution-artifact gemma-4-12b-q40-dequant-f16=/home/zhihao/models/gemma-4-12B-Q40-dequant-f16.gguf=text_decoder_f16_proxy \
>       --execution-artifact llama-3.2-1b-instruct-q4_0=/home/zhihao/models/Llama-3.2-1B-Instruct-Q4_0.gguf
>
> 1,100 is the log's p99, and 2,048 plus 1,100 stays inside the 4,096-token per-sequence context of both desktop
> parents, so nothing needs re-qualifying. Keep `--min-output` at its default for the headline trace: that is the
> untruncated real window. A second trace with `--min-output 256` is allowed only as an explicitly labelled
> best case, never as the headline. Derive inputs with
> `campaigns/burstgpt/prepare_trace_inputs_v2.py`, then run treatment and desktop-baseline arms and report the
> matched saving. Expect the run to take longer than the 3,555 s realistic trace; keep `--min-requests` low.
>
> TASK 3, second phone, only after tasks 1 and 2 have run: OnePlus 11 integration. Its v73 NPU crashes in the
> user-DMA `dmpoll` inside `hvx_mv_2d` with Bad VA 0, and `dma_queue_create` returning NULL is silently tolerated
> in `htp/main.c` around line 428. Fix candidates and the working OpenCL fallback are in the checklist section 8.
> Binaries, the OpenMP runtime and the Qwen shard for layers 18-23 are already on the phone. The rig manifest and
> about 300 single-phone assumptions across 55 files are the real engineering cost; the brief is checklist
> section 7.
>
> RIG PROCEDURE.
> - One run at a time under `flock -w 900 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock`.
> - Sync scheduler changes with `rsync -rlptc --exclude __pycache__ --exclude .venv --exclude campaigns/burstgpt/reports
>   research_dev/scheduler/ zhihao@172.20.74.85:/mnt/storage/s42-trace-v2-20260921-prep/source/research_dev/scheduler/`.
> - C++ changes: rsync the files, then
>   `/mnt/storage/s21_deps/cmake-4.2.3-linux-x86_64/bin/cmake --build /mnt/storage/s42-trace-v2-20260921-prep/cuda-build
>   --target llama-server` with `LD_LIBRARY_PATH=/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64`.
>   The build must keep `-DS41_SERVER_FFN_SPLIT=ON`; verify with
>   `strings cuda-build/bin/libllama-server-impl.so | grep -c S41SERVERFFN`, which must be 12 or more, or the server
>   silently ignores every `S41_SERVER_FFN_*` variable. After any rebuild, `rm -rf phone-identities`, move the old
>   `TRANSPORT_QUALIFICATION_IDENTITY.json` aside, run `bash /mnt/storage/s42-trace-v2-20260921-prep/MATERIALIZE_TRANSPORT.sh`,
>   and copy the new identity into every inputs dir you use.
> - New inputs per attempt: copy `campaign.json`, `rig.json`, `models.json`, `evidence.json` and the identity from
>   `/home/zhihao/s42-trace-v2a-m4a3-20260921-inputs` (both models release, probe attempts 4), rewrite the paths and
>   `campaign_id`. Launch from `/mnt/storage/s42-trace-v2-20260921-prep/source` with `LANG=C.UTF-8` and
>   `S42_UNIFIED_REPO_ROOT=/mnt/storage/s42-trace-v2-20260921-prep/source`:
>   `python3 research_dev/scheduler/campaigns/burstgpt/launch.py <campaign.json> <new dir> --preflight-only`, then the
>   same without the flag. Output directories must not already exist.
> - Compare with
>   `python3 /home/zhihao/s42-trace-v2a-20260921-inputs/compare_trace_energy.py --run baseline=/home/zhihao/s42-trace-v2a-baseline-20260921-inputs/run-baseline-1/run/RESULT.json --run yours=<RESULT.json>`.
> - Tests: `PYTHONPATH=.:research_dev/scheduler/tests python3 -m unittest <module>` from the repo root, plus pyflakes
>   on every changed file.
>
> HAZARDS THAT HAVE ALREADY COST RUNS.
> - Never `pkill -f` a pattern that your own ssh or adb command line contains; it kills your shell. Use bracketed
>   patterns in their own `adb shell` call, or kill by PID.
> - Wait on `RESULT.json` and `FAILURE.json` files, not on process patterns, and never put `set -e` before the run
>   line in a chain script: a failed launcher then skips its completion marker and stalls every follower.
> - Regenerate every derived inputs dir after changing an input patcher. Cache-policy flags on a model without
>   `ffn_host_share_release == 1` are refused at parse time and used to abort a whole run through route quarantine.
> - `fixed_phone_residency` is an evaluation-only preload that arrived work ignores; `phone_resident_limit_bytes` is
>   forwarded but never consumed. Neither can pin or evict a model from the phone. Use the shard index for that.
> - Never force-kill an in-flight phone worker. Phone kernel changes only as a user-authorized temporary
>   `fastboot boot`. Never start an adb server on a port other than 5037.
>
> REPORTING. Report every milestone as PASS or FAIL with the measured numbers, never as a claim, and say plainly
> what you did not verify. Append a timestamped entry to `research_dev/talks.md` newest-first for each result,
> extend the report README with the new arms, and keep the plan status line current.
>

## Out of scope

Attention, projections, MoE experts or any second operator family on the phone. Intra-request CPU/GPU overlap
in the ggml scheduler (Step 5 of the split-KV plan) unless M1 shows it worth more than 5 %. Any phone kernel
work.

## Rig and rules (unchanged)

Desktop `zhihao@172.20.74.85`, cmake `/mnt/storage/s21_deps/cmake-4.2.3-linux-x86_64/bin/cmake`, CUDA libs
`/mnt/storage/s21_deps/cuda-13.2.1/{lib,lib64}` on `LD_LIBRARY_PATH` for build and link. New deploy dir per
milestone under `/mnt/storage/` (the NVMe root recently reached 99 % full and now has about 24 GB free: keep models, traces and logs on `/mnt/storage`); the copy must include `cmake/build-info.cmake` and `common/build-info.cpp.in`
(an `--exclude 'build*'` rsync drops them). Rig lock
`/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock` (flock, nonblocking). OP15
serial `3C15AU002CL00000` is on the desktop (usb 2-2), using adb port **5037** as in the run7 rig config.
The user stopped the stray server that had held it on 5038. Use 5037 everywhere; never start an adb server
on another port. Do not kill another campaign's adb server. OP12
`5ae7a43d` is on FCHLLX01 until moved. Never force-kill an in-flight phone worker.
Phone kernel changes only as user-authorized temporary `fastboot boot`. Pick a free port per server, verify
the answering process, clean up in `finally`, never `pkill -f` with a pattern your own command line matches.
Re-materialize the transport identity after every server rebuild. No commit, no push, no AI-written commit
messages.
