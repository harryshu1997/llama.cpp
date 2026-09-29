# Llama CPU + phone HTP split calibration, 2026-09-10

## Result

The real Q4_0 FFN shard executes correctly through the existing OP15 Hexagon
worker. Partial CPU/NPU overlap works, but no tested partial split establishes
a complete-request speed-and-energy win over the corresponding CPU control.
Do not qualify or force this route into the scheduler from these measurements.

The final `balanced-v12` comparison uses four decode threads, eight prefill
threads, strict P-core affinity, polling disabled, and no CUDA device visible
to either owned server process. Two identical 915-input/292-output requests
were executed per row. Times include prefill and decode, not preparation.

| FFN columns on phone | Decode tokens/s | Request time, s | Fleet J at 4.5 W | Saving vs CPU |
| --- | ---: | ---: | ---: | ---: |
| 0%, CPU control | 33.14 | 10.44 | 994.17 | 0% |
| 50%, CPU + NPU | 37.18 | 10.46 | 1,162.70 | -16.95% |
| 75%, CPU + NPU | 33.86 | 11.31 | 1,124.18 | -13.08% |
| 100%, NPU FFNs | 31.26 | 11.88 | 1,024.48 | -3.05% |

The 50% split is 12.20% faster during decode than its CPU control and faster
than the 100%-FFN reference, but its extra prefill cost cancels the request-time
gain. CPU package energy increases from 899.28 J to 1,034.89 J; this is not just
the assumed phone power. Mean GPU board energy is 85.76 J versus 90.20 J.

At eight decode/eight prefill threads (`pcores8-v10`), 100%-FFN offload costs
798.58 J versus its own 1,252.25 J CPU control, a 36.23% saving, but takes
15.31 s versus 10.14 s. This is an energy/latency tradeoff, not a simultaneous
win. The lower-energy four/eight-thread CPU control above also shows why that
36.23% must not be presented as the gain over the best tested CPU configuration.

"100%" means all selected FFNs execute on NPU; attention, output projection,
and other operations remain on server CPU. A whole-model NPU-only reference
was **not** run. The available whole-phone Llama server is an OpenCL build,
not a verified whole-model HTP endpoint. Do not relabel its prior measurements.

## Power sensitivity and preparation

Fleet here means physical RAPL CPU-package energy plus physical NVML GPU-board
energy plus assumed phone energy. It is not a wall-plug measurement of the
entire desktop. Phone idle power is 0.875 W in both arms; the assisted decode
interval uses the indicated active power. CPU-only uses phone idle power.

| Phone FFN fraction | Fleet J, 3 W | Fleet J, 4.5 W | Fleet J, 6 W |
| --- | ---: | ---: | ---: |
| 0% | 994.17 | 994.17 | 994.17 |
| 50% | 1,150.92 | 1,162.70 | 1,174.48 |
| 75% | 1,111.24 | 1,124.18 | 1,137.12 |
| 100% | 1,010.47 | 1,024.48 | 1,038.50 |

For `balanced-v12`, staged-phone preparation to host-observed usability took
9.299414 s. Native phone timestamps separately give weight reads 1.418916 s,
HTP initialization 0.065655 s, and weight upload 0.260812 s. The remaining
7.554031 s includes process launch, FunctionFS/NCM setup and host readiness
detection; it is not all weight transfer. `READY.json` and native
`RESIDENTPHASE` rows preserve each clock's timestamps separately.

That preparation consumed 118.066103 J of measured server energy, or
145.96/159.91/173.86 J including assumed phone preparation power at 3/4.5/6 W.
Desktop initialization was 0.607086 s / 25.43 J for CPU and 0.405147 s / 22.25 J
for the assisted endpoint (including phone idle). These are separately measured
components, not reconstructed end-to-end cold runs. No positive amortization
count exists for the negative steady-request savings in the final table.

Initial deployment of the 452,988,576-byte shard reported 4.908 s at 88.0 MB/s.
That one-time staging transfer preceded the measured preparation intervals;
its energy was not captured. No preparation or staging cost is silently
declared free, and no cold fleet-energy saving is claimed.

## What was diagnosed and retested

1. **Initial probe launch validation:** a CPU launch still requires a declared
   GPU device identity in its contract, even with zero GPU layers. Corrected
   the diagnostic contract; validation was not relaxed.
2. **Legacy static FFN policy mismatch:** the last Llama prefill layer selects
   one output row, so a tensor-row-based M=1 client policy sent work absent from
   the prefill graph. The server failed during warmup with `FFN split input
   layer 15 arrived while layer 15 request 1 was pending`. The probe now uses
   the existing acknowledged request control path, enabling assistance only
   after prefill. No native code or scheduler correctness check was weakened.
   The legacy static-policy path remains a documented issue, not a repaired
   production path.
3. **ADB/TCP transport:** at 50%, NPU compute averaged 0.532 ms but RPC averaged
   18.391 ms, with a 43.442 ms p90. Switching to the existing direct FunctionFS
   DMA-BUF path brought the smoke-test RPC mean to 1.149 ms and decode from
   3.15 to 30.22 tokens/s. These are 32-output smoke checks only. The TCP delay
   resembles buffering/delayed-ACK stalls, but that individual mechanism was
   not independently isolated.
4. **FunctionFS identity/cleanup race:** an already verified Android identity
   was queried again during USB changeover. The probe now retains that exact
   identity and waits for enumeration during cleanup. The original failed
   cleanup is preserved; `COARSE_V6_RECOVERY.json` records the artifact-specific
   graceful close used to restore ADB, without a USB reset.
5. **Worker block size and CPU polling:** tested quarter-, half-, and full-width
   worker partitions, plus polling disabled. Neither established a consistent
   complete-request partial-split win. The unpartitioned 100%-FFN check is in
   `fullwidth-v9`; no new worker or shard format was built.
6. **Unintended CUDA context:** early CPU-only launches still initialized CUDA,
   adding 128 MiB and keeping the idle GPU at high clocks. Final CPU and assisted
   arms explicitly set `CUDA_VISIBLE_DEVICES` empty for their subprocesses only.
   GDM remains running. Earlier small fleet-energy differences are confounded
   by GPU settling and are not used as evidence of savings.
7. **Host thread configuration:** P-core affinity, four versus eight decode
   threads, and independent prefill thread counts were tested symmetrically.
   Four/eight avoids the slower four-thread prefill while retaining the
   lower-energy CPU control. No other process affinity was changed.

The current structural cost is consistent with the existing view-safe FFN
storage path: `src/llama-model-loader.cpp` selects ordinary CPU buffers for
FFN slicing instead of the normal preferred packed buffer. Final measured
prefill is 1.623 s on CPU versus 2.604 s at 50%. This establishes the prefill
penalty; it does not attribute every extra joule to repacking without a further
kernel-level profile. Removing the view-safe requirement would be unsafe.

The final 50% call statistics also show real but incomplete overlap: 0.400 ms
NPU compute, 0.640 ms CPU branch, 0.728 ms RPC and 0.092 ms exposed wait per
call. Each request makes 4,640 calls, with 4 KiB input and output per call.
This is a small-message latency/CPU-kernel problem, not evidence that the link
is bandwidth-saturated. Existing waits use condition variables and blocking
libusb event handling; no new busy-wait defect was established.

The next substantive optimization needs profiled, slice-compatible packed CPU
FFN prefixes and/or lower per-call overhead, with explicit memory accounting
and numerical tests. It is not safe to achieve this by disabling validation or
changing a scheduler fraction constant. No such native redesign was made here.

## Identity, calls and scope

- Llama 3.2 1B Instruct Q4_0; same 915 prompt-token IDs, 292 requested output
  tokens, seed 42, temperature 0, uncached prompts and ignore-EOS behavior.
- CPU placement: zero GPU layers, `--device none`, no KV offload, context 4096,
  batch 1024, ubatch 256, parallel 1. Per-attempt `SPEC.json` records thread,
  polling and affinity overrides. Both sides of each pair use those settings.
- Stored FFN set: layers 0-15, all 8,192 FFN columns, Q4_0; FFN tensors only.
  One HTP0 session, generation 1, with the same parent artifact and worker
  weight hash `16ef10a8af46b56a`. No HTP1/HTP2 independence claim is made here.
- Physical worker evidence names `backend=Hexagon`, `HTP0-REPACK`, the shard
  parent/slice and 432.02 MiB of weight buffers. The path and phone SHA are
  verified before launch. Admission includes a 512 MiB workspace allowance and
  768 MiB reserve within live memory and the declared 10 GB pool. That allowance
  is not misrepresented as a measured workspace peak.
- One weight load per successful configuration; no weight reload between
  fractions or repeated requests. An assisted desktop endpoint is reused across
  fractions. CPU control has its own FFN-disabled endpoint.
- Final configuration: 8/8 complete, 27,840 HTP0 calls, 4,640 per assisted
  request. Controls applied at token index 2; 290/292 token positions assisted.
  Fraction-weighted coverage over all output tokens is 49.658%, 74.486% and
  99.315% for 50%, 75% and 100%. This is not 100% coverage at every fraction.
- Across successful configurations: 54 full-output requests and six 32-output
  smoke requests completed. Three failed startup/debug attempts are separately
  preserved, including one successful CPU smoke request before a native failure.
- Successful native terminal logs show no transport recoveries and status 0;
  controls ACK their exact policy hashes and request identities. Output passes
  semantic-sanity-v1 and exact output-count checks, not a perplexity or bitwise
  CPU/NPU equivalence qualification.
- This is native calibration, not UnifiedScheduler execution: no fabricated
  scheduler ticket, lease or replacement proof is emitted. Worker residency
  phase generation 1 is distinct from the raw FunctionFS connection counter
  initially printed as generation 0; the latter is not a residency epoch.
- Final USB state is ptp,adb; no owned inference workers remain. Kernel and GDM
  are unchanged. See `FINAL_DEVICE_STATE.json`. No trace, native rebuild,
  production scheduler edit, commit or push occurred.

## Evidence and reproduction

Remote artifact root:
`/mnt/storage/s42-llama-htp-split-20260911-v1-tMRgjn`

Local `physical/` preserves every attempt's specifications, commands, native
logs, responses, control acknowledgements, readiness/health records, raw power
samples and cleanup results. The large GGUF remains on desktop and phone;
its index and hashes are copied locally. All completed reference directories
remain unchanged. `SUMMARY.json` is derived by strict checks in `summarize.py`:

```sh
python3 -B research_dev/scheduler/campaigns/burstgpt/reports/20260910-llama-htp-split/summarize.py --output /tmp/llama-htp-audit-new.json
```

Use a new output filename. The probe also requires a new artifact directory.
It checks for existing inference workers before loading. Resolved physical
commands are in `SPEC.json`, `WORKER_LAUNCH.json` and each arm's `LAUNCH.json`;
`REUSED_DESKTOP.json` identifies arms that share the first assisted process.

Validation: all nine completed configurations pass the native call/ACK/token/
identity audit; repeating the analysis produces byte-identical `SUMMARY.json`.
AST, ASCII and whitespace checks pass. The final comparison has 45/45 valid
phone health samples and 461 host power samples with no sampler errors. No
broad scheduler suite was run because production code is unchanged.

| Evidence | SHA-256 |
| --- | --- |
| Model | `4b90b1d7ae7324676194755a6dfce11cb6e457982c4c01a1db2857be1ed064ad` |
| Shard | `698facd8ef7e54f8f3057407d6f6a14741734112408e75c4f43a545d5b4ee7df` |
| Shard index | `66729a297f9d0565891a92e67ae4041d49cc0f7d395fb37bb09197d49fb2dd09` |
| Desktop server | `92950efd39902b798e7b4e2810e6bb0ce620994213e7a9b256cab8742535aef6` |
| Phone worker | `43adcb755f8ff10ba30073ae55a31c7af95ee4f14071cfd2a5ba961b8317da19` |
| Final balanced result | `31a3a6a907424534a743d7b5dd2bcbed9415982471bc238b39f8d8d787236521` |
| Eight-thread result | `bd994349cf48682a7f2ef0640f978615096cebd1dccd3e0f122277a5691eceab` |
| Summary | `f3178f2483ee0c7607841d3a85f539b8ec126ee8dc17b4903db424a64b996c1d` |

All successful configurations match server, library, model and shard-index
hashes. Current local HEAD is `5f89a2d9d33be547a1bdef5fd0f504a279c50800` with
pre-existing dirty changes; this is not asserted to be a clean upstream or
whole-tree source manifest. The inspected model loader, Llama graph and FFN
client files match the deployed native source tree byte-for-byte. The same
already-built binaries were used throughout.

Files added for this task: report-local `probe.py`, `summarize.py`, this README,
`SUMMARY.json`, `COARSE_V6_RECOVERY.json`, `FINAL_DEVICE_STATE.json`, the evidence
hash manifest, and `physical/` evidence. `research_dev/talks.md` gets one new
log entry. No production or unrelated dirty-worktree files were changed.
