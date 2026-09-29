# WiFi-server harness: do phones on a local WiFi 7 LAN save energy for FCHLLX01?

Stage S1 of `../PLAN.md` (mechanism microbenchmark) plus the phone-side launchers for W1/W2. Everything
phone-facing is dry-run and unit tested only; nothing here has touched a phone or the desktop.

Host: FCHLLX01 (2x RTX A6000 48 GB, Threadripper PRO 5995WX 64C/128T, 125 GB RAM). One GPU only
(`CUDA_VISIBLE_DEVICES=0`). Model: the phones' exact artifact
`/home/myid/zs89458/Documents/models/Qwen3-14B-Q4KM-dequant-f16.gguf`
(sha256 `d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718`, 29.5 GB, f16 weights).

## The physics being tested (and what the code can actually do)

Measured elsewhere:
- Server CPU streams weights at about 69 GB/s (Qwen3-14B f16, CPU-only decode 2.35 tok/s, 48 threads;
  memory note `a6000-host-cpu-decode-speed`, 2026-09-20). Today's arm (a) fit gives more (see below).
- OnePlus 15 (Hexagon v81): 8.9 ms compute per full Qwen3-14B FFN layer (535 MB f16), about 60 GB/s;
  1.5 ms USB FunctionFS transport per call on the desktop. Pixel 10 Pro: 6.4 ms per full layer
  (`ref_pixel_worker_log.txt`: compute_p50_us 6392, packed q4_K weights), about 84 GB/s f16-equivalent.
- Only activations cross the link: 10,240 B per row per call (f16, n_embd 5120).
- The server idles above 100 W, so time saved is energy saved. The GPU alone draws 84 W idle while the
  server holds a CUDA context and 102-109 W while it decodes CPU-bound (measured today; 26 W without a context).

So a phone is about the CPU per byte: moving a CPU-resident FFN layer entirely to a phone saves nothing
(the `phone` arm is predicted SLOWER here, since this CPU out-streams the OP15). Only a COLUMN SPLIT helps:
per layer the host computes the leading (1-f) of the FFN columns while the owner phone computes the
trailing f, concurrently (`src/llama-graph.cpp:1619-1690`, `build_dense_ffn_split`: host view
`[0, n_ff - columns)` + phone partial added at `ffn_split_sum`).

**Structural limit (blocker for the 2.6x hypothesis).** PLAN.md's `69 + 2 x 55 GB/s = 180 GB/s` assumes both
phones stream every layer concurrently with the CPU. llama-server gives every layer exactly ONE owner helper:
helper layer masks must be disjoint and cover the union (`tools/server/server.cpp:258-272`), the eval
callback routes each layer's `ffn_norm`/`ffn_phone_partial` to the single owning client
(`server.cpp:977`), and one union column policy applies to all helpers (`apply_policy`, `server.cpp:883`;
`consistent_geometry` `server.cpp:1017`). So per layer the bandwidth is CPU + ONE phone (69 + 60 = 129 GB/s
with the reference CPU rate, 83.6 + 60 = 144 GB/s with today's fit: at most 1.7-1.9x on the FFN part), the two phones take turns, and attention + the output head stay
on the CPU. Two more limits follow: the share f is the same for every phone (the faster Pixel cannot take a
bigger share), and the share moves in steps of the union column quantum 4352 = 25 % (LCM of the OP15's
2176 and the Pixel's 4352). The harness therefore checks two models (`benchlib.predict_cpu_resident_ms`):

- `overlap` (what the server does): per layer `attn + max(CPU (1-f) FFN, owner phone f FFN + RTT)`
- `aggregate` (PLAN physics): `bytes / (BW_cpu + sum BW_phones) + calls x RTT`

Predictions (`python3 analyze.py --predict --bw-cpu 83.6 --gpu-ms 16.1`: CPU bandwidth and GPU part fitted on
today's arm (a), OP15 8.9 ms and Pixel 6.4 ms per full layer, 24 phone calls per step), step time vs `cpu`:

| RTT per call | split-25 | split-50 | split-75 | phone (100 %) |
|---|---|---|---|---|
| 1.5 ms (USB-class) | -17 % | -8 % | +14 % | +36 % |
| 3 ms | -14 % | +8 % | +30 % | +52 % |
| 5 ms (W1 go limit) | +7 % | +29 % | +51 % | +74 % |

The saving hinges on the WiFi round trip: at 25 % the phone side (OP15 2.2 ms compute + RTT) hides under
the CPU's 75 % (4.8 ms at the fitted 83.6 GB/s) only while RTT stays below about 2.6 ms; above that each
call exposes the difference, and every call is on the token's critical path (24 synchronous calls per
token), so the p99 tail matters, not the mean. `phone` (100 %) loses on this host at any RTT because this CPU
streams faster than the OP15.

## Files

| file | what it does |
|---|---|
| `wifi_config.example.json` | the one config: server launch, phones (adb argv, WLAN IP, worker binary, shards, env, ports), physics defaults, workload. Copy to `wifi_config.json`. |
| `phone_workers.sh` | starts/stops the FFN workers on the phones in TCP mode bound to `0.0.0.0` (`start/stop/status`), the echo servers for `../wifi_rtt.py` (`echo-start/echo-stop`), `ip`, `wifi-tune`, `hash`; `--dry-run` prints the exact adb commands; `--local` runs stand-in workers of this host's build on 127.0.0.1 |
| `server_bench.py` | `run`: one llama-server per arm on GPU 0, fixed workload, NVML/RAPL/CPU sampling, control protocol, per-arm `RESULT-<arm>.json`; `probe`: HELLO geometry + EXECUTE round trips (rows 1/2/4) straight to each worker (W2); `plan`: prints argv/env/control payloads |
| `analyze.py` | per-arm table + helper table + time-model check; `--predict` for expectations |
| `benchlib.py` | pure functions: config, arms/columns, env/argv, stderr line parsing, protocol v6 codec, GGUF byte table, energy windows, time models, phone commands |
| `test_harness.py` | 25 unit tests (`python3 -m unittest` here), including `phone_workers.sh` against a fake adb/su/toybox |
| `results/20260928-cpu-arm/` | arm (a) today (cpu, cpu-helpers, gpu) on the real artifact |
| `results/20260928-cpu-pair-contended/` | cpu-helpers vs cpu A/B under another user's background load |
| `results/20260928-loopback-standin/` | mechanics run of every helper arm against local CPU stand-in workers (f16 dev model) |

## Phone worker mapping

One `llama-ffn-split-worker` per server helper, TCP (`--port P --bind 0.0.0.0`; `--bind` must be an IPv4
literal, `ffn-split-worker.cpp:2551`; TCP and FunctionFS are mutually exclusive, `parse_config` 351-441).

| helper label | phone | backend | layers (mask) | port | weights on the phone |
|---|---|---|---|---|---|
| op15-htp0 | OnePlus 15 `3C15AU002CL00000` | HTP0 | 0-5 (63) | 7071 | `/data/local/tmp/s42-ffn-shards-20260904-v1/qwen/HTP0.ffn.gguf` |
| op15-htp1 | OnePlus 15 | HTP1 | 6-11 (4032) | 7072 | `.../qwen/HTP1.ffn.gguf` |
| op15-htp2 | OnePlus 15 | HTP2 | 12-17 (258048) | 7073 | `.../qwen/HTP2.ffn.gguf` |
| pixel | Pixel 10 Pro `5A040DLCH004ES` | CPU (packed q4_K) | 18-23 (16515072) | 7074 | `/data/local/tmp/s43-pixel-cpugpu-20260925-v1/QWEN_PACKED.ffn.gguf` |

Why three OP15 workers: on the desktop the OP15's `llama-ffn-split-resident-workers` already forks one TCP
worker per HTP session with `--backend HTPn --bind 127.0.0.1 --port <session port>` and
`GGML_HEXAGON_NDEV=3` inherited (`ffn-split-resident-workers.cpp:182-216`, env from
`research_dev/scheduler/adapters/native/direct_phone_ffn_session.sh:262-266,317-319`; manifest rows
`phone_session_ops/weights.py:205`), and a router bridges FunctionFS to them. One HTP session cannot hold
18 layers (about 3.2 GB per 6-layer session), so over WiFi the three workers are exposed directly (bound
to 0.0.0.0 instead of 127.0.0.1) and the server gets three helpers for the OP15. Env per OP15 worker:
`LD_LIBRARY_PATH=ADSP_LIBRARY_PATH=/data/local/tmp/s42-ffn-shards-20260904-v1-bin GGML_HEXAGON_NDEV=3
GGML_HEXAGON_VMEM=3328 GGML_HEXAGON_MBUF=4192 GGML_HEXAGON_NHVX=4 S41_DISABLE_GRAPH_CACHE=1
S42_RESIDENCY_SESSION_ID=HTPn S42_RESIDENCY_SESSION_GENERATION=1`, `--column-quantum 2176 --max-tokens 4
--f16-io --columns 17408`. The Pixel worker is the adopted DVFS-fixed launch
(`reports/20260925-two-phone-eval/physical/server-identity-r1/WORKER_COMMAND.json`: `S42_PIXEL_*`,
`S43_PIXEL_CPU_BATCH_PAIR=1 S43_PIXEL_CPU_POLL=100 S43_PIXEL_UCLAMP_MIN=1024`, root, kernel flock
`/data/local/tmp/.s42-pixel-ffn-kernels.lock`, `--column-quantum 4352`). Shard and worker sha256 values are in
the config (`phone_workers.sh hash` prints the phone side). Workers run with `--max-requests 0` (resident);
they accept one client at a time and a new one after a disconnect, so they survive server restarts between
arms. Stop them only while no server is connected (SIGTERM).

## How the helpers are configured on the server

Build: `tools/server/CMakeLists.txt:57` option `S41_SERVER_FFN_SPLIT` is OFF by default and guards all
helper code (`#if defined(S41_SERVER_FFN_SPLIT)` in `server.cpp`). `build-cuda-s43` must be configured with
`-DS41_SERVER_FFN_SPLIT=ON` (the first configure of 2026-09-28 missed it and was rebuilt). Check:
`strings build-cuda-s43/bin/libllama-server-impl.so | grep -c S41SERVERFFNCAPS` must print at least 1;
`server_bench.py run` refuses helper arms otherwise.

Environment (`benchlib.server_ffn_environment`, parsed by `server.cpp:196-300`; same keys as the scheduler's
`adapters/llama_server_contracts.py:1161-1176` + `adapters/phone_helpers.py:222-231`):

```
S41_SERVER_FFN_HELPERS=4                      (<= 8, server.cpp:210; at most one functionfs-usb, :235)
S41_SERVER_FFN_HELPER<k>_{LABEL,LAYER_MASK,TRANSPORT=tcp,HOST=<phone WLAN IP>,PORT}
S41_SERVER_FFN_LAYER_MASK=16777215            (union: layers 0-23; helper masks disjoint, cover it)
S41_SERVER_FFN_COLUMNS=17408                  (max phone columns = worker --columns; HELLO must match)
S41_SERVER_FFN_ARTIFACT_SHA256=sha256:d89e9e82...  S41_SERVER_FFN_N_EMBD=5120
S41_SERVER_FFN_F16_IO=1  S41_SERVER_FFN_ACTIVATION=swiglu  S41_SERVER_FFN_TIMEOUT_MS=120000
S41_SERVER_FFN_RUNTIME_CONTROL=1  S41_SERVER_FFN_MAX_TOKENS=4   (max_tokens < ubatch only with runtime control, :379)
```

argv (`benchlib.server_argv`, mirrors `adapters/llama_server.py:804-862`): `--fit off --ctx-size 4096
--parallel 4 --batch-size 2048 --ubatch-size 512 --flash-attn on --cont-batching --kv-unified
--no-cache-idle-slots --cache-type-k/v f16 --split-mode none --n-gpu-layers 16 --main-gpu 0 --device CUDA0
--threads 48 --threads-batch 64 --metrics --slots --no-webui --log-timestamps`, env
`CUDA_VISIBLE_DEVICES=0 GGML_CUDA_DISABLE_GRAPHS=1`. `-ngl 16` puts layers 24-39 on the GPU and keeps
layers 0-23 plus the output head (1.56 GB) CPU-resident: the desktop placement. This is a MECHANISM test;
on this server the whole 29.5 GB model fits one A6000 (arm `gpu`).

Runtime control (desktop protocol, decode-boundary-v1): helper connections are deferred
(`server.cpp:538`); every request carries `X-Scheduler-Request-ID` (`server-context.cpp:4677`; with the FFN
runtime configured, a batch without it is rejected, `apply_ffn_split_ubatch_context` :3990) and a pinned
`id_slot`; after the first streamed token the harness posts
`POST /v1/chat/completions/control {"action":"ffn_split","request_id","slot_id","policy_hash","plan_generation":1,"layer_mask":16777215,"columns":<4352|8704|13056|17408>,"enabled":true}`
(`ffn_split_cohort` with `members` for concurrent requests; `server-context.cpp:2543,2727,5279`). The server
connects the helpers on the first policy and only decode rows (<= 4) ever reach the phones; prefill is host.
Arms: `cpu` (no FFN env), `cpu-helpers` (env set, policy never posted: runtime overhead), `phone`,
`split-25/50/75`, `gpu` (reference). Show them with `python3 server_bench.py plan --arms ...`.

Per-call data: over TCP the client prints only `S41SERVERFFNCALL request/layer/tokens/columns/payload_bytes`
(`ffn-split-client.cpp:1742`); the timed `S41SERVERFFNUSB started_ns/h2d_completed_ns/d2h_completed_ns/compute_us`
lines exist only for FunctionFS (`:825`). The harness therefore reads (a) the per-helper shutdown summary
`S41SERVERFFN {"helper":..,"rpc_p50_ms","rpc_p90_ms","compute_p50_ms","host_p50_ms","wait_p50_ms",..}`
(p50/p90 only) and (b) with `--tap`, a protocol-aware proxy on 127.0.0.1 that timestamps every EXECUTE and
reads the worker's `compute_us` (per-call rpc/compute/transport, p99/p99.9, `tap-<arm>.jsonl`). The tap adds
about 0.25 ms per call (measured in loopback: split-50 step 285 vs 279 ms), so headline arms run without it
and one extra tapped run measures the tail. The parser also handles the USB lines if a helper is USB.

## Run order and gates

0. G0 build + model (today, done): second `build exit 0` in `server-prep/build_cuda_s43.log`, the strings check
   above, `sha256sum Qwen3-14B-Q4KM-dequant-f16.gguf` = config `artifact_sha256`
   (or `server_bench.py run --verify-sha256`).
1. Network (NETWORK_SETUP.md): FCHLLX01 `enp2s0` 192.168.77.1/24, router in AP mode, phones joined.
   `./phone_workers.sh ip` -> put the addresses into `wifi_config.json` `phones[].wlan_ip`;
   `./phone_workers.sh wifi-tune` (stay awake, low-latency + hi-perf WiFi).
   adb: from the desktop keep `["adb","-P","5037","-s","<serial>"]`, or from this server run
   `adb -s <serial> tcpip 5555` on the desktop, then `adb connect <ip>:5555` here and set `"adb":["adb","-s","<ip>:5555"]`.
2. G1 WiFi RTT (W1): `./phone_workers.sh echo-start`, then per phone, idle and both at once:
   `python3 ../wifi_rtt.py --host <ip> --port 7070 --calls 5000 --gap-ms 8 --out W1-<phone>.json`.
   **Go:** p99 <= 5 ms and p99.9 <= 15 ms per call (PLAN). Better: RTT p50 <= 3 ms, else only split-25 can win
   (table above). `./phone_workers.sh echo-stop`.
3. Workers: `./phone_workers.sh --dry-run start` (review), `./phone_workers.sh start`, `./phone_workers.sh status`.
4. G2 helper probe (W2): `python3 server_bench.py probe --config wifi_config.json --calls 500 --gap-ms 8 --out results/PROBE.json`.
   **Go:** every helper `ok` (HELLO accepted, n_ff 17408, offset 0, quantum divides 4352) and rows-1
   transport p99 within the W1 limits; compute p50 near 8.9 ms x 1 (OP15, full width) / 6.4 ms (Pixel).
5. Arms (S1): `python3 server_bench.py run --config wifi_config.json --arms cpu,phone,split-25,split-50,split-75 --concurrency 1,4 --out results/<run>`
   then one tapped run for the per-call tail: `... run --arms split-25 --tap --out results/<run>-tap`
   (copy `RESULT-cpu.json` into the tap directory first to get identity checks).
   Keep the host quiet (another user's jobs run on this host; `host_before/host_after` in each RESULT
   record load average and top processes).
6. `python3 analyze.py results/<run> --rtt op15=W1-op15.json --rtt pixel=W1-pixel.json` -> `ANALYSIS.md`.
   **Go (energy win on this server):** the best split arm's step time at c=1 and c=4 is at least 5 % below
   `cpu` (run-to-run noise here is about 1-2 %), identity vs `cpu` holds (or diverges only after a long
   common prefix: phones use HTP f16 / packed q4_K arithmetic, so bit identity is not guaranteed), no
   helper errors/resets, and GPU J/token drops with it. With the >100 W static floor, the relative step
   saving is the relative energy saving to first order; wall power is not measurable on this host (no
   meter, RAPL root-only), phone energy needs the desktop's `run_phone_energy.sh` method.
   **No-go:** best split within noise of `cpu` (the RTT eats the parallelism) -> this server gains nothing
   from phones at this model size; move to Stage S2 only with a lower-latency link.
7. `./phone_workers.sh stop`.

Loopback rehearsal of steps 3-6 without phones (mechanics only; stand-ins share the server's memory bus
and are slower than phones, numbers mean nothing): set every `wlan_ip` to `127.0.0.1` in a scratch copy of
the config, `./phone_workers.sh --config <copy> --local start`, run steps 4-6 with `--config <copy>`,
then `--local stop`.

## Outputs

`results/<run>/`: `RUN.json` (config + argv), per arm `RESULT-<arm>.json` (server argv, FFN env, helper
preflight, load time, idle window, per level: per wave per request token ids, token timestamps, TTFT,
applied token index, steady period, server timings; decode-window and full-window GPU J / mean W / CPU
busy; level metrics: step ms, ms/token, decode tok/s, e2e tok/s, GPU J/token, CPU J/token, identity vs
cpu; `ffn` digest: helper lines, ready line, call counts by layer / rows / columns, per-helper shutdown
summaries, shapes, errors, resets, control count; `tap` summaries), `samples-<arm>.json` (5 Hz NVML power +
energy counter, RAPL when readable, /proc/stat), `server-<arm>.log`, `server-<arm>.ffn.tsv` (timestamped
S41/FFNCONTROL lines), `tap-<arm>.jsonl`, `SUMMARY.md`; `analyze.py` writes `ANALYSIS.md`.

Metric definitions: the steady window of a request starts at token max(skip=8, applied index + 2); a
level's step period is the median over waves of the median request period (all c rows advance once per
step); decode tok/s = c x 1000 / step; GPU J/token = NVML total-energy-counter delta over the window where
every request of the wave decodes, divided by the tokens produced in it. RAPL
`/sys/class/powercap/intel-rapl:0/energy_uj` is root-only on FCHLLX01, so CPU package energy is `null`
(the result says why); `perf` is blocked too (`perf_event_paranoid=4`).

## Verified today (2026-09-28, no phones)

Arm (a) on the phones' exact artifact (sha256 verified by the copy: `d89e9e82...`), 5 prompts x 128 tokens,
temperature 0, `ignore_eos`, host quiet at start (load 0.45), 48 threads, `-ngl 16`
(`results/20260928-cpu-arm/`, console log next to it):

| arm | c | step ms | decode tok/s | e2e tok/s | GPU W (decode) | GPU J/token | CPU J/token |
|---|---|---|---|---|---|---|---|
| cpu | 1 | 224.4 | 4.46 | 4.34 | 102.2 | 21.7 | null (RAPL root-only) |
| cpu | 4 | 246.0 | 16.26 | 15.56 | 109.2 | 6.34 | null |
| gpu (reference: all 40 layers on GPU 0) | 1 | 42.7 | 23.4 | 23.2 | 294 | 12.0 | null |
| gpu | 4 | 44.7 | 89.5 | 81.2 | 292 | 3.22 | null |

- Wave-to-wave spread of the cpu step: 223.2-225.8 ms (c=1), 244.6-246.0 ms (c=4). Load from page cache 10.8 s;
  VRAM 12.5 GB with `-ngl 16`, 28.5 GB with every layer on the GPU.
- GPU board power with the server's CUDA context idle: 84 W (26 W without a context); during CPU-bound decode
  102-109 W. That floor is what a shorter step saves (plus the CPU package, not readable here).
- CPU-resident bytes per step 17.41 GB (layers 0-23 + output head, from the GGUF header); GPU part of a step
  16.1 ms (gpu arm scaled by bytes) -> fitted CPU streaming rate 83.6 GB/s (the 69 GB/s reference was
  measured under other users' load on 2026-09-20).
- `cpu-helpers` (runtime configured with the 4 WLAN helpers, deferred, never activated) first measured +9 %:
  another user's job (load 20-65) started mid-run. Rerun as an A/B pair under the same load
  (`results/20260928-cpu-pair-contended/`): cpu-helpers 253.5 / 264.7 ms vs cpu 246.9 / 267.4 ms at c=1 / 4,
  i.e. the idle FFN runtime costs nothing measurable, while the background load itself costs about 10 %
  (224 -> 247 ms). Run the phone campaign on a quiet host and keep `cpu` in the same run.
- Identity: cpu-helpers 5/5 identical 128-token sequences at c=1; at c=4 identity also depends on how the
  server happened to batch the concurrent prefills (6/8), so judge identity at c=1. gpu vs cpu 4/5 (different
  arithmetic, as expected).

Loopback mechanics (`results/20260928-loopback-standin/`, Qwen3-14B-f16 dev model, four stand-in CPU workers
on 127.0.0.1 with the example config's ports/masks/quanta): all arms ran, server helpers deferred then
connected on the first control, 2,784 phone calls per helper arm (24 layers x 116 steps, decode only,
prefill_calls 0), rows 1 and 4, columns 17408 (phone) / 8704 (split-50), control applied at token 2-3,
token identity 4/4 vs cpu for cpu-helpers/phone/split-50, `cpu-helpers` step within 1 % of `cpu`, tap
per-call rows complete (transport p50 0.12 ms, p99 0.44 ms loopback). The probe returned HELLO ok for all
four helpers.

## Caveats

- Concurrency must stay <= `max_tokens` 4 (decode steps with more rows are not sent to the phones).
- In helper arms concurrent prefills can be split into separate batches (TTFT 2.1 s vs 1.1 s for some
  requests at c=4 in loopback); compare decode step periods, not e2e tok/s, across arms.
- NVML `power.draw` is a 1 s average on Ampere; energy comes from the NVML total-energy counter (mJ).
- The OP15 worker binary (2026-09-04) and the Pixel's (2026-09-25) speak protocol v6 like this build; the
  probe's HELLO fails fast if not.
- `phones[].lock_path` (Pixel kernel lock) is per phone and allowed only for a single-worker phone.
- Phone launches are detached with `nohup setsid` inside `su -c` (set `"setsid": false` per phone if the
  phone's toybox lacks it, `"as_root": false` to drop `su`); the pid file under `phones[].log_dir` is the
  worker's own pid, logs are `<log_dir>/<label>.log`. `adb shell -T` needs adb >= 1.0.37 (shell v2).
