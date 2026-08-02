# S41 Gemma desktop router smoke

Verdict: `B1_B8_SMOKE_PASS`

This is a geometry and placement smoke, not a quality result. The eight numeric
prompt arrays were copied from the prior Qwen smoke workload and rebound to
Gemma. Their token IDs are in range for Gemma, but the decoded text is not a
semantic workload.

## Frozen bindings

- Model: `gemma-4-12b-it-q8_0`
- GGUF bytes: `12669645856`
- GGUF SHA-256:
  `7b56cbd0e0d96d5c8d7df9c21b39d67264e58cbabf7a96d4681eff8b3d492848`
- Paired model in the S41 authority:
  `qwen3-14b-q4_k_m`, Q4_K_M
- GPU: RTX 4060 Ti,
  `GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08`
- llama-server: version 9875 (`7d1926dff`), SHA-256
  `f890165ce1f89d3084bc108d05b42e5ca4e1a0f25aa40fe7786d39ab87562d6a`
- S41 contract SHA-256:
  `cc1992fcac6c834671b5bce77c2a3b65ddf756b49482683376cbc41b1ed72b81`
- Executor bundle manifest SHA-256:
  `f315a0b737145da1f6c3532739db6ec38275c7aeffcae2ac8c8d5513eb8376f1`

## Realized route

The runtime certificate records CUDA, `--n-gpu-layers 99`, flash attention on,
8 parallel slots, continuous batching, context 4096, batch 2048, ubatch 512,
F16 K/V cache, and split mode none. Free GPU memory after load was 1959 MiB,
above the frozen 512 MiB minimum.

The successful run used `COLD_NVME`. The unmodified S40 warm-cache gate requires
at least 95 percent file residency. On this host the 12.67 GB Gemma file reached
only 72.49 percent under current RAM pressure, so the two warm pre-load attempts
failed closed before a child process or GPU model allocation. No threshold was
weakened.

## B1 and B8

| Geometry | Tokens | Elapsed (s) | Aggregate tokens/s |
| --- | ---: | ---: | ---: |
| B1 | 8 | 0.961673 | 8.319 |
| B8 | 64 | 3.330837 | 19.214 |

B8 delivered 2.310x the aggregate decode throughput of B1 in this short
one-token-quantum smoke.

## Cleanup

The model child exited with status 0. Harness cleanup reports no active model,
request session, or busy request. After stopping the parent router, port 48991
was closed, no llama-server or smoke process remained, and the GPU returned to
299 MiB used and 15649 MiB free.

The experimental native router printed `stack smashing detected` while handling
SIGTERM after its child had already exited cleanly. The same terminal symptom is
present in the earlier S40 Qwen smoke. It caused no leaked process or GPU state,
but it is a router shutdown bug that should be fixed before long campaigns.

`ROUTER_ARGV.json` and `SMOKE_ARGV.json` preserve the exact commands and
environment. `SMOKE_RESULT.json` is the canonical result. `PRE_RUN.txt` and
`POST_RUN.txt` preserve the terminal hardware checks.
