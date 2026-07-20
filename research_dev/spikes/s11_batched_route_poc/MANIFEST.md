# S11-B Reproduction Manifest

Date: 2026-07-16 EDT.

## Devices

```text
A6000: GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf
OP15:  3C15AU002CL00000, HTP v81, USB 8-3 at 5000M
OP12:  5ae7a43d, HTP v75, USB 6-2 at 5000M
```

## Original Sweep Binary Identity

```text
786defa77b157aad6745aa2b37a0e1d11a2cb89a31ac938e5ea81d0694717c4d  host llama-layersplit
9472537438612242b5ce9ce25aa739e531b0f1013af7245bfd5a0c3703728979  Android llama-layersplit
6fbf004a8ae62a13ce18d00059510546f63499634094fd4ec100840bd663ab71  Android libllama.so
```

The Android files were deployed under `/data/local/tmp/ls-npu-s11b/`, leaving
the historical S11 binary and libraries unchanged.

## Repaired V2 Checkpoint Identity

The fail-closed stage handshake and runner-v2 checkpoint use:

```text
914fd377acb5f604bbcad225ceb650b093adbcfa2faaa07e2347260a39140eb4  host llama-layersplit
e518ed4534b38d75f476d6bd9afdc012742ec3e4503aacceddd48ec650b20fa1  Android llama-layersplit
e61ee6fe4b760d16dd77ef40a57f948da7e8f7e17620f42933631597f993eca2  Android libllama.so
23462adb1c35fe08dcfab00af81e5d5f9c2e0c16fa6e49f3bd6c3e134e9e512a  runner v2
3c1060b9726865cda304f5eefea53b479b2fe7aba3798f5925fa6163693f5c21  layersplit.cpp
36c14f4d1a837f34ae6fc01ec0bf92a1f23663ec3b4fdb11d3c887b8534a3dd5  llama-model.cpp
```

The repaired Android files are under `/data/local/tmp/ls-npu-s11r/`. The
hash-bound reproduction is:

```text
scratchpad/s11_runner_v2_final_r2_b8_20260716/
plan sha256:    76ff53654c47b295816f548df38c167b3a5bfb601f90f4f0f6dd13256862f36e
summary sha256: 3a9bc141f740b63b06b6b6741e90bb9af8b75eb8014993fbe6c69a2ac8306ed2

scratchpad/s11_runner_v2_final_r2_two_phone_b1_20260716/
plan sha256:    f304444af338056b51f72d6f26a64442c3029c4db725ea5f9159dec42af19006
summary sha256: abb4f730ed19a8c946845576a7c158e5e64b680439ca4fdb4f755e26f12b0c96
```

## Build

```sh
cmake --build build-cuda --target llama-layersplit -j4
cmake --build build-cpu --target llama-layersplit -j4

docker run --rm -v "$PWD:/workspace" -w /workspace \
  snapdragon-toolchain-hostgcc:v0.3 bash -lc \
  'cmake --build build-snapdragon --target llama-layersplit --parallel 4'

python3 research_dev/spikes/s11_fixed_route_poc/test_fixed_route.py
```

Expected harness result: `26 tests`, all pass.

## Representative B=8 Command

```sh
python3 research_dev/spikes/s11_fixed_route_poc/run_fixed_route.py \
  --host-bin build-cuda/bin/llama-layersplit \
  --host-model /home/myid/zs89458/Documents/models/gemma-4-12B-it-f16.gguf \
  --gpu-uuid GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf \
  --route op15 \
  --output scratchpad/s11_batch_b8_repeat3_20260716 \
  --prompt 'In one sentence, explain why batching improves accelerator utilization.' \
  --chat --n-gen 4 --requests 16 --warmups 2 --pairs 3 \
  --batch-size 8 --driver-context 96 --driver-max-prefill 64 \
  --op15-dir /data/local/tmp/ls-npu-s11b \
  --phone-remote-bin llama-layersplit
```

No command in this manifest enables physical energy measurement.
