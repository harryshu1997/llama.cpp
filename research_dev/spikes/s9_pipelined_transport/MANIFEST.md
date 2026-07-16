# S9-V1A Reproduction Manifest

Date: 2026-07-14. Host tree: `/home/myid/zs89458/Documents/llama.cpp-release`.

## Identity

- Model shard: `scratchpad/phone_pim/12b-f16-mid-2-3.gguf`, 464,114,176 bytes,
  SHA-256 `5cfba18d2a47acc190f317d650895bcc53e914e9a0bc61631860be9591ed360d`, 111 x 4 MiB chunks.
- OP12 = `5ae7a43d`, HTP v75, USB 6-2.  OP15 = `3C15AU002CL00000`, HTP v81, USB 8-3.
- **Certified baseline worker (unchanged, frozen):** `0a50ca72...e749` - still present on both phones.
- Binaries built for this slice (all opt-in; v3 wire byte-identical):
  - host (windowed + instrumented) `ad01b6a9d4d026a15abfaa2a3d299ebf0dffd12909619fd1760e2ebc00e9aeac`
  - bench host `ff62cc0e4d95d8fd28ebbb07a6df8774a3736c862a613ab71b243d51d9e6d49a`
  - bench android `fd1141d44fbe0d8c5cc43241cd3effce5bcc30f1293d37403aab2f3166758c93`
  - measurement worker android (stderr timing only) `aaa928b85acaeb8beed75237dcf2afd681d7b9d8c7a49fcd10080298e5e4c662`
    deployed on-device as `llama-phone-pim-worker-meas` (does NOT overwrite the certified worker).

## Build

```sh
# host (Linux)
cmake --build build-phone-pim --target llama-phone-pim-host llama-phone-pim-worker \
  llama-phone-pim-bench test-phone-pim-protocol test-phone-pim-store test-phone-pim-stream --parallel
ctest --test-dir build-phone-pim -R phone-pim --output-on-failure          # 3/3
ctest --test-dir build-phone-pim-asan -R phone-pim --output-on-failure     # 3/3 (ASan/UBSan)

# android (docker toolchain)
docker run --rm -v "$PWD:/workspace" -w /workspace snapdragon-toolchain-hostgcc:v0.3 bash -lc \
  'cmake --build build-snapdragon --target llama-phone-pim-worker llama-phone-pim-bench --parallel'
```

## Reproduce

- CP1 accounting (instrumented, window=1):
  `research_dev/spikes/s9_pipelined_transport/run_prod_instrumented.sh <dev> HTP0 <dport> <hport> <out.json> 1`
- CP1 transport bench: `run_bench.sh <dev> <base_port> <label> <out.jsonl>` (needs android bench pushed to `/data/local/tmp/phone_pim/llama-phone-pim-bench`).
- CP2 gate sweep: `run_prod_sweep.sh <dev> HTP0 <base_port> <out.jsonl> 5` -> `python3 analyze_sweep.py`.
- CP2 adversarial: `adversarial_tests.sh <dev> HTP0 <base_port> <out.txt>`.

Device worker launch (measurement build, fresh empty store, no `--model`):
```sh
env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=3072 \
  ./llama-phone-pim-worker-meas --store-dir STORE --max-store-mib 2048 --max-model-mib 1024 \
  --min-free-mib 256 --backend HTP0 --bind 127.0.0.1 --port PORT --route-epoch 17 --generation 1
```
Host provision at a chosen window:
```sh
build-phone-pim/bin/llama-phone-pim-host -m scratchpad/phone_pim/12b-f16-mid-2-3.gguf \
  --host 127.0.0.1 --port HOST_PORT --prefix blk.2 --M 16 --repeat 7 \
  --route-epoch 17 --generation 1 --provision if-missing --chunk-mib 4 --stage-window 4 \
  --release --shutdown
```

## Artifacts

`artifacts/`: `cp0_reproduce.json`, `op1{2,5}_prod_instrumented.json` (+ `_worker.log`),
`op1{2,5}_bench.jsonl`, `adb_push_control.json`, `cp1_accounting.json`,
`op1{2,5}_sweep.jsonl`, `cp2_sweep_summary.json`, `op1{2,5}_adversarial.txt`,
`op12_w{1,4}_correct.json`.

## S9-V1A-R repair (2026-07-15)

Repaired, opt-in profiling + accounting/bounds fixes. v3 wire byte-identical; certified worker
0a50ca7299ee15a9... untouched on both phones. Repair binaries:
- host `8e6af30ac45a6ea4105c8ee9704de34e77c72f96713173244758945ecb90654b`
- host bench `ac4b8d21086dfb31663f25dae6efdc0a07b8c9adebc000aa674b89524ecde8d3`
- android measurement worker `c284227179b4048feef928f03f03e984cdbdae3bb2e90f25971b2c98cd764f27`
  (deployed as llama-phone-pim-worker-meas; requires --profile-recv to profile, default OFF)
- android bench `980bdf70fff16e49ddf4ce4b55fd01927252d3a17ec7b18cbd58fb28352830cc`

New opt-in flags: worker `--profile-recv`; host `--profile-transport`; host `--stage-window {1,2,4,8}`.

Repair harnesses (research_dev/spikes/s9_pipelined_transport/):
- run_matrix.py       - per-device matrix (profile/64/256/gate), Latin-square, provenance, thermal.
- run_simultaneous.py - both phones concurrently on separate USB buses.
- adversarial_tests.py - structural-JSON T1/T2/T3, fail-closed, persists records, retry/waste.
- analyze_sweep.py    - fail-closed full-shard gate + `--selftest` (18 mutation cases).
- analyze_profile.py  - CP1.7 honest accounting (host/phone timers kept separate).
- analyze_scaling.py  - 64/256 MiB + simultaneous goodput.
- asan_window_test.sh - ASan/UBSan test of the real host window>1 pipeline.
V1A originals preserved under v1a_historical/; V1A raw artifacts under artifacts/; repair raw
artifacts under artifacts_r/.

Reproduce the repaired matrix + gate:
```sh
python3 run_matrix.py 5ae7a43d OP12 6-2 HTP0 46000 artifacts_r/matrix v1a-r-1
python3 run_matrix.py 3C15AU002CL00000 OP15 8-3 HTP0 47000 artifacts_r/matrix v1a-r-1
python3 analyze_sweep.py --gate artifacts_r/matrix/OP12_gate.jsonl artifacts_r/matrix/OP15_gate.jsonl
python3 analyze_sweep.py --selftest
bash asan_window_test.sh
```
Device profiling run (opt-in): worker adds `--profile-recv`; host adds `--profile-transport`.
