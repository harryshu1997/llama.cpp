# Reproduction Manifest

## Identity

- Host tree: `/home/myid/zs89458/Documents/llama.cpp-release`
- Model: `12b-f16-mid-2-3.gguf`
- Bytes: 464114176
- SHA-256: `5cfba18d2a47acc190f317d650895bcc53e914e9a0bc61631860be9591ed360d`
- OP15 serial: `3C15AU002CL00000`, HTP v81, USB path 8-3 at 5000 Mbps
- OP12 serial: `5ae7a43d`, HTP v75, USB path 6-2 at 5000 Mbps

## Build

```sh
cmake --build build-phone-pim --target \
  test-phone-pim-protocol test-phone-pim-store test-phone-pim-stream \
  llama-phone-pim-worker llama-phone-pim-host --parallel
ctest --test-dir build-phone-pim -R phone-pim --output-on-failure

docker run --rm -v "$PWD:/workspace" -w /workspace \
  snapdragon-toolchain-hostgcc:v0.3 bash -lc \
  'cmake --build build-snapdragon --target \
   llama-phone-pim-worker test-phone-pim-protocol test-phone-pim-store --parallel'
```

## Device Command Shape

Run the worker in `/data/local/tmp/phone_pim` with its HTP libraries:

```sh
env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=3072 \
  ./llama-phone-pim-worker --model MODEL --backend HTP0 \
  --bind 127.0.0.1 --port PORT --route-epoch 17 --generation GENERATION
```

Forward one host port per phone, then run:

```sh
build-phone-pim/bin/llama-phone-pim-host \
  -m scratchpad/phone_pim/12b-f16-mid-2-3.gguf \
  --host 127.0.0.1 --port HOST_PORT --prefix blk.2 --M 16 --repeat 7 \
  --route-epoch 17 --generation GENERATION --island-id ISLAND \
  --release --shutdown
```

Exact result records are under `artifacts/`. Device, session, HTP revision,
thread-count, and HMX route evidence is recorded separately in
`artifacts/worker_route_provenance.txt`. Worker stderr proved HTP0 v81/v75
initialization and no CPU fallback.

## Dynamic V3 Identity

- Host binary SHA-256: `64db82ee4133f0d31dadfbe07cd90d21bd4ac771eee376ad5ae84f432a6e9e7c`
- Android worker SHA-256: `0a50ca7299ee15a9eb9d524a0dadea40b467721f011a57b7f89617b1bd30e749`
- Protocol test SHA-256: `1e2c41727d01bcf58209fedeec7b87026ef66d3e416d4ad4cff5d79b9170a1ab`
- Store test SHA-256: `71c2cd644f58fac0760ba0e1138b430c1b5125da489d74168b343c16f6b62579`
- Manifest SHA-256: `c07439ab9e72e294bf5b3cb4241c8436e1fd7c6e55a63a1f58cba7b2fe77e075`
- Chunk bytes/count: 4,194,304 / 111
- Published mode: 0400

Start a worker without `--model`, using an empty private store:

```sh
env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=3072 \
  ./llama-phone-pim-worker --store-dir STORE --max-store-mib 2048 \
  --max-model-mib 1024 --min-free-mib 256 --backend HTP0 \
  --bind 127.0.0.1 --port DEVICE_PORT --route-epoch ROUTE --generation 1
```

Provision and execute:

```sh
build-phone-pim/bin/llama-phone-pim-host \
  -m scratchpad/phone_pim/12b-f16-mid-2-3.gguf \
  --host 127.0.0.1 --port HOST_PORT --prefix blk.2 --M 16 --repeat 7 \
  --route-epoch ROUTE --generation 1 --provision if-missing --chunk-mib 4 \
  --release --shutdown
```

The exact OP15 and OP12 records are `artifacts/op15_dynamic_v3.json` and
`artifacts/op12_dynamic_v3.json`. `DYNAMIC_RESULTS.md` records the executed
failure tests and the performance boundary.
