# Phone PIM-Style FFN Prototype

This directory contains an experimental command-driven phone accelerator. It is
PIM-style because a verified model shard remains in phone storage and backend
memory while the host sends only commands and boundary activations. It is not
literal PIM, cache-coherent memory, or server-memory mapping.

The current capability is deliberately narrow:

- one pre-staged or dynamically provisioned Gemma4 dense FFN layer;
- one persistent backend context and graph per worker;
- HTP or another explicitly named GGML backend;
- a production Gemma4 CPU graph as the host correctness oracle; and
- bounded request/response frames over a forwarded TCP socket;
- one resumable UFS staging transfer with explicit replacement; and
- content-addressed publication before PREPARE.

It does not schedule multiple models, expose a llama-server API, authenticate the
wire, or prove capacity or energy savings. Dynamic provisioning moves a complete
GGUF shard. It does not yet stream individual tensors into a live graph.

## Data path

1. The host and worker negotiate a session epoch derived from a fresh worker-boot
   nonce, so frames from an earlier process cannot be replayed after restart.
2. Optional STAGE commands bind a ticket, object hash, ordered chunk map, and
   residency generation. Each chunk is hashed, written, and `fdatasync`ed before
   its durable-prefix acknowledgement.
3. The worker reserves the complete object, reconstructs a verified prefix after
   disconnect or process restart, verifies the full object, publishes it with
   `renameat2(RENAME_NOREPLACE)`, and `fsync`s the directory.
4. PREPARE explicitly requests either the pre-staged source or the published-store
   source. It cannot silently fall back across that boundary.
5. The worker opens the shard once, verifies and parses that same file descriptor,
   uploads the five dense-FFN tensors, builds the graph, and warms it.
6. EXECUTE transfers an `M x n_embd` F32 boundary tensor.
7. The resident graph runs RMS norm, gate/up projections, GEGLU, down projection,
   post-FFW RMS norm, and the residual add.
8. The result returns to the host and is compared with tensors captured from the
   production Gemma4 graph.
9. RELEASE destroys residency and advances its generation. Old generations are
   rejected after reconnect.

A verified descriptor cache removes redundant full-file hashes within one worker
process. It is keyed by content identity and checked against the current path,
inode, size, mtime, and ctime. A worker restart has no such cache and performs one
full-file hash before returning a published cache hit.

Backend failure clears and quarantines the resident island by advancing the same
generation. The random session nonce protects process restarts, while the route
epoch is still a prototype startup parameter. A future scheduler must own and
durably advance route epochs.

## Build

```sh
cmake -B build-phone-pim -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build-phone-pim --target \
    llama-phone-pim-worker llama-phone-pim-host llama-phone-pim-fleet \
    test-phone-pim-protocol test-phone-pim-client \
    test-phone-pim-store test-phone-pim-stream
ctest --test-dir build-phone-pim -R phone-pim --output-on-failure
```

## Run

Start the worker beside a pre-staged shard:

```sh
./llama-phone-pim-worker \
    --model /data/local/tmp/phone_pim/model.gguf \
    --backend HTP0 --bind 127.0.0.1 --port 9090 \
    --route-epoch 1 --generation 1
```

Forward the port and run the host against the byte-identical local shard:

```sh
adb forward tcp:19090 tcp:9090
./llama-phone-pim-host \
    --host 127.0.0.1 --port 19090 \
    --model model.gguf --prefix blk.2 --M 16 --repeat 7 \
    --route-epoch 1 --generation 1 --release --shutdown
```

The host exits nonzero for protocol, remote, or correctness failure. A successful
run emits one JSON record with `verdict=PRESTAGED_FFN_PASS` and
`oracle=production_gemma4_cb_eval`.

For dynamic provisioning, start the worker with a private store instead of a
pre-staged model:

```sh
./llama-phone-pim-worker \
    --store-dir /data/local/tmp/phone_pim/store \
    --max-store-mib 2048 --max-model-mib 1024 --min-free-mib 256 \
    --backend HTP0 --bind 127.0.0.1 --port 9090 \
    --route-epoch 1 --generation 1

./llama-phone-pim-host \
    --host 127.0.0.1 --port 19090 \
    --model model.gguf --prefix blk.2 --M 16 --repeat 7 \
    --route-epoch 1 --generation 1 --provision if-missing \
    --chunk-mib 4 --release --shutdown
```

`--test-stop-after-chunks N` closes the socket without a CLOSE command after N
durable acknowledgements. A later invocation resumes the verified prefix.
`--replace-active-staging` explicitly aborts a different incomplete transfer;
without it the host fails closed.

A successful dynamic run emits `verdict=DYNAMIC_FFN_PASS` and
`model_source=published_store`. That verdict certifies provisioning/execution
correctness for the one run; it is not a capacity or energy verdict.

## Live fleet runner

`llama-phone-pim-fleet` keeps one protocol session per phone and drains a real
FFN job queue across all READY devices. Each fresh session performs
`HELLO -> STATUS`; the STATUS generation is authoritative, so a stale command
line hint cannot authorize PREPARE or EXECUTE. PREPARE remains the exact
resident model-identity check because STATUS does not carry the model digest.

Start one worker per phone and forward each port, then run:

```sh
./llama-phone-pim-fleet \
    --device op12,127.0.0.1,19012,5ae7a43d \
    --device op15,127.0.0.1,19015,3C15AU002CL00000 \
    --model model.gguf --prefix blk.2 --M 16 --jobs 16 \
    --route-epoch 1 --generation-hint 1 --island-id 1
```

The runner loads the same FFN island on a local CPU backend before dispatch,
builds finite deterministic inputs, and checks every returned result against
that oracle. Each phone receives an initial job; after that, the first phone to
finish takes the next queue item. Successful output is labeled
`REAL_FLEET_FFN_PASS` and reports remote setup separately from the validated
dispatch makespan. The physical IDs are caller assertions and must be checked
against the external device/forward manifest.
Local source hashing and CPU-oracle preparation/execution are also reported and
are outside both remote setup and dispatch.

Fleet records use schema v2. Session, route, island, and generation uint64
identities are decimal strings so JSON consumers preserve their exact values.
On a failed setup or dispatch, a possibly created residency is reconciled with
STATUS and rebound by exact PREPARE identity before RELEASE.

The relative-L2 comparison for each returned job runs inline before that device
takes another queue item. `validated_dispatch_makespan_ms` therefore includes
this harness-only check. `last_rpc_complete_ms` marks the final RPC return but
does not remove earlier inline checks from queue pacing.

This is a live operator-island harness, not `llama-server` integration. It
currently uses the TCP path selected by the caller (the example ADB forwards
use USB), supports one Gemma4 dense FFN identity per worker, and keeps the
server CPU oracle enabled. It does not yet implement mixed-model placement,
server fallback, WiFi-input/USB-output separation, or an energy claim.

## Protocol boundary

The wire header is explicitly little-endian and binds request ID, command sequence,
session epoch, route epoch, and residency generation. Frames have a fixed maximum,
CRC32 header check, SHA-256 payload check, monotonic IDs, and one absolute I/O
deadline. Epochs and hashes are not authentication; the prototype assumes a
trusted lab host and a localhost-only worker reached through `adb forward`. It
implements the sequential chunk/ticket mechanics but not the complete S9 control
plane: there are no authoritative residency leases, multi-model cache policy,
live lane credits, or two-level scheduler in this executable.
