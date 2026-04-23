## ADDED Requirements

### Requirement: Disk offload ggml ops
Five new ggml ops SHALL be added to support Tier-2 disk-backed KV storage:
- `GGML_OP_KV_OFFLOAD` — marks a tensor for serialization to a per-layer file; compute is a metadata-only no-op that performs the actual write.
- `GGML_OP_KV_LOAD` — loads a tensor from disk at a given file offset into a GPU- or CPU-addressable buffer.
- `GGML_OP_KV_CLEAN` — deletes all disk-resident KV files associated with the current process/session.
- `GGML_OP_KV_PREFETCH_START` — initiates an asynchronous read of the next layer's offloaded KV block; returns immediately.
- `GGML_OP_KV_PREFETCH_WAIT` — blocks until a previously started prefetch completes and returns the loaded tensor.

All five ops SHALL be implemented on the CPU backend. OpenCL support is not required in this change (data lands in host memory; subsequent ops copy to device).

#### Scenario: KV_OFFLOAD writes then KV_LOAD reads
- **WHEN** a graph calls `KV_OFFLOAD(tensor_A, layer=3)` and later `KV_LOAD(layer=3, offset=0, size=tensor_A.nbytes)`
- **THEN** the loaded tensor has bit-identical contents to the original `tensor_A`

#### Scenario: KV_PREFETCH overlaps compute
- **WHEN** a graph issues `KV_PREFETCH_START(layer=L+1)` during layer `L`'s compute, then `KV_PREFETCH_WAIT` at the start of layer `L+1`
- **THEN** total wall-clock time is no worse than a synchronous `KV_LOAD` issued at the start of layer `L+1`
- **AND** strictly less when the underlying storage is faster than compute (measurable on NVMe + CPU-bound workloads)

#### Scenario: KV_CLEAN removes all files
- **WHEN** a session allocates and offloads KV for layers 0-34 and then invokes `KV_CLEAN`
- **THEN** no files remain under the session's temp directory after the call returns

### Requirement: Tier-2 storage directory lifecycle
The disk offload ops SHALL write to a per-session temporary directory whose path is:
1. Taken from env var `LLAMA_KV_OFFLOAD_DIR` if set, or
2. Created under `$TMPDIR/llama-kv-<pid>-<random>` otherwise.

The directory SHALL be created on the first offload call and removed by `KV_CLEAN` or at normal process termination.

#### Scenario: Custom directory honored
- **WHEN** `LLAMA_KV_OFFLOAD_DIR=/mnt/fast_nvme/kv` is set
- **THEN** offload files appear under that directory

#### Scenario: Auto-cleanup on llama_free
- **WHEN** the process calls `llama_free()` without explicit `KV_CLEAN`
- **THEN** the session's offload directory is removed as part of cleanup
