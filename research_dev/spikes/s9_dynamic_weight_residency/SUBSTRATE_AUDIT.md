# Substrate Audit (S9-V0)

Status: file/line-cited, read-only audit of the CURRENT tree at HEAD `933c722f6`
(dirty worktree preserved; nothing edited). Every mechanism a host-managed
dynamic weight-residency pipeline could reuse is classified REUSE / ADAPT /
REFERENCE_ONLY / REJECT through the residency lens. Twelve of the load-bearing
citations were re-verified by hand against the tree before this file was written
(see the verification note at the end of each section header where applicable);
the two most surprising -- the RPC identity hash is FNV-1a not SHA-256, and the
download path has zero fsync -- were confirmed directly.

This audit exists to prevent one specific error: treating a STRUCTURAL check
(name present, count matches, byte range in bounds) as a CONTENT-IDENTITY check
(these are exactly the expected bytes for exactly this model version, layout,
backend, SoC, boot epoch, and residency generation). Nowhere in the current tree
is any weight-tensor payload cryptographically verified: a full-file grep of the
loader, the downloader, the RPC cache, and the two sharing paths returns ZERO
SHA-256 / CRC of tensor bytes. The residency pipeline's VERIFYING state and its
derived-image identity are therefore NET-NEW, wrapped around reusable byte
plumbing -- not adapted from an existing verification layer.

Classification key (strict; the S8 audit downgraded several REUSE->lower on
review, so the bar here is "safe for a persistent host-orchestrated agent"):
- REUSE: adopt largely as-is.
- ADAPT: the shape/pattern is right; specific hardening required (stated).
- REFERENCE_ONLY: study the idea; do not adopt the code.
- REJECT: not suitable / absent; must be built.

The residency pipeline being audited FOR (two linked state machines, full
definition in WEIGHT_RESIDENCY_CONTRACT.md):

```
ABSENT -> RECEIVING -> VERIFYING -> VERIFIED_ON_DISK
VERIFIED_ON_DISK -> MATERIALIZING -> LPDDR_READY
LPDDR_READY -> PREPARING_HTP/GPU -> WARMING -> READY_HTP/GPU
READY -> LEASED -> DRAINING -> EVICTING
any state -> ERROR or QUARANTINED
```

Content identity is SHA-256. A derived (backend-prepared) image identity binds
model_version, tensor digest, graph hash, backend build, SoC, architecture,
layout version, boot epoch, and residency generation.

---

## A. llama_model_loader: tensor offsets, load_data_for(), materialization

Files: `src/llama-model-loader.{h,cpp}`, callers in `src/llama-model.cpp`.

This is the canonical MATERIALIZING -> LPDDR_READY path. It locates each tensor's
bytes by GGUF metadata offset and either mmaps them in place or read()s/uploads
them into a backend buffer. The physical-byte accounting is solid and reusable;
the content verification for a VERIFYING gate is entirely absent.

### A1. `llama_tensor_weight` per-tensor locator -- REUSE
- Struct = (source file idx, byte offset within file, tensor): the natural
  WeightSegment physical descriptor, immutable after construction.
  `src/llama-model-loader.h:33-50`.
- Offset provenance `offs = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i)`,
  guarded by an overflow AND end-of-file bounds check that throws "not within the
  file bounds, model is corrupted or incomplete". `src/llama-model-loader.h:45-48`.
- Main file registers all tensors with `idx=0` (`:587`); split shards register
  with their own idx and per-shard gguf context so per-split offset resolution is
  correct; duplicate tensor names across shards throw (`:644-653`, `:580-587`).
- GAP: binds a byte range to (name, split) only -- no content digest, no
  model_version / layout / backend identity. Reuse the struct as the physical
  locator; a residency WeightSegment adds `expected_sha256` and a transfer-chunk
  map on top.

### A2. Byte-count accounting (n_bytes, size_data, size_done, n_created) -- REUSE
- `n_elements`/`n_bytes` accumulated across shards at metadata load
  `:651-652`; `size_data` computed in `init_mappings` (`:1366-1368`; note
  TENSOR_DUPLICATED also adds to it at `:1284`, so the two must agree);
  `size_done += n_size` per tensor is the natural MATERIALIZING progress meter
  and the completion signal `size_done >= size_data` (`:1646`, `:1673`).
- GAP: byte-count only, never content. `size_done >= size_data` means "all bytes
  touched", not "all bytes correct". Reuse as a progress/quota meter, not as a
  VERIFIED gate.

### A3. `load_data_for()` single-tensor fetch (mmap-alias vs seek+read) -- ADAPT
- mmap path ALIASES the tensor onto mapped file pages (no copy) when
  `data==nullptr`; else memcpy from the mapping `:1391-1397`. The "materialized"
  RAM is thus page-cache backed by a mutable file -- not a durable independent
  copy.
- Non-mmap path `file->seek(w.offs); file->read_raw(...)` `:1398-1404`.
- Only optional validation is `ggml_validate_row_data` (NaN/Inf/enum-range scan),
  `:1406-1408`.
- GAP: to serve VERIFYING, hash `cur->data` after read and compare to the
  per-segment SHA-256 before publish; forbid/fence the mmap-alias path for the
  durable residency image (aliased pages are mutable via the file). Keep the
  seek/offset mechanics, wrap with hash-on-read.

### A4. `load_all_data()` materialization loop -- ADAPT
- mmap branch allocates onto mapped pages or `ggml_backend_tensor_set` copies into
  a device buffer `:1541-1568`; host-buffer branch seeks+reads into the final CPU
  buffer with optional async NaN/Inf validation `:1572-1579`; async GPU path uses
  4x pinned staging buffers with per-buffer events `:1582-1633`.
- Validation and async upload are MUTUALLY EXCLUSIVE: `check_tensors` forces
  `upload_backend=nullptr` (`:1446`), so real device offload gets speed OR the
  weak NaN check, never both, and never a content hash.
- Not transactional: tensors publish into live buffers as the loop runs; a throw
  (`:1669`) or a progress-callback cancel (`:1535`) leaves an unknown partial set
  materialized with no rollback and no per-segment journal. Tensors absent from
  `weights_map` are silently skipped (`:1526-1531`).
- GAP: make each segment atomic (stage -> hash -> verify -> publish), fail-fast,
  and record a per-segment READY marker. Verify SOURCE bytes on host independently
  of the device upload, then optionally read back.

### A5. mmap lifecycle (init_mappings / fragment unmap / mlock) -- REFERENCE_ONLY
- One mmap per file, optional `madvise(WILLNEED)` prefetch + mlock pinning,
  NUMA-aware `:1338-1362`; only the unused head/tail fragments are unmapped after
  load, and the tensor-bearing span stays mapped for the model lifetime
  (`:1371-1386`, `:1673-1683`; mappings moved into the model at
  `src/llama-model.cpp:1625-1628`).
- GAP: naive mmap makes weights file-backed and mutable-under-foot -- the opposite
  of a verified immutable LPDDR image bound to a residency generation. Study the
  fragment-unmap and mlock ideas; a residency store needs its own anonymous/locked
  buffer whose contents were hashed at VERIFYING.

### A6. Content-integrity of tensor payload (the missing VERIFYING gate) -- REJECT
- The only payload check is `ggml_validate_row_data` (NaN/Inf + per-quant
  enum-range), and it is gated by `check_tensors` which DEFAULTS FALSE
  (`src/llama-model.cpp:2302` `/*.check_tensors =*/ false`). Default load performs
  ZERO payload inspection. `src/llama-model-loader.cpp:822`, `:1406`.
- Full-file grep: no SHA-256, no CRC, no content digest, no fsync anywhere in the
  loader. Two different-but-finite weight files pass identically.
- GAP: the entire VERIFYING/VERIFIED_ON_DISK identity gate is unimplemented. It
  must be built around the loader (hash each segment's exact `[offs, offs+nbytes)`
  span, compare to a signed manifest, bind the full identity tuple). A residency
  loader must invert the `check_tensors` default to always-verify.

---

## B. GGUF split/shard behavior + the offline shard extractor

Files: `src/llama-model-loader.cpp` (split path), `src/llama.cpp`,
`src/llama-arch.cpp`, `research_dev/shard_gguf.py`.

llama.cpp's native multi-file "split" gives a usable skeleton for an atomic
multi-file WeightSet -- but its member identity is name+count only, never
content. Critically, the project's ACTUAL shards come from `shard_gguf.py`, which
is NOT a llama.cpp split at all, so the loader's split validation never runs on
them.

### B1. Split discovery + naming convention -- ADAPT
- Siblings enumerated purely by regenerating `"%s-%05d-of-%05d.gguf"` from a
  prefix + (idx, n_split); no directory listing, no manifest.
  `src/llama-model-loader.cpp:76-101`, `src/llama.cpp:505-549`.
- GAP: grouping identity is the FILENAME PATTERN only -- two different builds with
  the same base name + count collide silently. Replace name-pattern grouping with
  a content-addressed member list (per-shard sha256 + model_version).

### B2. Split-set validation (count vs file list, split.no, tensor count) -- ADAPT
- `split.count` vs discovered file count, entry file must self-declare
  `split.no==0`, each shard's `split.no` must equal its loop position, and a
  global `split.tensors.count` reconciles against the union assembled.
  `:589-610`, `:629-639`, `:657-665`; KV keys at `src/llama-arch.cpp:285-287`.
- GAP: completeness is proven by COUNTING only. A shard truncated mid-tensor-data
  but with the right tensor-count and split.no passes every check here. A missing
  shard is a fatal throw (`:663`), not a resumable state. VERIFIED_ON_DISK must
  gate on per-shard sha256 matching the manifest, and a missing member must be a
  RECEIVING-retry, not a throw.

### B3. Tensor -> (file idx, absolute offset) with bounds check -- ADAPT
- The per-tensor locator (A1) doubles as the cross-shard assembly key: every
  tensor keyed by NAME into one flat `weights_map` with its owning file idx;
  absolute `blk.N` names (preserved by the extractor) are what make cross-shard
  assembly work; name collision across shards is rejected. `:645-654`, `:580-587`.
- GAP: identity = (name, idx, offset, nbytes), bound to no content digest.

### B4. `shard_gguf.py` offline extractor -- ADAPT
- Half-open `[start,end)` selection keyed on the tensor's OWN absolute block index;
  `blk.17` stays `blk.17` (no renumbering); terminal-only tensors gated to the last
  stage; global tables copied to every shard. `research_dev/shard_gguf.py:30-39`.
- Strict range validation `0 <= start < end <= n_layer` from `<arch>.block_count`
  -- the only validation it performs `:60-61`. `block_count` and per-layer arrays
  (SWA pattern, rope) copied UNMODIFIED so on-device per-layer indexing matches the
  mono model `:65-73`. Absolute-layer-index preservation is GUARANTEED.
- GAP: the shard's actual `[start,end)` coverage is recorded NOWHERE in the output
  file (block_count stays full; no start/end KV emitted). Coverage travels only
  out-of-band via env `LLAMA_LAYER_START/END`, so the file is not self-describing.
  A residency ModelManifest must add an in-manifest coverage record + model_version.

### B5. Extractor/loader absence of identity, durability, atomic publish -- REJECT
- The extractor emits NO per-shard sha256, NO manifest, NO fsync; writes directly
  to an arbitrary output name via `GGUFWriter.close()` and writes NO `split.*` KV,
  so its outputs are standalone single-file ggufs INVISIBLE to the loader's split
  validation (that whole path B1-B2 is never exercised for this project's shards).
  `research_dev/shard_gguf.py:1-98`, `:42-49`; loader treats it as a lone file at
  `src/llama-model-loader.cpp:589-590`.
- GAP: for a residency agent the extractor output is neither an identity nor a
  durability source. The ModelManifest must add, on top: (1) per-shard sha256
  feeding VERIFYING; (2) an explicit atomic-publish member list per WeightSet
  generation; (3) transfer-chunk metadata for RECEIVING; (4) temp+fsync+rename
  durability; (5) an in-manifest layer-coverage + model_version record replacing
  the env convention.

---

## C. Downloader: resume, verification, fsync, atomic publication

Files: `common/download.cpp`, `common/hf-cache.cpp` (+ their headers).
Verified by hand: `grep fsync|fdatasync|fflush` over both files = NO MATCHES;
`std::rename` at `download.cpp:408`, `:66`, `hf-cache.cpp:190`, `:487`.

RECEIVING exists (HTTP range-resume + staging + atomic-rename publish). VERIFYING
and the durable VERIFIED_ON_DISK barrier do NOT exist and must be added wholesale.

### C1. HTTP range-resume (Range header, 206 gate, Accept-Ranges gate) -- ADAPT
- Resume is offset-only: append to the staging file and request `bytes=N-` from the
  current on-disk size; the already-written prefix is NEVER re-validated
  `download.cpp:216-225`. Correct 206-on-resume / 200-on-fresh gates `:232-239`.
  Accept-Ranges derived from HEAD; if absent, the staging file is deleted and
  re-downloaded whole `:346-349`, `:390-401`.
- GAP: (1) no resumable VERIFIED ranges -- a truncated/corrupt/wrong-content prefix
  from a prior run is trusted by length alone and appended to. (2) no fallback for
  the "Accept-Ranges advertised but 206 refused" case: all 3 attempts re-issue the
  same failing Range request and never truncate-and-full-GET (deadlock). (3) no
  If-Range/ETag validator, so a remote object that changed between runs can be
  spliced onto an old prefix undetected.

### C2. Atomic publication via std::rename from a staging file -- ADAPT
- `.downloadInProgress` staging name `:373`; publish `std::rename` `:407-414`. The
  rename is atomic-VISIBILITY on POSIX but is NOT preceded by any fsync, so it can
  make a durable name point at data still in the page cache -- a crash yields a
  full-named truncated/zero file. The payload `std::ofstream` is closed only by its
  destructor and its flush/close error is never checked before rename. Best-hardened
  writer `hf-cache.cpp:176-196` checks stream fail-state but still never fsyncs.
- GAP: add the durability half -- `fflush + fsync/fdatasync(file fd)` + checked
  close BEFORE rename, then `fsync(parent dir fd)` AFTER rename, ordered AFTER the
  content-hash gate so only verified bytes become durable.

### C3. HF cache layout (blobs by oid + snapshots + symlink) -- REFERENCE_ONLY
- The blob filename IS the LFS oid (a sha256 for LFS files; `is_valid_oid` accepts
  64-hex) -- the content digest is already known and materialized as the storage
  key, yet the downloaded bytes are NEVER hashed to confirm they match it.
  `hf-cache.cpp:304-357`, `:162`. Publishes into the snapshot tree as a relative
  symlink with a degraded rename-then-copy fallback, no fsync on any branch
  `:455-496`. Path-traversal guard on server-supplied paths is worth keeping
  `:165-174`.
- GAP: content-addressing is NOMINAL, not enforced. Study blob/snapshot separation;
  do not adopt the symlink indirection or unchecked rename/copy fallback.

### C4. Retry/backoff + ETag conditional caching -- ADAPT
- Hard-coded 3 attempts, exponential backoff 2s/4s, no jitter, no error-class
  distinction (a 404/401 is retried 3x) `:290-291`, `:380-388`. Freshness via a
  separate HEAD + ETag sidecar (not a conditional GET; no If-None-Match anywhere);
  on ETag mismatch it DELETES the published file before re-downloading `:321-365`.
- GAP: error-class-aware retry (do not retry 401/403/404), jitter across many
  phones, and NEVER delete a VERIFIED_ON_DISK image before its replacement reaches
  VERIFIED_ON_DISK (the delete-before-redownload at `:361` is a data-loss hazard).
  ETag is a freshness hint, not integrity -- bind versioning to the verified digest.

### C5. Content SHA-256 / manifest-digest verification of payload -- REJECT (absent)
- The ONLY sha256 code is the Docker path, which validates the FORMAT of the digest
  string and builds a blob URL from it -- it NEVER hashes the downloaded bytes
  `:826-895`. `grep sha256|verify|checksum|digest` over `hf-cache.cpp` = NO MATCHES.
  No length-vs-Content-Length assertion at completion (Content-Length only sizes the
  progress bar). A truncated/corrupt/MITM payload is renamed to its final name and
  used with no detection.
- GAP: this IS the VERIFYING state. Build from scratch: hash-on-write during
  RECEIVING, compare to the ticket digest at VERIFYING, refuse publication on
  mismatch (-> ERROR/QUARANTINED). Highest-priority addition in the subsystem.

### C6. Durability barriers + transactional rollback -- REJECT (absent)
- No fsync of file or parent dir around any rename; per-chunk write errors are
  checked but the stream is never flushed to disk `:250-256`, `:407-411`;
  `finalize_file` also never fsyncs `hf-cache.cpp:487-495`. Once rename succeeds
  there is no post-publish re-verify, no rollback, no quarantine directory `:407-417`.
- GAP: `any state -> ERROR/QUARANTINED` edges and the atomic swap between residency
  generations do not exist. Needs stage-verify-swap that retains the prior good
  image until the new one is VERIFIED_ON_DISK, plus a quarantine sink for
  VERIFYING failures.

---

## D. ggml-rpc tensor hash / content-cache protocol

Files: `ggml/src/ggml-rpc/ggml-rpc.cpp`, `transport.{h,cpp}`, `ggml-rpc.h`.
Verified by hand: the identity hash is FNV-1a 64-bit, NOT SHA-256
(`ggml-rpc.cpp:232-242`, basis `0xcbf29ce484222325`, prime `0x100000001b3`);
`HASH_THRESHOLD = 10 MiB` (`:80`); no crypto anywhere in the subsystem.

This is the clearest in-tree precedent for content-addressed weight dedup -- and
the clearest lesson in why it is not enough for a VERIFYING gate.

### D1. SET_TENSOR_HASH content-addressed dedup + server disk cache -- REFERENCE_ONLY
- Hashing kicks in only above 10 MiB `:79-80`; client computes the hash, asks the
  server, and on a hit the bulk bytes are never transmitted `:465-489`. Server
  writes the cache file (name = 16 hex chars of the 64-bit FNV hash, flat in
  cache_dir, no sidecar) `:1092-1101`; lookup is an existence test `:1111-1129`; on
  a hit it materializes cached bytes and reports success WITHOUT re-hashing or
  byte-comparing `:1131-1174`.
- GAP: (1) a 64-bit non-crypto digest with realistic birthday collisions over a
  fleet-scale corpus, and NO re-verification on a hit -> a collision silently loads
  the WRONG weights. (2) binds only raw payload bytes -- none of the residency
  identity fields. (3) no eviction (grep evict|unlink|lru -> none) -> unbounded disk
  growth. (4) no durability (plain ofstream, no fsync, no temp+rename). Reuse the
  send-hash-skip-bytes PATTERN for RECEIVING dedup; re-implement identity on SHA-256
  over the full binding.

### D2. Hash function primitive (fnv_hash) -- REJECT
- FNV-1a 64-bit, `:232-242`. Fast, adequate for opportunistic same-session dedup;
  unsuitable as an integrity boundary (not collision-resistant, forgeable, 8 bytes).
- GAP: S9 mandates SHA-256 content identity. Keep FNV only as a cheap non-crypto
  prefilter; the authoritative compare must be the full SHA-256 plus the binding
  tuple. Straight REJECT for the identity role.

### D3. HELLO version handshake + conn_caps -- ADAPT
- Explicit major.minor.patch with a compile-time assert tying op-count ABI to the
  patch version (`ggml-rpc.h:9-14`); client rule major must ==, minor must <=
  (`:330-346`); server enforces HELLO-first + exact request size (`:1455-1486`);
  `conn_caps` is a fixed 24-byte opaque blob for RDMA negotiation `:82-92`.
- GAP: to reach REUSE the handshake must also exchange and pin the residency
  identity context (backend build hash, SoC, arch, layout version, boot epoch) so a
  materialized image can be invalidated when the server incarnation changes; must
  surface a recoverable ERROR instead of `GGML_ABORT` (`:337`); caps must be
  typed/length-prefixed.

### D4. Wire framing (cmd byte + 8-byte host-endian length + packed structs) -- REFERENCE_ONLY
- Frame = 1 cmd byte + 8-byte length + payload; no magic, no per-frame version, no
  frame checksum, no request id `:290-304`. All wire structs `#pragma pack(1)`,
  memcpy'd host-endian, no byte-swap `:32-53`. Variable-length recv trusts the
  peer's 8-byte size with NO upper cap (only bad_alloc guard) -> allocation-DoS
  `:262-274`. Bulk chunks at 1 GiB with no send/recv timeout `transport.cpp:462-503`.
  Server-side tensor-region bounds checks are a good precedent `:1080-1089`.
- GAP: reuse the SHAPE (typed length-prefixed frames, chunked bulk), not the code.
  A residency transport needs bounded, length+checksum-framed, explicitly
  little-endian control and bulk channels, plus end-to-end SHA-256 of the
  transferred image.

### D5. Command enum + serial single-in-flight loop -- REFERENCE_ONLY
- 18 commands, HELLO pinned at ordinal 14 by static_assert; one flat opcode
  namespace shared by control and bulk; strictly serial request/response per
  connection `:56-77`, `:1487-1500`.
- GAP: no request ids -> no multiplexing, no async residency notification
  correlation; a large MATERIALIZING transfer head-of-line-blocks control/heartbeat.
  Study the opcode taxonomy; do not inherit the id-less single-in-flight loop.

### D6. Robustness/liveness absences (timeout, heartbeat, reconnect, idempotency) -- REJECT
- Client policy on ANY malformed response is process abort (`RPC_STATUS_ASSERT ->
  GGML_ABORT`, `:30`, used 15+ times); compute failure is a hard assert `:1408-1424`.
  `grep timeout|heartbeat|reconnect|request_id|retry|idempoten|SO_RCVTIMEO` -> NO
  matches; blocking recv/send with no `SO_RCVTIMEO` (only `SO_REUSEADDR`,
  `transport.cpp:584`) -> a silently dead peer hangs the agent forever.
- GAP: disqualifying for a persistent controller. Net-new: bounded send/recv
  timeouts + heartbeat; reconnect with resumable idempotent transfers
  (content-addressed chunk offsets so a retried chunk is a no-op); every
  any-state->ERROR/QUARANTINED edge must be a returned status, never abort;
  request-id multiplexing so control survives a long transfer.

---

## E. Per-tensor HTP <-> OpenCL one-physical-copy weight sharing

Files: `ggml/src/ggml-hexagon/ggml-hexagon.cpp`,
`ggml/src/ggml-opencl/ggml-opencl.cpp`. EXISTS in the dirty worktree, env-gated
`GGML_PHONE_SHARE_PUBLISH` (publish) / `GGML_PHONE_SHARE_IMPORT` (import), default
OFF. Verified by hand: name-keyed `g_hex_shared_tensors` at `ggml-hexagon.cpp:1017`,
first-insert-wins emplace `:1024`; the `.clear()` calls at `:1188-1190` are
UNRELATED maps (b_map/t_map/d_map).

Bit-correct for a single sequential dualengine load; NO content identity and NO
teardown. This is the raw PREPARING->READY aliasing primitive and supplies none of
the PreparedImage identity or the DRAINING/EVICTING teardown the contract needs.

### E1. Hexagon publish registry (name-keyed, first-insert-wins) -- REFERENCE_ONLY
- Registry is `unordered_map<string, entry>` keyed by the BARE tensor name; the
  entry carries only `{fd, base, import_size, offset, size}` -- no model_version,
  digest, generation, SoC, or epoch `:1016-1017`. `emplace` = first-insert-wins;
  on a second model load the pre-existing name is never overwritten (comment admits
  the reliance) `:1019-1025`. Publish trigger: native-linear F16/F32 `.weight`
  tensors only, selected by name substring + type, `off = t->data - sbuf->base`
  asserted 128B aligned `:376-389`.
- Verified absence: `grep sha256|digest|model_version|generation|boot.?epoch|
  graph.?hash|layout.?version|soc_id|checksum|verify` over the file -> ZERO matches.
  Identity == name alone.
- GAP: bind the full PreparedImage identity tuple and make the resolver reject a
  name whose bound identity does not match the caller's expected
  model_version + generation.

### E2. Hexagon registries have no unpublish/erase/teardown -- REJECT
- Both registries are process-global statics; the only mutations are `emplace`
  (`:1024`) and `push_back` (`:1044`) -- no erase, no clear, no unpublish
  `:1008-1017`. The shared-buffer destructor frees the rpcmem and closes the fd but
  does NOT remove the entry, so after free the registry holds a DANGLING fd/base ->
  use-after-free on the OpenCL importer `:318-328`.
- GAP: DRAINING->EVICTING has no code to build on. Needs `unpublish(name)` under the
  mutex, gated on residency generation, plus invalidation of already-issued aliases
  and generation-scoped keys so a freed-then-reallocated tensor cannot resolve the
  dead fd.

### E3. Hexagon extern-C name resolver (no identity args) -- ADAPT
- Cross-backend surface: lookup takes ONLY a name, returns fd+base+size+offset;
  mutex-guarded (correct) but semantically identity-blind `:1029-1040`. The older M4
  whole-buffer "take" path claims by size-match in publish order -- even weaker
  `:1051-1063`.
- GAP: signature must grow an identity token (caller passes expected
  {model_version, generation, layout_version}; resolver returns false on mismatch)
  and a generation-scoped key. Shape is right; hardening required.

### E4. OpenCL fd->cl_mem import (process-global, intentionally leaked) -- ADAPT
- QCOM ext-host-ptr import, memoized per integer fd; header comment states the
  handle "intentionally outlives every buffer that aliases it and is leaked at
  process exit" `ggml-opencl.cpp:6346-6373`. Both registries are process-global
  statics with NO erase; destructor releases only `ctx->buffer`, so every imported
  alias cl_mem AND its ion fd leak `:6326-6328`, `:5951-5953`. Resolver bound once
  via dlsym `:6333-6344`.
- GAP: an integer fd is a poor global key -- after the publisher frees its rpcmem,
  the same fd is recycled and the import returns a cl_mem wrapping a DIFFERENT
  buffer (stale alias). Leak-on-purpose is fine for a one-shot process; a persistent
  agent needs generation-scoped keys + `clReleaseMemObject` on EVICTING.

### E5. OpenCL alias placement by name + clFinish-only coherency -- REFERENCE_ONLY
- HIT path aliases Hexagon bytes purely by name match, reusing the Hexagon-side
  offset `hoff`, trusting both backends laid the tensor at the same 128B offset; the
  returned `tsz` is read but never compared to `ggml_nbytes(tensor)` `:6405-6417`.
  MISS path promotes a 1-byte dummy to a real buffer once `:6418-6437`. Coherency
  rests on a SEQUENCING ASSUMPTION (decode model loaded before prefill imports), not
  a barrier; a single `clFinish` does not order the DSP's writes `:6366-6368`. The
  `set_tensor` write-skip guard downgrades an offset bug to a benign mislocated READ
  `:7830-7841`.
- GAP: a name collision across models silently aliases wrong-sized/wrong-model bytes
  (fail-soft to a wrong read, not a crash). Coherency must become an explicit
  cross-engine barrier tied to MATERIALIZING->LPDDR_READY, and the write-skip needs a
  companion positive identity assertion.

---

## F. OpenCL "xmem" prepared-weight cache (Adreno prepacked GEMM)

File: `ggml/src/ggml-opencl/ggml-opencl.cpp`, kernel
`kernels/gemm_xmem_f16_f32_os8.cl`. Verified by hand: `s_xmem_weight_cache` keyed
by `(cl_mem, cl_ulong)` at `:13299`, env `GGML_OPENCL_XMEM_PREPACK_CACHE` at
`:13300`, "retained for reuse (leaks at exit; fine for benchmark)" at `:13335`.

This is the concrete PREPARING_GPU -> WARMING -> READY_GPU derived-image producer.
Adoption verdict REFERENCE_ONLY: the derived-image CONCEPT and the RAM-cost model
are directly instructive; every concrete artifact is disqualifying.

### F1. Two-layer env gate + dimension eligibility -- ADAPT
- Primary gate `GGML_OPENCL_ADRENO_XMEM_GEMM` read once at device init, inside
  `#ifdef GGML_OPENCL_USE_ADRENO_KERNELS`, presence-only, Adreno family only,
  default off `:4668-4672`. Runtime eligibility: f16/bf16 weight, f32 act+dst,
  contiguous, 2D only, K%8==0, `N>=16` (the "batch>=16" gate; N = src1 token count)
  `:13208-13237`. Sole dispatch inside the F16 mul_mat case `:15673-15679`.
- GAP: presence-only env, read once, no per-model/per-generation toggle; must become
  an explicit per-tensor policy bound to the residency state machine.

### F2. Prepared-weight transform + process-global prepack cache -- REJECT
- The cache is a function-local static `map<pair<cl_mem,cl_ulong>, cl_mem>` keyed ONLY
  by (device pointer, byte offset) -- no content digest, no K/M, no os, no layout, no
  SoC, no epoch `:13297-13302`. Transform reads the canonical linear f16 weight and
  writes a NEW transposed/os-packed device buffer `:13313-13336`. Two lifetimes:
  cache OFF = re-prepack + free EVERY call (pure overhead); cache ON = retained
  forever, no eviction/refcount/capacity/invalidation, deliberately leaked
  `:13382-13383`, `:13335`.
- GAP: identity is a raw pointer that a freed-then-reallocated cl_mem reuses ->
  silent wrong-weight collision; unbounded RAM; deliberate leak precludes clean
  hand-back. This is the concrete anti-pattern the S9 derived-image identity exists
  to replace.

### F3. Backend-derived RAM cost (prepacked buffer separate from canonical) -- REFERENCE_ONLY
- Prepacked weight size `~= (K/4)*ceil(M/4)*4*8 ~= 2*K*M` bytes -- roughly one EXTRA
  full copy of the f16 weight (canonical is `K*M*2`), resident on the GPU IN ADDITION
  to the canonical linear WeightSet `:13269-13274`. Plus per-call transient src/dst
  image2d (freed each call, scales with batch N) `:13283-13295` and a fixed 6144-byte
  `__constant` staging buffer `:13273-13276`.
- GAP: the code never accounts for derived RAM separately. The value is the FORMULA
  (~1x extra per prepacked tensor) for the S9 RAM ledger: a phone holding both
  HTP-linear and GPU-prepacked forms pays ~double weight RAM, unrecoverable by the
  linear one-copy sharing.

### F4. f16 (half4) accumulation -> ~1.86% rel_L2 (undocumented) -- REFERENCE_ONLY
- The os8 GEMM accumulators `r0..r7` are half4 (f16); the stock l4_lm path
  accumulates in f32 `kernels/gemm_xmem_f16_f32_os8.cl:108-115`, `:134-137`. f16
  accumulation over K is the mechanism of the measured ~1.86% rel_L2 error. There is
  NO comment/tolerance/assert anywhere flagging this -- source-only review would miss
  it; it is a MEASURED (S6) caveat, not represented in code.
- GAP: a GPU-prepared image is NOT bit-equivalent to the canonical weight, so
  residency verification of the derived image must be a TOLERANCE check bound to
  backend build + layout version, not a SHA-256 equality against canonical bytes.

### F5. Per-(SoC, layout, os) nature -- why the GPU image cannot be linearly shared with HTP -- REFERENCE_ONLY
- Prepack is parameterized by kpack/npack tiling + os=8, laid out for the os8 GEMM's
  6144-byte `__constant` sub-group-constant-load window; hard-gated to Adreno
  `:13211-13213`, `:13316-13332`; the kernel depends on Qualcomm Adreno sub-group
  extensions `gemm_xmem_f16_f32_os8.cl:1-3`, `:130`.
- GAP: PREPARING_GPU is a MANDATORY per-(SoC, layout, os) step. Unlike the
  F16-native linear weight (shareable HTP<->GPU one-copy), this transposed image is a
  distinct derived artifact that must be re-run on the target GPU and re-verified per
  residency generation -- the derived-image RAM cost is unavoidable per-backend.

---

## G. Partial model loading + arbitrary-model limitations

Files: `src/models/gemma4.cpp`, `src/llama-model.cpp`, `src/llama-model-loader.cpp`.
Verified by hand: `LLAMA_LAYER_START/END` appear ONLY in `gemma4.cpp:53-54` (load),
`:213-214` (graph), and `llama-model.cpp:1480` (presence-only tolerance flag) --
NO other `src/models/*.cpp` references them.

Layer-window partial load is a HARD, MODEL-SPECIFIC eligibility gate.

### G1. Layer-window tensor-creation skip in gemma4 -- REFERENCE_ONLY
- Stage window taken from unvalidated env via `atoi` (returns 0 on garbage; no
  error, no manifest binding); `ls`/`le` clamped to `[0,n_layer]` but NOT
  `ls<=le` `src/models/gemma4.cpp:52-58`. The core skip: out-of-range layers never
  reach any `create_tensor`, so no buffer is allocated `:86-92`. Head/tail tensors
  (output, output_norm) created only on the terminal stage `:60-66`.
- GAP: the slice boundary is a process-global env read inside model construction,
  unvalidated, bound to no content/graph/backend identity. Porting into a ModelManifest
  requires: replace env with a validated `[ls,le)` descriptor; record which tensor
  NAMES a stage owns so VERIFYING can checksum exactly that set; assert `ls<=le`.

### G2. Loader partial-load tolerance (done_getting_tensors partial flag) -- ADAPT
- In the SHARED `load_tensors`, the tolerance is armed by env PRESENCE (not value):
  `getenv("LLAMA_LAYER_START") != nullptr || getenv("LLAMA_LAYER_END") != nullptr`
  `src/llama-model.cpp:1477-1481`. The exact relaxation: without `partial`,
  `n_created < n_tensors` is a hard throw; with it, an INFO log
  `src/llama-model-loader.cpp:1320-1330`.
- GAP: (1) armed by env presence, so a stray `LLAMA_LAYER_END` silently disables the
  completeness check for ANY arch -- a corrupt full gguf missing tensors would then
  load with only an INFO line. (2) it only bounds `n_created <= n_tensors`; it does
  NOT verify the present tensors are the expected slice set (any subset of the right
  count passes). Replace "count is short AND env set" with "present set == manifest
  slice set AND each digest matches".

### G3. Model-specificity of layer windowing (the arbitrary-model REJECT boundary) -- REJECT
- Both halves of the split are gemma4-only: the load skip (`gemma4.cpp:86-92`) AND
  the graph (injects a relayed activation for `ls>0`, emits the cut hidden for
  `le<n_layer`, `:208-217`). Every other model's create_tensor loop has no such
  `continue`, so it always creates all `n_layer` layers; the partial flag is a no-op
  for them and the WHOLE model stays resident on every stage.
- GAP: partial/sliced residency is IMPOSSIBLE for any architecture other than
  gemma4 -- there is no generic layer-window hook in the base loader or graph builder.
  A ModelManifest may declare a layer window ONLY for gemma4-family models; for all
  others the sharding descriptor is unsatisfiable and the model is REJECT (resident
  whole or not eligible). This is encoded in `model_manifest.partial_load_supported`.

### G4. E2B per-layer-embedding + shared-KV constraints on legal cuts -- REFERENCE_ONLY
- E2B (`n_embd_per_layer>0`) forces `tok_embd` and the three `per_layer_*` tensors
  onto EVERY stage; `per_layer_tok_embd` is FULL-DEPTH
  (`n_embd_per_layer*n_layer x n_vocab`) and cannot be sliced to `[ls,le)` -- a
  per-layer stage carries global-depth embedding weight regardless of its window
  `gemma4.cpp:68-78`. A non-head E2B stage must be fed a DUAL batch `:233-245`.
  `has_kv(i)==false` layers omit their own K/V (shared-KV), so a stage boundary
  between a KV producer and consumer is illegal `:95-108`.
- GAP: even within the eligible arch the cut set is NOT arbitrary. The ModelManifest
  must encode a per-model LEGAL CUT SET derived from `has_kv` adjacency and
  `n_embd_per_layer`, and E2B stage footprints must include the full-depth
  `per_layer_*` tensors.

### G5. Missing-tensor + shape check (create_tensor / check_tensor_dims) -- ADAPT
- The shared `create_tensor`: a TENSOR_NOT_REQUIRED tensor absent returns NULL
  silently; a required absent tensor throws (name only) `:1272-1276`, `:870-874`;
  per-tensor shape verification catches gross layout mismatch `:877-891`. No sha256/
  fsync/digest in this file.
- GAP: the required/not-required + shape machinery is the right seam to extend, but
  it verifies NAME + SHAPE only. Insert per-tensor SHA-256 + derived-image identity
  binding here before promoting a slice to VERIFIED_ON_DISK and again before
  MATERIALIZING.

---

## H. Classification summary

| Mechanism | File(s) | Class |
|---|---|---|
| llama_tensor_weight locator (offset+split+bounds) | llama-model-loader.h:33-50 | REUSE |
| byte-count accounting (size_data/size_done/n_created) | llama-model-loader.cpp:651-1673 | REUSE |
| load_data_for (mmap-alias vs seek+read) | llama-model-loader.cpp:1391-1408 | ADAPT (hash-on-read) |
| load_all_data materialization loop | llama-model-loader.cpp:1541-1670 | ADAPT (make transactional) |
| mmap lifecycle / fragment unmap / mlock | llama-model-loader.cpp:1338-1683 | REFERENCE_ONLY |
| tensor-payload content integrity | llama-model-loader.cpp (absent) | REJECT (build VERIFYING) |
| split discovery + naming | llama-model-loader.cpp:76-101; llama.cpp:505-549 | ADAPT |
| split-set validation (count/split.no) | llama-model-loader.cpp:589-665 | ADAPT |
| shard_gguf.py extractor ([start,end), abs-index) | shard_gguf.py:30-93 | ADAPT |
| extractor/loader identity+durability | shard_gguf.py; loader (absent) | REJECT (manifest layer) |
| HTTP range-resume | download.cpp:216-401 | ADAPT (verified ranges) |
| atomic rename publish | download.cpp:407-414 | ADAPT (add fsync) |
| HF blob/snapshot/symlink cache | hf-cache.cpp:304-496 | REFERENCE_ONLY |
| retry/backoff + ETag conditional | download.cpp:290-388 | ADAPT |
| payload SHA-256 verification | download.cpp; hf-cache.cpp (absent) | REJECT (build VERIFYING) |
| fsync/dir-fsync durability + rollback | download.cpp; hf-cache.cpp (absent) | REJECT |
| SET_TENSOR_HASH dedup + disk cache | ggml-rpc.cpp:465-1174 | REFERENCE_ONLY |
| fnv_hash identity primitive | ggml-rpc.cpp:232-242 | REJECT (use SHA-256) |
| HELLO version handshake + caps | ggml-rpc.cpp:330-1486 | ADAPT |
| wire framing (cmd+len+packed structs) | ggml-rpc.cpp:290-304; transport.cpp | REFERENCE_ONLY |
| command enum + serial loop | ggml-rpc.cpp:56-1500 | REFERENCE_ONLY |
| timeout/heartbeat/reconnect/idempotency | ggml-rpc.cpp (absent) | REJECT |
| Hexagon publish registry (name-keyed) | ggml-hexagon.cpp:1016-1040 | REFERENCE_ONLY |
| Hexagon registry teardown | ggml-hexagon.cpp (absent) | REJECT |
| Hexagon extern-C resolver | ggml-hexagon.cpp:1029-1063 | ADAPT (add identity token) |
| OpenCL fd->cl_mem import (leaked) | ggml-opencl.cpp:6326-6373 | ADAPT (identity key + reclaim) |
| OpenCL alias-by-name + clFinish coherency | ggml-opencl.cpp:6405-7841 | REFERENCE_ONLY |
| xmem env gate + dim eligibility | ggml-opencl.cpp:4668-15679 | ADAPT |
| xmem prepack cache (pointer-keyed, leaked) | ggml-opencl.cpp:13297-13383 | REJECT |
| xmem derived-RAM cost model | ggml-opencl.cpp:13269-13295 | REFERENCE_ONLY |
| xmem f16-accum accuracy (~1.86% rel_L2) | gemm_xmem_f16_f32_os8.cl:108-137 | REFERENCE_ONLY |
| gemma4 layer-window skip | gemma4.cpp:52-92 | REFERENCE_ONLY |
| loader partial-load tolerance | llama-model.cpp:1477-1481 | ADAPT |
| arbitrary-model layer windowing | src/models/* (gemma4-only) | REJECT (eligibility gate) |
| E2B per-layer-embd / shared-KV cuts | gemma4.cpp:68-108,233-245 | REFERENCE_ONLY |
| create_tensor + check_tensor_dims | llama-model-loader.cpp:870-891,1272-1276 | ADAPT |

---

## I. Net-new components implied (nothing ported in S9-V0)

1. A VERIFYING layer: per-segment SHA-256 of the exact `[offs,offs+nbytes)` span,
   compared to a signed ModelManifest, with the `check_tensors` default inverted to
   always-verify. Absent from A6, C5, D2, G5.
2. A durable VERIFIED_ON_DISK barrier: `fflush + fsync(file) -> checked close ->
   rename -> fsync(parent dir)`, ordered AFTER the hash gate, plus stage-verify-swap
   and a QUARANTINE sink. Absent from C2, C6, D1.
3. A resumable VERIFIED transfer: TransferTicket carrying expected digest + per-chunk
   partial hashes so RECEIVING resumes without trusting a byte-length offset, and a
   truncate-and-full-GET fallback for the 206-refused deadlock. Extends C1.
4. Content identity on SHA-256 (not FNV-1a), and a derived-image identity binding
   model_version + tensor digest + graph hash + backend build + SoC + arch + layout
   version + boot epoch + residency generation. Absent everywhere (D2, E1, F2).
5. A generation-qualified, teardown-capable share/prepack registry: identity-checked
   resolvers, `clReleaseMemObject` / rpcmem-unpublish on EVICTING, and a cross-engine
   coherence barrier tied to MATERIALIZING->LPDDR_READY. Replaces E2, E4, F2.
6. Separate physical-byte accounting for canonical bytes vs backend-derived images
   (HTP linear shareable; GPU xmem NOT), so a phone's LPDDR ledger is exact. Formula
   from F3; no code does this today.
7. Bounded, versioned, checksummed, little-endian control + bulk channels with
   request-id multiplexing, heartbeat, reconnect, idempotency, and activation-over-
   bulk priority. Replaces D4-D6 (see TRANSPORT_CONTRACT.md).
8. A per-model LEGAL CUT SET (gemma4-only eligibility; E2B/`has_kv` constraints) as a
   hard ModelManifest gate. From G3-G4.

None of these are built in S9-V0. This audit classifies what exists and marks the
boundary between reusable byte plumbing and the net-new identity/verification/
lifetime plane that WEIGHT_RESIDENCY_CONTRACT.md, TRANSPORT_CONTRACT.md, and the
simulator specify.
