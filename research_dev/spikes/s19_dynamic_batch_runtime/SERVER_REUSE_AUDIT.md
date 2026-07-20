# S19 llama-server continuous-batching reuse audit

Status: design audit complete. No S19 runtime claim yet.

## Question

Which parts of `llama-server` continuous batching should become the request
lifecycle substrate for the distributed LayerSplit route?

The answer is to reuse its semantics and common llama.cpp batch/memory APIs,
not to copy `server_context` or expose intermediate activations through the
server API.

## What llama-server already proves

`llama-server` implements continuous batching with five load-bearing ideas:

1. A `server_slot` owns one llama sequence and moves through explicit prompt,
   decode, and idle states (`tools/server/server-context.cpp:58-65,161`).
2. A `server_batch` first collects logical rows and then renders one
   `llama_batch`. Each row carries a slot/sequence ID, token, position, and
   output flag (`tools/server/server-context.cpp:69-158`).
3. Every update begins with currently generating slots, then admits pending
   prompt rows into remaining batch capacity when continuous batching is
   enabled (`tools/server/server-context.cpp:2926-3059,3062-3084`).
4. Compatibility is checked before co-batching. The current server key includes
   task type and LoRA state (`tools/server/server-context.cpp:385-388`).
5. Completion and cancellation release one slot without clearing every active
   sequence. Prompt clearing, context shifts, and rollback target an explicit
   sequence with `common_context_seq_rm`; the cache policy may intentionally
   retain an idle slot's prompt state (`tools/server/server-context.cpp:252-267,
   483-502,2376-2457`).

The queue owns admission and deferral. After queued tasks are assigned to
slots, one `update_slots()` call builds and executes the shared batch
(`tools/server/server-queue.cpp:125-168`). Continuous batching is enabled by
default through `common_params::cont_batching` (`common/common.h:565`).

## What the current LayerSplit path has

The current StageNet protocol already accepts a variable-row decode batch with
explicit sequence IDs, positions, and tokens (`STAGE_BATCH_DECODE`). The host
tail also tags each activation row with the same sequence ID before
`llama_decode` (`examples/layersplit/layersplit.cpp:2763-2794,2901-2913`).

This is sufficient for a changing active subset, but not for continuous
admission:

- `run_pipebatchdriver` clears the complete tail and both phone stages at the
  start of every fixed group (`layersplit.cpp:2796-2805`).
- `STAGE_BATCH_PREFILL` assigns sequence IDs `0..B-1` and positions `0..N-1`;
  it cannot prefill an arbitrary free slot while other slots remain live.
- the persistent command rejects any request count other than the configured
  batch (`layersplit.cpp:3091-3093`);
- EOG only shrinks the current cohort. No pending request replaces the freed
  sequence before the cohort ends.

Therefore model and process persistence pass, but request/KV persistence is
still cohort-scoped.

## Reuse, adapt, reject

### Reuse

- explicit per-request slot states;
- one logical batch manifest rendered from `(seq_id, token, position, output)`;
- admission on every token-boundary update;
- compatibility filtering before batch construction;
- bounded slot credits and deferred admission;
- per-sequence KV removal on finish/cancel;
- prompt chunking into remaining physical batch capacity;
- one owner for terminal completion and cancellation.

### Adapt for Q-PIM

The distributed compatibility key must additionally bind:

~~~text
model and graph digest
route and layer cut
phone/backend/build and residency generation
activation dtype, shape, and layout
attention/context class
sampling contract
request, route, lease, and device-boot epochs
~~~

One logical request needs a global request ID plus a stable sequence mapping in
every state owner. A phone-local sequence number is not a global identity.

The same immutable batch manifest must be validated by the phone head and the
CUDA tail. An upstream result is accepted only for the exact request, sequence,
position, route, and epoch reserved for that update.

The server FIFO/defer policy is not reused as the Q-PIM policy. S19 retains the
existing priority, deadline, measured-batch-candidate, downstream-credit, and
power-frontier decisions. `llama-server` supplies lifecycle mechanics, not the
research scheduling objective.

### Reject

- copying the HTTP/router/JSON response stack into LayerSplit;
- copying all of `server_context`, which is coupled to sampling, LoRA,
  multimodal, checkpoint, and speculative-decoding behavior;
- exposing boundary activations through the public llama-server API;
- using `eval_callback` as a multi-sequence transport seam;
- changing a live request's route, cut, or KV owner;
- treating a fixed B32 cohort as continuous batching.

The public server documentation explicitly places intermediate-activation APIs
out of scope because the callback is not a stable multi-sequence interface
(`tools/server/README-dev.md:26-31`). Q-PIM therefore remains an experimental
executor beside the server, with native typed boundaries inside LayerSplit.

## Smallest safe implementation boundary

S19 should add a compact LayerSplit-local slot and batch lifecycle, using the
same llama.cpp primitives as `llama-server`. It should not edit
`tools/server/server-context.cpp`, `llama-graph.cpp`, or
`ggml_backend_sched`.

The first physical proof is one persistent phone head and one persistent CUDA
tail. Requests have staggered arrivals and unequal generation lengths, so a
new request must enter a freed sequence while another sequence continues
decoding. Only after that passes does the same manifest coordinator expand to
both phones.
