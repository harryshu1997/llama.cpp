# Tuned Pixel CPU versus server overlap budget

2026-09-23 18:54 UTC. Source/arithmetic review **PASS**. New CPU server measurement
**NOT RUN**: the shared rig lock is occupied by OP15 sweeps. No process was
interrupted and no server job was queued. No new energy or token-equality claim.

The server already overlaps a phone call with the desktop's remaining FFN
columns. The TCP worker's backend is transparent to the client: the newly
qualified CPU worker can use this same protocol and overlap mechanism. It
still needs qualification under actual server cadence. No new server C++
overlap machinery is required for this substitution.

## Gap using existing measurements

All values are mean milliseconds per selected FFN layer, one token. Historical
server values come from PIXEL_GEMV_SERVER.json; CPU times come from the latest
phone-local six-persistent-thread confirmation. The last two columns below
are estimates combining those sources, not a new physical server result.

| Phone FFN share | Host overlap window, measured earlier | CPU worker, measured locally | Outside-worker path, measured earlier | Estimated phone path | Estimated exposed wait |
| --- | ---: | ---: | ---: | ---: | ---: |
|50%|9.552|9.162|5.041|14.203|4.651|
|100%|0.014|18.291|4.789|23.080|23.066|

At50%, the compute branches are close: host9.552ms versus phone9.162ms.
Only0.390ms remains to hide framing, transport, copies and scheduling,
versus the earlier5.041ms outside-worker interval. The projected exposed wait
is4.651ms/layer, or27.907ms across six layers per
assisted token. Historical GPU server waits were15.337ms/layer. The outside-
worker interval includes4.859ms inside the RPC and0.181ms of outer client
handling; it is not a measurement of pure USB wire time.

At100%, the graph removes the host FFN branch. The0.014ms host interval is
bookkeeping, not an independent compute workload. The next Qwen layer depends
on the merged FFN output, so work elsewhere in the token cannot simply cover
this wait. Full offload therefore leaves about23.066ms exposed in this model.

The relevant condition is:

`exposed_wait ~= max(0, phone_compute + outside_worker_path - host_share)`

Input extraction before launch and output publication/addition after the join
are outside this overlap interval. A full end-to-end measurement must retain
those costs and compare generated tokens as well.

## Ways to close the gap

- At50%, keep the current CPU time and reduce the outside-worker critical
 path to at most0.390ms. That is a target, not an observed transport
 capability. CPU-only kernel improvement would instead need to lower the
 half-FFN from9.162 to4.511ms if the5.041ms overhead remains.
- Rebalance the fraction. A linear host model inferred from its50% time and
 an affine phone model fitted to the measured half/full CPU points balance
 at37.55% phone, with an estimated11.930ms interval. This ignores
 workload/cadence changes, copies outside the interval and merge cost; it is
 not a validated optimum.
- The current4352-column blocks permit25% steps. A25% phone/75% host split
 would have an estimated9.639ms phone path and14.328ms host path,
 hiding the phone in this model. That does not automatically beat50% total
 latency, and it offloads less host work. A37.5% split needs finer partition
 qualification; it cannot be selected with the current4352-column quantum.

The next bounded server comparison should select the qualified CPU worker
with6 persistent threads, use desktop controls around25%/50% phone shares,
and measure actual host/RPC/worker/join intervals, token identity and decode
time. A100% arm is useful as a reference for the no-host-branch case. CPU
frequency under the sparse server request cadence may differ materially from
continuous phone-local replay, so the current numbers cannot establish success.

## Source checks

- `examples/layersplit/ffn-split-client.cpp`: at`ffn_norm`, copy the input,
 record launch, notify the asynchronous exchange thread and return. At
 `ffn_phone_partial`, record host-ready and wait on completion. The worker
 backend is not part of this execution protocol.
- `src/llama-graph.cpp:build_dense_ffn_split`: partial splits compute host
 gate/up/down, publish the phone output and add the two parts. Full-width
 phone execution removes the local FFN branch.

[Machine-readable calculations](PIXEL_CPU_SERVER_OVERLAP_ASSESSMENT.json),
[historical server measurements](PIXEL_GEMV_SERVER.json),
[qualified CPU measurements](PIXEL_CPU_TUNE_RESULTS.json).
