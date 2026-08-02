# S39 desktop stock-default controls

## Question

Measure the same frozen 74-request, two-model trace using the stock
`llama-server` inference defaults rather than the explicit full-CUDA serving
envelope in CP0-D.

This is a separate control. It does not replace or modify the frozen CP0-D raw
campaign.

## Profiles

`STOCK_DEFAULT_SWAP` runs one server process. At each target change it closes
admission, drains active requests, stops the process, starts the target model,
waits for health, and resumes the matching FIFO.

`STOCK_DEFAULT_DUAL` starts one server process for each model before the paid
trace. Requests go directly to the matching process. The processes use the same
GPU. If the second process cannot become ready, that is a valid bounded
dual-load failure.

Both profiles pass only required identity, endpoint, and observability options
to `llama-server`. GPU layers, fit, context, slots, batching limits, flash
attention, KV layout, KV types, and continuous batching are left at the
binary's documented defaults.

## Acquisition

Run three fresh-process repetitions of:

1. `STOCK_DEFAULT_SWAP` with warm page cache.
2. `STOCK_DEFAULT_SWAP` with cold NVMe.
3. `STOCK_DEFAULT_DUAL` with warm page cache.

Use the exact CP0-D requests, target changes, payloads, and 30-second SLO. Use
only GPU 0. Bound each request and server start. Preserve a failed load or
failed replay rather than retrying or changing settings.

## Metrics

Record per-model and total throughput over time, selected-GPU cumulative board
energy, TTFT, completion latency, SLO goodput, queueing, model load placement,
VRAM, host RAM, and swap growth.

Phone energy, server-wall energy, and total-system energy remain unknown.

## Stop

Stop after raw acquisition, independent reduction, two graphs, manifests, and
an honest result. Do not run phones, change llama.cpp inference code, tune a
default after observing it, commit, or push.
