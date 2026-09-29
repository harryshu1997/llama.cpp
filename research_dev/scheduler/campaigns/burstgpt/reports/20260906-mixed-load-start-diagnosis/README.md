# Mixed-session load-start diagnosis

The v6 FAIL is preserved. No scheduler, native, qualification, or metric
logic changed in this diagnosis, and no new physical run was launched.

Raw desktop logs narrow the apparent phone-loading interruption to a
desktop CUDA graph recapture interval. After the reduced mask is applied,
token 45 contains 13 `CUDA graph warmup complete` events between its last
phone RPC and the next token's first phone RPC. The matching build source
sets `cuda_graph_update_required = true` at that exact log site, followed
by capture and executable creation/update. Source file hashes match the
local files inspected; the desktop binary hash is unchanged.

| Measurement | Observed |
| --- | ---: |
| Last retained RPC, request 168 | 11.393038 ms |
| Time from its D2H completion to next RPC submission | 907.483216 ms |
| Next retained RPC, request 169 | 12.032650 ms |
| First-to-last CUDA recapture log span | 569.337 ms |
| Token 44-to-45 latency | 1,136.030 ms |

The 569.337 ms is a log span, not an isolated GPU capture-time measurement.
Nevertheless, the 907 ms is demonstrably between host RPCs, not time
waiting for either of those RPCs to finish. Adjacent HTP compute times are
9.274 and 10.449 ms. This updates the earlier load-start correlation; it
does not establish that phone loading has no other contention cost.

The gate's matched phone-monotonic ratios remain 2.052805x and 2.031402x,
both above the unchanged 2x bound. Retained HTP0/HTP1 remain generation 1
and make 114 calls each during HTP2's replacement. Reverse replacement and
rollback remain unproven by these runs. The old metric and FAIL artifacts
were not modified. Moving the load boundary past CUDA warmup would hide,
not fix, the interruption.

## Safe next step

The current launcher has no declared CUDA-graph mode in its desktop
contract. Silently injecting `GGML_CUDA_DISABLE_GRAPHS` would change the
qualified runtime without changing its identity. That is not an acceptable
scheduler-only repair.

The smaller explicit experiment is to add a desktop-contract graph mode,
carry it into placement/qualification identity, and freshly calibrate
graph-disabled execution for control and phone-assisted routes. Keep
artifacts, GPU layer placement, prompts, decoding, and phone binaries
unchanged; do not inherit the old energy qualification. Then rerun only
the bounded mixed gate. The alternative is a native GPU graph-cache repair
that preserves exact buffer/topology validation across CPU/phone mask
changes. Either path needs an explicit scope decision before implementation.

## Reproduction and evidence

Artifact:
`/home/zhihao/s42-ffn-mixed-session-20260905-v6-gate/run/large-model-5-physical-hot-desktop.stderr`

```sh
python analyze_saved_host_gap.py /path/to/large-model-5-physical-hot-desktop.stderr --from-call 168 --to-call 169
```

This analyzer only reads an existing log. The call IDs select the frozen
v6 interval; they are not scheduling policy. Its decoded output and exact
source/binary hashes are in [HOST_GAP_AUDIT.json](HOST_GAP_AUDIT.json).
Stderr SHA-256:
`bf317a36177d240493a29dffa85d492c050f5dad458c5b979210df066f8f5d24`.

Changed files in this continuation: this report, its audit/analyzer, and
`research_dev/talks.md`. Production code and previous golden hashes are
unchanged. The prior 170 focused tests remain the latest scheduler test
result; no unrelated test suite or physical trace was run in this diagnosis.
