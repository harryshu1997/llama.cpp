# S14 real-device mixed-overlap smoke

Date: 2026-07-18

Verdict: `REAL_THREE_DEVICE_OVERLAP_MECHANICS_PASS`; relief and energy are not
established.

## Hardware and routes

- Server: one A6000 selected by UUID
  `GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf`. The other A6000 was excluded
  with `CUDA_VISIBLE_DEVICES`.
- OP12: serial `5ae7a43d`, persistent `HTP0` phone-PIM worker on forwarded
  port 19012.
- OP15: serial `3C15AU002CL00000`, persistent `HTP0` phone-PIM worker on
  forwarded port 19015.
- Server workload: eight real `llama-embedding` processes per round. Each
  process encodes 256 BGE requests as eight 32-sequence CUDA forwards, for
  2048 requests per round.
- Phone workload: 16 real Gemma dense-FFN island jobs at M=16, dynamically
  work-stolen across both phones.

The phone command started 150 ms after the server leg. The server leg remained
active for the full phone setup and dispatch window. No second server GPU was
used by the experiment.

## Matched results

| Round | BGE solo (ms) | BGE with phones (ms) | Phone dispatch (ms) | Max rel L2 |
|---:|---:|---:|---:|---:|
| 1 | 4536 | 4680 | 429.426 | 0.000303477 |
| 2 | 4608 | 4622 | 426.153 | 0.000304353 |
| 3 | 4512 | 4654 | 422.118 | 0.000304353 |
| 4 | 4544 | 4641 | 423.400 | 0.000303477 |
| 5 | 4550 | 4680 | 422.105 | 0.000303477 |

Median BGE wall time was 4544 ms solo and 4654 ms with both phones active, a
2.42% slowdown. All five phone runs returned `REAL_FLEET_FFN_PASS`; median
validated dispatch makespan was 423.4 ms and the worst relative L2 was
0.000304353, below the 0.005 gate. Every round assigned work to both phones.

Raw records and SHA-256 manifest are under
`scratchpad/s14_real_overlap/20260718_bge_ffn/`.

## Measurement correction

`energy/bge_server_bench.py` is not valid in its current form. Its generated
prompt files end with a newline, which `llama-embedding` parses as an extra
empty sequence, and it does not set `--parallel` to cover the sequences in a
physical batch. Those defects caused the observed output-reserve assertion and
null sequence embedding. The command used here has no trailing separator and
sets `--parallel 32` explicitly.

## Limits

- This is real concurrent execution, but the two workloads are independent.
  The phone FFN outputs do not yet feed a Gemma server suffix.
- The server leg reloads BGE eight times per round. Each process performs real
  batched forwards, but this is not yet a persistent mixed-workload server.
- No server power trace was acquired, so this result makes no energy claim.
- BGE correctness is covered by the separate CUDA-versus-CPU cosine result;
  this smoke checks process success and phone numerical output.
- OP12 also has an unrelated long-running HTP/OpenCL process owned by another
  user. The phone latency values are therefore not an uncontaminated atlas row.

The next real-device gate is an end-to-end priority route: high-priority BGE
stays on the selected A6000 while low-priority Gemma requests are batched across
resident phone islands and their validated activations continue through the
server suffix. Only that run can establish server relief and support a matched
GPU-board energy experiment.
