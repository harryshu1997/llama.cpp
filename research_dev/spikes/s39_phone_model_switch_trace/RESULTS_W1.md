# S39 W1 direct phone activation chain

Verdict:

`DIRECT_CHAIN_MECHANICS_PASS_REPEATS_PENDING`

No model-switch, latency, SLO, throughput, or energy benefit is claimed.

## Executed topology

```text
host coordinator
  | token, request, epoch, position
  v
OP15 relay + Qwen [0,30)
  | F32 cut activation over WiFi
  v
OP12 Qwen [30,40)
  | token, request, epoch, position
  v
host coordinator
```

Weights and binaries were provisioned over USB before execution. The relay
binary ran on OP15 and connected to OP15 through loopback and to OP12 at
`172.20.59.72:39312`. The host connected to the OP15 relay at
`172.20.173.218:39415`.

## Real-device result

The fixed relay executable SHA-256 was
`1c809cb50cae6aa86869d61068a05173c719e4542c851e478ee1e033c5456929`
on both the host build and OP15 after the run.

| Cohort | P50 service | Executed batches | Direct activation | Host activation |
|---:|---:|---|---:|---:|
| B1 | 5.653 s | 5, then 7 x 1 rows | 245,760 B | 0 B |
| B32 | 90.796 s | 160, then 7 x 32 rows | 7,864,320 B | 0 B |

All 264 generated-token checks matched the frozen CUDA token sequence:

```text
12095, 13, 3555, 374, 279, 6722, 315, 279
```

The same OP15 and OP12 worker process and boot nonce persisted across both
cohorts. B1 ended with `DETACH` and request-local KV reset; B32 ended with
`STOP`. The final placement records reported:

| Device | Layer range | OpenCL compute nodes | Declared CPU compute |
|---|---|---:|---:|
| OP15 | `[0,30)` | 88,320 | 128 `GET_ROWS` |
| OP12 | `[30,40)` | 30,080 | 0 |

Both workers reported zero missing compute buffers. OP12 logged a compile
failure for one specialized OpenCL flash-attention variant, then executed the
graph entirely on OpenCL. This is a backend-kernel fallback, not CPU fallback.

## Host-relay comparison

One separately loaded matched host-relay B1 control took 5.696 seconds. The
fixed direct B1 point was 0.76 percent lower. Earlier direct points ranged from
4.608 to 5.298 seconds, so process and device state variation is larger than
the fixed-point difference. The comparison is informational only and does not
establish a direct-transfer latency win.

Direct transfer does establish a structural result: the host no longer
receives and retransmits the cut activation. B32 host activation payload falls
from 15,728,640 bytes in the two-leg control to zero in the direct route. This
does not imply half the WiFi airtime because an access point may forward the
phone-to-phone frame.

## Verification

- host CMake build: pass;
- Android arm64 build: pass;
- strict `-Wall -Wextra -Wpedantic -Werror` build: pass;
- ASan and UBSan protocol self-test: pass;
- happy path, `n_batch` overflow, and invalid-status-capacity self-tests: pass;
- direct evidence reducer: pass;
- eight evidence mutation tests: pass;
- real B1 and B32 token, batch, placement, reset, and process-persistence gates:
  pass.

The reducer output is
`results/w1_direct_phone_chain/direct_chain_certificate.json`.

## Remaining boundary

The host remains authoritative for admission, batching, route epochs, and
result ownership, but W1 uses a single-client implicit reservation. It does not
yet carry a host-signed batch descriptor. WiFi interface counters, route
snapshots, retransmissions, and matched repeated controls were not captured.
Therefore W1 certifies direct-chain mechanics only and does not promote the
Qwen route beyond provisional status.
