# OP15 USB NCM operator transport

Date: 2026-08-02 EDT.

Verdict:
`NCM_STREAMING_GOODPUT_CONFIRMED; 1.75_MS_RPC_FLOOR_BLOCKS_DECODE_OPERATOR_SPLITS; LARGE_BATCH_OR_BACKGROUND_TRANSFER_ONLY; P99_NOT_STABLE`.

## Question

The OP15 USB NCM link showed high streaming throughput. This experiment asks
whether that throughput also reduces the response-ready time of the payloads
used by the phone operator paths, and whether it permits a larger phone split.

The phone was connected directly to the physical RTX 4060 Ti host. It
enumerated as CDC NCM at 5,000 Mbit/s USB and reported a 3,750 Mbit/s NCM
link. The transport probe used one persistent IPv6 TCP connection with
`TCP_NODELAY`, one request outstanding at a time, and no phone computation.
The phone daemon read the exact request and returned the exact requested
response size. Each of three fresh daemon processes ran 50 warmups and 300
paid exchanges for each case. Case order was changed between repetitions.

An exploratory single-stream test reached 2.61 Gbit/s from desktop to phone
and 3.37 Gbit/s from phone to desktop, or 326 and 421 decimal MB/s. One reverse
trial collapsed after two seconds before two stable reruns. Therefore the
accurate statement is not "400 MB/s in both directions": one direction was
about 326 MB/s, the other reached about 421 MB/s, they were tested separately,
and simultaneous full-duplex or stable p99 goodput was not established.
The raw streaming output was not retained, so these goodput figures remain an
exploratory observation rather than primary paper evidence.

## Response-ready result

The primary table is reproducible from the three raw JSON captures. Medians,
p90, and p99 are the medians of the corresponding fresh-process values. The
paired data cost subtracts the zero-payload control inside each repetition and
then takes the median. The effective rate divides total request plus response
payload by round-trip time; it is not a full-duplex throughput claim.

| case | request | response | median | paired data cost | p90 | p99 | effective aggregate rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero-payload control | 0 B | 0 B | 1.748 ms | 0.000 ms | 1.797 ms | 2.067 ms | 0.0 MB/s |
| attention state | 1.25 KiB | 1.25 KiB | 1.764 ms | 0.016 ms | 1.814 ms | 2.495 ms | 1.5 MB/s |
| hidden, M=1 | 10 KiB | 10 KiB | 1.872 ms | 0.124 ms | 1.944 ms | 2.235 ms | 10.9 MB/s |
| standalone SwiGLU | 68 KiB | 34 KiB | 2.068 ms | 0.282 ms | 2.107 ms | 2.234 ms | 50.5 MB/s |
| hidden, M=8 | 80 KiB | 80 KiB | 2.123 ms | 0.375 ms | 2.290 ms | 2.452 ms | 77.2 MB/s |
| hidden, M=32 | 320 KiB | 320 KiB | 3.700 ms | 1.935 ms | 4.474 ms | 5.387 ms | 177.1 MB/s |
| hidden, M=128 | 1.25 MiB | 1.25 MiB | 7.847 ms | 6.099 ms | 8.582 ms | 10.070 ms | 334.1 MB/s |

All three daemons completed exactly 2,450 exchanges. The raw captures contain
6,300 paid samples. The M=128 result independently confirms that NCM reaches
hundreds of MB/s once the transfer is large. It also shows why streaming
goodput alone is the wrong metric for decode: the zero-payload synchronous
request/response already costs about 1.75 ms.

The tail is not yet a publication-quality result. The first repetition had
6.82 ms control p99 and 6.88 ms attention p99, while the later control p99
values were about 2.0 ms. The medians repeat well, but a longer alternating
campaign is required before claiming tail behavior.

## Comparison with direct AOA

The earlier direct-AOA controls use the same physical phone and desktop:

| boundary | NCM transport only | best complete AOA path |
| --- | ---: | ---: |
| zero/small request | 1.748 ms | 0.256 ms no-op |
| 10 KiB hidden each way | 1.872 ms | 0.300 ms persistent RMSNorm, 0.414 ms HTP RMSNorm |
| 1.25 KiB attention state each way | 1.764 ms | 0.459 ms HTP attention including compute |
| 68 KiB/34 KiB SwiGLU | 2.068 ms | 0.943 ms HTP including compute |

For every latency-critical decode boundary tested here, NCM transport alone
is slower than AOA transport plus phone computation. Replacing AOA with the
current NCM/TCP path would remove the existing FFN, vocabulary-head, and
resident-KV attention wins.

NCM does become efficient for hundreds of KiB to MiB transfers. That makes it
useful for asynchronous weight or KV staging and potentially for a large
prefill boundary whose work can be pipelined. It does not by itself authorize
such a route: the existing real-device FFN campaign already shows CUDA batch
efficiency overtaking the phone at M>=8.

## Does it permit a larger split?

Not for the current best decode designs:

- A fused FFN column shard sends one hidden vector in and one hidden residual
  out. Increasing the resident phone column count does not increase this wire
  boundary, so phone compute and numerical margin determine the split.
- The sharded vocabulary head returns only local top-k candidates. It is
  compute-bound and gains nothing from NCM bandwidth.
- Resident-KV attention sends and returns compact per-token state. Kernel
  latency and the 1.75 ms NCM floor dominate.

NCM can carry a larger activation boundary, but that is different from making
the critical path faster. The next bounded transport experiment, if this
branch continues, should compare AOA, NCM/TCP, and NCM/UDP at the exact same
160 KiB, 640 KiB, and 2.5 MiB round trips. A useful NCM operator route must
either reduce its small-RPC floor below AOA or amortize the floor across a
large pipelined batch while still beating CUDA compute and total energy.

## Evidence and restoration

Evidence root:
`results/ncm_operator_transport_v1/run_20260802T1336EDT/`.

The root contains all raw JSON samples, worker logs, source snapshots, the
deployed Android binary, an independently generated analysis, and SHA-256
manifests. The temporary Android IPv6 policy rule was removed, NCM workers
were stopped, and OP15 was restored to `ptp,adb` at 5 Gbit/s USB. No model,
full layer, BurstGPT trace, power, or energy measurement was performed.
