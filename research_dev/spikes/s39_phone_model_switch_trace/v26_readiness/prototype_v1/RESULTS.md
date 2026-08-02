# S39 joint B8 prototype result

Status: `JOINT_B8_PROTOTYPE_EXECUTION_PASS_TOKEN_DIVERGENCE`

This is a prototype result, not V2.4 or V2.6 qualification evidence.
Thermal readiness, reboot freshness, task quality, energy, and authority were
not evaluated.

## Real-device result

One Qwen3-14B Q4_K_M B8 cohort ran concurrently on:

- RTX 4060 Ti CUDA0, layers `[0,40)`;
- OP15 OpenCL, layers `[0,30)`;
- OP12 OpenCL, layers `[30,40)`;
- direct OP15-to-OP12 Wi-Fi TCP activation transfer.

All eight requests completed eight continuation tokens on both routes. The
phone and CUDA workers each transitioned from zero to eight live sequences and
back to zero. Scoped teardown left no S39 process on the desktop or either
phone.

| measurement | result |
|---|---:|
| phone B8 execution | 45.239 s |
| CUDA B8 execution while phones were active | 1.601 s |
| CUDA process start to B8 start | 6.975 s |
| phone/CUDA greedy agreement | 60/64 tokens |
| activation relay | 19 batches, 763 rows |
| activation payload | 15,626,240 bytes |

The four cross-backend mismatches were sequence 6 positions 2, 7, and 8, and
sequence 8 position 2. Cross-backend greedy equality is diagnostic only. A
future handoff correctness gate must use a path-matched CUDA replay oracle.

All observed placement certificates passed:

- CUDA: scheduled compute on CUDA0, with only 29 `GET_ROWS` nodes in
  `CUDA_Host`;
- OP15: scheduled compute on OpenCL, with 19 `GET_ROWS` nodes on CPU;
- OP12: all scheduled compute on OpenCL;
- relay: `DIRECT_RELAY_OK`, exact model identity, cut 30, and direct
  OP15-to-OP12 endpoint binding.

## Evidence

Desktop run root:

`/home/zhihao/s39-v26-a-only/prototype-v1/run_20260727T034430Z`

Key SHA-256 values:

```text
RESULT.json                 ffddf6c0ba3a486c437f2adbd1cf856e9328352d8c315d5a9a2e5dfeb03fbfa3
phone-route-prototype.json  6ebb2261fb7372af1d4b29075296cc8f72175a682e1fec859c3f44f421f818a1
cuda-route-prototype.json   b6ee6712ac28b29c5c906c0c8a60ca869a31f59f721bc0aa97aec5136a62a409
cuda.log                    df0e20c5196b2d9ed7d6dc7f3e04cd7387837397bf5657772f4c78648d70e214
op12_stagenet.log           982e161368d9ba2d939c71d6aa0c82365817b9550047da89d356a279486a1637
op15_stagenet.log           d1bd96d3807e1a9a4b1c081f858af3b406714c25e59f672696efd5170dea3f40
op15_direct_relay.log       e1214a6bab62f9c6e4087c87134ea226e26974bff0293b11ef09883e0d104f39
```

The phone Wi-Fi addresses were discovered live rather than reused from the
frozen plan: OP15 `172.20.173.218`, OP12 `172.20.59.72`. Both phones stayed on
`PAWS-Secure`; the competing saved `ASUS_5G` profiles were removed before the
run. ADB on port 5038 remained the control plane. Wi-Fi TCP carried only the
runtime activation path.

## What this closes

This closes the first functional end-to-end milestone: the exact large model
loads and executes with continuous B8 state on the two-phone OpenCL route
while the same model executes on the target CUDA server.

It does not show a performance win. The phone route is about 28 times slower
than concurrent CUDA for this cohort. The next bounded experiment should
measure one actual switch:

1. start the phone route as the temporary owner;
2. start CUDA loading concurrently;
3. use `k_extra=0`;
4. replay the path-matched phone frontier on CUDA;
5. transfer ownership and finish on CUDA;
6. compare against matched CUDA-only warm and cold-load controls.

Before treating that run as qualification, move the prototype-only plan
repairs into the materializer and launcher under a separately reviewed
successor. Do not weaken the frozen V2.4/V2.6 authority in place.
