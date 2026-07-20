# S13 Runtime Fleet Manifest

Acquired on 2026-07-16 EDT from the shared ADB server.

## Device and Link Binding

```text
OP12: serial 5ae7a43d, model CPH2583, USB bus 006 port 002, negotiated 5000M
OP15: serial 3C15AU002CL00000, model CPH2749, USB bus 008 port 003, negotiated 5000M

5ae7a43d         tcp:19012 -> tcp:9090
3C15AU002CL00000 tcp:19015 -> tcp:9090
```

The physical IDs in the JSON records are caller assertions cross-checked
against `adb devices -l` and `adb forward --list`; protocol v3 does not
cryptographically bind a serial or worker build into HELLO.

## Content and Binary Hashes

```text
phone shard, both devices:
  5cfba18d2a47acc190f317d650895bcc53e914e9a0bc61631860be9591ed360d

deployed worker, both devices:
  0a50ca7299ee15a9eb9d524a0dadea40b467721f011a57b7f89617b1bd30e749

host fleet binary:
  d3cc65a65fb7d520b8205279a61a3872918cade3377af452b9d48ec37a188ec0

host source identities:
  phone_pim_fleet.cpp 14c99122e3a5a34ed392147714c7cd0b76c7bc661fd145329625b7def0978fe6
  phone_pim_client.cpp 2e253d8d6f3dcb740475ebb0ac3da84e26c891fb2e94ca68b4a1e9e8b2e771f0
  phone_pim_client.h   f3356ab2aa233dac0dce0d9662a68578216274ffba89626044756e580a5448e4
  phone_pim_json.h     8748e99b7aba7c8a53d670da6b1f33efc858ba7fbeb40cba40e0367347775b7f

cold.json:
  1b42a18a08b96f7730910ae8c652568d6fb6032ead9175880ce091c1dd731584

warm.json:
  ffe2df25c9ab465b59d87587cce165577c7461d57898cd6fed3906f009bd80cc

independent production-graph oracle records:
  OP12 12d3cf37e15b0f96170d6206e0f6d560835ce5b8165049090b418ef57b1906a6
  OP15 b6d8285d6cbbb8be7d0a7bf4f16f706f13ad9ca996d2e10d837601280d85bf2a
```

The oracle records are retained at
`../s10_power_frontier/artifacts/cp0_ffn_op12.json` and
`../s10_power_frontier/artifacts/cp0_ffn_op15.json`. They use the same worker
binary and model shard as S13.

## Commands

Cold and warm runs used the same command except for `--jobs`:

```sh
build-phone-pim/bin/llama-phone-pim-fleet \
  --device op12,127.0.0.1,19012,5ae7a43d \
  --device op15,127.0.0.1,19015,3C15AU002CL00000 \
  --model scratchpad/phone_pim/12b-f16-mid-2-3.gguf \
  --prefix blk.2 --M 16 --jobs 8 \
  --route-epoch 1 --generation-hint 999 --island-id 1
```

The cold run followed an explicit two-device `--release` run and began with
both workers non-READY at generation 7. The warm run immediately followed it,
used `--jobs 16`, and found both workers READY at generation 7. Both records use
schema v2 and encode protocol uint64 identities as decimal strings.

## Worker Logs

```text
OP12 worker log: 38 lines, 2405 bytes
sha256:cd4555ed2813fae4d94bb4a4fe6cf19c931c93b142cb7f2888f764ecfe73174e

OP15 worker log: 38 lines, 2408 bytes
sha256:23a4a911f9e3b645c88b96c6a2babc42ddf61b8e6ef73b28ba131633d089bcde
```

The logs are retained beside the JSON artifacts. They show HTP0 READY for
`blk.2`, M=16, with 342.0 MiB resident on each phone.
