# V2.4 desktop deployment preflight

Status: `NO_MODEL_TOPOLOGY_PASS; USB_LAUNCHER_MECHANICS_PASS; ARTIFACT_RECEIPT_IN_PROGRESS; A_ONLY_NOT_RUN`

This checkpoint validates only the control topology and prospective A_ONLY
input construction. It does not qualify Qwen3 14B, execute a model, validate
the remote runtime closure, or authorize B_ONLY, PAIR, a switch cycle, a trace
campaign, or an energy claim.

## Live no-model topology

The passing receipt is:

`results/no_model_topology_20260726T123018Z/topology.json`

Its SHA-256 is
`158e5dd224369c4a5923da5db9e6bb4808bc5e2b7e2a818fde48488877dce6f6`.
It was captured on the RTX 4060 Ti desktop through a temporary reverse tunnel
to the A6000 host's physical USB ADB server on port 5038.

The receipt binds:

- desktop `zhihao-Z690-C-ac`, boot
  `2f68fcf5-54e1-4306-ba25-63359be572c2`;
- RTX 4060 Ti UUID
  `GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08`;
- OP15 physical serial `3C15AU002CL00000`, USB `8-3`, boot
  `3eb99d7e-0b35-41a5-9e30-21867dc5dec7`;
- OP12 physical serial `5ae7a43d`, USB `6-2`, boot
  `2ca4b7a3-c9c1-4614-a0d2-5746c57c8c4d`;
- all six interface-pinned zero-payload ICMP edges between the desktop, OP15,
  and OP12.

The live validator printed `V24_NO_MODEL_TOPOLOGY_PASS`. A separate local
replay printed `V24_TOPOLOGY_REPLAY_VALID`. Every file in the passing result
directory verifies against its `SHA256SUMS.txt`.

Three earlier attempts are preserved and remain failed:

1. `20260726T122532Z`: relative contract path rejected.
2. `20260726T122613Z`: desktop self-SSH host identity was not configured.
3. `20260726T122839Z`: the exact two-newline `ping -q` trailer was rejected.

The verifier fixes only make those commands reproducible. They do not weaken
the identity, packet-count, endpoint, interface, chronology, or freshness
checks. The topology suite passes 51/51 tests.

## Prospective A_ONLY inputs

`materialize_a_only_inputs_v1.py` validates a digest-pinned desktop inventory
and derives the exact Qwen3 14B route:

- backend `GPUOpenCL`, cut 30;
- OP15 executes `[0,30)` from stored `[0,32)`;
- OP12 executes `[30,40)` from stored `[24,40)`;
- physical USB serial selectors on ADB port 5038;
- exact model, shard, runtime, command, port, and capture-entrypoint closure;
- unbound boot and network placeholders for the later post-reboot binder.

The adapter passes 20/20 tests. Its generated prospective spec is accepted by
the frozen V2.4 originator.

The historical managed-runtime launcher requires WiFi-style ADB selectors
containing `:` and a different component-stat shape. The versioned
`managed_runtime_launcher_usb_v1.py` successor securely loads that exact
historical source, accepts only the two physical serials on port 5038,
normalizes only the missing `build_id` field to `null`, and replaces only the
ADB prefix with `adb -P 5038 -s <physical-serial>`. All remaining validation
and execution stays in the historical launcher. The launcher suite passes
20/20 tests, and the adapter suite passes 22/22.

The exact successor now lets the adapter publish prospective outputs through
the frozen originator. This is still not acquisition readiness because remote
artifact evidence is not yet bound.

## Remaining gate

Before A_ONLY can run:

1. Produce the long pre-reboot artifact receipt for the CUDA model, both full
   phone shards, and all runtime components, including pre/post stat and hash.
2. Produce the short post-reboot receipt that binds fresh boot/network/device
   identity to the long receipt without rehashing multi-GB shards.
3. Bind running PIDs, start ticks, loaded components, shard mappings,
   placement, and direct OP15-to-OP12 socket peers.
4. Re-run the complete V2.4 authority and stop on any refusal.

Full local-UFS reprepare remains a later PAIR gate and is not part of A_ONLY.
