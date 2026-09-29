# Remote-resident phone: batch-capacity fix and fresh transport checks

## Outcome

The real-phone FFN relocation and owner-loss gates have NOT run. The new
preflight fails closed on a changed phone kernel, not on host software hashes.
No new model-memory, KV-capacity, output-correctness or energy-saving claim is
made here. The previous desktop TCP stand-in remains separate evidence.

Fresh remote artifact root:
`/mnt/storage/s42-remote-resident-phone-20260913-v2-WTUd4V/`.
The local `physical/` directory preserves receipts, commands and the blocked
preflight. Native binaries and the isolated deployment remain on the desktop.
Nothing overwrites earlier attempts.

## Physical timeline and blocker

- At the start, OP15 was on USB 2-2 at 5000 Mbps, ADB 5037, kernel
  `6.12.23-android16-5-o-g227664cbe007-4k`.
- The first bounded transport case completed with worker status 0. Android USB
  restored, but the extra ADB daemon on port 5038 claimed the device. That daemon
  had been accidentally started by this agent's earlier diagnostic (PID 257764,
  September 13 at 17:14). The canonical 5037 wait consequently timed out.
  Its receipt is retained under `physical/transport/`; it is not part of the
  completed qualification set. The task-created 5038 daemon was closed; the
  original 5037 daemon (PID 9886) reclaimed OP15 without a USB reset.
- Nine fresh directional/duplex checks completed by 18:07 EDT. Each has a zero
  worker exit status, no recorded kernel faults, zero transport reset recoveries,
  and a terminal record proving restored Android USB at SuperSpeed.
- A new qualification identity was materialized at 18:11:18 from those nine
  receipts, current host binary/library hashes, and actual phone binary hashes.
  The USB probe was rebuilt against the exact current USB client source. No
  FFN shard generator, ARM FFN worker or inference binary was rebuilt.
- USB records show an additional reboot beginning around 18:14. At 18:19:12,
  preflight rejected kernel
  `6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k`.
  The boot ID changed from `37fbaeca-60d5-44f2-987b-b93e82ecc7ff` to
  `649a2d68-ef7b-4ced-9a79-27b5bcb3f5bd`. Android reports `reboot`, and pstore
  is empty. This proves a boot change, not its initiator or cause. The agent
  issued no reboot, flash, kernel change, GDM stop or driver command.
- Final observation: OP15 is responsive on ADB 5037 with `ptp,adb`; GPU is idle,
  299 MiB used and 15,649 MiB free. No inference or FFN worker is left running
  by this gate. Only the original ADB daemon remains.

The expected kernel must be restored or the new kernel independently qualified
before physical FFN preparation can safely continue. The new identity must not
be reused on the changed kernel simply by editing its hardware fields.

## Transport measurements

These are payload-transfer measurements, not FFN inference latency or HTP proof.
Each direction used five warmups and 100 measured iterations, queue depth 1,
DMA-BUF and the devmem allocator. Rates use decimal MB/s.

| Payload bytes | H2D MB/s | D2H MB/s | Duplex median round-trip ms |
| ---: | ---: | ---: | ---: |
| 7,680 | 43.35 | 33.96 | 0.199 |
| 10,240 | 47.34 | 42.34 | 0.217 |
| 3,932,160 | 374.09 | 427.01 | 17.692 |

The final size covers a 512-token Gemma prefill ubatch in F16. These receipts
qualify USB transport on the earlier kernel, not the new prefill FFN execution
shape. The experimental catalog retains a conservative 768 MiB phone workspace
reservation, explicitly not a measured runtime allocation.

## Tested code retained in the active tree

1. `_internal/route_generation/costing_parameters.py`: a configured
   `ffn_max_tokens=512` was overwritten by decode parallelism 1. Preserve an
   explicit resident capacity, validate it against the requested shape and
   desktop ubatch, and size the transport for it. Unspecified capacities retain
   the existing behavior.
2. `_internal/route_generation/feasibility.py`: include the provisioned phone
   batch in the workspace estimate even when the current request is decode-only.
   Preserve larger existing demands and all other devices' demands. Existing
   configured workspace minima still apply.
3. `tests/test_remote_resident_gate.py`: five focused cases cover the declared
   capacity, unchanged defaults, invalid capacities, workspace and rejection
   when the required transport profile is absent.

Two before-image tests reproduce the capacity overwrite and invalid-capacity
acceptance. The after-image focused set passes 81 tests. Both saved replay
tests pass, with goldens unchanged. Compile, pyflakes and whitespace checks pass.
No broad suite or inference trace was run.

The unfinished phone-owner wiring is preserved as `UNVALIDATED_GATE_DRAFT.patch`,
not applied to the active gate. It is NOT ready to run: it still needs canonical
request execution authorization/leases, a valid preload payload, bounded
owner-loss control and terminal cleanup. In particular, the draft's standalone
empty-lease commands are not accepted as proof of scheduler-owned execution.
The active `remote_resident_gate.py` and `heterogeneous_rig.py` are byte-identical
to their before-images. The isolated remote deployment includes the draft used
only for preflight; do not launch its saved AB command. Re-sync reviewed code to
a new deployment before proceeding.

## Hashes

- Qualification identity canonical hash:
  `sha256:e9005ae70b264d6e759c7b079fc4d9f318d0c978b837a14b68f8043bc9eea733`.
- `physical/TRANSPORT_IDENTITY.json` file SHA-256:
  `fc392925e35550b0581e56f940d99efaaf02d50d4f3c47e8881ff0fed1ff391f`.
- Blocked `PHONE_OWNER_PREFLIGHT.json` file SHA-256:
  `12a6d7df109f36edceca7ff4e4758a95d8082970709cb6819289a12e6d33039c`.
- `GATE_CATALOG.json` file SHA-256:
  `fb475e6d6edde30514b362a1b4fbc5bf0edbcebf653c853bbdbe80f5b2d97eaa`.
- Source hashes: `SOURCE_CHANGES.sha256`. No commit, push or PR.

Replay goldens:

- Session COW v3: `sha256:5d52e8673fc60f974ca167964b3ad579e59abea28feb3d66357cf39c1ca7caf4`.
- Sparse-locality v8: `sha256:241924463ac27ead0c2d7bcb5da214e85097049c91d5800848e0b29effd2b917`.

## Resume order

1. Resolve and coordinate ownership of the OP15 kernel/boot state; verify the
   exact kernel, binaries, USB identity and live memory again.
2. Complete the existing canonical offline/session preparation and request-bound
   execution path. Use real scheduler leases, not the archived draft's commands.
3. Run only the bounded real-phone correctness/memory gate and isolated owner-loss
   recovery. Record actual shard load/verification/READY, physical calls and
   allocation evidence before claiming relocation.
4. Useful KV capacity is a later test. These CPU-resident FFN tensors can release
   host RAM, not GPU VRAM. No 24/84-request trace is authorized by this milestone.
