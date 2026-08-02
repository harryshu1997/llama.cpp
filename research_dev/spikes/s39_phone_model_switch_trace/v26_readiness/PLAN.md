# S39 CP0-R1 V2.6 bounded authority

Status: `MECHANICS_PASS; V2_4_1_DEFECT_FIX_APPLIED; A_ONLY_QUALIFICATION_BLOCKED_AT_FRESH_READINESS; PROTOTYPE_JOINT_B8_EXECUTION_PASS_TOKEN_DIVERGENCE`

## Prototype checkpoint

The user-authorized no-reboot, no-thermal prototype completed one real
Qwen3-14B B8 execution on 2026-07-27. OP15 ran layers `[0,30)` on OpenCL,
OP12 ran `[30,40)` on OpenCL, and the RTX 4060 Ti concurrently ran
`[0,40)` on CUDA0. All eight requests completed eight tokens, every route
returned from 8 to 0 sequences, all placement certificates passed, and
scoped teardown left zero S39 processes.

Phone/CUDA greedy agreement was 60/64. This is diagnostic cross-backend
divergence, not a handoff failure or qualification pass. The run used ADB USB
as the control plane and direct OP15-to-OP12 Wi-Fi TCP for 19 activation
batches. See `prototype_v1/RESULTS.md`.

The frozen V2.4/V2.6 qualification remains separately blocked at
`fresh_readiness`. No V2.4/V2.6 model-execution or authority claim is made
from the prototype.

## V2.4.1 successor digests

The authorized quoting fix produced one successor digest set. The V2.4
originals are preserved verbatim in `frozen_parents_v24/` with a
`SHA256SUMS.txt`; no gate, bound, threshold, or predicate changed.

| file | V2.4 parent | V2.4.1 successor |
|---|---|---|
| `production_plan_v1/preparation_v1.py` | `2aa020e1...66ef0d2` | `1e02e19b...2300a443` |
| `CP0_R1_EVIDENCE_CONTRACT_V2_4.json` | `20eea0fd...4731b455` | `264d16b3...b73a439f` |
| `desktop_deployment_v1/artifact_root_capture_v1.py` | `dda6f851...708ca78b` | `a70f72bf...804f9475` |
| `desktop_deployment_v1/fast_fresh_capture_v1.py` | `05d675b9...ec96eced9` | `bb77fa9f...b251c68cc` |
| `desktop_deployment_v1/verify_topology_v1.py` | `0f90228f...4998c129` | `602dc531...d3885c66` |
| `desktop_deployment_v1/materialize_a_only_inputs_v1.py` | `939b978f...9d7cd4561` | recomputed |
| `CP0_R1_EVIDENCE_CONTRACT_V2_6.json` | rebuilt | `8eaad17a...a3e57082` |

Only the first file changed behaviour; the rest carry the propagated digest.
All 149 V2.4 tests and the V2.6 suites pass against V2.4.1.

V2.6 is a thin successor around the frozen V2.4 raw predicates. It authorizes
only Qwen3-14B `A_ONLY`; B_ONLY, PAIR, model switching, trace replay, and
energy remain unauthorized.

The gate binds the exact candidate, MMLU64 corpus, V2.4 authority bytes,
capture producer bytes, runtime inventory, managed plans, phase lock, and
capture execution receipts. A receipt records the source executed from a
verified file descriptor, exact invocation, all input hashes and stat pairs,
stdout/stderr, result bytes, and monotonic/UTC intervals.

The outer validator reopens those records, validates the current runtime
inventory and managed plan semantics, checks lock ordering, and invokes the
V2.4 raw evaluator. A stored status string is never accepted as evidence.

Required order:

1. Build and adversarially test V2.6. [PASS]
2. Re-run the no-model topology and materialize the exact runtime inventory
   plus the A_ONLY phase lock on the RTX 4060 Ti, OP15, and OP12. [PASS -
   current root `results/production_materialization_20260726T220718Z/`,
   phase `cp0-r1-v26-a-only-20260726T220529Z-6176d8d7`, 40 components;
   produced by `materialize_production_v26.py`, independently validated and
   live-checked. The earlier 38-component root is superseded by the
   producer-pin rule after the route bundle gained the two capture copies.]
3. Materialize the missing V2.4 production plan set through one bounded
   V2.6-to-V2.4 adapter. [PASS - `materialize_v24_plan_set_v26.py` derives
   the closure input, operator input, and desktop inventory from the V2.6
   evidence and drives the UNMODIFIED frozen
   `materialize_a_only_inputs_v1.py` -> `originate_runtime_v1.py` chain on
   the acquisition desktop. All five prephase artifacts exist at
   `../v24_readiness/results/prephase_20260726T0915Z/` and on the desktop
   mirror; `materialize_config_v1.py` accepts them (config+plan built).]
4. Close the frozen executor seams before the joint capture can run. [DONE -
   `managed_runtime_launcher_snapshot_v1.py` and
   `phone_runtime_probe_snapshot_v1.py` wrap the frozen launcher and frozen
   v23 probe under `boot_id_source=phase_fresh_snapshot`; pinned into the
   plan set via `originate_from_inventory_v26.py` and validated, but NOT yet
   exercised against hardware.]
5. V2.4.1 defect fix - AUTHORIZED by the human owner on 2026-07-26 after the
   first real acquisition (`a-only-run-2`) passed `artifact_root`, physically
   rebooted both phones, and refused at `E_FIELD: cuda.status`. The frozen
   probe passed `sh -c <script>` as separate ssh arguments, so the remote
   shell ran `sh -c set` and returned 28 lines of shell variables. The fix
   quotes the script into ONE remote argument, exactly as the sibling frozen
   `capture_common.ssh_python_argv` already does; verified against the real
   desktop to return exactly the five required keys.

   Scope rules for V2.4.1: no gate, bound, threshold, or predicate changes -
   only the mechanical quoting that makes the frozen intent executable. The
   original V2.4 bytes are preserved as the immutable parent (snapshot with
   digests recorded in RESULTS.md), the successor digests are recorded, and
   both digest sets are disclosed so any reviewer can audit the delta.
6. Reboot/preflight immediately before one real A_ONLY acquisition, and
   re-materialize the V2.6 inventory + phase lock first. [BLOCKED at stage 5
   of 10, `fresh_readiness`, after stages 1-4 passed repeatedly on real
   hardware. No paid marker exists and no model process has run.]
7. Stop on any identity, source, placement, correctness, memory, swap, or
   cleanup refusal.

No model execution is authorized by this mechanics checkpoint itself.

## Current acquisition frontier

The V2.4.1 quoting repair is proven on the real desktop and both phones:
`artifact_root`, `preparation`, `phase_lock`, and `identity_binding` pass.
The next qualification gate is still `fresh_readiness`. Two independent
post-reboot races currently prevent a clean snapshot:

- the frozen Android status probe can make `dumpsys thermalservice` report a
  broken pipe when its downstream `awk` exits after the first match;
- OP15 can first bind `wlan0` as `192.168.1.97`, then switch to
  `172.20.173.218` before the fresh snapshot.

Blind retries are paused after the bounded four-attempt loop exhausted. Before
the next acquisition:

1. keep OP15 on one Wi-Fi network across reboot;
2. obtain explicit human authorization before changing the contract-pinned
   embedded capture-common used by `artifact_root_capture_v1.py` and
   `fast_fresh_capture_v1.py`, if the thermal race is to be removed instead
   of retried;
3. regenerate topology, the V2.6 inventory/phase lock, and the V2.4.1 plan
   set together, then run one bounded acquisition;
4. if `fresh_readiness` and `readiness_projection` pass, allow the first paid
   `cuda_monolithic` stage. Any failure from that point remains terminal.
