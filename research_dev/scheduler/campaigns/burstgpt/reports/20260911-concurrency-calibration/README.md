# Post-reboot concurrency calibration readiness

Status: exact-parent startup preload is implemented and physically verified.
Nine canonical calibration requests established QUALIFIED CPU/GPU execution and
loading evidence. The unchanged dev3 gate completed all three inference requests
but FAILED its phone terminal-proof check. Useful concurrent execution is not
yet demonstrated. Earlier failed attempts remain preserved.

## Qualified calibration and exact-parent startup

Preflight-v4 passed 101 checks, with two advisory cost-profile warnings.
calibration-01 and -02 failed before inference because the standalone NCM
whole-phone route advertised control/links before FunctionFS had bootstrapped
its authenticated control connection. The rig now reports that availability
truthfully; it neither changes USB modes nor silently changes transports.
calibration-03 through -11 then completed nine scheduler-selected, isolated
Llama requests with physical server energy and terminal proofs.

| Qualified parent | Samples | Mean execution | Mean load | Mean execution fleet energy |
| --- | ---: | ---: | ---: | ---: |
| CPU, 0 GPU layers | 4 | 10.828 s | 0.683 s | 965.57 J |
| GPU, 16 GPU layers | 5 | 1.390 s | 0.791 s | 173.80 J |

These are independently qualified launch configurations, not a matched CPU/GPU
A/B. All 18 execution/load receipts have isolated server attribution. Phone
power is assumed, and no phone calls occurred in these calibration requests.
SUMMARY.json and QUALIFICATION_AUDIT.json preserve the actual estimates,
sample counts, bounds and source/binary/artifact identities. The fifth GPU
sample is retained, as is its slower first load; no outlier was deleted.

Audit finding not changed: at four GPU samples, transition_estimate() already
reported QUALIFIED using mean held-out error, while state().qualified_transitions
still rejected one individually large held-out error. After five samples both
report qualified. The counter/estimate inconsistency remains documented here.

The user authorized preloading the exact qualified CPU parent at startup while
keeping the legacy CPU fallback available. startup_desktop_parents configures
an executor and exact desktop placement hash. The scheduler generates and
admits a bounded one-input/two-output verification ticket using existing
physical qualification, capacity-wide load leases, memory and receipt checks.
This is explicit initial residency, not a forced route for any trace request.
No placement epoch is published from the startup selection. Replan, fallback,
eviction or mismatched physical proof aborts the preload rather than substituting
another parent. The physical READY endpoint remains managed until cleanup.
Startup tickets and proofs are exported separately from the three workload
results; all startup load/verification energy remains inside the paid interval.

Focused validation: 54 tests passed (52 focused plus the two unchanged replay
goldens), followed by the new exact physical-proof and runner-boundary cases.
The latter caught and prevents confusing the client's accounting-only mode with
the scheduler's exact quality requirement. dev3-ready-parent-v1 preserves that
failure before any trace request. The corrected retry uses a fresh v2 directory.
There are 56 unique passing tests in STARTUP_TESTS.json, including both replays.
No native binary, shard, model artifact, baseline or historical result changed.

Replay goldens remain:

- v3: 5d52e8673fc60f974ca167964b3ad579e59abea28feb3d66357cf39c1ca7caf4
- v8: 241924463ac27ead0c2d7bcb5da214e85097049c91d5800848e0b29effd2b917

## Exact-parent physical result and dev3 blockers

The fresh dev3-ready-parent-v2 artifact proves startup loaded and verified
physical:cpu-parent:4b90b1d7ae732467 at http://127.0.0.1:18684, generation 2.
Its desktop placement is
sha256:c1ccad2708f47f9e9326f57f184d09b386b78c2a78fdc1a8c73c86e31584d895.
The legacy fallback at port 18484 remains a separate endpoint; its identity is
not substituted. No CPU-parent reload occurred during the workload.

| Exact-parent startup phase | Duration | Fleet energy |
| --- | ---: | ---: |
| Physical load receipt | 0.688205 s | 27.889316 J |
| Verification execution receipt | 0.070959 s | 2.004733 J |
| Complete exact-parent preparation interval | 1.590346 s | 47.691581 J |

The final row spans startup invocation to physical READY verification, including
scheduler and measurement gaps outside the two receipt windows. CPU package
energy is 34.557771 J and GPU board energy is 11.742257 J, physically measured.
Phone idle energy is explicitly assumed: 0.875 W for 1.590346 s, or 1.391553 J.
These startup costs are inside the paid accounting boundary. The complete
interval was integrated after failure from preserved power samples using the
existing server-energy integration; it is not a synthesized successful RESULT.
STARTUP_PHYSICAL_AUDIT.json records the calculation and monotonic timestamps.

Normal energy-aware scheduling was used for the unchanged three requests:

| Request | Execution duration | Arrival to execution | Arrival to completion | Phone inference calls |
| --- | ---: | ---: | ---: | ---: |
| Gemma 36, 292 tokens | 139.017 s | 9.823 s | 148.840 s | 0 |
| Llama 37, 292 tokens | 2.028 s | 90.635 s | 92.662 s | 0 |
| Qwen 50, 71 tokens | 47.960 s | 133.548 s | 181.508 s | 0 |

The full decision journal validates: four DECISION/ACQUIRED/COMPLETED sequences
(one startup plus three workload requests), two replans and no physical FALLBACK
event. Request execution overlap is zero. Protected-work slowdown is unmeasured,
not a measured zero. No useful-concurrency or matched-savings claim is made.

Two remaining causes are recorded, not bypassed:

1. Llama's exact READY CPU route is admitted with zero incremental allocation.
   Live selection prefers it, but the model-placement epoch proposes the GPU
   route. Both resolution passes return EPOCH_COMPONENT_MISMATCH. The existing
   DESKTOP_FALLBACK_AFTER_NONCONVERGENCE path then uses desktop-baseline selection,
   and Llama waits 90.635 s for GPU execution. The startup/qualification fix has
   exposed an epoch/live-selection disagreement, not a CPU memory rejection.
2. Gemma phone preparation starts at paid time 4.443 s and reaches READY at
   24.912 s. Rematerialization fails 23 times with no helper opportunity; candidate
   rejection reasons include MARGINAL_SYSTEM_COST_UNKNOWN,
   COLD_RESIDENCY_BREAK_EVEN and ROUTE_NOT_QUALIFIED. Another preparation completes
   at 159.992 s after its original owner finishes. There are no ATTACHED or
   FRACTION_APPLIED events; all 90 assistance decisions report
   PHONE_HELPER_UNAVAILABLE. The existing required-phone terminal check fails
   with `phone session has no ticket-bound execution proof`. It was not weakened.

Recovered worker/router logs independently confirm HTP0 and HTP1 loaded once
each, both at session generation 1, with zero router inference calls. Their
phone-local LOAD_AUTHORIZED-to-READY times are 10.482959 s and 9.485248 s.
The scheduler preparation intervals include additional transport and verification
work and must not be confused with these worker-only intervals. Retained HTP0
was not reloaded when HTP1 was added. The router's native terminal status is 0,
but that does not provide the missing ticket-bound request proof.

Cleanup restored ptp,adb and released the campaign's host inference processes.
NVML reports 299 MiB used, with GDM untouched. A sleeping owned watchdog shell
(PID 18985, sleep child 18987) remains until its original timeout; its active
marker is absent, so its guarded USB restore will not run. It was not signalled.
The readable phone-native logs were copied to recovered-phone-native/ without
changing phone permissions; root-owned descriptors.ready could not be read.

Artifact locations and integrity:

- Remote: /mnt/storage/s42-phase-concurrency-20260911-v1-nEEi6I/dev3-ready-parent-v2
- Local: physical/dev3-ready-parent-v2/ (including failure journal and raw logs).
- Startup failure before trace: physical/dev3-ready-parent-v1/.
- STARTUP_ARTIFACTS.json: hashes of 24 preserved evidence files, validated journal,
  native session timings and all 13 changed source/test identities.
- STARTUP_CHANGES.json: exact startup before/after file hashes; before-images
  remain under startup-source-before/.

The remote git head is 99449bafade0b2c15de4410feda832035c2f2d83, while the local
working-tree base is 5f89a2d9d33be547a1bdef5fd0f504a279c50800. The source manifest,
not either base revision alone, identifies deployed dirty-tree code. Each listed
startup source present in that manifest matches its current local hash. Its
manifest identity is
sha256:a6830cdd07f45ab5f9f16e1b8d303fd884e27bbb741d9bcd85a094388bee4cbb.
No baseline was rerun or compared as a matched control. No long trace, additional
gate, commit or push followed this failure.

Startup implementation changes, relative to startup-source-before/:

- configuration/campaign.py and config.py: exact-parent initial-residency config.
- campaigns/burstgpt/arguments.py and launch.py: explicit configuration transport.
- _unified/automated_requests.py and automated_requests_ops/submission.py:
  existing scheduler-owned startup verification/admission path, without an epoch.
- _internal/route_generation/compiler.py: honor explicit exact-parent generation.
- adapters/runtime.py and heterogeneous_rig.py: no startup substitution; verify
  live managed residency against the exact terminal execution proof.
- campaigns/burstgpt/runner.py: invoke preload and persist separate paid receipts.
- tests/test_work_conserving_start.py, test_physical_adapter.py and
  test_campaign_inputs.py: startup identity, admission, proof and config cases.

## Post-permission validation and exact-parent preflight

ENERGY_READY.json records fresh NVML samples and two advancing physical RAPL
readings at 2026-09-11 15:47:29 UTC. CPU energy is measured, not assumed. GDM and
other processes were left alone.

Preflight-v3 reached all 103 checks. Four blocked checks have one cause: with
more VRAM free, preflight selected an unqualified larger GPU suffix instead of
validating the frozen qualified parent. Both the previous and reconstructed
catalogs contain identical Gemma and Qwen parent parameters and evidence.
The frozen parents have 23 and 16 GPU layers; the unconstrained capacity search
proposed 29 and 20. The two shadow cost-profile warnings are advisory, not new
physical qualification failures.

Changed adapters/preflight.py to use the existing preserve_placement option.
This validates the exact qualified parent against live memory without changing
placement, qualification, graph mode or capacity requirements. Added a regression
with excess VRAM and a separate insufficient-capacity branch that still fails
closed. All 17 tests in test_physical_preflight, test_desktop_parent_capacity
and test_cuda_graph_mode pass. No replay golden was changed. Before-images are
in source-before/ and failed preflight-v3 is preserved under physical/.

Preflight-v4 uses this narrow repair in the same fresh deployment. Its result
and subsequent calibration evidence will be recorded below.

The user rebooted the RTX 4060 Ti desktop at 172.20.74.85. It now runs kernel
7.0.0-31-generic; both NVIDIA kernel driver and NVML are 595.91.07. The GPU probe
returns its exact expected UUID, live memory and board power. At 15:43:27 UTC it
reported 16,409,165,824 free bytes and 313,524,224 used bytes. OP15 is reachable
with the expected 6.12.23-android16-5-o-g227664cbe007-4k kernel. GDM is untouched.
See RECOVERED_HARDWARE.json.

## Deployment and narrow correction

Fresh remote directory:
`/mnt/storage/s42-phase-concurrency-20260911-v1-nEEi6I`.
The existing scheduler was synced into deploy/; native binaries, model artifacts
and phone shards are reused without modification. No historical result was
overwritten. Inputs register the independently proven CPU launch through normal
models.json, using its correctness/launch evidence, not the former single-point
energy profile. New energy qualification still requires real attributable
execution and transition receipts. Calibration is strict; the planned dev3
configuration explicitly selects the existing energy-budgeted policy.

The first canonical catalog materialization exposed a CLI defect: catalog.py
used a package-relative import in _register_overlay_whole_phone, but launch.py
invokes it as a script. Replaced that import with the same absolute import style
used by the rest of the script. Added a regression executing the catalog module
without package context and registering the NCM whole-phone contract. No route,
fraction, memory, evidence or phone-session policy changed.

Changed production/test files:

- campaigns/burstgpt/catalog.py
- tests/test_campaign_inputs.py

Before-images are in source-before/. All six test_campaign_inputs tests pass.
The existing scheduler fixes previously passed 457 affected tests and both
replay goldens; those broader tests were not rerun for this import-only fix.
No golden was edited.

## Preserved attempts and former permission block

- An initial invocation used unsupported named CLI arguments and exited before
  creating artifacts; the launcher takes positional campaign/output arguments.
- preflight-v1: catalog CLI import failed before physical inference. Its source
  and resolved configuration are preserved.
- preflight-v2: catalog generation succeeded (SHA-256
  2128811ce9cf29bce558fb70842e2bdfe3102938888829f7a236ea8f5e846555).
  The single-Llama calibration schedule fails the preflight tool's existing
  three-model coverage requirement. The fallback diagnostic report independently
  confirms CPU energy permission failure. A separate preflight-dev3.json now
  targets the unchanged three-model workload for the next preflight; the
  single-request calibration configuration remains separate and preserved.

The physical CPU counter /sys/class/powercap/intel-rapl:0/energy_uj is mode 0400,
root:root, with no ACL for zhihao. The campaign's exact rapl_package_snapshot()
raises PermissionError. /dev/cpu/0/msr is also root-only; no existing RAPL/powercap
grant service was found in the checked systemd/udev locations. The CPU counter
is not replaced by an assumed power value. The preflight's 27 diagnostic checks
are not a complete physical PASS.

An administrator must restore read access to that one CPU package energy
counter. Then run the three-model preflight, canonical bounded calibration,
verify actual QUALIFIED execution and load evidence separately, and only then
the unchanged dev3 gate from the exact scheduler-managed READY CPU endpoint.
READY CPU startup plumbing must still be verified: the existing runner preloads
the fallback CPU endpoint, not automatically the separately registered CPU
parent. No identity substitution or forced selection is authorized.

No new overlap, protected slowdown, qualification or savings result is claimed.
No baseline rerun or 24/84-request trace.

## Authorized permission repair attempt

After the user approved granting read access, attempted only
`sudo -n chgrp zhihao /sys/class/powercap/intel-rapl:0/energy_uj`.
Sudo rejected it with `interactive authentication is required`. The command
sequence stopped before chmod; permissions remain root:root 0400. No service
was stopped and no physical inference started. The user must authenticate in
their own terminal to set group zhihao and mode 0440 on this single counter;
no password is requested through the conversation.
