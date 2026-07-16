# S10-V0-R-E2 instrument capability audit

Date: 2026-07-15. Host: the live A6000 server. Method: capability inspection only,
no sustained measurement, no experiment.

## Verdict

**No SERVER_WALL instrument exists on this host.** Therefore
`SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED` is currently UNREACHABLE, not merely
unmeasured. The only physically collectable E2 boundary today is `GPU_BOARD`.
Even that boundary remains diagnostic until the all-pairs aggregate exists; its
strongest future label is `GPU_BOARD_RELIEF_ONLY_TOTAL_ENERGY_BLOCKED`.

There is a second, contract-level wall blocker: `ServerWallCapability` v1 binds a
hashed declaration, not topology, calibration validity, or raw acquisition
evidence. The validator rejects every `MEASURED` v1 capability with
`E_CAPABILITY_UNCERTIFIED`. Installing a meter would require a later evidence
version as well as the hardware.

`TOTAL_WALL` and `SYSTEM_ENERGY_SAVING` are out of scope for E2 by construction and
are not reachable by any instrument found here.

## 1. NVML / nvidia-smi -> GPU_BOARD only

Present at `/usr/bin/nvidia-smi`, driver `580.159.03`. Power management Enabled,
power limit 300.00 W, `power.draw` readable unprivileged.

**This host has TWO A6000 boards.** A `GPU_BOARD` scope is therefore ambiguous
unless it names which board:

| index | UUID | serial | PCI bus |
|---|---|---|---|
| 0 | `GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f` | 1715023010133 | `00000000:51:00.0` |
| 1 | `GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf` | 1714923004268 | `00000000:9C:00.0` |

E2 consequently requires `board_uuids` in every `GPU_BOARD` timeline and requires
control and treatment to name the SAME set. There is no separate GPU-board
capability record in v1. Comparing board 0 against board 1, or a one-board timeline
against a two-board timeline, is `E_SCOPE_MISMATCH`.

### Cadence and accuracy, quoted from the vendor

`nvidia-smi --help-query-gpu` states for `power.draw`:

> The last measured power draw for the entire board, in watts. On Ampere or newer
> devices, returns average power draw over 1 sec. ... This reading is accurate to
> within +/- 5 watts.

The A6000 is Ampere (GA102). So `power.draw`:

- is **the entire board** - never the host, never the wall;
- is a **1-second average**, not an instantaneous sample; and
- carries a **+/- 5 W** stated accuracy, which is the floor for
  `uncertainty_mw` on any NVML timeline (5000 mW).

This is confirmed first-hand in the only existing trace
(`../s10_power_frontier/artifacts/cp2_a6000_power_trace.csv`), re-measured for this
audit:

```
323 rows, span 32.26 s
poll rate            10.01 Hz   (the sampler asked 10x/s)
distinct value CHANGES 57       -> observed change rate 1.77 Hz
busy window (util>50) 23.84 s, 39 value changes -> 1.64 Hz
max poll gap 106.0 ms, median 100.0 ms
```

The sampler polled faster than the reported value changed. **322 samples contain
57 value changes.** A change count is a conservative quality gate, not proof of
statistical independence or the sensor's update cadence. Oversampling does not
create information, so E2 does not use raw row count as its evidence gate.

`power.draw.instant` IS exposed on this board (observed 25.12 W vs 25.93 W average),
and is a candidate for a higher-cadence integrator. **Its true update cadence is
UNMEASURED** - measuring it needs a sustained run, which this checkpoint forbids.
It is recorded as an opportunity, not a capability.

Classification: `NVML -> GPU_BOARD`. It can never be promoted. A GPU board sensor
cannot see the CPU, DRAM, PSU losses, fans, NIC, drives, or anything else.

## 2. RAPL / powercap -> unusable twice over

```
/sys/class/powercap/intel-rapl:0      name=package-0   energy_uj -> Permission denied
/sys/class/powercap/intel-rapl:0:0    name=core        energy_uj -> Permission denied
-r-------- 1 root root /sys/class/powercap/intel-rapl:0/energy_uj
```

CPU is an **AMD Ryzen Threadripper PRO 5995WX** (`AuthenticAMD`); the `intel-rapl`
powercap name is the generic RAPL interface the driver exposes for AMD too.

Two independent disqualifications:

1. **Permission.** `energy_uj` is `-r--------` root-only and `sudo` requires a
   password. No energy counter under `/sys/class/powercap` is readable by this user.
   (Root-only RAPL is the standard mitigation for the PLATYPUS side channel.)
2. **Coverage.** Only `package-0` and `core` domains exist. There is **no `dram`
   domain and no `psys`/platform domain.** Even with permission, package+core
   energy excludes DRAM, PSU conversion losses, fans, storage, NIC, and the second
   CPU rail path - it is a component counter, not a server wall.

Classification: `RAPL -> COMPONENT_PACKAGE`. E2 forbids promoting it to
`SERVER_WALL` under any name. Note that `psys`, where it exists on some Intel
parts, is still a platform-domain estimate and is not automatically a wall meter
either; the contract requires demonstrated complete coverage, not a domain name.

## 3. External / wall / BMC instruments -> none

- `ipmitool`, `ipmi-sensors`, `freeipmi`, `racadm`: **not installed**.
- `/dev/ipmi*`: **does not exist**. No BMC path.
- hwmon sensors present: `nvme`, `nvme`, `enp2s0`, `k10temp`, `dell_smm`. None is a
  PSU or wall meter. (`dell_smm`'s `power` entry is the sysfs runtime-PM directory,
  not a power sensor - a name match, not a capability.)
- No external meter (no PDU, clamp, or inline wattmeter) is present or referenced by
  any artifact on disk.

Classification: **no SERVER_WALL instrument exists.** A self-authored
`ServerWallCapability` v1 record would not change that finding: its hashed proof
would still be a declaration, and measured use is fail-closed.

## 4. USB / VBUS coverage -> none

No instrument covers the USB supply feeding the phones. This is unchanged from E1's
finding: the phone-side USB rail is pinned at ~497/500 mA and the battery coulomb
counter is dead while charging. Phone, USB supply, charger, and relay energy remain
`UNKNOWN`, never zero. E2 does not attempt to measure them and cannot claim
anything about them.

## 5. Clock and markers -> adequate

```
CLOCK_MONOTONIC resolution 1e-09 s, monotonic=True
time.monotonic_ns() available
perf_counter resolution 1e-09 s
```

A monotonic integer-microsecond timebase is available, so start/end markers can be
bound to the normalized timeline exactly. This is the one capability that is not a
blocker.

Caveat: a monotonic clock is per-boot and NOT comparable across hosts or reboots.
E2 requires both timelines of a pair to carry the same `clock_epoch_id`, so markers
from different boots cannot be compared.

## 6. Frozen classification table

| instrument | observed | classification | strongest label it can ever support |
|---|---|---|---|
| NVML `power.draw` | present, per-board, 1 s average, +/-5 W, 57 changes in 32.26 s | `GPU_BOARD` | `GPU_BOARD_RELIEF_ONLY_TOTAL_ENERGY_BLOCKED` |
| NVML `power.draw.instant` | present, cadence UNMEASURED | `GPU_BOARD` | same (not usable until cadence is measured) |
| RAPL `package-0`/`core` | present, root-only, no dram, no psys | `COMPONENT_PACKAGE` | nothing; never `SERVER_WALL` |
| IPMI / BMC / PDU / clamp | absent | `NONE` | nothing |
| USB / VBUS | absent | `NONE` | nothing; phone energy stays `UNKNOWN` |

Rule frozen in CONTRACT.md: **scope is typed, not named.** A timeline cannot
become `SERVER_WALL` by writing "server wall" in a free-form instrument string. The
validator maps a closed `instrument_kind` enum to an allowed scope and refuses any
mismatch (`E_SCOPE`). A measured wall scope additionally needs a later,
acquisition-backed capability version; v1 is mechanics-only.

## 7. What would unblock a real measurement

Concrete, in order of how much they buy:

1. **A GPU_BOARD A/B can be collected TODAY**, with two caveats that must be
   designed around rather than argued away: the observed 1.64-1.77 Hz value-change
   rate means a run needs roughly 60 s to collect 100 changes under similar
   behavior, and the +/-5 W accuracy must enter the decision as
   `uncertainty_mw`, not be dropped. `power.draw.instant`'s cadence should be
   measured first; it may reduce the required duration. Collection alone does not
   authorize a label; the all-pairs aggregate and pre-run attempt ledger are still
   absent.
2. **SERVER_WALL needs new hardware and a stronger evidence contract.** An inline
   wall meter or a BMC/PDU with readable input power is required. Nothing on this
   host can be configured into one, and unprivileged RAPL access would still not
   be a server wall. A later capability version must bind topology, calibration,
   validity, and raw acquisition artifacts rather than a declaration alone.
3. **TOTAL_WALL additionally needs the phone/USB/charger boundary**, which is the
   same blocker E1 recorded. Out of scope for E2 regardless.

After the all-pairs aggregate and attempt ledger exist, the honest ceiling with
this host's present instrumentation is
`GPU_BOARD_RELIEF_ONLY_TOTAL_ENERGY_BLOCKED`.
