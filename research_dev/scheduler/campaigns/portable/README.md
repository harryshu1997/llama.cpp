# The portable path: a new host or a new phone without a fork

One host with one llama-server, some of the model's layers on a GPU and the rest on the CPU, and helper
phones that each compute a column share of every CPU-resident FFN layer while the host computes the rest.
Everything specific to a machine lives in one site file and is read through drivers; nothing in this
directory names a device.

```
            scheduler (planner: placement and share, priced from the site profile)
 site.json -> discover, doctor      onboard-host, onboard-phone, provision -> PROFILE.json
              host driver           phone drivers (one interface)             meters (with a quality)
                       \___________________ run engine ___________________/
```

| Module | File | What it owns |
|---|---|---|
| Site | `configuration/site.py`, `drivers/discover.py` | What the machine is: facts read from it, and the few declared fields |
| Doctor | `drivers/doctor.py` | Every check that has cost a run, with the fix; also the gate before a run |
| Host driver | `drivers/host.py`, `drivers/gates.py` | Server launch on a GPU by index, slot erase, other users' load |
| Phone drivers | `drivers/phone.py`, `phone_adb_tcp.py`, `phone_functionfs.py` | One interface per phone; the transport stays inside the driver |
| Meters | `drivers/meter.py` | Host package + GPU board; a phone's power, `measured`, `assumed` or `absent` |
| Onboarding | `onboard.py`, `provision.py`, `_internal/shared_layer.py` | The measured profile, the share plan, the shards on the phone |
| Run engine | `engine.py`, `schedule.py`, `rundir.py`, `report.py` | Replays a schedule alone or shared; tables from run directories |

Import rule (tested): `configuration`, `_internal`, `adapters` < `drivers` < `campaigns/portable`.

All commands: `python3 -m research_dev.scheduler.campaigns.portable <command> --help`, run from the
repository root. Below, `M` stands for `python3 -m research_dev.scheduler.campaigns.portable`.

## A new host

```
M discover                                   # what is here: CPU, GPUs by index, RAPL zones, attached phones
M discover --init site.json --site-id ID --serving-gpu N --threads N --repo-root DIR \
           --server PATH/llama-server --adb PATH/adb --work-root DIR --model MODEL=PATH.gguf
M doctor --site site.json                    # fix what is blocking
M onboard-host --site site.json --model MODEL --gpu-layers N
```

`onboard-host` runs a few host-only requests and writes the host's record for that model into
`PROFILE.json` in the work root: time per token alone, loaded idle power, decode power, and whether the
host keeps its copy of the shared weights resident.

Declared by a person: which GPU serves, the thread count, the paths, and `--gpu-layers`. Everything
else is read from the machine.

## A new phone

```
M discover                                   # the phone appears with its serial and USB speed
# write declared-PHONE.json: serial, device_id, adb_port, transport, kind, backend, bundle_directory,
# work_directory, worker_port, forward_port, column_quantum, max_tokens, power
M discover --site site.json --add-phone declared-PHONE.json
M doctor --site site.json --phone PHONE --unmetered
M provision --site site.json --phone PHONE --model MODEL --probe [--bundle DIR]
M onboard-phone --site site.json --phone PHONE --model MODEL --binding .../bindings/PHONE-MODEL-sN-probe.json
M provision --site site.json --phone PHONE --model MODEL --from-profile
M make-plan --site site.json --model MODEL --port N --gpu-layers N --ctx N --parallel N \
            --binding .../bindings/PHONE-MODEL-sN.json --out PLAN.json
M run --site site.json --plan PLAN.json --schedule SCHEDULE.json --output RUN_DIR
M report seq RUN_DIR
```

What must exist on the phone: the worker bundle built for its backend (`llama-ffn-split-worker` and its
libraries; for a Snapdragon NPU the `libggml-htp-vNN.so` of its Hexagon version) in `bundle_directory`.
`provision --bundle DIR` copies one. Shards go under `work_directory`. Nothing is written elsewhere.

Transports:

| Transport | Needs | Cost per call | Use |
|---|---|---|---|
| `adb-tcp` | adb only; root optional | higher (measured at onboarding) | any phone, from the first day |
| `functionfs-bridge` | root and the qualified kernel | lowest | a phone prepared for the direct USB link |

`onboard-phone` sweeps the shares with host-only requests in between, fits

    layer time = max(c_host + (1 - f) t_host,  t_link + f t_phone)

and plans the column share for the memory the phone has. A result of `not-beneficial` (the link costs
more than the share saves) is a valid outcome; then nothing is provisioned.

A phone without readable power nodes (for example without root) declares
`"power": {"assumed_active_w": ..., "assumed_idle_w": ...}`; every report then marks its energy `assumed`.

## What the doctor checks

Blocking: no adb server on the port (none is ever started here), serial absent or unauthorized, the
serving GPU held by another job, unreadable CPU energy counter, missing worker or NPU library, an LLM
workload already running on the phone, the phone asleep or its CPU clocks capped, phone hot, ports in
use, the wrong kernel for the direct USB link. Degraded: no root, assumed power, a USB 2.0 link, an
uncalibrated gauge, other users' load on the host.

Never done by these tools: starting or stopping an adb server, rebooting or re-flashing a phone, booting
a kernel, changing a phone setting, touching another user's processes or another person's folders.

## On a shared host

A run takes the site lock, waits for a quiet host before every phase, cancels a phase that other users'
load disturbs and repeats it. The doctor does not touch a phone while a run holds the lock, and shard
slicing refuses to start.

## Run directories

The file names are those of the workstation harnesses of 2026-11-16/17 (`CONFIG.json`, `PHASE-<n>.json`,
`POWER_SAMPLES.json`, `phone-power.txt` ...), so `report` reads their recorded runs too, and
`onboard-host --ingest` / `onboard-phone --ingest` rebuild a profile from finished runs without machine
time.

## Not here yet

Run-time share feedback in the controller; the full multi-model scheduler on a site file (a generic
catalog builder); a driver for the Android accessory link; several sessions on one `adb-tcp` phone.
