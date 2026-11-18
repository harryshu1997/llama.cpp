"""Command line of the portable path.

    python3 -m research_dev.scheduler.campaigns.portable discover [--site SITE.json [--write]]
    python3 -m research_dev.scheduler.campaigns.portable discover --init SITE.json --site-id ID --serving-gpu N
            --threads N --repo-root DIR --server PATH --adb PATH --work-root DIR --model ID=PATH [--tool NAME=PATH]
    python3 -m research_dev.scheduler.campaigns.portable discover --site SITE.json --add-phone DECLARED.json
    python3 -m research_dev.scheduler.campaigns.portable doctor --site SITE.json [--phone ID ...] [--unmetered]
    python3 -m research_dev.scheduler.campaigns.portable onboard-host --site SITE.json --model ID --gpu-layers N
            [--port N] [--legs 3] [--output-tokens 128] [--artifact SHA] [--ingest RUN_DIR]
    python3 -m research_dev.scheduler.campaigns.portable provision --site SITE.json --phone ID --model ID
            (--probe [--layers 4] [--share 0.5] | --from-profile) [--bundle DIR]
    python3 -m research_dev.scheduler.campaigns.portable onboard-phone --site SITE.json --phone ID --model ID
            [--binding BINDING.json] [--port N] [--usable-gb X] [--stored-columns N] [--ingest RUN_DIR ...]
    python3 -m research_dev.scheduler.campaigns.portable adopt --site SITE.json --phone ID --model ID
            --index FFN_SHARDS.json --remote-dir PHONE_DIR [--out BINDING.json]
    python3 -m research_dev.scheduler.campaigns.portable make-plan --site SITE.json --model ID --port N
            --gpu-layers N --ctx N --parallel N [--binding BINDING.json | --n-embd N --n-ff N --artifact SHA]
            [--host-share resident|dormant] [--max-tokens N] --out PLAN.json
    python3 -m research_dev.scheduler.campaigns.portable run --site SITE.json --plan PLAN.json
            --schedule SCHEDULE.json --output RUN_DIR [--alone] [--dry-run] [--unmetered] [--min-window-s S]
    python3 -m research_dev.scheduler.campaigns.portable report {legs,sweep,rows,seq,trace} RUN ...

``discover`` without a site prints the host's facts and the phones attached to it (host-side facts only;
no command runs on a phone that the site does not name). With ``--site`` it refreshes the facts of the
site and its phones and lists what changed; ``--write`` saves them. ``--add-phone`` takes a JSON file of
the declared fields of one phone (serial, device id, transport, backend, directories, ports ...) and
adds its discovered facts. ``doctor`` exits 1 when a check is blocking.

``onboard-host`` measures the host with one model and ``onboard-phone`` sweeps the shares of one phone;
both write the site profile (``PROFILE.json`` in the work root) and both can ``--ingest`` finished run
directories instead of running. ``provision --probe`` puts a small probe shard on a phone for its sweep;
``provision --from-profile`` puts the shards of the profile's share plan there. ``--bundle DIR`` first
copies a worker bundle built for the phone.

``adopt`` writes a helper binding for shards that are already on a phone (read-only on the phone).
``make-plan`` writes what a run serves; ``run`` gates on the doctor (exit 2 when a check is blocking),
then replays the schedule. ``--alone`` runs the same plan on the host alone and does not touch the phone;
``--dry-run`` prints the server command, its environment and the phone-side launch and starts nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from ..._internal.shared_layer import PhoneCapacity, StaticSharePlan, plan_static_share
from ...configuration.site import SiteManifest
from ...configuration.site_profile import SiteProfile
from ...drivers import discover as discovery
from ...drivers import doctor as checks
from ...drivers import phone_adb_tcp, phone_functionfs  # noqa: F401  (register the phone drivers)
from ...drivers.adb import AdbGuard, AdbUnavailable
from ...drivers.gates import gpu_free, wait_until
from ...drivers.host import HostLaunchPolicy, ServerLoadFailure
from ...drivers.meter import PhonePowerCalibration
from ...drivers.phone import read_phone_caps_floor, read_phone_health
from . import engine, onboard, provision, report, schedule
from .rundir import RunDirectory


def _pairs(values: Sequence[str], name: str) -> dict[str, str]:
    out = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or not key or not path:
            raise SystemExit(f"{name} takes ID=PATH, got {value!r}")
        out[key] = path
    return out


def _save(path: Path, document: dict) -> None:
    SiteManifest.from_json(document, base=path.resolve().parent)        # never write a site that does not load
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="ascii")


def _discover(arguments: argparse.Namespace) -> int:
    host = discovery.host_facts()
    if arguments.init is not None:
        if arguments.init.exists():
            raise SystemExit(f"{arguments.init} exists; refresh it with --site")
        for name in ("site_id", "serving_gpu", "threads", "repo_root", "server", "adb", "work_root"):
            if getattr(arguments, name) is None:
                raise SystemExit("--init needs --" + name.replace("_", "-"))
        paths = {"repo_root": arguments.repo_root, "server_path": arguments.server, "adb_path": arguments.adb,
                 "work_root": arguments.work_root, "models": _pairs(arguments.model, "--model")}
        if arguments.tool:
            paths["host_tools"] = _pairs(arguments.tool, "--tool")
        document = discovery.new_site_document(arguments.site_id, host, serving_gpu_index=arguments.serving_gpu,
                                               threads=arguments.threads, paths=paths,
                                               cpu_affinity=arguments.cpu_affinity)
        _save(arguments.init, document)
        print(f"wrote {arguments.init}: host {host['cpu_model']}, serving GPU {arguments.serving_gpu}, no phone yet")
        return 0
    if arguments.site is None:
        print(json.dumps({"host": host}, indent=2))
        for port in arguments.adb_port:
            adb = AdbGuard(Path(arguments.adb or "adb"), port)
            try:
                print(json.dumps({"adb_port": port, "attached": discovery.attached_phones(adb)}, indent=2))
            except AdbUnavailable as error:
                print(f"adb port {port}: {error}")
        return 0
    before = json.loads(arguments.site.read_text(encoding="ascii"))
    site = SiteManifest.from_json(before, base=arguments.site.resolve().parent)
    if arguments.add_phone is not None:
        declared = json.loads(arguments.add_phone.read_text(encoding="ascii"))
        adb = AdbGuard(site.paths.adb_path, int(declared["adb_port"]))
        after = discovery.add_phone(before, discovery.phone_facts(adb, declared["serial"]), declared)
        _save(arguments.site, after)
        print(f"added phone {declared['device_id']} ({declared['serial']}) to {arguments.site}")
        return 0
    facts = {}
    for phone in site.phones:
        try:
            facts[phone.serial] = discovery.phone_facts(AdbGuard(site.paths.adb_path, phone.adb_port), phone.serial)
        except Exception as error:  # noqa: BLE001 - a phone that is away keeps its recorded facts
            print(f"phone {phone.device_id}: facts not refreshed ({str(error)[:120]})")
    after = discovery.refresh(before, host, facts)
    moved = discovery.changes(before, after)
    print("\n".join(moved) if moved else "site facts are current")
    if moved and arguments.write:
        _save(arguments.site, after)
        print("wrote", arguments.site)
    elif moved:
        print("(not saved; add --write)")
    return 0


def _doctor(arguments: argparse.Namespace) -> int:
    site = SiteManifest.load(arguments.site)
    rows = checks.site_checks(site, phones=arguments.phone or None, metered=not arguments.unmetered,
                              required_caps_by_phone=_recorded_caps(site))
    print(checks.format_checks(rows))
    stop = checks.blocking(rows)
    print(f"\n{len(stop)} blocking, {sum(1 for row in rows if row.status == 'degraded')} degraded, "
          f"{sum(1 for row in rows if row.status == 'ok')} ok")
    if arguments.json is not None:
        arguments.json.write_text(json.dumps([row.to_json() for row in rows], indent=2) + "\n", encoding="ascii")
    return 1 if stop else 0


def _adopt(arguments: argparse.Namespace) -> int:
    site = SiteManifest.load(arguments.site)
    document = provision.adopt(site, arguments.phone, arguments.model, arguments.index, arguments.remote_dir)
    binding = document["binding"]
    out = arguments.out or provision.binding_path(site, arguments.phone, arguments.model,
                                                  binding["shards"][0]["columns"])
    provision.save_binding(out, document)
    print(f"wrote {out}: {len(binding['shards'])} shards of {binding['shards'][0]['columns']} columns, "
          f"{sum(len(row['layers']) for row in binding['shards'])} layers, {len(binding['pins'])} bundle files pinned")
    return 0


def _recorded_caps(site: SiteManifest) -> dict[str, dict[str, int]]:
    """Per phone, the CPU clock caps recorded in the site profile when it was onboarded."""
    path = onboard.profile_path(site)
    if not path.exists():
        return {}
    profile = SiteProfile.load(path, site.site_id)
    return {name: dict(row["cpu_caps_khz"]) for name, row in profile.phones.items() if row.get("cpu_caps_khz")}


def _gate(site: SiteManifest, phones: Sequence[str], *, metered: bool,
          caps: dict[str, dict[str, int]] | None = None) -> bool:
    rows = checks.site_checks(site, phones=list(phones), metered=metered,
                              required_caps_by_phone=_recorded_caps(site) if caps is None else caps)
    stop = checks.blocking(rows)
    if stop:
        print(checks.format_checks(stop))
        print("\nnot started:", len(stop), "blocking")
    return not stop


def _onboard_host(arguments: argparse.Namespace) -> int:
    site = SiteManifest.load(arguments.site)
    profile = SiteProfile.load(onboard.profile_path(site), site.site_id)
    if arguments.ingest is not None:
        run = RunDirectory(arguments.ingest)
        available = None if arguments.available_gb is None else int(arguments.available_gb * 1e9)
    else:
        geometry = onboard.model_geometry(site.paths.models[arguments.model], arguments.gpu_layers)
        available = onboard.memory_available_bytes()
        if not _gate(site, [], metered=True):
            return 2
        policy = HostLaunchPolicy(gpu_layers=arguments.gpu_layers, ctx_size=4096, parallel=1)
        plan = engine.RunPlan(arguments.model, arguments.model, arguments.port, policy, geometry["n_embd"],
                              geometry["n_ff"], arguments.artifact or "sha256:" + "0" * 64)
        output = site.paths.work_root / f"run-onboard-host-{arguments.model}-{profile.revision}"
        engine.run(site, plan, schedule.host_baseline(f"host-{arguments.model}", schedule.PROBE_TEXT,
                                                      legs=arguments.legs, output_tokens=arguments.output_tokens),
                   output)
        run = RunDirectory(output)
    record = onboard.ingest_host(site, profile, arguments.model, run, gpu_layers=arguments.gpu_layers,
                                 artifact_sha256=arguments.artifact, memory_available=available)
    profile.save(onboard.profile_path(site))
    print(f"host {site.site_id} with {arguments.model}: {record['token_ms_alone']:.1f} ms per token alone, "
          f"{len(record['cpu_layers'])} CPU layers, idle {record['p_idle_w']:.0f} W, decode {record['p_decode_w']:.0f} W, "
          f"host weights {record['host_share']}; wrote {onboard.profile_path(site)}")
    return 0


def _onboard_phone(arguments: argparse.Namespace) -> int:
    site = SiteManifest.load(arguments.site)
    phone = site.phone(arguments.phone)
    profile = SiteProfile.load(onboard.profile_path(site), site.site_id)
    host = profile.host_model(arguments.model)
    health, floor = None, None
    if not arguments.ingest or arguments.usable_gb is None:
        health = read_phone_health(AdbGuard(site.paths.adb_path, phone.adb_port), phone)
    if arguments.usable_gb is not None:
        capacity = PhoneCapacity(int(arguments.usable_gb * 1e9), phone.sessions, phone.session_limit_bytes)
    else:
        if health.memory_available_bytes is None:
            raise SystemExit("the phone did not report its available memory; pass --usable-gb")
        capacity = onboard.phone_capacity(site, arguments.phone, health.memory_available_bytes)
    if arguments.ingest:
        # a run that recorded no power calibration (an earlier harness) is read with the site's
        power = phone.power
        fallback = None if power is None or not power.measured else PhonePowerCalibration(
            power.battery_current_unit, power.battery_discharge_sign, power.gauge_ratio, power.gauge_ratio_provenance)
        runs = [RunDirectory(path, None if (path / "PHONE_POWER.json").exists() else fallback)
                for path in arguments.ingest]
    else:
        if arguments.binding is None:
            raise SystemExit("onboard-phone needs --binding (provision --probe writes one) or --ingest")
        binding, document = provision.load_binding(arguments.binding)
        # the phone is measured as it is now: its present clock caps are recorded and become the bar for runs
        floor = read_phone_caps_floor(AdbGuard(site.paths.adb_path, phone.adb_port), phone)
        if not _gate(site, [arguments.phone], metered=True, caps={arguments.phone: floor}):
            return 2
        policy = HostLaunchPolicy(gpu_layers=int(host["gpu_layers"]), ctx_size=4096, parallel=1,
                                  host_share=str(host["host_share"]), max_tokens=phone.max_tokens)
        plan = engine.RunPlan(arguments.model, arguments.model, arguments.port, policy, binding.n_embd, binding.n_ff,
                              binding.artifact_sha256, arguments.phone, binding)
        columns = onboard.sweep_columns(binding.stored_columns, phone.column_quantum)
        output = site.paths.work_root / f"run-onboard-{arguments.phone}-{arguments.model}-{profile.revision}"
        engine.run(site, plan, schedule.sweep(f"sweep-{arguments.phone}", schedule.PROBE_TEXT, columns), output)
        runs = [RunDirectory(output)]
    pair = onboard.ingest_phone(site, profile, arguments.phone, arguments.model, runs, capacity=capacity,
                                health=health, stored_columns=arguments.stored_columns, caps_floor=floor)
    profile.save(onboard.profile_path(site))
    print(onboard.format_pair(pair, profile.host_model(arguments.model)))
    print("wrote", onboard.profile_path(site))
    return 0


def _provision(arguments: argparse.Namespace) -> int:
    site = SiteManifest.load(arguments.site)
    phone = site.phone(arguments.phone)
    profile = SiteProfile.load(onboard.profile_path(site), site.site_id)
    host = profile.host_model(arguments.model)
    if not _gate(site, [arguments.phone], metered=False):
        return 2
    if arguments.bundle is not None:
        pushed = provision.push_bundle(site, arguments.phone, arguments.bundle)
        print("bundle:", sum(1 for state in pushed.values() if state == "pushed"), "pushed,",
              sum(1 for state in pushed.values() if state == "present"), "already there")
    if arguments.probe:
        columns = int(host["n_ff"] * arguments.share) // phone.column_quantum * phone.column_quantum
        layers = tuple(host["cpu_layers"][:arguments.layers])
        specs = [f"P0={layers[0]}-{layers[-1]}:{columns}"]
        tag = f"probe-s{columns}"
    else:
        plan = StaticSharePlan.from_json(profile.pair(arguments.phone, arguments.model)["share_plan"])
        specs = provision.shard_specs(plan)
        tag = f"s{plan.columns}"
    local = site.paths.work_root / "shards" / f"{arguments.model}-{arguments.phone}-{tag}"
    remote = f"{phone.work_directory}/{arguments.model}-{tag}"
    index = provision.slice_shards(site, arguments.model, str(host["artifact_sha256"]), specs, local)
    pushed = provision.push_shards(site, arguments.phone, index, local, remote)
    document = provision.adopt(site, arguments.phone, arguments.model, local / provision.SHARD_INDEX_NAME, remote)
    out = provision.binding_path(site, arguments.phone, arguments.model, document["binding"]["shards"][0]["columns"])
    if arguments.probe:
        out = out.with_name(out.stem + "-probe.json")
    provision.save_binding(out, document)
    print(f"shards {specs} -> {remote}: {pushed}; wrote {out}")
    return 0


def _make_plan(arguments: argparse.Namespace) -> int:
    site = SiteManifest.load(arguments.site)
    binding, device = None, None
    if arguments.binding is not None:
        binding, document = provision.load_binding(arguments.binding)
        device = document["device_id"]
        if document["model_id"] != arguments.model or document["site_id"] != site.site_id:
            raise SystemExit("the binding belongs to another model or site")
        geometry = (binding.n_embd, binding.n_ff, binding.artifact_sha256)
    else:
        if None in (arguments.n_embd, arguments.n_ff, arguments.artifact):
            raise SystemExit("without --binding the plan needs --n-embd, --n-ff and --artifact")
        geometry = (arguments.n_embd, arguments.n_ff, arguments.artifact)
    policy = HostLaunchPolicy(gpu_layers=arguments.gpu_layers, ctx_size=arguments.ctx, parallel=arguments.parallel,
                              host_share=arguments.host_share, max_tokens=arguments.max_tokens)
    plan = engine.RunPlan(arguments.model, arguments.alias or arguments.model, arguments.port, policy, *geometry,
                          device, binding, arguments.idle_s, arguments.idle_s)
    arguments.out.write_text(json.dumps(plan.to_json(), indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(f"wrote {arguments.out}: {arguments.model}, -ngl {arguments.gpu_layers}, "
          + (f"shared with {device}" if device else "host alone"))
    return 0


def _run(arguments: argparse.Namespace) -> int:
    site = SiteManifest.load(arguments.site)
    plan = engine.RunPlan.from_json(json.loads(arguments.plan.read_text(encoding="ascii")))
    if arguments.alone:
        plan = plan.alone()
    if arguments.dry_run:
        print(json.dumps(engine.dry_run(site, plan, arguments.output), indent=2))
        return 0
    schedule = json.loads(arguments.schedule.read_text(encoding="utf-8"))
    options = engine.RunOptions(min_window_s=arguments.min_window_s, phase_retry=not arguments.no_phase_retry)
    for attempt in range(1, arguments.attempts + 1):
        rows = checks.site_checks(site, phones=[plan.device_id] if plan.shared else [],
                                  metered=not arguments.unmetered, required_caps_by_phone=_recorded_caps(site))
        stop = checks.blocking(rows)
        if stop:
            print(checks.format_checks(stop))
            print("\nrun not started:", len(stop), "blocking")
            return 2
        output = arguments.output
        try:
            result = engine.run(site, plan, schedule, output, options)
        except ServerLoadFailure as error:
            if error.kind != "gpu-out-of-memory" or attempt == arguments.attempts:
                raise
            # another job took the serving board while the model loaded: keep the directory, wait, try again
            output.rename(output.with_name(output.name + f".gpubusy-{attempt}"))
            print("RUN_GPU_BUSY attempt", attempt, flush=True)
            wait_until(lambda: bool(gpu_free(site.host.serving_gpu_index)["free"]), timeout_s=3600, period_s=30)
            continue
        print("RUN_DONE", result["status"], result["mode"], "phone calls", result["phone_calls"], output)
        return 0 if result["status"] == "PASS" else 1
    return 3


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m research_dev.scheduler.campaigns.portable", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    discover = commands.add_parser("discover", help="read the site's facts")
    discover.add_argument("--site", type=Path)
    discover.add_argument("--write", action="store_true")
    discover.add_argument("--add-phone", type=Path, metavar="DECLARED.json")
    discover.add_argument("--init", type=Path, metavar="SITE.json")
    discover.add_argument("--site-id")
    discover.add_argument("--serving-gpu", type=int)
    discover.add_argument("--threads", type=int)
    discover.add_argument("--cpu-affinity")
    discover.add_argument("--repo-root")
    discover.add_argument("--server")
    discover.add_argument("--adb")
    discover.add_argument("--work-root")
    discover.add_argument("--model", action="append", default=[], metavar="ID=PATH")
    discover.add_argument("--tool", action="append", default=[], metavar="NAME=PATH")
    discover.add_argument("--adb-port", type=int, action="append", default=None)

    doctor = commands.add_parser("doctor", help="check the host and the phones")
    doctor.add_argument("--site", type=Path, required=True)
    doctor.add_argument("--phone", action="append", default=[])
    doctor.add_argument("--unmetered", action="store_true", help="checks for an unmetered step (smoke, provisioning)")
    doctor.add_argument("--json", type=Path)

    host = commands.add_parser("onboard-host", help="measure the host with one model")
    host.add_argument("--site", type=Path, required=True)
    host.add_argument("--model", required=True)
    host.add_argument("--gpu-layers", type=int, required=True)
    host.add_argument("--port", type=int, default=27441)
    host.add_argument("--legs", type=int, default=3)
    host.add_argument("--output-tokens", type=int, default=128)
    host.add_argument("--artifact", help="sha256 of the model file (computed when absent)")
    host.add_argument("--ingest", type=Path, metavar="RUN_DIR")
    host.add_argument("--available-gb", type=float, help="with --ingest: host memory available before the load")

    phone = commands.add_parser("onboard-phone", help="sweep the shares of one phone")
    phone.add_argument("--site", type=Path, required=True)
    phone.add_argument("--phone", required=True)
    phone.add_argument("--model", required=True)
    phone.add_argument("--binding", type=Path)
    phone.add_argument("--port", type=int, default=27442)
    phone.add_argument("--usable-gb", type=float, help="memory the shards may take (default: 75 %% of what is free now)")
    phone.add_argument("--stored-columns", type=int)
    phone.add_argument("--ingest", type=Path, nargs="+", metavar="RUN_DIR")

    shards = commands.add_parser("provision", help="put shards (and a worker bundle) on a phone")
    shards.add_argument("--site", type=Path, required=True)
    shards.add_argument("--phone", required=True)
    shards.add_argument("--model", required=True)
    kind = shards.add_mutually_exclusive_group(required=True)
    kind.add_argument("--probe", action="store_true")
    kind.add_argument("--from-profile", action="store_true")
    shards.add_argument("--layers", type=int, default=4)
    shards.add_argument("--share", type=float, default=0.5)
    shards.add_argument("--bundle", type=Path)

    adopt = commands.add_parser("adopt", help="a binding for shards already on a phone")
    adopt.add_argument("--site", type=Path, required=True)
    adopt.add_argument("--phone", required=True)
    adopt.add_argument("--model", required=True)
    adopt.add_argument("--index", type=Path, required=True)
    adopt.add_argument("--remote-dir", required=True)
    adopt.add_argument("--out", type=Path)

    plan = commands.add_parser("make-plan", help="what a run serves")
    plan.add_argument("--site", type=Path, required=True)
    plan.add_argument("--model", required=True)
    plan.add_argument("--alias")
    plan.add_argument("--port", type=int, required=True)
    plan.add_argument("--gpu-layers", type=int, required=True)
    plan.add_argument("--ctx", type=int, required=True)
    plan.add_argument("--parallel", type=int, required=True)
    plan.add_argument("--binding", type=Path)
    plan.add_argument("--n-embd", type=int)
    plan.add_argument("--n-ff", type=int)
    plan.add_argument("--artifact")
    plan.add_argument("--host-share", choices=("resident", "dormant"), default="resident")
    plan.add_argument("--max-tokens", type=int, default=4)
    plan.add_argument("--idle-s", type=float, default=20.0)
    plan.add_argument("--out", type=Path, required=True)

    run = commands.add_parser("run", help="replay a schedule")
    run.add_argument("--site", type=Path, required=True)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--schedule", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--alone", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--unmetered", action="store_true")
    run.add_argument("--no-phase-retry", action="store_true")
    run.add_argument("--min-window-s", type=float, default=0.0)
    run.add_argument("--attempts", type=int, default=6)

    report.add_arguments(commands.add_parser("report", help="tables from run directories"))

    arguments = parser.parse_args(argv)
    if arguments.command == "discover":
        arguments.adb_port = arguments.adb_port or [5037]
        return _discover(arguments)
    if arguments.command == "doctor":
        return _doctor(arguments)
    if arguments.command == "onboard-host":
        return _onboard_host(arguments)
    if arguments.command == "onboard-phone":
        return _onboard_phone(arguments)
    if arguments.command == "provision":
        return _provision(arguments)
    if arguments.command == "adopt":
        return _adopt(arguments)
    if arguments.command == "make-plan":
        return _make_plan(arguments)
    if arguments.command == "run":
        if not arguments.dry_run and arguments.schedule is None:
            parser.error("run needs --schedule")
        return _run(arguments)
    report.run_report(arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
