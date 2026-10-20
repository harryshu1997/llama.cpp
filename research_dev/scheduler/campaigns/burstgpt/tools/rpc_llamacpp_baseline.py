#!/usr/bin/env python3
"""WS18 closest-prior-system baseline: llama.cpp's stock RPC offload of WHOLE layers to the same two phones.

    python3 -m ...rpc_llamacpp_baseline placement --models TEMPLATE/models.json --phone op15=BYTES --phone pixel=BYTES \\
        (--gpu-layers MODEL=N ... | --reference-placements DEFAULT_RUN/MODEL_PLACEMENTS.json) --out PLACEMENT.json
    python3 -m ...rpc_llamacpp_baseline plan --campaign TEMPLATE/campaign.json --placement PLACEMENT.json \\
        --phones PHONES.json --out PLAN.json
    python3 -m ...rpc_llamacpp_baseline warm --plan PLAN.json --phones PHONES.json --server BIN --out WARM_DIR
    python3 -m ...rpc_llamacpp_baseline run  --plan PLAN.json --phones PHONES.json --server BIN --out RUN_DIR
        [--phones-already-running]   (local loopback tests: endpoints served by host rpc servers)

What it is. The out-of-the-box arm (``default_llamacpp_baseline``: ONE stock llama-server in router mode, the
campaign trace, the campaign's request body, host energy over the paid window) plus the stock RPC backend: every
model with CPU-resident layers gets ``rpc``, ``n-gpu-layers``, ``tensor-split`` and ``ctx-size`` in its router
preset section, so llama.cpp places WHOLE layers (attention, KV cache and FFN) on the phones' ``ggml-rpc-server``
CPU devices. This is the latency-oriented distributed-offload design of prima.cpp / Exo / llama.cpp RPC; the
paper's system instead offloads only weight-streaming operators, batches and power-manages around them.

Placement (``placement``): llama.cpp orders devices as [RPC servers in ``--rpc`` order, then CUDA] and keeps the
first ``n_layer + 1 - n_gpu_layers`` layers on the CPU; ``tensor-split`` a,b,g+1 then gives RPC0 the next a layers,
RPC1 the next b, CUDA the last g layers plus the output layer (``src/llama-model.cpp`` split points). The GPU part
is the stock fit of the out-of-the-box arm (its ``offloaded N/M layers to GPU``), so only CPU-resident layers move.
Each phone takes as many of the remaining CPU layers, nearest the GPU block first, as fit in its budget: measured
``MemAvailable`` minus a margin, against the layer's GGUF tensor bytes plus a conservative KV cache (full context,
f16, every layer). Stock llama.cpp cannot do this itself: the CPU device of an rpc server reports total RAM as free
(``ggml-cpu.cpp`` get_memory), so its default split-by-free-memory and ``--fit`` would overcommit a phone; the
explicit split and the context are recorded as deviations in RESULT.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from research_dev.scheduler._internal.gguf_metadata import read_gguf_metadata
from research_dev.scheduler.campaigns.burstgpt.common import canonical, load_object
from research_dev.scheduler.campaigns.burstgpt.tools import default_llamacpp_baseline as baseline
from research_dev.scheduler.campaigns.burstgpt.tools import rpc_phone_servers as phones_mod

PLACEMENT_SCHEMA = "ws18-rpc-placement-v1"
ARM = "rpc-whole-layer-offload"
DEFAULT_CTX = 4096                       # what the stock fit chose for both f16 models in the out-of-the-box arm
DEFAULT_MARGIN_BYTES = 768 * 1024 ** 2   # compute buffers + rpc-server process + allocator slack per phone
KV_BYTES_PER_ELEMENT = 2                 # default f16 K/V cache

DEVIATIONS = (
    {"flag": "rpc = <phone endpoints> (per model preset)", "default": "no RPC devices",
     "category": "arm-definition",
     "reason": "the arm under test: llama.cpp's own RPC backend serving whole layers from ggml-rpc-server on the "
               "two phones (USB TCP)"},
    {"flag": "n-gpu-layers / tensor-split (per model preset)", "default": "fit to free device memory",
     "category": "necessity",
     "reason": "an rpc server's CPU device reports total RAM as free (ggml-cpu get_memory), so the default "
               "split-by-free-memory and --fit would overcommit the phones (Android kills the server). The CUDA "
               "share is the stock fit of the out-of-the-box arm; each phone gets the CPU-resident layers nearest "
               "the GPU block that fit its measured MemAvailable minus a margin (PLACEMENT.json)."},
    {"flag": "ctx-size (per model preset)", "default": "training context reduced by --fit",
     "category": "necessity",
     "reason": "with the placement fixed, --fit judges the phones by their reported RAM (= total) and keeps the "
               "training context; the KV cache of whole phone layers would then not fit the phones. The value is "
               "the context the stock fit chose for these models in the out-of-the-box arm (same KV size)."},
)
DEFAULTS_RELIED_ON = (
    "ggml-rpc-server defaults on the phones: CPU device, threads = hardware_concurrency/2, host 127.0.0.1",
    "ggml-rpc-server -c local tensor cache (stock option): repeated model loads send hashes, not weights",
    "llama.cpp layer order with RPC devices: CPU layers first, then RPC devices in --rpc order, then CUDA",
)


# ---------------------------------------------------------------- placement

def _scalar(fields: Mapping[str, Any], key: str, default: Any = None) -> Any:
    field = fields.get(key)
    if field is None:
        return default
    value = field.parts[0]
    return value


def _per_layer(value: Any, layer: int) -> int:
    if isinstance(value, (tuple, list)):
        return int(value[layer])
    return int(value)


def model_layers(path: str | Path, ctx: int) -> dict[str, Any]:
    """Per-layer GGUF tensor bytes and a conservative per-layer KV cache size at ``ctx``."""
    reader = read_gguf_metadata(path)
    fields = reader.fields
    arch = _scalar(fields, "general.architecture")
    n_layer = int(_scalar(fields, arch + ".block_count"))
    head_kv = _scalar(fields, arch + ".attention.head_count_kv")
    head = _scalar(fields, arch + ".attention.head_count")
    embd = int(_scalar(fields, arch + ".embedding_length"))
    key_len = _scalar(fields, arch + ".attention.key_length")
    value_len = _scalar(fields, arch + ".attention.value_length")
    weights = [0] * n_layer
    other = 0
    pattern = re.compile(r"blk\.(\d+)\.")
    for tensor in reader.tensors:
        match = pattern.match(tensor.name)
        if match and int(match.group(1)) < n_layer:
            weights[int(match.group(1))] += tensor.n_bytes
        else:
            other += tensor.n_bytes
    layers = []
    for index in range(n_layer):
        kv_heads = _per_layer(head_kv if head_kv is not None else head, index)
        heads = _per_layer(head, index) if head is not None else kv_heads
        k = int(key_len) if key_len is not None else embd // max(1, heads)
        v = int(value_len) if value_len is not None else k
        kv = ctx * kv_heads * (k + v) * KV_BYTES_PER_ELEMENT
        layers.append({"layer": index, "weight_bytes": weights[index], "kv_bytes": kv,
                       "bytes": weights[index] + kv})
    return {"architecture": arch, "n_layer": n_layer, "layers": layers, "non_layer_bytes": other, "ctx": ctx}


def plan_model(geometry: Mapping[str, Any], gpu_layers: int, budgets: Sequence[tuple[str, int]],
               margin_bytes: int) -> dict[str, Any]:
    """Phones take CPU-resident layers nearest the GPU block first; the LAST phone in --rpc order sits next to it."""
    n_layer = geometry["n_layer"]
    if not 0 <= gpu_layers <= n_layer:
        raise ValueError("gpu layers out of range")
    cpu_end = n_layer - gpu_layers          # layers [0, cpu_end) are CPU-resident without phones
    counts = {name: 0 for name, _ in budgets}
    used = {name: 0 for name, _ in budgets}
    cursor = cpu_end
    for name, budget in reversed(list(budgets)):
        room = budget - margin_bytes
        while cursor > 0 and used[name] + geometry["layers"][cursor - 1]["bytes"] <= room:
            cursor -= 1
            used[name] += geometry["layers"][cursor]["bytes"]
            counts[name] += 1
    ranges, start = {}, cursor
    for name, _ in budgets:
        ranges[name] = [start, start + counts[name]] if counts[name] else None
        start += counts[name]
    phone_layers = sum(counts.values())
    if phone_layers == 0:
        return {"phone_layers": 0, "cpu_layers": cpu_end, "gpu_layers": gpu_layers, "preset_options": {}}
    ngl = phone_layers + gpu_layers + 1
    split = [counts[name] for name, _ in budgets] + [gpu_layers + 1]
    return {
        "cpu_layers": cursor, "cpu_range": [0, cursor] if cursor else None, "gpu_layers": gpu_layers,
        "gpu_range": [cpu_end, n_layer], "phone_layers": phone_layers,
        "phones": {name: {"layers": counts[name], "range": ranges[name], "planned_bytes": used[name],
                          "budget_bytes": budget, "margin_bytes": margin_bytes}
                   for name, budget in budgets},
        "n_gpu_layers": ngl, "tensor_split": split,
    }


def preset_options(model_plan: Mapping[str, Any], endpoints: Sequence[str], ctx: int) -> dict[str, str]:
    if not model_plan.get("phone_layers"):
        return {}
    return {"ctx-size": str(ctx), "n-gpu-layers": str(model_plan["n_gpu_layers"]), "rpc": ",".join(endpoints),
            "tensor-split": ",".join(str(value) for value in model_plan["tensor_split"])}


def gpu_layers_from_reference(path: Path) -> dict[str, int]:
    """Repeating GPU layers the stock fit chose in an out-of-the-box run (``offloaded N/M`` includes the output)."""
    placements = load_object(path)
    out = {}
    for model, instances in (placements.get("by_model") or {}).items():
        values = {row.get("gpu_layers") for row in instances if row.get("gpu_layers") is not None}
        if len(values) != 1:
            raise ValueError("reference placement of %s is not unique: %s" % (model, sorted(values)))
        out[model] = max(0, values.pop() - 1)
    return out


def build_placement(models_manifest: Mapping[str, Any], gpu_layers: Mapping[str, int],
                    budgets: Sequence[tuple[str, int]], *, ctx: int = DEFAULT_CTX,
                    margin_bytes: int = DEFAULT_MARGIN_BYTES, geometry_fn=model_layers) -> dict[str, Any]:
    rows = {}
    for row in models_manifest["models"]:
        model_id = row["model_id"]
        geometry = geometry_fn(row["host_artifact_path"], ctx)
        if model_id not in gpu_layers:
            raise ValueError("no GPU layer count for " + model_id)
        rows[model_id] = {"geometry": {key: geometry[key] for key in ("architecture", "n_layer", "non_layer_bytes")},
                          "layer_bytes": [layer["bytes"] for layer in geometry["layers"]],
                          **plan_model(geometry, int(gpu_layers[model_id]), budgets, margin_bytes)}
    return {"schema": PLACEMENT_SCHEMA, "ctx": ctx, "margin_bytes": margin_bytes,
            "phones": [{"name": name, "budget_bytes": budget} for name, budget in budgets], "models": rows}


# ---------------------------------------------------------------- plan

def build_plan(trace_plan: Mapping[str, Any], placement: Mapping[str, Any],
               phone_list: Sequence[phones_mod.PhoneRpcServer]) -> dict[str, Any]:
    names = [row["name"] for row in placement["phones"]]
    if names != [phone.name for phone in phone_list]:
        raise ValueError("placement phones %s != phones file order %s" % (names, [p.name for p in phone_list]))
    endpoints = [phone.endpoint for phone in phone_list]
    plan = json.loads(json.dumps(trace_plan))
    for row in plan["models"]:
        model_plan = placement["models"].get(row["model_id"])
        if model_plan is None:
            raise ValueError("placement lacks " + row["model_id"])
        options = preset_options(model_plan, endpoints, int(placement["ctx"]))
        if options:
            row["preset_options"] = options
    plan["arm_extension"] = {
        "arm": ARM,
        "defaults_relied_on": list(DEFAULTS_RELIED_ON),
        "deviations": [dict(row) for row in DEVIATIONS],
        "placement": placement,
        "required_server_flags": ["--rpc", "--tensor-split"],
        "rpc_endpoints": endpoints,
        "rpc_phones": [{"name": p.name, "serial": p.serial, "transport": p.transport, "threads": p.threads,
                        "devices": p.devices, "cache": p.cache, "expected_sha256": dict(p.expected_sha256)}
                       for p in phone_list],
    }
    baseline.validate_plan(plan)
    return plan


# ---------------------------------------------------------------- phone lifecycle around a router run

def required_bytes(placement: Mapping[str, Any]) -> dict[str, int]:
    """Per phone: the largest planned footprint over the models (weights + KV) plus the placement margin; the
    phone's MemAvailable must cover it before its rpc server starts."""
    out: dict[str, int] = {}
    for row in (placement.get("models") or {}).values():
        for name, phone in (row.get("phones") or {}).items():
            need = int(phone.get("planned_bytes", 0)) + int(phone.get("margin_bytes", 0))
            out[name] = max(out.get(name, 0), need)
    return out


class PhoneServers:
    """Starts every phone's rpc server (preflight first) and stops them in reverse order, keeping receipts."""

    def __init__(self, phone_list, out: Path, *, runner=phones_mod.run_command, probe_fn=phones_mod.probe,
                 already_running: bool = False, required: Mapping[str, int] | None = None):
        required = dict(required or {})
        phone_list = [phone if required.get(phone.name, 0) <= phone.min_mem_available_bytes else
                      phones_mod.PhoneRpcServer(**{**phone.__dict__,
                                                   "min_mem_available_bytes": required[phone.name]})
                      for phone in phone_list]
        self.phones = list(phone_list)
        self.out = out
        self.runner = runner
        self.probe_fn = probe_fn
        self.already_running = already_running
        self.started: list[phones_mod.StartedServer] = []
        self.receipt: dict[str, Any] = {"schema": phones_mod.RECEIPT_SCHEMA, "preflight": [], "started": [],
                                        "stopped": [], "already_running": already_running}

    def __enter__(self):
        if self.already_running:
            self.receipt["started"] = [phones_mod.wait_for_probe(p.endpoint, 30, probe_fn=self.probe_fn)
                                       for p in self.phones]
            return self
        for phone in self.phones:
            self.receipt["preflight"].append(phones_mod.preflight(phone, self.runner))
        try:
            for phone in self.phones:
                started = phones_mod.start(phone, runner=self.runner, out_dir=self.out, probe_fn=self.probe_fn)
                self.started.append(started)
                self.receipt["started"].append(started.receipt())
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_exc):
        for started in reversed(self.started):
            self.receipt["stopped"].append(phones_mod.stop(started, runner=self.runner, out_dir=self.out))
        self.started = []
        self.receipt["clean"] = all(row.get("clean") for row in self.receipt["stopped"])
        return False


def _check_protocol(receipt: Mapping[str, Any], expected_major: int | None) -> None:
    for row in receipt.get("started", ()):
        probe = row.get("probe", row)
        major = int(str(probe.get("protocol", "0")).split(".")[0])
        if expected_major is not None and major != expected_major:
            raise SystemExit("rpc protocol %s on %s, client expects major %d" % (probe.get("protocol"),
                                                                                   probe.get("endpoint"),
                                                                                   expected_major))


def _rpc_major_from_source(repo: Path) -> int | None:
    header = repo / "ggml/include/ggml-rpc.h"
    if not header.is_file():
        return None
    match = re.search(r"#define RPC_PROTO_MAJOR_VERSION\s+(\d+)", header.read_text())
    return int(match.group(1)) if match else None


def run(args: argparse.Namespace, **baseline_kwargs) -> int:
    plan = load_object(Path(args.plan))
    if (plan.get("arm_extension") or {}).get("arm") != ARM:
        raise SystemExit("not an RPC baseline plan: " + args.plan)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("output directory is not empty: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    phone_list = phones_mod.load_phones(Path(args.phones))
    expected = [phone.endpoint for phone in phone_list]
    if expected != plan["arm_extension"]["rpc_endpoints"]:
        raise SystemExit("phones file endpoints %s != plan %s" % (expected, plan["arm_extension"]["rpc_endpoints"]))
    servers = PhoneServers(phone_list, out, already_running=args.phones_already_running,
                           required=required_bytes(plan["arm_extension"].get("placement") or {}),
                           **baseline_kwargs.pop("phone_kwargs", {}))
    code = 1
    run_dir = out / "router"
    with servers:
        _check_protocol(servers.receipt, _rpc_major_from_source(Path(baseline_kwargs.pop("repo", Path.cwd()))))
        plan["arm_extension"]["rpc_phone_receipt_start"] = servers.receipt.get("started")
        (out / "RPC_PLAN.json").write_bytes(canonical(plan))
        router_args = baseline.build_parser().parse_args(
            ["run", "--plan", str(out / "RPC_PLAN.json"), "--out", str(run_dir), "--server", args.server,
             "--port", str(args.port), "--max-duration-s", str(args.max_duration_s),
             *(["--gpu-index", str(args.gpu_index)] if args.gpu_index is not None else []),
             *(["--cuda-visible-devices", args.cuda_visible_devices] if args.cuda_visible_devices else []),
             *sum((["--library-dir", d] for d in (args.library_dir or [])), []),
             *(["--allow-partial-energy"] if args.allow_partial_energy else [])])
        code = baseline.run(router_args, **baseline_kwargs)
    (out / "RPC_PHONES.json").write_bytes(canonical(servers.receipt))
    _promote(run_dir, out, servers.receipt)
    if not servers.receipt.get("clean", True) and not args.phones_already_running:
        print("warning: a phone rpc server did not stop cleanly; see RPC_PHONES.json", file=sys.stderr)
        code = code or 3
    return code


def _promote(run_dir: Path, out: Path, receipt: Mapping[str, Any]) -> None:
    """The router run writes into OUT/router; the accountants read OUT directly (RESULT.json, streams/, ...)."""
    for item in run_dir.iterdir():
        item.rename(out / item.name)
    run_dir.rmdir()
    result_path = out / "RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        result["rpc_phone_servers"] = dict(receipt)
        result_path.write_bytes(canonical(result))


# ---------------------------------------------------------------- warm (rpc cache fill, not measured)

def warm(args: argparse.Namespace, *, popen=subprocess.Popen, health=None) -> int:
    """Load each phone-assisted model once (single llama-server, the plan's preset options) so the phones' rpc
    caches hold their layers; later loads send tensor hashes. Not part of any measured window."""
    plan = load_object(Path(args.plan))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    phone_list = phones_mod.load_phones(Path(args.phones))
    rig_libs = list(args.library_dir or [])
    environment, _, _ = baseline.router_environment(os.environ, rig_libs, Path(args.server).parent,
                                                    args.cuda_visible_devices)
    health = health or (lambda port: baseline.http_get_json("127.0.0.1", port, "/health").get("status") == "ok")
    report = []
    with PhoneServers(phone_list, out, already_running=args.phones_already_running,
                      required=required_bytes((plan.get("arm_extension") or {}).get("placement") or {})) as servers:
        for row in plan["models"]:
            options = row.get("preset_options") or {}
            if not options:
                continue
            command = [args.server, "-m", row["path"], "--host", "127.0.0.1", "--port", str(args.port)]
            for key, value in sorted(options.items()):
                command += ["--" + key, value]
            log = open(out / ("warm-%s.log" % row["model_id"]), "wb")
            started = time.monotonic()
            process = popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL)
            ok = False
            while time.monotonic() - started < args.timeout_s and process.poll() is None:
                try:
                    if health(args.port):
                        ok = True
                        break
                except OSError:
                    pass
                time.sleep(1.0)
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
            report.append({"model_id": row["model_id"], "loaded": ok, "seconds": round(time.monotonic() - started, 1),
                           "command": command})
        receipt = servers.receipt
    (out / "WARM.json").write_bytes(canonical({"models": report, "phones": receipt}))
    return 0 if report and all(row["loaded"] for row in report) else 1


# ---------------------------------------------------------------- CLI

def _budget(text: str) -> tuple[str, int]:
    name, _, value = text.partition("=")
    if not name or not value.isdigit():
        raise argparse.ArgumentTypeError("phone budget must be NAME=BYTES")
    return name, int(value)


def _gpu(text: str) -> tuple[str, int]:
    name, _, value = text.partition("=")
    if not name or not value.isdigit():
        raise argparse.ArgumentTypeError("gpu layers must be MODEL=N")
    return name, int(value)


def _run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", required=True)
    parser.add_argument("--phones", required=True)
    parser.add_argument("--server", required=True, help="llama-server built with -DGGML_RPC=ON")
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, default=18610)
    parser.add_argument("--library-dir", action="append")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--phones-already-running", action="store_true",
                        help="endpoints are served already (local loopback tests); only probe them")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("placement")
    p.add_argument("--models", required=True)
    p.add_argument("--phone", type=_budget, action="append", required=True, help="NAME=BUDGET_BYTES in --rpc order")
    p.add_argument("--gpu-layers", type=_gpu, action="append", default=[])
    p.add_argument("--reference-placements", help="MODEL_PLACEMENTS.json of an out-of-the-box run")
    p.add_argument("--ctx", type=int, default=DEFAULT_CTX)
    p.add_argument("--margin-bytes", type=int, default=DEFAULT_MARGIN_BYTES)
    p.add_argument("--out", required=True)
    p = sub.add_parser("plan")
    baseline._add_input_arguments(p)
    p.add_argument("--placement", required=True)
    p.add_argument("--phones", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("warm")
    _run_arguments(p)
    p.add_argument("--timeout-s", type=float, default=900.0)
    p = sub.add_parser("run")
    _run_arguments(p)
    p.add_argument("--max-duration-s", type=float, default=14_400.0)
    p.add_argument("--allow-partial-energy", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "placement":
        gpu = dict(args.gpu_layers)
        if args.reference_placements:
            gpu = {**gpu_layers_from_reference(Path(args.reference_placements)), **gpu}
        placement = build_placement(load_object(Path(args.models)), gpu, args.phone, ctx=args.ctx,
                                    margin_bytes=args.margin_bytes)
        Path(args.out).write_bytes(canonical(placement))
        for model, row in placement["models"].items():
            print("%s: cpu %s, phones %s, gpu %s (+output) -> n-gpu-layers %s, tensor-split %s" % (
                model, row.get("cpu_layers"), {k: v["layers"] for k, v in (row.get("phones") or {}).items()},
                row["gpu_layers"], row.get("n_gpu_layers"), row.get("tensor_split")))
        return 0
    if args.command == "plan":
        args.plan = None
        trace_plan, _ = baseline._load_inputs(args)
        plan = build_plan(trace_plan, load_object(Path(args.placement)), phones_mod.load_phones(Path(args.phones)))
        Path(args.out).write_bytes(canonical(plan))
        print("plan: %d requests -> %s" % (len(plan["requests"]), args.out))
        return 0
    if args.command == "warm":
        return warm(args)
    return run(args, repo=Path(__file__).resolve().parents[5])


if __name__ == "__main__":
    raise SystemExit(main())
